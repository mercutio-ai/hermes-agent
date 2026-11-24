"""Acknowledgment-contract regression tests for webhook routes with ``script:``.

Contract (defined in t_2fc0131f, attached as ``ack-contract.md``): for a route
with ``script:`` the handler must verify HMAC + parse + filter-match on the
request path, record the delivery id, return 2xx (202 accepted) within 2 s
(hard cap: GitHub's ~10 s delivery timeout), and run the route script in a
tracked background task. Background failures and timeouts are logged, never
surfaced into the HTTP response. Fast ack must never turn a rejected delivery
into a success.

Integrated state: ``_handle_webhook`` now acks script routes with 202 via
``_accept_script_route`` and runs ``run_route_script`` in the tracked
``_background_tasks`` task (``_run_script_route``,
``gateway/platforms/webhook.py``). The tests below validate that integrated
behavior. They are contract-first: against the old synchronous
implementation the latency, background-failure, and exactly-once tests FAIL
with explicit per-case messages (verified via ``git stash`` during
development) -- the 2 s threshold is the contract's and is deliberately NOT
relaxed. The invalid-auth and quick-delivery tests pass on both.

All cases use fakes (a monkeypatched ``run_route_script`` and a stubbed
durable producer): no real ``gh``, Discord, or network calls.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import subprocess
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms import webhook_ingress
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH

# The acknowledgment budget from the contract: well under GitHub's ~10 s
# delivery cap. Do NOT raise this to accommodate synchronous execution.
ACK_BUDGET_SECONDS = 2.0

ROUTE = "factory-gate"


# ---------------------------------------------------------------------------
# Helpers (same direct-handler harness as test_webhook_coalescing.py)
# ---------------------------------------------------------------------------


def _make_adapter(routes) -> WebhookAdapter:
    extra = {"host": "127.0.0.1", "port": 0, "routes": routes, "rate_limit": 100}
    return WebhookAdapter(PlatformConfig(enabled=True, extra=extra))


def _mock_request(
    body: bytes,
    route_name: str = ROUTE,
    delivery_id: str = "d-1",
    extra_headers: dict | None = None,
):
    req = MagicMock()
    headers = {"X-GitHub-Delivery": delivery_id}
    if extra_headers:
        headers.update(extra_headers)
    req.headers = headers
    req.content_length = len(body)
    req.match_info = {"route_name": route_name}
    req.method = "POST"

    async def _read():
        return body

    req.read = _read
    return req


def _capture_admissions(adapter, monkeypatch):
    """Stand in for the durable producer with honest duplicate detection.

    The real producer dedups provider retries; the stub keeps its own seen
    set and marks replays via ``event._webhook_duplicate``, which
    ``_admit_agent_run`` already translates into a ``"duplicate"`` outcome.
    """
    admitted: list = []
    seen: set = set()

    async def _admit(_adapter, event):
        if event.message_id in seen:
            event._webhook_duplicate = True
            return SimpleNamespace(
                admission_id="adm-dup", ref=SimpleNamespace(profile_id="default")
            )
        seen.add(event.message_id)
        admitted.append(event)
        return SimpleNamespace(
            admission_id=f"adm-{len(admitted)}",
            ref=SimpleNamespace(profile_id="default"),
        )

    async def _finalize(event, authority, receipt):
        return None

    monkeypatch.setattr(webhook_ingress, "admit_producer", _admit)
    monkeypatch.setattr(adapter, "_finalize_delivery", _finalize)
    adapter._message_handler = SimpleNamespace(__self__=SimpleNamespace())
    monkeypatch.setattr(
        "gateway.session_authorities.authority_for_profile_id",
        lambda runner, pid: object(),
    )
    return admitted


async def _drain_background_tasks(adapter: WebhookAdapter) -> None:
    tasks = [t for t in list(adapter._background_tasks) if not t.done()]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def _wait_for(predicate, timeout: float = 5.0):
    """Poll ``predicate`` until truthy; raise TimeoutError on expiry."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise TimeoutError(f"condition not met within {timeout:.1f}s")


def _script_route(**overrides):
    route = {"secret": _INSECURE_NO_AUTH, "script": "factory-webhook-gate.py"}
    route.update(overrides)
    return route


# ===================================================================
# 1. Blocking script: 2xx within budget, returned BEFORE release
# ===================================================================


class TestBlockingScriptAck:
    @pytest.mark.asyncio
    async def test_response_precedes_blocked_script(self, monkeypatch):
        """A script that blocks until released must not hold the response.

        CONTRACT-FIRST: fails against the old synchronous implementation,
        which awaits ``run_route_script`` on the request path and therefore
        cannot answer until ``release`` is set (``wait_for`` expires).
        """
        adapter = _make_adapter({ROUTE: _script_route()})
        admitted = _capture_admissions(adapter, monkeypatch)

        release = threading.Event()
        started = threading.Event()
        calls: list = []

        def _blocking(script, payload):
            calls.append(script)
            started.set()
            assert release.wait(timeout=30), "script released by test teardown"
            return True, dict(payload)

        monkeypatch.setattr(adapter._route_processor, "run_route_script", _blocking)

        body = json.dumps({"event_type": "push", "n": 1}).encode()
        t0 = time.monotonic()
        resp_task = asyncio.create_task(
            adapter._handle_webhook(_mock_request(body, delivery_id="d-block"))
        )
        # The script must at least have STARTED (proves the request was
        # accepted and work was dispatched, not dropped).
        await asyncio.to_thread(started.wait, 10)
        assert started.is_set(), "route script never started"

        try:
            resp = await asyncio.wait_for(
                asyncio.shield(resp_task), timeout=ACK_BUDGET_SECONDS
            )
        except TimeoutError:
            release.set()  # let the stranded request-path run finish
            await resp_task
            pytest.fail(
                "contract-first failure: webhook response did not arrive within "
                f"{ACK_BUDGET_SECONDS}s while the route script was blocked -- "
                "the handler is still awaiting run_route_script on the request "
                "path instead of acking 202 and running it in the background"
            )

        elapsed = time.monotonic() - t0
        assert elapsed <= ACK_BUDGET_SECONDS, f"ack took {elapsed:.2f}s"
        # The response was produced while the script was still blocked.
        assert not release.is_set(), "test bug: release set before response"
        assert resp.status in (200, 202), f"expected 2xx, got {resp.status}"
        assert json.loads(resp.text)["status"] == "accepted"

        # Release the script: the accepted work must then run exactly once.
        release.set()
        await _wait_for(lambda: len(admitted) == 1)
        await _drain_background_tasks(adapter)
        assert calls == ["factory-webhook-gate.py"]
        assert len(admitted) == 1
        assert admitted[0].message_id == "d-block"


# ===================================================================
# 2. Script exception / timeout: still 2xx, failure is logged
# ===================================================================


def _logged(caplog, *needles: str) -> bool:
    haystacks = [
        r.getMessage().lower() for r in caplog.records if r.levelno >= logging.WARNING
    ]
    return any(all(n.lower() in h for n in needles) for h in haystacks)


class TestBackgroundScriptFailures:
    @pytest.mark.asyncio
    async def test_script_exception_still_acks_and_logs(self, monkeypatch, caplog):
        """A raising script must not turn the delivery into a 5xx.

        CONTRACT-FIRST: fails against the old implementation, where the
        exception propagates out of the awaited ``run_route_script`` instead
        of being contained in a background task and logged.
        """
        adapter = _make_adapter({ROUTE: _script_route()})
        admitted = _capture_admissions(adapter, monkeypatch)
        calls: list = []

        def _boom(script, payload):
            calls.append(script)
            raise RuntimeError("gate exploded")

        monkeypatch.setattr(adapter._route_processor, "run_route_script", _boom)

        body = json.dumps({"event_type": "push"}).encode()
        try:
            resp = await adapter._handle_webhook(
                _mock_request(body, delivery_id="d-boom")
            )
        except RuntimeError as exc:
            pytest.fail(
                f"contract-first failure: script exception escaped the handler ({exc!r}) -- "
                "it must be contained in the background task with a 2xx already returned"
            )
        assert resp.status in (200, 202), f"expected 2xx, got {resp.status}"
        assert json.loads(resp.text)["status"] == "accepted"
        await _drain_background_tasks(adapter)
        assert calls == ["factory-webhook-gate.py"]
        assert admitted == [], "a raising script must not admit a run"
        assert (
            _logged(caplog, ROUTE)
            or _logged(caplog, "d-boom")
            or _logged(caplog, "script")
        ), (
            f"background script failure was not logged; records: "
            f"{[(r.levelname, r.getMessage()[:160]) for r in caplog.records]}"
        )

    @pytest.mark.asyncio
    async def test_script_timeout_still_acks_and_logs(self, monkeypatch, caplog):
        """A script timeout must not hold or fail the delivery.

        CONTRACT-FIRST: same expectation as the exception case; the timeout
        surfaces as a background log mentioning the timeout, not as a slow
        or failed HTTP response.
        """
        adapter = _make_adapter({ROUTE: _script_route()})
        admitted = _capture_admissions(adapter, monkeypatch)
        calls: list = []

        def _hang(script, payload):
            calls.append(script)
            raise subprocess.TimeoutExpired(cmd=["factory-webhook-gate.py"], timeout=60)

        monkeypatch.setattr(adapter._route_processor, "run_route_script", _hang)

        body = json.dumps({"event_type": "push"}).encode()
        t0 = time.monotonic()
        try:
            resp = await adapter._handle_webhook(
                _mock_request(body, delivery_id="d-timeout")
            )
        except subprocess.TimeoutExpired as exc:
            pytest.fail(
                f"contract-first failure: script timeout escaped the handler ({exc!r}) -- "
                "it must be contained in the background task with a 2xx already returned"
            )
        elapsed = time.monotonic() - t0
        assert elapsed <= ACK_BUDGET_SECONDS, f"ack took {elapsed:.2f}s"
        assert resp.status in (200, 202), f"expected 2xx, got {resp.status}"
        await _drain_background_tasks(adapter)
        assert calls == ["factory-webhook-gate.py"]
        assert admitted == [], "timed-out script must not admit a run"
        assert (
            _logged(caplog, "timed out")
            or _logged(caplog, "timeout")
            or _logged(caplog, "TimeoutExpired")
        ), (
            f"background script timeout was not logged; records: "
            f"{[(r.levelname, r.getMessage()[:160]) for r in caplog.records]}"
        )


# ===================================================================
# 3. Normal quick delivery still executes the expected path
# ===================================================================


class TestQuickDeliveryPassthrough:
    @pytest.mark.asyncio
    async def test_fast_script_transform_reaches_admission(self, monkeypatch):
        """Happy path passes on old and new implementations alike."""
        adapter = _make_adapter({ROUTE: _script_route()})
        admitted = _capture_admissions(adapter, monkeypatch)
        calls: list = []

        def _fast(script, payload):
            calls.append(script)
            return True, {**payload, "gate": "kept"}

        monkeypatch.setattr(adapter._route_processor, "run_route_script", _fast)

        body = json.dumps({"event_type": "push", "n": 3}).encode()
        resp = await adapter._handle_webhook(_mock_request(body, delivery_id="d-fast"))
        assert resp.status in (200, 202)
        assert json.loads(resp.text)["status"] == "accepted"
        # New implementations admit from the background task: poll, don't assume.
        await _wait_for(lambda: len(admitted) == 1)
        await _drain_background_tasks(adapter)
        assert calls == ["factory-webhook-gate.py"]
        assert "kept" in admitted[0].text, (
            "transformed payload did not reach the prompt"
        )

    @pytest.mark.asyncio
    async def test_duplicate_delivery_runs_script_once(self, monkeypatch):
        """A provider retry (same delivery id) is answered duplicate and the
        gate script is not re-run.

        CONTRACT-FIRST: fails against the old implementation, which runs the
        script BEFORE any delivery-id record exists, so the retry re-runs the
        gate (calls == 2). The contract records the id pre-ack.
        """
        adapter = _make_adapter({ROUTE: _script_route()})
        admitted = _capture_admissions(adapter, monkeypatch)
        calls: list = []

        def _fast(script, payload):
            calls.append(script)
            return True, dict(payload)

        monkeypatch.setattr(adapter._route_processor, "run_route_script", _fast)

        body = json.dumps({"event_type": "push"}).encode()
        first = await adapter._handle_webhook(
            _mock_request(body, delivery_id="d-retry")
        )
        assert json.loads(first.text)["status"] == "accepted"
        second = await adapter._handle_webhook(
            _mock_request(body, delivery_id="d-retry")
        )
        assert second.status in (200, 202)
        assert json.loads(second.text)["status"] == "duplicate"
        await _wait_for(lambda: len(admitted) == 1)
        await _drain_background_tasks(adapter)
        assert admitted[0].message_id == "d-retry"
        assert calls == ["factory-webhook-gate.py"], (
            f"contract-first failure: retried delivery re-ran the gate {len(calls)}x -- "
            "the delivery id must be recorded before the 202 so retries are duplicates"
        )


# ===================================================================
# 4. Invalid authentication: fast ack must not bless rejections
# ===================================================================

_HMAC_SECRET = "correct-horse-battery-staple"
_HMAC_BODY = json.dumps({"event_type": "push", "n": 9}).encode()


def _github_sig_headers(delivery_id: str) -> dict:
    mac = hmac.new(_HMAC_SECRET.encode(), _HMAC_BODY, hashlib.sha256).hexdigest()
    return {"X-Hub-Signature-256": "sha256=" + mac, "X-GitHub-Delivery": delivery_id}


class TestInvalidAuthStillRejected:
    @pytest.mark.asyncio
    async def test_bad_signature_rejected_and_script_never_starts(self, monkeypatch):
        """No signature -> 404 (same as unknown route); the script is untouched.

        Passes on old and new implementations: auth stays on the request path
        and precedes any ack by construction.
        """
        adapter = _make_adapter(
            {ROUTE: {"secret": _HMAC_SECRET, "script": "factory-webhook-gate.py"}}
        )
        admitted = _capture_admissions(adapter, monkeypatch)
        calls: list = []
        monkeypatch.setattr(
            adapter._route_processor,
            "run_route_script",
            lambda script, payload: calls.append(script) or (True, payload),
        )

        resp = await adapter._handle_webhook(
            _mock_request(_HMAC_BODY, delivery_id="d-nosig")
        )
        assert resp.status == 404, f"bad signature must stay 404, got {resp.status}"
        assert calls == [], (
            "route script must not start for an unauthenticated delivery"
        )
        assert admitted == []
        await _drain_background_tasks(adapter)
        assert adapter._background_tasks == set()

    @pytest.mark.asyncio
    async def test_valid_signature_still_accepted(self, monkeypatch):
        """Positive control: a correctly signed delivery is accepted and runs."""
        adapter = _make_adapter(
            {ROUTE: {"secret": _HMAC_SECRET, "script": "factory-webhook-gate.py"}}
        )
        admitted = _capture_admissions(adapter, monkeypatch)
        calls: list = []

        def _fast(script, payload):
            calls.append(script)
            return True, dict(payload)

        monkeypatch.setattr(adapter._route_processor, "run_route_script", _fast)

        req = _mock_request(
            _HMAC_BODY,
            delivery_id="d-signed",
            extra_headers=_github_sig_headers("d-signed"),
        )
        resp = await adapter._handle_webhook(req)
        assert resp.status in (200, 202), f"valid signature must 2xx, got {resp.status}"
        await _wait_for(lambda: len(admitted) == 1)
        await _drain_background_tasks(adapter)
        assert calls == ["factory-webhook-gate.py"]
