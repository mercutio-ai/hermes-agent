"""A dispatched worktree task's branch must match the branch its worktree is on.

A task row's ``branch_name`` can be written by two independent actors: the
dispatcher (which derives ``wt/<task-id>`` for a worktree task that carries no
branch) and a caller that sets a conventional branch on the row afterwards --
e.g. an orchestrator stamping ``worktrees/fix/<slug>`` onto a freshly created
card, or a CLI caller using ``kanban create --branch``.

The dispatcher used to read ``branch_name`` from the task snapshot it took at
CLAIM time and materialise the worktree on that value. If another writer set
the branch between the claim and the worktree being resolved, the snapshot was
stale: the worktree was created on the derived ``wt/<task-id>`` fallback, and
the row was then written back with it -- silently discarding the branch the
caller had asked for, and leaving the row describing a branch that no checkout
uses. Anything that later matches ``task.branch_name`` against a PR head ref
(PR ownership, review routing) then fails to find an owner.

The fix re-reads ``branch_name`` from the row immediately before resolving the
worktree, so the worktree is created on the branch the row currently names and
the row is written back with that same value. The row and the worktree can no
longer disagree.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def spawnable(monkeypatch):
    """Make every assignee dispatchable."""
    from hermes_cli import profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda _name: True)


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        [
            "git", "-C", str(cwd),
            "-c", "user.name=Test User",
            "-c", "user.email=test@example.com",
            "-c", "commit.gpgsign=false",
            *args,
        ],
        check=True, capture_output=True, text=True,
    )


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main", str(repo)],
        check=True, capture_output=True, text=True,
    )
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init")
    return repo


def _head_branch(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def _dispatch(conn, repo: Path, wanted_branch: str | None):
    """Create a worktree card, set its branch, and run ONE real dispatch tick.

    ``wanted_branch`` is written to the row BEFORE the tick -- that is the
    ordering that loses the branch, because the dispatcher claims the card
    inside the tick and the claim-time snapshot is taken there.

    Returns ``(task_id, workspace_path)``.
    """
    spawned = []

    def fake_spawn(task, workspace, board=None):
        spawned.append((task.id, str(workspace)))
        return 4242

    tid = kb.create_task(
        conn,
        title="fix: worktree branch matches row branch",
        assignee="alice",
        workspace_kind="worktree",
        workspace_path=str(repo),
        branch_name=wanted_branch,
    )
    res = kbd.dispatch_once(conn, spawn_fn=fake_spawn)
    assert res.spawned, "the task should have been dispatched"
    assert spawned, "spawn_fn should have been called"
    return tid, Path(spawned[0][1])


def test_dispatch_honours_branch_set_before_the_tick(kanban_home, spawnable, tmp_path):
    """The worktree lands on the branch the row names, and the row still says so."""
    repo = _make_repo(tmp_path)
    wanted = "worktrees/fix/2026-11-08-money-tracker-override-cleanup"

    with kbc.connect() as conn:
        tid, workspace = _dispatch(conn, repo, wanted)
        stored = conn.execute(
            "SELECT branch_name FROM tasks WHERE id = ?", (tid,)
        ).fetchone()["branch_name"]

    # The row and the checkout agree -- the invariant PR ownership relies on.
    assert stored == wanted
    assert _head_branch(workspace) == wanted


def test_dispatch_derives_fallback_branch_when_none_is_set(kanban_home, spawnable, tmp_path):
    """With no branch anywhere, the derived fallback is still used.

    The re-read must not break the ordinary path: a task that never had a
    branch still gets ``wt/<task-id>``, and the row still says so.
    """
    repo = _make_repo(tmp_path)

    with kbc.connect() as conn:
        tid, workspace = _dispatch(conn, repo, None)
        stored = conn.execute(
            "SELECT branch_name FROM tasks WHERE id = ?", (tid,)
        ).fetchone()["branch_name"]

    assert stored == f"wt/{tid}"
    assert _head_branch(workspace) == f"wt/{tid}"


def test_dispatch_branch_set_after_the_claim_is_not_lost(
    kanban_home, spawnable, tmp_path, monkeypatch
):
    """A branch written AFTER the claim still wins over the derived fallback.

    This is the exact ordering that broke the money-tracker Socket chain: the
    card was created without a branch, the dispatcher claimed it and derived
    ``wt/<card-id>``, and the orchestrator's post-hoc write of the intended
    branch was then discarded -- leaving the card pointing at a branch that
    its PR head ref did not use, so the reconciler could not find an owner.

    The race is reproduced at its true point: the concurrent write lands
    between the dispatcher's claim and its worktree resolution. The claim
    returns a snapshot with ``branch_name`` still None (that is what the
    dispatcher sees), while the row already carries the intended branch.

    The dispatcher must re-read the row at resolve time instead of trusting
    the claim-time snapshot.
    """
    repo = _make_repo(tmp_path)
    wanted = "worktrees/fix/2026-11-08-money-tracker-override-cleanup"

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn,
            title="fix: branch set after the claim survives",
            assignee="alice",
            workspace_kind="worktree",
            workspace_path=str(repo),
        )
        # The card exists with NO branch: this is what a create without
        # --branch looks like.
        assert kb.get_task(conn, tid).branch_name is None

        real_claim = kb.claim_task

        def claim_then_concurrent_write(conn_, task_id, **kwargs):
            """The dispatcher claims; another writer stamps the branch."""
            claimed = real_claim(conn_, task_id, **kwargs)
            # The other writer lands right after the claim, before the
            # dispatcher resolves the worktree.
            conn_.execute(
                "UPDATE tasks SET branch_name = ? WHERE id = ?", (wanted, task_id)
            )
            conn_.commit()
            return claimed

        monkeypatch.setattr(kb, "claim_task", claim_then_concurrent_write)

        spawned = []

        def fake_spawn(task, workspace, board=None):
            spawned.append((task.id, str(workspace)))
            return 4242

        kbd.dispatch_once(conn, spawn_fn=fake_spawn)
        assert spawned, "the task should have been dispatched"

        stored = conn.execute(
            "SELECT branch_name FROM tasks WHERE id = ?", (tid,)
        ).fetchone()["branch_name"]

    workspace = Path(spawned[0][1])
    # The intended branch survived, and the worktree is ON it.
    assert stored == wanted
    assert _head_branch(workspace) == wanted
