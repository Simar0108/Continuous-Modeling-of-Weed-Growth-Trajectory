"""Git hash / dirty-tree guard for training jobs."""

from __future__ import annotations

import subprocess
from pathlib import Path


def git_status(repo: Path) -> dict:
    out = {
        "git_hash": "NO_REPO",
        "git_dirty": True,
        "git_ok": False,
        "git_message": "not a git repository",
    }
    try:
        hash_p = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo, capture_output=True, text=True, check=False,
        )
        if hash_p.returncode != 0:
            return out
        dirty_p = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo, capture_output=True, text=True, check=False,
        )
        dirty = bool(dirty_p.stdout.strip())
        out.update({
            "git_hash": hash_p.stdout.strip(),
            "git_dirty": dirty,
            "git_ok": not dirty,
            "git_message": "dirty working tree" if dirty else "clean",
        })
        return out
    except FileNotFoundError:
        out["git_message"] = "git executable not found"
        return out


def assert_clean_or_allowed(repo: Path, allow_dirty: bool) -> dict:
    status = git_status(repo)
    print(
        f"[repro] git_hash={status['git_hash']} dirty={status['git_dirty']} "
        f"({status['git_message']})"
    )
    if status["git_ok"] or allow_dirty:
        return status
    raise SystemExit(
        "REFUSE: dirty or missing git tree. Commit, or pass --allow-dirty."
    )
