"""Git hash / dirty-tree guard and launch sentinels for training jobs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def assert_no_sourceless_bytecode(package: str = "ode") -> None:
    """Refuse if any imported ``ode.*`` module is bytecode without a sibling .py."""
    bad: list[str] = []
    for name, mod in list(sys.modules.items()):
        if name != package and not name.startswith(package + "."):
            continue
        path = getattr(mod, "__file__", None)
        if not path:
            continue
        p = Path(path)
        if p.suffix != ".pyc":
            continue
        if p.parent.name == "__pycache__":
            stem = p.name.split(".cpython-")[0]
            src = p.parent.parent / f"{stem}.py"
        else:
            src = p.with_suffix(".py")
        if not src.is_file():
            bad.append(f"{name} -> {p}")
    if bad:
        raise SystemExit(
            "REFUSE: imported sourceless bytecode:\n  " + "\n  ".join(bad)
        )


def write_complete(out_dir: Path, extra: str = "") -> None:
    """Write the COMPLETE sentinel after the final checkpoint is on disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = "COMPLETE\n" + extra
    (out_dir / "COMPLETE").write_text(payload)


def is_complete_ckpt(ckpt_dir: Path, ckpt_name: str = "best.ckpt") -> bool:
    """True iff this seed dir has COMPLETE and the expected checkpoint."""
    return (ckpt_dir / "COMPLETE").is_file() and (ckpt_dir / ckpt_name).is_file()


def list_complete_ckpts(ckpt_dirs: list[Path], ckpt_name: str = "best.ckpt") -> list[Path]:
    """Return dirs that have COMPLETE + ckpt. Incomplete dirs are skipped."""
    return [d for d in ckpt_dirs if is_complete_ckpt(d, ckpt_name)]


def require_complete_ckpts(ckpt_dirs: list[Path], ckpt_name: str = "best.ckpt") -> None:
    """All-or-nothing refuse. Prefer list_complete_ckpts for afterany eval."""
    if not ckpt_dirs:
        raise SystemExit("REFUSE: no checkpoint directories to check")
    missing: list[str] = []
    for d in ckpt_dirs:
        if not (d / "COMPLETE").is_file():
            missing.append(f"{d}: missing COMPLETE")
        if not (d / ckpt_name).is_file():
            missing.append(f"{d}: missing {ckpt_name}")
    if missing:
        raise SystemExit("REFUSE: incomplete NCDE run\n  " + "\n  ".join(missing))


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
        assert_no_sourceless_bytecode()
        return status
    raise SystemExit(
        "REFUSE: dirty or missing git tree. Commit, or pass --allow-dirty."
    )
