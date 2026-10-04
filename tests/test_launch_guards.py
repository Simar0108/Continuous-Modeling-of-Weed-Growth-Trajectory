"""COMPLETE sentinel and sourceless-bytecode guards. CPU only."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from ode.repro import assert_no_sourceless_bytecode, require_complete_ckpts, write_complete

REPO = Path(__file__).resolve().parent.parent


def test_sourceful_imports():
    assert_no_sourceless_bytecode()


def test_eval_refuses_without_complete(tmp_path: Path):
    d = tmp_path / "h1_ncde_seed0"
    d.mkdir(parents=True)
    (d / "best.ckpt").write_text("stub")
    try:
        require_complete_ckpts([d])
    except SystemExit as exc:
        assert "missing COMPLETE" in str(exc)
    else:
        raise AssertionError("expected REFUSE")


def test_eval_refuses_without_ckpt(tmp_path: Path):
    d = tmp_path / "h1_ncde_seed0"
    d.mkdir(parents=True)
    write_complete(d)
    try:
        require_complete_ckpts([d])
    except SystemExit as exc:
        assert "missing best.ckpt" in str(exc)
    else:
        raise AssertionError("expected REFUSE")


def test_eval_accepts_complete_and_ckpt(tmp_path: Path):
    d = tmp_path / "h1_ncde_seed0"
    d.mkdir(parents=True)
    write_complete(d)
    (d / "best.ckpt").write_text("stub")
    require_complete_ckpts([d])


def test_killed_job_never_writes_complete(tmp_path: Path):
    """Kill a dummy trainer mid-run: COMPLETE is withheld and eval refuses."""
    out = tmp_path / "killed_seed"
    out.mkdir(parents=True)
    script = tmp_path / "slow_train.py"
    script.write_text(
        "import time\n"
        "from pathlib import Path\n"
        "from ode.repro import write_complete\n"
        f"out = Path({str(out)!r})\n"
        "(out / 'best.ckpt').write_text('partial')\n"
        "time.sleep(30)\n"
        "write_complete(out)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        cwd=str(REPO),
        env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONPATH": str(REPO)},
    )
    time.sleep(0.4)
    proc.kill()
    proc.wait(timeout=5)
    assert not (out / "COMPLETE").is_file()
    try:
        require_complete_ckpts([out])
    except SystemExit as exc:
        assert "missing COMPLETE" in str(exc)
    else:
        raise AssertionError("expected REFUSE")
