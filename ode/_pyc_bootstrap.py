"""Load the Step-6 modules from bytecode after the .py sources were deleted.

The 2026-09-24 16:46 / 2026-09-25 09:50 ``__pycache__`` files are the
last compiled Step-6 tree (bounded heads, track embedding, shared t_lag).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

_CACHE = Path(__file__).resolve().parent / "__pycache__"
_LOADED = False

_PYC = {
    "ode.model": "model.cpython-310.pyc",
    "ode.training_loop": "training_loop.cpython-310.pyc",
    "ode.overfit_test": "overfit_test.cpython-310.pyc",
    "ode.train_multi": "train_multi.cpython-310.pyc",
}


def _ensure_pkg(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]
    sys.modules[name] = mod


def bootstrap() -> None:
    global _LOADED
    if _LOADED:
        return
    here = Path(__file__).resolve().parent
    _ensure_pkg("ode", here)
    _ensure_pkg("ode.data", here / "data")
    import ode.data.dataset  # noqa: F401
    import ode.data.datamodule  # noqa: F401
    import ode.data.transforms  # noqa: F401

    for modname, fname in _PYC.items():
        pyc = _CACHE / fname
        if not pyc.is_file():
            raise FileNotFoundError(f"missing bytecode {pyc}")
        if modname in sys.modules and getattr(sys.modules[modname], "__file__", "").endswith(".py"):
            continue
        mod = types.ModuleType(modname)
        sys.modules[modname] = mod
        spec = importlib.util.spec_from_file_location(modname, pyc)
        spec.loader.exec_module(mod)
    _LOADED = True
