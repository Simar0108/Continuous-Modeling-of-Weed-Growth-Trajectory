"""Import Step-6 modules from restored source. Refuse sourceless bytecode."""

from __future__ import annotations

from pathlib import Path

_CACHE = Path(__file__).resolve().parent / "__pycache__"
_LOADED = False

_REQUIRED_SOURCES = (
    "ode.model",
    "ode.training_loop",
    "ode.overfit_test",
    "ode.train_multi",
)


def bootstrap() -> None:
    """Import the four restored Step-6 modules. Refuse if any .py is missing."""
    global _LOADED
    if _LOADED:
        return
    here = Path(__file__).resolve().parent
    missing = []
    for modname in _REQUIRED_SOURCES:
        stem = modname.rsplit(".", 1)[-1]
        src = here / f"{stem}.py"
        if not src.is_file():
            missing.append(modname)
    if missing:
        raise SystemExit(
            "REFUSE: sourceless bytecode — restore "
            + ", ".join(missing)
            + " before launching."
        )
    import ode.model  # noqa: F401
    import ode.overfit_test  # noqa: F401
    import ode.train_multi  # noqa: F401
    import ode.training_loop  # noqa: F401
    _LOADED = True
