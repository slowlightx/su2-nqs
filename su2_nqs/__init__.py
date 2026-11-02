"""Exact-SU(2) neural quantum states in fusion-space coordinates."""

from importlib import import_module


_EXPORTS = {
    name: (".su2.fusion_path", name)
    for name in (
        "allowed_next_steps",
        "count_paths",
        "enumerate_paths",
        "is_valid_path",
    )
}

__all__ = [
    "allowed_next_steps",
    "count_paths",
    "enumerate_paths",
    "is_valid_path",
]


def __getattr__(name: str):
    """Resolve top-level convenience exports without eager submodule imports."""

    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy convenience exports in interactive discovery."""

    return sorted(set(globals()) | set(__all__))
