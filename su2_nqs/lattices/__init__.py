"""Bond generators for chain and two-dimensional benchmark lattices."""

from importlib import import_module


_EXPORTS = {
    "chain_bonds": (".chain", "chain_bonds"),
    "square_bonds": (".square", "square_bonds"),
    "triangular_bonds": (".triangular", "triangular_bonds"),
}

__all__ = ["chain_bonds", "square_bonds", "triangular_bonds"]


def __getattr__(name: str):
    """Resolve a lattice generator only when requested."""

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
