"""Hamiltonian operators with compatibility-preserving lazy exports."""

from importlib import import_module


_EXPORTS = {
    **{
        name: (".heisenberg_chain", name)
        for name in ("HeisenbergChain", "connected_configs", "get_conn")
    },
    **{
        name: (".hubbard", name)
        for name in (
            "HubbardHamiltonian",
            "hubbard_fusion_connectivity",
            "hubbard_fusion_matrix",
            "hubbard_reduced_fusion_matrix",
            "hubbard_reduced_fusion_sparse",
        )
    },
    "pair_spin_energy": (".pair_spin", "pair_spin_energy"),
    **{
        name: (".sz_heisenberg", name)
        for name in ("SzHeisenbergChain", "get_conn_sz")
    },
    **{
        name: (".tree_path_connectivity", name)
        for name in (
            "TreeHeisenbergGraphConnectivityPlan",
            "compile_tree_heisenberg_graph_connectivity_plan",
        )
    },
}


__all__ = [
    "HeisenbergChain",
    "HubbardHamiltonian",
    "SzHeisenbergChain",
    "TreeHeisenbergGraphConnectivityPlan",
    "compile_tree_heisenberg_graph_connectivity_plan",
    "connected_configs",
    "get_conn",
    "get_conn_sz",
    "hubbard_fusion_connectivity",
    "hubbard_fusion_matrix",
    "hubbard_reduced_fusion_matrix",
    "hubbard_reduced_fusion_sparse",
    "pair_spin_energy",
]


def __getattr__(name: str):
    """Resolve a historical operator convenience export on first use."""

    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy compatibility exports in interactive discovery."""

    return sorted(set(globals()) | set(__all__))
