"""Exact small-system reference calculations.

Public convenience exports are loaded lazily so a CPU-only reference module
does not initialize JAX or optional accelerator plugins merely by importing
``su2_nqs.benchmarks``.
"""

from importlib import import_module


_EXPORTS = {
    "coupled_basis_matrix": (".coupled_basis", "coupled_basis_matrix"),
    "coupled_state_to_sz": (".coupled_basis", "coupled_state_to_sz"),
    "heisenberg_sz_matrix": (".exact_diagonalization", "heisenberg_sz_matrix"),
    "spin_squared_matrix": (".exact_diagonalization", "spin_squared_matrix"),
    "local_code_decoupling_defect": (
        ".local_indistinguishability",
        "local_code_decoupling_defect",
    ),
    "local_code_qddm_singular_values": (
        ".local_indistinguishability",
        "local_code_qddm_singular_values",
    ),
    "TOMOGRAPHY_VECTOR_NAMES": (
        ".sampled_local_tomography",
        "TOMOGRAPHY_VECTOR_NAMES",
    ),
    "TwoRootLocalTomography": (
        ".sampled_local_tomography",
        "TwoRootLocalTomography",
    ),
    "local_diagonal_rdm_from_moments": (
        ".sampled_local_tomography",
        "local_diagonal_rdm_from_moments",
    ),
    "two_root_three_spin_tomography": (
        ".sampled_local_tomography",
        "two_root_three_spin_tomography",
    ),
    "two_root_tomography_vector": (
        ".sampled_local_tomography",
        "two_root_tomography_vector",
    ),
    "two_root_two_spin_tomography": (
        ".sampled_local_tomography",
        "two_root_two_spin_tomography",
    ),
}

__all__ = [
    "coupled_basis_matrix",
    "coupled_state_to_sz",
    "heisenberg_sz_matrix",
    "local_code_decoupling_defect",
    "local_code_qddm_singular_values",
    "local_diagonal_rdm_from_moments",
    "spin_squared_matrix",
    "TOMOGRAPHY_VECTOR_NAMES",
    "TwoRootLocalTomography",
    "two_root_three_spin_tomography",
    "two_root_tomography_vector",
    "two_root_two_spin_tomography",
]


def __getattr__(name: str):
    """Resolve a public benchmark helper on first use."""

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
