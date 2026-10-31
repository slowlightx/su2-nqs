"""SU(2) primitives with compatibility-preserving lazy public exports.

Submodules are independent mathematical layers. In particular, importing a
Clebsch--Gordan table or generic tree no longer initializes direct fusion-path
coordinates or Hubbard spinor reference machinery.
"""

from importlib import import_module


def _exports(module: str, *names: str) -> dict[str, tuple[str, str]]:
    return {name: (module, name) for name in names}


_EXPORTS = {
    **_exports(
        ".fusion_path",
        "allowed_next_steps",
        "count_paths",
        "enumerate_paths",
        "is_valid_path",
        "next_step_mask",
    ),
    **_exports(
        ".hubbard_fusion",
        "HubbardFusionState",
        "count_hubbard_fusion_states",
        "decode_hubbard_fusion_state",
        "encode_hubbard_fusion_state",
        "enumerate_hubbard_fusion_states",
        "is_valid_hubbard_fusion_state",
    ),
    **_exports(
        ".hubbard_spinor",
        "Q_OP_FERMION_SPINOR",
        "SPINOR_COMPONENTS",
        "canonical_hopping_parity",
        "directed_hopping_reduced_element",
    ),
    **_exports(".quantum_numbers", "coupling_range", "is_admissible_triangle"),
    **_exports(
        ".reduced_vector",
        "Q_OP_VECTOR",
        "VECTOR_DELTA_Q",
        "advance_weighted_spin_vector",
        "appended_spin_scalar_contraction_coefficient",
        "appended_spin_vector_source_coefficient",
        "compile_weighted_spin_vector",
        "contract_reduced_vector_scalar",
        "old_block_vector_advance_coefficient",
    ),
    **_exports(
        ".reduced_tensor_product",
        "ReducedTensorTable",
        "VECTOR_PRODUCT_RANKS",
        "advance_reduced_vector_product",
        "appended_spin_product_source_coefficient",
        "compile_weighted_spin_vector_jordan",
        "compile_weighted_spin_vector_product",
        "couple_reduced_vectors",
        "couple_reduced_vectors_commutator",
        "couple_reduced_vectors_jordan",
        "contract_reduced_tensors_scalar",
        "old_block_tensor_advance_coefficient",
        "old_vector_appended_spin_product_coefficient",
        "tensor_delta_q",
    ),
}


__all__ = [
    "allowed_next_steps",
    "advance_reduced_vector_product",
    "advance_weighted_spin_vector",
    "appended_spin_product_source_coefficient",
    "appended_spin_scalar_contraction_coefficient",
    "appended_spin_vector_source_coefficient",
    "compile_weighted_spin_vector",
    "compile_weighted_spin_vector_jordan",
    "compile_weighted_spin_vector_product",
    "couple_reduced_vectors",
    "couple_reduced_vectors_commutator",
    "couple_reduced_vectors_jordan",
    "contract_reduced_tensors_scalar",
    "count_paths",
    "count_hubbard_fusion_states",
    "contract_reduced_vector_scalar",
    "coupling_range",
    "canonical_hopping_parity",
    "decode_hubbard_fusion_state",
    "encode_hubbard_fusion_state",
    "enumerate_hubbard_fusion_states",
    "HubbardFusionState",
    "directed_hopping_reduced_element",
    "enumerate_paths",
    "is_admissible_triangle",
    "is_valid_path",
    "is_valid_hubbard_fusion_state",
    "next_step_mask",
    "old_block_vector_advance_coefficient",
    "old_block_tensor_advance_coefficient",
    "old_vector_appended_spin_product_coefficient",
    "Q_OP_VECTOR",
    "Q_OP_FERMION_SPINOR",
    "ReducedTensorTable",
    "tensor_delta_q",
    "SPINOR_COMPONENTS",
    "VECTOR_DELTA_Q",
    "VECTOR_PRODUCT_RANKS",
]


def __getattr__(name: str):
    """Resolve a historical SU(2) convenience export on first use."""

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
