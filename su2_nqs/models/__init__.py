"""Normalized neural wavefunctions with compatibility-preserving lazy exports.

Keeping the package initializer lazy is part of the physical-input dependency
boundary: importing a physical-input tree model must not initialize the
legacy direct-fusion-path models. Every historical public name remains
available and resolves to the same defining module on first access.
"""

from importlib import import_module


_EXPORTS = {
    "sample_compact_tree_indexed": (
        ".compact_tree_gru",
        "sample_compact_tree_indexed",
    ),
    **{
        name: (".fusion_gru", name)
        for name in (
            "FusionGRU",
            "FusionGRUConfig",
            "init_fusion_gru",
            "log_psi",
            "sample",
        )
    },
    **{
        name: (".full_hilbert_gru", name)
        for name in (
            "FullHilbertGRU",
            "FullHilbertGRUConfig",
            "full_hilbert_log_psi",
            "init_full_hilbert_gru",
            "sample_full_hilbert",
        )
    },
    **{
        name: (".hubbard_fusion_gru", name)
        for name in (
            "HubbardFusionGRUConfig",
            "hubbard_log_psi",
            "hubbard_probabilities",
            "init_hubbard_fusion_gru",
            "sample_hubbard_fusion",
        )
    },
    **{
        name: (".multihead_fusion_gru", name)
        for name in (
            "MultiHeadFusionGRUConfig",
            "init_multihead_fusion_gru",
            "multihead_log_psi",
            "multihead_path_probabilities",
            "sample_multihead",
        )
    },
    **{
        name: (".persistent_head_fusion_gru", name)
        for name in (
            "PersistentHeadFusionGRUConfig",
            "init_persistent_head_fusion_gru",
            "persistent_head_log_psi",
            "persistent_head_probabilities",
        )
    },
    **{
        name: (".projected_singlet_agp", name)
        for name in (
            "ProjectedAGPBatch",
            "ProjectedAGPTrainabilityAudit",
            "ProjectedSingletAGPConfig",
            "audit_projected_singlet_agp_batch",
            "checked_projected_singlet_agp_log_amplitudes",
            "init_projected_singlet_agp_parameters",
            "invariant_pairing_fast_amplitudes",
            "projected_singlet_agp_amplitudes",
            "spatially_invariant_pairing_from_parameters",
            "symmetric_pairing_from_parameters",
        )
    },
    **{
        name: (".sz_gru", name)
        for name in (
            "SzGRU",
            "SzGRUConfig",
            "init_sz_gru",
            "sample_sz",
            "sz_log_psi",
        )
    },
    **{
        name: (".symmetry_blind_gru", name)
        for name in (
            "SymmetryBlindGRU",
            "SymmetryBlindGRUConfig",
            "init_symmetry_blind_gru",
            "sample_symmetry_blind",
            "symmetry_blind_log_psi",
        )
    },
    **{
        name: (".token_position_gru", name)
        for name in (
            "TokenPositionGRU",
            "TokenPositionGRUConfig",
            "init_token_position_gru",
            "sample_token_position",
            "token_position_log_psi",
        )
    },
}


# Deliberately retain the historical public set. Physical-input tree APIs
# live in ``su2_nqs.models.physical_tree`` rather than widening a legacy
# convenience namespace whose contents are used by downstream scripts.
__all__ = [
    "FusionGRU",
    "FusionGRUConfig",
    "FullHilbertGRU",
    "FullHilbertGRUConfig",
    "HubbardFusionGRUConfig",
    "MultiHeadFusionGRUConfig",
    "PersistentHeadFusionGRUConfig",
    "ProjectedAGPBatch",
    "ProjectedAGPTrainabilityAudit",
    "ProjectedSingletAGPConfig",
    "SzGRU",
    "SzGRUConfig",
    "SymmetryBlindGRU",
    "SymmetryBlindGRUConfig",
    "TokenPositionGRU",
    "TokenPositionGRUConfig",
    "init_fusion_gru",
    "init_full_hilbert_gru",
    "init_hubbard_fusion_gru",
    "init_multihead_fusion_gru",
    "init_persistent_head_fusion_gru",
    "init_projected_singlet_agp_parameters",
    "init_sz_gru",
    "init_symmetry_blind_gru",
    "init_token_position_gru",
    "log_psi",
    "full_hilbert_log_psi",
    "hubbard_log_psi",
    "hubbard_probabilities",
    "multihead_log_psi",
    "multihead_path_probabilities",
    "persistent_head_log_psi",
    "persistent_head_probabilities",
    "audit_projected_singlet_agp_batch",
    "checked_projected_singlet_agp_log_amplitudes",
    "invariant_pairing_fast_amplitudes",
    "projected_singlet_agp_amplitudes",
    "spatially_invariant_pairing_from_parameters",
    "symmetric_pairing_from_parameters",
    "sample",
    "sample_compact_tree_indexed",
    "sample_full_hilbert",
    "sample_hubbard_fusion",
    "sample_multihead",
    "sample_sz",
    "sample_symmetry_blind",
    "sz_log_psi",
    "symmetry_blind_log_psi",
    "sample_token_position",
    "token_position_log_psi",
]


def __getattr__(name: str):
    """Resolve a historical convenience export on first access."""

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
