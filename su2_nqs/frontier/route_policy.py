"""Fail-closed research-route policy for production experiment entry points."""

from __future__ import annotations

from collections.abc import Iterable


DIRECT_FUSION_STATUS = "frozen_exact_reference_only"
_FROZEN_TOKENS = (
    "direct_fusion",
    "direct-fusion",
    "fusion_path",
    "fusion-path",
    "fusiongru",
    "fusion_gru",
)

# One authoritative import/runtime registry for the frozen direct-path route.
# Binary ``fusion_tree`` topology is intentionally absent: physical-input CG
# trees reuse it without representing amplitudes in fusion-path coordinates.
FROZEN_DIRECT_FUSION_MODULE_PREFIXES = (
    "su2_nqs.su2.fusion_path",
    "su2_nqs.su2.hubbard_fusion",
    "su2_nqs.su2.hubbard_spinor",
    "su2_nqs.su2.reduced_vector",
    "su2_nqs.models.fusion_gru",
    "su2_nqs.models.hubbard_fusion_gru",
    "su2_nqs.models.tree_gru",
    "su2_nqs.models.compact_tree_gru",
    "su2_nqs.models.factorized_tree_gru",
    "su2_nqs.models.branching_tree_gru",
    "su2_nqs.models.multihead_fusion_gru",
    "su2_nqs.models.persistent_head_fusion_gru",
    "su2_nqs.models.exact_carrier_tree",
    "su2_nqs.models.free_k2_tree",
    "su2_nqs.models.k2_pair_carrier_tree",
    "su2_nqs.models.sparse_causal_tree",
    "su2_nqs.models.causal_boundary_tree",
    "su2_nqs.models.hamiltonian_conditioned_tree",
    "su2_nqs.models.orbit_quotient_gru",
    "su2_nqs.models.compact_tree_trie",
    "su2_nqs.operators.heisenberg_chain",
    "su2_nqs.operators.permutation",
    "su2_nqs.operators.hubbard",
    "su2_nqs.operators.tree_path_connectivity",
    "su2_nqs.benchmarks.optimization",
    "su2_nqs.benchmarks.compare_grus",
    "su2_nqs.benchmarks.coupled_basis",
    "su2_nqs.benchmarks.reduced_vector_reference",
    "su2_nqs.benchmarks.fusion_spatial_bridge",
    "su2_nqs.benchmarks.projected_tree_optimization",
    "su2_nqs.benchmarks.sector_spectroscopy",
    "su2_nqs.benchmarks.triangular_locality",
)

# Abandoned physical-input prototypes remain available as historical/exact
# references, but must not silently re-enter a production campaign.  Keeping
# these separate from direct fusion documents why they are frozen without
# misclassifying the active ClebschTree backbones as fusion-path models.
FROZEN_EXPLORATORY_MODULE_PREFIXES = (
    "su2_nqs.models.shared_cut",
    "su2_nqs.models.spin_clebsch_tree_routes",
    "su2_nqs.models.cell_multiplicity_clebsch_tree",
    "su2_nqs.models.hubbard_reduced_fusion_dag",
    "su2_nqs.models.reduced_fusion_dag",
    "su2_nqs.models.persistent_route_pair_dag",
)

# Backwards-compatible aggregate consumed by production runners and written to
# their provenance records.  New frozen routes belong in one of the two named
# registries above, never directly in this aggregate.
FROZEN_PRODUCTION_MODULE_PREFIXES = (
    FROZEN_DIRECT_FUSION_MODULE_PREFIXES + FROZEN_EXPLORATORY_MODULE_PREFIXES
)


class FrozenResearchRouteError(RuntimeError):
    """Raised when a production run attempts to reactivate a frozen route."""


def require_active_production_method(method: str) -> str:
    """Validate that a requested production method is not direct fusion."""

    normalized = str(method).strip().lower()
    if not normalized:
        raise ValueError("production method name must be nonempty")
    if any(token in normalized for token in _FROZEN_TOKENS):
        raise FrozenResearchRouteError(
            f"{method!r} belongs to the frozen direct-fusion route; "
            "use it only from an explicitly historical exact-reference runner"
        )
    return normalized


def frozen_production_modules(
    module_names: Iterable[str],
) -> tuple[str, ...]:
    """Return frozen-route modules present in an import/runtime name set."""

    return tuple(
        sorted(
            {
                str(name)
                for name in module_names
                if any(
                    str(name) == prefix or str(name).startswith(prefix + ".")
                    for prefix in FROZEN_PRODUCTION_MODULE_PREFIXES
                )
            }
        )
    )


def assert_no_frozen_runtime_modules(
    module_names: Iterable[str],
) -> tuple[str, ...]:
    """Fail closed when a production process imported a frozen-route module."""

    offenders = frozen_production_modules(module_names)
    if offenders:
        raise FrozenResearchRouteError(
            f"frozen research-route modules entered production runtime: {offenders}"
        )
    return offenders


__all__ = [
    "DIRECT_FUSION_STATUS",
    "FROZEN_DIRECT_FUSION_MODULE_PREFIXES",
    "FROZEN_EXPLORATORY_MODULE_PREFIXES",
    "FROZEN_PRODUCTION_MODULE_PREFIXES",
    "FrozenResearchRouteError",
    "assert_no_frozen_runtime_modules",
    "frozen_production_modules",
    "require_active_production_method",
]
