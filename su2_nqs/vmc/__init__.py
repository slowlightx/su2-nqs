"""Variational Monte Carlo utilities with compatibility-preserving lazy exports.

The lazy package boundary prevents importing one physical-input VMC action
from initializing unrelated legacy direct-fusion samplers. Historical public
names and their defining modules are unchanged.
"""

from importlib import import_module


def _exports(module: str, *names: str) -> dict[str, tuple[str, str]]:
    return {name: (module, name) for name in names}


_EXPORTS = {
    **_exports(".adam", "train_adam"),
    **_exports(
        ".adaptive_tree",
        "AdaptiveTreeExecution",
        "AdaptiveTreeVMCGradient",
        "BucketedTreeConnectivityBatch",
        "DEFAULT_TREE_CONNECTIVITY_BUCKETS",
        "ParallelTreeSampleBatch",
        "TreeConnectivityBucket",
        "adaptive_compact_tree_vmc_energy_gradient_from_connectivity",
        "build_bucketed_tree_connectivity",
        "compact_tree_vmc_energy_gradient_auto",
        "sample_compact_tree_data_parallel",
    ),
    **_exports(".checkpoint", "load_checkpoint", "save_checkpoint"),
    **_exports(
        ".resumable_checkpoint",
        "RESUMABLE_CHECKPOINT_SCHEMA_VERSION",
        "canonical_json_sha256",
        "load_latest_resumable_checkpoint",
        "load_resumable_checkpoint",
        "save_resumable_checkpoint_generation",
        "save_resumable_checkpoint",
    ),
    **_exports(
        ".constrained_uniform",
        "fusion_completion_count",
        "sample_uniform_fusion_paths",
        "sample_uniform_sz_configs",
        "uniform_fusion_action_probabilities",
        "uniform_fusion_log_probability",
        "uniform_fusion_path_probability",
        "uniform_sz_log_probability",
    ),
    **_exports(".gradients", "energy_gradient", "vmc_loss"),
    **_exports(
        ".grassmann_sr",
        "GrassmannEnergySRDirection",
        "GrassmannSRDirection",
        "exact_grassmann_energy_sr_direction",
        "exact_grassmann_sr_direction",
    ),
    **_exports(
        ".local_energy", "energy_and_variance", "local_energies", "local_energy"
    ),
    **_exports(
        ".instrumented_sr",
        "InstrumentedSRResult",
        "conjugate_gradient_with_count",
        "train_instrumented_padded_sparse_sr",
    ),
    **_exports(
        ".multistate",
        "exact_ky_fan_loss",
        "exact_projector_infidelity_loss",
        "exhaustive_state_matrix",
        "generalized_ritz_analysis",
        "ky_fan_subspace_energy",
        "subspace_projector_overlap",
        "subspace_principal_fidelities",
    ),
    **_exports(".padded_sparse", "train_padded_sparse_sr"),
    **_exports(
        ".projected_agp",
        "DirectAmplitudeRayleighEstimate",
        "ProjectedAGPHeisenbergBatch",
        "ProjectedAGPHeisenbergConfig",
        "direct_amplitude_rayleigh_estimate",
        "projected_agp_heisenberg_action",
        "require_valid_direct_amplitude_estimate",
    ),
    **_exports(
        ".direct_amplitude_multistate_force",
        "DirectAmplitudeKyFanForce",
        "DirectAmplitudeKyFanForceReference",
        "direct_amplitude_ky_fan_horizontal_force",
        "prepare_direct_amplitude_ky_fan_force_reference",
    ),
    **_exports(
        ".resumable_tree_adam",
        "ResumableTreeAdamResult",
        "pytree_sha256",
        "run_resumable_compact_tree_adam",
    ),
    **_exports(
        ".resumable_tree_multistate_adam",
        "ResumableThreeRootTreeAdamResult",
        "run_resumable_three_root_tree_adam",
    ),
    **_exports(
        ".sampled_multistate",
        "EXACT_OVERLAP_DIAGONAL",
        "FailClosedGeneralizedRitz",
        "HermiticityDiagnostics",
        "NumericalSubspaceRankError",
        "SAMPLED_OVERLAP_DIAGONAL",
        "SampledPhysicalResiduals",
        "SampledSubspaceMoments",
        "fail_closed_generalized_ritz",
        "sampled_physical_residuals",
        "sampled_subspace_moments",
    ),
    **_exports(
        ".sampled_multistate_force",
        "SampledKyFanForceReference",
        "SampledKyFanHorizontalForce",
        "prepare_sampled_ky_fan_force_reference",
        "sampled_ky_fan_horizontal_force",
    ),
    **_exports(
        ".sampled_folded_force",
        "SampledFoldedForce",
        "SampledFoldedForceReference",
        "prepare_sampled_folded_force_reference",
        "sampled_folded_spectrum_force",
    ),
    **_exports(
        ".tree_multistate_gate",
        "D1A_TREE_GATE_PROPOSAL",
        "D1A_TREE_GATE_SCIENTIFIC_SCOPE",
        "D1_TREE_GATE_DEFAULT_EXACT_NORMALIZED_OVERLAP_DIAGONAL",
        "StratifiedThreeRootTreeSamples",
        "ThreeRootTreeExecution",
        "ThreeRootTreeRitzForceGate",
        "compact_tree_three_root_ritz_force_gate",
        "sample_stratified_three_root_compact_tree",
    ),
    **_exports(
        ".transition_moments",
        "EffectiveSampleSizeDiagnostics",
        "JointBlockBootstrap",
        "NonlinearBootstrapSummary",
        "TransitionMomentEstimate",
        "estimate_transition_moments",
        "flatten_complex_moments",
        "hermitize_transition_moments",
        "joint_block_bootstrap",
        "stratified_observation_weights",
        "unflatten_complex_moments",
    ),
}

# ``local_energy`` is both a public function name and a submodule basename.
# Import machinery assigns the submodule object to ``vmc.local_energy`` when
# another helper loads it, bypassing module ``__getattr__`` thereafter. Bind
# this one safe (direct-fusion-independent) collision eagerly so the historic
# function export remains stable regardless of submodule import order.
local_energy = getattr(import_module(".local_energy", __name__), "local_energy")


__all__ = [
    "AdaptiveTreeExecution",
    "AdaptiveTreeVMCGradient",
    "BucketedTreeConnectivityBatch",
    "D1A_TREE_GATE_PROPOSAL",
    "D1A_TREE_GATE_SCIENTIFIC_SCOPE",
    "D1_TREE_GATE_DEFAULT_EXACT_NORMALIZED_OVERLAP_DIAGONAL",
    "DirectAmplitudeRayleighEstimate",
    "DirectAmplitudeKyFanForce",
    "DirectAmplitudeKyFanForceReference",
    "DEFAULT_TREE_CONNECTIVITY_BUCKETS",
    "EffectiveSampleSizeDiagnostics",
    "EXACT_OVERLAP_DIAGONAL",
    "ParallelTreeSampleBatch",
    "ProjectedAGPHeisenbergBatch",
    "ProjectedAGPHeisenbergConfig",
    "RESUMABLE_CHECKPOINT_SCHEMA_VERSION",
    "ResumableTreeAdamResult",
    "ResumableThreeRootTreeAdamResult",
    "energy_and_variance",
    "energy_gradient",
    "direct_amplitude_rayleigh_estimate",
    "direct_amplitude_ky_fan_horizontal_force",
    "fusion_completion_count",
    "exact_grassmann_sr_direction",
    "exact_ky_fan_loss",
    "exact_projector_infidelity_loss",
    "exhaustive_state_matrix",
    "FailClosedGeneralizedRitz",
    "fail_closed_generalized_ritz",
    "generalized_ritz_analysis",
    "GrassmannSRDirection",
    "GrassmannEnergySRDirection",
    "HermiticityDiagnostics",
    "InstrumentedSRResult",
    "JointBlockBootstrap",
    "TreeConnectivityBucket",
    "adaptive_compact_tree_vmc_energy_gradient_from_connectivity",
    "build_bucketed_tree_connectivity",
    "canonical_json_sha256",
    "compact_tree_three_root_ritz_force_gate",
    "compact_tree_vmc_energy_gradient_auto",
    "sample_compact_tree_data_parallel",
    "exact_grassmann_energy_sr_direction",
    "ky_fan_subspace_energy",
    "load_checkpoint",
    "load_latest_resumable_checkpoint",
    "load_resumable_checkpoint",
    "local_energies",
    "local_energy",
    "NumericalSubspaceRankError",
    "NonlinearBootstrapSummary",
    "save_checkpoint",
    "save_resumable_checkpoint_generation",
    "save_resumable_checkpoint",
    "pytree_sha256",
    "prepare_sampled_ky_fan_force_reference",
    "prepare_direct_amplitude_ky_fan_force_reference",
    "projected_agp_heisenberg_action",
    "require_valid_direct_amplitude_estimate",
    "run_resumable_compact_tree_adam",
    "run_resumable_three_root_tree_adam",
    "SampledKyFanForceReference",
    "SampledKyFanHorizontalForce",
    "sampled_ky_fan_horizontal_force",
    "SampledFoldedForce",
    "SampledFoldedForceReference",
    "prepare_sampled_folded_force_reference",
    "sampled_folded_spectrum_force",
    "SampledPhysicalResiduals",
    "SAMPLED_OVERLAP_DIAGONAL",
    "SampledSubspaceMoments",
    "sample_stratified_three_root_compact_tree",
    "sampled_physical_residuals",
    "sampled_subspace_moments",
    "sample_uniform_fusion_paths",
    "sample_uniform_sz_configs",
    "subspace_projector_overlap",
    "subspace_principal_fidelities",
    "StratifiedThreeRootTreeSamples",
    "ThreeRootTreeExecution",
    "ThreeRootTreeRitzForceGate",
    "TransitionMomentEstimate",
    "train_adam",
    "conjugate_gradient_with_count",
    "train_instrumented_padded_sparse_sr",
    "train_padded_sparse_sr",
    "estimate_transition_moments",
    "flatten_complex_moments",
    "hermitize_transition_moments",
    "joint_block_bootstrap",
    "stratified_observation_weights",
    "unflatten_complex_moments",
    "uniform_fusion_action_probabilities",
    "uniform_fusion_log_probability",
    "uniform_fusion_path_probability",
    "uniform_sz_log_probability",
    "vmc_loss",
]


def __getattr__(name: str):
    """Resolve a historical convenience export on first use."""

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
