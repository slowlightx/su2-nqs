from __future__ import annotations

import jax
import numpy as np
import pytest

from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.vmc.scale_first_arms import (
    EXACT_SU2_ARM,
    FIXED_SZ_ARM,
    SCALE_FIRST_ARM_IDS,
    SYMMETRY_UNAWARE_ARM,
    build_scale_first_arm_runtime,
)
from su2_nqs.vmc.matched_work_sr import (
    MatchedWorkSRControls,
    run_resumable_matched_work_sr,
)


jax.config.update("jax_enable_x64", True)


def _runtime(arm_id: str):
    return build_scale_first_arm_runtime(
        arm_id,
        geometry="square",
        rows=2,
        columns=2,
        bonds=square_j1j2_bonds(2, 2, j1=1.0, j2=0.25, periodic=False),
        shared_model={
            "embedding_dim": 8,
            "head_count": 2,
            "depth": 1,
            "mlp_ratio": 1,
            "phase_scale": 0.25,
            "initialization_scale": 0.1,
            "direct_phase_initialization_scale": 1.0,
            "initial_distance_decay": 1.0,
            "use_distance_envelope": True,
        },
        initialization_key=jax.random.PRNGKey(19),
        su2_connected_capacity=16,
    )


def test_scale_first_adapters_share_parameters_and_execute_one_full_h() -> None:
    runtimes = [_runtime(arm_id) for arm_id in SCALE_FIRST_ARM_IDS]
    reference = jax.tree_util.tree_leaves(runtimes[0].params)
    for runtime in runtimes[1:]:
        assert all(
            np.array_equal(np.asarray(left), np.asarray(right))
            for left, right in zip(
                reference,
                jax.tree_util.tree_leaves(runtime.params),
                strict=True,
            )
        )
    for order, runtime in enumerate(runtimes):
        samples = runtime.sample_fn(
            runtime.params, jax.random.PRNGKey(100 + order), 3
        )
        local = runtime.local_energy_fn(runtime.params, samples)
        logs = runtime.log_amplitude_fn(runtime.params, samples)
        assert local.bond_count == runtimes[0].bond_count
        assert local.overflow_count == 0
        assert np.all(np.isfinite(np.asarray(local.local_energies)))
        assert np.all(np.isfinite(np.asarray(logs)))


def test_scale_first_adapters_keep_the_three_spin_constraints_distinct() -> None:
    su2 = _runtime(EXACT_SU2_ARM)
    fixed = _runtime(FIXED_SZ_ARM)
    generic = _runtime(SYMMETRY_UNAWARE_ARM)
    assert su2.model_config.target_q == 0
    assert fixed.model_config.target_m2 == 0
    assert generic.model_config.target_m2 is None


def test_scale_first_adapter_rejects_an_unknown_arm() -> None:
    with pytest.raises(ValueError, match="unknown scale-first arm"):
        _runtime("renamed_control")


def test_scale_first_adapter_rejects_malformed_bonds() -> None:
    with pytest.raises(ValueError, match="invalid bond"):
        build_scale_first_arm_runtime(
            EXACT_SU2_ARM,
            geometry="square",
            rows=2,
            columns=2,
            bonds=((0, 1, 1.0, 2.0),),
            shared_model={
                "embedding_dim": 8,
                "head_count": 2,
                "depth": 1,
                "mlp_ratio": 1,
            },
            initialization_key=jax.random.PRNGKey(1),
            su2_connected_capacity=16,
        )


@pytest.mark.parametrize("arm_id", SCALE_FIRST_ARM_IDS)
def test_every_scale_first_arm_runs_the_identical_checkpointed_driver(
    arm_id: str, tmp_path
) -> None:
    runtime = _runtime(arm_id)
    result = run_resumable_matched_work_sr(
        runtime.params,
        sample_fn=runtime.sample_fn,
        local_energy_fn=runtime.local_energy_fn,
        log_amplitude_fn=runtime.log_amplitude_fn,
        controls=MatchedWorkSRControls(
            total_steps=1,
            sample_count=3,
            seed=211,
            damping=0.1,
            cg_tolerance=1.0e-6,
            cg_max_iterations=20,
            step_size=0.001,
            maximum_relative_parameter_step=0.1,
        ),
        checkpoint_directory=tmp_path / arm_id,
        run_identity={"test": "scale_first_common_driver", "arm_id": arm_id},
    )
    assert result["completed_steps"] == 1
    assert result["cumulative_samples"] == 3
    assert result["cumulative_full_h_bond_terms"] == 3 * runtime.bond_count
    assert result["history"][0]["full_h_evaluations"] == 1
