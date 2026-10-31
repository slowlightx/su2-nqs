from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.vmc.matched_work_sr import (
    MatchedWorkSRControls,
    SampledLocalEnergyBatch,
    run_resumable_matched_work_sr,
)


jax.config.update("jax_enable_x64", True)


def test_matched_work_driver_resumes_without_repeating_h_calls(tmp_path) -> None:
    features = jnp.asarray(
        [[1.0, -0.5], [1.0, -0.1], [1.0, 0.4], [1.0, 0.9]],
        dtype=jnp.float64,
    )
    local = np.asarray([-1.0, -0.2, 0.4, 0.9], dtype=np.complex128)
    params = jnp.asarray([0.2, -0.1], dtype=jnp.float64)
    calls = {"count": 0}

    def sample_fn(_params, _key, count):
        assert count == 4
        return jnp.arange(4, dtype=jnp.int32)[:, None]

    def log_fn(candidate, samples):
        return (features[samples[:, 0]] @ candidate).astype(jnp.complex128)

    def local_fn(_params, samples):
        calls["count"] += 1
        return SampledLocalEnergyBatch(
            local_energies=local[samples[:, 0]],
            bond_count=7,
            evaluation_seconds=0.01,
        )

    controls = MatchedWorkSRControls(
        total_steps=3,
        sample_count=4,
        seed=73,
        damping=0.1,
        cg_tolerance=1.0e-10,
        cg_max_iterations=10,
        step_size=0.01,
        maximum_relative_parameter_step=0.01,
    )
    checkpoint = tmp_path / "matched"
    identity = {"task": "matched_work_resume_test", "arm": "toy"}
    first = run_resumable_matched_work_sr(
        params,
        sample_fn=sample_fn,
        local_energy_fn=local_fn,
        log_amplitude_fn=log_fn,
        controls=controls,
        checkpoint_directory=checkpoint,
        run_identity=identity,
        maximum_steps_this_call=1,
    )
    assert first["completed_steps"] == 1
    assert calls["count"] == 1
    resumed = run_resumable_matched_work_sr(
        params,
        sample_fn=sample_fn,
        local_energy_fn=local_fn,
        log_amplitude_fn=log_fn,
        controls=controls,
        checkpoint_directory=checkpoint,
        run_identity=identity,
    )
    assert resumed["resumed"]
    assert resumed["completed_steps"] == 3
    assert calls["count"] == 3
    assert resumed["cumulative_samples"] == 12
    assert resumed["cumulative_full_h_bond_terms"] == 84
    assert len(resumed["history"]) == 3
    assert all(row["full_h_evaluations"] == 1 for row in resumed["history"])
