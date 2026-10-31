"""Independent nonunit raw-norm plumbing test, not a physical training run."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


def test_normalized_sampling_H_is_rescaled_to_original_active_coordinates(monkeypatch, tmp_path):
    import projects.clebsch_tree_nqs.workflows.triangle_active_two_step_pilot as module

    def draw(banks, runtime, count, seed, bridge):
        rng = np.random.default_rng(seed)
        return rng.integers(0, 2, (count, 144), dtype=np.int32), {"seed": seed}

    def canonical(params, config, profile):
        return {"w": params["w"] / 2}, {"original_raw_norm": 4.}

    def amplitude(params, row):
        return params["w"][0] + params["w"][1] * (row[0] - .5)

    def h_products(params, rows):
        psi = jax.vmap(lambda row: amplitude(params, row))(rows)
        return psi, -2 * psi, psi**2, psi**2

    def update(params, moments, rows, hpsi, density, learning_rate):
        # H was evaluated in normalized sampling coordinates, whereas the
        # amplitude Jacobian and params use the original coordinate scale.
        expected = -2 * jax.vmap(lambda row: amplitude(params, row))(rows)
        np.testing.assert_allclose(hpsi, expected, atol=1e-15, rtol=1e-15)
        assert np.isfinite(np.asarray(density)).all() and np.all(np.asarray(density) > 0)
        return {"w": params["w"] + jnp.asarray([0., .001])}, (
            moments[0]+1, moments[1], moments[2]), {"accepted": True}

    monkeypatch.setattr(module, "exact_draw", draw)
    monkeypatch.setattr(module, "canonical_active_sampling_bank", canonical)
    monkeypatch.setattr(module, "make_feature_lut_amplitude", lambda *args, **kwargs: amplitude)
    runtime = dict(cfg=None, profile=[], images=np.tile(np.arange(144), (16, 1)),
        H=h_products, update=update, prepare=lambda params: params)
    plan = dict(H_chunk=8, projection_chunk=16, action_chunk=4,
        maximum_peak_device_bytes=70 * 1024**3)
    params = {"w": jnp.asarray([1., .05])}
    zero = {"w": jnp.zeros(2)}
    state = (params, (jnp.asarray(0), zero, zero), np.random.default_rng(1234).bit_generator.state)
    updated, receipt = module.transition(plan, runtime, state, tmp_path / "step1")
    assert receipt["accepted"]
    assert int(updated[1][0]) == 1
