import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fnqs.basis import all_bits
from fnqs.diagnostics.response import (bootstrap_response, dense_response,
    response_records, summarize_response, tangent_comparison)
from fnqs.exact import TFIMReference
from fnqs.hamiltonians import IsingCouplings
from fnqs.models import GRUConfig, init_model, log_psi


def kron_operator(n, entries):
    result = np.ones((1, 1))
    for site in range(n):
        result = np.kron(result, entries.get(site, np.eye(2)))
    return result


def independent_pauli_h(n, vector):
    """Independent Kronecker products, including every repeated PBC origin."""
    z, x = np.diag([-1., 1.]), np.array([[0., 1.], [1., 0.]])
    out = np.zeros((1 << n, 1 << n))
    for site in range(n):
        out -= vector[0]*kron_operator(n, {site: z, (site+1) % n: z})
        out -= vector[1]*kron_operator(n, {site: z, (site+2) % n: z})
        out -= vector[2]*kron_operator(n, {site: x, (site+1) % n: x})
        out -= vector[3]*kron_operator(n, {site: x})
    return out


@pytest.mark.parametrize("n", [4, 6, 8, 10])
def test_local_h_tangent_and_source_against_independent_pauli(n):
    cfg = GRUConfig(3)
    params = init_model(jax.random.PRNGKey(219), cfg)
    bits = all_bits(n).astype(np.uint8)
    g = np.array([1., -.11, .07, 1.02])
    direction = (.23, -.17, .31, 1.)
    log = lambda p, b, c: log_psi(p, b, c, cfg)
    kernel = jax.jit(lambda p, b, c: response_records(p, b, c, log, direction))
    records = {k: np.asarray(v) for k, v in kernel(params, bits, g).items()}
    psi = np.exp(np.asarray(log(params, bits, g)))
    h, v = independent_pauli_h(n, g), independent_pauli_h(n, direction)
    np.testing.assert_allclose(records["energy"], (h@psi)/psi, atol=2e-11, rtol=0)
    np.testing.assert_allclose(records["source"], (v@psi)/psi, atol=2e-11, rtol=0)
    np.testing.assert_allclose(records["h_score"], (h@(psi*records["score"]))/psi, atol=2e-11, rtol=0)
    summary = summarize_response(records, psi**2)
    s = records["score"]-np.sum(psi**2*records["score"])
    tangent = psi*s
    energy = psi@h@psi
    raw = (h-energy*np.eye(len(psi)))@tangent+v@psi
    projected = raw-psi*(psi@raw)
    np.testing.assert_allclose(summary["response_residual_norm"], np.linalg.norm(projected), atol=2e-11, rtol=0)
    delta = 5e-5
    def energy_at(c):
        state = np.exp(np.asarray(log(params, bits, c)))
        return state@independent_pauli_h(n, c)@state
    finite = (energy_at(g+delta*np.array(direction))-energy_at(g-delta*np.array(direction)))/(2*delta)
    np.testing.assert_allclose(summary["variational_energy_derivative"], finite, atol=2e-8, rtol=0)


def linear_witness(p, bits, g):
    n = bits.shape[1]
    label = bits.astype(jnp.int32) @ (1 << jnp.arange(n-1, -1, -1))
    psi = p["state"]+p["scale"]*(g[3]-p["h"])*p["tangent"]
    psi = psi/jnp.linalg.norm(psi)
    return jnp.log(psi[label])


@pytest.mark.parametrize("n", [4, 6, 8])
def test_exact_response_and_correct_state_wrong_tangent_witness(n):
    g = IsingCouplings(h=1.02)
    reference = dense_response(n, g)
    np.testing.assert_allclose(reference.energy, TFIMReference(n, g.h).energy(), atol=3e-12, rtol=0)
    np.testing.assert_allclose(reference.chi_total, TFIMReference(n, g.h).fidelity_susceptibility(), atol=3e-12, rtol=0)
    assert reference.residual_norm < 2e-11
    for scale in (1., 1.25, 0.):
        params = {"state": jnp.asarray(reference.state), "tangent": jnp.asarray(reference.tangent),
                  "h": g.h, "scale": scale}
        records = response_records(params, all_bits(n), g.vector(), linear_witness, fd_steps=(1e-5, 5e-6))
        summary = summarize_response(records, reference.state**2)
        np.testing.assert_allclose(summary["energy_total"], reference.energy, atol=1e-11, rtol=0)
        np.testing.assert_allclose(summary["response_relative_residual"], abs(scale-1), atol=2e-11, rtol=0)
        np.testing.assert_allclose(summary["chi_total"], scale**2*reference.chi_total, atol=2e-11, rtol=0)
        np.testing.assert_allclose(summary["hf_derivative_defect"], 0, atol=2e-11, rtol=0)
        assert max(summary["fd_score_rms_error"]) < 2e-9
        comparison = tangent_comparison(reference.state, np.asarray(records["score"]), reference)
        assert comparison["state_infidelity"] < 2e-14
        np.testing.assert_allclose(comparison["tangent_relative_error"], abs(scale-1), atol=2e-12)


def test_gauge_invariance_bootstrap_and_zero_source():
    cfg = GRUConfig(3)
    params = init_model(jax.random.PRNGKey(219), cfg)
    records = {k: np.asarray(v, dtype=complex) for k, v in response_records(params, all_bits(4),
        np.array([1., 0., 0., 1.]), lambda p, b, g: log_psi(p, b, g, cfg)).items()}
    gauge = .7+.4j
    shifted = {k: v.copy() for k, v in records.items()}
    shifted["score"] += gauge
    shifted["fd_scores"] += gauge
    shifted["h_score"] += gauge*records["energy"]
    before, after = summarize_response(records), summarize_response(shifted)
    for key in ("chi_total", "response_residual_norm", "hf_derivative_defect"):
        np.testing.assert_allclose(before[key], after[key], atol=5e-14, rtol=0)
    boot = bootstrap_response(records, 8, 12)
    assert boot == bootstrap_response(records, 8, 12)
    assert all(x >= 0 for x in boot["bootstrap_se"].values())
    constant = {k: np.zeros_like(v) for k, v in records.items()}
    result = summarize_response(constant)
    assert result["response_relative_residual"] is None
    assert result["fd_score_relative_rms_error"] is None
    with pytest.raises(ValueError, match="source vanishes"):
        bootstrap_response(constant, 1)


def test_invalid_response_inputs():
    log = lambda p, b, g: jnp.ones(len(b))*g[3]
    for direction, steps in [((0, 0, 0, 0), (1e-3,)), ((0, 0, 1), (1e-3,)), ((0, 0, 0, 1), (0.,))]:
        with pytest.raises(ValueError):
            response_records({}, all_bits(4), np.ones(4), log, direction, steps)
    invalid = response_records({}, np.full((2, 4), 2), np.ones(4), log)
    with pytest.raises(ValueError, match="nonfinite"):
        summarize_response(invalid)
    with pytest.raises(ValueError):
        dense_response(12, IsingCouplings())
