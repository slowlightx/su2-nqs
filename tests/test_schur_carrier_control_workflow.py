"""Protocol helpers are checked independently of any favorable force outcome."""
import jax
import jax.numpy as jnp
import numpy as np

from projects.clebsch_tree_nqs.workflows.spin_schur_carrier_control import (
    parameter_direction, projective_metric, force_summary,
)


def test_parameter_directions_are_deterministic_and_block_balanced():
    params = {key: {'x': jnp.zeros((count,))} for key, count in
              (('base', 9), ('directions', 17), ('attention', 3))}
    for block in ('joint', 'base', 'attention'):
        first = parameter_direction(params, block, 481)
        second = parameter_direction(params, block, 481)
        for a, b in zip(jax.tree_util.tree_leaves(first), jax.tree_util.tree_leaves(second), strict=True):
            np.testing.assert_array_equal(a, b)
        norms = {key: sum(float(jnp.sum(x*x)) for x in jax.tree_util.tree_leaves(value))
                 for key, value in first.items()}
        assert abs(sum(norms.values())-1) < 1e-14
        for key in norms:
            np.testing.assert_allclose(norms[key], 1/3 if block == 'joint' else float(key == block), atol=1e-14)


def test_projective_metric_matches_explicit_orthogonal_tangent_and_gauge():
    rng = np.random.default_rng(281)
    c = rng.normal(size=32)+1j*rng.normal(size=32)
    t = rng.normal(size=(32, 3))+1j*rng.normal(size=(32, 3))
    q = np.exp(rng.normal(size=32))
    norm = np.sum(abs(c)**2/q)
    perpendicular = t-c[:, None]*(np.sum(c.conj()[:, None]*t/q[:, None], axis=0)/norm)
    expected = np.sum(abs(perpendicular)**2/q[:, None], axis=0)/norm
    np.testing.assert_allclose(projective_metric(c, t, q), expected, atol=2e-14)
    np.testing.assert_allclose(projective_metric(c, t+c[:, None]*(.3+.2j), q), expected, atol=2e-14)


def test_whole_chain_summary_respects_pairing_and_scale_invariance():
    rng = np.random.default_rng(188)
    n = 48; c = rng.normal(size=n)+1j*rng.normal(size=n)
    hc = rng.normal(size=n)+1j*rng.normal(size=n)
    t = rng.normal(size=(n, 3))+1j*rng.normal(size=(n, 3))
    d = rng.normal(size=n); q = np.exp(rng.normal(size=n))
    data = dict(carrier=c, hcarrier=hc, diagonal=d, q=q, groups=np.tile(np.arange(8), 6))
    spec = dict(bootstrap_seed=2, bootstrap_replicates=100)
    scales = np.sqrt(projective_metric(c, t, q))
    result = force_summary(data, t, scales, spec)
    changed = {**data, 'q': q*3.2, 'carrier': c*(.3+.1j), 'hcarrier': hc*(.3+.1j)}
    invariant = force_summary(changed, t*(.3+.1j), scales, spec)
    np.testing.assert_allclose(result['variance_reduction'], invariant['variance_reduction'], rtol=1e-13)
    for name in ('control', 'diagonal'):
        a, b = result['forces'][name], invariant['forces'][name]
        np.testing.assert_allclose(a['pooled'], b['pooled'], atol=2e-14)
        np.testing.assert_allclose(a['max_sample_squared_influence_fraction'], b['max_sample_squared_influence_fraction'], atol=2e-14)
        # Independent finite-difference quadrature-weight influence verifies
        # both complex centering and the ratio-normalization subtraction.
        action = hc if name == 'control' else d*c
        def force(w):
            norm = np.sum(w*abs(c)**2); energy = np.sum(w*c.conj()*action)/norm
            return 2*np.real(np.sum(w[:, None]*(t/scales).conj()*(action-energy*c)[:, None], axis=0))/norm
        w = 1/q; eps = 1e-6; influences = []
        for index in range(n):
            delta = np.eye(1, n, index)[0]*w[index]*eps
            influences.append((force(w+delta)-force(w-delta))/(2*eps))
        squared = np.sum(np.asarray(influences)**2, axis=1)
        np.testing.assert_allclose(a['max_sample_squared_influence_fraction'], squared.max()/squared.sum(), rtol=1e-7)
