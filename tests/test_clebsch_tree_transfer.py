import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.models.clebsch_tree_transfer import transfer_clebsch_parameters, calibrate_clebsch_parameters
from su2_nqs.models.literature_clebsch_tree import LiteratureClebschTreeConfig, init_literature_clebsch_tree, make_clebsch_amplitudes, clebsch_layers
from su2_nqs.su2.fusion_tree import FusionTree
from su2_nqs.benchmarks.clebsch_tree_exact import fixed_m_spin_basis
from su2_nqs.benchmarks.exact_diagonalization import spin_squared_matrix

jax.config.update('jax_enable_x64', True)


def test_wider_same_tree_embedding_preserves_every_amplitude_and_gradient_direction():
    tree = FusionTree.balanced(8)
    narrow = LiteratureClebschTreeConfig(tree, multiplicity_cap=2, maximum_q=4)
    wide = LiteratureClebschTreeConfig(tree, multiplicity_cap=5, maximum_q=8)
    p, _ = init_literature_clebsch_tree(jax.random.PRNGKey(2), narrow)
    random, _ = init_literature_clebsch_tree(jax.random.PRNGKey(3), wide)
    transferred, report = transfer_clebsch_parameters(narrow, p, wide, random, mode='embedding')
    x = jnp.asarray(fixed_m_spin_basis(8, 0))
    first, second = make_clebsch_amplitudes(narrow), make_clebsch_amplitudes(wide)
    np.testing.assert_allclose(first(p, x), second(transferred, x), atol=2e-14, rtol=2e-14)
    d = jax.tree.map(jnp.ones_like, p)
    zero = jax.tree.map(jnp.zeros_like, random)
    embedded_d, _ = transfer_clebsch_parameters(narrow, d, wide, zero, mode='embedding')
    a = jax.jvp(lambda z: first(z, x), (p,), (d,))[1]
    b = jax.jvp(lambda z: second(z, x), (transferred,), (embedded_d,))[1]
    np.testing.assert_allclose(a, b, atol=2e-13, rtol=2e-13)
    assert report['exact_physical_ray_preserved']


def test_cross_size_keeps_unmatched_spin_channels_and_remains_singlet():
    small = LiteratureClebschTreeConfig(FusionTree.balanced(4), multiplicity_cap=3, maximum_q=4)
    large = LiteratureClebschTreeConfig(FusionTree.balanced(8), multiplicity_cap=3, maximum_q=4)
    p, _ = init_literature_clebsch_tree(jax.random.PRNGKey(4), small)
    random, _ = init_literature_clebsch_tree(jax.random.PRNGKey(5), large)
    transferred, report = transfer_clebsch_parameters(small, p, large, random)
    # Four-spin source root only has q=0, but the larger model must keep q>0.
    layer = clebsch_layers(large)[1]
    offset = 0
    for key, shape in zip(layer.keys, layer.shapes, strict=True):
        count = int(np.prod(shape))
        if key[2] > 0:
            np.testing.assert_array_equal(transferred['merges'][1][offset:offset+count], random['merges'][1][offset:offset+count])
        offset += count
    np.testing.assert_array_equal(transferred['readout'], random['readout'])
    x = fixed_m_spin_basis(8, 0)
    psi = np.asarray(make_clebsch_amplitudes(large)(transferred, jnp.asarray(x)))
    s2, basis = spin_squared_matrix(8, n_up=4)
    lookup = {row: i for i, row in enumerate(basis)}
    order = [lookup[int(''.join(map(str, row)), 2)] for row in x]
    assert np.linalg.norm(s2[np.ix_(order, order)] @ psi) / np.linalg.norm(psi) < 2e-12
    assert not report['exact_physical_ray_preserved']
    assert not report['projected_small_state_embedding_claim']


def test_recalibration_only_changes_overall_ray_and_invalid_transfers_fail():
    cfg = LiteratureClebschTreeConfig(FusionTree.balanced(8), multiplicity_cap=3, maximum_q=4)
    p, _ = init_literature_clebsch_tree(jax.random.PRNGKey(6), cfg)
    x = fixed_m_spin_basis(8, 0)
    q, report = calibrate_clebsch_parameters(cfg, p, x)
    fn = make_clebsch_amplitudes(cfg)
    a, b = np.asarray(fn(p, x)), np.asarray(fn(q, x))
    np.testing.assert_allclose(a / np.linalg.norm(a), b / np.linalg.norm(b), atol=2e-13)
    assert max(abs(r['std_after'] - 1) for r in report['layers']) < 1e-12
    with pytest.raises(ValueError, match='identical tree'):
        transfer_clebsch_parameters(cfg, p, LiteratureClebschTreeConfig(FusionTree.balanced(4)), p, mode='embedding')
    with pytest.raises(ValueError, match='mode'):
        transfer_clebsch_parameters(cfg, p, cfg, p, mode='truncate')


def test_cost_gate_checks_optional_parent_hashes_and_runs_exact_invariants(tmp_path):
    import json
    from projects.clebsch_tree_nqs.workflows.spin_clebsch_tree_sampled import cost
    from su2_nqs.vmc.checkpoint import load_checkpoint
    from su2_nqs.frontier.artifacts import pytree_sha256
    spec = dict(geometry='square', rows=2, columns=2, j2_over_j1=.5,
                multiplicity_cap=2, maximum_q=2, projection_chunk=2,
                require_gpu=False, seed=12, calibration_seed=13, sampling_seed=14)
    gate = cost(spec, tmp_path / 'source')
    parent = load_checkpoint(tmp_path / 'source' / 'checkpoint.pkl')
    frozen = dict(directory=str(tmp_path / 'source'),
                  registration_sha256=gate['registration_sha256'],
                  checkpoint_sha256=gate['checkpoint_sha256'],
                  params_sha256=pytree_sha256(parent['params']))
    result = cost(dict(spec, seed=21, initialization_parent=frozen), tmp_path / 'target')
    assert result['status'] == 'passed'
    report = json.loads((tmp_path / 'target' / 'initialization.json').read_text())
    assert report['small_projected_physical_state_is_not_embedded']
    assert report['reduced_bank_transfer']['copied_scalar_parameters'] > 0
    with pytest.raises(ValueError, match='parent changed'):
        cost(dict(spec, initialization_parent=dict(frozen, checkpoint_sha256='0' * 64)), tmp_path / 'invalid')


def test_square6_uses_complete_space_group_without_an_illegal_root_swap():
    from projects.clebsch_tree_nqs.workflows.spin_clebsch_tree_sampled import components
    from su2_nqs.models.clebsch_tree_projected import make_projected_clebsch
    cfg, _, _, _, images, order = components(dict(geometry='square', rows=6, columns=6,
        j2_over_j1=.5, multiplicity_cap=2, maximum_q=4, projection_chunk=8))
    fn, report = make_projected_clebsch(cfg, order, images, orbit_chunk=8, fold_root=False)
    assert report['full_group_order'] == report['evaluation_images'] == 288
    assert not report['root_fold']
    # Asking the compiler to look for a valid fold must not invent one.
    _, discovered = make_projected_clebsch(cfg, order, images, orbit_chunk=8)
    assert not discovered['root_fold']
    p, _ = init_literature_clebsch_tree(jax.random.PRNGKey(19), cfg)
    row = np.random.default_rng(20).permutation([0] * 18 + [1] * 18)
    values = np.asarray(jax.jit(fn)(p, jnp.asarray([row, row[images[13]], row[images[111]]])))
    np.testing.assert_allclose(values, values[0], rtol=2e-12, atol=1e-16)
