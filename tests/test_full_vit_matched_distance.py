"""Frozen-ray and protocol boundaries; no optimizer or long training."""
import json
import numpy as np
import jax.numpy as jnp
from su2_nqs.frontier.artifacts import pytree_sha256
from projects.clebsch_tree_nqs.workflows.spin_full_vit_matched_distance import CONFIG, ray_params, candidate_name


def test_rescaled_rays_keep_parent_and_direction():
    parent = {'a': jnp.asarray([.1, -.2]), 'b': jnp.asarray([[.3]])}
    end = {'a': parent['a']+jnp.asarray([.001, .03]), 'b': parent['b']-.02}
    sha = pytree_sha256(parent)
    for alpha in (0, .25, 2, 4):
        p = ray_params(parent, end, alpha)
        for k in parent:
            np.testing.assert_allclose(p[k]-parent[k], alpha*(end[k]-parent[k]), atol=1e-15)
    assert pytree_sha256(parent) == sha


def test_fixed_radius_budget_and_fresh_calibration_audit_seeds():
    spec = json.loads(CONFIG.read_text())
    seeds = [s['seed'] for key in ('calibration', 'validation') for s in spec[key]['streams']]
    assert len(seeds) == len(set(seeds)) == 4
    # Numeric seed identifiers are not experiment dates. Check every prior
    # registered config here, not just the immediate ancestor.
    for path in CONFIG.parent.glob('*.json'):
        if path != CONFIG:
            assert not any(str(seed) in path.read_text() for seed in seeds), path
    cal, val = [len(spec[k]['streams'])*spec[k]['snapshots']*spec[k]['chains'] for k in ('calibration', 'validation')]
    names = [candidate_name(seed, arm, target) for seed in spec['training_seeds']
             for arm in spec['directions'] for target in spec['target_fs_sines']]
    assert len(set(names)) == 8 and cal+val+len(names)*val == 9728
    assert spec['bootstrap']['family_tests'] == 2*(8+4)
