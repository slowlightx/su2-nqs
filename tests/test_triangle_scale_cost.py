"""Size extension keeps the existing charge convention and common arity state."""
import jax
import json
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.triangle_scale_cost import (
    balanced_block_probe_states, collect_accounting, scale_arity_pair, scale_structure,
)
from su2_nqs.models.triangular_mixed_clebsch import (
    MixedClebschConfig, init_mixed_clebsch_family, mixed_clebsch_bank,
    make_mixed_clebsch_amplitude,
)
from su2_nqs.models.triangular_arity_clebsch import init_arity_pair, arity_clebsch_bank

jax.config.update('jax_enable_x64', True)


def test_production_structure_is_not_naively_four_times_the_cost():
    a = scale_structure(MixedClebschConfig(length=6), .125)
    b = scale_structure(MixedClebschConfig(length=12), .125)
    assert a['top_real_parameters'] == 345280
    assert b['top_real_parameters'] == 4104448
    assert b['layer_counts'] == [144, 48, 16, 8, 4, 2, 1]
    assert (a['full_A1_images'], b['full_A1_images']) == (432, 1728)
    assert (a['Hamiltonian_bonds'], b['Hamiltonian_bonds']) == (216, 864)
    assert not b['physical_state_preserved_by_size_change']


def test_transfer_keeps_trained_local_banks_without_copying_the_old_root():
    source = MixedClebschConfig(length=6, multiplicity_cap=9, maximum_q=3)
    target = MixedClebschConfig(length=12, multiplicity_cap=9, maximum_q=3)
    family = init_mixed_clebsch_family(jax.random.PRNGKey(172001), source)
    old_bank = mixed_clebsch_bank(family['ternary_c3'], source, 'ternary_c3')
    parent, _ = init_arity_pair(old_bank, source, 2)
    pair, note = scale_arity_pair(parent['binary_truncated'], source, target,
        source_arm='binary_truncated', rank=2, key=jax.random.PRNGKey(172002))
    banks = {arm: arity_clebsch_bank(p, target, arm, 2) for arm, p in pair.items()}
    reference = arity_clebsch_bank(parent['binary_truncated'], source, 'binary_truncated', 2)
    for index, node in enumerate(banks['ternary_joint']['local']):
        for a, b in zip(node, reference['local'][index % 4], strict=True):
            np.testing.assert_allclose(a, b, atol=3e-14, rtol=3e-12)
    assert note['copied_eighteen_site_merges'] == list(range(8))
    assert note['same_target_initial_physical_state'] and note['fresh_target_audit_required']
    assert not note['source_root_copied'] and not note['parent_physical_state_preserved']
    for a, b in zip(jax.tree.leaves(banks['binary_truncated']), jax.tree.leaves(banks['ternary_joint']), strict=True):
        np.testing.assert_allclose(a, b, atol=3e-14, rtol=3e-12)
    raw = jax.jit(jax.vmap(make_mixed_clebsch_amplitude(target), in_axes=(None, 0)))
    rng = np.random.default_rng(172003)
    states = balanced_block_probe_states(target, count=2, seed=172003)
    np.testing.assert_array_equal(states.sum(axis=1), [72, 72])
    a = np.asarray(raw(banks['binary_truncated'], states))
    b = np.asarray(raw(banks['ternary_joint'], states))
    assert np.linalg.norm(a) > 1e-100
    np.testing.assert_allclose(a, b, atol=1e-100, rtol=1e-10)
    # A root singlet satisfies S+|psi>=0, not merely fixed Sz=0.
    bra = states[0].copy()
    bra[np.flatnonzero(bra == 0)[0]] = 1
    lowering = np.tile(bra, (73, 1))
    lowering[np.arange(73), np.flatnonzero(bra)] = 0
    ward = np.asarray(raw(banks['ternary_joint'], lowering))
    assert abs(ward.sum())/np.linalg.norm(ward) < 2e-11


def test_size_transfer_rejects_silent_capacity_changes():
    with pytest.raises(ValueError, match='width budgets'):
        scale_arity_pair({}, MixedClebschConfig(),
            MixedClebschConfig(length=12, multiplicity_cap=9),
            source_arm='binary_truncated', rank=8, key=jax.random.PRNGKey(1))


def test_hard_timeout_retains_last_phase_as_censored(tmp_path, monkeypatch):
    output = tmp_path/'cost'
    output.mkdir()
    config = tmp_path/'config.json'
    config.write_text(json.dumps(dict(output=str(output), task='triangle_ClebschTree_size_extension_cost_only')))
    (output/'progress.json').write_text(json.dumps(dict(
        current_operation='complete_H_one_configuration', elapsed_seconds=12.,
        operation_start_elapsed_seconds=13.)))
    monkeypatch.setattr('subprocess.check_output', lambda *a, **k:
        '123|TIMEOUT|0:15|2700|billing=1,cpu=4,gres/gpu=1|\n')
    report = collect_accounting(config, '123')
    assert report['status'] == 'censored'
    assert report['last_operation'] == 'complete_H_one_configuration'
    assert report['operation_start_elapsed_seconds'] == 13.
    assert not report['complete_H_gradient_proven']
    final = json.loads((output/'results.json').read_text())
    assert final['censored'] and final['last_progress']['elapsed_seconds'] == 12.
