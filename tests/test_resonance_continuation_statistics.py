"""Independent whole-chain statistics oracle; no N36 queries or sampling."""
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.spin_resonance_continuation import primary_audit


def _fixture():
    rng = np.random.default_rng(17101)
    raw = rng.normal(size=(6, 6)) + 1j*rng.normal(size=(6, 6))
    h = (raw+raw.conj().T)/2
    waves = rng.normal(size=(6, 7)) + 1j*rng.normal(size=(6, 7))
    waves[0, 1] = 0.; waves[2, 4] = 0.; waves[5, 6] = 0.
    actions = h @ waves
    # Four equally long independent synthetic chain blocks; repeated
    # configurations are deliberately retained, with different block mixes.
    draws = np.array([[0,1,2,3,4], [1,2,2,4,5], [0,0,1,3,5], [0,2,3,4,5]]).reshape(-1)
    density = np.array([.2, .8, .45, 1.3, .7, .35])
    groups = np.repeat(np.array([4,9,21,55]), 5)
    names = ['carrier','matched_free0','attention0','matched_free1','attention1','matched_free2','attention2']
    return waves[draws], actions[draws], density[draws], groups, names


def _direct(psi, hp, q, names):
    """Amplitude-only oracle, independent of transition-Gram implementation."""
    norm = np.sum(abs(psi)**2/q[:,None], axis=0)
    energy = np.sum(psi.conj()*hp/q[:,None], axis=0)/norm
    variance = np.sum(abs(hp-energy*psi)**2/q[:,None], axis=0)/norm/36
    contrasts = np.array([[(energy[names.index(f'attention{i}')].real-energy[names.index(f'matched_free{i}')].real)/36,
                           variance[names.index(f'attention{i}')]-variance[names.index(f'matched_free{i}')]] for i in range(3)])
    return np.r_[contrasts.ravel(), contrasts.mean(axis=0)]


def _independent_bootstrap(psi, hp, q, groups, names, spec):
    labels = np.unique(groups)
    counts = np.random.default_rng(spec['bootstrap_seed']).multinomial(
        len(labels), np.full(len(labels), 1/len(labels)), size=spec['bootstrap_replicates'])
    replicates = []
    for row in counts:
        selected = np.concatenate([np.tile(np.flatnonzero(groups == label), number)
                                   for label,number in zip(labels,row,strict=True)])
        # Repeat entire chains, not independent observations or start labels.
        replicates.append(_direct(psi[selected], hp[selected], q[selected], names))
    return np.array(replicates)


def test_primary_mean_SE_covariance_and_six_adjusted_intervals_match_direct_chain_resampling():
    psi,hp,q,groups,names = _fixture()
    spec = dict(bootstrap_replicates=128,bootstrap_seed=17102)
    observed = primary_audit(psi,hp,q,groups,names,spec)
    point = _direct(psi,hp,q,names)
    replicates = _independent_bootstrap(psi,hp,q,groups,names,spec)
    labels = [f'pair{i}.{metric}' for i in range(3) for metric in ('energy','variance')]+['mean.energy','mean.variance']
    standard_errors = replicates.std(axis=0,ddof=1)
    nominal = np.quantile(replicates,[.025,.975],axis=0).T
    adjusted = np.quantile(replicates[:,:6],[.05/12,1-.05/12],axis=0).T
    maximum_error = 0.
    for index,name in enumerate(labels):
        result = observed['metrics'][name]
        np.testing.assert_allclose(result['value'],point[index],atol=3e-16)
        np.testing.assert_allclose(result['chain_bootstrap_se'],standard_errors[index],atol=3e-16)
        np.testing.assert_allclose(result['chain_bootstrap_95'],nominal[index],atol=4e-16)
        maximum_error = max(maximum_error,abs(result['value']-point[index]),abs(result['chain_bootstrap_se']-standard_errors[index]))
        if index < 6:
            np.testing.assert_allclose(result['bonferroni_six_comparisons_interval'],adjusted[index],atol=4e-16)
            assert adjusted[index,0] <= nominal[index,0] and adjusted[index,1] >= nominal[index,1]
    np.testing.assert_allclose(observed['bootstrap_covariance'],np.cov(replicates,rowvar=False,ddof=1),atol=3e-17)
    assert observed['primary'] == 'mean.energy' and observed['family_size'] == 6
    assert observed['adjusted_individual_confidence'] == pytest.approx(1-.05/6)
    print('continuation primary/direct whole-chain statistic maximum error',maximum_error)


def test_identical_pair_contrasts_do_not_gain_fake_three_start_statistics():
    psi,hp,q,groups,_ = _fixture()
    free, neural, hfree, hneural = psi[:,1],psi[:,2],hp[:,1],hp[:,2]
    phases = [1.,np.exp(.4j),np.exp(-.7j)]
    waves = np.column_stack([phase*v for phase in phases for v in (free,neural)])
    actions = np.column_stack([phase*v for phase in phases for v in (hfree,hneural)])
    names = [name for i in range(3) for name in (f'matched_free{i}',f'attention{i}')]
    observed = primary_audit(waves,actions,q,groups,names,dict(bootstrap_replicates=64,bootstrap_seed=17103))
    for metric in ('energy','variance'):
        mean = observed['metrics']['mean.'+metric]
        pair = observed['metrics']['pair0.'+metric]
        assert pair['chain_bootstrap_se'] > 1e-5
        np.testing.assert_allclose(mean['value'],pair['value'],atol=3e-16)
        np.testing.assert_allclose(mean['chain_bootstrap_se'],pair['chain_bootstrap_se'],atol=3e-16)
        assert not np.isclose(mean['chain_bootstrap_se'],pair['chain_bootstrap_se']/np.sqrt(3),rtol=.01)


def test_configuration_nodes_contribute_H_squared_even_though_their_energy_term_is_zero():
    psi,hp,q,groups,names = _fixture()
    nodes = psi == 0
    assert np.count_nonzero(nodes) >= 3 and np.linalg.norm(hp[nodes]) > 1
    spec = dict(bootstrap_replicates=20,bootstrap_seed=17104)
    correct = primary_audit(psi,hp,q,groups,names,spec)
    wrong = primary_audit(psi,np.where(nodes,0.,hp),q,groups,names,spec)
    energy_changes = []
    variance_changes = []
    for index in range(3):
        energy_changes.append(correct['metrics'][f'pair{index}.energy']['value']-wrong['metrics'][f'pair{index}.energy']['value'])
        variance_changes.append(correct['metrics'][f'pair{index}.variance']['value']-wrong['metrics'][f'pair{index}.variance']['value'])
    np.testing.assert_allclose(energy_changes,0.,atol=1e-16)
    assert np.linalg.norm(variance_changes) > 1e-3


def test_relabeling_arms_scaling_Q_and_repeating_within_chains_preserve_uncertainty():
    psi,hp,q,groups,names = _fixture()
    spec = dict(bootstrap_replicates=32,bootstrap_seed=17105)
    reference = primary_audit(psi,hp,q,groups,names,spec)
    order = np.array([4,0,6,1,5,2,3])
    changed = primary_audit(np.repeat(psi[:,order],2,axis=0),np.repeat(hp[:,order],2,axis=0),
        np.repeat(q*113.,2),np.repeat(groups,2),[names[i] for i in order],spec)
    for name,entry in reference['metrics'].items():
        for key,value in entry.items():
            np.testing.assert_allclose(changed['metrics'][name][key],value,atol=4e-16)


def test_invalid_importance_weights_or_unequal_chain_masses_are_rejected():
    psi,hp,q,groups,names = _fixture()
    spec = dict(bootstrap_replicates=20,bootstrap_seed=17106)
    for bad in (q*0,-q,q*float('nan')):
        with pytest.raises(ValueError):
            primary_audit(psi,hp,bad,groups,names,spec)
    with pytest.raises(ValueError):
        primary_audit(psi[:-1],hp[:-1],q[:-1],groups[:-1],names,spec)
