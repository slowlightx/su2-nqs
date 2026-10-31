"""Saved-array workflow integrity, without requiring a favorable physics result."""
from pathlib import Path
import json

import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.analyze_carrier_residuals import (
    DEFAULT, ROOT, configuration_orbits, label_mass, load_saved_corpora,
)
from su2_nqs.frontier.artifacts import canonical_sha256
from su2_nqs.frontier.provenance import file_sha256


def test_orbit_labels_preserve_repeated_physical_observations():
    states=np.asarray([[1,0,0,0],[0,1,0,0],[1,0,0,0],[1,1,0,0]])
    group=np.asarray([np.roll(np.arange(4),k) for k in range(4)])
    configs,orbits=configuration_orbits(states,group)
    assert len(set(configs))==3 and len(set(orbits))==2
    assert configs[0]=='80' and orbits[0]==orbits[1]==orbits[2]
    fractions=np.asarray([.1,.2,.3,.4])
    assert label_mass(fractions,configs)['max_fraction']==pytest.approx(.4)
    assert label_mass(fractions,orbits)['max_fraction']==pytest.approx(.6)
    with pytest.raises(ValueError):
        configuration_orbits(states,np.asarray([[0,0,2,3]]))
    with pytest.raises(ValueError):
        label_mass(fractions*2,configs)


def test_new_geometry_adapter_replays_saved_units_and_keeps_corpora_distinct():
    corpora,probes,meta,paths=load_saved_corpora(DEFAULT)
    assert all(Path(p).is_file() for p in paths)
    # Carrier unit conversion involves amplitudes much larger than one; judge
    # its floating-point roundoff relative to the actual amplitude scale.
    carrier_scale=np.max(abs(corpora['known_development']['carrier']))
    assert meta['replay']['carrier']/carrier_scale < 4*np.finfo(float).eps
    assert max(v for k,v in meta['replay'].items() if k!='carrier') < 1e-15
    assert not meta['previous_gate']['passed']
    assert [len(corpora[k]['q']) for k in ('fresh_calibration','known_development','combined_development','viewed_audit')]==[128,256,384,128]
    assert 'groups' not in corpora['combined_development']
    assert all('hcandidates' not in corpora[k] for k in ('fresh_calibration','known_development','combined_development'))
    assert probes['known']['tangents'].shape==(256,6)
    for name in ('fresh_calibration','known_development'):
        d=corpora[name]
        np.testing.assert_allclose(np.sum(abs(d['carrier'])**2/d['q']),.5,atol=2e-15)


def test_saved_structure_seals_resources_and_replayed_endpoints():
    folder=DEFAULT/'structure'
    result=json.loads((folder/'results.json').read_text())
    manifest=json.loads((folder/'manifest.json').read_text())
    for record in (result,manifest):
        assert record['sha256']==canonical_sha256({k:v for k,v in record.items() if k!='sha256'})
    assert result['manifest_sha256']==manifest['sha256']
    for section in ('inputs','sources'):
        for name,digest in manifest[section].items():
            assert file_sha256(ROOT/name)==digest,name
    assert file_sha256(folder/'profiles.npz')==result['files']['profiles.npz']
    assert result['post_hoc'] and not result['new_variational_win_claim']
    assert not result['production_authorized'] and not result['source_gate_passed']
    assert not any(result['resources'].values())
    assert result['chart']['initial_carrier_exact']
    assert result['chart']['saved_endpoints_preserved']
    assert result['chart']['global_manifold_equivalence_not_asserted']
    for arm in result['chart']['endpoints'].values():
        assert max(arm.values())<1e-14
    assert len(result['corpora']['viewed_audit']['leave_one_group_out'])==16
    assert not result['corpora']['combined_development']['leave_one_group_out']


def test_saved_structure_retains_all_mass_and_finite_Q_cross_terms():
    folder=DEFAULT/'structure'
    result=json.loads((folder/'results.json').read_text())
    with np.load(folder/'profiles.npz',allow_pickle=False) as profiles:
        for name in ('carrier','reference','force','diagonal','control'):
            assert len(profiles[name])==128
            np.testing.assert_allclose(profiles[name].sum(),1.,atol=2e-15)
        assert set(profiles['groups'])==set(range(16))
        for arm in result['arms']:
            event=result['events'][arm]
            i=event['row']
            assert event['weighted_fraction']==pytest.approx(profiles[arm][i])
            np.testing.assert_array_equal(event['state'],profiles['states'][i])
    audit=result['corpora']['viewed_audit']['pooled']
    for arm in audit['arms']:
        h=arm['hamiltonian']; split=h['reference_split']
        np.testing.assert_allclose(h['delta_energy_per_site'],
            h['interference_left_per_site']+h['interference_right_per_site']+h['quadratic_per_site'],atol=2e-15)
        for side in ('left','right'):
            np.testing.assert_allclose(h[f'interference_{side}_per_site'],
                split[f'{side}_reference_per_site']+split[f'{side}_innovation_per_site'],atol=2e-15)
        np.testing.assert_allclose(h['quadratic_per_site'],
            sum(split[f'quadratic_{piece}_per_site'] for piece in ('reference','innovation','mixed')),atol=2e-15)
    assert not audit['finite_Q_Hermiticity_assumed']
    assert audit['quadrature_is_not_full_support_certification']
    assert result['six_probe_span']['finite_probe_span_not_full_parameter_tangent']
    # Test the fit/evaluation boundary, not a desired physical advantage.
    fitted=result['chart']['prior64_fitted_JVP_transfer']['prior_calibration']
    assert max(row['anchor_plane_fraction'] for row in fitted)<1e-25
