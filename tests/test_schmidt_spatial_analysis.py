"""Independent physical controls and immutable-source analysis checks."""
import json
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.schmidt_spatial_analysis import (
    boundary_triplet_control, screening_summary)
from projects.clebsch_tree_nqs.workflows.dmrg_region_probe import _canonical
from projects.clebsch_tree_nqs.workflows.dmrg_archive import file_sha256


def test_boundary_dimers_are_an_exact_highest_weight_triplet_not_QSL():
    control = boundary_triplet_control()
    assert control['q'] == control['m'] == 2
    assert control['raising_norm'] < 1e-14
    assert control['norm_error'] < 1e-14
    np.testing.assert_allclose(control['row_spin_z'], [.5, 0., .5], atol=1e-14, rtol=0)
    assert not control['QSL']


def test_cached_screening_revalidates_physical_records_and_retains_control(tmp_path):
    levels = {str(q): [dict(higher_spin_weight_upper_bound_nominal=0.,
        physical_observable_expectations={'interior_row1': -.2679269324 if q==2 else 0.},
        physical_casimir_expectation=q*(q+2)/4, entanglement_energy=float(q))]
        for q in (2,4,6)}
    worker = dict(status='complete', parent_fingerprint_before='unchanged',
        parent_fingerprint_after='unchanged', readout=dict(cut=18,n_sites=36,groups_partition_prefix=True,
        spatial_spin_groups={name:list(range(6*i,6*(i+1))) for i,name in enumerate(
            ('boundary_row0','interior_row1','boundary_row2'))},
        leading_spin_content=dict(sectors={q:dict(leading_levels=v) for q,v in levels.items()})))
    worker['record_sha256'] = _canonical(worker)
    raw=tmp_path/'worker.json'; raw.write_text(json.dumps(worker))
    profiles={'0':dict(lowest_entanglement_energy=0.),
        **{q:dict(all_levels=v) for q,v in levels.items()}}
    saved=dict(records=[dict(chi=10240,input=str(raw),input_sha256=file_sha256(raw),profiles=profiles)])
    source=tmp_path/'spatial.json'; source.write_text(json.dumps(saved))
    result=screening_summary(source,tmp_path/'analysis')
    assert not result['qsl_specific'] and not result['independent_new_observable']
    assert result['ordinary_boundary_control']['negative_channel_weight'] == pytest.approx(5/12)
    assert result['records'][0]['lowest_triplet']['bound']['nominal_negative_channel_weight_lower_bound'] == pytest.approx(.2679269324/1.5)
    raw.write_text(raw.read_text()+' ')
    with pytest.raises(ValueError,match='cached physical input changed'):
        screening_summary(source,tmp_path/'refused')
