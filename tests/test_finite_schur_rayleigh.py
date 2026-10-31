"""Full-gradient protocol and summed-slope (not individual-sign) eligibility."""
import json
import numpy as np
import pytest
from projects.clebsch_tree_nqs.workflows.spin_finite_schur_rayleigh import CONFIG, registration, chain_regions, total_slope_gate


def test_full_Rayleigh_protocol_preserves_architecture_and_bounded_independent_decision():
    spec=json.loads(CONFIG.read_text());reg=registration(spec)
    assert spec['steps']==1 and spec['direction_damping']==.1 and spec['matching_bisections']==12
    assert spec['protection']['maximum_step_fs']==.01
    assert (spec['maximum_q'],spec['multiplicity_width'],spec['depth'])==(4,2,2)
    assert not any('audit' in path for path in reg['inputs'])
    with pytest.raises(ValueError,match='fresh'):registration({**spec,'seeds':{**spec['seeds'],'audit':2026100244}})
    with pytest.raises(ValueError,match='one attempt'):registration({**spec,'steps':2})
    with pytest.raises(ValueError,match='physical'):registration({**spec,'j2_over_j1':.1})


def test_chain_regions_cover_individual_chains_and_summed_descent_allows_opposed_coherences():
    regions=chain_regions(np.repeat(np.arange(8),8))
    assert len(regions)==11 and regions['half_0'].sum()==regions['half_1'].sum()==32
    row=dict(energy_slope_per_site=-.2,forward_coherence_per_site=.1,reverse_coherence_per_site=-.3,tangent_fs_speed=1.)
    records=dict(full=row,halves=[row,{**row,'tangent_fs_speed':7.}])
    result=total_slope_gate(records)
    assert result['eligible'] and result['half_speed_ratio']==7.
    assert result['individual_coherence_signs_not_a_physical_gate']
    assert not total_slope_gate({**records,'halves':[row,{**row,'energy_slope_per_site':.1}]})['eligible']
    with pytest.raises(ValueError):chain_regions(np.repeat(np.arange(6),8))
