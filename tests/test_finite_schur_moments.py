"""Six-gradient, strict residual protection and conditional-audit protocol."""
import json
import pytest
from projects.clebsch_tree_nqs.workflows.spin_finite_schur_moments import CONFIG,registration,joint_slope_gate


def test_joint_moment_protocol_is_fixed_and_reuses_unmodified_state_and_kernel():
    spec=json.loads(CONFIG.read_text());reg=registration(spec)
    assert len(spec['objectives'])==6 and spec['steps']==1
    assert spec['protection']['maximum_variance_ratio']==1.
    assert spec['protection']['maximum_step_fs']==.01 and spec['matching_bisections']==12
    assert spec['direction_damping']==.1 and spec['maximum_q']==4
    assert not any('audit' in p for p in reg['inputs'])
    with pytest.raises(ValueError,match='STRICT'):registration({**spec,'protection':{**spec['protection'],'maximum_variance_ratio':1.1}})
    with pytest.raises(ValueError,match='fresh'):registration({**spec,'seeds':{**spec['seeds'],'audit':2026100254}})


def test_energy_descent_without_residual_descent_is_not_an_allowed_update():
    row=dict(energy_slope_per_site=-.1,variance_slope_per_site=-.2,tangent_fs_speed=1.)
    records=dict(full=row,halves=[row,row])
    assert joint_slope_gate(records,{'common_descent':True})['eligible']
    failed=joint_slope_gate({**records,'halves':[row,{**row,'variance_slope_per_site':.001}]},{'common_descent':True})
    assert not failed['eligible'] and failed['checks']['energy_descent_full_and_each_half']
    assert not joint_slope_gate(records,{'common_descent':False})['eligible']
