"""JVP budget/conditional-update protocol and fixed chain partition semantics."""
import json
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.spin_finite_schur_jvp import CONFIG,registration,tangent_records


def test_JVP_protocol_keeps_directions_architecture_and_conditional_budget_fixed():
    spec=json.loads(CONFIG.read_text());reg=registration(spec)
    assert spec['methods']==['born','importance'] and spec['steps']==1
    assert spec['finite_difference_microbatches']==1 and spec['H_microbatch']==4
    assert spec['tangent_policy']['maximum_speed_ratio']==2.
    assert spec['protection']['maximum_step_fs']==.01
    assert spec['maximum_q']==4 and spec['multiplicity_width']==2
    assert reg['config']==spec
    assert not any('audit' in path for path in reg['inputs'])
    with pytest.raises(ValueError,match='fresh'):registration({**spec,'seeds':{**spec['seeds'],'audit':2026100234}})
    with pytest.raises(ValueError,match='fixed'):registration({**spec,'finite_difference_microbatches':2})


def test_coherent_records_use_all_nodes_and_exact_predetermined_halves():
    p=np.tile([1.,0.],8);hp=np.tile([0.,1.],8)
    d=np.tile([0.,-.1],8);hd=np.tile([-.1,-10.],8)
    data=dict(carrier=p,hcarrier=hp/2,q=np.ones(16),groups=np.repeat(np.arange(8),2))
    r=tangent_records(data,d,hd,2)
    assert len(r['halves'])==2 and len(r['chains'])==8
    assert r['full']['exact_parent_node_records']==8
    for h in r['halves']:
        assert h['exact_parent_node_records']==4
        assert h['energy_slope_per_site']==pytest.approx(-.1)
        assert h['tangent_fs_speed']==pytest.approx(.1)
