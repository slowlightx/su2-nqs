import copy
import json
from pathlib import Path

import pytest

from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_training_plan import materialize,prepare
from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import sealed_json,DEFAULT
from su2_nqs.frontier.provenance import file_sha256


def fixture(tmp_path):
    spec=json.loads(DEFAULT.read_text()); spec['stages']=[dict(projection='gamma_a1',steps=3)]
    parent=tmp_path/'parent'; parent.mkdir()
    reg=sealed_json(parent/'registration.json',dict(config=spec))
    state=parent/'quality/step00003.pkl'; state.parent.mkdir(); state.write_bytes(b'fixture only')
    common=dict(completed_steps=3,status='fixed_steps_complete',final_params_sha256='params',registration_sha256=reg['result_sha256'])
    sealed_json(parent/'progress.json',common); sealed_json(parent/'training.json',common)
    audit=dict(step=3,checkpoint_sha256=file_sha256(state),params_sha256='params',registration_sha256=reg['result_sha256'],audit_projection='gamma_a1',energy={'energy_per_site':-0.5})
    sealed_json(parent/'audits/step00003/energy.json',audit)
    child=copy.deepcopy(spec); child['output']=str(tmp_path/'child'); child['quality_milestones']=[0,3]
    child['audit']['seeds']=[12341,12342]
    plan=dict(parent=dict(directory=str(parent),registration_sha256=reg['result_sha256'],completed_steps=3),child_config=child,generated_config=str(tmp_path/'generated.json'))
    return plan,parent,audit


def test_materialize_has_no_access_to_audit_values(tmp_path):
    p,parent,a=fixture(tmp_path); first=materialize(p)
    a['energy']={'energy_per_site':100}; sealed_json(parent/'audits/step00003/energy.json',a)
    assert materialize(p) == first
    assert first['repair_parameters_from']['params_sha256'] == 'params'
    bad=copy.deepcopy(p); bad['child_config']['model']['embedding_dim']+=2
    with pytest.raises(ValueError,match='changed model'): materialize(bad)
    bad=copy.deepcopy(p); bad['parent']['registration_sha256']='wrong'
    with pytest.raises(ValueError,match='registration'): materialize(bad)
    a['checkpoint_sha256']='wrong'; sealed_json(parent/'audits/step00003/energy.json',a)
    with pytest.raises(ValueError,match='audit'): materialize(p)


def test_parent_must_be_finished_and_plan_bytes_are_frozen(tmp_path):
    p,parent,a=fixture(tmp_path); path=tmp_path/'plan.json'; path.write_text(json.dumps(p))
    digest=file_sha256(path); assert prepare(path,digest) == Path(p['generated_config'])
    assert prepare(path,digest) == Path(p['generated_config'])
    with pytest.raises(ValueError,match='bytes'): prepare(path,'wrong')
    progress=dict(completed_steps=3,status='running',final_params_sha256='params',registration_sha256=p['parent']['registration_sha256'])
    sealed_json(parent/'progress.json',progress)
    with pytest.raises(ValueError,match='incomplete'): materialize(p)


def test_full_physics_parent_audit_is_explicit_not_an_energy_selected_fallback(tmp_path):
    plan,parent,audit=fixture(tmp_path)
    before=materialize(plan)
    audit['training_projection']='gamma_a1'
    full=parent/'audits/step00003/results.json';sealed_json(full,audit)
    plan['parent']['audit_file']='results.json'
    assert materialize(plan)==before
    audit['training_projection']='none';sealed_json(full,audit)
    with pytest.raises(ValueError,match='trained full-A1'):materialize(plan)
    plan['parent']['audit_file']='../results.json'
    with pytest.raises(ValueError,match='declare'):materialize(plan)
    plan['parent']['audit_file']='results.json';full.unlink()
    # Existing energy.json must not rescue a missing explicitly selected file.
    with pytest.raises(FileNotFoundError):materialize(plan)


def test_full_state_continuation_preserves_protocol_and_cumulative_budget(tmp_path):
    p,parent,_=fixture(tmp_path)
    original=p.pop('child_config')
    parent_spec=json.loads((parent/'registration.json').read_text())['config']
    p.update(mode='full_state_continuation',
        continuation=dict(additional_steps=2,time_budget_extension_seconds=600),
        child_overrides=dict(output=original['output'],
            stages=parent_spec['stages']+[dict(projection='gamma_a1',steps=2)],
            quality_milestones=[0,3,5],
            maximum_training_seconds=parent_spec['maximum_training_seconds']+600,
            audit=dict(parent_spec['audit'],seeds=[12341,12342])))
    result=materialize(p)
    assert result['netket_continue_from']['additional_steps'] == 2
    assert result['netket_continue_from']['progress_sha256'] == json.loads((parent/'progress.json').read_text())['result_sha256']
    assert 'repair_parameters_from' not in result
    bad=copy.deepcopy(p); bad['child_overrides']['sr']=dict(parent_spec['sr'],step_size=1.0)
    with pytest.raises(ValueError,match='protocol'): materialize(bad)
    bad=copy.deepcopy(p); bad['child_overrides']['j2_over_j1']=parent_spec['j2_over_j1']+.01
    with pytest.raises(ValueError,match='j2'): materialize(bad)


def test_explicit_cross_coupling_is_not_silent_parameter_repair(tmp_path):
    p,parent,_=fixture(tmp_path)
    p['mode']='coupling_warm_start'
    parent_j2=p['child_config']['j2_over_j1']
    p['child_config']['j2_over_j1']=parent_j2+.05
    p['child_config']['repair_parameters_from']={'stale':'ancestry'}
    result=materialize(p)
    assert result['coupling_warm_start_from']['parent_j2_over_j1'] == parent_j2
    assert 'repair_parameters_from' not in result
    bad=copy.deepcopy(p); bad['child_config']['j2_over_j1']=parent_j2
    with pytest.raises(ValueError,match='different Hamiltonian'): materialize(bad)
    bad=copy.deepcopy(p); bad['child_config']['model']['depth']+=1
    with pytest.raises(ValueError,match='changed model'): materialize(bad)
    bad=copy.deepcopy(p); bad['child_overrides']={}
    with pytest.raises(ValueError,match='exactly one'): materialize(bad)
