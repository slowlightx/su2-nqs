"""Independent audits and preregistered design/decision integrity."""
from pathlib import Path
import json
import jax
import numpy as np
import pytest
from fnqs.pilots.factorial import prepare,decision,audit,points,PRIMARY,SECONDARY,RESERVED,SEEDS
from fnqs.pilots.engine import init_model
from fnqs.pilots.lineage import validate_splits,sha
from fnqs.variants.size_features import SizeFeatureConfig
from fnqs.hamiltonians import IsingCouplings,dense_hamiltonian
from fnqs.basis import all_bits
from fnqs.sectors import parity_reference


def synthetic(narrow150=.9,wide150=.9,narrow600=.9,wide600=.9):
    values={('narrow',150):narrow150,('wide',150):wide150,('narrow',600):narrow600,('wide',600):wide600}
    return [dict(window=w,step=t,seed=s,primary=dict(chi=v)) for (w,t),v in values.items() for s in SEEDS]


@pytest.mark.parametrize('values,window,coverage,budget',[
    ((.9,.7,.9,.7),'wide',True,False),((.9,.9,.7,.7),'narrow',False,True),
    ((.9,.8,.7,.5),'wide',True,True),((.9,.9,.9,.9),None,False,False)])
def test_decision_consistent_paired_effects(values,window,coverage,budget):
    result=decision(synthetic(*values))
    assert result['selected_window']==window
    assert result['coverage_consistent']==coverage and result['budget_consistent']==budget


def test_one_seed_does_not_define_effect():
    rows=synthetic();next(r for r in rows if r['window']=='wide' and r['step']==600 and r['seed']==113)['primary']['chi']=.1
    assert not decision(rows)['identified_under_declared_budget']
    with pytest.raises(ValueError):decision(rows[:-1])
    with pytest.raises(ValueError):decision(rows+[rows[0]])


def test_sealed_design_matched_tasks_primary_inside_both_intervals(tmp_path):
    output=tmp_path/'experiment';prepare(output)
    p=json.loads((output/'protocol.json').read_text())
    assert len(p['trajectories'])==6
    for t in p['trajectories']:
        assert len(t['train_points'])==5 and len(t['schedule'])==600
        assert t['endpoint_costs']['150']['born_samples']==9600
        assert t['endpoint_costs']['600']['born_samples']==38400
        assert all(len(row)==1 for row in t['schedule'])
        fields=[x[-1] for x in t['train_points']]
        assert all(min(fields)<g[-1]<max(fields) for g in p['primary_points'])
        validate_splits(t['train_points'],p['primary_points']+p['secondary_points'],p['reserved_points'])
    with pytest.raises(FileExistsError):prepare(output)


def test_full_correlations_and_energy_against_direct_pauli_reference(tmp_path):
    config=SizeFeatureConfig(3);params=init_model(jax.random.PRNGKey(7),config)
    n=4;g=[1.,.1,.08,1.03];ref=parity_reference(n,IsingCouplings(*g))
    bits=all_bits(n);z=2*bits-1
    reference_correlations=np.array([np.sum(ref.state**2*np.mean(z*np.roll(z,-r,axis=1),axis=1)) for r in range(n)])
    reference=dict(energy=np.array(ref.energy),chi=np.array(ref.chi_total),state=ref.state,correlations=reference_correlations)
    path=tmp_path/'rows.npz';result=audit(params,n,g,config,reference,path)
    with np.load(path,allow_pickle=False) as data:psi=data['psi'];score=data['score']
    # Kronecker construction of each independent Pauli ZZ, bit zero has Z=-1.
    identity=np.eye(2);pauli=np.diag([-1.,1.]);direct=[]
    for r in range(n):
        total=0.
        for i in range(n):
            factors=[identity]*n
            if r:
                factors[i]=pauli;factors[(i+r)%n]=pauli
            operator=factors[0]
            for factor in factors[1:]:operator=np.kron(operator,factor)
            total+=psi@operator@psi/n
        direct.append(total)
    np.testing.assert_allclose(result['correlations'],direct,atol=1e-12)
    energy=psi@dense_hamiltonian(n,IsingCouplings(*g))@psi
    np.testing.assert_allclose(result['energy_total'],energy,atol=2e-12)
    np.testing.assert_allclose(result['chi_total'],np.sum(psi**2*(score-np.sum(psi**2*score))**2),atol=1e-12)
    assert result['archive_sha256']==sha(path)
    assert result['normalization_error']<1e-12
    assert result['fd_score_relative_rms_error'][1]<result['fd_score_relative_rms_error'][0]


def test_solution_stage_does_not_run_without_identified_factor(tmp_path):
    from fnqs.artifacts import write_json
    from fnqs.pilots.factorial_followup import prepare_followup
    base=tmp_path/'factorial';base.mkdir()
    write_json(base/'review.json',dict(decision=decision(synthetic())))
    with pytest.raises(ValueError,match='no consistent identified factor'):
        prepare_followup(tmp_path/'solution',base)
