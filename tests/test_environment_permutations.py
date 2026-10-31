from itertools import combinations

import numpy as np

from su2_nqs.benchmarks.environment_tangent import importance_tangent_moments
from su2_nqs.benchmarks.physical_interpretation import FixedSzPairActions
from su2_nqs.operators.cell_spin_channels import four_site_spin_modes, weighted_cell_exchange_action
from su2_nqs.operators.exchange_environment import spectator_environment_action
from su2_nqs.operators.environment_permutations import environment_permutation_bank, evaluate_environment_actions


def test_permutation_actions_match_independent_full_vector_actions():
    basis=[]
    for up in combinations(range(16),8):
        row=np.zeros(16,dtype=np.int8); row[list(up)]=1
        basis.append(tuple(row))
    states=np.asarray(basis)
    cells=((0,1,4,5),(2,3,6,7),(8,9,12,13),(10,11,14,15))
    pairs=FixedSzPairActions(tuple(basis))
    pair=lambda v,i,j:.5*v[pairs.indices(i,j)]-.25*v
    rng=np.random.default_rng(2026092741)
    vector=rng.normal(size=len(states))+1j*rng.normal(size=len(states))
    o=[weighted_cell_exchange_action(vector,pair,*cells[:2],w,w) for w in four_site_spin_modes()[1:]]
    f=spectator_environment_action(vector,pair,*cells[2:],"dimer_correlation")
    of=sum(weighted_cell_exchange_action(f,pair,*cells[:2],w,w) for w in four_site_spin_modes()[1:])/np.sqrt(3)
    powers=1<<np.arange(15,-1,-1)
    lookup=np.zeros(2**16,dtype=complex); lookup[states@powers]=vector
    indices=rng.choice(len(states),32,replace=False)
    permutations,coefficients=environment_permutation_bank(16,cells)
    actual,cost=evaluate_environment_actions(states[indices],lambda x:lookup[x@powers],permutations,coefficients)
    np.testing.assert_allclose(actual,np.column_stack((*o,f,of))[indices],atol=3e-14)
    assert cost["unique_amplitude_requests"]<=cost["raw_requests"]
    np.testing.assert_allclose(coefficients.sum(axis=0),0,atol=3e-14)


def test_full_support_moments_include_carrier_nodes():
    # psi=(1,0), H=O=X; q_raw=(1.2,.2), exact q probabilities=(6/7,1/7).
    psi=np.asarray([1.]*6+[0.])
    hpsi=np.asarray([0.]*6+[1.])
    q=np.asarray([1.2]*6+[.2])
    result=importance_tangent_moments(psi,hpsi,hpsi[:,None],q)
    assert abs(result["energy_total"])<1e-14
    assert abs(result["variance_total"]-1)<1e-14
    assert abs(result["energy_derivative_total"][0]-2)<1e-14
    # Carrier-only samples would incorrectly report zero variance/gradient.
    bad=importance_tangent_moments(psi[:6],hpsi[:6],hpsi[:6,None],q[:6])
    assert bad["variance_total"]==bad["energy_derivative_total"][0]==0


def test_a1_first_derivative_does_not_require_intermediate_projection():
    swap=np.eye(4)[[1,0,3,2]]
    p=(np.eye(4)+swap)/2
    rng=np.random.default_rng(84)
    x=rng.normal(size=(4,4)); h=p@(x+x.T)@p
    x=rng.normal(size=(4,4)); o=x+x.T
    psi=p@rng.normal(size=4); psi/=np.linalg.norm(psi)
    hpsi=h@psi; e=psi@hpsi
    assert abs(2*(hpsi-e*psi)@o@psi-2*(hpsi-e*psi)@p@o@psi)<3e-14
    assert np.linalg.norm(p@o@psi-o@psi)>1e-3


def test_local_preflight_artifacts_are_sealed_and_untrained():
    import json
    from pathlib import Path
    from su2_nqs.frontier.artifacts import canonical_sha256
    from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
    from projects.clebsch_tree_nqs.workflows import spin_environment_preflight as workflow

    root=Path(__file__).resolve().parents[1]
    base=root/"projects/clebsch_tree_nqs/results/spin_environment_preflight_v1"
    seen=set()
    for directory,count in ((base,512),(base/"audit",16384)):
        r=json.loads((directory/"results.json").read_text())
        assert r["result_sha256"]==canonical_sha256({k:v for k,v in r.items() if k!="result_sha256"})
        assert r["source_manifest"]==transitive_local_source_manifest((Path(workflow.__file__).resolve(),),root=root,package_prefixes=("su2_nqs","projects.clebsch_tree_nqs"))
        assert not r["remote_training_licensed"] and not r["hamiltonian_diagonalization_performed"]
        assert r["config"]["optimization_steps"]==0
        assert r["aggregate"]["sample_count"]==count
        assert r["carrier_replay_relative_error"]<2e-12
        assert r["proposal_kappa"]>0
        assert not seen.intersection(r["config"]["replicate_seeds"])
        seen.update(r["config"]["replicate_seeds"])
        for row in r["replicates"]:
            assert row["bundle_sha256"]==file_sha256(directory/f"replicate_{row['index']}.npz")
            with np.load(directory/f"replicate_{row['index']}.npz") as b:
                np.testing.assert_allclose(b["proposal_squared"],np.abs(b["psi"])**2+r["proposal_kappa"],rtol=1e-14,atol=0)
                assert np.all(np.sum(b["configurations"],axis=1)==18)
                assert np.isfinite(b["hpsi"]).all() and np.isfinite(b["actions"]).all()
