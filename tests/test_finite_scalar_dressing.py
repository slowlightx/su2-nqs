import numpy as np

from su2_nqs.benchmarks.finite_scalar_dressing import finite_step_moments


def test_finite_norm_energy_and_variance_match_dense_including_carrier_nodes():
    # Uniform exhaustive samples: no assumption that the carrier is nonzero.
    p=np.array([1.,0.,0.,0.])
    d=np.array([0.,1.,.3,0.])
    h=np.array([[.1,1.,0.,0.],[1.,-.2,.5,0.],[0.,.5,.3,1.],[0.,0.,1.,-.4]])
    for epsilon in (-1.,-.5,0.,.5,1.):
        r=finite_step_moments(p,d,h@p,h@d,np.ones(4),epsilon)
        state=p+epsilon*d;norm=state@state;state/=np.sqrt(norm)
        energy=state@h@state;variance=np.linalg.norm(h@state-energy*state)**2
        np.testing.assert_allclose([r["norm_ratio"],r["energy_total"],r["variance_total"]],[norm,energy,variance],atol=2e-14)
        assert r["normalization_quadratic_error"]<2e-14
        if epsilon==0:
            assert r["energy_change_total"]==r["variance_change_total"]==0


def test_finite_dressing_artifact_is_fixed_exact_projected_and_replayable():
    import json
    from pathlib import Path
    from su2_nqs.frontier.artifacts import canonical_sha256
    from su2_nqs.frontier.provenance import file_sha256,transitive_local_source_manifest
    from projects.clebsch_tree_nqs.workflows import spin_finite_environment_dressing as workflow

    root=Path(__file__).resolve().parents[1]
    directory=root/"projects/clebsch_tree_nqs/results/spin_finite_environment_dressing_v1"
    result=json.loads((directory/"results.json").read_text())
    assert result["result_sha256"]==canonical_sha256({k:v for k,v in result.items() if k!="result_sha256"})
    assert result["config_sha256"]==canonical_sha256(result["config"])
    assert result["source_manifest"]==transitive_local_source_manifest((Path(workflow.__file__).resolve(),),root=root,package_prefixes=("su2_nqs","projects.clebsch_tree_nqs"))
    assert not result["remote_training_licensed"]
    assert result["config"]["optimization_steps"]==0
    assert result["sample_count"]==128 and result["independent_chain_count"]==64
    assert [r["seed"] for r in result["replicates"]]==result["config"]["replicate_seeds"]
    assert len(set(result["config"]["replicate_seeds"]))==4
    assert result["bank_cost"]["group_order"]==432
    assert max(result["n16_validation"]["checks"].values())<2e-10
    assert max(result["n36_validation"].values())<2e-9
    assert [r["epsilon"] for r in result["aggregate"]]==result["config"]["epsilon_grid"]
    data=[]
    for row in result["replicates"]:
        assert row["config_sha256"]==result["config_sha256"]
        assert row["source_manifest_sha256"]==canonical_sha256(result["source_manifest"])
        assert row["full_H_sampled_centers"]==row["sample_count"]==32
        path=directory/f"replicate_{row['index']}.npz"
        assert row["bundle_sha256"]==file_sha256(path)
        with np.load(path,allow_pickle=False) as bundle:
            data.append({k:bundle[k] for k in ("psi","direction","hpsi","hdirection","proposal_squared")})
    combined={k:np.concatenate([d[k] for d in data]) for k in data[0]}
    for row in result["aggregate"]:
        replay=finite_step_moments(*(combined[k] for k in combined),row["epsilon"])
        for key,value in replay.items():
            np.testing.assert_allclose(value,row[key],atol=2e-13,rtol=2e-13)
        assert row["normalization_quadratic_error"]<1e-12
        assert row["norm_ratio"]>0 and row["variance_total"]>=0
        if row["epsilon"]==0:
            assert row["energy_change_total"]==row["variance_change_total"]==0
