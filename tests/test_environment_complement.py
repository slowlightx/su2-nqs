import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.spin_environment_complement import orthogonal_residual
from su2_nqs.benchmarks.spectral_error_tomography import lowest_subspace_state


def test_residualization_preserves_enlarged_ritz_space():
    rng=np.random.default_rng(9027)
    q,_=np.linalg.qr(rng.normal(size=(9,4)))
    d=rng.normal(size=9)
    residual,fraction=orthogonal_residual(d,q)
    np.testing.assert_allclose(q.T@residual,0,atol=2e-14)
    np.testing.assert_allclose(d,q@(q.T@d)+residual,atol=2e-14)
    assert 0<fraction<1
    h=rng.normal(size=(9,9)); h=h+h.T
    old,_=lowest_subspace_state(np.column_stack((q,d)),h)
    new,_=lowest_subspace_state(np.column_stack((q,residual)),h)
    assert abs(np.vdot(old,new))>1-1e-13
    redundant, fraction=orthogonal_residual(q[:,0],q)
    assert np.linalg.norm(redundant)<1e-14 and fraction<1e-28
    with pytest.raises(ValueError,match="orthonormal"):
        orthogonal_residual(d,2*q)


def test_complement_artifact_keeps_the_complete_strong_anchor():
    import json
    from pathlib import Path
    from su2_nqs.frontier.artifacts import canonical_sha256
    from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
    from projects.clebsch_tree_nqs.workflows import spin_environment_complement as workflow

    root=Path(__file__).resolve().parents[1]
    directory=root/"projects/clebsch_tree_nqs/results/spin_environment_complement_v1"
    r=json.loads((directory/"results.json").read_text())
    assert r["result_sha256"]==canonical_sha256({k:v for k,v in r.items() if k!="result_sha256"})
    assert r["bundle_sha256"]==file_sha256(directory/"states.npz")
    assert r["source_manifest"]==transitive_local_source_manifest((Path(workflow.__file__).resolve(),),root=root,package_prefixes=("su2_nqs","projects.clebsch_tree_nqs"))
    assert len(r["cases"])==6 and not r["remote_training_licensed"]
    for row in r["cases"].values():
        assert row["base"]["rank"]==5
        assert row["rank"]==row["matched_krylov"]["rank"]==row["additive"]["rank"]==6
        assert row["residual_orthogonality_max"]<1e-12
        assert row["scramble_anchor_max_error"]<1e-12
        assert 0<=row["outside_norm_fraction"]<=1
        assert row["energy_per_site"]<=row["base"]["energy_per_site"]+1e-12
        assert row["energy_per_site"]>=row["exact_energy_per_site"]-1e-12
        assert len(row["scrambles"])==5
