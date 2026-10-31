"""Independent projector, physical cut, gradient and resumability checks."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.coupled_basis import coupled_basis_matrix
from su2_nqs.benchmarks.spin_sector import spin_squared_sz_sparse
from su2_nqs.benchmarks.singlet_vap import singlet_projector_matrix, singlet_isometry, physical_cut_metrics
from su2_nqs.su2.fusion_path import enumerate_paths
from su2_nqs.vmc.exact_fs_adam import train_exact_fs_adam
from su2_nqs.models.fixed_sz_spatial_vit import FixedSzSpatialViTConfig, init_fixed_sz_spatial_vit
from projects.clebsch_tree_nqs.workflows.spin import _prepare_case
from projects.clebsch_tree_nqs.workflows.spin_full_vit_vap import prepare_vap


@pytest.mark.parametrize("n", [2,4,6,8])
def test_projector_matches_independent_cg_isometry(n):
    s2,basis = spin_squared_sz_sparse(n)
    indices = np.asarray(basis)@(1 << np.arange(n-1,-1,-1))
    u = coupled_basis_matrix(enumerate_paths(n,0),0,0)[indices]
    p = singlet_projector_matrix(s2.toarray(), n_sites=n)
    np.testing.assert_allclose(p, u@u.conj().T, atol=3e-14, rtol=0)
    np.testing.assert_allclose(p@p,p,atol=3e-14,rtol=0)
    q = singlet_isometry(s2.toarray(),n_sites=n)
    np.testing.assert_allclose(q@q.conj().T,u@u.conj().T,atol=3e-14,rtol=0)


def test_projector_rejects_wrong_spectrum_and_size():
    with pytest.raises(ValueError):
        singlet_projector_matrix(np.diag([0.,.7]), n_sites=4)
    with pytest.raises(ValueError):
        singlet_projector_matrix(np.eye(2), n_sites=36)


@pytest.mark.parametrize("geometry", ["square","triangular"])
def test_projected_vit_gradient_and_tangent_stay_singlet(geometry):
    # Full triangular point-group A1 needs a compatible rhombic torus;
    # rectangular 2x4 intentionally has no gamma_a1 table in the lattice API.
    rows = 4 if geometry == "triangular" else 2
    data = _prepare_case(dict(geometry=geometry,rows=rows,columns=4,
        j2_over_j1=.125, exact_energy_anchor=-100.), "fixed_m0")
    config = FixedSzSpatialViTConfig(geometry=geometry,rows=rows,columns=4,
        patch_rows=rows//2,patch_columns=2,embedding_dim=4,head_count=2)
    params = init_fixed_sz_spatial_vit(jax.random.PRNGKey(43),config)
    problem = prepare_vap(data,config)
    tangent = jax.tree_util.tree_map(lambda x: jnp.ones_like(x)*.01,params)
    psi,derivative = jax.jvp(problem["projected"],(params,),(tangent,))
    np.testing.assert_allclose(problem["s2"]@psi,0.,atol=2e-12)
    np.testing.assert_allclose(problem["s2"]@derivative,0.,atol=2e-12)
    loss = lambda p: jnp.vdot(problem["projected"](p),jnp.asarray(problem["h"])@problem["projected"](p)).real
    got = jax.jvp(loss,(params,),(tangent,))[1]
    shifts = [jax.tree_util.tree_map(lambda p,t:p+sign*1e-4*t,params,tangent) for sign in (-1,1)]
    expected = (loss(shifts[1])-loss(shifts[0]))/2e-4
    np.testing.assert_allclose(got,expected,atol=2e-8,rtol=1e-5)


def test_physical_cut_of_singlet_product_has_correct_local_q():
    s2,basis = spin_squared_sz_sparse(8)
    # Four disjoint nearest-neighbor singlets (Condon--Shortley signs).
    amplitudes = []
    for b in basis:
        amplitudes.append(np.prod([(b[i]-b[i+1])/np.sqrt(2) for i in (0,2,4,6)]))
    result = physical_cut_metrics(np.asarray(amplitudes),np.asarray(basis),(0,1,2,3))
    assert result["q_probabilities"]["0"] == pytest.approx(1,abs=1e-14)
    assert result["entropy_nats"] == pytest.approx(0,abs=1e-14)
    assert result["su2_commutator_max"] < 1e-14
    cross = physical_cut_metrics(np.asarray(amplitudes),np.asarray(basis),(0,2,4,6))
    assert cross["entropy_nats"] == pytest.approx(np.log(16),abs=1e-14)
    np.testing.assert_allclose(list(cross["q_probabilities"].values()),[2/16,9/16,5/16],atol=1e-14)


def test_exact_fs_adam_resume_and_radius(tmp_path):
    state = lambda p: jnp.array([jnp.cos(p["x"][0]),jnp.sin(p["x"][0])])*jnp.exp(1j*p["x"][1])
    initial = {"x":jnp.array([.8,1.])}
    args = dict(state_fn=state,hamiltonian=np.diag([0.,1.]),identity={"test":1},
                steps=8,learning_rate=.1,radius=.025,record_every=1)
    expected,report = train_exact_fs_adam(initial,checkpoint=tmp_path/"full.pkl",**args)
    def stop(p,row):
        if row["step"] == 3:
            raise RuntimeError("interrupted")
    with pytest.raises(RuntimeError,match="interrupted"):
        train_exact_fs_adam(initial,checkpoint=tmp_path/"resume.pkl",callback=stop,**args)
    actual,resumed = train_exact_fs_adam(initial,checkpoint=tmp_path/"resume.pkl",resume=True,**args)
    np.testing.assert_array_equal(expected["x"],actual["x"])
    assert resumed["full_H_calls"] == report["full_H_calls"]
    assert report["completed"] == 8 and report["accepted"] == 8
    assert all(r["fs_sine"] <= .025 for r in report["history"])
    assert report["history"][-1]["energy"] < report["history"][0]["energy"]
    with pytest.raises(ValueError,match="contract"):
        train_exact_fs_adam(initial,checkpoint=tmp_path/"resume.pkl",resume=True,**{**args,"steps":9})
