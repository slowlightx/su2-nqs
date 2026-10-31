import numpy as np
import pytest

from su2_nqs.operators.cell_spin_channels import weighted_cell_exchange_action
from su2_nqs.operators.exchange_environment import spectator_environment_action, environment_modulated_exchange_action


def _swaps(n):
    packed=np.arange(2**n)
    def pair(v,i,j):
        a,b=1<<(n-1-i),1<<(n-1-j)
        different=((packed&a)!=0)!=((packed&b)!=0)
        indices=np.where(different,packed^a^b,packed)
        return .5*v[indices]-.25*v
    return pair


def _active_pauli_reference():
    spin=[]
    for i in range(4):
        components=[]
        for p in (np.array([[0,1],[1,0]])/2,np.array([[0,1j],[-1j,0]])/2,np.diag([-.5,.5])):
            value=np.ones((1,1))
            for j in range(4):
                value=np.kron(value,p if i==j else np.eye(2))
            components.append(value)
        spin.append(components)
    return sum((spin[0][k]-spin[1][k])@(spin[2][k]-spin[3][k]) for k in range(3))


@pytest.mark.parametrize("feature",("spin_content","spin_pair"))
def test_environment_exchange_agrees_with_independent_tensor_product(feature):
    singlet=np.array([0,-1,1,0])/np.sqrt(2)
    casimir=2*(np.eye(4)-np.outer(singlet,singlet))
    f=(np.kron(casimir,np.eye(4))+np.kron(np.eye(4),casimir))/4 if feature=="spin_content" else np.kron(casimir,casimir)/4
    expected=np.kron(_active_pauli_reference(),f)
    pair=_swaps(8)
    observed=environment_modulated_exchange_action(np.eye(256),pair,((0,1),(2,3),(4,5),(6,7)),(1,-1),(1,-1),feature)
    np.testing.assert_allclose(observed,expected,atol=3e-14)
    np.testing.assert_allclose(observed,observed.conj().T,atol=3e-14)
    reverse=spectator_environment_action(weighted_cell_exchange_action(np.eye(256),pair,(0,1),(2,3),(1,-1),(1,-1)),pair,(4,5),(6,7),feature)
    np.testing.assert_allclose(observed,reverse,atol=3e-14)
    with pytest.raises(ValueError,match="disjoint"):
        environment_modulated_exchange_action(np.eye(256),pair,((0,1),(2,3),(3,4),(6,7)),(1,-1),(1,-1),feature)


def test_dimer_environment_is_hermitian_and_preserves_singlet():
    pair=_swaps(8)
    matrix=spectator_environment_action(np.eye(256),pair,(0,1,2,3),(4,5,6,7),"dimer_correlation")
    np.testing.assert_allclose(matrix,matrix.T.conj(),atol=3e-14)
    singlet=np.array([0,-1,1,0])/np.sqrt(2)
    state=np.kron(np.kron(singlet,singlet),np.kron(singlet,singlet))
    image=matrix@state
    assert np.linalg.norm(image)>1e-3
    raising=np.zeros_like(image)
    for old in range(256):
        for bit in range(8):
            if not old&(1<<bit):
                raising[old|(1<<bit)]+=image[old]
    assert np.linalg.norm(raising)<3e-14


def test_dimer_environment_matches_independent_pauli_products():
    spin = []
    for site in range(4):
        components = []
        for pauli in (np.array([[0, 1], [1, 0]])/2,
                      np.array([[0, 1j], [-1j, 0]])/2,
                      np.diag([-.5, .5])):
            value = np.ones((1, 1))
            for j in range(4):
                value = np.kron(value, pauli if j == site else np.eye(2))
            components.append(value)
        spin.append(components)
    def dot(i, j):
        return sum(spin[i][k] @ spin[j][k] for k in range(3))
    d0, d1, d2 = dot(0, 1)+dot(2, 3), dot(0, 2)+dot(1, 3), dot(0, 3)+dot(1, 2)
    r0, r1 = (d0-d1)/np.sqrt(2), (d0+d1-2*d2)/np.sqrt(6)
    reference = (np.kron(r0, r0)+np.kron(r1, r1))/6
    observed = spectator_environment_action(np.eye(256), _swaps(8), (0,1,2,3), (4,5,6,7), "dimer_correlation")
    np.testing.assert_allclose(observed, reference, atol=3e-14)


def test_sealed_environment_experiment_is_complete_and_unsaturated():
    import json
    from pathlib import Path
    from su2_nqs.frontier.artifacts import canonical_sha256
    from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
    from projects.clebsch_tree_nqs.workflows import spin_exchange_environment as workflow

    root = Path(__file__).resolve().parents[1]
    directory = root/"projects/clebsch_tree_nqs/results/spin_exchange_environment_v1"
    result = json.loads((directory/"results.json").read_text())
    assert result["result_sha256"] == canonical_sha256({k:v for k,v in result.items() if k != "result_sha256"})
    assert result["bundle_sha256"] == file_sha256(directory/"states.npz")
    assert result["source_manifest"] == transitive_local_source_manifest(
        (Path(workflow.__file__).resolve(),), root=root,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"))
    assert not result["remote_training_licensed"]
    assert max(result["diagnostics"]["checks"].values()) < 2e-10
    assert set(result["cases"]) == {"0.0", "0.125", "0.175"}
    for case in result["cases"].values():
        assert set(case["modes"]) == {"collective", "mode_1", "mode_2", "mode_3"}
        for mode in case["modes"].values():
            assert mode["constant"]["rank"] == 4
            assert set(mode["environments"]) == {"spin_content", "spin_pair", "dimer_correlation"}
            for row in mode["environments"].values():
                assert row["rank"] == row["matched_krylov"]["rank"] == row["environment_only_addition"]["rank"] == 5
                assert row["rank"] < case["a1_dimension"]
                assert len(row["scrambles"]) == 5
                assert row["energy_per_site"] <= mode["constant"]["energy_per_site"]+1e-12
                assert row["energy_per_site"] >= case["exact_energy_per_site"]-1e-12
                assert row["affine_coefficient_fit"]["reconstruction_error"] < 1e-12
                assert row["scramble_anchor_max_error"] < 1e-12
                assert row["scramble_gram_max_error"] < 1e-12


def test_intermediate_spatial_projection_must_not_be_inserted_between_factors():
    pair=_swaps(4)
    a=pair(np.eye(16),0,1)
    b=pair(np.eye(16),2,3)
    # Z2 spatial operation swaps the two dimers.
    indices=np.array([((i&3)<<2)|(i>>2) for i in range(16)])
    projector=(np.eye(16)+np.eye(16)[indices])/2
    correct=projector@a@b@projector
    prematurely_projected=projector@a@projector@b@projector
    assert np.linalg.norm(correct-prematurely_projected)>.1
