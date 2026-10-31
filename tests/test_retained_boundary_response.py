"""Independent linear-algebra checks for fixed-q response enrichment."""
import numpy as np
import pytest
from su2_nqs.frontier.retained_boundary_response import (
    change_child_coordinates, correlated_bath_response_factor, density_protection_count, fixed_q_response_frame,
    interpolate_fixed_q_frame, response_capture,
)


def test_complex_correlated_bath_factor_equals_explicit_tensor_product_trace():
    rng=np.random.default_rng(4030)
    responses=rng.normal(size=(3,7,4))+1j*rng.normal(size=(3,7,4))
    bath=rng.normal(size=(5,4))+1j*rng.normal(size=(5,4))
    couplings=rng.normal(size=(3,4));gram=bath.conj().T@bath
    factor,diagnostics=correlated_bath_response_factor(2,responses,couplings,gram)
    # Direct JOINT target/source/environment tensor; no Gram formula used.
    joint=np.zeros((7,4,5),complex)
    for i in range(3):
        for j in range(4):
            joint+=couplings[i,j]*responses[i,:,:,None]*bath[:,j][None,None,:]
    matrix=joint.reshape(7,-1)
    np.testing.assert_allclose(factor@factor.conj().T,matrix@matrix.conj().T,atol=3e-12,rtol=3e-14)
    # This genuinely complex oracle detects using G rather than G.T.
    wrong=couplings@gram@couplings.T
    wrong_cov=np.einsum('ik,ias,kbs->ab',wrong,responses,responses.conj())
    assert np.linalg.norm(wrong_cov-matrix@matrix.conj().T)>1
    assert diagnostics['q']==2 and not diagnostics['extra_q_plus_one_factor']
    assert not diagnostics['spherical_Cartesian_conversion_inserted']
    assert diagnostics['target_covariance_constructed'] is False


def test_correlated_bath_diagonal_limit_is_old_weighted_response_SVD():
    rng=np.random.default_rng(4031);responses=rng.normal(size=(4,11,3))
    weights=np.asarray([.5,1.3,.8,2.])
    factor,diagnostics=correlated_bath_response_factor(4,responses,np.eye(4),np.diag(weights))
    old=np.concatenate([np.sqrt(w)*r for w,r in zip(weights,responses,strict=True)],axis=1)
    np.testing.assert_allclose(factor@factor.T,old@old.T,atol=2e-14)
    np.testing.assert_allclose(np.linalg.svd(factor,compute_uv=False),np.linalg.svd(old,compute_uv=False),atol=4e-15)
    assert diagnostics['bath_Gram_numerical_rank']==4


def test_correlated_bath_factor_is_unitary_gauge_covariant():
    rng=np.random.default_rng(4032)
    def unitary(n):
        return np.linalg.qr(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))[0]
    responses=rng.normal(size=(3,9,4))+1j*rng.normal(size=(3,9,4))
    bath=rng.normal(size=(7,5))+1j*rng.normal(size=(7,5));couplings=rng.normal(size=(3,5))
    factor,_=correlated_bath_response_factor(2,responses,couplings,bath.conj().T@bath)
    rotated_bath=unitary(7)@bath
    same,_=correlated_bath_response_factor(2,responses,couplings,rotated_bath.conj().T@rotated_bath)
    np.testing.assert_allclose(same@same.conj().T,factor@factor.conj().T,atol=8e-12,rtol=4e-14)
    target,source=unitary(9),unitary(4)
    moved=np.einsum('ab,ibs,st->iat',target,responses,source)
    actual,_=correlated_bath_response_factor(2,moved,couplings,bath.conj().T@bath)
    np.testing.assert_allclose(actual@actual.conj().T,target@factor@factor.conj().T@target.conj().T,
                               atol=8e-12,rtol=4e-14)


def test_correlated_bath_rejects_invalid_psd_and_only_repairs_roundoff():
    responses=np.ones((2,3,1));couplings=np.eye(2)
    with pytest.raises(ValueError,match='negative'):
        correlated_bath_response_factor(2,responses,couplings,np.diag([1.,-.01]))
    with pytest.raises(ValueError,match='negative'):
        correlated_bath_response_factor(2,responses,couplings,np.diag([1e-20,-1e-22]))
    for bad_j,bad_g in ((couplings+0j,np.eye(2)),(couplings,np.ones((3,3))),
                        (couplings,np.asarray([[1,1j],[0,1]])),(couplings,np.diag([np.nan,1]))):
        with pytest.raises(ValueError):
            correlated_bath_response_factor(2,responses,bad_j,bad_g)
    factor,notes=correlated_bath_response_factor(2,responses,couplings,np.diag([1.,-1e-14]))
    np.testing.assert_allclose(factor@factor.T,np.ones((3,3)),atol=1e-15)
    assert notes['bath_Gram_clipped_negative_roundoff']==1e-14
    zero,notes=correlated_bath_response_factor(2,responses,couplings,np.zeros((2,2)))
    assert zero.shape==(3,0) and notes['response_covariance_trace']==0


def test_procrustes_polar_path_has_analytic_rank_one_angle_and_endpoints():
    angle = .83; step = 1/16
    old = np.array([[1.], [0.]])
    new = np.array([[np.cos(angle)], [np.sin(angle)]])
    frame, notes = interpolate_fixed_q_frame(2, old, new, step=step)
    expected = np.arctan2(step*np.sin(angle), 1-step+step*np.cos(angle))
    np.testing.assert_allclose(frame[:, 0], [np.cos(expected), np.sin(expected)], atol=2e-15)
    np.testing.assert_array_equal(interpolate_fixed_q_frame(2, old, new, step=0)[0], old)
    endpoint = interpolate_fixed_q_frame(2, old, -new, step=1)[0]
    np.testing.assert_allclose(endpoint @ endpoint.T, new @ new.T, atol=2e-15)
    assert notes['q'] == 2 and notes['multiplicity_width'] == 1
    assert notes['old_retracted_projector_Frobenius'] < notes['old_proposed_projector_Frobenius']


def test_procrustes_polar_path_is_independently_complex_gauge_covariant():
    rng = np.random.default_rng(4020)
    def unitary(n):
        return np.linalg.qr(rng.normal(size=(n, n))+1j*rng.normal(size=(n, n)))[0]
    old = unitary(21)[:, :6]; proposed = unitary(21)[:, :6]
    physical, old_gauge, new_gauge = unitary(21), unitary(6), unitary(6)
    a, notes = interpolate_fixed_q_frame(4, old, proposed, step=1/16)
    b, _ = interpolate_fixed_q_frame(4, physical @ old @ old_gauge,
                                   physical @ proposed @ new_gauge, step=1/16)
    np.testing.assert_allclose(a.conj().T @ a, np.eye(6), atol=3e-15)
    np.testing.assert_allclose(b @ b.conj().T, physical @ a @ a.conj().T @ physical.conj().T, atol=4e-15)
    assert min(notes['chord_Gram_eigenvalues']) >= notes['chord_Gram_lower_bound']-2e-15
    with pytest.raises(ValueError):
        interpolate_fixed_q_frame(2, np.eye(4)[:, :2], np.eye(4)[:, 2:], step=1/16)
    with pytest.raises(TypeError):
        interpolate_fixed_q_frame(1.5, old, proposed, step=1/16)


def test_response_frame_protects_columns_and_maximizes_remaining_response():
    rng = np.random.default_rng(4001)
    energy = np.linalg.qr(rng.normal(size=(31, 8)))[0]
    responses = rng.normal(size=(31, 17))
    selected = fixed_q_response_frame(2, energy, responses, protect=4)
    np.testing.assert_array_equal(selected.frame[:, :4], energy[:, :4])
    assert selected.q == 2 and selected.frame.shape == energy.shape
    assert selected.isometry_error < 1e-14
    residual = responses-energy[:, :4] @ (energy[:, :4].T @ responses)
    expected = np.linalg.norm(energy[:, :4].T @ responses)**2+np.sum(np.linalg.svd(residual, compute_uv=False)[:4]**2)
    np.testing.assert_allclose(response_capture(selected.frame, responses)["numerator"], expected, atol=3e-12)
    random = fixed_q_response_frame(2, energy, responses, protect=4, seed=4002)
    assert response_capture(selected.frame, responses)["fraction"] >= response_capture(random.frame, responses)["fraction"]


def test_response_projector_is_complex_unitary_gauge_covariant():
    rng = np.random.default_rng(4003)
    energy = np.linalg.qr(rng.normal(size=(23, 8))+1j*rng.normal(size=(23, 8)))[0]
    responses = rng.normal(size=(23, 13))+1j*rng.normal(size=(23, 13))
    gauge = np.linalg.qr(rng.normal(size=(23, 23))+1j*rng.normal(size=(23, 23)))[0]
    a = fixed_q_response_frame(2, energy, responses, protect=4).frame
    b = fixed_q_response_frame(2, gauge @ energy, gauge @ responses, protect=4).frame
    np.testing.assert_allclose(b @ b.conj().T, gauge @ a @ a.conj().T @ gauge.conj().T, atol=2e-14)


def test_null_response_falls_back_and_invalid_frames_fail_closed():
    energy = np.eye(12, 8)
    selected = fixed_q_response_frame(2, energy, energy[:, :2], protect=4)
    np.testing.assert_array_equal(selected.frame, energy)
    assert selected.response_columns == 0 and selected.fallback_columns == 4
    with pytest.raises(ValueError):
        fixed_q_response_frame(2, energy*2, energy, protect=4)
    with pytest.raises(ValueError):
        fixed_q_response_frame(2, energy, energy, protect=9)
    closed = fixed_q_response_frame(2, energy, np.ones((12, 3)), protect=8)
    np.testing.assert_array_equal(closed.frame, energy)
    assert closed.response_columns == 0


def test_child_overlap_then_qr_is_exact_physical_projection_not_state_preservation():
    rng = np.random.default_rng(4004)
    old = np.linalg.qr(rng.normal(size=(17, 4))+1j*rng.normal(size=(17, 4)))[0]
    new = np.linalg.qr(rng.normal(size=(17, 4))+1j*rng.normal(size=(17, 4)))[0]
    overlap = new.conj().T @ old
    channels = tuple((2, a, 0, b) for a in range(4) for b in range(3))
    merge = np.linalg.qr(rng.normal(size=(12, 5))+1j*rng.normal(size=(12, 5)))[0]
    root = rng.normal(size=(5, 2))+1j*rng.normal(size=(5, 2))
    projected = change_child_coordinates(merge, channels, left={2: overlap})
    q, r = np.linalg.qr(projected)
    direct = np.kron(new @ overlap, np.eye(3)) @ merge @ root
    canonical = np.kron(new, np.eye(3)) @ q @ (r @ root)
    np.testing.assert_allclose(canonical, direct, atol=4e-15)
    original = np.kron(old, np.eye(3)) @ merge @ root
    exact_projector = np.kron(new @ new.conj().T, np.eye(3))
    np.testing.assert_allclose(canonical, exact_projector @ original, atol=4e-15)
    assert np.linalg.norm(canonical) < np.linalg.norm(original)


def test_physical_density_protection_bounds_raw_norm_loss():
    rng = np.random.default_rng(4005)
    old = np.linalg.qr(rng.normal(size=(29, 8))+1j*rng.normal(size=(29, 8)))[0]
    factor = rng.normal(size=(8, 8))+1j*rng.normal(size=(8, 8))
    rho = factor @ factor.conj().T; rho /= np.trace(rho)
    values, axes = np.linalg.eigh(rho); order = np.argsort(values)[::-1]
    values, axes = values[order], axes[:, order]
    reference = old @ axes
    responses = rng.normal(size=(29, 20))+1j*rng.normal(size=(29, 20))
    selected = fixed_q_response_frame(2, reference, responses, protect=4).frame
    overlap = selected.conj().T @ old
    retained = float(np.trace(rho @ (overlap.conj().T @ overlap)).real)
    assert retained >= values[:4].sum()-2e-14
    p_q2 = .6
    raw_fidelity = 1-p_q2+p_q2*retained
    assert raw_fidelity >= 1-p_q2*values[4:].sum()-2e-14


def test_environment_allocation_is_minimal_and_prospective_loss_bounded():
    rng = np.random.default_rng(4006)
    values = np.array([.57, .29, .08, .025, .015, .009, .007, .004])
    p = .64; budget = .005
    k = density_protection_count(p, values, maximum_lost_norm=budget)
    assert p*values[k:].sum() <= budget
    assert k == 0 or p*values[k-1:].sum() > budget
    old = np.linalg.qr(rng.normal(size=(31, 8)))[0]
    responses = rng.normal(size=(31, 20))
    new = fixed_q_response_frame(2, old, responses, protect=k).frame
    overlap = new.T @ old
    raw_fidelity = 1-p+p*np.trace(np.diag(values) @ (overlap.T @ overlap))
    assert raw_fidelity >= 1-budget-2e-14
    assert density_protection_count(.001, values, maximum_lost_norm=budget) == 0
    assert density_protection_count(p, values, maximum_lost_norm=0.) == 8
    with pytest.raises(ValueError):
        density_protection_count(p, values[::-1], maximum_lost_norm=budget)


def test_cached_isomorphic_node_maps_are_detached_before_local_updates():
    from copy import deepcopy
    from projects.clebsch_tree_nqs.workflows.triangle_boundary_response_enrichment import (
        _checked_changed_nodes, _independent_node_map_copy,
    )
    shared36 = {0: np.eye(3), 2: np.eye(4)}
    shared72 = {0: np.eye(3), 2: np.eye(4)}
    original = {**{i: shared36 for i in (8, 9, 10, 11)},
                **{i: shared72 for i in (12, 13)}, 14: {0: np.ones((3, 1))}}
    detached = _independent_node_map_copy(original)
    detached[8][2] = -detached[8][2]
    detached[12][0][:] *= -1
    detached[14][0][0, 0] = 2
    for node in (9, 10, 11, 13):
        for q in original[node]:
            np.testing.assert_array_equal(detached[node][q], original[node][q])
            assert not np.shares_memory(detached[node][q], original[node][q])
    assert [node for node in original if any(not np.array_equal(original[node][q], detached[node][q])
                                            for q in original[node])] == [8, 12, 14]
    np.testing.assert_array_equal(original[8][2], np.eye(4))
    np.testing.assert_array_equal(original[12][0], np.eye(3))
    assert _checked_changed_nodes(original, detached, allowed={8, 12, 14}) == [8, 12, 14]
    unsafe = deepcopy(original)
    unsafe[8][2] = -unsafe[8][2]
    with pytest.raises(ArithmeticError, match="cached-peer"):
        _checked_changed_nodes(original, unsafe, allowed={8, 12, 14})


def test_workflow_creates_case_directory_before_array_io(tmp_path, monkeypatch):
    """I/O-only integration regression; no physics result is mocked as evidence."""
    from copy import deepcopy
    from types import SimpleNamespace
    from projects.clebsch_tree_nqs.workflows import triangle_boundary_response_enrichment as workflow
    source_run = tmp_path/"sources"
    for tag in ("j0", "j0125"):
        directory = source_run/tag/"physical"; directory.mkdir(parents=True)
        (directory/"state.pkl").write_bytes(b"test-input-placeholder")
    energy = -64/144
    block = SimpleNamespace(h_by_q={0: np.array([[-64.]]), 2: np.eye(8)})
    state = dict(maps={8: {2: np.eye(8)}}, blocks=[block]*31)
    data = dict(train=np.eye(8), heldout=np.eye(8), all=np.eye(8))
    frozen = dict(global_region_q_weights={2: .6})
    monkeypatch.setattr(workflow, "load_source", lambda j2: ({}, {"unit_test": True}))
    monkeypatch.setattr(workflow, "load_checkpoint", lambda path: {"params": deepcopy(state)})
    monkeypatch.setattr(workflow, "frozen_responses", lambda a, b: (data, dict(notes=deepcopy(frozen), down_q2=np.eye(8)*.025)))
    monkeypatch.setattr(workflow, "project_candidate", lambda a, b, c, d: (deepcopy(state),
        dict(raw_energy_per_site=energy, normalized_RAW_fidelity_to_parent=1.,
             reduced_H_columns_evaluated=0, full_A1_energy_known=False)))
    monkeypatch.setattr(workflow, "native_bank_from_state", lambda a, b: ({"local": (), "top": ()}, {}))
    monkeypatch.setattr(workflow, "triangular_raw_overlap", lambda a, b, c: 1.+0j)
    result = workflow.run(tmp_path/"output", source_run, 30, protection="density")
    assert len(result["reports"]) == 2
    for tag in ("j0", "j0125"):
        assert (tmp_path/"output"/tag/"frozen_response_columns.npz").exists()
        assert (tmp_path/"output"/tag/"density4_response4"/"initialized_bank.pkl").exists()
