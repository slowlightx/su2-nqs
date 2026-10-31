"""Independent physical N12 oracle for an entangled conditional boundary bath."""
from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest

from su2_nqs.operators.su2_block_transfer import BlockIsometry, project_block
from su2_nqs.operators.su2_sector_merge import (
    compact_reduced_block, project_sector_maps, reduced_block_from_projected,
)
from su2_nqs.operators.su2_conditional_response import (
    ConditionalResponseFactor, make_conditional_cut_response,
    nested_response_expansion, root_coefficient_matrices,
)
from su2_nqs.su2.clebsch_gordan import clebsch_gordan
from su2_nqs.su2.quantum_numbers import coupling_range
from su2_nqs.su2.triangle_blocks import triangle_block_tables


def _orthogonal(rng, rows, columns):
    x = rng.normal(size=(rows, columns))+1j*rng.normal(size=(rows, columns))
    return np.linalg.qr(x, mode="reduced")[0]


def _triangle(rng, truncated=False):
    mags = np.array([2*i.bit_count()-3 for i in range(8)])
    physical = {}
    for q, table in triangle_block_tables()["three"].items():
        u = np.stack([table*(mags == m)[:, None] for m in range(-q, q+1, 2)], axis=-1)
        width = 1 if truncated and q == 1 else u.shape[1]
        physical[q] = np.einsum("pak,ab->pbk", u, _orthogonal(rng, u.shape[1], width))
    projected = project_block(BlockIsometry(3, physical), np.zeros((8, 8), complex))
    return physical, compact_reduced_block(reduced_block_from_projected(projected))


def _physical_merge(left, right, maps):
    """Explicit magnetic CG sums, independent of production recoupling."""
    result = {}
    for q, matrix in maps.items():
        physical = np.zeros((next(iter(left.values())).shape[0]*next(iter(right.values())).shape[0],
                             matrix.shape[1], q+1), complex)
        row = 0
        for ql, ul in sorted(left.items()):
            for qr, ur in sorted(right.items()):
                if q not in coupling_range(ql, qr):
                    continue
                for a in range(ul.shape[1]):
                    for b in range(ur.shape[1]):
                        for k, m in enumerate(range(-q, q+1, 2)):
                            state = np.zeros(physical.shape[0], complex)
                            for ml in range(-ql, ql+1, 2):
                                mr = m-ml
                                if abs(mr) <= qr and (mr-qr) % 2 == 0:
                                    state += clebsch_gordan(ql, ml, qr, mr, q, m)*np.kron(
                                        ul[:, a, (ml+ql)//2], ur[:, b, (mr+qr)//2])
                            physical[:, :, k] += state[:, None]*matrix[row][None, :]
                        row += 1
        assert row == matrix.shape[0]
        result[q] = physical
    return result


def _count(left, right, q):
    return sum(ul.shape[1]*ur.shape[1] for ql, ul in left.items()
               for qr, ur in right.items() if q in coupling_range(ql, qr))


def _fixture(truncated_bath=False):
    rng = np.random.default_rng(2610235901)
    pl, left = _triangle(rng); pr, right = _triangle(rng)
    pb, b = _triangle(rng, truncated=truncated_bath); pd, d = _triangle(rng)
    maps = {q: _orthogonal(rng, _count(pl, pr, q), width)
            for q, width in ((0, 2), (2, 2), (4, 1), (6, 1))}
    a = project_sector_maps(left, right, (), maps=maps, compact_boundary=True)
    pa = _physical_merge(pl, pr, maps)
    parents = {p: _orthogonal(rng, _count(pa, pb, p), 2) for p in (1, 3)}
    pc = _physical_merge(pa, pb, parents)
    root = {p: rng.normal(size=(2, pd[p].shape[1]))+1j*rng.normal(size=(2, pd[p].shape[1]))
            for p in (1, 3)}
    norm = np.sqrt(sum(np.vdot(v, v).real for v in root.values()))
    root = {p: v/norm for p, v in root.items()}
    psi = np.zeros(4096, complex)
    for p, gamma in root.items():
        for m in range(-p, p+1, 2):
            cg = clebsch_gordan(p, m, p, -m, 0, 0)
            for c in range(gamma.shape[0]):
                for e in range(gamma.shape[1]):
                    psi += cg*gamma[c, e]*np.kron(pc[p][:, c, (m+p)//2], pd[p][:, e, (-m+p)//2])
    ab = [(0, 0, 1.), (3, 0, .4), (1, 2, .125), (5, 1, .7)]
    ad = [(2, 1, .8), (4, 2, -.15), (0, 0, .35)]
    args = dict(a_children=(left, right), a_source=a, a_maps=maps, b=b, d=d,
                parent_maps=parents, root_coefficients=root,
                cross_bonds_ab=ab, cross_bonds_ad=ad)
    return args, (pl, pr, pb, pd), psi, ab, ad


def _physical_cut_action(psi, ab, ad):
    ids = np.arange(4096, dtype=np.int64); result = np.zeros_like(psi)
    bonds = [(i, 6+j, g) for i, j, g in ab]+[(i, 9+j, g) for i, j, g in ad]
    for i, j, g in bonds:
        bi = (ids >> (11-i)) & 1; bj = (ids >> (11-j)) & 1
        result += g*(bi-.5)*(bj-.5)*psi
        different = bi != bj
        flip = ids[different] ^ ((1 << (11-i)) | (1 << (11-j)))
        result[flip] += .5*g*psi[different]
    return result


@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("target_q", [0, 2, 4])
def test_coherent_response_matches_independent_full_N12_physical_action(target_q, center):
    args, (pl, pr, pb, pd), psi, ab, ad = _fixture()
    observed = make_conditional_cut_response(**args, target_q=target_q, center=center)
    action = _physical_cut_action(psi, ab, ad)
    mean = np.vdot(psi, action)
    assert abs(np.vdot(psi, psi)-1) < 3e-13
    if center:
        action -= mean*psi
        np.testing.assert_allclose(observed.diagnostics["cut_expectation_real"], mean.real, atol=3e-13)
        assert abs(mean.imag) < 3e-13
    ua = _physical_merge(pl, pr, {target_q: np.eye(_count(pl, pr, target_q))})[target_q]
    ue = _physical_merge(pb, pd, {target_q: np.eye(_count(pb, pd, target_q))})[target_q]
    # Highest A magnetic index and opposite E magnetic index: singlet CG is
    # positive 1/sqrt(q+1). This independently fixes every magnetic factor.
    expected = np.sqrt(target_q+1)*ua[:, :, -1].conj().T @ action.reshape(64, 64) @ ue[:, :, 0].conj()
    dense = observed.to_dense()
    np.testing.assert_allclose(dense, expected, atol=5e-13)
    np.testing.assert_allclose(dense @ dense.conj().T, expected @ expected.conj().T, atol=5e-13)
    rng = np.random.default_rng(5902)
    v = rng.normal(size=dense.shape[1])+1j*rng.normal(size=dense.shape[1])
    w = rng.normal(size=dense.shape[0])+1j*rng.normal(size=dense.shape[0])
    np.testing.assert_allclose(observed.matvec(v), dense @ v, atol=5e-13)
    np.testing.assert_allclose(observed.rmatvec(w), dense.conj().T @ w, atol=5e-13)
    np.testing.assert_allclose(observed.as_linear_operator() @ np.column_stack((v, 2*v)),
                               dense @ np.column_stack((v, 2*v)), atol=5e-13)
    np.testing.assert_allclose(observed.as_linear_operator().H @ np.column_stack((w, 2*w)),
                               dense.conj().T @ np.column_stack((w, 2*w)), atol=5e-13)
    np.testing.assert_allclose(np.vdot(w, observed.matvec(v)), np.vdot(observed.rmatvec(w), v), atol=5e-13)
    assert not observed.diagnostics["physical_bath_leakage_included"]
    assert not observed.diagnostics["complete_A1"]


def test_truncated_bath_response_is_a_projection_not_full_physical_variance():
    args, (pl, pr, pb, pd), psi, ab, ad = _fixture(truncated_bath=True)
    response = make_conditional_cut_response(**args, target_q=2, center=True)
    action = _physical_cut_action(psi, ab, ad)
    action -= np.vdot(psi, action)*psi
    ua = _physical_merge(pl, pr, {2: np.eye(_count(pl, pr, 2))})[2][:, :, -1]
    ue = _physical_merge(pb, pd, {2: np.eye(_count(pb, pd, 2))})[2][:, :, 0]
    partial = np.sqrt(3)*ua.conj().T @ action.reshape(64, 64)
    projected = partial @ ue.conj()
    dense = response.to_dense()
    np.testing.assert_allclose(dense, projected, atol=5e-13)
    leakage = partial @ partial.conj().T-dense @ dense.conj().T
    assert np.linalg.eigvalsh((leakage+leakage.conj().T)/2).min() > -5e-13
    assert np.trace(leakage).real > 1e-4
    assert not response.diagnostics["physical_bath_leakage_included"]


def test_center_is_defined_by_actual_maps_not_an_inconsistent_cached_A_operator():
    args, _, _, _, _ = _fixture()
    original = make_conditional_cut_response(**args, target_q=2, center=True)
    a = args["a_source"]
    permuted = list(a.site_reduced_spin)
    permuted[0], permuted[1] = permuted[1], permuted[0]
    bad_cache = replace(a, site_reduced_spin=tuple(permuted))
    checked = make_conditional_cut_response(**dict(args, a_source=bad_cache), target_q=2, center=True)
    # Total-spin/Hermiticity checks cannot diagnose such a site permutation.
    # Ignoring this redundant cache gives one unique physical mean and Y.
    assert checked.diagnostics["cutmean_source"] == original.diagnostics["cutmean_source"]
    np.testing.assert_array_equal(checked.to_dense(), original.to_dense())
    assert checked.diagnostics["cut_expectation_real"] == original.diagnostics["cut_expectation_real"]


def test_source_charge_interference_is_retained_and_nested_expansion_preserves_old_state():
    args, _, _, _, _ = _fixture()
    response = make_conditional_cut_response(**args, target_q=2, center=False)
    assert response.diagnostics["coherent_source_q"] == [0, 2, 4]
    sites = len(response.diagnostics["boundary_inside_sites"])
    parts = [sum((a @ z for a, z in response.terms[i*sites:(i+1)*sites]),
                 np.zeros(response.shape, complex)) for i in range(3)]
    coherent = response.to_dense(); incoherent = sum(x @ x.conj().T for x in parts)
    assert np.linalg.norm(coherent @ coherent.conj().T-incoherent) > 1e-4
    nested = nested_response_expansion(2, args["a_maps"][2], response, additional=2)
    assert np.array_equal(nested.frame[:, :2], args["a_maps"][2])
    np.testing.assert_allclose(nested.frame.conj().T @ nested.frame, np.eye(4), atol=5e-13)
    old = args["a_maps"][2]
    np.testing.assert_allclose(nested.frame @ (nested.frame.conj().T @ old), old, atol=5e-13)
    assert nested.diagnostics["exact_old_state_feasible_after_ancestor_zero_extension"]


def test_real_degenerate_matrix_free_svd_keeps_new_columns_exactly_real():
    channels = tuple((2, a, 0, 0) for a in range(520))
    eye = np.eye(520)
    a = eye[:, 3:7].astype(complex)
    z = np.diag([3., 3., 1., .5]) @ a.conj().T
    response = ConditionalResponseFactor(2, channels, channels, ((a, z),), {}, {})
    expanded = nested_response_expansion(2, eye[:, :3], response, additional=2)
    assert expanded.diagnostics["real_singular_solver_used"]
    assert np.max(abs(expanded.frame.imag)) == 0.
    added = expanded.frame[:, 3:]
    np.testing.assert_allclose(added @ added.conj().T, eye[:, 3:5] @ eye[:, 3:5].T, atol=5e-13)


def test_root_unpack_and_invalid_inputs_are_rejected_before_large_allocation():
    args, _, _, _, _ = _fixture()
    channels = tuple((q, a, q, b) for q, x in sorted(args["root_coefficients"].items())
                     for a in range(x.shape[0]) for b in range(x.shape[1]))
    root = np.concatenate([x.reshape(-1) for _, x in sorted(args["root_coefficients"].items())])[:, None]
    unpacked = root_coefficient_matrices(root, channels)
    for q, v in unpacked.items():
        np.testing.assert_array_equal(v, args["root_coefficients"][q])
    with pytest.raises(ValueError, match="normalized"):
        make_conditional_cut_response(**dict(args, root_coefficients={q: 2*v for q, v in unpacked.items()}))
    with pytest.raises(ValueError, match="isometric"):
        make_conditional_cut_response(**dict(args, a_maps={q: 2*v for q, v in args["a_maps"].items()}))
    with pytest.raises(ValueError, match="cut bonds"):
        make_conditional_cut_response(**dict(args, cross_bonds_ad=[(0, 0, 1j)]))
    with pytest.raises(MemoryError):
        make_conditional_cut_response(**args, memory_limit_bytes=1)
    response = make_conditional_cut_response(**args)
    with pytest.raises(MemoryError):
        response.to_dense(maximum_entries=1)
    with pytest.raises(ValueError):
        nested_response_expansion(4, args["a_maps"][2], response)
