"""Independent finite-support limits of projected residual learning targets.

These references do not train a model.  In particular, parent ``H psi``
identifies an energy derivative but not finite-candidate energy or variance.
Coherent A1 projection must precede the logarithmic derivative.
"""
import numpy as np
import pytest

from su2_nqs.benchmarks.coupled_basis import coupled_basis_matrix
from su2_nqs.benchmarks.exact_diagonalization import (
    heisenberg_sz_matrix, spin_squared_matrix,
)
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.lattices.spatial_symmetry import rectangular_gamma_a1_group
from su2_nqs.su2.fusion_path import enumerate_paths
from su2_nqs.su2.triangle_blocks import triangle_block_tables
from su2_nqs.vmc.tangent_residual_diagnostic import project_residual_on_tangents


def _moments(psi, hamiltonian):
    normalized = psi / np.linalg.norm(psi)
    action = hamiltonian @ normalized
    energy = np.vdot(normalized, action).real
    return energy, np.vdot(action, action).real - energy**2


def test_identical_parent_h_action_does_not_identify_finite_candidate_moments():
    # Both states have full sampling support; no node pathology is involved.
    basis = np.array([[1, 1j], [1j, 1]], dtype=complex) / np.sqrt(2)
    psi, tangent = basis[:, 0], basis[:, 1]
    h0 = basis @ np.array([[0., 1.], [1., 0.]]) @ basis.conj().T
    h7 = basis @ np.array([[0., 1.], [1., 7.]]) @ basis.conj().T
    np.testing.assert_allclose(h0 @ psi, h7 @ psi, atol=2e-15)
    np.testing.assert_allclose((h0 @ psi) / psi, (h7 @ psi) / psi, atol=2e-15)
    for hamiltonian in (h0, h7):
        exact_slope = 2 * np.vdot(tangent, hamiltonian @ psi).real
        assert exact_slope == pytest.approx(2., abs=3e-15)
    candidate = psi - .12 * tangent
    e0, v0 = _moments(candidate, h0)
    e7, v7 = _moments(candidate, h7)
    assert abs(e0 - e7) > .09
    assert abs(v0 - v7) > .5


def test_parent_nodes_can_hide_a_tangent_and_its_energy_derivative():
    # A parent-Born cache is blind to the second coordinate, although the
    # physical tangent and Hpsi have support there.  Do not certify complete
    # Hilbert-space tangent coverage from ratios on parent samples alone.
    psi = np.array([1., 0.])
    tangent = np.array([0., 1.])
    hamiltonian = np.array([[0., 1.], [1., 0.]])
    observed = abs(psi) > 0
    score = tangent[observed] / psi[observed]
    local_energy = (hamiltonian @ psi)[observed] / psi[observed]
    sampled_slope = 2 * np.real(np.sum(abs(psi[observed])**2 * score.conj() * local_energy))
    exact_slope = 2 * np.vdot(tangent, hamiltonian @ psi).real
    assert sampled_slope == 0
    assert exact_slope == 2


def test_identical_local_su2_environments_do_not_identify_interblock_residual_force():
    # Two triangles, each carrying q=1 with two internal multiplicities.
    # A relative sign between joint multiplicity sectors changes inter-block
    # coherence but neither triangle's *complete* local density matrix.
    # Thus even perfect local raw environments cannot predict every residual
    # force: the controller needs genuinely inter-block state information.
    tables = triangle_block_tables()
    channels = tables['six_channels'][0]
    coefficients = np.zeros(len(channels))
    coefficients[channels.index((1, 1, 0, 0))] = np.sqrt(.7)
    coefficients[channels.index((1, 1, 1, 1))] = np.sqrt(.3)
    plus = tables['six'][0] @ coefficients
    coefficients[channels.index((1, 1, 1, 1))] *= -1
    minus = tables['six'][0] @ coefficients
    s2, _ = spin_squared_matrix(6)
    for psi in (plus, minus):
        np.testing.assert_allclose(s2 @ psi, 0, atol=2e-14)
        np.testing.assert_allclose(psi @ psi, 1, atol=2e-15)
    for first in (True, False):
        factors = [p.reshape(8, 8) for p in (plus, minus)]
        rdms = [p @ p.T if first else p.T @ p for p in factors]
        np.testing.assert_allclose(rdms[0], rdms[1], atol=5e-16)
    bits = (np.arange(8)[:, None] >> np.arange(2, -1, -1)) & 1
    magnetic = np.sum(2*bits-1, axis=1)
    scalar = np.zeros((8, 8))
    for m in (-1, 1):
        transform = tables['three'][1] * (magnetic == m)[:, None]
        scalar += transform @ np.array([[0., 1.], [1., 0.]]) @ transform.T
    direction_operator = np.kron(scalar, np.eye(8))
    hamiltonian, _ = heisenberg_sz_matrix(6, [(0, 3, 1.), (1, 4, .7), (2, 5, .4)])
    forces = []
    for psi in (plus, minus):
        energy = psi @ hamiltonian @ psi
        delta = direction_operator @ psi
        forces.append(2*delta @ (hamiltonian @ psi-energy*psi))
    np.testing.assert_allclose(forces, [.20061609586175888, .041871017197883796], atol=2e-14)
    assert abs(forces[0]-forces[1]) > .15


def test_screening_texture_connected_spin_context_resolves_local_rdm_ambiguity():
    """Actual cross-block correlations, not bond counts, resolve the sign.

    Each direction is a local SU(2) scalar rotating multiplicities at fixed
    q=1.  Its screening texture f_i=delta<S_i.S_A> sums to zero; the block
    total spin and inter-block total screening are unchanged.  Nevertheless
    the normalized f_A.C_AB.f_B changes sign between the coherent states.
    This is contextual information, not a QSL or beyond-two-body witness.
    """
    tables = triangle_block_tables()
    channels = tables['six_channels'][0]
    coefficients = np.zeros(len(channels))
    coefficients[channels.index((1, 1, 0, 0))] = np.sqrt(.7)
    coefficients[channels.index((1, 1, 1, 1))] = np.sqrt(.3)
    states = [tables['six'][0] @ coefficients]
    coefficients[channels.index((1, 1, 1, 1))] *= -1
    states.append(tables['six'][0] @ coefficients)
    local_bits = (np.arange(8)[:, None] >> np.arange(2, -1, -1)) & 1
    local_m = np.sum(2*local_bits-1, axis=1)
    scalar = np.zeros((8, 8))
    for m in (-1, 1):
        transform = tables['three'][1] * (local_m == m)[:, None]
        scalar += transform @ np.array([[0., 1.], [1., 0.]]) @ transform.T
    screening_operators = []
    for site in range(3):
        operator, _ = heisenberg_sz_matrix(3, [
            (min(site, other), max(site, other), 1.)
            for other in range(3) if other != site])
        screening_operators.append(operator+.75*np.eye(8))
    bits = (np.arange(64)[:, None] >> np.arange(5, -1, -1)) & 1
    z = bits-.5
    contexts, textures = [], []
    for psi in states:
        probability = abs(psi)**2
        mean = probability @ z
        covariance = 3*((z.T*probability) @ z-mean[:, None]*mean[None, :])
        np.testing.assert_allclose(mean, 0, atol=2e-16)
        np.testing.assert_allclose(covariance[:3, 3:].sum(), -.75, atol=5e-16)
        pair = []
        for block in (0, 1):
            operator = np.kron(scalar, np.eye(8)) if block == 0 else np.kron(np.eye(8), scalar)
            delta = operator @ psi-psi*np.vdot(psi, operator @ psi)
            texture = []
            for screening in screening_operators:
                observable = np.kron(screening, np.eye(8)) if block == 0 else np.kron(np.eye(8), screening)
                texture.append(2*np.vdot(delta, observable @ psi).real)
            pair.append(np.asarray(texture))
        first, second = pair
        np.testing.assert_allclose([first.sum(), second.sum()], 0, atol=5e-16)
        textures.append(pair)
        contexts.append(first @ covariance[:3, 3:] @ second / np.linalg.norm(first)/np.linalg.norm(second))
    np.testing.assert_allclose(textures[0], textures[1], atol=5e-16)
    np.testing.assert_allclose(contexts, [-np.sqrt(.21), np.sqrt(.21)], atol=8e-16)


@pytest.fixture(scope='module')
def projected_reference():
    """Eight spins: complete S=0 and complete rectangular Gamma-A1."""
    n = 8
    hamiltonian, states = heisenberg_sz_matrix(
        n, square_j1j2_bonds(2, 4, j2=.17, periodic=True), n_up=4)
    coupled = coupled_basis_matrix(enumerate_paths(n, 0), 0, 0)[list(states)]
    positions = {state: i for i, state in enumerate(states)}
    images = []
    for permutation in rectangular_gamma_a1_group(2, 4):
        matrix = np.zeros_like(hamiltonian)
        for column, state in enumerate(states):
            bits = [(state >> (n-1-i)) & 1 for i in range(n)]
            image = sum(bits[permutation[i]] << (n-1-i) for i in range(n))
            matrix[positions[image], column] = 1
        images.append(matrix)
    projector = np.mean(images, axis=0)
    s2, _ = spin_squared_matrix(n, n_up=4)
    np.testing.assert_allclose(coupled.conj().T @ coupled, np.eye(14), atol=2e-14)
    np.testing.assert_allclose(s2 @ coupled, 0, atol=2e-14)
    np.testing.assert_allclose(projector @ projector, projector, atol=2e-14)
    np.testing.assert_allclose(projector @ hamiltonian, hamiltonian @ projector, atol=2e-14)
    rng = np.random.default_rng(492312)
    raw = coupled @ (rng.normal(size=14) + 1j*rng.normal(size=14))
    local_scalar, _ = heisenberg_sz_matrix(n, [(0, 1, 1.)], n_up=4)
    raw_tangent = local_scalar @ raw
    normalization = np.linalg.norm(projector @ raw)
    parent = projector @ raw / normalization
    tangent = projector @ raw_tangent / normalization
    np.testing.assert_allclose(s2 @ tangent, 0, atol=2e-14)
    np.testing.assert_allclose(projector @ tangent, tangent, atol=2e-14)
    assert np.min(abs(parent)) > 1e-5
    return parent, tangent, hamiltonian, raw/normalization, raw_tangent/normalization, images


def test_complete_projected_covariance_matches_finite_energy_derivative(projected_reference):
    psi, tangent, hamiltonian, *_ = projected_reference
    hpsi = hamiltonian @ psi
    probability = abs(psi)**2
    scores, local_energy = tangent / psi, hpsi / psi
    centered_score = scores - np.sum(probability*scores)
    slope = 2*np.real(np.sum(probability*centered_score.conj()*local_energy))
    step = 1e-6
    finite_difference = (_moments(psi+step*tangent, hamiltonian)[0] -
                         _moments(psi-step*tangent, hamiltonian)[0])/(2*step)
    np.testing.assert_allclose(slope, finite_difference, atol=8e-10, rtol=0)
    diagnostic = project_residual_on_tangents(psi, hpsi, tangent[:, None], n_sites=8)
    np.testing.assert_allclose(
        diagnostic.metrics['unit_FS_descent_slope_per_site'],
        -abs(slope)/(8*np.sqrt(np.sum(probability*abs(centered_score)**2))),
        atol=2e-14)


def test_coherent_projected_score_is_not_an_unweighted_mean_of_image_scores(projected_reference):
    psi, tangent, hamiltonian, raw, raw_tangent, images = projected_reference
    amplitudes = np.stack([image @ raw for image in images])
    derivatives = np.stack([image @ raw_tangent for image in images])
    score = derivatives.sum(axis=0) / amplitudes.sum(axis=0)
    np.testing.assert_allclose(score, tangent/psi, atol=3e-14)
    naive_score = np.mean(derivatives/amplitudes, axis=0)
    assert np.linalg.norm(naive_score-score) > .1
    probability = abs(psi)**2
    local_energy = (hamiltonian @ psi)/psi
    correct = 2*np.real(np.sum(probability*(score-np.sum(probability*score)).conj()*local_energy))
    naive = 2*np.real(np.sum(probability*(naive_score-np.sum(probability*naive_score)).conj()*local_energy))
    assert abs(correct-naive) > .01
