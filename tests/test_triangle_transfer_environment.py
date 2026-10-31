"""Independent physical references for the bounded correlated-bath probe."""
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.triangle_transfer_initialization import (
    RETAIN, _boundary_product_density, _boundary_probe_ritz, _hamiltonian,
    boundary_residual_coefficients, fixed_gap_boundary_correction, multiplicity_density, reference_site_spins,
    schur_multiplets, twelve_singlet_basis,
)


def singlet():
    return np.asarray([0., -1., 1., 0.])/np.sqrt(2)


def test_site_spin_convention_and_complete_algebra():
    for x, y, z in reference_site_spins(2):
        assert np.max(abs(x@y-y@x-1j*z)) < 1e-15
        assert np.max(abs(x@x+y@y+z@z-.75*np.eye(4))) < 1e-15
    plus = reference_site_spins(2)[0][0]+1j*reference_site_spins(2)[0][1]
    assert plus[2,0] == 1
    assert plus[0,2] == 0


@pytest.mark.parametrize('couplings', [np.ones((2,2)), np.asarray([[1.,1.],[0.,0.]])])
def test_two_dimer_exact_quantum_bath_cancellation(couplings):
    s = singlet(); c = np.outer(s,s); spin = reference_site_spins(2)
    bonds = [(i,j,couplings[i,j]) for i in range(2) for j in range(2) if couplings[i,j]]
    residual, info = boundary_residual_coefficients(c,spin,spin,bonds)
    assert np.linalg.norm(residual) < 1e-15
    assert info['boundary_action_norm_squared'] < 1e-28
    rho = np.outer(s,s)
    product, kernel = _boundary_product_density(rho,rho,spin,spin,couplings,diagonal_only=False)
    assert np.linalg.norm(product) < 1e-15
    assert np.linalg.norm(kernel) < 1e-15
    # An infinite-temperature bath misses the exact external-singlet cancellation.
    wrong, _ = _boundary_product_density(rho,np.eye(4)/4,spin,spin,couplings,diagonal_only=False)
    if np.count_nonzero(couplings) == 2:
        assert np.trace(wrong).real > .1


def test_coherent_kernel_retains_internal_total_spin_cancellation():
    s = singlet(); rho = np.outer(s,s); spin = reference_site_spins(2)
    couplings = np.asarray([[1.,0.],[1.,0.]])
    coherent, kernel = _boundary_product_density(rho,rho,spin,spin,couplings,diagonal_only=False)
    diagonal, same_kernel = _boundary_product_density(rho,rho,spin,spin,couplings,diagonal_only=True)
    assert np.max(abs(kernel-.25*np.ones((2,2)))) < 1e-15
    assert np.array_equal(kernel,same_kernel)
    assert np.linalg.norm(coherent) < 1e-15
    assert np.trace(diagonal).real == pytest.approx(.375,abs=1e-14)


def test_single_cross_bond_full_variance_and_product_bath_identity():
    s = singlet(); c = np.outer(s,s); spin = reference_site_spins(2)
    r, info = boundary_residual_coefficients(c,spin,spin,[(1,0,1.)])
    assert info['boundary_expectation'] == pytest.approx(0.,abs=1e-15)
    assert info['residual_norm_squared'] == pytest.approx(3/16,abs=1e-15)
    rho = np.outer(s,s); coupling = np.asarray([[0.,0.],[1.,0.]])
    product, _ = _boundary_product_density(rho,rho,spin,spin,coupling,diagonal_only=False)
    assert np.max(abs(product-r@r.conj().T)) < 1e-15


def test_complex_basis_gauge_uses_right_transpose_not_adjoint():
    rng = np.random.default_rng(2610234901); spin = reference_site_spins(2)
    c = rng.normal(size=(4,4))+1j*rng.normal(size=(4,4)); c /= np.linalg.norm(c)
    bonds = [(0,1,.7),(1,0,-.3)]
    r, info = boundary_residual_coefficients(c,spin,spin,bonds)
    explicit = sum((coupling*np.kron(a,b) for i,j,coupling in bonds
                    for a,b in zip(spin[i],spin[j],strict=True)),np.zeros((16,16),complex))
    reference = (explicit@c.ravel()).reshape(4,4)-info['boundary_expectation']*c
    assert np.max(abs(r-reference)) < 2e-15
    left = np.linalg.qr(rng.normal(size=(4,4))+1j*rng.normal(size=(4,4)))[0]
    right = np.linalg.qr(rng.normal(size=(4,4))+1j*rng.normal(size=(4,4)))[0]
    spin_left = tuple(tuple(left.conj().T@a@left for a in triple) for triple in spin)
    spin_right = tuple(tuple(right.conj().T@a@right for a in triple) for triple in spin)
    transformed, _ = boundary_residual_coefficients(left.conj().T@c@right.conj(),spin_left,spin_right,bonds)
    assert np.max(abs(transformed-left.conj().T@r@right.conj())) < 3e-15
    assert np.max(abs(transformed@transformed.conj().T-left.conj().T@(r@r.conj().T)@left)) < 3e-15


def test_complete_multiplicity_density_recovers_complex_scalar_blocks():
    base = schur_multiplets(6); rng = np.random.default_rng(2610234902)
    rho = np.zeros((64,64),complex); original = {}
    for q,u in base.items():
        z = rng.normal(size=(u.shape[1],3))+1j*rng.normal(size=(u.shape[1],3))
        block = z@z.conj().T; original[q] = block
        rho += sum((u[:,:,m]@block@u[:,:,m].conj().T/(q+1) for m in range(q+1)),np.zeros_like(rho))
    norm = np.trace(rho).real; rho /= norm
    recovered, info = multiplicity_density(rho,base)
    assert info['scalar_reconstruction_relative_error'] < 3e-15
    for q in base:
        assert np.max(abs(recovered[q]-original[q]/norm)) < 3e-15
    # A pure magnetic component of a triplet is not SU(2) invariant.
    triplet = base[2][:,0,-1]
    with pytest.raises(ArithmeticError,match=r'SU\(2\)'):
        multiplicity_density(np.outer(triplet,triplet),base)


def test_correlated_parent_uses_all_source_charges_and_not_product_marginals():
    rng = np.random.default_rng(2610234904); base = schur_multiplets(6)
    coeff = rng.normal(size=132)+1j*rng.normal(size=132); coeff /= np.linalg.norm(coeff)
    c = (twelve_singlet_basis()@coeff).reshape(64,64)
    spin = reference_site_spins(6); bonds = [(1,4,1.),(2,0,.7),(5,2,.125)]
    residual,info = boundary_residual_coefficients(c,spin,spin,bonds)
    exact = residual@residual.conj().T
    _,scalar = multiplicity_density(exact,base)
    assert scalar['scalar_reconstruction_relative_error'] < 3e-15
    coupling = np.zeros((6,6))
    for i,j,value in bonds:
        coupling[i,j] = value
    product,_ = _boundary_product_density(c@c.conj().T,c.T@c.conj(),spin,spin,coupling,diagonal_only=False)
    assert np.linalg.norm(product-exact) > .01
    projectors = {q:sum((u[:,:,m]@u[:,:,m].conj().T for m in range(q+1)),np.zeros((64,64),complex))
                  for q,u in base.items()}
    pieces = {}
    for q,p in projectors.items():
        source = p@c
        action,_ = boundary_residual_coefficients(source,spin,spin,bonds,center=False)
        pieces[q] = projectors[2]@(action-info['boundary_expectation']*source)
    assert np.linalg.norm(sum(pieces.values())-projectors[2]@residual) < 3e-15
    assert all(np.linalg.norm(pieces[q]) > .01 for q in (0,2,4))
    assert np.linalg.norm(pieces[6]) < 3e-15


def test_complex_left_right_frames_preserve_parent_and_full_H_variance():
    base = schur_multiplets(6); rng = np.random.default_rng(2610234903)
    left = {}; right = {}; expanded_left = {}; expanded_right = {}
    for q,u in base.items():
        frames = [np.linalg.qr(rng.normal(size=(u.shape[1],u.shape[1]))+
                  1j*rng.normal(size=(u.shape[1],u.shape[1])))[0] for _ in range(2)]
        left[q],right[q] = [x[:,:RETAIN[q]] for x in frames]
        expanded_left[q],expanded_right[q] = [x[:,:4 if q==2 else RETAIN[q]] for x in frames]
    basis = twelve_singlet_basis()
    h = _hamiltonian(12,[(0,1,1.),(6,7,1.),(5,6,.7),(2,10,.125)])
    hp = h@basis
    parent,_,old = _boundary_probe_ritz(left,right,basis,hp)
    c,m,new = _boundary_probe_ritz(expanded_left,expanded_right,basis,hp)
    assert m.shape == (132,22)
    assert np.linalg.norm(m@(m.conj().T@parent)-parent) < 4e-15
    assert new['energy_per_site'] <= old['energy_per_site']+2e-14
    psi = basis@c; hpsi = h@psi
    e = np.vdot(psi,hpsi); variance = np.linalg.norm(hpsi-e*psi)**2
    assert new['energy_per_site'] == pytest.approx(e.real/12,abs=1e-14)
    assert new['variance_per_site'] == pytest.approx(variance/12,abs=2e-14)
    assert variance > 1e-5  # Root Ritz residual does NOT imply full physical variance zero.
    assert new['root_Ritz_residual'] < 1e-12


def test_reference_bound_and_zero_parent_rejected():
    with pytest.raises(ValueError):
        reference_site_spins(7)
    with pytest.raises(ValueError):
        boundary_residual_coefficients(np.zeros((4,4)),reference_site_spins(2),reference_site_spins(2),[])


def test_fixed_gap_resolvent_matches_restricted_dense_solve_and_complex_gauge():
    rng = np.random.default_rng(2610234910)
    u = np.linalg.qr(rng.normal(size=(8,8))+1j*rng.normal(size=(8,8)))[0]
    energies = np.asarray([-3.,-2.7,-1.3,-.4,.2,.9,1.6,3.2])
    h0 = (u*energies)@u.conj().T
    retained = u[:,:3]; complement = u[:,3:]
    response = rng.normal(size=8)+1j*rng.normal(size=8)
    delta,info = fixed_gap_boundary_correction(h0,retained,response,reference_energy=-3.,gap=1.)
    restricted = complement.conj().T@(h0+4*np.eye(8))@complement
    exact = complement@np.linalg.solve(restricted,complement.conj().T@response)
    assert np.max(abs(delta-exact)) < 3e-15
    assert info['protected_leakage'] < 3e-15
    assert info['projected_solve_residual'] < 5e-15
    gauge = np.linalg.qr(rng.normal(size=(8,8))+1j*rng.normal(size=(8,8)))[0]
    transformed,_ = fixed_gap_boundary_correction(gauge.conj().T@h0@gauge,
        gauge.conj().T@retained,gauge.conj().T@response,reference_energy=-3.,gap=1.)
    assert np.max(abs(transformed-gauge.conj().T@delta)) < 3e-15


def test_fixed_gap_preserves_singlet_charge_on_four_spin_boundary_oracle():
    spin = reference_site_spins(4)
    h0 = sum((a@b for i,j in ((0,1),(2,3))
              for a,b in zip(spin[i],spin[j],strict=True)),np.zeros((16,16),complex))
    v = sum((a@b for a,b in zip(spin[1],spin[2],strict=True)),np.zeros((16,16),complex))
    parent = np.kron(singlet(),singlet())
    delta,info = fixed_gap_boundary_correction(h0,parent[:,None],v@parent,
                                              reference_energy=-1.5,gap=1.)
    # The boundary perturbation creates two triplets coupled to a singlet;
    # its uncoupled excitation cost is exactly two J1 plus the fixed gap.
    assert np.max(abs(delta-v@parent/3)) < 2e-15
    for component in range(3):
        total = sum((site[component] for site in spin),np.zeros((16,16),complex))
        assert np.linalg.norm(total@delta) < 2e-15
    assert info['minimum_denominator'] == pytest.approx(1.,abs=2e-15)


def test_fixed_gap_rejects_noncommuting_or_nonpositive_denominator():
    h0 = np.diag([0.,1.,2.]); response = np.ones(3)
    with pytest.raises(ValueError,match='commute'):
        fixed_gap_boundary_correction(h0,np.asarray([[1.],[1.],[0.]])/np.sqrt(2),
                                      response,reference_energy=0.)
    with pytest.raises(ValueError,match='positive definite'):
        fixed_gap_boundary_correction(h0,np.eye(3)[:,:1],response,reference_energy=2.)
    for gap in (0.,-1.,np.nan,1j):
        with pytest.raises(ValueError):
            fixed_gap_boundary_correction(h0,np.eye(3)[:,:1],response,
                                          reference_energy=0.,gap=gap)
