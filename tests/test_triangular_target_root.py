"""Target-root validation independent of optimized checkpoint values."""
import jax
import numpy as np
import pytest
from itertools import combinations
from math import factorial

from su2_nqs.models.triangular_mixed_clebsch import (
    MixedClebschConfig, init_mixed_clebsch_family, make_mixed_clebsch_amplitude,
)
from su2_nqs.models.triangular_target_root import (
    make_target_root_amplitude, singlet_root_coefficients, target_root_layout,
    collective_spin_root_coefficients,
)
from su2_nqs.lattices.triangular_schur_hierarchy import triangular_schur_hierarchy
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group


def test_target_root_replays_singlet_and_all_total_spin_sectors():
    config = MixedClebschConfig(6, 9, 6)
    bank = init_mixed_clebsch_family(jax.random.PRNGKey(781), config)["ternary_free"]
    rng = np.random.default_rng(782)
    legacy = jax.jit(make_mixed_clebsch_amplitude(config))
    for q in (0, 2, 4, 6):
        roots = (singlet_root_coefficients(bank, config) if q == 0 else
                 tuple(rng.normal(size=(dl, dr)) for _, _, dl, dr in target_root_layout(config, q)))
        query = jax.jit(jax.vmap(make_target_root_amplitude(config, q, q), in_axes=(None, None, 0)))
        bits = rng.permutation(np.r_[np.ones(18+q//2, np.int32), np.zeros(18-q//2, np.int32)])
        rows = [bits]
        for i in np.flatnonzero(bits):
            for j in np.flatnonzero(1-bits):
                row = bits.copy(); row[i], row[j] = row[j], row[i]
                rows.append(row)
        amplitudes = np.asarray(query(bank, roots, np.asarray(rows)))
        # S² in a fixed physical M=q/2 slice: diagonal M²+N/2,
        # plus every unlike-spin exchange with coefficient one.
        residual = ((q*q/4+18-q*(q+2)/4)*amplitudes[0]+amplitudes[1:].sum())
        scale = np.linalg.norm(amplitudes)
        assert scale > 0
        assert abs(residual)/scale < 2e-11
        if q == 0:
            np.testing.assert_allclose(amplitudes[0], legacy(bank, bits), rtol=2e-12, atol=1e-15)


@pytest.mark.parametrize("q,m", [(1,1), (2,1), (38,0), (-2,0)])
def test_invalid_target_quantum_numbers_fail_closed(q,m):
    with pytest.raises((ValueError, TypeError)):
        make_target_root_amplitude(MixedClebschConfig(6,9,6), q,m)


def test_collective_descendants_match_independent_physical_spin_operations():
    config=MixedClebschConfig(6,9,6)
    bank=init_mixed_clebsch_family(jax.random.PRNGKey(789),config)["ternary_free"]
    tree=triangular_schur_hierarchy(6)
    left=tree.coarse_supports[tree.binary_merges[-1][0]]
    parent=jax.jit(jax.vmap(make_mixed_clebsch_amplitude(config),in_axes=(None,0)))
    rng=np.random.default_rng(790)
    for spin in (1,2,3):
        q=2*spin
        bits=rng.permutation(np.r_[np.ones(18+spin,np.int32),np.zeros(18-spin,np.int32)])
        rows=[]
        for subset in combinations([i for i in left if bits[i]],spin):
            value=bits.copy();value[list(subset)]=0;rows.append(value)
        physical=factorial(spin)*np.asarray(parent(bank,np.asarray(rows))).sum()
        roots=collective_spin_root_coefficients(bank,config,q)
        answer=float(make_target_root_amplitude(config,q,q)(bank,roots,bits))
        assert abs(answer-physical)/max(abs(physical),abs(answer),1e-200)<2e-11


def test_actual_CT6_A1_uniform_cover_excludes_false_collective_dipole_signal():
    tree=triangular_schur_hierarchy(6)
    region=tree.coarse_supports[tree.binary_merges[-1][0]]
    images=np.asarray(triangular_hnf_gamma_a1_group(((6,0),(0,6))))
    weights=np.bincount(images[:,region].ravel(),minlength=36)/len(images)
    # Thus sum_g U_g S_A+ U_g^-1 / |G| = .5 S_total+.
    # Applied to the SAME A1 singlet parent this is identically zero.
    np.testing.assert_array_equal(weights,np.full(36,.5))
