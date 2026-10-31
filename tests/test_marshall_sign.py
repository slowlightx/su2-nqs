import numpy as np
import pytest

from su2_nqs.benchmarks.exact_diagonalization import (
    computational_basis,
    heisenberg_sz_matrix,
)
from su2_nqs.benchmarks.marshall_sign import (
    bipartite_marshall_signs,
    diagonal_gauge_transform,
    offdiagonal_stoquastic_audit,
)


def _bits(n_sites: int) -> np.ndarray:
    return np.asarray(
        [
            [
                (state >> (n_sites - site - 1)) & 1
                for site in range(n_sites)
            ]
            for state in computational_basis(n_sites)
        ],
        dtype=np.int32,
    )


def test_marshall_sign_uses_msb_first_down_on_even_sites() -> None:
    configurations = np.asarray(
        (
            (0, 0, 0, 0),
            (1, 0, 0, 1),
            (1, 0, 1, 0),
            (0, 1, 1, 0),
        ),
        dtype=np.int32,
    )
    # Even A sites are 0 and 2; bit zero denotes down.
    np.testing.assert_array_equal(
        bipartite_marshall_signs(configurations),
        np.asarray((1.0, -1.0, 1.0, -1.0)),
    )


@pytest.mark.parametrize("n_sites", (4, 6))
def test_marshall_gauge_makes_even_periodic_chain_stoquastic(n_sites: int) -> None:
    bonds = tuple(
        (min(site, (site + 1) % n_sites), max(site, (site + 1) % n_sites), 1.0)
        for site in range(n_sites)
    )
    hamiltonian, _ = heisenberg_sz_matrix(n_sites, bonds)
    gauge = bipartite_marshall_signs(_bits(n_sites))
    gauged = diagonal_gauge_transform(hamiltonian, gauge)
    audit = offdiagonal_stoquastic_audit(gauged)
    assert audit["maximum_positive_real_offdiagonal"] == 0.0
    assert audit["maximum_absolute_imaginary_offdiagonal"] == 0.0
    np.testing.assert_allclose(
        np.linalg.eigvalsh(gauged), np.linalg.eigvalsh(hamiltonian), atol=1e-12
    )
    np.testing.assert_allclose(
        diagonal_gauge_transform(gauged, gauge), hamiltonian, atol=1e-12
    )


def test_marshall_helpers_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="bits"):
        bipartite_marshall_signs(np.asarray(((0, 2),), dtype=np.int32))
    with pytest.raises(ValueError, match="unique"):
        bipartite_marshall_signs(
            np.asarray(((0, 1),), dtype=np.int32), sublattice_a=(0, 0)
        )
    with pytest.raises(ValueError, match="unit modulus"):
        diagonal_gauge_transform(np.eye(2), np.asarray((1.0, 0.5)))

