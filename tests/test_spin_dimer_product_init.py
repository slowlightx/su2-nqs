from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.clebsch_tree_exact import fixed_m_spin_basis
from su2_nqs.benchmarks.exact_diagonalization import spin_squared_matrix
from su2_nqs.frontier.spin_dimer_product_init import (
    adjacent_singlet_product_amplitudes,
    init_spin_clebsch_tree_dimer_product,
)
from su2_nqs.lattices.pair_balanced_tree import pair_balanced_fusion_tree
from su2_nqs.models.spin_clebsch_tree import (
    SpinClebschTreeConfig,
    spin_clebsch_tree_amplitudes,
)
from su2_nqs.su2.fusion_tree import FusionTree


@pytest.mark.parametrize("n_sites", (4, 8, 12))
def test_pair_balanced_initializer_equals_independent_dimer_product(
    n_sites: int,
) -> None:
    config = SpinClebschTreeConfig(
        pair_balanced_fusion_tree(n_sites),
        target_q=0,
        target_m2=0,
        feature_dim=3,
        maximum_intermediate_q=4,
    )
    params = init_spin_clebsch_tree_dimer_product(config)
    basis = np.asarray(fixed_m_spin_basis(n_sites, 0), dtype=np.int32)
    actual = np.asarray(
        spin_clebsch_tree_amplitudes(params, jnp.asarray(basis), config)
    )
    expected = adjacent_singlet_product_amplitudes(basis)
    np.testing.assert_allclose(actual, expected, atol=2.0e-15, rtol=0.0)

    spin_squared, integer_basis = spin_squared_matrix(
        n_sites, n_up=n_sites // 2
    )
    positions = {value: index for index, value in enumerate(integer_basis)}
    integers = [
        sum(bit << (n_sites - site - 1) for site, bit in enumerate(row))
        for row in basis
    ]
    order = [positions[value] for value in integers]
    spin_squared = spin_squared[np.ix_(order, order)]
    assert np.linalg.norm(spin_squared @ actual) / np.linalg.norm(actual) < 2.0e-12


def test_leaf_balanced_n36_rejects_dimer_product_channel() -> None:
    config = SpinClebschTreeConfig(
        FusionTree.balanced(36),
        target_q=0,
        target_m2=0,
        feature_dim=2,
        maximum_intermediate_q=10,
    )
    with pytest.raises(ValueError, match="does not retain"):
        init_spin_clebsch_tree_dimer_product(config)


@pytest.mark.parametrize("n_sites", (1, 3, True))
def test_pair_balanced_tree_rejects_invalid_site_count(n_sites: object) -> None:
    with pytest.raises(ValueError, match="even integer"):
        pair_balanced_fusion_tree(n_sites)  # type: ignore[arg-type]
