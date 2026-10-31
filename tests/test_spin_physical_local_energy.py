import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.clebsch_tree_exact import fixed_m_spin_basis
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.models.spin_clebsch_tree import (
    SpinClebschTreeConfig,
    init_spin_clebsch_tree,
)
from su2_nqs.operators.spin_config import heisenberg_spin_connectivity
from su2_nqs.operators.sz_graph_sparse import heisenberg_sz_graph_sparse
from su2_nqs.su2.fusion_tree import FusionTree
from su2_nqs.vmc.spin_physical import (
    padded_spin_connectivity,
    spin_physical_hpsi,
)
from su2_nqs.vmc.spin_device import spin_device_hpsi


def test_single_column_connectivity_combines_repeated_bonds() -> None:
    states, values = heisenberg_spin_connectivity(
        (0, 1), ((0, 1, 1.0), (1, 0, 0.5))
    )
    assert states == ((0, 1), (1, 0))
    np.testing.assert_allclose(values, (-0.375, 0.75), atol=0.0)


def test_sampled_spin_hpsi_matches_full_exact_action() -> None:
    bonds = square_j1j2_bonds(
        2, 2, j1=1.0, j2=0.3, periodic=False
    )
    matrix, matrix_basis = heisenberg_sz_graph_sparse(
        4, bonds, n_up=2
    )
    basis = fixed_m_spin_basis(4, 0)
    assert tuple(matrix_basis) == basis
    samples = np.asarray(basis, dtype=np.int32)
    connected, elements, valid, _ = padded_spin_connectivity(samples, bonds)
    config = SpinClebschTreeConfig(
        FusionTree.balanced(4), target_q=0, target_m2=0, feature_dim=2
    )
    params = init_spin_clebsch_tree(jax.random.PRNGKey(41), config)

    @jax.jit
    def action(candidate):
        return spin_physical_hpsi(
            candidate,
            jnp.asarray(samples),
            jnp.asarray(connected),
            jnp.asarray(elements),
            jnp.asarray(valid),
            config,
        )

    amplitudes, hpsi = action(params)
    np.testing.assert_allclose(
        np.asarray(hpsi), matrix @ np.asarray(amplitudes), atol=3.0e-12
    )

    @jax.jit
    def device_action(candidate):
        return spin_device_hpsi(
            candidate,
            jnp.asarray(samples),
            bonds,
            config,
        )

    device_amplitudes, device_hpsi = device_action(params)
    np.testing.assert_allclose(device_amplitudes, amplitudes, atol=2.0e-13)
    np.testing.assert_allclose(device_hpsi, hpsi, atol=3.0e-12)
