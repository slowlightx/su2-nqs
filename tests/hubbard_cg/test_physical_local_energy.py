import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.lattices.square import square_bonds
from su2_nqs.models.physical_tree import (
    MatchedHubbardAbelianTreeConfig,
    PhysicalHubbardTreeConfig,
    init_matched_hubbard_abelian_tree,
    init_physical_hubbard_tree,
    matched_hubbard_abelian_tree_amplitude,
)
from su2_nqs.operators.hubbard_fock import hubbard_fock_sector_matrix
from su2_nqs.vmc.hubbard_physical import (
    hubbard_physical_hpsi,
    padded_hubbard_fock_connectivity,
)


jax.config.update("jax_enable_x64", True)


def test_sampled_physical_hpsi_matches_full_exact_action() -> None:
    config = PhysicalHubbardTreeConfig(4, 4, 0, 0, 2, 0.7)
    bonds = square_bonds(2, 2, periodic=True)
    matrix, basis = hubbard_fock_sector_matrix(
        4, 4, 0, bonds, interaction=4.0
    )
    samples = np.asarray(basis, dtype=np.int32)
    connected, elements, valid, widths = padded_hubbard_fock_connectivity(
        samples, bonds, interaction=4.0
    )
    params = init_physical_hubbard_tree(jax.random.PRNGKey(19), config)

    @jax.jit
    def action(candidate):
        return hubbard_physical_hpsi(
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
    assert int(np.max(widths)) == connected.shape[1]


def test_same_tree_abelian_control_uses_identical_physical_action() -> None:
    config = MatchedHubbardAbelianTreeConfig(
        4, 4, target_m2=0, feature_dim=2, initialization_scale=0.7
    )
    bonds = square_bonds(2, 2, periodic=True)
    matrix, basis = hubbard_fock_sector_matrix(
        4, 4, 0, bonds, interaction=4.0
    )
    samples = np.asarray(basis, dtype=np.int32)
    connected, elements, valid, _ = padded_hubbard_fock_connectivity(
        samples, bonds, interaction=4.0
    )
    params = init_matched_hubbard_abelian_tree(
        jax.random.PRNGKey(20), config
    )

    amplitudes, hpsi = hubbard_physical_hpsi(
        params,
        jnp.asarray(samples),
        jnp.asarray(connected),
        jnp.asarray(elements),
        jnp.asarray(valid),
        config,
        amplitude_fn=matched_hubbard_abelian_tree_amplitude,
    )
    np.testing.assert_allclose(
        np.asarray(hpsi), matrix @ np.asarray(amplitudes), atol=3.0e-12
    )
