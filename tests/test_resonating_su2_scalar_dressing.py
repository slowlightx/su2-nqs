from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.exact_diagonalization import (
    computational_basis,
    spin_squared_matrix,
)
from su2_nqs.benchmarks.projected_singlet_agp import permutation_group_closure
from su2_nqs.models.projected_singlet_agp import (
    ProjectedSingletAGPConfig,
    projected_singlet_agp_amplitudes,
)
from su2_nqs.models.resonating_su2_scalar_dressing import (
    ResonatingSU2ScalarDressingConfig,
    additive_su2_scalar_dressing,
    init_resonating_su2_scalar_dressing,
    projected_resonating_amplitudes,
    resonating_su2_scalar_pairing_bank,
)
from su2_nqs.su2.sz_basis import enumerate_sz_configs


def _dihedral_ring_group(n_sites: int) -> tuple[tuple[int, ...], ...]:
    translation = tuple((site + 1) % n_sites for site in range(n_sites))
    reflection = tuple((-site) % n_sites for site in range(n_sites))
    return permutation_group_closure((translation, reflection))


def _pairing_problem(n_sites: int = 8):
    rng = np.random.default_rng(2026092201)
    raw = rng.normal(size=(n_sites, n_sites))
    base = 0.5 * (raw + raw.T)
    np.fill_diagonal(base, 0.0)
    directions = []
    for distance in (1, 2, 3):
        value = np.zeros_like(base)
        for site in range(n_sites):
            value[site, (site + distance) % n_sites] = 1.0
            value[(site + distance) % n_sites, site] = 1.0
        directions.append(value)
    return base, np.stack(directions)


def _integer_bits(state: int, n_sites: int) -> tuple[int, ...]:
    return tuple(
        (int(state) >> (n_sites - site - 1)) & 1 for site in range(n_sites)
    )


def test_pair_attention_bank_is_symmetric_zero_onsite_and_scale_fixed() -> None:
    config = ResonatingSU2ScalarDressingConfig(
        rows=2, columns=4, embedding_dim=8, head_count=2, component_count=3
    )
    parameters = init_resonating_su2_scalar_dressing(
        jax.random.key(2026092202), config
    )
    base, directions = _pairing_problem()
    bank = np.asarray(
        resonating_su2_scalar_pairing_bank(parameters, base, directions, config)
    )
    assert bank.shape == (3, 8, 8)
    np.testing.assert_allclose(bank, np.swapaxes(bank, 1, 2), atol=2.0e-15)
    np.testing.assert_allclose(
        np.diagonal(bank, axis1=1, axis2=2), 0.0, atol=2.0e-15
    )
    np.testing.assert_allclose(
        np.linalg.norm(bank, axis=(1, 2)), np.linalg.norm(base), atol=2.0e-13
    )
    assert np.linalg.matrix_rank(bank.reshape(3, -1)) == 3


def test_projected_columns_and_additive_state_are_exact_singlets_and_scalars() -> None:
    n_sites = 8
    group = _dihedral_ring_group(n_sites)
    carrier_config = ProjectedSingletAGPConfig(
        n_sites=n_sites, group_images=group, parameter_mode="real"
    )
    dressing_config = ResonatingSU2ScalarDressingConfig(
        rows=2, columns=4, embedding_dim=8, head_count=2, component_count=2
    )
    parameters = init_resonating_su2_scalar_dressing(
        jax.random.key(2026092203), dressing_config
    )
    base, directions = _pairing_problem(n_sites)
    configurations = np.asarray(
        enumerate_sz_configs(n_sites, n_sites // 2), dtype=np.int8
    )
    resonating = projected_resonating_amplitudes(
        parameters,
        configurations,
        base,
        directions,
        dressing_config,
        carrier_config,
        microbatch_size=19,
    )
    base_packed = base[np.triu_indices(n_sites, 1)]
    carrier = projected_singlet_agp_amplitudes(
        {"pairing_real": jnp.asarray(base_packed)},
        configurations,
        carrier_config,
        microbatch_size=23,
    ).amplitudes
    coefficients = jnp.asarray((0.9, -0.35, 0.27), dtype=jnp.float64)
    dressed = additive_su2_scalar_dressing(
        carrier, resonating, coefficients
    ).dressed_amplitudes
    assert float(jnp.linalg.norm(dressed)) > 1.0e-9

    # Independent dense S^2 reference in the same MSB-first Sz=0 ordering.
    spin_squared, integer_basis = spin_squared_matrix(n_sites, n_up=n_sites // 2)
    positions = {
        tuple(row): index for index, row in enumerate(configurations.tolist())
    }
    ordered = np.asarray(
        [
            dressed[positions[_integer_bits(state, n_sites)]]
            for state in computational_basis(n_sites, n_up=n_sites // 2)
        ]
    )
    np.testing.assert_array_equal(integer_basis, computational_basis(n_sites, n_up=4))
    assert np.linalg.norm(spin_squared @ ordered) / np.linalg.norm(ordered) < 4.0e-13

    # Reynolds invariance is checked independently by actively translating
    # each requested spin configuration.
    translation = group[1]
    pulled = configurations[:, np.asarray(translation, dtype=np.int64)]
    translated_columns = projected_resonating_amplitudes(
        parameters,
        pulled,
        base,
        directions,
        dressing_config,
        carrier_config,
        microbatch_size=17,
    )
    np.testing.assert_allclose(translated_columns, resonating, atol=4.0e-12)


def test_additive_channel_exactly_nests_carrier_and_dressing_is_configuration_dependent() -> None:
    n_sites = 8
    group = _dihedral_ring_group(n_sites)
    carrier_config = ProjectedSingletAGPConfig(n_sites=n_sites, group_images=group)
    dressing_config = ResonatingSU2ScalarDressingConfig(
        rows=2, columns=4, embedding_dim=8, head_count=2, component_count=2
    )
    parameters = init_resonating_su2_scalar_dressing(
        jax.random.key(2026092204), dressing_config
    )
    base, directions = _pairing_problem(n_sites)
    configurations = np.asarray(enumerate_sz_configs(n_sites, 4), dtype=np.int8)
    carrier = projected_singlet_agp_amplitudes(
        {"pairing_real": jnp.asarray(base[np.triu_indices(n_sites, 1)])},
        configurations,
        carrier_config,
    ).amplitudes
    resonating = projected_resonating_amplitudes(
        parameters,
        configurations,
        base,
        directions,
        dressing_config,
        carrier_config,
    )
    nested = additive_su2_scalar_dressing(
        carrier, resonating, jnp.asarray((1.0, 0.0, 0.0))
    ).dressed_amplitudes
    np.testing.assert_array_equal(nested, carrier)

    dressed = additive_su2_scalar_dressing(
        carrier, resonating, jnp.asarray((1.0, 0.21, -0.13))
    ).dressed_amplitudes
    support = np.abs(np.asarray(carrier)) > 1.0e-10
    ratios = np.asarray(dressed)[support] / np.asarray(carrier)[support]
    assert float(np.std(ratios.real)) > 1.0e-4

    def objective(candidate):
        columns = projected_resonating_amplitudes(
            candidate,
            configurations[:11],
            base,
            directions,
            dressing_config,
            carrier_config,
            microbatch_size=7,
        )
        return jnp.sum(jnp.abs(columns) ** 2)

    gradient = jax.grad(objective)(parameters)
    assert all(
        np.all(np.isfinite(np.asarray(leaf)))
        for leaf in jax.tree_util.tree_leaves(gradient)
    )
    assert np.sqrt(
        sum(float(np.sum(np.asarray(leaf) ** 2)) for leaf in jax.tree_util.tree_leaves(gradient))
    ) > 1.0e-10
