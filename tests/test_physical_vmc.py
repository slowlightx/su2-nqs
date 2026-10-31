from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.models.physical_tree import (
    PhysicalSpinTreeConfig,
    init_physical_spin_tree,
    physical_spin_tree_amplitude,
)
from su2_nqs.operators.spin_config import heisenberg_spin_connectivity
from su2_nqs.su2.fusion_tree import FusionTree
from su2_nqs.su2.sz_basis import enumerate_sz_configs
from su2_nqs.vmc.gradients import vmc_loss
from su2_nqs.vmc.physical_vmc import (
    physical_hpsi_energy_variance,
    physical_hpsi_vmc_loss,
    physical_hpsi_vmc_loss_with_aux,
)
from su2_nqs.vmc.spin_physical import (
    padded_spin_connectivity,
    spin_physical_hpsi,
)


jax.config.update("jax_enable_x64", True)


def _assert_gradient_trees_close(left: object, right: object) -> None:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    assert left_tree == right_tree
    for actual, expected in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_allclose(actual, expected, atol=3.0e-11, rtol=2.0e-11)


def test_physical_hpsi_surrogate_matches_generic_log_vmc_gradient() -> None:
    config = PhysicalSpinTreeConfig(
        FusionTree.balanced(4),
        target_q=0,
        target_m2=0,
        feature_dim=2,
        maximum_intermediate_q=4,
        initialization_scale=0.3,
    )
    params = init_physical_spin_tree(jax.random.PRNGKey(50), config)
    samples = np.asarray(enumerate_sz_configs(4, 2), dtype=np.int32)
    bonds = square_j1j2_bonds(2, 2, j1=1.0, j2=0.2, periodic=True)
    connected, elements, valid, widths = padded_spin_connectivity(
        samples, bonds, padded_width=len(bonds) + 1
    )
    assert connected.shape[1] == len(bonds) + 1
    assert int(np.max(widths)) <= connected.shape[1]

    def hpsi(candidate):
        return spin_physical_hpsi(
            candidate,
            samples,
            connected,
            elements,
            valid,
            config,
        )

    physical_value, physical_gradient = jax.value_and_grad(
        lambda candidate: physical_hpsi_vmc_loss(candidate, hpsi_fn=hpsi)
    )(params)
    log_fn = lambda candidate, state: jnp.log(
        physical_spin_tree_amplitude(candidate, state, config).astype(jnp.complex128)
    )
    generic_value, generic_gradient = jax.value_and_grad(
        lambda candidate: vmc_loss(
            candidate,
            samples,
            log_fn,
            lambda state: heisenberg_spin_connectivity(state, bonds),
        )
    )(params)
    np.testing.assert_allclose(physical_value, generic_value, atol=2.0e-12, rtol=0.0)
    _assert_gradient_trees_close(physical_gradient, generic_gradient)
    (aux_value, auxiliary), aux_gradient = jax.value_and_grad(
        lambda candidate: physical_hpsi_vmc_loss_with_aux(
            candidate, hpsi_fn=hpsi
        ),
        has_aux=True,
    )(params)
    np.testing.assert_allclose(aux_value, physical_value, atol=2.0e-12, rtol=0.0)
    _assert_gradient_trees_close(aux_gradient, physical_gradient)
    assert int(auxiliary["nonfinite_local_count"]) == 0

    energy, variance, local = physical_hpsi_energy_variance(params, hpsi_fn=hpsi)
    np.testing.assert_allclose(energy, physical_value, atol=2.0e-12, rtol=0.0)
    assert float(variance) >= 0.0
    assert np.all(np.isfinite(np.asarray(local)))


def test_fixed_padding_rejects_underallocated_width() -> None:
    samples = np.asarray(enumerate_sz_configs(4, 2), dtype=np.int32)
    bonds = square_j1j2_bonds(2, 2, j1=1.0, j2=0.2, periodic=True)
    with pytest.raises(ValueError, match="smaller than required"):
        padded_spin_connectivity(samples, bonds, padded_width=1)
