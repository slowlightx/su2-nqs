from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.rvb_covering_oracle import (
    covering_relation_features,
    dimer_matching_orbits,
    enumerate_perfect_matchings,
)
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_hnf import triangular_hnf_j1j2_bonds
from su2_nqs.models.rvb_covering_transformer import (
    RVBCoveringTransformerConfig,
    init_rvb_covering_transformer,
    rvb_covering_transformer_coefficients,
    rvb_covering_transformer_parameter_count,
)


SUPERCELL = ((6, 4), (0, 2))


def _triangular_orbit_features():
    bonds = triangular_hnf_j1j2_bonds(SUPERCELL, j1=1.0, j2=0.0)
    group = triangular_hnf_gamma_a1_group(SUPERCELL)
    matchings = enumerate_perfect_matchings(12, bonds)
    orbit = dimer_matching_orbits(matchings, group)[-1]
    features = covering_relation_features(orbit.members, bonds, group)
    return features


def test_covering_transformer_rejects_invalid_capacity() -> None:
    with pytest.raises(ValueError):
        RVBCoveringTransformerConfig(6, 1, 12, embedding_dim=3, head_count=2)
    with pytest.raises(ValueError):
        RVBCoveringTransformerConfig(0, 1, 12)


def test_deeper_transformer_initialization_consumes_the_right_number_of_keys() -> None:
    config = RVBCoveringTransformerConfig(
        token_count=6,
        bond_class_count=1,
        relation_class_count=12,
        embedding_dim=4,
        depth=2,
    )
    params = init_rvb_covering_transformer(jax.random.PRNGKey(19), config)
    assert len(params["layers"]) == 2


def test_relation_labels_and_attention_are_exactly_space_group_invariant() -> None:
    features = _triangular_orbit_features()
    assert features.bond_class_count == 1
    assert features.relation_class_count == 12
    config = RVBCoveringTransformerConfig(
        token_count=6,
        bond_class_count=features.bond_class_count,
        relation_class_count=features.relation_class_count,
        embedding_dim=4,
        head_count=1,
        depth=1,
    )
    params = init_rvb_covering_transformer(jax.random.PRNGKey(17), config)
    values = rvb_covering_transformer_coefficients(
        params,
        jnp.asarray(features.bond_classes),
        jnp.asarray(features.relation_classes),
        config,
    )
    np.testing.assert_allclose(values, values[0], atol=2.0e-14)
    assert rvb_covering_transformer_parameter_count(params) == 202


def test_parameter_tangent_cannot_leak_out_of_space_group_orbit() -> None:
    features = _triangular_orbit_features()
    config = RVBCoveringTransformerConfig(6, 1, 12, embedding_dim=4)
    params = init_rvb_covering_transformer(jax.random.PRNGKey(23), config)
    tangent = jax.tree_util.tree_map(
        lambda value: jnp.full_like(value, 0.013), params
    )

    def evaluate(candidate):
        return rvb_covering_transformer_coefficients(
            candidate,
            jnp.asarray(features.bond_classes),
            jnp.asarray(features.relation_classes),
            config,
        )

    values, derivatives = jax.jvp(evaluate, (params,), (tangent,))
    np.testing.assert_allclose(values, values[0], atol=2.0e-14)
    np.testing.assert_allclose(derivatives, derivatives[0], atol=3.0e-13)
