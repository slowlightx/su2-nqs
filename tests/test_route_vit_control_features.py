from __future__ import annotations

import numpy as np

from su2_nqs.lattices.route_vit_control_features import (
    maximally_mismatched_cyclic_permutation,
    permute_route_features,
)


def test_maximum_cyclic_scramble_is_deranged_and_preserves_token_multiset() -> None:
    features = (
        (0.0, 1.0),
        (1.0, 0.0),
        (2.0, -1.0),
        (-1.0, 2.0),
    )
    permutation = maximally_mismatched_cyclic_permutation(features)
    assert sorted(permutation) == list(range(len(features)))
    assert all(source != target for target, source in enumerate(permutation))
    scrambled = permute_route_features(features, permutation)
    np.testing.assert_allclose(
        np.sort(np.asarray(scrambled), axis=0),
        np.sort(np.asarray(features), axis=0),
        atol=0.0,
        rtol=0.0,
    )
