import numpy as np

from su2_nqs.benchmarks.doob_conditioner import compile_doob_candidate_features


def test_doob_candidate_features_are_causal_and_telescope_exactly():
    actions = np.asarray(
        ((0, 0), (0, 1), (1, 0), (1, 1)), dtype=np.int32
    )
    probabilities = np.asarray((0.1, 0.2, 0.3, 0.4))
    values = np.asarray((1.0 + 0.2j, -0.4j, 2.0, -1.0 + 0.7j))
    result = compile_doob_candidate_features(
        probabilities, values, actions, action_dim=3
    )
    assert result.real_imag_features.shape == (4, 2, 3, 2)
    assert result.feature_count == 2
    assert result.maximum_prefix_invariance_error == 0.0
    assert result.maximum_selected_reconstruction_error < 1.0e-14
    np.testing.assert_array_equal(
        result.innovations[0, 0], result.innovations[1, 0]
    )
    np.testing.assert_array_equal(
        result.innovations[2, 0], result.innovations[3, 0]
    )
