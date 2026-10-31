import numpy as np

from su2_nqs.benchmarks.coset_cut_diagnostics import coset_cut_diagnostics


def test_coset_cut_diagnostics_reconstructs_kl_response_and_residual():
    actions = np.asarray(((0, 0), (0, 1), (1, 0), (1, 1)), dtype=np.int32)
    source = np.asarray((1.0, 2.0, 3.0, 4.0), dtype=np.complex128)
    target = np.asarray((2.0, 1.0j, -1.0, 3.0), dtype=np.complex128)
    hamiltonian = np.asarray(
        (
            (1.0, 0.2, 0.0, 0.0),
            (0.2, 2.0, 0.3, 0.0),
            (0.0, 0.3, -1.0, 0.4),
            (0.0, 0.0, 0.4, 0.5),
        ),
        dtype=np.complex128,
    )
    result = coset_cut_diagnostics(source, target, hamiltonian, actions)
    assert result["conditional_born_kl"]["chain_rule_error"] < 1.0e-13
    assert result["hamiltonian_doob"]["orthogonality_error"] < 1.0e-13
    assert result["complex_response"]["maximum_state_reconstruction_error"] < 1.0e-13
    assert len(result["decisions"]) == 2
