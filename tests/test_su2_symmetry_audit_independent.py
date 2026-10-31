"""Independent Haar-average and whole-chain quadratic symmetry oracles.

The oracle integrates physical collective rotations without CG tables, and
the statistical oracle explicitly sums distinct-chain pairs. Neither local
rotation invariance nor a noise-compatible defect certifies a global singlet.
"""

import numpy as np
import pytest

from su2_nqs.benchmarks.su2_physics_protocol import (
    rotational_defect_chain_audit,
    rotational_symmetry_defect,
    regional_reference_chain_audit,
)


def _physical_spins(n):
    spin = [np.asarray([[0, 1], [1, 0]], complex) / 2,
            np.asarray([[0, 1j], [-1j, 0]], complex) / 2,
            np.diag([-.5, .5]).astype(complex)]
    sites = []
    for location in range(n):
        axes = []
        for single in spin:
            matrix = np.ones((1, 1), complex)
            for index in range(n):
                matrix = np.kron(matrix, single if index == location else np.eye(2))
            axes.append(matrix)
        sites.append(axes)
    return sites, [sum(site[axis] for site in sites) for axis in range(3)]


def _physical_haar_average(matrix):
    """Exact finite quadrature on even 2/4/6-spin polynomial representations.

    Alpha/gamma integrals select equal physical S^z indices. The remaining
    beta integral uses n+1 Gauss-Legendre nodes, independent of a Schur basis.
    """
    n = matrix.shape[0].bit_length() - 1
    _, totals = _physical_spins(n)
    magnetic = np.diag(totals[2]).real
    selection = magnetic[:, None] == magnetic[None, :]
    initial = matrix * selection
    eigenvalues, vectors = np.linalg.eigh(totals[1])
    nodes, weights = np.polynomial.legendre.leggauss(n + 1)
    average = np.zeros_like(matrix, dtype=np.complex128)
    for cosine, weight in zip(nodes, weights, strict=True):
        rotation = (vectors * np.exp(-1j * np.arccos(cosine) * eigenvalues)) @ vectors.conj().T
        average += weight / 2 * (rotation @ initial @ rotation.conj().T)
    return average * selection


def test_schur_defect_matches_independent_physical_haar_average_and_scalar_data():
    rng = np.random.default_rng(2026100851)
    z = rng.normal(size=(16, 6)) + 1j * rng.normal(size=(16, 6))
    rho = z @ z.conj().T
    rho /= np.trace(rho)
    original = rho.copy()
    sites, totals = _physical_spins(4)
    twirled = _physical_haar_average(rho)
    np.testing.assert_allclose(np.trace(twirled), 1., atol=3e-14, rtol=0)
    assert np.min(np.linalg.eigvalsh(twirled)) > -3e-14
    for spin in totals:
        np.testing.assert_allclose(twirled @ spin - spin @ twirled, 0., atol=4e-14, rtol=0)
    delta = rho - twirled
    report = rotational_symmetry_defect(rho)
    np.testing.assert_allclose(
        report["rotational_defect_trace_distance"], .5 * abs(np.linalg.eigvalsh(delta)).sum(),
        atol=4e-14, rtol=0,
    )
    np.testing.assert_allclose(report["rotational_defect_frobenius"], np.linalg.norm(delta), atol=4e-14, rtol=0)
    np.testing.assert_array_equal(rho, original)
    assert report["input_repaired"] is False
    assert report["strict_entanglement_spectrum_reported"] is False
    assert report["strict_su2_eligible"] is False
    for first in range(4):
        for second in range(first + 1, 4):
            scalar = sum(sites[first][axis] @ sites[second][axis] for axis in range(3))
            np.testing.assert_allclose(np.trace(rho @ scalar), np.trace(twirled @ scalar), atol=4e-14, rtol=0)
            raw_zz = sites[first][2] @ sites[second][2]
            np.testing.assert_allclose(
                np.trace(twirled @ raw_zz), np.trace(rho @ scalar) / 3, atol=4e-14, rtol=0,
            )


def _pairwise_cross_chain_estimate(means):
    deltas = np.stack([row - _physical_haar_average(row) for row in means])
    count = len(means)
    numerator = sum(np.vdot(deltas[first], deltas[second]).real
                    for first in range(count) for second in range(count) if first != second)
    numerator /= count * (count - 1)
    average_normalization = np.trace(means, axis1=1, axis2=2).real.mean()
    return numerator / average_normalization**2, numerator


def test_noise_audit_matches_explicit_distinct_chain_pairs_and_delete_chain_error():
    norms = np.asarray([.8, 1., 1.2, .9, 1.1])
    coefficients = np.asarray([-.7, .6, -.3, .8, -.4])
    anisotropy = np.diag([1., -1., -1., 1.]) * .04
    means = np.stack([normalization * np.eye(4) / 4 + coefficient * anisotropy
                      for normalization, coefficient in zip(norms, coefficients, strict=True)])
    records = np.repeat(means, 3, axis=0)
    chain_ids = np.repeat(np.arange(5), 3)
    report = rotational_defect_chain_audit(records, chain_ids)
    expected, numerator = _pairwise_cross_chain_estimate(means)
    deleted = np.asarray([_pairwise_cross_chain_estimate(np.delete(means, index, axis=0))[0]
                          for index in range(5)])
    expected_se = np.sqrt(4 * np.mean((deleted-deleted.mean())**2))
    np.testing.assert_allclose(report["cross_chain_squared_defect"], expected, atol=3e-14, rtol=0)
    np.testing.assert_allclose(report["cross_chain_numerator"], numerator, atol=3e-14, rtol=0)
    np.testing.assert_allclose(report["whole_chain_jackknife_SE"], expected_se, atol=3e-14, rtol=0)
    assert expected < 0
    assert report["negative_estimate_retained"] is True
    assert report["normalization_ratio_bias_removed"] is False
    assert report["exact_SU2_certified"] is False
    assert report["mixing_certified"] is False
    sites, _ = _physical_spins(2)
    scalar = sum(sites[0][axis] @ sites[1][axis] for axis in range(3))
    anisotropy_operator = sites[0][2] @ sites[1][2] - scalar / 3
    summed = means.sum(axis=0)
    grand = summed / np.trace(summed).real
    delete_values = np.asarray([
        np.trace((summed - row) @ anisotropy_operator).real / np.trace(summed - row).real
        for row in means
    ])
    linear = report["linear_observable_chain_estimates"]["axis_anisotropy_pair_0_1"]
    np.testing.assert_allclose(linear["value"], np.trace(grand @ anisotropy_operator).real, atol=3e-14, rtol=0)
    np.testing.assert_allclose(
        linear["whole_chain_jackknife_SE"], np.sqrt(4 * np.mean((delete_values-delete_values.mean())**2)),
        atol=3e-14, rtol=0,
    )
    # Relabelling temporal records into unequal pseudo-chains is rejected.
    with pytest.raises(ValueError, match="equal-size"):
        rotational_defect_chain_audit(records, np.asarray([0] * 4 + [1] * 3 + [2] * 3 + [3] * 5))


def test_local_rotation_invariance_is_not_a_global_singlet_certificate():
    up, down = np.eye(4)[3], np.eye(4)[0]
    triplet_zero = np.asarray([0., 1., 1., 0.]) / np.sqrt(2)
    rho = (np.outer(up, up) + np.outer(down, down) + np.outer(triplet_zero, triplet_zero)) / 3
    _, totals = _physical_spins(2)
    s2 = sum(axis @ axis for axis in totals)
    np.testing.assert_allclose(np.trace(rho @ s2), 2., atol=2e-14, rtol=0)
    report = rotational_symmetry_defect(rho)
    assert report["strict_su2_eligible"] is True
    assert report["strict_entanglement_spectrum_reported"] is False
    assert report["rotational_defect_frobenius"] < 2e-14


def test_nonpositive_invariant_estimator_does_not_get_density_matrix_bounds():
    # This estimator is exactly invariant as an operator, but it is not a
    # physical density matrix. Invariance cannot license fake entropies.
    singlet = np.asarray([0., -1., 1., 0.]) / np.sqrt(2)
    triplet_projector = np.eye(4) - np.outer(singlet, singlet)
    rho = -.1 * np.outer(singlet, singlet) + (1.1 / 3) * triplet_projector
    before = rho.copy()
    report = rotational_symmetry_defect(rho)
    assert report["rotational_defect_frobenius"] < 2e-14
    assert report["physical_matrix_valid"] is False
    assert report["strict_su2_eligible"] is False
    assert report["continuity_bounds"] is None
    np.testing.assert_array_equal(rho, before)


def test_numerical_positivity_tolerance_does_not_license_theorem_bounds():
    singlet = np.asarray([0., -1., 1., 0.]) / np.sqrt(2)
    triplet_projector = np.eye(4) - np.outer(singlet, singlet)
    rho = -1e-10 * np.outer(singlet, singlet) + ((1 + 1e-10) / 3) * triplet_projector
    report = rotational_symmetry_defect(rho, tolerance=1e-9)
    # Deliberately distinguish a numerical regional tolerance diagnostic
    # from the positive-density-matrix assumptions of continuity theorems.
    assert report["physical_matrix_valid"] is True
    assert report["continuity_bounds_eligible"] is False
    assert report["continuity_bounds"] is None
    assert report["input_repaired"] is False
    assert report["strict_entanglement_spectrum_reported"] is False


def test_saved_physics_audit_preserves_source_identity_and_no_new_state_claims():
    import hashlib
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    directory = root / "projects/clebsch_tree_nqs/results/spin_symmetry_defect_quality_v1"
    if not directory.exists():
        pytest.skip("optional immutable cached-state symmetry audit absent")
    result = json.loads((directory / "results.json").read_text())
    registration = json.loads((directory / "registration.json").read_text())
    seal = json.loads((directory / "completion.json").read_text())
    for name, key in (("results.json", "results_sha256"),
                      ("registration.json", "registration_sha256"),
                      ("actual_rdms.npz", "actual_rdms_sha256")):
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == seal[key]
    for relative, digest in registration["inputs"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest
    for relative, digest in registration["sources"].items():
        assert hashlib.sha256((directory / "source_snapshots" / relative).read_bytes()).hexdigest() == digest
    assert len(result["records"]) == 90
    assert len(registration["inputs"]) == 152
    assert all(result[key] == 0 for key in ("new_training", "new_ED", "new_samples",
               "new_MPS_contractions", "new_H_calls", "new_amplitude_queries"))
    assert not result["phase_claim"]
    assert not result["strict_entanglement_spectrum_reported"]
    parents = [row for row in result["records"]
               if row["method"] == "ordinary_spatial_vit_4032" and row["order"] == 942]
    assert len(parents) == 3
    assert {row["params_sha256"] for row in parents} == {
        "2b2afb846a3b5501ce5014fd1e0f9832175e0818da872722ddc02a691899aa82"
    }
    assert all(row["quality"]["parameter_count"] == 4032 for row in parents)
    assert all(row["globally_strict_SU2_claim"] is False for row in parents)
    assert all(not row["chain_noise_audit"]["exact_SU2_certified"] for row in parents)
    assert all(len(set(row["independent_stream_seeds"])) == 2 for row in parents)
    # The currently independent cached readout belongs to the old local
    # network, not the separately running 99,904-parameter Kim production.
    assert all("stable_sr/audits/step00942" in row["source_audit"] for row in parents)


def test_native_spin_readout_replays_diagonal_projectors_covariance_and_operator_bound():
    """Check frozen Born labels without querying either a model or its H kernel."""
    import hashlib
    import json
    from math import comb
    from pathlib import Path
    from su2_nqs.su2.fusion_tree import Leaf
    from su2_nqs.su2.many_cell_schur import rectangular_four_spin_layout

    root = Path(__file__).resolve().parents[1]
    directory = root / "projects/clebsch_tree_nqs/results/spin_reduced_vit_train_v1"
    path = directory / "native_spin_readout.json"
    if not path.exists():
        pytest.skip("optional frozen native Schur diagonal readout absent")
    result = json.loads(path.read_text())
    payload = {key: value for key, value in result.items() if key != "result_sha256"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert digest == result["result_sha256"]
    for relative, expected in result["inputs_sha256"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
    for relative, expected in result["native_source_snapshot_sha256"].items():
        assert hashlib.sha256((directory / "source_snapshot" / relative).read_bytes()).hexdigest() == expected
    assert result["exact_global_SU2_singlet"] and not result["full_A1"]
    assert all(result[key] == 0 for key in (
        "new_H_calls", "new_amplitude_queries", "new_samples", "new_training", "new_RDM_contractions"))
    assert not result["multiplicity_coherences_inferred"]
    assert not result["strict_entanglement_spectrum_reported"]
    with np.load(directory / "untouched_audit.npz") as archive:
        labels = archive["labels"]
    assert labels.shape == (512, 35)
    layout = rectangular_four_spin_layout(6, 6)
    nodes = layout.tree.internal_nodes
    positions = {node: index for index, node in enumerate(nodes)}

    def leaf_indices(node):
        return [node.index] if isinstance(node, Leaf) else leaf_indices(node.left) + leaf_indices(node.right)

    def reflected_physical(model_site):
        row, column = divmod(layout.site_order[model_site], 6)
        return 6 * row + (-column) % 6

    def statistic(values):
        return float(np.mean(values)), float(np.std(values, ddof=1) / np.sqrt(512))

    def verify_summary(summary, values):
        expected_value, expected_error = statistic(values)
        np.testing.assert_allclose(summary["value"], expected_value, atol=3e-15, rtol=0)
        np.testing.assert_allclose(summary["iid_draw_SE"], expected_error, atol=3e-15, rtol=0)

    for index, node in enumerate(nodes):
        left = np.ones(512, int) if isinstance(node.left, Leaf) else labels[:, positions[node.left]]
        right = np.ones(512, int) if isinstance(node.right, Leaf) else labels[:, positions[node.right]]
        q = labels[:, index]
        assert np.all(abs(left - right) <= q) and np.all(q <= left + right)
        assert np.all((left + right - q) % 2 == 0)
    assert np.all(labels[:, -1] == 0)
    for record in result["cells"] + result["intercell_nodes"]:
        index = record["postorder_node_index"]
        model_sites = leaf_indices(nodes[index])
        assert record["model_sites"] == model_sites
        assert record["physical_sites"] == [reflected_physical(site) for site in model_sites]
        q = labels[:, index]
        for row in record["spin_weight_estimates"]:
            indicator = (q == row["q"]).astype(float)
            assert row["count"] == int(indicator.sum())
            verify_summary(row, indicator)
        verify_summary(record["spin_squared"], q * (q + 2) / 4)
    cell_q = labels[:, layout.cell_root_indices]
    pooled = result["pooled_cell_readout"]
    probability_draws = np.stack([(cell_q == q).mean(axis=1) for q in (0, 2, 4)], axis=1)
    for column, row in enumerate(pooled["spin_weight_estimates"]):
        verify_summary(row, probability_draws[:, column])
    np.testing.assert_allclose(pooled["covariance_of_spatial_average_probabilities_mean"],
                               np.cov(probability_draws, rowvar=False, ddof=1) / 512,
                               atol=3e-15, rtol=0)
    casimir_draw = (cell_q * (cell_q + 2) / 4).mean(axis=1)
    verify_summary(pooled["spin_squared"], casimir_draw)
    verify_summary(pooled["all_six_pair_spin_dot_sum"], (casimir_draw - 3) / 2)

    # Independent periodic metric construction, not the production bond API.
    def coupling(first, second):
        row_a, col_a = divmod(first, 6)
        row_b, col_b = divmod(second, 6)
        distances = [(row_b-row_a+6*x)**2 + (col_b-col_a+6*y)**2
                     - (row_b-row_a+6*x)*(col_b-col_a+6*y)
                     for x in (-1, 0, 1) for y in (-1, 0, 1)]
        return 1. if 1 in distances else .125 if 3 in distances else 0.

    sites, totals = _physical_spins(4)
    s2 = sum(axis @ axis for axis in totals)
    lower_bound_matrix = (s2 - 3*np.eye(16)) / 2 - .21875*np.eye(16)
    bound = result["intracell_H_operator_bound"]
    for cell in bound["cell_bonds"]:
        physical = result["cells"][cell["cell"]]["physical_sites"]
        expected = {(min(physical[a], physical[b]), max(physical[a], physical[b])): coupling(physical[a], physical[b])
                    for a in range(4) for b in range(a+1, 4)}
        actual = {tuple(edge["sites"]): edge["coupling"] for edge in cell["edges"]}
        assert actual == expected
        assert list(expected.values()).count(1.) == 5
        assert list(expected.values()).count(.125) == 1
        matrix = sum(coupling(physical[a], physical[b]) * sum(sites[a][axis] @ sites[b][axis] for axis in range(3))
                     for a in range(4) for b in range(a+1, 4))
        assert np.linalg.eigvalsh(matrix - lower_bound_matrix).min() >= -3e-14
    verify_summary(bound["mean_cell_lower_bound_expectation_estimate"], (casimir_draw - 3) / 2 - .21875)

    def multiplicity(n, q):
        k = (n-q) // 2
        return comb(n, k) - (comb(n, k-1) if k else 0)

    reference = result["kinematic_uniform_complete_singlet_reference"]
    dimension = multiplicity(36, 0)
    assert reference["complete_singlet_dimension"] == dimension == 477638700
    counts = [multiplicity(4, q) * multiplicity(32, q) for q in (0, 2, 4)]
    assert sum(counts) == dimension
    for row, count in zip(reference["spin_weights"], counts, strict=True):
        assert row["joint_multiplicity_count"] == count
        np.testing.assert_allclose(row["probability"], count / dimension, atol=3e-16, rtol=0)


def test_reference_discrepancy_uses_explicit_cross_chain_pairs_and_physical_haar_decomposition():
    coefficients = np.asarray([-.7, .6, -.3, .8, -.4])
    anisotropy = np.diag([1., -1., -1., 1.]) * .04
    reference = np.eye(4, dtype=complex)/4
    means = np.stack([reference + coefficient*anisotropy for coefficient in coefficients])
    records = np.repeat(means, 3, axis=0)
    groups = np.repeat(np.arange(5), 3)
    actual = regional_reference_chain_audit(records, groups, reference)
    raw = means-reference
    invariant = np.stack([_physical_haar_average(row) for row in raw])
    features = (raw, invariant, raw-invariant)
    def quadratic(values):
        n = len(values)
        return sum(np.vdot(values[i], values[j]).real for i in range(n) for j in range(n) if i != j)/(n*(n-1))
    deletes = []
    for name, values in zip(('raw', 'invariant_operator', 'rotation_noninvariant'), features, strict=True):
        expected = quadratic(values)
        deleted = np.asarray([quadratic(np.delete(values, i, axis=0)) for i in range(5)])
        deletes.append(deleted)
        row = actual['features'][name]
        np.testing.assert_allclose(row['cross_chain_squared_frobenius_discrepancy'], expected, atol=3e-14, rtol=0)
        np.testing.assert_allclose(row['whole_chain_delete_jackknife_SE'],
                                   np.sqrt(4*np.mean((deleted-deleted.mean())**2)), atol=3e-14, rtol=0)
        assert row['negative_estimate_retained'] == (expected < 0)
    centered = np.stack(deletes, axis=1)
    centered -= centered.mean(axis=0)
    np.testing.assert_allclose(actual['whole_chain_delete_covariance'], 4*centered.T@centered/5, atol=3e-14, rtol=0)
    assert actual['decomposition_closure_error'] < 3e-14
    assert not actual['normalization_ratio_bias_present']
    assert not actual['physical_state_replaced']
    assert not actual['strict_entanglement_spectrum_reported']


def test_actual_5600_vit_physics_identity_whole_chain_errors_and_scalar_spin_projectors():
    import hashlib
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    directory = root/'projects/clebsch_tree_nqs/results/spin_spatial_vit_5600_physics_v1'
    if not (directory/'results.json').exists():
        pytest.skip('optional frozen Kim5600 physics artifact absent')
    result = json.loads((directory/'results.json').read_text())
    registration = json.loads((directory/'registration.json').read_text())
    for value in (result, registration):
        expected = hashlib.sha256(json.dumps({k:v for k,v in value.items() if k != 'result_sha256'},
                           sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        assert expected == value['result_sha256']
    assert result['registration_sha256'] == registration['result_sha256']
    assert hashlib.sha256((directory/'physics.npz').read_bytes()).hexdigest() == result['raw_sha256']
    for relative, digest in registration['source_manifest'].items():
        assert hashlib.sha256((directory/'source_snapshots'/relative).read_bytes()).hexdigest() == digest
    for relative, digest in registration['inputs_sha256'].items():
        local = root/'projects/clebsch_tree_nqs/results'/relative.split('/results/', 1)[1] if relative.startswith('/share/kim/') else root/relative
        assert hashlib.sha256(local.read_bytes()).hexdigest() == digest
    assert len(registration['source_manifest']) == 75
    assert result['step'] == 5600 and result['model_parameter_count'] == 99904
    assert result['params_sha256'] == '4336e48aedee5451654de2e366ec8ee796f948d5d6ff3080e06247489a37f116'
    assert result['full_A1'] and not result['strict_SU2']
    assert result['new_H_calls'] == result['new_samples'] == result['new_training'] == 0
    with np.load(directory/'physics.npz') as archive:
        rho, matrices, ids, scalar = [archive[key] for key in ('rho', 'per_record_matrices', 'chain_ids', 'scalar')]
        assert tuple(archive['region_sites']) == (14, 15, 20, 21)
        assert archive['states'].shape == (512, 36)
    chains, counts = np.unique(ids, return_counts=True)
    assert len(chains) == 128 and set(counts) == {4}
    np.testing.assert_allclose(np.trace(matrices, axis1=1, axis2=2), 1., atol=3e-14, rtol=0)
    np.testing.assert_allclose(matrices.mean(axis=0), rho, atol=3e-14, rtol=0)
    means = np.stack([scalar[ids == chain].mean(axis=0) for chain in chains])
    for column, name in enumerate(result['structure_factors']):
        report = result['structure_factors'][name]['scalar']
        np.testing.assert_allclose(report['value'], means[:, column].real.mean(), atol=3e-14, rtol=0)
        np.testing.assert_allclose(report['whole_chain_SE'], means[:, column].real.std(ddof=1)/np.sqrt(128), atol=3e-14, rtol=0)
    total = 36*means[:, 0].real
    np.testing.assert_allclose(result['total_spin_squared']['value'], total.mean(), atol=3e-14, rtol=0)
    np.testing.assert_allclose(result['total_spin_squared']['whole_chain_SE'], total.std(ddof=1)/np.sqrt(128), atol=3e-14, rtol=0)
    np.testing.assert_allclose(result['scalar_whole_chain_covariance_of_means'], np.cov(means.real, rowvar=False)/128,
                               atol=3e-14, rtol=0)
    _, spins = _physical_spins(4)
    s2 = sum(spin @ spin for spin in spins)
    eye = np.eye(16)
    # Independent polynomial Casimir projectors, not Schur columns.
    projectors = {0:(s2-2*eye)@(s2-6*eye)/12, 2:-s2@(s2-6*eye)/8, 4:s2@(s2-2*eye)/24}
    chain_matrices = np.stack([matrices[ids == chain].mean(axis=0) for chain in chains])
    grand = chain_matrices.mean(axis=0)
    deleted = np.stack([(chain_matrices.sum(axis=0)-row)/127 for row in chain_matrices])
    for q, projector in projectors.items():
        report = result['region']['spin_weight_observables'][f'p_q{q}']
        value = np.trace(grand@projector).real
        estimates = np.einsum('cij,ji->c', deleted, projector)
        error = np.sqrt(127*np.mean(abs(estimates-estimates.mean())**2))
        np.testing.assert_allclose(report['value'], value, atol=3e-14, rtol=0)
        np.testing.assert_allclose(report['chain_jackknife_se'], error, atol=3e-14, rtol=0)
    expected_twirl = _physical_haar_average(rho)
    distance = .5*abs(np.linalg.eigvalsh(rho-expected_twirl)).sum()
    np.testing.assert_allclose(result['region']['rotational_audit']['rotational_defect_trace_distance'], distance,
                               atol=3e-14, rtol=0)
    assert not result['region']['strict_multiplicity_spectrum_reported']
    assert not result['region']['input_repaired']
    assert not result['physical_phase_claim']


def test_actual_5600_reference_comparison_seals_and_physical_whole_chain_quadratic_oracle():
    """Replay finite-quality comparisons without their production Schur/statistical helpers."""
    import hashlib
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    candidate = root/'projects/clebsch_tree_nqs/results/spin_spatial_vit_5600_physics_v1'
    directory = candidate/'reference_quality_comparison_v1'
    if not (directory/'results.json').exists():
        pytest.skip('optional frozen actual5600 reference comparison absent')
    result = json.loads((directory/'results.json').read_text())
    registration = json.loads((directory/'registration.json').read_text())
    completion = json.loads((directory/'completion.json').read_text())
    for name in ('results', 'registration'):
        assert hashlib.sha256((directory/f'{name}.json').read_bytes()).hexdigest() == completion[f'{name}_sha256']
    assert result['registration'] == registration
    assert registration['candidate_params_sha256'] == '4336e48aedee5451654de2e366ec8ee796f948d5d6ff3080e06247489a37f116'
    assert registration['config']['step'] == 5600
    assert registration['config']['problem']['regions']['compact4'] == [14, 15, 20, 21]
    assert len(registration['inputs']) == 221
    assert len(registration['sources']) == 27
    for relative, digest in registration['inputs'].items():
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest() == digest
    for relative, digest in registration['sources'].items():
        assert hashlib.sha256((directory/'source_snapshots'/relative).read_bytes()).hexdigest() == digest
    assert result['new_training'] == result['new_ED'] == result['new_samples'] == result['new_MPS_contractions'] == result['new_H_calls'] == result['new_amplitude_queries'] == 0
    assert not result['strict_entanglement_spectrum_reported']
    assert not result['phase_claim'] and not result['mixing_certified']
    assert not result['candidate_own_quality_sequence_available']

    with np.load(candidate/'physics.npz') as cache:
        matrices, groups = cache['per_record_matrices'], cache['chain_ids']
        scalar, zz = cache['scalar'], cache['zz']
    ids = np.unique(groups)
    count = len(ids)
    assert count == 128
    means = np.stack([matrices[groups == group].mean(axis=0) for group in ids])
    # Haar integration is an independent physical rotation oracle, not CG columns.
    invariant_means = np.stack([_physical_haar_average(row) for row in means])
    reference_dir = root/registration['config']['reference_protocol']
    reference_raw = np.load(reference_dir/'readouts.npz')
    try:
        for row in result['records']:
            assert row['chi'] in (3072, 4096)
            key = '__'.join(('triangular_n36', 'dimer_reference', row['state_id'], 'compact4'))+'_raw_rho'
            reference = reference_raw[key]
            assert reference.shape == (16, 16)
            invariant_reference = _physical_haar_average(reference)
            raw = means-reference
            invariant = invariant_means-invariant_reference
            features = (raw, invariant, raw-invariant)
            deleted_values = []
            for name, values in zip(('raw', 'invariant_operator', 'rotation_noninvariant'), features, strict=True):
                flat = values.reshape(count, -1)
                gram = (flat.conj() @ flat.T).real
                np.fill_diagonal(gram, 0.)
                actual = gram.sum()/(count*(count-1))
                deleted = (gram.sum()-2*gram.sum(axis=1))/((count-1)*(count-2))
                error = np.sqrt((count-1)*np.mean((deleted-deleted.mean())**2))
                report = row['squared_distance']['features'][name]
                np.testing.assert_allclose(report['cross_chain_squared_frobenius_discrepancy'], actual, atol=4e-14, rtol=0)
                np.testing.assert_allclose(report['whole_chain_delete_jackknife_SE'], error, atol=4e-14, rtol=0)
                assert report['negative_estimate_retained'] == (actual < 0)
                deleted_values.append(deleted)
            centered = np.stack(deleted_values, axis=1)
            centered -= centered.mean(axis=0)
            np.testing.assert_allclose(row['squared_distance']['whole_chain_delete_covariance'],
                                       (count-1)*centered.T@centered/count, atol=4e-14, rtol=0)
            assert not row['squared_distance']['phase_claim']
            assert not row['squared_distance']['strict_entanglement_spectrum_reported']
        quality = result['reference_quality_sequence']
        for row in quality:
            key_a = '__'.join(('triangular_n36', 'dimer_reference', row['state_a'], 'compact4'))+'_raw_rho'
            key_b = '__'.join(('triangular_n36', 'dimer_reference', row['state_b'], 'compact4'))+'_raw_rho'
            np.testing.assert_allclose(row['raw_squared_frobenius_drift'],
                                       np.linalg.norm(reference_raw[key_b]-reference_raw[key_a])**2, atol=4e-14, rtol=0)
            assert not row['distance']['strict_entanglement_spectrum_comparison']
    finally:
        reference_raw.close()
    # Retain covariance by forming these contrasts within each whole chain.
    scalar_means = np.stack([scalar[groups == group].mean(axis=0).real for group in ids])
    zz_means = np.stack([zz[groups == group].mean(axis=0).real for group in ids])
    m_means = scalar_means[:, 3:6].mean(axis=1)
    m_report = result['structure_factor_paired_diagnostics']['M_scalar_orbit_mean']
    np.testing.assert_allclose(m_report['value'], m_means.mean(), atol=4e-14, rtol=0)
    np.testing.assert_allclose(m_report['whole_chain_SE'], m_means.std(ddof=1)/np.sqrt(count), atol=4e-14, rtol=0)
    names = ('Gamma', 'K', 'minus_K', 'M1', 'M2', 'M3')
    for i, name in enumerate(names):
        contrast = scalar_means[:, i]-3*zz_means[:, i]
        report = result['structure_factor_paired_diagnostics']['scalar_minus_three_zz'][name]
        np.testing.assert_allclose(report['value'], contrast.mean(), atol=4e-14, rtol=0)
        np.testing.assert_allclose(report['whole_chain_SE'], contrast.std(ddof=1)/np.sqrt(count), atol=4e-14, rtol=0)
