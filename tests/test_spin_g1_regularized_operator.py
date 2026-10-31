from __future__ import annotations

from projects.clebsch_tree_nqs.workflows.spin_g1_regularized_operator import (
    LABELS,
    _validate,
    _words,
)


def _spec(phase: str = "development") -> dict[str, object]:
    result: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": "spin_g1_regularized_operator",
        "phase": phase,
        "seed": 17,
        "j2_over_j1": 0.125,
        "crossfit_folds": 4,
        "sampling": {
            "chain_count": 8,
            "burn_sweeps": 1,
            "measurement_blocks": 1,
            "decorrelation_sweeps": 1,
            "amplitude_batch_size": 32,
        },
        "kernel": {"terminal_capacity": 64, "entry_capacity": 100},
    }
    if phase == "untouched_audit":
        result["development_artifact"] = "sealed/results.json"
    return result


def test_registered_three_token_contract_has_thirty_one_words() -> None:
    words = _words()
    assert len(LABELS) == 3
    assert len(words) == 31
    assert max(map(len, words)) == 3
    assert len(set(words)) == len(words)


def test_development_and_separate_audit_configs_validate() -> None:
    _validate(_spec("development"))
    _validate(_spec("untouched_audit"))


def test_audit_config_must_name_a_frozen_development_artifact() -> None:
    spec = _spec("untouched_audit")
    del spec["development_artifact"]
    try:
        _validate(spec)
    except ValueError as error:
        assert "development_artifact" in str(error)
    else:
        raise AssertionError("audit must fail closed without frozen development")
