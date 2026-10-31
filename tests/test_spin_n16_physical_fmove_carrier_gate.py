import json

import pytest

from projects.clebsch_tree_nqs.workflows.spin_n16_physical_fmove_carrier_gate import (
    _validate,
)


def _spec():
    return {
        "schema_version": 1,
        "domain": "spin",
        "task": "spin_n16_physical_fmove_carrier_gate",
        "preregistered_before_run": True,
        "source": {
            "result": "result.json",
            "result_sha256": "abc",
            "bundle": "states.npz",
            "bundle_sha256": "def",
        },
        "scramble_seed": 3,
        "gates": {"minimum_non_krylov_norm_fraction": 1.0e-6},
        "remote_training_on_pass": False,
    }


def test_physical_fmove_gate_config_contract():
    _validate(json.loads(json.dumps(_spec())))
    broken = _spec()
    broken["remote_training_on_pass"] = True
    with pytest.raises(ValueError, match="must not submit remote"):
        _validate(broken)
