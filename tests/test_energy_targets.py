from __future__ import annotations

import pytest

from su2_nqs.frontier.energy_targets import observed_energy_to_target


def test_observed_energy_to_target_records_first_checkpoint_and_time_scope() -> None:
    rows = observed_energy_to_target(
        {
            "compile_seconds": 2.0,
            "training_seconds": 8.0,
            "history": [
                {"step": 10.0, "loss_before_update": -8.0},
                {"step": 20.0, "loss_before_update": -9.2},
            ],
        },
        final_energy=-9.8,
        exact_energy=-10.0,
        n_sites=10,
        total_steps=40,
        targets_per_site=(0.1, 0.03, 0.01),
    )
    assert rows[0]["first_observed_step"] == 19
    assert rows[0]["first_observed_completed_updates"] == 19
    assert rows[1]["first_observed_step"] == 40
    assert rows[1]["training_seconds_to_checkpoint"] == pytest.approx(8.0)
    assert rows[1]["cold_seconds_to_checkpoint"] == pytest.approx(10.0)
    assert rows[2]["reached"] is False


def test_initial_state_is_update_zero_and_final_state_is_total_steps() -> None:
    rows = observed_energy_to_target(
        {
            "initial_loss": -9.5,
            "compile_seconds": 1.0,
            "training_seconds": 4.0,
            "history": [{"step": 1, "loss_before_update": -9.5}],
        },
        final_energy=-9.9,
        exact_energy=-10.0,
        n_sites=10,
        total_steps=1,
        targets_per_site=(0.1, 0.02),
    )
    assert rows[0]["first_observed_completed_updates"] == 0
    assert rows[0]["training_seconds_to_checkpoint"] == 0.0
    assert rows[1]["first_observed_completed_updates"] == 1


def test_observed_energy_to_target_rejects_invalid_trajectory() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        observed_energy_to_target(
            {
                "history": [
                    {"step": 2, "loss_before_update": 0.0},
                    {"step": 2, "loss_before_update": 0.0},
                ]
            },
            final_energy=0.0,
            exact_energy=-1.0,
            n_sites=2,
            total_steps=2,
            targets_per_site=(1.0,),
        )
