"""Microscopic family splits and evidence-based research-stage readiness.

A/B/C label operator content, not a learned universality-class target.
Planning a held-family protocol does not authorize its physical conclusions.
"""

from __future__ import annotations

from .training import Task


def candidate_family(task: Task) -> str:
    """Identify the declared A/B/C cuts from physical couplings, not metadata."""
    g = task.couplings
    if g.j1 != 1:
        raise ValueError("current candidates require j1=1")
    if g.j2 == 0 and g.jx == 0:
        return "A"
    if g.j2 > 0 and g.jx == 0:
        return "B"
    if g.jx > 0 and g.j2 == 0:
        return "C"
    raise ValueError("operator content is outside the declared candidate families")


def validate_family_holdout(pretraining: list[Task], model_selection: list[Task],
                            held_family: str) -> None:
    """Reject whole-family leakage, including mislabeled coupling vectors."""
    if held_family not in {"A", "B", "C"} or not pretraining:
        raise ValueError("a nonempty pretraining split and A/B/C holdout are required")
    for role, tasks in (("pretraining", pretraining), ("model_selection", model_selection)):
        for task in tasks:
            physical = candidate_family(task)
            if physical != task.family:
                raise ValueError(f"{role} family metadata contradicts coupling operator content")
            if physical == held_family:
                raise ValueError(f"held family leaked into {role}")


def assess_stages(phase0: dict, phase1: dict | None = None) -> dict:
    """Read complete gate evidence; warm replicas do not add cold initializations.

    The Phase 0 gate is produced by the observable/exponent audit. This report
    further requires three independently initialized optimization runs before
    enabling Phase 1 shared training. Phase 2 needs a separate completed
    Phase 1 reproduction gate; an ED reconnaissance report cannot supply it.
    """
    gate = phase0.get("phase0_scientific_gate", {})
    arms = phase0.get("arms", [])
    cold_seeds = {a["seed"] for a in arms if a.get("warm_start") is None}
    phase0_ready = (phase0.get("kind") == "shared_neural_phase0" and
        gate.get("passed") is True and bool(gate.get("checks")) and
        all(value is True for value in gate["checks"].values()) and
        len(cold_seeds) >= 3 and all(a.get("observable_checks_passed") is True for a in arms))
    phase1 = phase1 or {}
    gate1 = phase1.get("phase1_scientific_gate", {})
    phase1_ready = (phase0_ready and phase1.get("kind") == "shared_neural_phase1" and
                   gate1.get("passed") is True and bool(gate1.get("checks")) and
                   all(value is True for value in gate1["checks"].values()))
    return {"phase0_cold_initialized_reproduction_passed": phase0_ready,
            "phase1_reference_development_allowed": True,
            "phase1_shared_training_allowed": phase0_ready,
            "phase1_universality_claim_allowed": phase1_ready,
            "phase2_protocol_development_allowed": True,
            "phase2_transfer_execution_allowed": phase1_ready,
            "note": "ED brackets are finite-size references; they do not establish a class or transfer advantage."}
