import copy

import pytest

from fnqs.hamiltonians import IsingCouplings
from fnqs.staging import assess_stages, validate_family_holdout
from fnqs.training import Task


def test_holdout_checks_physical_couplings_and_model_selection_leakage():
    a = Task(16, IsingCouplings(), "A")
    b = Task(16, IsingCouplings(j2=.2), "B")
    c = Task(16, IsingCouplings(jx=.1), "C")
    validate_family_holdout([a, b], [a, b], "C")
    with pytest.raises(ValueError): validate_family_holdout([a, c], [], "C")
    with pytest.raises(ValueError): validate_family_holdout([a, b], [c], "C")
    disguised = Task(16, IsingCouplings(jx=.1), "A")
    with pytest.raises(ValueError): validate_family_holdout([a, disguised], [], "C")


def test_stages_reject_warm_replicas_or_ed_tables_as_reproduction():
    phase0 = {"kind":"shared_neural_phase0", "phase0_scientific_gate":
              {"passed":True, "checks":{"hc":True,"nu":True,"eta":True,"observables":True}},
              "arms":[{"seed":seed,"warm_start":None,"observable_checks_passed":True}
                      for seed in [17, 29, 43]]}
    assert assess_stages(phase0)["phase1_shared_training_allowed"]
    warm = copy.deepcopy(phase0)
    for arm in warm["arms"]: arm["warm_start"] = {"parent_seed":17}
    assert not assess_stages(warm)["phase1_shared_training_allowed"]
    inconsistent = copy.deepcopy(phase0)
    inconsistent["phase0_scientific_gate"]["checks"]["nu"] = False
    assert not assess_stages(inconsistent)["phase1_shared_training_allowed"]
    assert not assess_stages(phase0, {"kind":"phase1_ed_reconnaissance"})["phase2_transfer_execution_allowed"]
    assert assess_stages(phase0)["phase2_protocol_development_allowed"]
