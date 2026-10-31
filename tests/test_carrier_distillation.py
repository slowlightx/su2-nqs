import numpy as np
import pytest

from su2_nqs.vmc.carrier_distillation import normalized_carrier_audit


def test_normalized_carrier_audit_counts_probability_outside_teacher_sector():
    target = np.asarray([1.0, 0.0], dtype=np.complex128)
    student = np.asarray([0.5, 0.5], dtype=np.complex128)
    audit = normalized_carrier_audit(student, target, native_norm=1.0)
    assert audit["fidelity"] == pytest.approx(0.25)
    assert audit["teacher_sector_weight"] == pytest.approx(0.5)


def test_normalized_carrier_audit_rejects_nonpositive_native_norm():
    with pytest.raises(ValueError, match="positive"):
        normalized_carrier_audit(np.ones(2), np.ones(2), native_norm=0.0)
