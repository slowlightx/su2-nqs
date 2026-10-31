"""Independent dense-H checks of a fixed-direction calibration rule."""
import numpy as np
import pytest

from su2_nqs.vmc.update_calibration import independent_update_calibration
from su2_nqs.vmc.carrier_protection import physical_moments


def _inputs():
    # Each chain has a complete two-level state; no random favorable outcome.
    p = np.tile([np.sqrt(.8), np.sqrt(.2)], 8).astype(complex)
    hp = np.tile([0., np.sqrt(.2)], 8).astype(complex)
    d = -(hp-.2*p)
    hd = np.tile([0., -.8*np.sqrt(.2)], 8).astype(complex)
    return p, hp, d, hd, np.ones(16), np.repeat(np.arange(8), 2)


def test_dense_H_both_moments_direction_units_and_gauge():
    p, hp, d, hd, q, ids = _inputs()
    result = independent_update_calibration(p, hp, d, hd, q, ids, 2, maximum_alpha=2.)
    assert result['eligible'] and 0 < result['maximum_alpha'] <= 2
    for row in result['regions'].values():
        assert row['energy_slope_per_site'] == pytest.approx(-.16)
        assert row['variance_slope_per_site'] == pytest.approx(-.096)
    eps = 1e-6
    plus = physical_moments(p+eps*d, hp+eps*hd, q, 2)
    minus = physical_moments(p-eps*d, hp-eps*hd, q, 2)
    for key, slope in [('energy_per_site', 'energy_slope_per_site'), ('variance_per_site', 'variance_slope_per_site')]:
        observed = (plus[key]-minus[key])/(2*eps)
        assert observed == pytest.approx(result['regions']['full'][slope], abs=2e-10)
    shifted = independent_update_calibration(p, hp+4*p, 3*d+(2+.7j)*p,
        3*(hd+4*d)+(2+.7j)*(hp+4*p), 7*q, ids, 2, maximum_alpha=2/3)
    assert shifted['eligible']
    assert shifted['maximum_alpha']*3 == pytest.approx(result['maximum_alpha'])
    assert shifted['regions']['full']['residual_response_per_site'] == pytest.approx(9*result['regions']['full']['residual_response_per_site'])


def test_fold_reversal_or_null_cannot_be_repaired_by_radius():
    p, hp, d, hd, q, ids = _inputs()
    d[ids % 2 == 1] *= -1
    hd[ids % 2 == 1] *= -1
    result = independent_update_calibration(p, hp, d, hd, q, ids, 2, maximum_alpha=.1)
    assert not result['eligible'] and result['maximum_alpha'] == 0
    assert not result['checks']['energy_descent_all_folds']
    null = independent_update_calibration(p, hp, 1j*p, 1j*hp, q, ids, 2, maximum_alpha=.1)
    assert not null['eligible'] and null['maximum_alpha'] == 0


def test_incomplete_quadrature_keeps_complex_energy_nodes_and_Hdelta():
    rng = np.random.default_rng(171)
    a = rng.normal(size=(30,30))+1j*rng.normal(size=(30,30))
    h = a+a.conj().T
    p = rng.normal(size=30)+1j*rng.normal(size=30); p[0] = 0
    d = rng.normal(size=30)+1j*rng.normal(size=30)
    keep = np.arange(16); q = np.linspace(.2,1.8,16)
    hp, hd = (h@p)[keep], (h@d)[keep]; p, d = p[keep], d[keep]
    result = independent_update_calibration(p, hp, d, hd, q, np.repeat(np.arange(8),2), 6, maximum_alpha=.1)
    eps = 2e-6
    plus = physical_moments(p+eps*d,hp+eps*hd,q,6)
    minus = physical_moments(p-eps*d,hp-eps*hd,q,6)
    row = result['regions']['full']
    assert row['exact_parent_node_records'] == 1
    assert row['hermiticity_quadrature_defect_per_site'] > 1e-3
    for key,slope in [('energy_per_site','energy_slope_per_site'),('variance_per_site','variance_slope_per_site')]:
        assert (plus[key]-minus[key])/(2*eps) == pytest.approx(row[slope],abs=2e-8)


def test_invalid_calibration_contracts_fail_closed():
    p,hp,d,hd,q,ids = _inputs()
    for opts in ({'maximum_alpha':0}, {'maximum_alpha':True}, {'response_safety':.6}):
        with pytest.raises(ValueError):
            independent_update_calibration(p,hp,d,hd,q,ids,2,**({'maximum_alpha':.1}|opts))
    with pytest.raises(ValueError):independent_update_calibration(p,hp,d,hd,q,np.zeros(16,int),2,maximum_alpha=.1)
    with pytest.raises(ValueError):independent_update_calibration(p,hp,d,hd,np.zeros(16),ids,2,maximum_alpha=.1)
