"""Independent noisy scaling law; run explicitly while v6 stays frozen."""

import numpy as np
import pytest

from fnqs.scaling import collapse_fit


def _observations(seed):
    n=np.repeat([16,24,32,48,64],41).astype(float)
    x=np.tile(np.linspace(-1,1,41),5)
    h=1.07+x/n**(1/.9)
    mean=.03*n**(2/.9)*(1-.25*x*x+.05*x**4+(.4-.1*x*x)/n)
    sigma=.01*mean
    return n,h,mean+np.random.default_rng(seed).normal(size=len(n))*sigma,sigma


@pytest.mark.parametrize("seed",[17,29,43])
def test_noisy_free_exponent_with_fixed_physical_errors(seed):
    n,h,chi,sigma=_observations(seed)
    fit=collapse_fit(n,h,chi,degree=4,chi_se=sigma)
    assert abs(fit["nu"]-.9)<.005
    assert abs(fit["hc"]-1.07)<.0003
    assert not fit["at_bound"]
    assert fit["objective_normalization"]=="fixed_physical_standard_errors"


def test_invalid_error_vector_is_rejected():
    n,h,chi,sigma=_observations(17)
    for errors in [np.zeros_like(sigma),sigma[:-1],sigma*np.nan]:
        with pytest.raises(ValueError,match="chi SE"):
            collapse_fit(n,h,chi,degree=4,chi_se=errors)
