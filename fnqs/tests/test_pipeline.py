import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fnqs.configuration import validate_config
from fnqs.controls import control_configs
from fnqs.entropy import summarize_swap, swap_ratios
from fnqs.hamiltonians import IsingCouplings, all_bits, connectivity, ground_state
from fnqs.observables import renyi2_from_state
from fnqs.scaling import collapse_fit


def test_configs_validate_and_leakage_or_unknown_keys_fail():
    directory=Path(__file__).parents[1]/"configs"
    for path in directory.glob("*.json"):
        command="reference" if "reference" in path.name else "recon" if "recon" in path.name else "train"
        validate_config(command,json.loads(path.read_text()))
    config=json.loads((directory/"phase0_smoke.json").read_text())
    config["test_fields"]=[1.0]
    with pytest.raises(ValueError):validate_config("train",config)
    config["test_fields"]=[1.1];config["misspelled_budget"]=10
    with pytest.raises(ValueError):validate_config("train",config)


def test_control_protocol_has_matched_queries_and_disjoint_confirmation():
    configs=control_configs()
    confirmation={round(.805+.01*i,8) for i in range(40)}
    for config in configs.values():
        validate_config("train",config)
        assert config["tasks_per_step"]==5 and config["steps"]==1200
        assert not confirmation & set(config["train_fields"]+config["test_fields"])
    assert len(configs)==6
    config=dict(configs["fixed_dense_sr"],sr_backtracking=True,mode="vmc")
    with pytest.raises(ValueError):validate_config("train",config)
    config=dict(configs["fixed_dense_adam"],tasks_per_step=22)
    with pytest.raises(ValueError):validate_config("train",config)


def test_swap_estimator_against_exact_schmidt_purity():
    n=4
    _,psi,_=ground_state(n,IsingCouplings(h=.9))
    basis=all_bits(n);labels=np.arange(1<<n)
    first=jnp.asarray(np.repeat(basis,len(basis),axis=0));second=jnp.asarray(np.tile(basis,(len(basis),1)))
    weights=np.repeat(psi**2,len(psi))*np.tile(psi**2,len(psi))
    def logs(bits):
        integer=bits @ jnp.asarray(1 << np.arange(n-1,-1,-1))
        return jnp.log(jnp.asarray(psi)[integer])
    for ell in [1,2,3]:
        ratios=np.asarray(swap_ratios(logs,first,second,ell))
        np.testing.assert_allclose(-np.log(weights@ratios),renyi2_from_state(psi,n,ell),atol=2e-13)
        result=summarize_swap(ratios)
        assert np.isfinite(result["renyi2"]) and result["replica_pairs"]==256


def test_collapse_exponent_is_not_hardcoded_to_ising():
    sizes=np.repeat([16,24,32,48,64],31)
    fields=np.concatenate([np.linspace(1.015-.5/n**(1/.9),1.015+.5/n**(1/.9),31) for n in [16,24,32,48,64]])
    x=(fields-1.015)*sizes**(1/.9)
    chi=sizes**(2/.9)*(0.1-.02*x*x+.003*x**4+(0.03+.02*x)/sizes)
    fit=collapse_fit(sizes,fields,chi,degree=4)
    np.testing.assert_allclose(fit["hc"],1.015,atol=2e-6)
    np.testing.assert_allclose(fit["nu"],.9,atol=2e-5)


def test_connectivity_retains_msb_and_signed_spin_values_at_n64():
    bits=np.zeros(64,dtype=np.uint8)
    new,values=connectivity(bits,IsingCouplings())
    assert len(new)==65
    assert values[0]==-64
    assert np.any(np.all(new==np.eye(64,dtype=int)[0],axis=1))
