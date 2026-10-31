"""CPU-only protocol/reference tests; actual B200 acceptance is a separate job."""
from copy import deepcopy
import json
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows import accelerator_runtime_validation as module
from su2_nqs.frontier.artifacts import canonical_sha256


def _plan() -> dict:
    case = dict(source_registration=dict(path="registration.json", sha256="a"*64),
        checkpoint=dict(path="checkpoint.pkl", sha256="b"*64),
        audit_report=dict(path="audit.json", sha256="c"*64),
        audit_products=dict(path="products.npz", sha256="d"*64),
        parameter_sha256="e"*64, optimizer_sha256="f"*64, rng_sha256="1"*64)
    return dict(task="accelerator_runtime_validation_v1", wall_cap_seconds=1800,
        warnings_visible=True, stress_repetitions=4, gemm_seed=100,
        approved_distributions={"jax":"0.11.1", "jaxlib":"0.11.1", "jax-cuda13-plugin":"0.11.1",
            "jax-cuda13-pjrt":"0.11.1", "nvidia-cublas":"13.8.2.12"},
        approved_libraries={name:"2"*64 for name in module.LIBRARY_NAMES},
        cases=[dict(deepcopy(case), tag="j0", j2=0., jvp_seed=101),
               dict(deepcopy(case), tag="j0125", j2=.125, jvp_seed=102)])


def test_exact_patch_source_seals_and_no_silent_panel_reduction():
    plan = _plan(); module.validate_plan(plan)
    for key, value in (("warnings_visible",False), ("stress_repetitions",1), ("wall_cap_seconds",3600)):
        with pytest.raises(ValueError): module.validate_plan(dict(plan, **{key:value}))
    for version in ("12.9.2.10", "13.2.0.0", "latest"):
        bad=deepcopy(plan);bad["approved_distributions"]["nvidia-cublas"]=version
        with pytest.raises(ValueError): module.validate_plan(bad)
    bad=deepcopy(plan);bad["cases"][1]["j2"]=0.
    with pytest.raises(ValueError): module.validate_plan(bad)
    bad=deepcopy(plan);bad["cases"][0]["checkpoint"]["sha256"]="wrong"
    with pytest.raises(ValueError): module.validate_plan(bad)


@pytest.mark.parametrize("environment", [
    {"JAX_SKIP_CUDA_CONSTRAINTS_CHECK":"1"}, {"TF_CPP_MIN_LOG_LEVEL":"2"},
    {"PYTHONWARNINGS":"ignore"}, {"JAX_DISABLE_JIT":"true"},
])
def test_warning_and_constraint_bypass_rejected(environment):
    with pytest.raises(ValueError): module.require_visible_checks(environment)


def test_actual_maps_unique_and_cuda12_contamination_rejected():
    maps="100-200 r-xp 0 00:00 1 /new/libcublas.so.13\n200-300 r-xp 0 00:00 2 /new/libcublasLt.so.13\n"
    assert set(module.mapped_libraries(maps))==set(module.LIBRARY_NAMES)
    with pytest.raises(ValueError): module.mapped_libraries(maps+"300-400 r-xp 0 00:00 3 /old/libcublas.so.12\n")
    with pytest.raises(ValueError): module.mapped_libraries(maps+"300-400 r-xp 0 00:00 3 /other/libcublas.so.13\n")
    with pytest.raises(ValueError): module.mapped_libraries(maps.splitlines()[0])


def test_only_official_sealed_user_space_compatibility_directory_allowed(tmp_path):
    plan=_plan()
    with pytest.raises(ValueError):module.verify_compatibility_libraries(plan,{'LD_LIBRARY_PATH':'/unapproved'})
    library=tmp_path/'libcuda.so.615.71.10';library.write_bytes(b'official-test-fixture-not-a-driver')
    plan['forward_compat']=dict(package='cuda-compat-13-4',version='615.71.10-1ubuntu1',
        directory=str(tmp_path),libraries={library.name:module.file_sha256(library)})
    module.validate_plan(plan)
    maps='100-200 r-xp 0 00:00 1 '+str(library)+'\n'
    env={'LD_LIBRARY_PATH':str(tmp_path)}
    result=module.verify_compatibility_libraries(plan,env,maps)
    assert result['kernel_driver_modified'] is False
    with pytest.raises(ValueError):module.verify_compatibility_libraries(plan,{},maps)
    with pytest.raises(ValueError):module.verify_compatibility_libraries(plan,dict(env,LD_PRELOAD='/other'),maps)
    with pytest.raises(ValueError):module.verify_compatibility_libraries(plan,env,maps.replace(str(tmp_path),'/other'))
    library.write_bytes(b'changed')
    with pytest.raises(ValueError):module.verify_compatibility_libraries(plan,env,maps)


def test_quantized_half_reference_and_complex_conjugate_transpose():
    a=np.asarray([[1.0001, .3333], [-.7501, 2.0003]],dtype=np.float16)
    b=np.asarray([[.7001], [1.2003]],dtype=np.float16)
    c=np.asarray([[.2], [.4]],dtype=np.float32)
    expected=a.astype(np.float64)@b.astype(np.float64)+c
    np.testing.assert_array_equal(module.gemm_reference(a,b,c,operation=0,beta=1),expected)
    z=a.astype(np.complex128)+1j*np.asarray([[1.,2.],[3.,4.]])
    np.testing.assert_array_equal(module.gemm_reference(z,b,c,operation=2,beta=0),z.conj().T@b.astype(np.complex128))
    assert not np.array_equal(z.T@b,z.conj().T@b)


def test_relative_error_fails_nonfinite_shape_and_zero_reference():
    assert module._relative_error(np.zeros(2),np.zeros(2))==0
    with pytest.raises(ArithmeticError):module._relative_error(np.asarray([np.nan]),np.ones(1))
    with pytest.raises(ArithmeticError):module._relative_error(np.ones(2),np.ones(1))


def test_actual_CPU_complex_real_gradient_convention_and_SPD_solve():
    result=module.jax_derivative_and_solve_checks(2610289910)
    assert result["passed"]
    assert len(result["gradients"])==4 and len(result["SPD_solve"])==2
    assert max(x["relative_residual"] for x in result["SPD_solve"])<2e-11
    assert max(x.get("CPU_FD_relative_error",0.) for x in result["gradients"])<2e-7


def test_artifact_path_escape_and_changed_bytes_rejected(tmp_path):
    source=tmp_path/"source";source.write_bytes(b"safe")
    item=dict(path="source",sha256=module.file_sha256(source))
    assert module._artifact(item,tmp_path)==source
    source.write_bytes(b"changed")
    with pytest.raises(ValueError):module._artifact(item,tmp_path)
    outside=tmp_path.parent/"outside_unread"
    with pytest.raises(ValueError):module._artifact(dict(path=str(outside),sha256="a"*64),tmp_path)


def test_CPU_stage_preserves_all_physical_fixture_roles_and_sources(monkeypatch,tmp_path):
    # Synthetic transport test, explicitly NOT a replacement for the actual
    # two144-site CT12 probes performed in the registered acceptance job.
    plan=_plan();rows=np.tile(np.r_[np.zeros(72),np.ones(72)],(2,1)).astype(np.int32)
    products=np.asarray([[1.,-2.,3.,4.],[2.,-4.,6.,8.]])
    identity=dict(parameter_sha256="e"*64,optimizer_sha256="f"*64,rng_sha256="1"*64)
    monkeypatch.setattr(module,"load_physical_case",lambda *a: ({}, {}, identity, rows, products))
    monkeypatch.setattr(module,"_case_probes",lambda *a: (dict(rows=rows,products=products,
        amplitude=products[:,0],jvp=np.ones(2)),dict(n_sites=144,complete_A1_images=1728)))
    monkeypatch.setattr(module,"_source_manifest",lambda:{"entry.py":"sourcehash"})
    monkeypatch.setattr(module,"_verify_frozen_inputs",lambda *a:None)
    result=module.cpu_prepare(plan,tmp_path,tmp_path)
    assert result["passed"] and result["new_physical_H_centers"]==4
    assert result["plan_sha256"]==canonical_sha256(plan)
    with np.load(tmp_path/"golden.npz") as f:
        assert set(f.files)=={t+"_"+k for t in ("j0","j0125") for k in ("rows","products","amplitude","jvp")}
        np.testing.assert_array_equal(f["j0125_products"],products)


def test_source_changes_during_acceptance_fail_closed(monkeypatch,tmp_path):
    monkeypatch.setattr(module,"_source_manifest",lambda:{"entry.py":"new"})
    with pytest.raises(ValueError):module._verify_frozen_inputs(_plan(),tmp_path,{"entry.py":"old"})
    with pytest.raises(TimeoutError):module._check_deadline(module.time.monotonic()-1810,1800)
