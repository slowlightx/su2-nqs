"""Sealed CPU-to-B200 numerical acceptance; never a production training run.

The physical probes are actual144-site CT12 states from independently audited
A100 checkpoints, with complete1728-image coherent A1 and every nonzero H
bond. A passed finite test is not proof of absence of all accelerator bugs.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import importlib.metadata as im
import json
import logging
import os
from pathlib import Path
import re
import time
import traceback
from typing import Any, Mapping

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from . import triangle_active_two_step_pilot as c12

ROOT = Path(__file__).resolve().parents[3]
CT_TOLERANCE = 2e-11
LIBRARY_NAMES = ("libcublas.so.13", "libcublasLt.so.13")


def validate_plan(plan: dict) -> None:
    """Require two sealed physical CT12 sources and one approved exact stack."""
    if (plan.get("task") != "accelerator_runtime_validation_v1"
            or plan.get("warnings_visible") is not True
            or plan.get("wall_cap_seconds") != 1800
            or plan.get("stress_repetitions") != 4):
        raise ValueError("fixed30-minute acceptance protocol, visible warnings and four stress rounds required")
    approved = plan.get("approved_distributions", {})
    for name in ("jax", "jaxlib", "jax-cuda13-plugin", "jax-cuda13-pjrt"):
        if approved.get(name) != "0.11.1":
            raise ValueError("explicit matched JAX0.11.1 CUDA13 distributions required")
    version = approved.get("nvidia-cublas", "")
    if not re.fullmatch(r"\d+\.\d+\.\d+\.\d+", version) or tuple(map(int, version.split("."))) < (13, 2, 2, 0):
        raise ValueError("an explicitly approved cuBLAS patch at least13.2.2 is required")
    if set(plan.get("approved_libraries", {})) != set(LIBRARY_NAMES):
        raise ValueError("approve exact hashes of both cuBLAS13 and cuBLASLt13 libraries")
    for digest in plan["approved_libraries"].values():
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("exact native-library SHA256 required")
    compat=plan.get('forward_compat')
    if compat is not None:
        if (compat.get('package')!='cuda-compat-13-4'
                or not isinstance(compat.get('directory'),str)
                or not Path(compat['directory']).is_absolute()
                or not any(name.startswith('libcuda.so.') for name in compat.get('libraries',{}))
                or any(not re.fullmatch(r'[0-9a-f]{64}',value) for value in compat['libraries'].values())):
            raise ValueError('exact official user-space forward-compat library seals required')
    cases = plan.get("cases", [])
    if len(cases) != 2 or {row.get("j2") for row in cases} != {0., .125} or len({row.get("tag") for row in cases}) != 2:
        raise ValueError("exactly two distinct J0/J.125 CT12 cases required")
    for case in cases:
        if isinstance(case["j2"], bool) or case.get("tag") not in ("j0", "j0125"):
            raise ValueError("registered physical case identity required")
        if (case["tag"] == "j0") != (case["j2"] == 0):
            raise ValueError("tag and actual Hamiltonian coupling differ")
        for key in ("source_registration", "checkpoint", "audit_report", "audit_products"):
            item = case.get(key, {})
            if not isinstance(item.get("path"), str) or not item["path"] or not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", "")):
                raise ValueError("every physical input needs a path and SHA256: " + key)
        for key in ("parameter_sha256", "optimizer_sha256", "rng_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", case.get(key, "")):
                raise ValueError("saved parameter/optimizer/RNG identity required")
    seeds = [plan.get("gemm_seed"), *(row.get("jvp_seed") for row in cases)]
    if any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**32 for s in seeds) or len(set(seeds)) != 3:
        raise ValueError("three distinct deterministic uint32 probe seeds required")


def require_visible_checks(environment: Mapping[str, str]) -> None:
    """Reject warning/check bypasses rather than muting or changing them."""
    if environment.get("JAX_SKIP_CUDA_CONSTRAINTS_CHECK"):
        raise ValueError("CUDA constraints checks must not be bypassed")
    if environment.get("TF_CPP_MIN_LOG_LEVEL", "0") not in ("", "0"):
        raise ValueError("native warnings must remain visible")
    if "ignore" in environment.get("PYTHONWARNINGS", "").lower():
        raise ValueError("Python warning suppression is forbidden")
    if environment.get("JAX_DISABLE_JIT", "").lower() in ("1", "true", "yes"):
        raise ValueError("acceptance must exercise the real compiled kernels")
    for name in ("", "jax", "jax_plugins.cuda"):
        logger = logging.getLogger(name)
        if logger.disabled or logger.getEffectiveLevel() > logging.WARNING:
            raise ValueError("warning logger disabled: " + name)


def mapped_libraries(maps: str) -> dict[str, str]:
    """Extract unique actual Linux mappings, rejecting stale CUDA12 libraries."""
    paths = {line.split()[-1] for line in maps.splitlines()
        if line.split() and line.split()[-1].startswith("/")}
    if any(re.search(r"/libcublas(?:Lt)?\.so\.12(?:\.|$)", path) for path in paths):
        raise ValueError("a CUDA12 cuBLAS library is actually mapped")
    result = {}
    for name in LIBRARY_NAMES:
        found = {str(Path(path).resolve()) for path in paths
            if Path(path).name == name or Path(path).name.startswith(name + ".")}
        if len(found) != 1:
            raise ValueError("exactly one actually mapped " + name + " required")
        result[name] = found.pop()
    return result


def verify_compatibility_libraries(plan:dict,environment:Mapping[str,str],maps:str|None=None) -> dict|None:
    """Permit only the presealed official driver compatibility directory.

    CUDA13's userspace API can use NVIDIA's supported forward-compat path on
    datacenter GPUs with R570 kernels; this does not alter the kernel driver.
    Never accept a generic LD_LIBRARY_PATH or an injected BLAS replacement.
    """
    if environment.get('LD_PRELOAD'):raise ValueError('unapproved preloaded libraries')
    compat=plan.get('forward_compat')
    expected=compat['directory'] if compat else ''
    if environment.get('LD_LIBRARY_PATH','')!=expected:
        raise ValueError('library search path differs from the registered compatibility directory')
    if compat is None or maps is None:return None
    root=Path(expected).resolve();loaded={}
    paths={line.split()[-1] for line in maps.splitlines() if line.split() and line.split()[-1].startswith('/')}
    for filename in paths:
        path=Path(filename).resolve();name=path.name
        if not name.startswith(('libcuda.so.','libnvidia-gpucomp.so.','libnvidia-nvvm',
                                'libnvidia-ptxjitcompiler.so.','libnvidia-tileiras.so.')):continue
        if path.parent!=root or name not in compat['libraries'] or file_sha256(path)!=compat['libraries'][name]:
            raise ValueError('actual driver/JIT library differs from sealed compatibility package: '+str(path))
        loaded[name]=dict(path=str(path),sha256=compat['libraries'][name])
    if not any(name.startswith('libcuda.so.') for name in loaded):
        raise ValueError('actual sealed forward-compatible libcuda is not mapped')
    return dict(package=compat['package'],version=compat['version'],loaded_libraries=loaded,
                kernel_driver_modified=False)


def _native_call(function: Any, *args: Any) -> None:
    status = int(function(*args))
    if status != 0:
        raise RuntimeError(f"native call {getattr(function, '__name__', 'unknown')} failed: {status}")


def _function(library: Any, name: str, arguments: list[Any], restype: Any = ct.c_int) -> Any:
    fn = getattr(library, name)
    fn.argtypes, fn.restype = arguments, restype
    return fn


def loaded_cuda_stack(plan: dict) -> tuple[dict, Any, Any]:
    """Inspect libraries JAX already loaded, never load a replacement to pass."""
    require_visible_checks(os.environ)
    verify_compatibility_libraries(plan,os.environ)
    installed = {}
    for name, expected in plan["approved_distributions"].items():
        installed[name] = im.version(name)
        if installed[name] != expected:
            raise ValueError("installed distribution differs from approved patch: " + name)
    for name in ("jax-cuda12-plugin", "jax-cuda12-pjrt"):
        try:
            version = im.version(name)
        except im.PackageNotFoundError:
            continue
        raise ValueError("CUDA12 plugin/PJRT still installed: " + name + "=" + version)
    if jax.default_backend() != "gpu" or not all("B200" in d.device_kind for d in jax.devices()):
        raise ValueError("a real allocated B200 is required")
    if not jax.config.x64_enabled:
        raise ValueError("x64 must remain enabled")
    # Force a real JAX GEMM before reading /proc/self/maps.
    jax.block_until_ready(jax.jit(lambda a: a @ a)(jnp.eye(129, dtype=jnp.float64)))
    raw_maps = Path("/proc/self/maps").read_text()
    compatibility=verify_compatibility_libraries(plan,os.environ,raw_maps)
    mapped = mapped_libraries(raw_maps)
    distribution = im.distribution("nvidia-cublas")
    owned = {str(Path(distribution.locate_file(p)).resolve()) for p in distribution.files or ()}
    identities = {}
    for name, path in mapped.items():
        if path not in owned or file_sha256(Path(path)) != plan["approved_libraries"][name]:
            raise ValueError("mapped native library not owned by the approved wheel/hash: " + path)
        identities[name] = dict(path=path, sha256=file_sha256(Path(path)), distribution="nvidia-cublas",
            patch=installed["nvidia-cublas"])
    if not hasattr(os, "RTLD_NOLOAD"):
        raise RuntimeError("RTLD_NOLOAD is required to inspect, not replace, native libraries")
    mode = os.RTLD_NOLOAD | os.RTLD_NOW
    blas = ct.CDLL(mapped["libcublas.so.13"], mode=mode)
    lt = ct.CDLL(mapped["libcublasLt.so.13"], mode=mode)
    create = _function(blas, "cublasCreate_v2", [ct.POINTER(ct.c_void_p)])
    destroy = _function(blas, "cublasDestroy_v2", [ct.c_void_p])
    get_version = _function(blas, "cublasGetVersion_v2", [ct.c_void_p, ct.POINTER(ct.c_int)])
    handle, version = ct.c_void_p(), ct.c_int()
    _native_call(create, ct.byref(handle))
    try:
        _native_call(get_version, handle, ct.byref(version))
    finally:
        _native_call(destroy, handle)
    lt_version = int(_function(lt, "cublasLtGetVersion", [], ct.c_size_t)())
    if min(version.value, lt_version) < 130200:
        raise ValueError("actual cuBLAS/cuBLASLt version remains below13.2")
    from jax_cuda13_plugin import _versions
    plugin_version = int(_versions.cublas_get_version())
    if plugin_version != version.value:
        raise ValueError("JAX CUDA13 plugin and mapped cuBLAS version disagree")
    expected_version = tuple(map(int, installed["nvidia-cublas"].split(".")))
    if version.value != 10000*expected_version[0]+100*expected_version[1]+expected_version[2]:
        raise ValueError("mapped cuBLAS API version differs from the approved precise wheel patch")
    runtime_paths = {line.split()[-1] for line in raw_maps.splitlines()
        if line.split() and re.search(r"/libcudart\.so\.13(?:\.|$)", line.split()[-1])}
    runtime_paths = {str(Path(p).resolve()) for p in runtime_paths}
    if len(runtime_paths) != 1:
        raise ValueError("exactly one actually mapped CUDA13 runtime required")
    runtime_path = runtime_paths.pop()
    cuda = ct.CDLL(runtime_path, mode=mode)
    runtime_api, driver_api, device = ct.c_int(), ct.c_int(), ct.c_int()
    _native_call(_function(cuda, "cudaRuntimeGetVersion", [ct.POINTER(ct.c_int)]), ct.byref(runtime_api))
    _native_call(_function(cuda, "cudaDriverGetVersion", [ct.POINTER(ct.c_int)]), ct.byref(driver_api))
    _native_call(_function(cuda, "cudaGetDevice", [ct.POINTER(ct.c_int)]), ct.byref(device))
    major, minor = ct.c_int(), ct.c_int()
    attribute = _function(cuda, "cudaDeviceGetAttribute", [ct.POINTER(ct.c_int), ct.c_int, ct.c_int])
    _native_call(attribute, ct.byref(major), 75, device.value)
    _native_call(attribute, ct.byref(minor), 76, device.value)
    free_bytes, total_bytes = ct.c_size_t(), ct.c_size_t()
    _native_call(_function(cuda, "cudaMemGetInfo", [ct.POINTER(ct.c_size_t), ct.POINTER(ct.c_size_t)]),
        ct.byref(free_bytes), ct.byref(total_bytes))
    if min(runtime_api.value, driver_api.value) < 13000 or major.value != 10:
        raise ValueError("actual CUDA13 driver/runtime and SM100 B200 required")
    report = dict(distributions=installed, mapped_libraries=identities,forward_compat=compatibility,
        cublas_version=version.value, cublasLt_version=lt_version,
        cuda13_plugin_reported_cublas_version=plugin_version, cuda_runtime_path=runtime_path,
        cuda_runtime_API_version=runtime_api.value, cuda_driver_API_version=driver_api.value,
        allocated_device_index=device.value, compute_capability=[major.value, minor.value],
        device_memory_free_bytes=free_bytes.value, device_memory_total_bytes=total_bytes.value,
        cuda12_plugin_present=False, warnings_visible=True,
        environment={k: os.environ.get(k) for k in ("XLA_FLAGS", "LD_LIBRARY_PATH", "JAX_PLATFORMS",
            "JAX_SKIP_CUDA_CONSTRAINTS_CHECK", "TF_CPP_MIN_LOG_LEVEL", "NVIDIA_TF32_OVERRIDE",
            "CUBLAS_EMULATE_DOUBLE_PRECISION", "CUBLAS_EMULATE_SINGLE_PRECISION")},
        finite_test_does_not_prove_absence_of_all_bugs=True)
    return report, blas, cuda


def _artifact(item: dict, input_root: Path) -> Path:
    path = (input_root/item["path"]).resolve()
    if not path.is_relative_to(input_root.resolve()) or file_sha256(path) != item["sha256"]:
        raise ValueError("physical source path/hash mismatch: " + str(path))
    return path


def _state_identity(saved: dict) -> dict:
    return dict(parameter_sha256=pytree_sha256(saved["params"]),
        optimizer_sha256=pytree_sha256(saved["optimizer_state"]),
        rng_sha256=canonical_sha256(saved["metadata"]["rng"]))


def load_physical_case(case: dict, input_root: Path) -> tuple[dict, dict, dict, np.ndarray, np.ndarray]:
    """Bind doubled-q profile and full144-site state to its safe A100 origin."""
    paths = {key: _artifact(case[key], input_root) for key in
        ("source_registration", "checkpoint", "audit_report", "audit_products")}
    registration = json.loads(paths["source_registration"].read_text())
    plan = registration["plan"]
    if plan.get("j2") != case["j2"] or plan.get("model") != dict(length=12, multiplicity_cap=20, maximum_q=10):
        raise ValueError("actual144-site Hamiltonian/model differs from source registration")
    devices = registration.get("runtime", {}).get("devices", [])
    if not devices or not all("A100" in str(device) for device in devices):
        raise ValueError("numerical acceptance must start from the safe A100, not B200 trained state")
    audit = json.loads(paths["audit_report"].read_text())
    if (audit.get("status") != "complete_fixed_prefix_independent_audit"
            or not audit.get("independent_from_training_and_geometry")
            or audit.get("images") != 1728
            or audit.get("physical_bonds") != len(c12.registered_bonds(case["j2"]))):
        raise ValueError("complete independent A100 physical audit required")
    saved = load_checkpoint(paths["checkpoint"])
    identity = _state_identity(saved)
    if any(identity[key] != case[key] or identity[key] != saved["metadata"].get(key) for key in identity):
        raise ValueError("restored native parameter/optimizer/RNG identity mismatch")
    if saved["metadata"].get("completed_updates") != 2 or int(np.asarray(saved["optimizer_state"][0])) != 2:
        raise ValueError("the preselected genuine two-update A100 source is required")
    if any(np.asarray(x).dtype != np.dtype("float64") for x in jax.tree.leaves(saved["params"])):
        raise ValueError("original float64 active parameters must not change dtype")
    with np.load(paths["audit_products"]) as source:
        rows, products = source["rows"][:2].copy(), source["products"][:2, 1, :].copy()
    if rows.shape != (2, 144) or products.shape != (2, 4) or not np.all((rows == 0)|(rows == 1)) or not np.all(rows.sum(1) == 72):
        raise ValueError("two actual physical fixed-Sz=0 source rows required")
    if not np.isfinite(products).all():
        raise ValueError("finite A100 golden four-products required")
    return saved, registration, identity, rows, products


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ArithmeticError("nonfinite numeric result or changed shape")
    return float(np.max(np.abs(actual-expected))/max(float(np.max(np.abs(expected))), 1e-300))


def _case_probes(case: dict, saved: dict, registration: dict, rows: np.ndarray) -> tuple[dict, dict]:
    plan = registration["plan"]
    runtime = c12.runtime(plan, saved["params"])
    params = saved["params"]
    bank, normalization = c12.canonical_active_sampling_bank(params, runtime["cfg"], runtime["profile"])
    products = np.stack([np.asarray(v) for v in jax.block_until_ready(runtime["H"](bank, rows))], axis=1)
    rng = np.random.default_rng(case["jvp_seed"])
    count = sum(np.size(x) for x in jax.tree.leaves(params))
    direction = jax.tree.map(lambda x: jnp.asarray(rng.normal(size=x.shape)/np.sqrt(count)), params)
    # Fixed parent raw norm only: do NOT differentiate through a NumPy QR or
    # confuse original native coordinates with the canonical sampler gauge.
    scale = np.sqrt(normalization["original_raw_norm"])
    amplitude = lambda p: runtime["amplitude"](runtime["prepare"](p), jnp.asarray(rows))/scale
    phi, jvp = jax.block_until_ready(jax.jit(lambda p, d: jax.jvp(amplitude, (p,), (d,)))(params, direction))
    if _relative_error(np.asarray(phi), products[:, 0]) > CT_TOLERANCE:
        raise ArithmeticError("canonical physical products and fixed-native-scale amplitude differ")
    if not np.isfinite(np.asarray(jvp)).all() or not np.any(np.asarray(jvp) != 0):
        raise ArithmeticError("all-parameter physical JVP must be finite and nonzero")
    return dict(rows=rows, products=products, amplitude=np.asarray(phi), jvp=np.asarray(jvp)), dict(
        n_sites=144, complete_A1_images=len(runtime["images"]), physical_H_bonds=len(runtime["bonds"]),
        active_real_parameters=count, active_profile=runtime["profile"], normalization=normalization,
        jvp_definition="all native active coordinates, complete A1 amplitude divided by fixed sqrt(parent raw norm); no derivative of QR",
        jvp_seed=case["jvp_seed"], no_new_samples=True, no_training=True)


def _check_deadline(start: float, seconds: float) -> None:
    if time.monotonic()-start >= seconds:
        raise TimeoutError("acceptance deadline reached; incomplete is not passed")


def _source_manifest() -> dict[str, str]:
    return transitive_local_source_manifest([Path(__file__)], root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"))


def _verify_frozen_inputs(plan: dict, input_root: Path, sources: dict[str, str]) -> None:
    if _source_manifest() != sources:
        raise ValueError("project source changed during numerical acceptance")
    for case in plan["cases"]:
        for key in ("source_registration", "checkpoint", "audit_report", "audit_products"):
            _artifact(case[key], input_root)


def cpu_prepare(plan: dict, input_root: Path, output: Path) -> dict:
    """Create deterministic full-CT12 CPU probes from sealed A100 sources."""
    if jax.default_backend() != "cpu":
        raise ValueError("CPU golden preparation must use the actual CPU backend")
    start = time.monotonic()
    sources = _source_manifest()
    arrays, reports = {}, {}
    for case in plan["cases"]:
        _check_deadline(start, plan["wall_cap_seconds"])
        saved, reg, identity, rows, a100 = load_physical_case(case, input_root)
        probe, details = _case_probes(case, saved, reg, rows)
        error = np.max(np.abs(probe["products"]-a100), axis=0)/np.maximum(np.max(np.abs(a100), axis=0), 1e-300)
        if not np.isfinite(error).all() or error.max() > CT_TOLERANCE:
            raise ArithmeticError("safe A100 to CPU complete-H physical replay failed")
        arrays.update({case["tag"]+"_"+key: value for key, value in probe.items()})
        reports[case["tag"]] = dict(identity=identity, physical=details,
            A100_four_product_relative_errors=error.tolist(), source_row_indices=[0, 1])
        atomic_json(output/(case["tag"]+"_CPU_stage.json"), reports[case["tag"]])
        _check_deadline(start, plan["wall_cap_seconds"])
    _verify_frozen_inputs(plan, input_root, sources)
    golden = output/"golden.npz"
    np.savez_compressed(golden, **arrays)
    result = dict(status="CPU_golden_prepared", passed=True, plan_sha256=canonical_sha256(plan),
        golden_sha256=file_sha256(golden), cases=reports, seconds=time.monotonic()-start,
        source_manifest=sources,
        no_production_training=True, new_physical_H_centers=4, new_samples=0)
    atomic_json(output/"CPU_registration.json", result)
    return result


def gpu_check(plan: dict, input_root: Path, output: Path, golden_directory: Path) -> dict:
    """Run approved-library, numerical stress and real-state B200 replay gates."""
    start = time.monotonic()
    cpu = json.loads((golden_directory/"CPU_registration.json").read_text())
    source_manifest = _source_manifest()
    if (not cpu.get("passed") or cpu.get("plan_sha256") != canonical_sha256(plan)
            or file_sha256(golden_directory/"golden.npz") != cpu["golden_sha256"]
            or cpu["source_manifest"] != source_manifest):
        raise ValueError("CPU numerical registration/plan/project source/golden changed")
    stack, blas, cuda = loaded_cuda_stack(plan)
    atomic_json(output/"loaded_stack.json", stack)
    stress = concurrent_gemm_stress(blas, cuda, seed=plan["gemm_seed"], repetitions=plan["stress_repetitions"],
        remaining=lambda: plan["wall_cap_seconds"]-(time.monotonic()-start))
    atomic_json(output/"native_GEMM_stress.json", stress)
    if not stress["passed"]:
        raise ArithmeticError("explicit-stream concurrency acceptance incomplete: " + stress["status"])
    _check_deadline(start, plan["wall_cap_seconds"])
    derivatives = jax_derivative_and_solve_checks(plan["gemm_seed"]+1)
    atomic_json(output/"jax_derivative_solve.json", derivatives)
    _check_deadline(start, plan["wall_cap_seconds"])
    reports = {}
    with np.load(golden_directory/"golden.npz") as golden:
        for case in plan["cases"]:
            _check_deadline(start, plan["wall_cap_seconds"])
            saved, reg, identity, rows, _ = load_physical_case(case, input_root)
            # Explicit save/load of original parameters, optimizer and RNG;
            # no update or coordinate canonicalization touches this payload.
            restored_path = output/(case["tag"]+"_checkpoint_roundtrip.pkl")
            save_checkpoint(restored_path, saved["params"], optimizer_state=saved["optimizer_state"], metadata=saved["metadata"])
            restored = load_checkpoint(restored_path)
            if _state_identity(restored) != identity:
                raise ArithmeticError("GPU checkpoint parameter/optimizer/RNG roundtrip mismatch")
            probe, details = _case_probes(case, restored, reg, rows)
            errors = {key: _relative_error(value, golden[case["tag"]+"_"+key]) for key, value in probe.items()}
            if max(errors.values()) > CT_TOLERANCE:
                raise ArithmeticError("actual144-site CPU/B200 amplitude/H/Q/JVP replay failed")
            reports[case["tag"]] = dict(identity=identity, physical=details, relative_errors=errors,
                checkpoint_roundtrip_verified=True, checkpoint_roundtrip_scope="in_process_save_load",
                roundtrip_checkpoint_sha256=file_sha256(restored_path))
            atomic_json(output/(case["tag"]+"_GPU_stage.json"), reports[case["tag"]])
            _check_deadline(start, plan["wall_cap_seconds"])
    _verify_frozen_inputs(plan, input_root, source_manifest)
    _check_deadline(start, plan["wall_cap_seconds"])
    return dict(status="bounded_software_acceptance_complete", passed=True, stack=stack,
        plan_sha256=canonical_sha256(plan), source_manifest=source_manifest,
        CPU_registration_sha256=file_sha256(golden_directory/"CPU_registration.json"),
        native_GEMM_stress=stress, jax_derivative_solve=derivatives, cases=reports,
        seconds=time.monotonic()-start, no_production_training=True, automatic_production_release=False,
        new_physical_H_centers=4, new_samples=0,
        finite_acceptance_is_not_a_proof_all_software_bugs_absent=True)


# Native explicit-stream stress and JAX derivative checks are deliberately
# independent of the physical CT kernel and of its CPU golden preparation.


class _DoubleComplex(ct.Structure):
    _fields_ = [("real", ct.c_double), ("imag", ct.c_double)]


def gemm_reference(a: np.ndarray, b: np.ndarray, initial: np.ndarray, *, operation: int, beta: int) -> np.ndarray:
    """CPU reference from actual quantized inputs, with explicit transpose."""
    if operation not in (0, 1, 2) or beta not in (0, 1):
        raise ValueError("registered transpose/beta required")
    dtype = np.complex128 if np.iscomplexobj(a) else np.float64
    left, right = a.astype(dtype), b.astype(dtype)
    if operation:
        left = left.T if operation == 1 else left.conj().T
    return left @ right + beta*initial.astype(dtype)


def concurrent_gemm_stress(blas: Any, cuda: Any, *, seed: int, repetitions: int,
        remaining: Any) -> dict:
    """Use distinct handles/workspaces and explicit CUDA streams, no host race.

    CUDA-event stream interval overlap is recorded, not misrepresented as an
    instruction-level trace of simultaneous kernels or TMEM use.
    """
    void, integer, size = ct.c_void_p, ct.c_int, ct.c_size_t
    malloc = _function(cuda, "cudaMalloc", [ct.POINTER(void), size])
    free = _function(cuda, "cudaFree", [void])
    copy = _function(cuda, "cudaMemcpy", [void, void, size, integer])
    copy_async = _function(cuda, "cudaMemcpyAsync", [void, void, size, integer, void])
    stream_create = _function(cuda, "cudaStreamCreateWithFlags", [ct.POINTER(void), ct.c_uint])
    stream_sync = _function(cuda, "cudaStreamSynchronize", [void])
    stream_destroy = _function(cuda, "cudaStreamDestroy", [void])
    event_create = _function(cuda, "cudaEventCreate", [ct.POINTER(void)])
    event_record = _function(cuda, "cudaEventRecord", [void, void])
    event_sync = _function(cuda, "cudaEventSynchronize", [void])
    event_destroy = _function(cuda, "cudaEventDestroy", [void])
    event_time = _function(cuda, "cudaEventElapsedTime", [ct.POINTER(ct.c_float), void, void])
    create = _function(blas, "cublasCreate_v2", [ct.POINTER(void)])
    destroy = _function(blas, "cublasDestroy_v2", [void])
    set_stream = _function(blas, "cublasSetStream_v2", [void, void])
    set_workspace = _function(blas, "cublasSetWorkspace_v2", [void, void, size])
    set_math = _function(blas, "cublasSetMathMode", [void, integer])
    normal_arguments = [void, integer, integer, integer, integer, integer,
        void, void, integer, void, integer, void, void, integer]
    gemms = {name: _function(blas, "cublas"+letter+"gemm_v2", normal_arguments)
        for name, letter in (("float64", "D"), ("complex128", "Z"), ("float32", "S"))}
    half_gemm = _function(blas, "cublasGemmEx", [void, integer, integer, integer, integer, integer,
        void, void, integer, integer, void, integer, integer, void, void, integer, integer, integer, integer])
    cases, pointers, streams, handles, events = [], [], [], [], []
    rng = np.random.default_rng(seed)
    origin = void()
    try:
        _native_call(event_create, ct.byref(origin)); events.append(origin)
        for name in ("float64", "complex128", "float32", "float16"):
            dtype = np.dtype(name)
            for shape_name, m, n, k, transpose, beta in (
                ("square", 1024, 1024, 1024, 0, 0),
                ("ragged", 257, 193, 389, 0, 1),
                ("transpose", 383, 257, 193, 1, 0),
            ):
                if remaining() < 60:
                    raise TimeoutError("native stress panel incomplete; no reduction of registered cases")
                operation = 2 if name == "complex128" and transpose else transpose
                a_shape = (k, m) if transpose else (m, k)
                a, b = rng.normal(size=a_shape), rng.normal(size=(k, n))
                if name == "complex128":
                    a = a+1j*rng.normal(size=a_shape); b = b+1j*rng.normal(size=(k, n))
                a, b = np.asfortranarray(a.astype(dtype)), np.asfortranarray(b.astype(dtype))
                output_dtype = np.float32 if name == "float16" else dtype
                initial = np.asfortranarray(rng.normal(size=(m, n)).astype(output_dtype))
                reference = gemm_reference(a, b, initial, operation=operation, beta=beta)
                stream, handle = void(), void()
                _native_call(stream_create, ct.byref(stream), 1); streams.append(stream)
                _native_call(create, ct.byref(handle)); handles.append(handle)
                _native_call(set_stream, handle, stream)
                # SetStream resets workspace: allocate/set it AFTER the stream.
                device = []
                for array in (a, b, initial):
                    pointer = void(); _native_call(malloc, ct.byref(pointer), array.nbytes)
                    pointers.append(pointer); device.append(pointer)
                    _native_call(copy, pointer, void(array.ctypes.data), array.nbytes, 1)
                workspace = void(); _native_call(malloc, ct.byref(workspace), 4*1024**2)
                pointers.append(workspace)
                _native_call(set_workspace, handle, workspace, 4*1024**2)
                # PEDANTIC_MATH: no TF32 shortcut or FP64 emulation is enabled.
                _native_call(set_math, handle, 2)
                starts, stops = void(), void()
                _native_call(event_create, ct.byref(starts)); events.append(starts)
                _native_call(event_create, ct.byref(stops)); events.append(stops)
                if name == "complex128":
                    alpha, beta_value = _DoubleComplex(1., 0.), _DoubleComplex(float(beta), 0.)
                else:
                    scalar = ct.c_double if name == "float64" else ct.c_float
                    alpha, beta_value = scalar(1.), scalar(beta)
                cases.append(dict(name=name, shape=shape_name, m=m, n=n, k=k, operation=operation,
                    beta=beta, a=a, b=b, initial=initial, reference=reference, device=device,
                    stream=stream, handle=handle, start=starts, stop=stops,
                    alpha=alpha, beta_value=beta_value, maximum_cpu_relative_error=0., maximum_serial_relative_error=0.))

        def enqueue(case: dict, *, timed: bool, reset: bool = True) -> None:
            a, b, c = case["device"]
            if reset:
                _native_call(copy_async, c, void(case["initial"].ctypes.data), case["initial"].nbytes, 1, case["stream"])
            if timed: _native_call(event_record, case["start"], case["stream"])
            args = [case["handle"], case["operation"], 0, case["m"], case["n"], case["k"],
                ct.byref(case["alpha"]), a, case["a"].shape[0], b, case["b"].shape[0],
                ct.byref(case["beta_value"]), c, case["m"]]
            if case["name"] == "float16":
                # CUDA_R_16F=2, CUDA_R_32F=0, CUBLAS_COMPUTE_32F=68,
                # CUBLAS_GEMM_DEFAULT=-1; output/scalars are float32.
                args = [case["handle"], case["operation"], 0, case["m"], case["n"], case["k"],
                    ct.byref(case["alpha"]), a, 2, case["a"].shape[0], b, 2, case["b"].shape[0],
                    ct.byref(case["beta_value"]), c, 0, case["m"], 68, -1]
            _native_call(half_gemm if case["name"] == "float16" else gemms[case["name"]], *args)
            if timed: _native_call(event_record, case["stop"], case["stream"])

        # Serial warmup and reference; each subsequent round resets C0 before
        # beta1, so no recurrence or accidental cross-stream output sharing.
        for case in cases:
            enqueue(case, timed=False); _native_call(stream_sync, case["stream"])
            result = np.empty_like(case["initial"], order="F")
            _native_call(copy, void(result.ctypes.data), case["device"][2], result.nbytes, 2)
            case["serial"] = result.copy()
            case["maximum_cpu_relative_error"] = _relative_error(result, case["reference"])
        overlap_pairs, interval_rounds = 0, []
        for _ in range(repetitions):
            if remaining() < 60: raise TimeoutError("full four-round concurrent stress not completed")
            # Complete pageable H2D resets BEFORE the timed launch panel;
            # interleaved copies must not accidentally serialize all GEMMs.
            for case in cases:
                _native_call(copy_async, case["device"][2], void(case["initial"].ctypes.data),
                    case["initial"].nbytes, 1, case["stream"])
            for stream in streams: _native_call(stream_sync, stream)
            _native_call(event_record, origin, void()); _native_call(event_sync, origin)
            for case in cases: enqueue(case, timed=True, reset=False)
            for stream in streams: _native_call(stream_sync, stream)
            intervals = []
            for case in cases:
                begin, end = ct.c_float(), ct.c_float()
                _native_call(event_time, ct.byref(begin), origin, case["start"])
                _native_call(event_time, ct.byref(end), origin, case["stop"])
                intervals.append([begin.value, end.value])
                result = np.empty_like(case["initial"], order="F")
                _native_call(copy, void(result.ctypes.data), case["device"][2], result.nbytes, 2)
                case["maximum_cpu_relative_error"] = max(case["maximum_cpu_relative_error"], _relative_error(result, case["reference"]))
                case["maximum_serial_relative_error"] = max(case["maximum_serial_relative_error"], _relative_error(result, case["serial"]))
            overlap_pairs += sum(max(a[0], b[0]) < min(a[1], b[1]) for i, a in enumerate(intervals) for b in intervals[i+1:])
            interval_rounds.append(intervals)
        reports = []
        for case in cases:
            tolerance = 2e-11 if case["name"] in ("float64", "complex128") else 3e-5
            if max(case["maximum_cpu_relative_error"], case["maximum_serial_relative_error"]) > tolerance:
                raise ArithmeticError("native explicit-stream GEMM failed for " + case["name"] + "/" + case["shape"])
            reports.append({key: case[key] for key in ("name", "shape", "m", "n", "k", "operation", "beta",
                "maximum_cpu_relative_error", "maximum_serial_relative_error")})
        return dict(passed=overlap_pairs > 0,
            status="explicit_stream_numeric_checks_passed" if overlap_pairs else "inconclusive_no_stream_interval_overlap",
            seed=seed, repetitions=repetitions, cases=reports,
            explicitly_created_independent_streams=len(streams), independent_handles_and_workspaces=True,
            reset_C0_before_each_beta1=True, cuda_event_intervals_ms=interval_rounds,
            stream_interval_overlap_verified=overlap_pairs > 0, overlapping_interval_pairs=overlap_pairs,
            kernel_overlap_trace_verified=False, TMEM_instruction_path_verified=False,
            half_inputs_quantized_before_CPU_reference=True, half_GEMM_output_and_accumulation="float32",
            math_mode="CUBLAS_PEDANTIC_MATH", no_bug_absence_theorem=True)
    finally:
        # Do not leave asynchronous work using freed buffers on a failed check.
        for stream in streams:
            try: _native_call(stream_sync, stream)
            except Exception: pass
        for handle in handles: _native_call(destroy, handle)
        for event in events: _native_call(event_destroy, event)
        for pointer in pointers: _native_call(free, pointer)
        for stream in streams: _native_call(stream_destroy, stream)


def jax_derivative_and_solve_checks(seed: int) -> dict:
    """Check real-loss gradients and the SPD solve used by SR, not training."""
    import jax.scipy.linalg as jl
    rng = np.random.default_rng(seed)
    derivatives, systems = [], []
    for name in ("float64", "complex128", "float32", "float16"):
        complex_values = name == "complex128"
        a, b = rng.normal(size=(17, 13)), rng.normal(size=(13, 11))
        da, db = rng.normal(size=a.shape), rng.normal(size=b.shape)
        if complex_values:
            a = a+1j*rng.normal(size=a.shape); b = b+1j*rng.normal(size=b.shape)
            da = da+1j*rng.normal(size=da.shape); db = db+1j*rng.normal(size=db.shape)
        a, b, da, db = [jnp.asarray(x, dtype=name) for x in (a, b, da, db)]
        def loss(x, y):
            product = jnp.matmul(x, y, precision=jax.lax.Precision.HIGHEST)
            return jnp.vdot(product, product).real/product.size
        (value, (ga, gb)), tangent = jax.block_until_ready((jax.jit(jax.value_and_grad(loss, (0, 1)))(a, b),
            jax.jit(lambda x, y, dx, dy: jax.jvp(loss, (x, y), (dx, dy))[1])(a, b, da, db)))
        if not all(np.isfinite(np.asarray(x)).all() for x in (value, ga, gb, tangent)):
            raise ArithmeticError("nonfinite JAX GEMM gradient")
        adjoint = float(np.real(np.sum(np.asarray(ga)*np.asarray(da))+np.sum(np.asarray(gb)*np.asarray(db))))
        tolerance = 2e-10 if name in ("float64", "complex128") else (1e-4 if name == "float32" else .03)
        error = abs(float(tangent)-adjoint)/max(abs(float(tangent)), abs(adjoint), 1.)
        if error > tolerance: raise ArithmeticError("real-inner-product JVP/VJP adjoint mismatch")
        row = dict(dtype=name, finite=True, value=float(value), JVP=float(tangent), real_VJP_contraction=adjoint,
            real_adjoint_relative_error=error, complex_gradient_convention="Re(sum(grad*d)), not vdot(grad,d)")
        if name in ("float64", "complex128"):
            aa, bb, dda, ddb = map(np.asarray, (a, b, da, db))
            def cpu_loss(x, y):
                product = x @ y
                return np.vdot(product, product).real/product.size
            epsilon = 1e-5
            fd = float((cpu_loss(aa+epsilon*dda, bb+epsilon*ddb)-cpu_loss(aa-epsilon*dda, bb-epsilon*ddb))/(2*epsilon))
            fd_error = abs(fd-float(tangent))/max(abs(fd), abs(float(tangent)), 1.)
            if fd_error > 2e-7: raise ArithmeticError("CPU centered finite difference gradient mismatch")
            row.update(CPU_centered_FD=fd, CPU_FD_relative_error=fd_error)
        derivatives.append(row)
    for name in ("float64", "complex128"):
        r, rhs = rng.normal(size=(65, 65)), rng.normal(size=65)
        if name == "complex128": r = r+1j*rng.normal(size=r.shape); rhs = rhs+1j*rng.normal(size=rhs.shape)
        matrix = r.conj().T @ r+2*np.eye(65)
        reference = np.linalg.solve(matrix, rhs)
        actual = np.asarray(jax.block_until_ready(jax.jit(lambda h, b: jl.solve(h, b, assume_a="pos"))(
            jnp.asarray(matrix), jnp.asarray(rhs))))
        error = _relative_error(actual, reference)
        residual = float(np.linalg.norm(matrix @ actual-rhs)/np.linalg.norm(rhs))
        if max(error, residual) > 2e-11: raise ArithmeticError("SPD solve/Cholesky failed against CPU")
        systems.append(dict(dtype=name, dimension=65, relative_CPU_error=error, relative_residual=residual))
    solver_paths = []
    maps = Path("/proc/self/maps")
    if maps.exists():
        solver_paths = sorted({str(Path(line.split()[-1]).resolve()) for line in maps.read_text().splitlines()
            if line.split() and re.search(r"/libcusolver\.so", line.split()[-1])})
    return dict(passed=True, gradients=derivatives, SPD_solve=systems, loaded_cuSOLVER_paths=solver_paths,
        CPU_reference_only_small_linear_systems=True, no_physical_SR_update=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("cpu_prepare", "gpu_check"), required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--golden", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output/(args.phase+"_result.json")
    if result_path.exists() or (args.phase == "cpu_prepare" and (args.output/"CPU_registration.json").exists()):
        raise FileExistsError("acceptance output already frozen; do not overwrite")
    try:
        plan = json.loads(args.plan.read_text()); validate_plan(plan)
        if args.phase == "cpu_prepare":
            result = cpu_prepare(plan, args.input_root, args.output)
        else:
            if args.golden is None: raise ValueError("GPU check requires a frozen CPU golden directory")
            result = gpu_check(plan, args.input_root, args.output, args.golden)
        atomic_json(result_path, result)
    except Exception as error:
        atomic_json(result_path, dict(status="acceptance_failed_or_incomplete", passed=False,
            phase=args.phase, error=repr(error), traceback=traceback.format_exc(),
            no_production_training=True, automatic_production_release=False))
        raise
    print(json.dumps(dict(phase=args.phase, status=result["status"], passed=result["passed"])), flush=True)


if __name__ == "__main__":
    main()
