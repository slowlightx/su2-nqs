"""CPU-only shell/certificate regressions; fake guard never accesses a GPU."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


WRAPPER = Path(__file__).resolve().parents[1]/'configs/remote/b200_runtime_fix_20261028/verified_python.sh'


def test_wrapper_shell_syntax_and_default_usage():
    assert subprocess.run(['bash', '-n', str(WRAPPER)]).returncode == 0
    result = subprocess.run(['bash', str(WRAPPER)], capture_output=True, text=True)
    assert result.returncode == 2 and 'Usage:' in result.stderr


@pytest.mark.parametrize('reason', ['no_sha', 'library_override'])
def test_wrapper_refuses_before_interpreter(reason):
    env = dict(os.environ)
    env.pop('NQS_B200_ACCEPTANCE_SHA256', None)
    env.pop('LD_LIBRARY_PATH', None)
    env.pop('LD_PRELOAD', None)
    if reason == 'library_override':
        env.update(NQS_B200_ACCEPTANCE_SHA256='a'*64, LD_LIBRARY_PATH='/untrusted')
    result = subprocess.run(['bash', str(WRAPPER), '--', '-c', 'print("SCIENCE")'],
        env=env, capture_output=True, text=True)
    assert result.returncode == 2 and 'Refusing:' in result.stderr
    assert 'SCIENCE' not in result.stdout


@pytest.mark.parametrize('failure', [None, 'certificate_sha', 'incomplete', 'plan', 'source', 'library'])
def test_certificate_and_guard_before_exec(tmp_path, failure):
    # Substitute only deployment paths, never the certificate validation code.
    text = WRAPPER.read_text().replace('/home/yx789/nqs-b200-validation-20261028', str(tmp_path))
    text = text.replace('/home/yx789/nqs-b200-runtime-20261028/venv/bin/python', sys.executable)
    wrapper = tmp_path/'verified_python.sh'
    wrapper.write_text(text)
    files = {
        'projects/clebsch_tree_nqs/workflows/accelerator_runtime_validation.py':
            'def validate_plan(p):\n    assert p["task"] == "test"\n'
            'def loaded_cuda_stack(p):\n    return ({"mapped_libraries": {"lib": {"sha256": "a"*64}}}, None, None)\n',
        'su2_nqs/frontier/artifacts.py':
            'import hashlib,json\ndef canonical_sha256(p):\n    return hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()\n',
        'su2_nqs/frontier/provenance.py':
            'def transitive_local_source_manifest(*a,**k):\n    return {"test_source": "sealed"}\n',
    }
    for name, contents in files.items():
        destination = tmp_path/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(contents)
    plan = dict(task='test', approved_distributions={'jax': 'test'}, approved_libraries={'lib': 'a'*64})
    plan_path = tmp_path/'configs/remote/b200_runtime_fix_20261028/plan.json'
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(json.dumps(plan))
    certificate = dict(status='bounded_software_acceptance_complete', passed=True,
        plan_sha256=hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
        source_manifest={'test_source': 'sealed'},
        stack=dict(distributions={'jax': 'test'}, mapped_libraries={'lib': dict(sha256='a'*64)}))
    if failure == 'incomplete': certificate['passed'] = False
    if failure == 'plan': certificate['plan_sha256'] = 'b'*64
    if failure == 'source': certificate['source_manifest'] = {}
    if failure == 'library': certificate['stack']['mapped_libraries']['lib']['sha256'] = 'b'*64
    cert_path = tmp_path/'gpu/gpu_check_result.json'
    cert_path.parent.mkdir()
    cert_path.write_text(json.dumps(certificate))
    digest = hashlib.sha256(cert_path.read_bytes()).hexdigest()
    env = dict(os.environ, NQS_B200_ACCEPTANCE_SHA256='b'*64 if failure == 'certificate_sha' else digest)
    for key in ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'PYTHONPATH'):
        env.pop(key, None)
    result = subprocess.run(['bash', str(wrapper), '--', '-c', 'print("SCIENCE")'],
        cwd=tmp_path, env=env, capture_output=True, text=True)
    if failure is None:
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == 'SCIENCE'
        assert 'verified_B200_runtime_startup_passed' in result.stderr
    else:
        assert result.returncode != 0 and 'SCIENCE' not in result.stdout
