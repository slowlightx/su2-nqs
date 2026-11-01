"""Read installed distribution versions only; do not modify the shared env."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

expected = json.loads(Path(__file__).with_name('runtime_packages.json').read_text())
actual = {name:importlib.metadata.version(name) for name in expected}
if actual != expected:
    raise RuntimeError('shared runtime version mismatch: '+','.join(k for k in expected if actual[k] != expected[k]))
print(json.dumps(dict(runtime_packages=actual, runtime_sha256=hashlib.sha256(
    json.dumps(actual, sort_keys=True).encode()).hexdigest())), flush=True)
