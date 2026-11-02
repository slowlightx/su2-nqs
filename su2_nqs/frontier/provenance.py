"""Reproducibility seals for production experiment entry points.

The source manifest follows local Python imports recursively instead of relying
on a hand-maintained list.  It deliberately records source files; importing a
module remains the responsibility of the experiment and its route gate.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import platform
import sys
from typing import Iterable


def file_sha256(path: Path) -> str:
    """Return the SHA256 digest of one source file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _module_name(path: Path, root: Path) -> str:
    relative = path.resolve().relative_to(root.resolve()).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _module_path(module: str, root: Path) -> Path | None:
    parts = module.split(".")
    source = root.joinpath(*parts).with_suffix(".py")
    if source.is_file():
        return source.resolve()
    package = root.joinpath(*parts, "__init__.py")
    if package.is_file():
        return package.resolve()
    return None


def _package_initializers(module: str, root: Path) -> tuple[Path, ...]:
    parts = module.split(".")
    result = []
    for stop in range(1, len(parts) + 1):
        candidate = root.joinpath(*parts[:stop], "__init__.py")
        if candidate.is_file():
            result.append(candidate.resolve())
    return tuple(result)


def _import_candidates(path: Path, root: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(), filename=str(path))
    current_module = _module_name(path, root)
    current_package = (
        current_module.split(".")
        if path.name == "__init__.py"
        else current_module.split(".")[:-1]
    )
    candidates: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            candidates.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                remove = node.level - 1
                prefix = current_package[: len(current_package) - remove]
                suffix = [] if node.module is None else node.module.split(".")
                base = ".".join(prefix + suffix)
            else:
                base = node.module or ""
            if base:
                candidates.add(base)
            for alias in node.names:
                if alias.name != "*" and base:
                    candidates.add(f"{base}.{alias.name}")
    return tuple(sorted(candidates))


def transitive_local_source_manifest(
    entry_paths: Iterable[Path],
    *,
    root: Path,
    package_prefixes: tuple[str, ...] = ("su2_nqs",),
) -> dict[str, str]:
    """Hash entry files and their recursively imported local dependencies."""

    root = root.resolve()
    queue = [Path(path).resolve() for path in entry_paths]
    visited: set[Path] = set()
    while queue:
        path = queue.pop()
        if path in visited:
            continue
        if not path.is_file() or path.suffix != ".py":
            raise FileNotFoundError(path)
        visited.add(path)
        for module in _import_candidates(path, root):
            if not any(
                module == prefix or module.startswith(prefix + ".")
                for prefix in package_prefixes
            ):
                continue
            resolved = _module_path(module, root)
            if resolved is not None and resolved not in visited:
                queue.append(resolved)
            for initializer in _package_initializers(module, root):
                if initializer not in visited:
                    queue.append(initializer)
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(visited)
    }


def numerical_runtime_identity() -> dict[str, object]:
    """Return the numerical software/device identity relevant to resuming."""

    import jax
    import numpy

    distributions = {}
    for name in ("jax", "jaxlib", "numpy", "scipy", "optax"):
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = None
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "byteorder": sys.byteorder,
        "distributions": distributions,
        "numpy_version_runtime": numpy.__version__,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "optimizer_implementation": (
            "optax" if importlib.util.find_spec("optax") is not None else "local_fallback"
        ),
        "selected_environment": {
            name: os.environ.get(name)
            for name in ("JAX_PLATFORM_NAME", "JAX_ENABLE_X64", "XLA_FLAGS")
        },
    }


__all__ = [
    "file_sha256",
    "numerical_runtime_identity",
    "transitive_local_source_manifest",
]
