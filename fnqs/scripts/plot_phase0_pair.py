"""Export paired held-field response and correlation diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {name: args.pair / name / "neural.json" for name in ("warm", "scratch")}
    runs = {name: json.loads(path.read_text()) for name, path in paths.items()}
    protocol = json.loads((args.pair / "protocol.json").read_text())
    for name, run in runs.items():
        if run["config"] != protocol["configs"][name] or run["source_sha256"] != protocol["source_sha256"]:
            raise ValueError("result does not match the predeclared paired experiment")
    config = runs["scratch"]["config"]
    seeds = config["seeds"]
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fnqs-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, len(seeds), figsize=(14, 10), layout="constrained", squeeze=False)
    for column, seed in enumerate(seeds):
        for size_index, n in enumerate(config["sizes"]):
            color = f"C{size_index}"
            for name, style in (("warm", "-"), ("scratch", "--")):
                arm = next(arm for arm in runs[name]["arms"] if arm["seed"] == seed)
                rows = sorted((row for row in arm["rows"] if row["n"] == n and row["split"] == "held_field"),
                              key=lambda row: row["h"])
                h = np.array([row["h"] for row in rows])
                chi = np.array([row["chi_total"] for row in rows])
                chi_se = np.array([row["chi_se"] for row in rows])
                label = f"{name}, L={n}"
                axes[0, column].plot(h, chi, style, color=color, label=label)
                axes[0, column].fill_between(h, np.maximum(0, chi - 2 * chi_se), chi + 2 * chi_se,
                                             color=color, alpha=.07)
                axes[1, column].plot(h, [row["chi_relative_error"] for row in rows], style,
                                     color=color, label=label)
                axes[2, column].plot(h, [row["max_correlation_abs_error"] for row in rows], style,
                                     color=color, label=label)
                if name == "warm":
                    axes[0, column].plot(h, [row["chi_reference"] for row in rows], ":", color=color,
                                         linewidth=1, label=f"exact, L={n}")
        axes[0, column].set(title=f"seed {seed}", ylabel="Total fidelity susceptibility")
        axes[1, column].set(ylabel="Relative response error (point estimate)")
        axes[2, column].set(xlabel="h (Pauli units)", ylabel="max r |C_FNQS(r) - C_exact(r)|")
        axes[1, column].axhline(.1, color="black", linestyle=":", linewidth=.8)
        axes[2, column].axhline(.03, color="black", linestyle=":", linewidth=.8)
        axes[1, column].set_ylim(bottom=0)
        axes[2, column].set_ylim(bottom=0)
    axes[0, 0].legend(fontsize=7, ncol=3)
    fig.suptitle(f"Paired Phase 0: {config['steps']} updates and {config['audit_samples']} fresh audit draws/point\n"
                 "Warm solid / scratch dashed / exact dotted; held fields only; shading +/-2 bootstrap SE\n"
                 "Accuracy gates also include sampling uncertainty; warm parental training costs are additional")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    fig.savefig(args.output, dpi=180, metadata={
        "Software": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "Description": json.dumps(hashes, sort_keys=True)})
    plt.close(fig)


if __name__ == "__main__":
    main()
