"""Static scientific plots of the complete read-only response diagnosis."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fnqs.artifacts import write_json
from fnqs.scripts.analyze_size_capacity import digest, load_json
from fnqs.scripts.diagnose_response import GROUPS, validate

LABELS = ("Fixed 16, 1k", "Shared 16, 1k", "Shared 16, 2k", "Shared 32, 1k", "Shared 32, 2k")


def plot(directory: Path) -> dict:
    """Bands represent three-seed ranges, not confidence intervals."""
    protocol, _, _ = validate(directory)
    data = load_json(directory/"diagnosis.json")
    if not data["complete"] or data["protocol_sha256"] != digest(directory/"protocol.json"):
        raise ValueError("need complete, sealed diagnostics")
    rows = data["rows"]
    for c in protocol["cases"]:
        path = directory/"cases"/f"case_{c['index']:03d}.json"
        if load_json(path) != rows[c["index"]]:
            raise ValueError("changed input row")
    colors = plt.get_cmap("tab10").colors[:5]

    def draw(ax, n, metric):
        for group, label, color in zip(GROUPS, LABELS, colors):
            fields = sorted({r["case"]["h"] for r in rows})
            values = []
            for h in fields:
                selected = [r for r in rows if r["case"]["n"] == n and r["case"]["h"] == h and r["case"]["group"] == group]
                values.append([metric(r) for r in selected])
            values = np.array(values)
            ax.plot(fields, values.mean(axis=1), "o-", ms=3, lw=1.4, color=color, label=label)
            ax.fill_between(fields, values.min(axis=1), values.max(axis=1), color=color, alpha=.12)
        ax.set_xlabel("h/J (J=1)")
        ax.grid(alpha=.2)
        ax.set_xlim(.946, 1.053)

    fig, axes = plt.subplots(3, 3, figsize=(12.6, 9.2), constrained_layout=True)
    for row, n in enumerate((16, 24, 32)):
        draw(axes[row, 0], n, lambda r: r["original_v8_chi_ratio"])
        draw(axes[row, 1], n, lambda r: r["response_relative_residual"])
        draw(axes[row, 2], n, lambda r: r["fd_score_relative_rms_error"][0])
        axes[row, 0].axhline(1, ls="--", color="black", lw=.8)
        axes[row, 1].axhline(0, ls="--", color="black", lw=.8)
        axes[row, 2].set_yscale("log")
        axes[row, 0].set_ylabel(f"L={n}\nchi / exact chi")
        axes[row, 1].set_ylabel("Response defect / source")
        axes[row, 2].set_ylabel("Centered AD-FD RMS / sqrt(chi)")
    axes[0, 0].set_title("Response accuracy (original 8192-draw audit)")
    axes[0, 1].set_title(f"Physical response equation ({protocol['samples_per_point']} draws)")
    axes[0, 2].set_title("Input derivative check (FD step=0.001)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Fixed-theta coupling response: three seeds per group\nLines: seed means; bands: seed min-max. Development diagnosis, no new gate.", fontsize=12)
    main = directory/"response_diagnosis.png"
    fig.savefig(main, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(9.4, 8.8), constrained_layout=True)
    for row, n in enumerate((16, 24, 32)):
        draw(axes[row, 0], n, lambda r: r["state_residual_norm"]/np.sqrt(n))
        draw(axes[row, 1], n, lambda r: r["hf_derivative_defect"]/n)
        axes[row, 0].set_ylabel(f"L={n}\nState residual / sqrt(L)")
        axes[row, 1].set_ylabel("Derivative defect / L")
        axes[row, 1].axhline(0, ls="--", color="black", lw=.8)
    axes[0, 0].set_title("State residual: sqrt(Var H / L)")
    axes[0, 1].set_title("Hellmann-Feynman derivative defect / L")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Finite-energy neural states: residual and derivative covariance\nBands: ranges across three seeds; Monte Carlo bootstrap SE retained in JSON.", fontsize=12)
    secondary = directory/"state_and_energy_derivative.png"
    fig.savefig(secondary, dpi=180)
    plt.close(fig)
    metadata = {"input_sha256": digest(directory/"diagnosis.json"),
        "plot_source_sha256": digest(Path(__file__)),
        "figures": {p.name: digest(p) for p in (main, secondary)},
        "bands": "min-max across three seeds; not confidence intervals",
        "confirmation_measured": False}
    write_json(directory/"figures.json", metadata)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("fnqs/results/coupling_response_diagnostics_v9"))
    args = parser.parse_args()
    print(plot(args.directory))
