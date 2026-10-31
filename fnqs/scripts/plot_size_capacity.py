"""Standalone scientific figures for the validated development controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fnqs.artifacts import write_json


COLORS = ["#555555", "#218a8f", "#185ea4", "#e69820", "#b95128"]


def plot(directory: Path) -> None:
    """Plot every seed, retaining the near-capacity-matched fixed aggregate."""
    result = json.loads((directory / "comparison.json").read_text())
    protocol = json.loads((directory / "protocol.json").read_text())
    base = protocol["base_steps"]
    names = [f"fixed16_{base}", f"shared16_{base}", f"shared16_{2*base}",
             f"shared32_{base}", f"shared32_{2*base}"]
    labels = ["Fixed 16\n1× budget", "Shared 16\n1× budget", "Shared 16\n2× budget",
              "Shared 32\n1× budget", "Shared 32\n2× budget"]
    seeds = result["seeds"]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2), constrained_layout=True)
    metrics = [("chi_max_relative_plus_2se", 100., .1, "Maximum χ relative error + 2SE (%)"),
               ("correlation_max_absolute_plus_2se", 1., .03, "Maximum |C − reference| + 2SE"),
               ("energy_max_relative_plus_2se", 1., .002, "Maximum E relative error + 2SE")]
    for ax, (metric, scale, threshold, title) in zip(axes[0], metrics):
        for j, seed in enumerate(seeds):
            y = [result["groups_by_seed"][str(seed)][name]["development"][metric]*scale for name in names]
            ax.plot(np.arange(len(names)), y, color="#888888", lw=.7, alpha=.6)
            ax.scatter(np.arange(len(names))+(j-1)*.055, y, c=COLORS, marker=["o","s","^"][j],
                       s=40, label=f"seed {seed}", zorder=3)
        ax.axhline(threshold*scale, color="#a12828", ls="--", lw=1, label="Existing margin")
        ax.set_xticks(range(len(names)), labels, fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=.2)
    axes[0, 0].legend(fontsize=8, loc="best")
    axes[0, 2].set_yscale("log")
    for ax, n in zip(axes[1], protocol["sizes"]):
        for j, seed in enumerate(seeds):
            values = [100*result["groups_by_seed"][str(seed)][name]["development"]["by_size"][str(n)]["chi_max_relative_plus_2se"] for name in names]
            ax.plot(range(len(names)), values, color="#888888", lw=.7, alpha=.6)
            ax.scatter(np.arange(len(names))+(j-1)*.055, values, c=COLORS, marker=["o","s","^"][j], s=40)
        ax.axhline(10., color="#a12828", ls="--", lw=1)
        ax.set_xticks(range(len(names)), labels, fontsize=8)
        ax.set_title(f"L={n}: maximum χ error + 2SE (%)", fontsize=11)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Size sharing × GRU width × optimization budget — development fields\n"
                 "Three independent cold seeds; 1× = 6,000 aggregate training Hamiltonians / 768,000 Born draws",
                 fontsize=13)
    fig.savefig(directory / "comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(seeds), 3, figsize=(14.5, 10.), sharex=True, constrained_layout=True)
    for i, seed in enumerate(seeds):
        for ax, n in zip(axes[i], protocol["sizes"]):
            for name, label, color in zip(names, labels, COLORS):
                raw = result["raw_rows_by_seed"][str(seed)]
                keys = [f"fixed16_L{n}_{base}"] if name.startswith("fixed") else [name]
                rows = sorted([r for key in keys for r in raw[key] if r["n"] == n and r["split"] == "development"], key=lambda r:r["h"])
                h = [r["h"] for r in rows]
                y = np.array([r["chi_total"] / r["chi_reference"] for r in rows])
                se = np.array([r["chi_se"] / r["chi_reference"] for r in rows])
                ax.plot(h, y, color=color, lw=1.5, label=label.replace("\n", ", "))
                ax.fill_between(h, y-2*se, y+2*se, color=color, alpha=.09)
            ax.axhline(1., color="black", lw=.8)
            ax.axhspan(.9, 1.1, color="#777777", alpha=.05)
            ax.set_title(f"seed {seed}, L={n}", fontsize=11)
            ax.grid(alpha=.2)
            if ax is axes[i,0]:
                ax.set_ylabel("χ / independent reference")
            if i == len(seeds)-1:
                ax.set_xlabel("h (development fields)")
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle("Total coupling QGT at fixed parameters; shading = ±2 bootstrap SE\n"
                 "Development measurements do not establish a universality class", fontsize=13)
    fig.savefig(directory / "response_profiles.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.4), sharex="col", constrained_layout=True)
    for col, seed in enumerate(seeds):
        for arm_name in protocol["arms"]:
            step = max(protocol["arms"][arm_name]["milestones"])
            path = directory / "runs" / arm_name / f"seed_{seed}" / f"step_{step}" / "checkpoint.npz"
            with np.load(path, allow_pickle=False) as archive:
                history = json.loads(str(archive["metadata"]))["history"]
            groups = [history[j:j+25] for j in range(0,len(history),25)]
            x = [g[-1]["step"] for g in groups]
            for row, metric in enumerate(["gradient_norm", "variance_per_site"]):
                axes[row,col].plot(x, [np.mean([r[metric] for r in g]) for g in groups], lw=1.25, label=arm_name)
                axes[row,col].set_yscale("log")
                axes[row,col].grid(alpha=.2)
                axes[row,col].axvline(base, color="#777777", ls="--", lw=.7)
        axes[0,col].set_title(f"seed {seed}")
        axes[1,col].set_xlabel("Accepted optimizer updates")
    axes[0,0].set_ylabel("Gradient norm (25-step mean)")
    axes[1,0].set_ylabel("Variance / site (25-step mean)")
    axes[0,0].legend(fontsize=8)
    fig.suptitle("Training diagnostics measured before each update\n"
                 "Fixed-size curves each use two tasks/update; shared curves use six", fontsize=13)
    fig.savefig(directory / "convergence.png", dpi=180)
    plt.close(fig)
    write_json(directory / "figures.json", {"plot_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "comparison_sha256": hashlib.sha256((directory / "comparison.json").read_bytes()).hexdigest(),
        "figures": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                    for name in ["comparison.png", "response_profiles.png", "convergence.png"]}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    plot(parser.parse_args().directory)
