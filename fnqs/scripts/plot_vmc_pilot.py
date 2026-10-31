"""Export independent research-size response and correlation audit diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def main() -> None:
    """Plot saved point estimates and bootstrap SE without fitting exponents."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fnqs-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = data["arms"]
    fig, axes = plt.subplots(2, len(arms), figsize=(14, 7), layout="constrained", squeeze=False)
    colors = {n: f"C{i}" for i, n in enumerate(data["config"]["sizes"])}
    for column, arm in enumerate(arms):
        upper, lower = axes[:, column]
        for n, color in colors.items():
            rows = sorted((row for row in arm["rows"] if row["n"] == n), key=lambda row: row["h"])
            h = np.array([row["h"] for row in rows])
            chi = np.array([row["chi_total"] for row in rows])
            chi_se = np.array([row["chi_se"] for row in rows])
            upper.plot(h, chi, color=color, label=f"FNQS L={n}")
            upper.fill_between(h, np.maximum(0, chi - 2 * chi_se), chi + 2 * chi_se,
                               color=color, alpha=.12)
            upper.plot(h, [row["chi_reference"] for row in rows], "--", color=color,
                       label=f"Exact L={n}", linewidth=1)
            lower.plot(h, [row["max_correlation_abs_error"] for row in rows], color=color,
                       label=f"L={n}")
            held = [row for row in rows if row["split"] == "held_field"]
            upper.scatter([row["h"] for row in held], [row["chi_total"] for row in held],
                          facecolors="none", edgecolors=color, s=12)
        upper.set(title=f"seed {arm['seed']}", xlabel="h (Pauli units)",
                  ylabel="Total fidelity susceptibility")
        lower.axhline(data["observable_thresholds"]["correlation_absolute"], color="black",
                      linestyle=":", label="Accuracy threshold", linewidth=1)
        lower.set(xlabel="h (Pauli units)", ylabel="max r |C_FNQS(r) - C_exact(r)|", ylim=(0, .32))
    axes[0, 0].legend(fontsize=7, ncol=2)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle("Research-size SR pilot: 200 updates/seed; independent 4096-draw audits\n"
                 "Solid FNQS / dashed exact; open markers held fields; shading +/-2 bootstrap SE (descriptive)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180,
                metadata={"Software": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          "Description": "Input SHA256: " + hashlib.sha256(args.input.read_bytes()).hexdigest()})
    plt.close(fig)


if __name__ == "__main__":
    main()
