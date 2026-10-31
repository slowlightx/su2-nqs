"""Plot finite-size exact references without assigning universality classes."""

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
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir()) / "fnqs-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    names = sorted({cut["name"] for cut in data["cuts"]})
    even, odd = [], []
    for name in names:
        cuts = sorted((cut for cut in data["cuts"] if cut["name"] == name), key=lambda cut:cut["n"])
        n = np.array([cut["n"] for cut in cuts])
        axes[0].plot(1 / n, [cut["peak_field"] for cut in cuts], "o-", label=name)
        axes[1].plot(n, [max(row["chi_total"] for row in cut["rows"]) for cut in cuts], "o-", label=name)
        largest = cuts[-1]
        peak = max(largest["rows"], key=lambda row:row["chi_total"])
        even.append(peak["same_parity_gap"]); odd.append(peak["opposite_parity_gap"])
    positions = np.arange(len(names))
    axes[2].bar(positions - .18, odd, width=.36, label="Opposite parity")
    axes[2].bar(positions + .18, even, width=.36, label="Same parity (response)")
    axes[2].set_xticks(positions, names, rotation=35)
    axes[0].set(xlabel="1/L", ylabel="Finite-size fidelity peak h*(L)", title="Peaks, not thermodynamic critical fields")
    axes[1].set(xlabel="L", ylabel="Total susceptibility at finite-size peak", title="Exact response; no exponent imposed")
    axes[2].set(ylabel="Gap in Pauli units", title="L=14 gaps at each finite-size peak", yscale="log")
    axes[0].legend(fontsize=7,ncol=2); axes[2].legend(fontsize=7)
    fig.suptitle("Phase 1 candidate references: parity ED and projected response; no class assignment")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=180,metadata={"Software":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    plt.close(fig)


if __name__ == "__main__":
    main()
