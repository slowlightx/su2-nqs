"""Export exact-sum small-system response comparisons from saved results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from fnqs.exact import TFIMReference


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=Path("fnqs/results"))
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"fnqs-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files={"Initial Adam 600":args.results/"phase0_response_pilot_v2"/"neural.json",
           "SR 1200, task minibatches":args.results/"phase0_shared_control_v3"/"shared_dense_sr"/"neural.json",
           "SR 2400, task minibatches":args.results/"phase0_shared_sr_refinement_v3"/"neural.json",
           "Then 50 full-ensemble steps":args.results/"phase0_full_ensemble_refinement_v3"/"neural.json"}
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout="constrained")
    for axis,n in zip(axes,[4,6,8],strict=True):
        for label,path in files.items():
            rows=sorted([r for r in json.loads(path.read_text())["arms"][0]["rows"] if r["n"]==n],key=lambda r:r["h"])
            axis.plot([r["h"] for r in rows],[r["chi_total"] for r in rows],label=label,linewidth=1.4)
        fields=np.linspace(.8,1.2,201)
        axis.plot(fields,[TFIMReference(n,float(h)).fidelity_susceptibility() for h in fields],"k--",label="Exact TFIM",linewidth=1.2)
        axis.set(xlabel="h (Pauli units)",ylabel="Total fidelity susceptibility",title=f"L={n}")
    axes[0].legend(fontsize=7)
    fig.suptitle("Shared GRU response calibration: seed 17, exact Born sums; no critical exponents")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=180,metadata={"Software":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    plt.close(fig)


if __name__=="__main__":main()
