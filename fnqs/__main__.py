"""CLI: python -m fnqs {reference,train,recon} --config ... --output ..."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiments import run_neural, run_reconnaissance, run_reference
from .configuration import validate_config


def main() -> None:
    """Run the requested reproducible benchmark with an explicit config."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["reference","train","recon"])
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--resume",action="store_true",help="continue the same neural run for config.steps additional updates")
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    validate_config(args.command,config)
    if args.resume and args.command!="train":parser.error("--resume is only supported for train")
    if args.command=="reference":result=run_reference(config,args.output)
    elif args.command=="train":result=run_neural(config,args.output,args.resume)
    else:result=run_reconnaissance(config,args.output)
    summary={"kind":result["kind"],"elapsed_seconds":result["elapsed_seconds"],"output":str(args.output)}
    if "phase0_scientific_gate" in result:summary["phase0_scientific_gate"]=result["phase0_scientific_gate"]
    if "collapse" in result:summary["collapse"]=result["collapse"]
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=="__main__":main()
