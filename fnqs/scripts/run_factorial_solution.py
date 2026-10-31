"""Conditional A/B/C solution retest after the coverage/budget experiment."""
import argparse
from pathlib import Path
from fnqs.pilots.factorial_followup import prepare_followup,run_followup

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--factorial',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':
        if args.factorial is None:parser.error('--factorial is required for prepare')
        prepare_followup(args.output,args.factorial)
    else:run_followup(args.output)

if __name__=='__main__':main()
