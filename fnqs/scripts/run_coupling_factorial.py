"""CLI for the preregistered fixed-L coupling-coverage/budget experiment."""
import argparse
from pathlib import Path
from fnqs.pilots.factorial import prepare,run

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();(prepare if args.command=='prepare' else run)(args.output)

if __name__=='__main__':main()
