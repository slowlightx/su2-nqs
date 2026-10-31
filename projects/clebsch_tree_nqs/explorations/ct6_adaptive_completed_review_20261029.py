"""Read-only mirror and independent review of completed CT6 adaptive parents.

The ongoing campaign is not sampled, modified, or used to select endpoints.
Reuse the independent NumPy oracle and full checkpoint/RNG lineage checks.
"""
from __future__ import annotations

import json
from pathlib import Path

from su2_nqs.frontier.artifacts import atomic_json
from . import ct6_completed_review_20261028 as previous


def run(output: Path, *, fetch: bool = True) -> dict:
    """Verify both completed 8-update parents with their untouched8192 audits."""
    previous.BASE = '/home/yx789/nqs-wavefunctions/nqs/ct6_adaptive_production_20261028'
    previous.FILES = (
        'checkpoint.pkl', 'registration.json', 'endpoint_frozen.json',
        'audit_result.json', 'train_result.json', 'training_progress.json',
        'replay_result.json', 'terminal_audit/audit.json',
        'terminal_audit/products.npz', 'terminal_audit/draw.npz',
        *(f'step{i}/transition.json' for i in range(1, 9)),
    )
    if fetch:
        previous.fetch(output)
    report = previous.review(output)
    for tag in ('ct6_j0', 'ct6_j0125'):
        directory = output / tag
        result = json.loads((directory / 'audit_result.json').read_text())
        registration = json.loads((directory / 'registration.json').read_text())
        audit = json.loads((directory / 'terminal_audit/audit.json').read_text())
        assert result['independent_from_training_geometry_and_adaptation']
        assert audit['no_endpoint_selection']
        assert audit == result['terminal_audit']
        assert registration['plan']['audit_records'] == audit['records']
        assert audit['seed'] == registration['plan']['seeds']['audit']
        assert audit['statistics_seed'] == registration['plan']['seeds']['audit_statistics']
    report['best_mean_reporting_note'] = (
        'Both paired endpoint gains are positive evidence of optimization, '
        'but neither new absolute audit mean is lower than selected466/492; '
        'keep the same selected means and retain all new quality points.'
    )
    atomic_json(output / 'independent_review.json', report)
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--review-existing', action='store_true')
    args = parser.parse_args()
    result = run(args.output, fetch=not args.review_existing)
    print(json.dumps({tag: {k: v for k, v in row.items() if k != 'moments'}
                      for tag, row in result['runs'].items()}, indent=2))
