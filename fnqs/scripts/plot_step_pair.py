"""Export the sealed fixed-L16 learning-rate pair without new measurements."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fnqs.artifacts import write_json
from fnqs.pilots.lineage import sha
from fnqs.scripts.run_l16_pair import metrics


def plot(directory: Path) -> None:
    """Compare both equal-cost branches with their common physical parent."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    protocol = json.loads((directory/'protocol.json').read_text())
    review = json.loads((directory/'review.json').read_text())
    if len(review['rows']) != 6:
        raise ValueError('complete six-endpoint review required')
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey='row')
    inputs = {}
    for column, seed in enumerate((71, 83, 97)):
        spec = next(r for r in protocol['runs'] if r['seed']==seed)
        parent = Path(spec['parent']).parent
        branches = [('Parent (chart200)', parent, 'C3^-'),
                    ('+100, learning rate 0.5', directory/'runs'/f'slow_seed_{seed}', 'C0o-'),
                    ('+100, learning rate 4', directory/'runs'/f'fast_seed_{seed}', 'C2s-')]
        worst = []
        for label, folder, style in branches:
            path = folder/'audit.json'
            data = json.loads(path.read_text())['points']
            inputs[str(path)] = sha(path)
            worst.append(100*metrics(data)['chi'])
            axes[1, column].plot([p['couplings'][3] for p in data],
                                [p['chi_reference_ratio'] for p in data], style, ms=3, label=label)
        axes[0, column].bar(range(3), worst, color=['C3','C0','C2'])
        axes[0, column].set_xticks(range(3), ['Parent', '+100 slow', '+100 fast'], rotation=15)
        axes[0, column].axhline(10, color='grey', ls=':')
        axes[0, column].set_title(f'L16 parent seed {seed}')
        axes[1, column].axhline(1, color='black', lw=1)
        axes[1, column].axhspan(.9, 1.1, color='grey', alpha=.13)
        axes[1, column].set_xlabel('Transverse field h')
        for row in (0, 1):
            axes[row, column].grid(alpha=.2)
    axes[0, 0].set_ylabel('Worst chi relative error (%)')
    axes[1, 0].set_ylabel('chi / independent JW reference')
    axes[1, 2].legend(fontsize=7)
    fig.tight_layout()
    target = directory/'comparison.png'
    fig.savefig(target, dpi=170)
    plt.close(fig)
    write_json(directory/'figure.json', dict(figure_sha256=sha(target), input_sha256=inputs,
        plot_source_sha256=sha(Path(__file__)),
        slow_fast_incremental_work_equal=True, parent_has_no_extra_updates=True,
        new_hamiltonian_actions=0, new_born_samples=0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot(args.output)


if __name__ == '__main__':
    main()
