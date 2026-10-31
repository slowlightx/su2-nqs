"""Plot existing development profiles without additional neural queries."""
from pathlib import Path
import argparse,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fnqs.artifacts import write_json
from fnqs.pilots.lineage import sha

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    directory=parser.parse_args().directory
    fig,axes=plt.subplots(2,3,figsize=(11.5,6),sharex=True)
    inputs={}
    for col,seed in enumerate((71,83,97)):
        for label,path,color in [('Parent',directory/'parents'/f'seed_{seed}'/'audit.json','grey'),
            ('Energy',directory/'runs'/f'energy_seed_{seed}'/'audit.json','C0'),
            ('Response',directory/'runs'/f'response_seed_{seed}'/'audit.json','C1')]:
            inputs[str(path)]=sha(path)
            rows=json.loads(path.read_text())['points']
            fields=[x['couplings'][-1] for x in rows]
            axes[0,col].errorbar(fields,[x['chi_total']/x['reference_chi_total'] for x in rows],
                yerr=[x['bootstrap_se']['chi_total']/x['reference_chi_total'] for x in rows],
                marker='o',color=color,label=label,capsize=2)
            axes[1,col].errorbar(fields,[x['response_relative_residual'] for x in rows],
                yerr=[x['bootstrap_se']['response_relative_residual'] for x in rows],
                marker='o',color=color,capsize=2)
        axes[0,col].axhline(1,color='black',ls=':',lw=1)
        axes[0,col].set_title(f'L=16, seed {seed}')
        axes[0,col].legend(fontsize=8)
        axes[1,col].set_xlabel('h')
        for row in (0,1):axes[row,col].grid(alpha=.2)
    axes[0,0].set_ylabel('chi / exact chi (total)')
    axes[1,0].set_ylabel('Projected response residual rho')
    fig.suptitle('Phase 0 repair: existing development points, error bars = 1 bootstrap SE')
    fig.tight_layout();path=directory/'repair_profiles.png';fig.savefig(path,dpi=160);plt.close(fig)
    write_json(directory/'repair_profiles_figure.json',dict(inputs_sha256=inputs,figure_sha256=sha(path),
        plot_source_sha256=sha(Path(__file__)),new_born_samples=0,new_hamiltonian_actions=0))

if __name__=='__main__':main()
