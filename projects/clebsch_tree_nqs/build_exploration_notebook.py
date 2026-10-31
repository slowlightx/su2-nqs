#!/usr/bin/env python3
"""Build and execute the comprehensive exploratory ClebschTree notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).resolve().parent
OUTPUT = PROJECT / "notebooks/clebsch_tree_exploration.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def channel_attention_cells():
    """Self-contained, incrementally executable local-channel study."""
    cells = [markdown(r"""
## Local spin transitions and conditional scalar attention (2026-09-26)

The new question is whether **extra local conservation laws**, rather than
global SU(2), obstruct useful learning. Uniform modes $Q_A\cdot Q_B$ preserve
each cell spin. Nonuniform rank-one modes $T_A^\mu\cdot T_B^\mu$ allow local
spin transitions while remaining global scalars. Attention conditions their
coefficients on spectator-cell $q=0,2,4$ projectors, never magnetic indices.

All five arms start from the same projected-pi-flux two-Lanczos state. Joint
and staged arms allocate 620 parameters and use 180 updates; staged freezes
attention for the first 60. The local-only ablation has four active parameters.
The scramble is **nonlocal isospectral**, fixes the carrier, and preserves
the residual Gram matrix; it is not a locality-preserving lattice scramble.
No ED state is used in the training loss. All final seeds are displayed;
training lines are medians and shaded bands are seed ranges, not error bars.

**Boundary:** this exact-support oracle uses a 14-dimensional $S=0,\Gamma-A_1$
block. The conditional linear span is 14/14, whereas the local-only span has
rank 3. Neither near-exact energies nor beating degree-four Krylov establishes
a scalable win. No fixed-$S^z$ / GPU-work comparison is performed here.
"""), code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
root = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'su2_nqs').is_dir())
project = root / 'projects/clebsch_tree_nqs'
channel = json.loads((project / 'results/spin_channel_attention_v1/results.json').read_text())
out = project / 'figures/exploration'
out.mkdir(parents=True, exist_ok=True)
arms = ('local_only', 'joint', 'staged', 'scrambled_joint', 'scrambled_staged')
colors = dict(zip(arms, ('#777777','#0072B2','#009E73','#D55E00','#CC79A7')))
labels = dict(zip(arms, ('Local only','Joint','Staged','Scrambled joint','Scrambled staged')))
print('Result hash:', channel['result_sha256'])
print('Exact checks:', channel['checks'])

fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)
for col, (coupling, case) in enumerate(channel['cases'].items()):
    e0 = case['controls']['exact_a1']['energy_per_site']
    for arm in arms:
        rows = [r for r in case['runs'] if r['arm'] == arm]
        steps = [p['step'] for p in rows[0]['history']]
        es = np.array([[p['energy_per_site']-e0 for p in r['history']] for r in rows])
        vs = np.array([[p['variance_total'] for p in r['history']] for r in rows])
        for ax, values in zip(axes[:,col], (es,vs)):
            ax.plot(steps, np.median(values,axis=0), c=colors[arm], label=labels[arm])
            ax.fill_between(steps, np.min(values,axis=0), np.max(values,axis=0), color=colors[arm], alpha=.1)
            ax.set_yscale('log')
            ax.axvline(60,c='0.8',lw=.7,ls=':')
    axes[0,col].axhline(case['controls']['degree_four_krylov']['energy_per_site']-e0,c='black',ls='--',label='K4')
    axes[1,col].axhline(case['controls']['degree_four_krylov']['variance_total'],c='black',ls='--')
    axes[0,col].set_title(f'Triangular 4x4, J2/J1={coupling}')
    axes[1,col].set_xlabel('Exact objective updates (not matched H work)')
axes[0,0].set_ylabel('Energy error / site')
axes[1,0].set_ylabel('Total energy variance')
axes[0,2].legend(fontsize=7,frameon=False)
fig.savefig(out/'fig53_channel_attention_training.png',dpi=180)
fig.savefig(out/'fig53_channel_attention_training.pdf')
plt.show()
"""), code(r"""
case = channel['cases']['0.125']
fig, axes = plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
e0 = case['controls']['exact_a1']['energy_per_site']
for arm in arms:
    rows = [r for r in case['runs'] if r['arm']==arm]
    axes[0].scatter([r['energy_per_site']-e0 for r in rows],
        [r['physics']['errors_to_exact']['trace_distance'] for r in rows],
        c=colors[arm],label=labels[arm],s=30)
    # Plot errors in q weights, with individual seeds, not bars.
    reference = case['controls']['exact_a1']['physics']['sectors']
    for index,row in enumerate(rows):
        delta = [row['physics']['sectors'][str(q)]['weight']-reference[str(q)]['weight'] for q in (0,2,4)]
        axes[1].plot((0,2,4),delta,'o-',c=colors[arm],alpha=.45,lw=.8)
for name,label,marker in (('two_lanczos','K2','s'),('degree_four_krylov','K4','D')):
    row=case['controls'][name]
    axes[0].scatter(row['energy_per_site']-e0,row['physics']['errors_to_exact']['trace_distance'],marker=marker,c='black')
    axes[0].annotate(label,(row['energy_per_site']-e0,row['physics']['errors_to_exact']['trace_distance']),
        xytext=(-4,-10) if name=='two_lanczos' else (3,3),textcoords='offset points',
        ha='right' if name=='two_lanczos' else 'left',fontsize=8)
axes[0].set(xscale='log',yscale='log',xlabel='Energy error / site',ylabel='Four-site RDM trace distance',title='Energy and local-state accuracy differ')
axes[0].legend(fontsize=7,frameon=False)
axes[1].axhline(0,c='black',lw=.7)
axes[1].set(xlabel='Doubled cell spin q = 2S',ylabel='p(q) - exact p(q)',xticks=(0,2,4),title='Local spin weights: all seeds')
names=('spin_weight_entropy','magnetic_entropy','multiplicity_entropy')
for arm in ('joint','staged'):
    rows=[r for r in case['runs'] if r['arm']==arm]
    values=[[r['physics']['entropy_components'][n]-case['controls']['exact_a1']['physics']['entropy_components'][n] for n in names] for r in rows]
    for v in values:
        axes[2].plot(range(3),v,'o-',c=colors[arm],alpha=.6,lw=1)
axes[2].axhline(0,c='black',lw=.7)
axes[2].set(xticks=range(3),xticklabels=('Spin weights','Magnetic','Multiplicity'),ylabel='Entropy component error (nats)',title='Joint (blue) vs staged (green)')
axes[2].tick_params(axis='x',labelrotation=20)
fig.suptitle('Triangular 4x4, J2/J1=0.125; exact S=0, Gamma-A1',fontsize=11)
fig.savefig(out/'fig54_channel_attention_physics.png',dpi=180)
fig.savefig(out/'fig54_channel_attention_physics.pdf')
plt.show()
for coupling,case in channel['cases'].items():
    print('\nJ2/J1 =',coupling,'conditional rank',case['conditional_span_rank'],'/',case['a1_dimension'])
    for arm in arms:
        rows=[r for r in case['runs'] if r['arm']==arm]
        print(arm,'median E/N',np.median([r['energy_per_site'] for r in rows]),
              'median Var',np.median([r['variance_total'] for r in rows]))
    print('Paired comparisons:',case['comparisons'])
"""), markdown(r"""
### Interpretation and the next discriminating test

Joint physical attention beats its isospectral scramble in energy and variance
for all nine paired seeds. At J2/J1=0.125 it does **not** reproducibly beat K4;
at 0.175 all three seeds do, in a saturated exact-small space. Staging gives
larger variance in all nine pairs. At 0.125 it nevertheless has smaller median
four-site RDM error (0.000244 vs 0.000687 for joint; K4: 0.000563).

For a singlet, the exact identity
$S_A=H(p_q)+\sum_q p_q\log(q+1)+\sum_q p_q S(\sigma_q)$ separates sector
probabilities, fixed magnetic degeneracies and multiplicity entanglement.
All three are measured here; attention maps are not observables. The strong
K2 carrier already has accurate local spin weights. Its remaining error is
not diagnosed by simply asking for more triplet support.

The next useful experiment separates q-changing transitions from transitions
between multiplicity copies at fixed q, with restricted, **non-saturated**
operator rank. Only then compare rank and actual action cost to Krylov and
fixed-Sz controls, followed by local two-cell density matrices / dimer
correlations. No evidence here supports a universal local-first curriculum,
a QSL phase boundary, or a remote production release.

Context: [SU(2) tensor structure and degeneracy spaces](https://arxiv.org/abs/1208.3919),
[exact SU(2) and lattice VMC](https://arxiv.org/abs/2104.14869),
[2D Transformer wavefunctions](https://arxiv.org/abs/2311.16889).
""")]
    for cell in cells:
        cell.metadata["experiment_section"] = "channel_attention_20260926"
    energy_cell = code(r"""
# Post-hoc observable analysis of the same sealed states; no retraining.
with np.load(project/'results/spin_channel_attention_v1/states.npz',allow_pickle=False) as stored:
    final_states={k:stored[k] for k in stored.files}
with np.load(project/'results/spin_carrier_cell_scalar_circuit_v1/states.npz',allow_pickle=False) as stored:
    h1=stored['coupling_0_h_a1']
    h2=(stored['coupling_0.125_h_a1']-h1)/.125
case=channel['cases']['0.125']
def bond_energies(v):
    return np.array([np.vdot(v,h1@v).real/16,np.vdot(v,h2@v).real/16])
exact_bonds=bond_energies(final_states['j0.125_exact_a1'])
fig,ax=plt.subplots(figsize=(6.3,4.2),constrained_layout=True)
for arm in arms:
    vs=[final_states[f'j0.125_{arm}_{seed}'] for seed in channel['config']['seeds']]
    delta=np.array([(bond_energies(v)-exact_bonds)*np.array([1,.125]) for v in vs])
    ax.scatter(delta[:,0],delta[:,1],c=colors[arm],label=labels[arm],s=35)
for name,label,marker in (('two_lanczos','K2','s'),('degree_four_krylov','K4','D')):
    components=bond_energies(final_states['j0.125_'+name])
    delta=(components-exact_bonds)*np.array([1,.125])
    ax.scatter(*delta,c='black',marker=marker)
    ax.annotate(label,delta,xytext=(4,3),textcoords='offset points',fontsize=9)
    print(name, 'J1 bond sum / N, J2 bond sum / N:',components)
print('Exact bond components:',exact_bonds)
x=np.linspace(-.00112,.00006,100)
ax.plot(x,-x,'k--',lw=1,label='Equal total energy to exact')
ax.scatter(0,0,marker='*',s=85,c='black',label='Exact local balance')
ax.set(xlabel=r'Error in $\langle H_1\rangle/N$',
       ylabel=r'Error in $J_2\langle H_2\rangle/N$',
       title='Competing bond energies: triangular 4x4, J2/J1=0.125')
ax.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
ax.legend(frameon=False,fontsize=8)
fig.savefig(out/'fig55_channel_attention_bond_balance.png',dpi=180)
fig.savefig(out/'fig55_channel_attention_bond_balance.pdf')
plt.show()
""")
    energy_cell.metadata["experiment_section"] = "channel_attention_20260926"
    cells.insert(-1, energy_cell)
    return cells


def channel_mechanism_cells():
    """Non-saturated physical channel intervention, not another neural fit."""
    cells = [markdown(r"""
## Local-q transfer versus pure multiplicity: non-saturated test (2026-09-27)

The entire K2 anchor space (rank 3) is retained. Eight preregistered operator
ablations add fixed physical directions, with final ranks 5--10 out of 14.
Each is compared to **same-rank** Krylov and five random rotations fixing
the entire K2 space. There is no neural training, best-seed selection, or
ED-target loss. All comparisons reuse the existing exact A1 Hamiltonian.

For a local exchange O, define $O_{keep}=\sum_{q_A,q_B}P_{q_A,q_B}OP_{q_A,q_B}$
and $O_{transfer}=O-O_{keep}$. The q-preserving exchange still contains
magnetic exchange. A separate control uses products of two cell dimer scalars,
which act as identity on magnetic indices and nontrivially on multiplicities.
This distinction is tested explicitly with independent spin matrices and
complete local Schur blocks.

All labels refer to the **pre-space-projection local operator**. Space-group
averaging need not preserve q for one fixed partition; even a q-preserving
nonunitary dressing can reweight existing q probabilities. Selection rules
are not the same thing as fixed probabilities.
"""), code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from su2_nqs.frontier.artifacts import canonical_sha256
root = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p/'su2_nqs').is_dir())
project = root/'projects/clebsch_tree_nqs'
mechanism = json.loads((project/'results/spin_channel_mechanism_v1/results.json').read_text())
assert canonical_sha256({k:v for k,v in mechanism.items() if k!='result_sha256'}) == mechanism['result_sha256']
out = project/'figures/exploration'
out.mkdir(parents=True,exist_ok=True)
styles = {
 'q_transfer': ('q transfer','#0072B2','o'),
 'q_preserving_exchange': ('q-preserving exchange','#D55E00','s'),
 'multiplicity_dimer': ('Multiplicity dimer','#CC79A7','^'),
 'unsplit_exchange': ('Unsplit exchange','#009E73','D'),
 'q_transfer_plus_multiplicity': ('Transfer + multiplicity','#56B4E9','v'),
 'q_preserving_plus_multiplicity': ('Keep + multiplicity','#E69F00','P'),
 'split_exchange': ('Split exchange','#882255','X'),
 'all_channels': ('All channels','#444444','*'),
}
fig,axes=plt.subplots(1,3,figsize=(12,4.2),constrained_layout=True)
for ax,(coupling,case) in zip(axes,mechanism['cases'].items()):
    e0=case['controls']['exact_a1']['energy_per_site']
    controls={3:case['controls']['two_lanczos']['energy_per_site']}
    for name,row in case['arms'].items():
        label,color,marker=styles[name]
        ax.scatter(row['rank'],max(row['energy_per_site']-e0,1e-15),c=color,marker=marker,s=48,label=label)
        controls[row['rank']]=row['matched_krylov']['energy_per_site']
    ranks=sorted(controls)
    ax.plot(ranks,[max(controls[r]-e0,1e-15) for r in ranks],'k.--',lw=1,label='Matched-rank Krylov')
    ax.set(yscale='log',xlabel='Variational subspace rank',xticks=(3,5,6,8,10),title=f'J2/J1={coupling}')
axes[0].set_ylabel('Energy error / site (floor 1e-15)')
axes[2].legend(fontsize=6.5,frameon=False,loc='lower left')
fig.suptitle('Triangular 4x4: all tested operator sets lose to same-rank Krylov',fontsize=12)
fig.savefig(out/'fig56_channel_mechanism_rank_frontier.png',dpi=180)
fig.savefig(out/'fig56_channel_mechanism_rank_frontier.pdf')
plt.show()
print('Result hash:',mechanism['result_sha256'])
print('Checks and construction costs:',mechanism['diagnostics'])
"""), code(r"""
case=mechanism['cases']['0.125']
e0=case['controls']['exact_a1']['energy_per_site']
fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
selected=('q_transfer','q_preserving_exchange','multiplicity_dimer','unsplit_exchange','split_exchange','all_channels')
reference=case['controls']['exact_a1']['physics']
for name in selected:
    row=case['arms'][name]
    label,color,marker=styles[name]
    axes[0].scatter(row['energy_per_site']-e0,row['physics']['two_cells']['errors_to_exact']['trace_distance'],c=color,marker=marker,s=50,label=label)
    delta=np.array([row['physics']['h1_per_site']-reference['h1_per_site'],.125*(row['physics']['h2_per_site']-reference['h2_per_site'])])
    axes[1].scatter(*delta,c=color,marker=marker,s=50)
carrier=case['controls']['two_lanczos']
axes[0].scatter(carrier['energy_per_site']-e0,carrier['physics']['two_cells']['errors_to_exact']['trace_distance'],c='black',marker='+',s=60,label='K2')
x=np.linspace(-.0013,.0001,100)
axes[1].plot(x,-x,'k--',lw=.8,label='Exact total energy')
axes[1].scatter(0,0,c='black',marker='*',s=80)
axes[0].set(xscale='log',yscale='log',xlabel='Energy error / site',ylabel='Two-cell RDM trace distance',title='Tested multiplicity-dimer channel is insufficient')
from matplotlib.ticker import LogLocator, NullFormatter
axes[0].set_xlim(1e-5,3e-4)
axes[0].xaxis.set_major_locator(LogLocator(base=10,subs=(1.,)))
axes[0].xaxis.set_minor_formatter(NullFormatter())
axes[0].legend(fontsize=7,frameon=False)
axes[1].set(xlabel=r'Error in $\langle H_1\rangle/N$',ylabel=r'Error in $J_2\langle H_2\rangle/N$',title='Unsplit exchange improves competing bonds')
axes[1].ticklabel_format(axis='both',style='sci',scilimits=(0,0))
fig.suptitle('Triangular 4x4, J2/J1=0.125; fixed physical operator ablation')
fig.savefig(out/'fig57_channel_mechanism_physics.png',dpi=180)
fig.savefig(out/'fig57_channel_mechanism_physics.pdf')
plt.show()
for coupling,case in mechanism['cases'].items():
    print('\nJ2/J1',coupling)
    for name,row in case['arms'].items():
        print(name,'rank',row['rank'],'E/N',row['energy_per_site'],'Var',row['variance_total'],
              'E minus same-rank K',row['energy_minus_matched_krylov_per_site'],
              'paired scramble wins',row['scramble_joint_wins'],'/5')
"""), markdown(r"""
### Decision: retain physical exchange coherence; do not scale this fixed bank

At J2/J1=0.125, pure multiplicity dimer changes K2 energy by only
-7.41e-6/site and worsens both one- and two-cell RDM errors. The unsplit
exchange lowers energy by -1.81e-4/site and reduces the two-cell RDM error
from 0.02090 to 0.00478. Its rank-5 state is better than either separated
exchange component at rank 6. This is an operational complementarity result,
not a unique decomposition of quantum interference or evidence for a QSL.

All 24 physical arms lose in energy to same-rank Krylov. The literal spectral
pinching costs 576 pair actions per mode and input vector, versus 16 for the
unsplit exchange (before any shared-input or kernel optimization). Thus this
is a diagnostic, not an efficient ansatz. Isospectral scramble wins alone are
insufficient; nearby high-rank Krylov is already nearly exact in this cluster.

Do not prescribe a q-first or multiplicity-first curriculum from an RDM
decomposition. Retain complete physical exchange processes. Any next neural
claim must test environment-dependent reduced coefficients / correlations
beyond this fixed two-cell bank, with non-saturated capacity and measured
action cost. No additional fixed heads or remote production are authorized
by these data. The paper notebook is unchanged.
""")]
    for cell in cells:
        cell.metadata['experiment_section'] = 'channel_mechanism_20260927'
    return cells


def exchange_environment_cells():
    """Environment-modulated complete exchange, with same-rank controls."""
    cells = [markdown(r"""
## What surrounding information modulates complete exchange? (2026-09-27)

No q-first or multiplicity-first curriculum is imposed. On triangular 4x4
PBC, keep exact $S=0,\Gamma-A_1$ and the entire rank-3 K2 anchor. The primary
exchange is the preregistered collective nonuniform mode
$O=(O_1+O_2+O_3)/\sqrt3$. Add $O|\psi_{K2}\rangle$ and then one
$O F|\psi_{K2}\rangle$: the maximum rank is **5 of 14**, not saturated.
The three resolved modes are separate robustness tests, never selected as a
best mode after the experiment.

The coefficient is an affine **operator** on two disjoint spectator cells:
$c(F)=\alpha+\beta F$. The three registered features are spin content
$(Q_C^2+Q_D^2)/12$, spin pair $Q_C^2Q_D^2/36$, and dimer correlation
$F_D=[R_0(C)R_0(D)+R_1(C)R_1(D)]/6$. Here R0/R1 are the two traceless
combinations of the three four-site dimer matchings. The dimer feature is a
coherent multiplicity operator, not a function of classical magnetic bits.

Optimize coefficients by Ritz energy, without exact-state targets. Compare
against constant exchange, adding $F$ **alone** instead of $OF$, rank-5
Krylov (K4), and five rotations fixing the full constant-exchange space.
Form $OF$ **before** A1 compression: $POFP\ne(POP)(PFP)$.
The final state is exactly projected; the intermediate operator factors need
not each stay in A1. This is a low-rank mechanism oracle, not trained ViT or
a sampled cost comparison. All energies are finite-cluster quantities.
"""), code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from su2_nqs.frontier.artifacts import canonical_sha256
root = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p/'su2_nqs').is_dir())
project = root/'projects/clebsch_tree_nqs'
environment = json.loads((project/'results/spin_exchange_environment_v1/results.json').read_text())
assert canonical_sha256({k:v for k,v in environment.items() if k!='result_sha256'}) == environment['result_sha256']
out = project/'figures/exploration'
out.mkdir(parents=True,exist_ok=True)
couplings = np.array([float(x) for x in environment['cases']])
cases = list(environment['cases'].values())
primary = [c['modes']['collective'] for c in cases]
styles = {'spin_content':('Spin content','#0072B2','o'),
          'spin_pair':('Spin pair','#E69F00','s'),
          'dimer_correlation':('Dimer correlation','#009E73','D')}
fig,axes=plt.subplots(1,2,figsize=(10.8,5.3),constrained_layout=True)
e0=np.array([c['exact_energy_per_site'] for c in cases])
for feature,(label,color,marker) in styles.items():
    rows=[m['environments'][feature] for m in primary]
    axes[0].plot(couplings,np.array([r['energy_per_site'] for r in rows])-e0,color=color,marker=marker,label=label)
    axes[1].plot(couplings,[r['energy_minus_additive_environment_per_site']/1e-5 for r in rows],color=color,marker=marker,label=label)
axes[0].plot(couplings,np.array([m['constant']['energy_per_site'] for m in primary])-e0,'k:',label='Constant exchange (rank 4)')
axes[0].plot(couplings,np.array([m['environments']['dimer_correlation']['matched_krylov']['energy_per_site'] for m in primary])-e0,'k--',marker='+',label='K4 (rank 5)')
# An important pre-existing alternative: all three unsplit modes together.
previous=json.loads((project/'results/spin_channel_mechanism_v1/results.json').read_text())
assert canonical_sha256({k:v for k,v in previous.items() if k!='result_sha256'}) == previous['result_sha256']
old_rows=[previous['cases'][str(j)]['arms']['unsplit_exchange'] for j in couplings]
axes[0].plot(couplings,np.array([r['energy_per_site'] for r in old_rows])-e0,color='#882255',ls='-.',marker='^',label='Previous 3-mode exchange (rank 5)')
axes[0].set(yscale='log',ylabel='Energy error / site',title='Primary collective exchange; no new energy record')
axes[1].axhline(0,c='black',lw=.8)
axes[1].set(ylabel=r'$(E_{OF}-E_{F\ alone})/N$  [$10^{-5}$]',title='Negative: modulation beats independent environment')
for ax in axes:
    ax.set(xlabel='J2/J1',xticks=couplings)
    ax.grid(alpha=.15)
axes[0].legend(fontsize=7,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.2),ncol=2)
axes[1].legend(fontsize=8,frameon=False)
fig.suptitle('Triangular 4x4: dimer environment helps in the frustrated tests',fontsize=12)
fig.savefig(out/'fig58_exchange_environment_energy.png',dpi=180)
fig.savefig(out/'fig58_exchange_environment_energy.pdf')
plt.show()
print('Sealed result:',environment['result_sha256'])
print('All exact operator checks:',environment['diagnostics']['checks'])
"""), code(r"""
fig,axes=plt.subplots(1,3,figsize=(12.5,4.1),constrained_layout=True)
dimer=[m['environments']['dimer_correlation'] for m in primary]
for label,key,color,marker in [('Constant exchange','constant','#777777','o'),
                             ('Dimer-modulated exchange','modulated','#009E73','D'),
                             ('Exact finite cluster','exact','black','*')]:
    axes[0].plot(couplings,[r['correlation_'+key]['connected_OF'] for r in dimer],color=color,marker=marker,label=label)
axes[0].set(ylabel=r'$\langle O F_D\rangle-\langle O\rangle\langle F_D\rangle$',title='Connected exchange–environment correlation')
axes[0].legend(fontsize=7,frameon=False)
for label,rows,color,marker in [('Constant exchange',[m['constant'] for m in primary],'#777777','o'),
                               ('Dimer modulation',dimer,'#009E73','D'),
                               ('Dimer alone',[r['environment_only_addition'] for r in dimer],'#D55E00','s'),
                               ('K4',[r['matched_krylov'] for r in dimer],'black','+')]:
    axes[1].plot(couplings,[r['physics']['two_cells']['errors_to_exact']['trace_distance'] for r in rows],color=color,marker=marker,label=label)
    axes[2].plot(couplings,[r['variance_total'] for r in rows],color=color,marker=marker,label=label)
axes[1].set(ylabel='Two-cell RDM trace distance',title='Two-cell reduced density matrix')
axes[2].set(ylabel='Total energy variance',yscale='log',title='K4 remains stronger')
axes[1].legend(fontsize=7,frameon=False)
for ax in axes:
    ax.set(xlabel='J2/J1',xticks=couplings)
    ax.grid(alpha=.15)
fig.suptitle('Beyond energy: collective exchange with coherent dimer environment',fontsize=12)
fig.savefig(out/'fig59_exchange_environment_physics.png',dpi=180)
fig.savefig(out/'fig59_exchange_environment_physics.pdf')
plt.show()
print('Primary affine coefficients in the fixed convention; NOT probabilities or effective Hamiltonian couplings:')
for j,r in zip(couplings,dimer):
    fit=r['affine_coefficient_fit']
    print(j,'alpha',fit['alpha'],'beta',fit['beta'],'unique',fit['unique_with_fixed_carrier_phase'],
          'joint wins vs five scrambles',r['scramble_joint_wins'])
print('\nComplete registered inventory (no best-mode selection):')
for j,case in environment['cases'].items():
    for name,mode in case['modes'].items():
        for feature,r in mode['environments'].items():
            print(j,name,feature,'rank',r['rank'],'E/N',r['energy_per_site'],'Var',r['variance_total'],
                  'delta E vs same-rank K',r['energy_minus_matched_krylov_per_site'],
                  'E+Var win vs independent F',r['wins_additive_energy_and_variance'],
                  'scramble joint wins',r['scramble_joint_wins'])
"""), markdown(r"""
### Decision: an environment signal, not a variational-frontier win

At J2/J1=0.125 and 0.175, the **preregistered collective** dimer-modulated
exchange beats its constant baseline, independent-F addition, and all five
same-rank constant-space-fixing rotations in both energy and variance. Its
two-cell RDM errors fall from 0.01352 to 0.01082 and from 0.01353 to 0.00819.
The connected exchange–dimer covariance also moves toward the exact value.
At J2=0 the same dimer feature is not superior to F alone and worsens the RDM
slightly. Spin-content information alone is not an adequate modulation here.

With the fixed feature normalization and carrier-overlap phase, the dimer
coefficient $\beta$ is +0.1083, -0.3280, -0.5051 at the three couplings.
These are identifiable variational coordinates in this chosen affine family,
**not** measured probabilities, an effective exchange coupling, or a unique
decomposition of the exact wavefunction. They suggest that the relevant
correction couples exchange to surrounding dimer structure, rather than
requiring a q-first/multiplicity-first optimization schedule.

All 36 modulated trials lose to rank-5 Krylov in energy. The primary trials
also lose to the previously tested rank-5 space containing all three unsplit
exchange modes. Only 7/36 trials beat the independent-environment control in
both energy and variance. Five scramble realizations on one finite cluster
are a bounded diagnostic, not a statistical discovery or evidence of QSL.
There is no new lowest-energy, efficiency, scale, or PRL claim.

Next test **one** residualized environment direction beyond the already
strong three-mode exchange space, with a same-rank Krylov and fixed-anchor
scramble. Hold the dimer feature fixed and check a second cell partition;
reject a geometry-specific claim if the signal depends on the selected
partition. A future neural coefficient must be an SU(2)-scalar operator on
reduced data, not multiplication by an arbitrary spin-bit function. Do not
expand a bank until it spans all 14 states. Remote production and the paper
notebook remain unchanged.
""")]
    for cell in cells:
        cell.metadata['experiment_section'] = 'exchange_environment_20260927'
    return cells


def environment_complement_cells():
    """Strong-anchor complementarity and bounded N36 CPU feasibility."""
    cells=[markdown(r"""
## Does the environment add information beyond all three exchanges? (2026-09-27)

Keep the entire $B=\mathrm{span}\{\pi,H\pi,H^2\pi,O_1\psi_{K2},
O_2\psi_{K2},O_3\psi_{K2}\}$, whose actual rank is 5. Remove its component
from the **same** preregistered collective dimer direction:
$r=(I-P_B)P_{A_1}OF_D|\psi_{K2}\rangle$. Optimize in $B\oplus r$, rank 6
of 14. Compare independent $F_D$ addition, rank-6 Krylov (K5), and five fresh
rotations fixing the whole B. The original partition and one prespecified
relative spectator shift are both reported, with no best-partition selection.

The environment retains 7.0–13.6% of its **squared norm** outside B. It lowers
energy and variance and improves the two-cell RDM in all six finite-cluster
tests. At J2=0.125 and 0.175 it also beats independent-F addition for both
partitions. It still loses to K5 in every case; scramble wins are 2/5,4/5
for the primary frustrated tests and 1/5,4/5 for the shifted ones. This is
real but modest additional correlation, not a new competitive energy result.
"""),code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from su2_nqs.frontier.artifacts import canonical_sha256
root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs'
def sealed_result(path):
    result=json.loads(path.read_text())
    assert canonical_sha256({k:v for k,v in result.items() if k!='result_sha256'})==result['result_sha256']
    return result
complement=sealed_result(project/'results/spin_environment_complement_v1/results.json')
out=project/'figures/exploration'; out.mkdir(parents=True,exist_ok=True)
jvalues=np.array(complement['config']['couplings'])
original=[complement['cases'][f'original/j{j}'] for j in jvalues]
shifted=[complement['cases'][f'spectator_row_shift/j{j}'] for j in jvalues]
e0=np.array([r['exact_energy_per_site'] for r in original])
fig,axes=plt.subplots(1,3,figsize=(12.4,4.4),constrained_layout=True)
for label,rows,color,marker in [('Strong 3-mode anchor',[r['base'] for r in original],'#777777','o'),
                               ('+ dimer environment',original,'#009E73','D'),
                               ('+ shifted environment',shifted,'#0072B2','s'),
                               ('K5 (rank 6)',[r['matched_krylov'] for r in original],'black','+')]:
    axes[0].plot(jvalues,np.array([r['energy_per_site'] for r in rows])-e0,color=color,marker=marker,label=label)
    axes[2].plot(jvalues,[r['physics']['two_cells']['errors_to_exact']['trace_distance'] for r in rows],color=color,marker=marker,label=label)
for label,rows,color,marker in [('Original environment',original,'#009E73','D'),('Shifted environment',shifted,'#0072B2','s')]:
    axes[1].plot(jvalues,[r['hamiltonian_residual_capture_fraction'] for r in rows],color=color,marker=marker,label=label)
axes[0].set(yscale='log',ylabel='Energy error / site',title='Extra information; still above K5')
axes[1].set(ylabel='Fraction of residual norm squared captured',title='Overlap with missing Hamiltonian direction')
axes[2].set(ylabel='Two-cell RDM trace distance',title='Physical RDM improvement')
axes[0].legend(fontsize=7,frameon=False)
axes[1].legend(fontsize=7,frameon=False)
for ax in axes:
    ax.set(xlabel='J2/J1',xticks=jvalues)
    ax.grid(alpha=.15)
fig.suptitle('Triangular 4x4: one residualized environment direction beyond the strong exchange space')
fig.savefig(out/'fig60_environment_beyond_strong_exchange.png',dpi=180)
fig.savefig(out/'fig60_environment_beyond_strong_exchange.pdf')
plt.show()
print('Result seal:',complement['result_sha256'])
for case,row in complement['cases'].items():
    print(case,'outside norm fraction',row['outside_norm_fraction'],
          'gain/site',row['energy_change_from_strong_per_site'],
          'variance',row['base']['variance_total'],'->',row['variance_total'],
          'same-rank K energy difference',row['energy_minus_matched_krylov_per_site'],
          'scramble joint wins',row['scramble_joint_wins'])
"""),markdown(r"""
### Local 6x6 calculation: completed, but no trained-state energy comparison

The 512-sample timing pilot is followed by 16,384 **new** audit samples from
128 persistent chains, with the carrier/operators/proposal calibration fixed.
The carrier is the analytic three-determinant exact-$S=0,\Gamma-A_1$ pi/0-flux
state, **not** K2 or the optimized three-mode state. There is no ED or training.

Use a full-support proposal $q(\sigma)\propto|\psi(\sigma)|^2+\kappa$, with
kappa fixed from independent uniform configurations. This does not floor or
alter physical amplitudes. Full-support importance moments include carrier
nodes, which can contribute to $|H\psi|^2$ and $\langle H\psi|O\psi\rangle$.
Variance and gradients from carrier-only samples can miss these terms.

At an A1 carrier, $\langle H\psi|P_{A_1}O\psi\rangle=
\langle H\psi|O\psi\rangle$. Thus local actions give the first derivative
for a symmetry-averaged correction without 432-image averaging. This identity
does **not** give the projected tangent metric or finite-step dressed energy.
All figures below are diagnostics of the untrained carrier. The chain-level
errors include snapshot correlation, but do not certify mixing or eliminate
finite-sample self-normalization bias.
"""),code(r"""
preflight=sealed_result(project/'results/spin_environment_preflight_v1/results.json')
audit_dir=project/'results/spin_environment_preflight_v1/audit'
audit=sealed_result(audit_dir/'results.json')
fig,axes=plt.subplots(1,2,figsize=(10.5,4.4),constrained_layout=True)
energies=[]; errors=[]
for row in audit['replicates']:
    with np.load(audit_dir/f"replicate_{row['index']}.npz") as d:
        norm=np.abs(d['psi'])**2/d['proposal_squared']
        numerator=(d['psi'].conj()*d['hpsi']/d['proposal_squared']).real
        nchains=audit['config']['chain_count']; nblocks=audit['config']['measurement_blocks']
        a=numerator.reshape(nblocks,nchains).mean(axis=0)
        b=norm.reshape(nblocks,nchains).mean(axis=0)
        e=np.sum(a)/np.sum(b)
        se=np.std(a-e*b,ddof=1)/np.sqrt(nchains)/b.mean()
        energies.append(e/36); errors.append(se/36)
axes[0].errorbar(np.arange(1,5),energies,yerr=errors,fmt='o',color='#0072B2',capsize=4,label='Independent streams; chain SE')
mean=audit['aggregate']['energy_total']/36
se=audit['aggregate']['chain_bootstrap_se']['energy_total']/36
axes[0].axhline(mean,c='black',lw=1,label='Pooled carrier energy')
axes[0].axhspan(mean-se,mean+se,color='black',alpha=.1)
axes[0].set(xlabel='Audit stream',xticks=(1,2,3,4),ylabel='Carrier energy / site',title='6x6: immutable projected pi/0-flux carrier')
axes[0].legend(fontsize=7,frameon=False)
labels=[r'$\overline{O_1}$',r'$\overline{O_2}$',r'$\overline{O_3}$',r'$\overline{F_D}$',r'$\overline{O F_D}$']
for r,offset,color,marker,label in [(preflight,-.10,'#999999','o','512-sample pilot'),(audit,.10,'#009E73','D','16,384-sample audit')]:
    m=r['aggregate']
    values=np.array(m['energy_derivative_total'])/36/1e-3
    stderr=np.array(m['chain_bootstrap_se']['energy_derivative_total'])/36/1e-3
    axes[1].errorbar(values,np.arange(5)+offset,xerr=stderr,fmt=marker,color=color,capsize=3,label=label)
axes[1].axvline(0,c='black',lw=.8)
axes[1].set(yticks=np.arange(5),yticklabels=labels,xlabel=r'$d(E/N)/d\epsilon$  [$10^{-3}$]',title='First derivatives, not variational energy gains')
axes[1].legend(fontsize=7,frameon=False)
fig.suptitle('Local CPU feasibility: environment slope remains statistically unresolved')
fig.savefig(out/'fig61_n36_environment_local_preflight.png',dpi=180)
fig.savefig(out/'fig61_n36_environment_local_preflight.pdf')
plt.show()
print('Audit result:',audit['result_sha256'])
print('Energy/site and chain SE:',mean,se)
print('Total variance and chain SE:',audit['aggregate']['variance_total'],audit['aggregate']['chain_bootstrap_se']['variance_total'])
print('Wall seconds: pilot/audit',preflight['wall_seconds_this_invocation'],audit['wall_seconds_this_invocation'])
print('Importance ESS fraction:',audit['aggregate']['importance_ess_fraction'])
print('Projected-environment slope/site and SE:',audit['aggregate']['energy_derivative_total'][-1]/36,audit['aggregate']['chain_bootstrap_se']['energy_derivative_total'][-1]/36)
"""),markdown(r"""
### Interpretation and next boundary

Small cluster: the environment direction is not redundant with the strong
three-mode space. At J2=0.125 it lowers energy by 7.58e-6/site and variance
by about 20%, while capturing 22% of the remaining Hamiltonian residual norm
squared. A shifted environment retains a smaller benefit. This still loses
to same-rank Krylov, so it is a modest correlation mechanism, not a frontier.

Large cluster: CPU feasibility is established (47.1 s for the fixed audit;
not a hardware-independent timing guarantee). Carrier energy is
-0.5077489 +/- 0.0001914/site. The environment slope is
(-2.39 +/- 1.84)e-4/site; its sign is unresolved and changes between the
small pilot and audit. Do not claim energy lowering, greater large-size gain,
or a SU(2)-versus-ViT advantage from this. High importance ESS alone does not
certify mixing or the stability of the operator-gradient estimator.

The next meaningful local task is a **finite-step exact-A1 amplitude and
normalization** test of a few fixed scalar directions, retaining a strong
carrier and independent full-support audits. The present first-derivative
shortcut cannot replace that evaluator or establish a strong-carrier
complement at N36. Start with measured tiny batches; do not multiply the
47-second timing into a speculative neural-training/GPU forecast. No remote
production is released; these figures remain in the exploration notebook.
""")]
    for cell in cells:
        cell.metadata['experiment_section']='environment_complement_20260927'
    return cells


def finite_environment_cells():
    """Actual finite-step A1 states, not a projected first-derivative shortcut."""
    cells=[markdown(r"""
## Finite-step strict-A1 environment dressing on 6x6 (2026-09-27)

The state is now $\phi_\epsilon=\psi_\pi+\epsilon\bar A\psi_\pi$, with
$A=OF_D$ and $\bar A=|G|^{-1}\sum_g U_g A U_g^{-1}$, using **all 432**
space-group elements. Every amplitude is in $S=0,\Gamma-A_1$. Factors O and
F are multiplied before projection. This is a finite coherent state, not the
linearized energy formula from the previous experiment.

Repeated conjugate permutations are combined once. On the analytic
closed-shell carrier, disjoint swaps use exact low-rank determinant updates;
ill-conditioned components fall back to direct determinants. The shortcut
is independently tested, including exchange signs. Neither the projection
nor the Hamiltonian is sampled or truncated.

Exact N16 checks validate amplitudes, singlet structure, and full-support
normalization. At N36, direct versus low-rank amplitude/H-action replay and
Tx, Ty, C6, mirror invariance are checked before sampling. Four new streams
then estimate actual finite-state energy, variance, relative normalization,
and carrier fidelity with a full-support proposal. Absolute normalization
is not known analytically at N36. The quadratic norm is included in every
energy/variance calculation.

The epsilon grid (-1,-0.5,-0.25,0,0.25,0.5,1) and primary epsilon=+0.5 were
fixed before these new samples, using earlier exploratory information. All
steps and streams are reported, with no post-hoc choice of the best step.
The reference is the **zero-Lanczos** carrier, not optimized K2 or a ViT.
"""),code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from su2_nqs.frontier.artifacts import canonical_sha256
root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs'
directory=project/'results/spin_finite_environment_dressing_v1'
finite=json.loads((directory/'results.json').read_text())
assert canonical_sha256({k:v for k,v in finite.items() if k!='result_sha256'})==finite['result_sha256']
out=project/'figures/exploration';out.mkdir(parents=True,exist_ok=True)
rows=finite['aggregate'];eps=np.array([r['epsilon'] for r in rows])
fig,axes=plt.subplots(1,3,figsize=(12.3,4.2),constrained_layout=True)
for ax,key,factor,label in zip(axes,('norm_ratio','energy_change_total','variance_change_total'),(1,1/36,1/36),
        ('Norm relative to carrier','Paired energy change / site','Paired variance change / site')):
    values=np.array([r[key] for r in rows])*factor
    errors=np.array([r['paired_chain_bootstrap_se'][key] for r in rows])*factor
    ax.errorbar(eps,values,yerr=errors,color='#0072B2',fmt='o-',capsize=3)
    ax.axhline(1 if key=='norm_ratio' else 0,c='black',lw=.8)
    ax.axvline(finite['config']['primary_epsilon'],c='#009E73',ls=':',lw=1,label='Registered primary step')
    ax.set(xlabel=r'$\epsilon$',ylabel=label,xticks=(-1,-.5,0,.5,1))
    ax.grid(alpha=.15)
axes[0].legend(fontsize=7,frameon=False)
axes[0].set_title('Quadratic normalization is retained')
axes[1].set_title('Actual finite-state energy')
axes[2].set_title('Full-H finite-state variance')
fig.suptitle('Triangular 6x6, J2/J1=0.125: fixed strict-SU(2)/A1 environment correction')
fig.savefig(out/'fig62_finite_A1_environment_steps.png',dpi=180)
fig.savefig(out/'fig62_finite_A1_environment_steps.pdf')
plt.show()
print('Result seal:',finite['result_sha256'])
print('N16 exact checks:',finite['n16_validation']['checks'])
print('N36 exact projection / low-rank replay checks:',finite['n36_validation'])
print('Bank:',finite['bank_cost'])
print('Measured direct/low-rank H seconds per validation center:',finite['direct_H_seconds_one_center'],finite['low_rank_H_seconds_one_center'])
for r in rows:
    print('epsilon',r['epsilon'],'E/N',r['energy_total']/36,'Var/N',r['variance_total']/36,
          'Delta E/N',r['energy_change_total']/36,'paired SE',r['paired_chain_bootstrap_se']['energy_change_total']/36,
          'Delta Var/N',r['variance_change_total']/36,'paired SE',r['paired_chain_bootstrap_se']['variance_change_total']/36,
          'norm',r['norm_ratio'],'fidelity',r['carrier_fidelity'],'importance ESS',r['importance_ess_fraction'])
"""),code(r"""
from su2_nqs.benchmarks.finite_scalar_dressing import finite_step_moments
primary=finite['config']['primary_epsilon']
fig,axes=plt.subplots(1,2,figsize=(9.5,4),constrained_layout=True)
for stream in finite['replicates']:
    with np.load(directory/f"replicate_{stream['index']}.npz") as bundle:
        arrays={k:bundle[k] for k in ('psi','direction','hpsi','hdirection','proposal_squared')}
    names=tuple(arrays)
    summary=finite_step_moments(*(arrays[k] for k in names),primary)
    nc=finite['config']['chain_count'];nb=finite['config']['measurement_blocks']
    indices=np.arange(nc*nb).reshape(nb,nc).T
    rng=np.random.default_rng(finite['config']['bootstrap_seed']+100+stream['index'])
    boot=[]
    for _ in range(300):
        ix=indices[rng.integers(0,nc,nc)].ravel()
        m=finite_step_moments(*(arrays[k][ix] for k in names),primary)
        boot.append([m['energy_change_total']/36,m['variance_change_total']/36])
    se=np.std(boot,axis=0,ddof=1)
    for axis,key,error in zip(axes,('energy_change_total','variance_change_total'),se):
        axis.errorbar(stream['index']+1,summary[key]/36,yerr=error,fmt='o',color='#0072B2',capsize=4)
pooled=next(r for r in rows if r['epsilon']==primary)
for ax,key,label in zip(axes,('energy_change_total','variance_change_total'),('Paired energy change / site','Paired variance change / site')):
    value=pooled[key]/36;error=pooled['paired_chain_bootstrap_se'][key]/36
    ax.axhline(0,c='black',lw=.8)
    ax.axhline(value,c='#009E73',lw=1)
    ax.axhspan(value-error,value+error,color='#009E73',alpha=.12)
    ax.set(xlabel='Independent audit stream',xticks=(1,2,3,4),ylabel=label)
fig.suptitle(f'Registered primary epsilon={primary}: independent streams, paired chain errors')
fig.savefig(out/'fig63_finite_A1_environment_independent_streams.png',dpi=180)
fig.savefig(out/'fig63_finite_A1_environment_independent_streams.pdf')
plt.show()
print('Primary joint resolved decrease (diagnostic, not a discovery p-value):',
      pooled['energy_change_total']+2*pooled['paired_chain_bootstrap_se']['energy_change_total']<0 and
      pooled['variance_change_total']+2*pooled['paired_chain_bootstrap_se']['variance_change_total']<0)
print('Samples/chains:',finite['sample_count'],finite['independent_chain_count'])
print('Wall time:',finite['wall_seconds_this_invocation'])
print('Historical baseline tail diagnostic, NOT pooled with this audit:')
for subdir in ('spin_environment_preflight_v1/audit','spin_finite_environment_dressing_v1'):
    source=project/'results'/subdir
    historical=json.loads((source/'results.json').read_text())
    assert canonical_sha256({k:v for k,v in historical.items() if k!='result_sha256'})==historical['result_sha256']
    blocks=[]
    for file in sorted(source.glob('replicate_*.npz')):
        with np.load(file) as bundle:
            blocks.append({k:bundle[k] for k in ('psi','hpsi','proposal_squared')})
    p,h,q=[np.concatenate([b[k] for b in blocks]) for k in ('psi','hpsi','proposal_squared')]
    norm=np.sum(abs(p)**2/q);energy=np.sum(p.conj()*h/q).real/norm
    contributions=abs(h-energy*p)**2/q
    count=max(1,int(np.ceil(.01*len(p))))
    print(subdir,'samples',len(p),'total carrier variance',contributions.sum()/norm,
          'fraction from largest 1% of residual contributions',np.sort(contributions)[-count:].sum()/contributions.sum())
"""),markdown(r"""
### Claim boundary

These are finite-step amplitudes and actual full-H energy/variance estimates,
not an inference from the sign of a first derivative. The exact state keeps
both symmetries at every listed step. Normalization at N36 remains sampled,
and the small number of independent chains can leave rare-node contributions
or equilibration bias unresolved. Paired errors benefit from a common
proposal, but do not remove those systematic risks.

**Observed outcome:** at the registered $\epsilon=0.5$, all four streams
lower energy and variance. The pooled changes are
$\Delta E/N=-4.34(2.23)\times10^{-5}$ and
$\Delta\mathrm{Var}(H)/N=-7.45(3.54)\times10^{-5}$: only 1.94 and 2.11
paired SE, respectively. This is an encouraging finite-state signal, not a
resolved joint win. In particular, the carrier's total variance is 0.392
here versus 0.810 in the older 16,384-sample audit; the latter's largest 1%
of residual contributions carry 38.36% of its variance. The present small
bootstrap cannot account for unobserved tails. Do not confuse the per-site
variance plotted above with the variance of $H/N$, which divides by $N^2$.

Even a favorable step here would only improve one zero-Lanczos carrier;
there is no matched Krylov/ViT or GPU-efficiency comparison. All finite steps
were fixed in advance and displayed. Do not select the lowest audit energy
and call it an independent win. Remote production and the paper notebook
remain unchanged. Detailed outcome and the next decision are in the sealed
experiment's REPORT.md.
""")]
    for cell in cells:cell.metadata['experiment_section']='finite_environment_20260927'
    return cells


def full_vit_budget_cells():
    """Full-wavefunction capacity/optimization and bounded N36, not paper claims."""
    cells = [markdown(r"""
## Full-wavefunction ViT baseline and bounded N36 pilot (2026-09-28)

Fixed OF extensions and additional audits are **frozen**. These runs reuse the
complete factored Spatial-ViT and train all weights, with no carrier or small
residual head. They impose fixed $S^z=0$ and exact lattice $\Gamma-A_1$, **not
exact SU(2)**. Both capacities have two layers and four heads. Every seed is
shown; no seed spread is described as Monte Carlo uncertainty.

N16: 2000 Adam + 500 L-BFGS iterations, 90 active-evaluation-second limit per
stage, cached ED scalar references, no new eigensolve. The actual runs finish
their step budgets within the caps. Larger width does not perform better.
Longer training strongly improves the smaller model; neither its convergence
nor a literature-frontier reproduction is established. Old translation-only
runs are not a paired comparison to this A1 protocol.

N36: fresh random initialization, 32 chains, 64 steps or 180 training seconds,
and independent initial/final audits. Exact patch-translation cosets reduce
spatial images to 48 (triangular) / 32 (square). This is a short **feasibility
pilot**, not convergence, a symmetry win, or permission for remote production.
"""), code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from su2_nqs.frontier.artifacts import canonical_sha256, atomic_json
root = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p/'su2_nqs').is_dir())
project = root/'projects/clebsch_tree_nqs'
folder = project/'results/spin_full_vit_budget_v1'
trials = [json.loads(p.read_text()) for p in sorted(folder.glob('*/results.json'))]
assert len(trials)==8
for trial in trials:
    assert trial['result_sha256']==canonical_sha256({k:v for k,v in trial.items() if k!='result_sha256'})
controls = json.loads((project/'results/spin_strong_carrier_innovation_v1/results.json').read_text())
assert controls['result_sha256']==canonical_sha256({k:v for k,v in controls.items() if k!='result_sha256'})
k2 = controls['cases']['original/j0.125/K2']['states']['carrier']
k4case = controls['cases']['original/j0.125/K4']['states']
krylov = {'K2':k2, 'K4':k4case['carrier'], 'K5':k4case['rank_matched_direct_krylov'],
          'K6':k4case['rank_matched_conditional_krylov']}
summary = dict(task='spin_full_vit_budget_summary', trials=[dict(trial=t['trial'],
    result_sha256=t['result_sha256'], parameters=t['parameter_count'],metrics=t['metrics'],
    spin=t['spin'],wall_seconds=t['wall_seconds']) for t in trials],
    cached_triangular_krylov=krylov, krylov_source_seal=controls['result_sha256'],
    production_released=False, paper_claim=False)
summary['result_sha256']=canonical_sha256(summary)
atomic_json(folder/'summary.json',summary)
out=project/'figures/exploration';out.mkdir(parents=True,exist_ok=True)
colors={'d16_l2':'#0072B2','d32_l2':'#D55E00'}
fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for col,geometry in enumerate(('triangular','square')):
    for trial in [t for t in trials if t['case']['geometry']==geometry]:
        hist=trial['training']['history']; color=colors[trial['arm']]
        label=f"{trial['parameter_count']} params, seed {str(trial['seed'])[-2:]}"
        style='-' if trial['seed']==2026092901 else '--'
        for ax,key in zip(axes[:,col],('energy_error_per_site','variance_per_site')):
            ax.plot([h['evaluations'] for h in hist],[max(h[key],1e-14) for h in hist],
                    color=color,ls=style,lw=1.1,label=label)
            ax.set_yscale('log');ax.grid(alpha=.15)
        adam=next(h for h in hist if h['stage']=='adam_terminal')
        axes[0,col].scatter(adam['evaluations'],adam['energy_error_per_site'],marker='|',s=70,c=color)
    axes[0,col].set_title(f'{geometry.capitalize()} 4x4 PBC')
    axes[1,col].set_xlabel('Exact objective evaluations (line searches counted)')
    axes[0,col].legend(fontsize=7,frameon=True,facecolor='white',framealpha=.95)
for name,row in krylov.items():
    axes[0,0].axhline(row['energy_error_per_site'],c='0.5',lw=.6,ls=':')
    axes[0,0].text(2600,row['energy_error_per_site'],name,fontsize=7,va='bottom')
axes[0,0].set_ylabel('Energy error / site');axes[1,0].set_ylabel('Var(H) / site')
fig.suptitle('Full ViT: all layers trained; vertical ticks mark end of Adam\nKrylov lines are cached references, not matched-work competitors',fontsize=11)
fig.savefig(out/'fig72_full_vit_training.png',dpi=180);fig.savefig(out/'fig72_full_vit_training.pdf');plt.show()
"""), code(r"""
fig,axes=plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
for geometry,marker in (('triangular','o'),('square','s')):
    for trial in [t for t in trials if t['case']['geometry']==geometry]:
        color=colors[trial['arm']]; err=trial['metrics']['energy_error_per_site']
        axes[0].scatter(trial['wall_seconds'],err,c=color,marker=marker,s=40)
        spin=trial['spin']; projected=spin['singlet_projected_energy_error_per_site']
        axes[1].plot([0,1],[err,max(projected,1e-14)],c=color,marker=marker,lw=.8,alpha=.75)
        axes[2].scatter(1-spin['singlet_weight'],spin['fraction_of_raw_energy_error_explained_by_spin_projection'],
                        c=color,marker=marker,s=40)
axes[0].set(xlabel='Total trial wall time (CPU seconds)',ylabel='Final energy error / site',yscale='log',
            title='Capacity is not yet exploitable')
axes[1].set(xticks=[0,1],xticklabels=['Raw fixed-Sz + A1','Post-project S=0'],yscale='log',
            ylabel='Energy error / site',title='Exact-small projection headroom')
axes[2].set(xscale='log',xlabel='Non-singlet probability',ylabel='Fraction of energy error removed',
            title='Small leakage, large energy cost')
fig.suptitle('Blue: 3992 parameters; orange: 15112. Circles: triangular; squares: square.\nPost-projection is a diagnostic, not scalable trained-SU(2) performance.',fontsize=10)
fig.savefig(out/'fig73_full_vit_capacity_spin.png',dpi=180);fig.savefig(out/'fig73_full_vit_capacity_spin.pdf');plt.show()
for t in trials:
    print(t['trial'],t['metrics'], 'singlet weight',t['spin']['singlet_weight'])
"""), markdown(r"""
### Initialization is near a high-spin, high-energy eigenstate

Replaying the saved initialization hashes gives 96.3%--99.2% probability in
the **maximal total-spin $S=8$ state** for N16. This is not singlet weight.
At fixed $M=0$ the maximal-spin vector is the normalized uniform vector,
so its weight is $|\sum_\sigma\psi(\sigma)|^2/\binom{16}{8}$ for normalized
$\psi$. The following cell also checks the independent sparse $S^2$ action.

This explains why low initial variance is not a ground-state quality target:
a high-energy eigenstate can also have zero variance. It does not by itself
explain the width dependence, or prove that an SU(2) training architecture
will be more efficient. No additional optimization or samples are used.
"""), code(r"""
import jax
from projects.clebsch_tree_nqs.workflows.spin import _prepare_case
from projects.clebsch_tree_nqs.workflows.spin_full_vit_budget import prepare_objective
from su2_nqs.models.fixed_sz_spatial_vit import FixedSzSpatialViTConfig, init_fixed_sz_spatial_vit
from su2_nqs.frontier.artifacts import pytree_sha256
initial_spin=[]
for geometry in ('triangular','square'):
    selected=[t for t in trials if t['case']['geometry']==geometry]
    data=_prepare_case(selected[0]['case'],'fixed_m0')
    for t in selected:
        cfg=FixedSzSpatialViTConfig(**t['model_config'])
        p=init_fixed_sz_spatial_vit(jax.random.PRNGKey(t['seed']),cfg)
        assert pytree_sha256(p)==t['initial_params_sha256']
        objective=prepare_objective(data,cfg)
        psi=objective['physical'](objective['state_fn'](p))
        row=dict(trial=t['trial'],maximal_spin_probability=float(abs(psi.sum())**2/len(psi)),
                 spin_squared_expectation=float(np.vdot(psi,data['spin_squared']@psi).real))
        initial_spin.append(row);print(row)
summary.pop('result_sha256',None)
summary['initial_spin_replay']=initial_spin
summary['result_sha256']=canonical_sha256(summary)
atomic_json(folder/'summary.json',summary)
"""), code(r"""
pilots=[json.loads((project/f'results/spin_full_vit_n36_pilot_v1/{g}/results.json').read_text())
        for g in ('triangular','square')]
references=json.loads((project/'references/finite_torus_6x6_targets.json').read_text())
fig,axes=plt.subplots(2,2,figsize=(11,6.5),constrained_layout=True)
for col,p in enumerate(pilots):
    assert p['result_sha256']==canonical_sha256({k:v for k,v in p.items() if k!='result_sha256'})
    h=p['history'];x=[v['active_seconds'] for v in h]
    axes[0,col].plot(x,[v['energy_before_update_per_site'] for v in h],c='#0072B2',lw=1,label='Training estimate')
    for label,t,a in [('Initial audit',0,p['initial_audit']),('Terminal audit',x[-1],p['final_audit'])]:
        axes[0,col].errorbar(t,a['energy_per_site'],yerr=2*a['chain_mean_standard_error'],
                           fmt='o',ms=4,c='black',capsize=3,label=label)
    key='triangular_6x6_torus' if p['geometry']=='triangular' else 'square_6x6_pbc'
    axes[0,col].axhline(references[key]['exact_energy_per_site'],c='0.4',ls='--',label='Cached finite-size ED reference')
    axes[1,col].plot([v['step'] for v in h],[v['seconds'] for v in h],c='#009E73')
    axes[0,col].set(title=f"{p['geometry'].capitalize()} 6x6, J2={p['j2_over_j1']}",xlabel='Active training time (s)',ylabel='Energy / site')
    axes[1,col].set(xlabel='Completed update',ylabel='Seconds / sampled update',
                  title=f"{p['amplitude_images']} A1 images, {p['parameter_count']} parameters")
    axes[0,col].legend(fontsize=7,frameon=False)
    print(p['geometry'],'wall',p['wall_seconds'],'seconds; full-H configuration calls',p['full_H_configuration_calls'])
    print('Untouched terminal audit:',p['final_audit'])
fig.suptitle('Short random-start feasibility only: 32 chains, 64 updates, no production release\nAudit bars: twice chain-mean SE; thermalization and asymptotic uncertainty are NOT certified',fontsize=10)
fig.savefig(out/'fig74_full_vit_n36_pilot.png',dpi=180);fig.savefig(out/'fig74_full_vit_n36_pilot.pdf');plt.show()
""")]
    for cell in cells:
        cell.metadata['experiment_section']='full_vit_budget_20260928'
    return cells


def decoder_pilot_cells(*, physical_initialization=False):
    """Bounded N36 training and independent audits; no small-size optimization."""
    cells=[markdown(r"""
## N36 strict decoder: bounded chain-health check and joint energy pilot

Same4492-parameter full-q local-Schur/CG ansatz, exact SU2 and full A1, original
faster H evaluator. **No new ViT/heads, q truncation, tile sweep or N16 work.**
Two lattices: triangular J2/J1=0.125 and square J2/J1=0.5. One inherited weight
initialization;16 chains per phase (8 uniform,8 ordered),32 proposals/block.
Fixed24 updates,32 centers/update, Adam3e-4 and empirical FS cap0.03. The cap
is measured on training samples and is not an untouched-state guarantee.

Parent and terminal audits have separate fresh RNG streams, disjoint from
training and from each other;256 energy centers each after1024 burn proposals.
Use the **terminal** checkpoint, never best noisy training energy. Plot all
training steps. Chain-based intervals remain conditional on equilibration;
short-chain Rhat/ESS and cohort comparisons are diagnostics, not certificates.
No competitive baseline, multiple weight seeds or GPU/PRL win is claimed.

Energy context only:18 disjoint nearest-neighbor singlets have E/N=-3/8 on
both Hamiltonians (every other bond expectation is zero). This product is
S=0 but **not lattice-A1 projected or work matched**. It is an analytic
reference, not a trained comparison arm. The parent here is the random N36
cost-test initialization, not a transferred high-quality N16 VAP state.
"""),code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from su2_nqs.frontier.artifacts import canonical_sha256
root=next(p for p in (Path.cwd(),*Path.cwd().parents)if(p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs';folder=project/'results/spin_decoder_pilot_v1'
results={g:json.loads((folder/g/'results.json').read_text())for g in ('triangular','square')}
for r in results.values():
    assert r['result_sha256']==canonical_sha256({k:v for k,v in r.items()if k!='result_sha256'})
    assert not r['equilibrium_certified'] and r['checkpoint_selection']=='fixed_terminal_not_best_energy'
out=project/'figures/exploration';out.mkdir(parents=True,exist_ok=True)
colors={'triangular':'#0072B2','square':'#D55E00'}
fig,axes=plt.subplots(2,2,figsize=(10,6.8),constrained_layout=True)
for col,(geometry,r)in enumerate(results.items()):
    h=r['history'];x=[row['step']for row in h]
    for ax,field in zip(axes[:,col],('energy_before_update_per_site','variance_before_update_per_site')):
        ax.plot(x,[row[field]for row in h],marker='.',color=colors[geometry],label='Training batch before update')
        ax.grid(alpha=.15);ax.set_xlabel('Fixed-budget update');ax.legend(fontsize=8)
    if not r['training_blocked']:
        for label,style in [('audit_parent','--'),('audit_terminal',':')]:
            s=r['summaries'][label]
            axes[0,col].axhline(s['energy_per_site'],color='0.3',ls=style,label=label.replace('audit_','Fresh audit: '))
        axes[0,col].legend(fontsize=8)
    axes[0,col].axhline(-.375,color='#777777',ls='-.',lw=1,label='S=0 NN dimers; no lattice projection')
    axes[0,col].legend(fontsize=7)
    axes[0,col].set_title(geometry.capitalize());axes[0,col].set_ylabel('E / N')
    axes[1,col].set_ylabel('Variance / N');axes[1,col].set_yscale('log')
fig.suptitle('N36 short pilot: noisy training trajectory is not an independent variational result')
fig.savefig(out/'fig98_decoder_n36_pilot.png',dpi=180);fig.savefig(out/'fig98_decoder_n36_pilot.pdf');plt.show()
"""),code(r"""
fig,axes=plt.subplots(1,2,figsize=(10,3.5),constrained_layout=True)
for y,(geometry,r)in enumerate(results.items()):
    if r['comparison'] is None:continue
    c=r['comparison']
    for ax,field,interval in zip(axes,('delta_energy_per_site','delta_variance_per_site'),
                                  ('delta_energy_bootstrap_95','delta_variance_bootstrap_95')):
        low,high=c[interval];value=c[field]
        ax.plot([low,high],[y,y],color=colors[geometry],lw=2)
        ax.plot(value,y,'o',color=colors[geometry])
for ax,label in zip(axes,('Terminal - parent: energy / N','Terminal - parent: variance / N')):
    ax.axvline(0,color='0.6',ls='--');ax.set_yticks([0,1],['Triangular','Square']);ax.set_ylim(-.6,1.6)
    ax.set_xlabel(label);ax.grid(axis='x',alpha=.15)
fig.suptitle('Independent audit streams; 95% chain-bootstrap intervals CONDITIONAL on equilibration',fontsize=10)
fig.savefig(out/'fig99_decoder_n36_audit.png',dpi=180);fig.savefig(out/'fig99_decoder_n36_audit.pdf');plt.show()
for geometry,r in results.items():
    print(geometry,'steps',r['completed_steps'],'H centers',r['full_H_centers'],'seconds',r['active_seconds'])
    for phase,s in r['summaries'].items():
        print(phase,'E/N',s['energy_per_site'],'chain SE',s['energy_chain_se'],'variance/N',s['variance_per_site'],
              'screening',s['screening_ok'],'Rhat',{k:v['split_rhat']for k,v in s['diagnostics'].items()},
              'cohort delta',s['cohort_energy_difference'],'cohort SE',s['cohort_energy_difference_se'])
    print('comparison',r['comparison'])
"""),code(r"""
fig,axes=plt.subplots(3,2,figsize=(10,8),constrained_layout=True)
for col,(geometry,r)in enumerate(results.items()):
    with np.load(folder/geometry/'chains.npz')as archive:
        for row,phase in enumerate(('health','audit_parent','audit_terminal')):
            if phase+'_energies'not in archive:continue
            e=archive[phase+'_energies'].real/36;steps=archive[phase+'_energy_indices']*32+1024
            for a,b,label,color in ((0,8,'Uniform starts','#0072B2'),(8,16,'Ordered starts','#D55E00')):
                values=e[:,a:b];axes[row,col].plot(steps,values.mean(axis=1),marker='.',label=label,color=color)
            axes[row,col].set_title(f'{geometry}: {phase}');axes[row,col].set_ylabel('Snapshot cohort mean E / N')
            axes[row,col].set_xlabel('Proposal attempts / chain');axes[row,col].grid(alpha=.15);axes[row,col].legend(fontsize=8)
fig.suptitle('Fixed-state start-cohort traces; no iid-snapshot confidence bands')
fig.savefig(out/'fig100_decoder_n36_chain_health.png',dpi=180)
fig.savefig(out/'fig100_decoder_n36_chain_health.pdf');plt.show()
"""),markdown(r"""
### Interpretation boundary

This is a bounded trainability and chain-health experiment. A short descending
training trace is not proof of equilibrium, competitive ground-state energy
or a method advantage. Inspect independent terminal-minus-parent changes
together with start-cohort dependence, complex-energy diagnostics and short-
chain Rhat. Acceptance alone is not a mixing test. No checkpoint is chosen by
the audit. No automatic extension, N16 refinement or remote job follows.
See `results/spin_decoder_pilot_v1/REPORT.md`; article notebook is unchanged.
""")]
    if physical_initialization:
        replacements={
            'spin_decoder_pilot_v1':'spin_decoder_physical_pilot_v1',
            'fig98_decoder':'fig101_decoder', 'fig99_decoder':'fig102_decoder',
            'fig100_decoder':'fig103_decoder',
            "axhline(-.375,color='#777777'":"axhline(-.4296875 if geometry=='triangular' else -.4375,color='#777777'",
            'S=0 NN dimers; no lattice projection':'Unperturbed cell product; no A1 projection',
            'N36 strict decoder: bounded chain-health check and joint energy pilot':
                'N36 physical singlet initialization: unchanged decoder, bounded joint training',
            'One inherited weight\ninitialization':'One local-singlet-based weight\ninitialization',
            'The parent here is the random N36\ncost-test initialization, not a transferred high-quality N16 VAP state.':
                'The parent embeds the open four-spin ground singlet in the existing reduced tensors,\n'
                'then perturbs EVERY tensor with a fixed Gaussian SD .05/sqrt(input fan-in).\n'
                'This is a parameter perturbation, NOT a state norm ratio. No q channels\n'
                'are cut or frozen. It is not a high-quality VAP transfer or a new RVB mechanism.\n'
                'Only local N4 eigenvectors are calculated; no N16 or N36 ED is used.',
            'N36 short pilot: noisy training trajectory is not an independent variational result':
                'N36 local-singlet start: fixed 24 updates; independent endpoint audits below',
        }
        for cell in cells:
            for old,new in replacements.items():cell.source=cell.source.replace(old,new)
        cells.append(code(r"""
bond=json.loads((folder/'bond_diagnostics.json').read_text())
assert bond['sha256']==canonical_sha256({k:v for k,v in bond.items()if k!='sha256'})
fig,axes=plt.subplots(1,3,figsize=(12,3.6),constrained_layout=True)
for col,geometry in enumerate(('triangular','square')):
    phases=bond['diagnostics'][geometry]
    for name,marker,color in [('J1','o','#0072B2'),('J2','s','#D55E00')]:
        vals=[phases[k][name]for k in ('audit_parent','audit_terminal')]
        axes[col].errorbar([0,1],[v['mean']for v in vals],yerr=[v['chain_se']for v in vals],
                          fmt=marker+'-',capsize=3,color=color,label=name+' contribution')
    axes[col].set(xticks=[0,1],xticklabels=['Parent','Terminal'],ylabel='Energy / N (3Hz estimator)',title=geometry)
    axes[col].grid(alpha=.15);axes[col].legend(fontsize=8)
    for x,phase in enumerate(('audit_parent','audit_terminal')):
        v=phases[phase]['H_minus_3Hz']
        axes[2].errorbar(x+col*.12,v['mean'],yerr=v['chain_se'],fmt='o',capsize=3,color=colors[geometry],
                        label=geometry if x==0 else None)
axes[2].axhline(0,color='0.5',ls='--');axes[2].set(xticks=[.06,1.06],xticklabels=['Parent','Terminal'],
                                               ylabel='(Full H - 3Hz) / N',title='Same-chain consistency check')
axes[2].grid(alpha=.15);axes[2].legend(fontsize=8)
fig.suptitle('Singlet bond energies: 1 chain SE; neither mixing certification nor physical variance',fontsize=11)
fig.savefig(out/'fig104_decoder_physical_bonds.png',dpi=180)
fig.savefig(out/'fig104_decoder_physical_bonds.pdf');plt.show()
print('Fixed-partition contributions (A1 relates equivalent bonds; NOT a cluster-order diagnosis):',bond['diagnostics'])
"""))
        cells.append(markdown(r"""
### Completed outcome: useful initialization, no resolved optimization win

Triangular independent E/N: -0.43256 +/-0.00563 -> -0.43632 +/-0.00577;
square: -0.44341 +/-0.00509 -> -0.44490 +/-0.00555. Both energy-change
and variance-change95% intervals contain zero. **All six frozen-state
sampling screens fail**, so chain error bars do not cover possible mixing
bias. Neither these nominal energies nor the small sampled variances are
certified competitive variational results. Full-H/3Hz agreement also cannot
override the chain diagnostics. No tail points or poor chains were deleted.

This is the existing local-singlet/RVB initialization idea embedded in the
current decoder, not a new SU2+ViT mechanism or a proof of a capacity ceiling.
The fixed budget is closed; no N16 refinement or remote production follows.
"""))
    for cell in cells:
        cell.metadata['experiment_section']=('decoder_physical_20260930' if physical_initialization else 'decoder_pilot_20260929')
    return cells


def decoder_sampler_cells():
    """Fixed physical parent, equal-H proposal controls; never a variational win."""
    cells=[markdown(r"""
## N36 fixed-parent sampling reliability: nearest / global / coherent-frame mixture

Triangular6x6 J2/J1=.125 and square6x6 J2/J1=.5. **No parameter updates.**
Use the saved physical initialization, not its trained endpoint, with all4492
parameters, exact SU2, and all432/288 A1 images. All arms start from the same
8 uniform and8 ordered configurations, but use independent transition RNG.

Nearest exchanges a J1 edge. Global exchanges any two sites (the old kernel).
Frame mix chooses50% global exchange and50% independent frame-mixture MH.
Reuse the ordered coherent plaquette frames, with their group multiplicities:
Q(x)=.9 sum_f w_f |phi_f(x)|² + .1/binomial(36,18). The uniform term gives
full m=0 support; acceptance uses the **full Q**, not the chosen frame density.
This changes the proposal, not the coherent learned target amplitude.

Per arm:1024 burn proposals/chain, then32 snapshots separated by32 attempts;
256 measured full-H centers plus8 identical parent replay probes. Equal H and
target-amplitude query counts do not imply equal wall time or independent
samples. No additional samples, parameter/seed search, ED or remote jobs.
SEs are between whole-chain means and omit mixing bias. Rhat<=1.2 is only a
loose short-chain screen. **Any apparent energy change is sampling error or
bias at a fixed state, not variational improvement.** Rare near-node H-ratio
contributions can remain missing even when the screens improve.
"""),code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from su2_nqs.frontier.artifacts import canonical_sha256
root=next(p for p in (Path.cwd(),*Path.cwd().parents)if(p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs';folder=project/'results/spin_decoder_sampler_comparison_v1'
geometries=('triangular','square');arms=('nearest','global','frame_mix')
labels=('J1 exchange','Any-pair exchange','Frame mixture')
colors=('#777777','#0072B2','#D55E00')
results={(g,a):json.loads((folder/g/a/'results.json').read_text())for g in geometries for a in arms}
raw={(g,a):dict(np.load(folder/g/a/'chains.npz'))for g in geometries for a in arms}
for r in results.values():
    assert r['result_sha256']==canonical_sha256({k:v for k,v in r.items()if k!='result_sha256'})
    assert r['initial_parameters_sha256']==r['final_parameters_sha256']
    assert r['measurement_H_centers']==256 and r['full_H_centers']==264 and r['training_steps']==0
out=project/'figures/exploration';out.mkdir(parents=True,exist_ok=True)
fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
for col,g in enumerate(geometries):
    for i,a in enumerate(arms):
        s=results[g,a]['summary']
        for offset,cohort,marker,name in ((-.17,s['cohorts'][0],'o','Uniform starts'),
                                         (.17,s['cohorts'][1],'s','Ordered starts'),(0,s,'D','All chains')):
            axes[0,col].errorbar(i+offset,cohort['energy_per_site'],yerr=cohort['energy_chain_se'],
                fmt=marker,capsize=3,color=colors[i],mfc='white' if offset else colors[i],
                label=name if i==0 else None)
    for field,marker in (('energy','o'),('log_weight','s'),('nn_correlation','^')):
        axes[1,col].plot(range(3),[results[g,a]['summary']['diagnostics'][field]['split_rhat'] for a in arms],
            marker=marker,label=field.replace('_',' '))
    axes[0,col].set(title=g.capitalize(),ylabel='Fixed-parent E / N (1 chain SE)')
    axes[1,col].axhline(1.2,color='0.5',ls='--',label='Loose screen, not certification')
    axes[1,col].set_ylabel('Classical split Rhat')
    for ax in axes[:,col]:ax.set_xticks(range(3),labels);ax.grid(alpha=.15);ax.legend(fontsize=7)
fig.suptitle('Same wavefunction and H budget: test reliability, not variational improvement')
fig.savefig(out/'fig105_decoder_sampler_comparison.png',dpi=180)
fig.savefig(out/'fig105_decoder_sampler_comparison.pdf');plt.show()
for g in geometries:
    for a in arms:
        r=results[g,a];s=r['summary']
        print(g,a,'screen=',s['screening_ok'],'move=',round(s['mean_nontrivial_move_fraction'],4),
              'seconds=',round(r['active_seconds'],2),'descriptive E-ESS/s=',round(s['energy_ess_per_second'],3))
"""),code(r"""
fig,axes=plt.subplots(2,3,figsize=(12,6.4),sharex=True,constrained_layout=True)
for row,g in enumerate(geometries):
    for col,a in enumerate(arms):
        d=raw[g,a];e=d['energies'].real/36;x=d['energy_indices']*32
        for start,color,label in ((0,'#009E73','Uniform starts'),(8,'#CC79A7','Ordered starts')):
            values=e[:,start:start+8].mean(axis=1)
            axes[row,col].plot(x,values,ls=':',alpha=.5,color=color,lw=1)
            axes[row,col].plot(x,np.cumsum(values)/np.arange(1,len(values)+1),color=color,label=label)
        axes[row,col].set(title=g+' / '+labels[col],xlabel='Post-burn proposal attempts / chain',ylabel='E / N')
        axes[row,col].grid(alpha=.15);axes[row,col].legend(fontsize=7)
fig.suptitle('Start-cohort traces: dotted batch means, solid cumulative means (not independent points)')
fig.savefig(out/'fig106_decoder_sampler_traces.png',dpi=180)
fig.savefig(out/'fig106_decoder_sampler_traces.pdf');plt.show()
"""),code(r"""
fig,axes=plt.subplots(2,2,figsize=(10,6.6),constrained_layout=True)
for col,g in enumerate(geometries):
    for i,a in enumerate(arms):
        d=raw[g,a];e=d['energies'];tail=np.sort(abs(e-e.mean()).ravel()**2)[::-1]
        x=np.arange(1,len(tail)+1)/len(tail)
        axes[0,col].plot(x,np.cumsum(tail)/tail.sum(),color=colors[i],label=labels[i])
        logw=np.sort((d['log_weights']-d['log_q']).ravel())
        axes[1,col].plot(np.arange(1,len(logw)+1)/len(logw),logw,color=colors[i],label=labels[i])
        s=results[g,a]['summary']
        from math import comb, log
        outside_mask=d['log_q']<log(.1/comb(36,18))+1e-8
        outside=int(outside_mask.sum());measured_outside=outside_mask[d['energy_indices']-1]
        residual=abs(e-e.mean())**2
        print(g,a,'variance/N=',s['variance_per_site'],'largest-chain fraction=',s['largest_chain_variance_fraction'],
              'leave-one-chain-out variance range=',s['leave_one_chain_out_variance_range'],
              'H-3Hz=',s['H_minus_3Hz_per_site'],'+/-',s['H_minus_3Hz_chain_se'],
              'saved samples outside pure-core support=',outside,'/',d['log_q'].size,
              'outside-core measured variance fraction=',float(residual[measured_outside].sum()/residual.sum()),
              'imaginary energy / chain SE=',s['imaginary_energy_per_site']/s['imaginary_chain_se'])
    axes[0,col].set(title=g.capitalize(),xscale='log',xlabel='Fraction of measured samples (largest residual first)',
                    ylabel='Cumulative fraction of sampled variance')
    axes[1,col].set(xlabel='Empirical quantile of saved chain configurations',ylabel='log(|psi|² / Q), unnormalized')
    for ax in axes[:,col]:ax.grid(alpha=.15);ax.legend(fontsize=8)
fig.suptitle('Tail sensitivity and importance-weight range; unseen rare events remain unbounded')
fig.savefig(out/'fig107_decoder_sampler_tails.png',dpi=180)
fig.savefig(out/'fig107_decoder_sampler_tails.pdf');plt.show()
"""),markdown(r"""
### Completed decision: bulk mixing improves; second-moment reliability does not follow

Both nearest and frame-mixture arms pass the loose screens; both arbitrary-pair
arms fail. Energy estimates remain mutually compatible at a **fixed** state.
This is operational progress, not a lower-energy ansatz or a PRL claim.

One triangular frame-mixture event outside every unperturbed product-frame
support contributes **68.54% of sampled variance**. It is kept. Square Im(E)
is3.09--5.76 conditional chain SE from zero across all three arms, despite good
bulk diagnostics for nearest/frame. The exact mean must be real. Common bias,
rare tails and uncertainty calibration remain unresolved; no training is released.

No extra sampling follows this budget. A next separately bounded importance
audit should cover the core and its H-connected neighbors to check the second
moment, rather than merely lengthen identical Born chains. No such audit has
been performed here. All six runs total1584 full-H centers and11.57 active
CPU-process wall minutes, with unchanged parent hashes and exact H replay.
""")]
    for cell in cells:cell.metadata['experiment_section']='decoder_sampler_20260930'
    return cells


def resonance_continuation_cells():
    """Read sealed continuation endpoints and their fresh audit; no new physics."""
    cells = [markdown(r"""
## Fixed-architecture energy continuation: attention versus matched free tables

Triangular $6\times6$, $J_2/J_1=0.125$. This is a bounded optimization and
generalization test, not a symmetry ablation: **every state has exact SU(2),
complete $A_1$, and the same analytic projected pi/0-flux carrier**. The
carrier is not optimized two-Lanczos. Shared attention730 remains contained
in the unrestricted free360 generator table.

Three frozen additive-teacher students initialize three attention runs.
For each, a free360 run starts from its **exact copied generator table**, so
the initial physical states match. Independently continue the old additive110
and free360 teacher endpoints as two further controls: eight trained arms in
total. Each has fresh zero Adam moments, 60 fixed updates, empirical FS radius
0.001 and 12 fixed step-search candidates (maximum step plus 11 bisections).
All use the same saved 128-record training feature corpus and differentiate
both sides of the finite-Q full-H energy quotient. This is not a new global
optimization certificate; supervised initialization costs are inherited.

All initial and final states were sealed before drawing 128 fresh audit
records on 16 independent chains. Fourteen audited states include the
carrier, five distinct initial states, and eight endpoints. The preregistered
primary contrast is the **mean energy difference of the three fixed
attention-minus-matched-free pairs**. These warmstarts are not independent
statistical replicas: whole-chain bootstrap jointly resamples all states.
The six individual pair/metric comparisons also retain Bonferroni-adjusted
intervals. Endpoint-minus-own-initializer intervals are descriptive nominal
95% intervals, not additional multiplicity-corrected primary claims.
"""), code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from su2_nqs.frontier.artifacts import canonical_sha256
from su2_nqs.frontier.provenance import file_sha256

root = Path.cwd()
if not (root/'su2_nqs').exists():
    root = next(p for p in root.parents if (p/'su2_nqs').exists())
project = root/'projects/clebsch_tree_nqs'
folder = project/'results/spin_resonance_continuation_v1'
result = json.loads((folder/'results.json').read_text())
unsigned = dict(result); result_hash = unsigned.pop('result_sha256')
assert canonical_sha256(unsigned) == result_hash and result['status'] == 'complete'
assert file_sha256(folder/'registration.json') == result['registration_sha256']
for name, sha in result['artifacts'].items():
    assert file_sha256(folder/name) == sha, name
assert result['fresh_audit'] and not result['remote_released']
assert result['config']['training_steps'] == 60
assert result['config']['step_fs_radius'] == .001
assert result['config']['chains'] == 16 and result['config']['audit_blocks'] == 8
training = result['training']['arms']
arms = ['continued_additive110', 'continued_free360',
        *[name for i in range(3) for name in (f'matched_free{i}', f'attention{i}')]]
state_names = ['carrier', 'initial_additive', 'initial_free',
               *[f'initial{i}' for i in range(3)], *arms]
assert set(training) == set(arms) and len(state_names) == 14
assert all(len(training[arm]['history']) == 60 for arm in arms)
assert max(result['training']['initial_replay_max'].values()) < 1e-10
with np.load(folder/'audit_values.npz', allow_pickle=False) as saved:
    assert list(saved['names']) == state_names
    assert saved['psi'].shape == saved['hpsi'].shape == (128, 14)
    assert len(np.unique(saved['groups'])) == 16
audit = result['audit']; primary = result['primary_audit']
assert primary['primary'] == 'mean.energy' and primary['family_size'] == 6
start_colors = ['#0072B2', '#D55E00', '#009E73']
labels = {'continued_additive110': 'Additive110 continuation',
          'continued_free360': 'Free360 teacher continuation'}
colors = {'continued_additive110': '#444444', 'continued_free360': '#999999'}
styles = {'continued_additive110': '-.', 'continued_free360': ':'}
for i in range(3):
    labels[f'matched_free{i}'] = f'Start {i+1}: matched free360'
    labels[f'attention{i}'] = f'Start {i+1}: attention730'
    colors[f'matched_free{i}'] = colors[f'attention{i}'] = start_colors[i]
    styles[f'matched_free{i}'] = '--'; styles[f'attention{i}'] = '-'

def interval_point(ax, y, item, color, adjusted=False, marker='o'):
    if adjusted:
        lo, hi = item['bonferroni_six_comparisons_interval']
        ax.hlines(y, lo, hi, color=color, lw=7, alpha=.16)
    lo, hi = item['chain_bootstrap_95']
    ax.hlines(y, lo, hi, color=color, lw=1.7)
    ax.plot(item['value'], y, marker, color=color, ms=5)

fig, axes = plt.subplots(3, 2, figsize=(13, 12.5), constrained_layout=True)
for arm in arms:
    history = training[arm]['history']; steps = [row['step'] for row in history]
    for ax, metric in zip(axes[0], ['energy_per_site', 'variance_per_site']):
        ax.plot(steps, [row[metric] for row in history], color=colors[arm],
                ls=styles[arm], lw=1.5, label=labels[arm])
axes[0,0].set(title='Eight independent continuations: training energy',
              ylabel='Finite-Q training E/N', xlabel='Fixed update')
axes[0,1].set(title='Training variance: cached full-H corpus',
              ylabel='Finite-Q training Var(H)/N', xlabel='Fixed update')
axes[0,0].legend(frameon=False, fontsize=7, ncol=2)
axes[0,1].text(.03, .97, 'Solid: attention; dashed: matched free\nFinite-Q curves are not variational upper bounds',
              transform=axes[0,1].transAxes, va='top', fontsize=8)
pair_labels = [f'Start {i+1}: NN - matched free' for i in range(3)] + ['Mean of 3 pairs']
for ax, metric, title in zip(axes[1], ['energy', 'variance'],
        ['Fresh audit: matched energy contrast', 'Fresh audit: matched variance contrast']):
    for i in range(3):
        interval_point(ax, i, primary['metrics'][f'pair{i}.{metric}'], start_colors[i], adjusted=True)
    interval_point(ax, 3, primary['metrics']['mean.'+metric], '#222222', marker='D')
    ax.set(yticks=range(4), yticklabels=pair_labels, title=title,
           xlabel='Change per site; negative favors attention')
    ax.set_ylim(3.65, -.35)
    ax.locator_params(axis='x', nbins=5)
    ax.text(.02, .015, 'Thin: nominal 95%; pale: Bonferroni 6-comparison intervals',
            transform=ax.transAxes, fontsize=7)
for ax, metric, title in zip(axes[2], ['energy_per_site', 'variance_per_site'],
        ['Fresh audit: energy change from own initializer',
         'Fresh audit: variance change from own initializer']):
    for i, arm in enumerate(arms):
        parent = training[arm]['parent']
        interval_point(ax, i, audit['paired'][arm+'-minus-'+parent][metric], colors[arm],
                       marker='o' if 'attention' in arm else 's')
    ax.set(yticks=range(len(arms)), yticklabels=[labels[arm] for arm in arms], title=title,
           xlabel='Change per site; nominal whole-chain 95% interval')
    ax.invert_yaxis()
for ax in axes[1:].flat:
    ax.axvline(0, color='0.4', ls='--', lw=1)
for ax in axes.flat:
    ax.grid(axis='x', alpha=.2)
    ax.ticklabel_format(axis='x', style='plain', useOffset=False)
fig.suptitle('Fixed architecture, fixed work, fresh paired audit\nAll arms retain SU(2) and full A1; this is not a symmetry ablation', fontsize=13)
out = project/'figures/exploration'; out.mkdir(parents=True, exist_ok=True)
fig.savefig(out/'fig121_resonance_energy_continuation.png', dpi=180)
fig.savefig(out/'fig121_resonance_energy_continuation.pdf')
plt.show()
"""), code(r"""
# Dynamic tables deliberately make no a-priori claim about the outcome.
rows = ['| Audited state | E/N (whole-chain SE) | Var(H)/N (whole-chain SE) | Norm ESS fraction | Largest residual chain share |',
        '| --- | ---: | ---: | ---: | ---: |']
for name in state_names:
    state = audit['arms'][name]; e = state['energy_per_site']; v = state['variance_per_site']
    rows.append(f"| {name} | {e['value']:.8f} ± {e['chain_bootstrap_se']:.8f} | {v['value']:.8f} ± {v['chain_bootstrap_se']:.8f} | {state['weight_ess_fraction']:.3f} | {state['max_residual_chain_fraction']:.3f} |")
display(Markdown('\n'.join(rows)))
rows = ['| Prespecified NN − matched-free contrast | Difference/site | Joint-resampled nominal 95% interval | Bonferroni six-comparison interval |',
        '| --- | ---: | --- | --- |']
for name, item in primary['metrics'].items():
    lo, hi = item['chain_bootstrap_95']
    adjusted = item.get('bonferroni_six_comparisons_interval')
    adjusted_text = '— (not an individual six-test contrast)' if adjusted is None else f"[{adjusted[0]:+.7g}, {adjusted[1]:+.7g}]"
    rows.append(f"| {name} | {item['value']:+.7g} | [{lo:+.7g}, {hi:+.7g}] | {adjusted_text} |")
display(Markdown('\n'.join(rows)))
rows = ['| Continued arm | Parameters | Matched steps / 60 | Training FS from initializer | Audit FS from initializer | Cumulative optimizer seconds |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
for arm in arms:
    record = training[arm]
    matched = sum(row['physical_step']['matched'] for row in record['history'])
    rows.append(f"| {arm} | {record['parameter_count']} | {matched} | {record['train_endpoint_distance']['fs']:.5g} | {result['audit_endpoint_distances'][arm]['fs']:.5g} | {record['cumulative_optimizer_active_seconds']:.2f} |")
display(Markdown('\n'.join(rows)))
print('Primary:', primary['primary'], '|', primary['interpretation'])
print('Primary covariance shape:', np.asarray(primary['bootstrap_covariance']).shape)
print('Cumulative experiment seconds:', result['seconds'])
print('Fresh full-H feature centers / additional cheap carrier-H centers:',
      result['physical_H_feature_centers'], result['additional_cheap_carrier_H_centers'])
print('Features shared by all arms:', result['all_arms_share_full_H_features'])
print('Fresh audit / equilibrium certified / variational win claimed / remote released:',
      result['fresh_audit'], result['equilibrium_certified'], result['variational_win_claimed'], result['remote_released'])
print('Immutable result SHA256:', result_hash)
"""), markdown(r"""
### Interpretation boundaries

**Observed outcome:** no fresh-audit continuation gain was established.
The primary attention-minus-matched-free energy contrast is
$+0.000191\pm0.000412$ per site (whole-chain bootstrap SE), with nominal 95%
interval $[-0.000559,+0.001048]$. All eight endpoint-minus-initializer energy
and variance point estimates are positive; most intervals include zero, so
this is not evidence that every endpoint significantly worsens. Attention
start 2 has an adverse variance contrast even after the six-test correction.

The physical-distance comparison is incomplete: the three attention runs
matched the target radius in only 5, 4 and 1 of 60 steps, versus 60/60 for
each free-table arm. Every update was nonzero, finite and bracketed, but the
fixed absolute bisection grid was too coarse at the attention step scale;
attention accumulated 11–18% less FS distance than its paired free table.
This does not explain why the free-table continuations also failed to
transfer their training gains. Do not interpret this as an expressivity
failure or silently retune the sealed experiment.

The unchanged carrier itself shifted from $-0.503521$ to $-0.509852$ per site
between the old and fresh audits, approximately 2.92 combined standard errors.
Earlier teacher-minus-carrier energy gains are not replicated here. This
warns about estimator stability; it neither identifies a unique cause nor
proves a sampler bug. Compare states within the paired audit, not across
audit absolute energies. No extra audit or training was appended.

Read the prespecified **mean paired energy contrast** alongside variance,
the six adjusted individual comparisons, and the change from each arm's own
initial state. Do not choose the best warmstart after seeing the audit.
A nominal interval excluding zero is not itself a global variational bound
or proof of equilibration. Norm-weight ESS does not certify the H-squared or
residual tails; chain concentration and the saved leave-one-chain-out
diagnostics remain relevant. The training FS radius is a sampled metric,
not a guarantee of equal all-Hilbert updates or equal parameter counts.

The three warmstarts share one teacher, training corpus, and audit corpus;
their spread must not be treated as a three-seed Monte Carlo error bar.
The full free-table control is independently continued, not frozen at the
attention initializer. Any finite-budget attention advantage would concern
optimization or inductive bias inside that larger table family, not an
enlarged state manifold, a new SU(2) advantage, or a QSL conclusion. Conversely,
a null result here does not prove a general attention/SU(2) incompatibility.
This figure remains exploratory; no outcome automatically releases a GPU
campaign, and the condensed article notebook is unchanged.
""")]
    for cell in cells:
        cell.metadata['experiment_section'] = 'resonance_continuation_20261001'
    return cells


def resonance_distillation_cells():
    """Saved-feature teacher reproduction; no fitting or new H in notebook."""
    cells=[markdown(r"""
## Attention reproduction: table bottleneck versus physical-state optimization

Triangular $6\times6$, $J_2/J_1=0.125$. This is a **post-hoc expressivity and
optimization diagnostic**, not new ground-state evidence. Same730-parameter
attention, original frozen anchor, strict SU(2), full $A_1$, same analytic
carrier. No new configurations, carrier queries, physical H calls, or GPU.

Two saved teachers: independently trained additive110 and free360 endpoints.
Three starts each: old attention endpoint, original initialization, independent
initialization. Fit all36 generator contexts first, then the saved128-point
training-state innovation and its shifted-H response. Each L-BFGS fit has a
fixed1000-iteration/1500-evaluation ceiling. One direct physical fit per teacher
tests bypassing table pretraining. All14 final endpoints are reported.

The width8 linear head forces $\operatorname{rank}\,\mathrm{center}(F_\theta)
\leq8$. The correct anchored target is $F_0+G_T$, not $G_T$. Its SVD tail
gives a necessary **table-error** floor, not a physical-state-error floor.

For fixed-Q inner products, define $a_v=\langle C,v\rangle_Q/\langle C,C\rangle_Q$
and $d_v=v/a_v-C$. We fit relative squared errors in $d_v$ and
$(H-E_T)d_v$. These retain the small carrier improvement that total-state
fidelity can hide. $E_T$ is fixed for each corpus, in total-H units; train
and transfer response metrics therefore use different recorded shifts.
The old128-point audit is now a **seen-transfer diagnostic**, not an untouched
audit; it was not used to select a start or fitting endpoint.
"""),code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from su2_nqs.frontier.artifacts import canonical_sha256
from su2_nqs.frontier.provenance import file_sha256

root=Path.cwd()
if not (root/'su2_nqs').exists():root=next(p for p in root.parents if (p/'su2_nqs').exists())
project=root/'projects/clebsch_tree_nqs'
folder=project/'results/spin_resonance_distillation_v1'
r=json.loads((folder/'results.json').read_text())
unsealed=dict(r);seal=unsealed.pop('result_sha256')
assert canonical_sha256(unsealed)==seal and r['status']=='complete'
assert file_sha256(folder/'registration.json')==r['registration_sha256']
for name,sha in r['artifacts'].items():assert file_sha256(folder/name)==sha
assert r['new_physical_H_calls']==r['new_carrier_queries']==r['new_samples']==0
assert not r['fresh_audit'] and not r['variational_win_claimed']
teachers=r['config']['teachers'];starts=r['config']['starts']
colors={'additive110':'#009E73','free360':'#0072B2'}
styles=['-','--',':']
fig,axes=plt.subplots(2,2,figsize=(11.5,8),constrained_layout=True)
for teacher in teachers:
    c=colors[teacher]
    for j,start in enumerate(starts):
        for ax,stage in zip(axes[0],['table','physical']):
            hist=r['fits'][teacher+'__'+start+'__'+stage]['history']
            ax.semilogy([h['iteration'] for h in hist],[h['loss'] for h in hist],
                        color=c,ls=styles[j],lw=1.3,label=teacher+' / '+str(j+1))
    old=r['evaluations'][teacher]['old_attention']['table']
    denominator=360*old['absolute_mse']/old['relative_mse']
    floor=r['rank_certificates'][teacher]['weighted_frobenius_error_lower_bound']**2/denominator
    axes[0,0].axhline(floor,color=c,ls='-.',alpha=.7)
axes[0,0].set(title='Whole-table fit; dash-dot = necessary head floor',xlabel='L-BFGS iteration',ylabel='Gauge-invariant relative squared error')
axes[0,1].set(title='Physical fit after table pretraining',xlabel='L-BFGS iteration',ylabel='Mean innovation / H-response relative MSE')
axes[0,1].legend(frameon=False,fontsize=7,ncol=2)
for teacher in teachers:
    c=colors[teacher];rows=r['evaluations'][teacher];offset=-.09 if teacher=='additive110' else .09
    for ax,metric in zip(axes[1],['energy_per_site','variance_per_site']):
        value=rows['teacher']['seen_transfer'][metric]
        ax.axhline(value,color=c,ls='--',lw=1,label=teacher+' teacher')
        groups=[['old_attention'],[teacher+'__'+s+'__table' for s in starts],
                [teacher+'__'+s+'__physical' for s in starts],[teacher+'__direct_physical']]
        for x,names in enumerate(groups):
            y=[rows[n]['seen_transfer'][metric] for n in names]
            ax.vlines(x+offset,min(y),max(y),color=c,lw=1)
            ax.scatter([x+offset]*len(y),y,color=c,s=22,alpha=.8)
        ax.set(xticks=range(4),xticklabels=['Old NN','Table fit','Then state fit','Direct state fit'])
axes[1,0].set(title='Seen-transfer energy replay',ylabel='E/N (not a fresh energy benchmark)')
axes[1,1].set(title='Seen-transfer variance replay',ylabel='Var(H)/N')
axes[1,1].legend(frameon=False,fontsize=8)
for ax in axes.flat:ax.grid(alpha=.2)
fig.suptitle('Fixed architecture: teacher reproduction, not a new variational win\nDots/ranges show all starts, not statistical confidence intervals')
out=project/'figures/exploration';out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/'fig120_resonance_attention_reproduction.png',dpi=180)
fig.savefig(out/'fig120_resonance_attention_reproduction.pdf')
plt.show()
rows=['| Teacher / endpoint | Table relative RMS | Transfer innovation relative RMS | Transfer H-response relative RMS | Transfer E/N | Transfer Var(H)/N |',
      '| --- | ---: | ---: | ---: | ---: | ---: |']
for teacher in teachers:
    for name,v in r['evaluations'][teacher].items():
        m=v['seen_transfer'];label=name.split('__',1)[-1]
        rows.append(f"| {teacher} / {label} | {np.sqrt(v['table']['relative_mse']):.4g} | {np.sqrt(m['innovation_relative_mse']):.4g} | {np.sqrt(m['h_response_relative_mse']):.4g} | {m['energy_per_site']:.8f} | {m['variance_per_site']:.8f} |")
display(Markdown('\n'.join(rows)))
print('wall seconds:',r['seconds'],'optimizer evaluations:',sum(v['optimizer_function_evaluations'] for v in r['fits'].values()))
print('All fits reached the fixed iteration cap:',all(v['iterations']==1000 for v in r['fits'].values()))
print('Numerical reconstruction of old NN table, max:',r['neural_table_replay_max'])
"""),markdown(r"""
**Interpretation.** All three table-pretrained students reproduce the additive
teacher closely: seen-transfer innovation relative RMS0.40–0.50%, shifted-H
RMS0.74–0.86%. For the full-table teacher these are6.4–9.2% and9.8–13.1%:
substantial but not exact reproduction. Energy differences from teachers are
at most $6.2\times10^{-6}$/site (additive) and $7.2\times10^{-5}$/site (full).
Variance differences are at most $9.5\times10^{-6}$ and $2.9\times10^{-4}$ per
site. Direct physical fitting gives appreciably larger innovation errors, but
uses1000 rather than2000 total iterations. This is not a matched-work test of
whether pretraining itself is more efficient.

Thus the previous attention deficit is **not explained solely by physical
expressivity**: the unchanged network can encode nearly all of the observed
teacher improvement using a better supervised optimization path. There is
also a real table-head bottleneck, but it is not a comparable state-space
no-go. Free-table pretraining supplies all contexts, not new Monte Carlo data.
No global convergence, all-Hilbert replay, or new energy win is certified;
all fits stop at their1000-iteration cap. Do not widen the model, add samples,
or release GPU production solely because of this diagnostic. A later energy
comparison should start from a common improved physical state, keep the
independently optimized free-table control live, and reserve genuinely new
evaluation chains. The article notebook is unchanged.
""")]
    for cell in cells:cell.metadata['experiment_section']='resonance_distillation_20261001'
    return cells


def resonance_training_cells():
    """Three independently trained arms and untouched audit; saved arrays only."""
    cells = [markdown(r"""
## N36 joint environments: bounded independent optimization and untouched audit

Triangular $6\times6$, $J_2/J_1=0.125$. Every arm starts from the same analytic
projected pi/0-flux carrier, with strict SU(2), all432 spatial images and216
Hamiltonian bonds. The carrier is **not optimized two-Lanczos**.

Compare independently optimized additive110, free360, and shared attention730.
The exact local operator has1176 fixed basis coefficients; precontracting
these with the carrier and full H changes evaluation cost, not the ansatz.
The attention remains contained in the unrestricted free table.

128 training records /16 chains;60 fixed updates per arm,12 fixed FS-search
candidates per update. The finite-Q full-H quotient differentiates both bra
and ket. Final endpoints are sealed before drawing128 new audit records on
16 independent chains. Training curves are **not variational upper bounds**.
The primary comparison is free360 minus additive110; paired whole-chain
intervals retain correlation between arms. There is no added ED or GPU work.
"""), code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from su2_nqs.frontier.artifacts import canonical_sha256
from su2_nqs.frontier.provenance import file_sha256

root=Path.cwd()
if not (root/'su2_nqs').exists():
    root=next(p for p in root.parents if (p/'su2_nqs').exists())
project=root/'projects/clebsch_tree_nqs'
folder=project/'results/spin_resonance_training_v1'
result=json.loads((folder/'results.json').read_text())
unsealed=dict(result);seal=unsealed.pop('result_sha256')
assert canonical_sha256(unsealed)==seal and result['status']=='complete'
assert file_sha256(folder/'registration.json')==result['registration_sha256']
for name,sha in result['artifacts'].items():assert file_sha256(folder/name)==sha
arms=result['config']['arms'];audit=result['audit']
colors={'additive110':'#009E73','free360':'#0072B2','attention730':'#CC79A7'}
labels={'additive110':'Additive (110)','free360':'Full free table (360)','attention730':'Shared attention (730)'}
fig,axes=plt.subplots(2,2,figsize=(11,7.5),constrained_layout=True)
for arm in arms:
    hist=result['training'][arm]['history'];steps=[h['step'] for h in hist]
    axes[0,0].plot(steps,[h['energy_per_site'] for h in hist],color=colors[arm],label=labels[arm])
    axes[0,1].plot(steps,[h['variance_per_site'] for h in hist],color=colors[arm],label=labels[arm])
axes[0,0].set(xlabel='Fixed optimizer step',ylabel='Training E/N',title='Finite-Q training, not an upper bound')
axes[0,1].set(xlabel='Fixed optimizer step',ylabel='Training Var(H)/N',title='Independent parameters; same samples')
for ax in axes[0]:ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=8)
pairs=['additive110-minus-carrier','free360-minus-additive110','attention730-minus-additive110']
pair_labels=['Additive − carrier','Full − additive (primary)','Attention − additive']
for ax,metric,title in [(axes[1,0],'energy_per_site','Untouched audit: paired energy change'),
                        (axes[1,1],'variance_per_site','Untouched audit: paired variance change')]:
    for i,pair in enumerate(pairs):
        item=audit['paired'][pair][metric];lo,hi=item['chain_bootstrap_95'];value=item['value']
        color=list(colors.values())[i]
        ax.hlines(i,lo,hi,color=color,lw=2)
        ax.plot(value,i,'o',color=color)
    ax.axvline(0,color='0.45',ls='--',lw=1)
    ax.set(yticks=range(3),yticklabels=pair_labels,xlabel='Change per site; whole-chain 95% interval',title=title)
    ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
fig.suptitle('Bounded N36 experiment — fixed endpoints, independent audit')
out=project/'figures/exploration';out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/'fig119_resonance_three_arm_training.png',dpi=180)
fig.savefig(out/'fig119_resonance_three_arm_training.pdf')
plt.show()
rows=['| Arm | Audit E/N (chain SE) | Audit Var(H)/N (chain SE) | Norm weight ESS fraction | Largest chain residual share |',
      '| --- | ---: | ---: | ---: | ---: |']
for arm in ['carrier',*arms]:
    a=audit['arms'][arm];e=a['energy_per_site'];v=a['variance_per_site']
    rows.append(f"| {arm} | {e['value']:.7f} ± {e['chain_bootstrap_se']:.7f} | {v['value']:.7f} ± {v['chain_bootstrap_se']:.7f} | {a['weight_ess_fraction']:.3f} | {a['max_residual_chain_fraction']:.3f} |")
display(Markdown('\n'.join(rows)))
for arm in arms:
    t=result['training'][arm];hist=t['history']
    print(arm,'last10 training E change',hist[-1]['energy_per_site']-hist[-10]['energy_per_site'],
          'final gradient norm',hist[-1]['gradient_norm'],
          'matched steps',sum(h['physical_step']['matched'] for h in hist),
          'dev FS',t['endpoint_fs']['fs'],'audit FS',result['endpoint_audit_distances'][arm]['fs'])
    print('  global operator-bound diagnostic:',t['structure'])
    print('  original cached trajectory seconds:',hist[-1]['seconds'],
          'last invocation training/endpoint replay seconds:',t['seconds'])
for phase in ['train','audit']:
    with np.load(folder/(phase+'_features.npz'),allow_pickle=False) as cached:
        records=json.loads(str(cached['records']))
        print(phase,'shared feature setup seconds',sum(r['seconds'] for r in records),
              'carrier queries',sum(r['carrier_queries'] for r in records),
              'dense fallback components',sum(r['fallback_components'] for r in records))
historical_peak=result['peak_rss_gib']
for phase in ['train','audit']:
    with np.load(folder/(phase+'_features.npz'),allow_pickle=False) as saved:
        historical_peak=max(historical_peak,max(r['peak_rss_gib'] for r in json.loads(str(saved['records']))))
print('Total seconds / final-process peak / historical peak GiB:',
      result['seconds'],result['peak_rss_gib'],historical_peak)
if (folder/'failure.json').exists():
    print('Resource stop retained; manual checkpoint resume, same cumulative budget:',
          json.loads((folder/'failure.json').read_text()))
print('No automatic variational win:',result['variational_win_claimed'])
"""), markdown(r"""
### Reading this comparison

The fixed audit supports carrier improvement by additive110 and free360:
both energy and variance differences have nominal paired95% intervals below
zero. But the preregistered full-minus-additive energy difference is only
-0.000209 +/-0.000617/site (bootstrap SE), with a95% interval crossing zero.
Attention-minus-additive is +0.001154 +/-0.000407/site. Thus this is a positive
carrier-dressing result, not an attention or nonadditive-context win.
Attention's variance gain over C is marginal: bootstrap95% barely excludes
zero, whereas a conservative two-jackknife-SE interval includes zero.
These secondary intervals are nominal, not multiplicity-adjusted claims.

A lower training quotient may reflect finite-sample overfitting. The three
arms are compared at fixed work, not certified global minima; inspect final
gradients and curve slopes before calling the additive control converged.
Full360 contains the attention outputs, so an attention advantage at this
budget would concern optimization/inductive bias, not a larger state space.

Variance uses the complete H action, including nodes with nonzero Hpsi.
Norm weight ESS does not measure chain mixing or certify H-squared tails.
Whole-chain intervals, leave-one-chain-out ranges and residual concentration
must be read together. The training FS radius is only an empirical metric;
independent endpoint distances are reported separately. A positive support
floor makes importance sampling legal but does not certify equilibration.
This exploration figure is not automatically a PRL or QSL physics result.
""")]
    for cell in cells:cell.metadata['experiment_section']='resonance_training_20261001'
    return cells


def resonance_reuse_cells():
    """Exact same-state cost comparison and two-environment interface, saved only."""
    cells=[markdown(r"""
## Exact carrier-query reuse, then genuinely joint environmental attention

For an already invariant carrier $C$, use
$C(\mathrm{localchange}(x[g]))=C(\mathrm{localchange}(x[g])[g^{-1}])$.
This pulls every carrier query back near the **same physical center**, where
duplicates and Slater row replacements can be reused. It does NOT remove any
of the432 dressed-operator weights. All216 Hamiltonian bonds, nodes, carrier,
and old730-parameter network state remain unchanged in the cost comparison.

Two-environment gates then act on the SAME three-cell/216-input support:

$$K_1=\sum_{a,c}P_a^A\otimes U_B(a,c)\otimes P_c^C,\quad
K_2=\sum_{a,b}P_a^A\otimes P_b^B\otimes U_C(b,a),\quad
\psi=P_{A_1}K_2K_1C.$$

Each projector contains the complete magnetic multiplet. Information from two
environment labels jointly changes a reduced tensor, with no measured-label
substitution or intermediate spatial projection. A single shared attention
has730 parameters; its complete two-environment table has360 coordinates.
Free360 is therefore a stronger expressive control, not a smaller neural
model that has already been beaten. Additive/one-environment controls,
position reassignment and physically scrambled supports are retained.

All results below are from the same2 fixed configurations and232 full-H
outputs as figure117, not new Monte Carlo samples. No optimization or energy/
variance/entanglement estimate is made. Exact evaluation speedup is meaningful
engineering evidence; it is not a new variational-state or QSL result.
"""),code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
from su2_nqs.frontier.provenance import file_sha256

root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs'
folder=project/'results/spin_resonance_reuse_v1'
reuse=json.loads((folder/'results.json').read_text())
assert reuse['status']=='completed'
assert reuse['registration_sha256']==file_sha256(folder/'registration.json')
assert reuse['replay_sha256']==file_sha256(folder/'replay.npz')
assert reuse['checkpoint_sha256']==file_sha256(folder/'multicontext_checkpoint.npz')
assert reuse['energy_estimate'] is None and reuse['training_steps']==0
fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
rows=['| Support | H+AD old/new (s/center) | Speedup | Unique C queries after reuse | Max amplitude/H/gradient relative discrepancy |',
      '| --- | ---: | ---: | ---: | ---: |']
controls=['| Support / context control | Parameters | Full-H AD-probe gradient norm | Pointwise difference from full |',
          '| --- | ---: | ---: | ---: |']
with np.load(folder/'replay.npz') as saved:
    for label,color,marker in [('physical','#0072B2','o'),('scrambled','#CC79A7','s')]:
        p=reuse['geometries'][label]
        old_queries=sum(c['plan']['unique_carrier_queries'] for c in p['baseline_chunks'])
        new_queries=p['pullback_plan']['unique_carrier_queries']
        axes[0].plot([0,1],[p['baseline_fresh_H_AD_seconds']/2,p['fast_fresh_H_AD_seconds']/2],
                     marker=marker,color=color,label=label.title())
        axes[1].plot([0,1],[old_queries,new_queries],marker=marker,color=color,label=label.title())
        discrepancy=max(p[k] for k in ('amplitude_relative_error','full_H_relative_error','reverse_gradient_relative_error'))
        rows.append(f"| {label} | {p['baseline_fresh_H_AD_seconds']/2:.3f} / {p['fast_fresh_H_AD_seconds']/2:.3f} | {p['H_AD_speedup']:.2f}x | {new_queries:,} | {discrepancy:.2e} |")
        full=saved[label+'_multicontext_full']
        for control,record in p['multicontext'].items():
            value=saved[label+'_multicontext_'+control]
            distance=np.linalg.norm(value-full)/np.linalg.norm(full)
            controls.append(f"| {label} / {control} | {record['parameter_count']} | {record['gradient_norm']:.5g} | {distance:.4g} |")
        print(label,'dense carrier seconds:',p['pullback_dense_carrier_seconds'],
              'row-update full fresh seconds:',p['minor_fresh_carrier_seconds'],
              'selected:',p['selected_carrier_evaluator'])
        print(label,'minor-reference max condition:',p['minor_maximum_reference_condition'],
              'fallback component count:',p['minor_fallback_components'],
              'minor/dense amplitude discrepancy:',p['minor_dense_relative_error'])
        print(label,'nonadditive generator fraction:',p['generator_interaction_fraction'])
for ax in axes:
    ax.set_xticks([0,1],['Original complete evaluation','Exact reuse'])
    ax.set_yscale('log');ax.grid(axis='y',alpha=.2);ax.legend(frameon=False)
axes[0].set(ylabel='CPU wall seconds / full-H center + reverse AD',title='Identical physical state; compile and staging included')
axes[1].set(ylabel='Carrier queries for the two fixed H centers',title='All432 projection weights retained')
fig.suptitle('Exact evaluation reuse — no variational optimization')
out=project/'figures/exploration';out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/'fig118_resonance_exact_reuse.png',dpi=180)
fig.savefig(out/'fig118_resonance_exact_reuse.pdf')
plt.show()
display(Markdown('\n'.join(rows)))
display(Markdown('\n'.join(controls)))
print('Diagnostic total seconds / peak GiB:',reuse['total_seconds'],reuse['peak_rss_gib'])
"""),markdown(r"""
### Interpretation and strong nulls

The efficiency comparison above is paired on the same CPU, same states,
same full projection and complete H. It includes host query planning,
carrier evaluation, transfers/staging and fresh JIT cost; warm repeats are
separate diagnostics. Old queries were summed across bounded chunks, while
pullback queries share the entire fixed H support. This is a measured path
speedup, not an asymptotic scaling law, GPU forecast or competitive energy.

The context table proves two-environment dependence survives complete A1
projection. Its pointwise norm is NOT a Hilbert/FS distance or observable.
Nonadditive generators do not establish a physical advantage over fully
optimized additive generators: exponentiation itself can create interactions.
Position reassignment can be absorbed by relabelling a free table; it is
not evidence of a distinct optimized state family. Free360 and adequately
optimized additive/one-environment controls remain mandatory in future energy
comparisons. More environments may improve parameter sharing but do not
remove the potentially exponential coherent-contraction cost.
""")]
    for cell in cells:cell.metadata['experiment_section']='resonance_reuse_20261001'
    return cells


def overlap_attention_cost_cells():
    """Saved N36 full-H costs; never execute a Hamiltonian or fit a state here."""
    cells = [markdown(r"""
## Overlapping conditional resonance with shared attention: full-H cost

Three four-spin cells are ordered $A\to B\to C$. One shared reduced-label
attention supplies the scalar generators of $K_1(B|A)$ and $K_2(C|B)$:

$$\psi_{\rm cascade}=P_{A_1}K_2K_1 C,\qquad
\psi_{\rm no\ cascade}=P_{A_1}(K_1+K_2-I)C.$$

The difference is exactly $P_{A_1}(K_2-I)(K_1-I)C$. Intermediate B labels
are coherently summed, not measured or copied from the output configuration;
only the final spatial projection is applied. Each fixed-partition cell spin
q is preserved; internal multiplicity a carries the new conditional information.
The final spatial sum mixes partitions. All magnetic multiplets remain complete.

This probe uses triangular $6\times6$, $J_2/J_1=0.125$, all216 bonds and all432
space-group images. C is the analytic three-determinant pi/0-flux singlet,
**not optimized two-Lanczos**. Two preregistered uniform fixed-M configurations
are cost probes, not wavefunction samples: **no energy/variance is estimated**.
730 neural parameters generate one shared60-coordinate table; this does not
demonstrate expressivity beyond a free table or a full Spatial-ViT.

v1 stopped at4 million unique carrier queries before completing H. v2 keeps
the same two centers and streams all232 H-output configurations. Queries
repeated across chunks are counted again; no bond or image is removed. Strong
controls are retained: no-cascade, joint-diagonal-conditioned/shared-rotation,
factorized, and physical-site-scrambled supports. None is trained here.
"""), code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
from su2_nqs.frontier.provenance import file_sha256

root = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p/'su2_nqs').is_dir())
project = root/'projects/clebsch_tree_nqs'
folder = project/'results/spin_overlap_attention_cost_v2'
cost = json.loads((folder/'results.json').read_text())
assert cost['status'] == 'completed'
assert cost['registration_sha256'] == file_sha256(folder/'registration.json')
assert cost['checkpoint_sha256'] == file_sha256(folder/'checkpoint.npz')
assert cost['replay_sha256'] == file_sha256(folder/'full_h_replay.npz')
assert cost['energy_estimate'] is None and cost['variance_estimate'] is None
centers = cost['h_stencil']['center_count']
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
table = ['| Support | C queries (sum over chunks) | Full H incl. compile (s/center) | Full H + reverse AD (s/center) | Identity H error | FD/JVP H error |',
         '| --- | ---: | ---: | ---: | ---: | ---: |']
for label, color, marker in [('physical','#0072B2','o'), ('scrambled','#CC79A7','s')]:
    rec = cost['geometries'][label]
    chunks = rec['chunks']
    # IMPORTANT: the actual current implementation creates new JIT closures
    # and clears caches per chunk. Include their compile+execution cost.
    # Stored fresh_* fields use warm replay and are NOT this actual fresh cost.
    planning = sum(c['planning_seconds'] for c in chunks)
    carrier = sum(c['carrier_query_inclusive_seconds'] for c in chunks)
    forward = sum(c['forward_compile_inclusive_seconds'] for c in chunks)
    reverse = sum(c['gradient_compile_inclusive_seconds'] for c in chunks)
    actual_h = planning+carrier+forward
    actual_ad = actual_h+reverse
    cached_h = rec['arms']['cascade']['cached_full_H_warm_seconds']
    cached_ad = cached_h+sum(np.median(c['gradient_warm_seconds']) for c in chunks)
    axes[0].plot(range(4), np.array([planning,carrier,forward,reverse])/centers,
                 marker=marker, color=color, label=label.title())
    axes[1].plot(range(3), np.array([cached_ad,actual_h,actual_ad])/centers,
                 marker=marker, color=color, label=label.title())
    table.append(f"| {label} | {rec['carrier_queries_summed_over_chunks']:,} | {actual_h/centers:.3f} | {actual_ad/centers:.3f} | {rec['identity_relative_errors'][1]:.2e} | {rec['jvp_finite_difference_relative_errors'][1]:.2e} |")
    print(label, 'All gradient block norms:', rec['gradient_leaf_norms'])
    print(label, 'Pointwise cascade minus no-cascade relative change (NOT Hilbert norm):',
          rec['mixed_point_amplitude_relative_norm'], rec['mixed_point_H_relative_norm'])
for ax in axes:
    ax.set_yscale('log')
    ax.set_ylabel('CPU wall seconds / one full-H center')
    ax.grid(axis='y', alpha=.2)
    ax.legend(frameon=False)
axes[0].set(xticks=range(4), xticklabels=('Query\nplanning','Carrier\nevaluation','Projected\nforward + JIT','Reverse\nAD + JIT'),
            title='Actual cost components: all bonds and images')
axes[1].set(xticks=range(3), xticklabels=('Cached H+AD\nreplay only','Fresh full H\nactual','Fresh H+AD\nactual'),
            title='Cached neural gradient is not training cost')
fig.suptitle('N36 bounded cost probe — 2 fixed configurations; no energy estimate')
out = project/'figures/exploration'
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out/'fig117_overlap_attention_full_h_cost.png', dpi=180)
fig.savefig(out/'fig117_overlap_attention_full_h_cost.pdf')
plt.show()
display(Markdown('\n'.join(table)))
print('Complete diagnostic run seconds / peak GiB:', cost['total_seconds'],cost['peak_rss_gib'])
print('Two fixed centers, all full-H queries:',cost['h_stencil'])
print('No sampling/optimization/remote release:',cost['training_steps'],cost['remote_released'])
"""), markdown(r"""
### Decision boundary

Nonzero cascade-minus-no-cascade amplitude and full-network gradients validate
the interface, **not long-range physical correlations or an energy gain**.
The next discriminating test must retain the freely optimized joint-diagonal
and full60-table controls: masking one finite network is not proof of nested
optimized neural function families. A spatial scramble changes the physical
support; equal parameter count alone does not guarantee equal state-space rank.

Do not extrapolate these two fixed configurations into a GPU throughput or
scaling claim. All determinant evaluations, host planning, complete projection,
full H and compilation are counted above. A larger-context attention can be
interesting only when it controls more than a freely learned60-entry table;
its value must eventually appear in competitive energy/variance and physical
singlet correlations. Keep this cost figure in exploration, not the PRL figures.
""")]
    for cell in cells:
        cell.metadata['experiment_section'] = 'overlap_attention_cost_20260930'
    return cells


def conditional_carrier_resonance_cells():
    """Plot the sealed exact-small local-filter study without new model calls."""
    cells = [markdown(r"""
## Conditional singlet resonance inside a strong carrier: exact-small test

The eight-site filter acts **on the physical carrier**, not on an independently
initialized weak decoder. Four active spins have complete reduced blocks
$Q=0,2,4$ with multiplicities $2,3,1$; a disjoint four-spin spectator supplies
six complete $(q,a)$ projectors. The state is
$\psi=P_{A_1}U_\theta C$, with
$U_\theta=\sum_z P_z\otimes\bigoplus_Q(I_{m_Q}\otimes e^{G_Q(z)})$.
Thus $G=0$ exactly retains C, all magnetic multiplets remain complete, and
the full spatial projection is applied **after** the local action.

The factorized null permits a shared active generator and independent
spectator weights. The stronger **diagonal-conditioned** null permits arbitrary
joint spectator/active diagonal weights but only shared off-diagonal matrix
elements. The conditional arm additionally learns spectator-dependent
off-diagonal elements. The fourth arm scrambles the **physical site support**,
not freely learned label names. This is the discriminating question:
does environment-dependent coherent rotation help beyond reweighting?

Both lattices have $N=16$, but different carriers: triangular $J_2/J_1=0.125$
uses the saved projected-VMC/Jastrow/two-Lanczos state; square $J_2/J_1=0.5$
uses a saved high-quality VAP state, **not a pi-flux carrier**. Their complete
singlet $\Gamma-A_1$ spaces have dimensions 14 and 31, respectively. This is
not a matched cross-lattice comparison, a full ViT, or a scaling claim.
All three registered starts (identity plus two perturbations) are shown.
These are exact expectations, so start-to-start spread is optimization
variation, **not a Monte Carlo standard error**. The finite iteration cap does
not certify globally optimized nulls; terminal gradients are reported below.
"""), code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
from su2_nqs.frontier.artifacts import canonical_sha256
from su2_nqs.frontier.provenance import file_sha256

root = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p/'su2_nqs').is_dir())
project = root/'projects/clebsch_tree_nqs'
folder = project/'results/spin_conditional_carrier_resonance_v1'
resonance = json.loads((folder/'results.json').read_text())
registration = json.loads((folder/'registration.json').read_text())
assert resonance['registration_sha256'] == registration['sha256']
assert registration['sha256'] == canonical_sha256(
    {k:v for k,v in registration.items() if k != 'sha256'})
assert resonance['result_sha256'] == canonical_sha256(
    {k:v for k,v in resonance.items() if k != 'result_sha256'})
assert resonance['states_sha256'] == file_sha256(folder/'states.npz')
arms = ('factorized', 'diagonal_conditioned', 'conditional', 'site_scrambled')
labels = {'factorized':'Factorized', 'diagonal_conditioned':'Joint diagonal\n+ shared rotation',
          'conditional':'Conditional\nrotation', 'site_scrambled':'Site-scrambled\nconditional'}
colors = {'factorized':'#777777', 'diagonal_conditioned':'#D55E00',
          'conditional':'#0072B2', 'site_scrambled':'#CC79A7'}
coherence_labels = {'factorized':'Factorized', 'diagonal_conditioned':'Joint diag. + shared',
                    'conditional':'Conditional', 'site_scrambled':'Site-scrambled'}
starts = registration['config']['starts']
start_seeds = [s['seed'] for s in starts]
assert len(starts) == 3
start_markers = ('o', '^', 's')
offsets = np.linspace(-.13, .13, len(starts))
case_order = ('triangular_n16_j2_0125', 'square_n16_j2_05')
case_titles = ('Triangular 4x4, J2/J1=0.125; dim=14',
               'Square 4x4, J2/J1=0.5; dim=31')
fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
for row, (name, title) in enumerate(zip(case_order, case_titles)):
    case = resonance['cases'][name]
    e0 = case['exact_reference_energy_per_site']
    carrier = case['carrier']
    carrier_joint = np.asarray(case['carrier_coherence']['joint_singlet_triplet_coherence'])
    for index, arm in enumerate(arms):
        records = case['arms'][arm]
        assert len(records) == len(starts)
        by_seed = {r['start']['seed']:r for r in records}
        assert set(by_seed) == set(start_seeds)
        for k, seed in enumerate(start_seeds):
            record = by_seed[seed]
            for col, value in enumerate((record['energy_per_site']-e0,
                                         record['variance_per_site'])):
                axes[row,col].scatter(index+offsets[k], value,
                    marker=start_markers[k], color=colors[arm], s=39,
                    edgecolor='white', linewidth=.45, zorder=4)
            # Signed, unconditioned joint expectation: no division by rare p(z).
            joint = np.asarray(record['coherence']['joint_singlet_triplet_coherence'])
            axes[row,2].plot(range(6), joint-carrier_joint,
                marker=start_markers[k], color=colors[arm], lw=.85, ms=3.5,
                alpha=.6, label=coherence_labels[arm] if k==0 else None)
    for col, baseline in enumerate((carrier['energy_per_site']-e0,
                                    carrier['variance_per_site'])):
        ax = axes[row,col]
        ax.axhline(baseline, color='black', lw=1, ls='--', label='Unchanged carrier')
        ax.set(xticks=range(4), xticklabels=[labels[a] for a in arms],
               xlim=(-.45,3.45), ylabel=('(E - exact E0) / N' if col==0 else 'Var(H) / N'))
        # Preserve zero and roundoff-negative values rather than clipping them.
        ax.set_yscale('symlog', linthresh=1e-12)
        values = [baseline] + [
            record['energy_per_site']-e0 if col==0 else record['variance_per_site']
            for records in case['arms'].values() for record in records]
        lo, hi = min(values), max(values)
        # Default linear margins otherwise create a huge empty negative branch.
        bottom = min(-.35e-12,lo*1.25) if lo<=0 or hi/max(lo,1e-300)>1e5 else lo*.8
        ax.set_ylim(bottom, hi*1.3)
        ax.grid(axis='y', alpha=.18)
        ax.tick_params(axis='x', labelsize=8)
        ax.legend(frameon=False, fontsize=8)
    ranks = case['construction']
    axes[row,0].set_title(title+'\nExact energy error', fontsize=10)
    axes[row,1].set_title('Exact variance: every terminal state', fontsize=10)
    axes[row,2].axhline(0, color='black', lw=1, ls='--')
    axes[row,2].set(xticks=range(6),
        xticklabels=('(0,0)','(0,1)','(2,0)','(2,1)','(2,2)','(4,0)'),
        xlabel='Spectator reduced label (q,a), q=2S',
        ylabel=r'$\Delta\langle P_{q,a} X_{\mathrm{active}}\rangle$',
        title=f"Joint singlet resonance change\nPhysical/scrambled span ranks {ranks['physical']['rank']}/{ranks['scrambled']['rank']}")
    axes[row,2].title.set_fontsize(10)
    axes[row,2].grid(alpha=.15)
    axes[row,2].legend(frameon=False, fontsize=7, ncol=2)
fig.suptitle('Local conditional filter inside C — all 3 starts; no statistical error bars', fontsize=12)
out = project/'figures/exploration'
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out/'fig116_conditional_carrier_resonance.png', dpi=180)
fig.savefig(out/'fig116_conditional_carrier_resonance.pdf')
plt.show()

table = ['| Lattice / arm / start | Parameters | E/N | Error/N | Var(H)/N | Iterations | Max gradient | Stop |',
         '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
for name in case_order:
    case = resonance['cases'][name]
    geometry = name.split('_')[0]
    e0 = case['exact_reference_energy_per_site']
    c = case['carrier']
    table.append(f"| {geometry}: carrier | — | {c['energy_per_site']:.12f} | {c['energy_per_site']-e0:.4g} | {c['variance_per_site']:.4g} | — | — | fixed |")
    for arm in arms:
        for record in case['arms'][arm]:
            label = f"{geometry}: {arm}, {record['start']['seed']}"
            stop = record['message'].replace('|','/')
            table.append(f"| {label} | {record['parameters']} | {record['energy_per_site']:.12f} | {record['energy_per_site']-e0:.4g} | {record['variance_per_site']:.4g} | {record['iterations']} | {record['gradient_max']:.3g} | {stop} |")
display(Markdown('\n'.join(table)))
for name in case_order:
    case = resonance['cases'][name]
    print('\nCase:', name, '\nCarrier:', case['carrier_role'])
    print('Sector dimension / retained carrier weight:', case['sector_dimension'],
          case['carrier_projection_retained_weight'])
    print('Identity tangent comparison:', case['identity_tangent'])
    print('Complex linear-envelope ceiling (not nonlinear optimum):',
          case['single_filter_linear_span_ceiling'])
    print('Largest spin-raising residual:', max(
        r['spin_raising_residual'] for records in case['arms'].values() for r in records))
print('\nN36 cost only, not sampled energy:', resonance['n36_cost_profile'])
print('New ED / remote jobs:', resonance['new_ed_diagonalizations'], resonance['remote_jobs'])
print('Registration / result:', resonance['registration_sha256'], resonance['result_sha256'])
"""), markdown(r"""
### What this comparison can and cannot establish

The right panels show changes in the **actual signed expectation**
$\langle P_{q,a}X\rangle$, where
$X=|ss\rangle\langle(tt)_{S=0}|+\mathrm{h.c.}$ for the two active dimers.
They use the same physical support for every arm and do not divide by a small
spectator probability. They are neither attention maps nor probabilities.
A changed coherence by itself is not evidence of a lower-energy physical
mechanism; the energy and variance comparisons remain necessary.

Improvement over the factorized null alone can reflect joint reweighting.
Improvement over the diagonal-conditioned null would instead support the
usefulness of **spectator-dependent off-diagonal reduced generators** in this
bounded local family. Three terminal starts are optimization diagnostics,
not confidence intervals; iteration-limited runs must not be called globally
optimized controls. Parameter counts and generator boxes differ, so this is
not an equal-capacity or matched-work superiority test.

**Observed outcome:** triangular diagonal-conditioned, full conditional and
site-scrambled arms all reach the stored exact energy within numerical
precision. Their 14-dimensional space cannot separate the off-diagonal
mechanism. On square, the median full-conditional endpoint has
$E/N=-0.5286176124$ and $\mathrm{Var}(H)/N=1.6747\times10^{-5}$, versus
$-0.5286167345$ and $2.3727\times10^{-5}$ for diagonal-conditioned, and
$-0.5285964252$ and $1.7304\times10^{-4}$ for C. Full conditional has lower
energy and variance than diagonal-conditioned in all three starts, but both
hit the 120-iteration cap with gradients of order $10^{-6}$. This is a
**bounded optimization lead**, not superiority to a fully optimized null.
Moreover, the physical and site-scrambled square banks have complex-linear
span ranks 24 and 16: their energy difference is **not a capacity-matched
locality advantage**. At identity the real conditional tangent rank is 23
versus 15 for the diagonal-conditioned null, with 2.55% of its squared
Frobenius norm outside that null tangent. This describes available local
directions, not the fraction of energy gained or of physical entanglement.

A single eight-site gate is still a finite local-operator family, even when
its coefficients are supplied by a network. The printed complex-linear
outer-envelope rank can saturate a 14- or 31-dimensional sector; its ceiling
is not the optimum of the real nonlinear filter. Neither small-space
saturation nor the fixed-query N36 cost probe establishes long-range ViT
expressivity, a QSL result, or a production variational win. Keep this section
in the exploration notebook. A later question would concern **overlapping
resonance composition and growing environmental context**, not simply more
independent low-dimensional additive directions.
""")]
    for cell in cells:
        cell.metadata['experiment_section'] = 'conditional_carrier_resonance_20260930'
    return cells


def carrier_mixture_comparison_cells():
    """Read frozen nested-span fits and their fresh paired full-H audit."""
    cells=[markdown(r"""
## Does the trained network improve on fully optimized old-state mixing?

Triangle $6\times6$, $J_2/J_1=0.125$, the same singlet/full-$A_1$ carrier.
The old 128-record audit is now explicitly **development**. A full-complex
Rayleigh–Ritz solve minimizes its real energy in $\{C,D_0\}$ and separately
in $\{C,D_0,\psi_{\mathrm{diagonal}}\}$ and $\{C,D_0,\psi_{\mathrm{control}}\}$.
No ridge, rank truncation, real-coefficient restriction or undertrained mixing
baseline is used. A lower nested-space *development* minimum is inevitable,
not a variational success.

Coefficients are frozen before drawing new 128 records from 16 independent
chains. All audit states share the same four amplitude/H-action columns;
coefficients are never refitted on the audit or inside its bootstrap.
The prespecified primary comparison is **network_control minus old_mixture**;
the diagonal extension is secondary, not an alternative winner to select.

These are optimized linear closures of already trained full-network endpoints,
**not a new backbone training or a fully optimized nonlinear neural family**.
Neither fit is FS-constrained; this is not another matched-distance run.
The original training cost is inherited and reported separately.
"""),code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Markdown,display
from su2_nqs.frontier.artifacts import canonical_sha256
from su2_nqs.frontier.provenance import file_sha256
root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs'
folder=project/'results/spin_schur_control_training_v1/mixture_comparison'
reg=json.loads((folder/'registration.json').read_text())
fit=json.loads((folder/'fit/results.json').read_text())
audit=json.loads((folder/'audit/results.json').read_text())
assert reg['registration_sha256']==canonical_sha256({k:v for k,v in reg.items() if k!='registration_sha256'})
assert audit['fit_seal']['fit_sha256']==file_sha256(folder/'fit/results.json')
assert fit['coefficients_sha256']==file_sha256(folder/'fit/coefficients.npz')
for name,digest in audit['files'].items(): assert file_sha256(folder/'audit'/name)==digest
assert not audit['coefficients_refit_on_audit'] and not audit['network_retrained']
arms=('old_mixture','network_diagonal','network_control','raw_diagonal','raw_control')
labels={'carrier':'Carrier','old_mixture':'Optimized old mixture',
    'network_diagonal':'Mixture + diagonal shape (secondary)',
    'network_control':'Mixture + control shape (primary)',
    'raw_diagonal':'Original diagonal endpoint','raw_control':'Original control endpoint'}
colors={'old_mixture':'#333333','network_diagonal':'#D55E00','network_control':'#0072B2',
        'raw_diagonal':'#C99974','raw_control':'#81B1D2'}
fig,axes=plt.subplots(2,2,figsize=(12,7.5),constrained_layout=True)
def interval(ax,x,entry,color,label=None):
    lo,hi=entry['chain_bootstrap_95']
    ax.vlines(x,lo,hi,color=color,lw=1.5)
    ax.hlines((lo,hi),x-.04,x+.04,color=color,lw=1.1)
    ax.scatter(x,entry['value'],color=color,s=34,zorder=4,label=label)
for column,(metric,title) in enumerate((('energy_per_site','Energy'),('variance_per_site','Variance'))):
    ax=axes[0,column]
    for x,arm in enumerate(arms):
        interval(ax,x,audit['paired'][arm+'-minus-carrier'][metric],colors[arm])
    ax.axhline(0,color='black',lw=.8,alpha=.6)
    ax.set(xticks=range(5),xticklabels=('Old mix','+ diagonal','+ control','Raw diag.','Raw control'),
           ylabel=('Delta E / N' if column==0 else 'Delta Var(H) / N'),
           title=f'({chr(97+column)}) {title} relative to the original carrier')
    ax=axes[1,column]
    for x,arm in enumerate(('network_diagonal','network_control')):
        observed=audit['paired'][arm+'-minus-old_mixture'][metric]
        interval(ax,x+.055,observed,colors[arm],label='Fresh audit: paired 95%' if x==0 else None)
        development=fit['development_moments'][arm][metric]-fit['development_moments']['old_mixture'][metric]
        ax.scatter(x-.055,development,facecolors='none',edgecolors=colors[arm],marker='D',s=38,
                   label='Development fit (not a CI)' if x==0 else None,zorder=4)
    ax.axhline(0,color='black',lw=.8,alpha=.6)
    ax.set(xticks=(0,1),xticklabels=('Diagonal shape\nsecondary','Control shape\nprimary'),
           xlim=(-.4,1.4),ylabel=('Extra Delta E / N' if column==0 else 'Extra Delta Var(H) / N'),
           title=f'({chr(99+column)}) Extra {title.lower()} change beyond optimized old mixing')
    ax.legend(frameon=False,fontsize=8,loc='upper left')
for ax in axes.flat: ax.grid(alpha=.15);ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3))
fig.suptitle('Frozen coefficients, new full-H chains — no backbone retraining',fontsize=12)
out=project/'figures/exploration';out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/'fig115_optimized_carrier_mixture_comparison.png',dpi=180)
fig.savefig(out/'fig115_optimized_carrier_mixture_comparison.pdf');plt.show()
table=['| Fixed state | E/N | Var(H)/N | Observed FS from C |', '| --- | ---: | ---: | ---: |']
for arm in audit['arms']:
    value=audit['arms'][arm]
    table.append(f"| {labels[arm]} | {value['energy_per_site']['value']:.9f} | {value['variance_per_site']['value']:.9f} | {audit['parent_distances'][arm]['fs']:.5f} |")
display(Markdown('\n'.join(table)))
table=['| Development fit | Rank | Gram condition | Largest LOCO state rotation (rad) |', '| --- | ---: | ---: | ---: |']
for arm,record in fit['fits'].items():
    entries=fit['leave_one_chain_out'][arm]['records']
    drift=max(e['full_development_distance_to_pooled']['fs'] for e in entries if e['passed'])
    table.append(f"| {labels[arm]} | {record['rank']} | {record['condition_number']:.4g} | {drift:.5f} |")
display(Markdown('\n'.join(table)))
for contrast in (audit['primary_contrast'],audit['secondary_contrast']):
    print(contrast,json.dumps(audit['paired'][contrast],indent=2))
print('New work / CPU seconds / peak GiB:',audit['ledger'],audit['seconds'],audit['peak_rss_bytes']/2**30)
print('Inherited neural training (not free):',audit['inherited_network_training'])
print('No automatic production or N64 release:',audit['production_authorized'],audit['n64_authorized'])
    """),markdown(r"""
**This run finds no resolved extra benefit.** Primary control-shape minus old
mix gives $\Delta E/N=+0.0001480\pm0.0002207$ and
$\Delta\mathrm{Var}(H)/N=+0.0004261\pm0.0004139$ (paired chain jackknife SE).
Both 95% bootstrap intervals contain zero, as do the secondary comparison's
intervals. Old mixing itself does not significantly improve on C. Thus the
result is neither a neural variational win nor proof that mixing explains all
possible neural benefits.

The error bars resample whole chains and describe the fixed endpoints, not
independently optimized roots on each bootstrap. They do not certify chain
equilibration or unseen-tail coverage. Development LOCO refits are diagnostic
only; the tail-bearing chain is never discarded and no LOCO endpoint is used.

Any resolved gain would show useful information in a **saved trained shape**
beyond old reference mixing. It would not establish an SU(2)/attention advantage
over matched non-symmetric networks or optimized VMC/Lanczos. Conversely, an
unresolved gain is not an equivalence proof or a no-go theorem for the full
neural manifold. No further samples, directions or training are appended to
this fixed-budget comparison. Keep this figure in exploration, not the article.
""")]
    for cell in cells:cell.metadata['experiment_section']='carrier_mixture_comparison_20260930'
    return cells


def carrier_residual_geometry_cells():
    """Plot sealed saved-array geometry only; no new physical evaluations."""
    cells=[markdown(r"""
## Why did lower force noise not yield a better state? Saved-residual geometry

This is a **post-hoc structural diagnosis** of the frozen triangular
$6\times6$, $J_2/J_1=0.125$ endpoints, not another experiment. It reads only
the sealed `structure/results.json`, `manifest.json` and `profiles.npz`.
No samples, neural amplitudes, JVPs, H actions or training steps are added.
The previously untouched 128-record audit has already been viewed and is
explicitly called **viewed audit** throughout this section.

In each displayed empirical Q-inner product, remove the carrier ray from
the initial decoder $D_0$, giving $s_0\propto P_C^\perp D_0$. Compare this
reference direction with the phase-aligned endpoint residual
$R=(\psi-bC)/b$, $b=\langle C,\psi\rangle_Q/\langle C,C\rangle_Q$.
The reference inherits a perturbed plaquette initialization; it is not the
physical pi/0-flux carrier and does not impose a hard $q=0$ cutoff.

Panel (a) shows pooled projections and **leave-one-chain-out (LOCO) ranges,
not confidence intervals**. Projection axes are recomputed within each
corpus and each omission. The combined development metric has no pooled
chain uncertainty: it mixes fresh128 and previously seen256 empirical Grams.
Panel (b) ranks each saved mass profile separately; the rank axes do not
match configurations between curves. Panel (c) reprojects each JVP locally
in each corpus; panel (d) instead transfers one chart fitted on cal64 without
refitting its coefficients on known256. These are different questions.
"""),code(r"""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Markdown,display
from su2_nqs.frontier.artifacts import canonical_sha256
root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs'
folder=project/'results/spin_schur_control_training_v1/structure'
geometry=json.loads((folder/'results.json').read_text())
manifest=json.loads((folder/'manifest.json').read_text())
assert geometry['sha256']==canonical_sha256({k:v for k,v in geometry.items() if k!='sha256'})
assert manifest['sha256']==canonical_sha256({k:v for k,v in manifest.items() if k!='sha256'})
assert geometry['manifest_sha256']==manifest['sha256']
assert hashlib.sha256((folder/'profiles.npz').read_bytes()).hexdigest()==geometry['files']['profiles.npz']
profiles=dict(np.load(folder/'profiles.npz',allow_pickle=False))
assert geometry['post_hoc'] and not geometry['new_variational_win_claim']
assert all(geometry['resources'][k]==0 for k in
    ('new_physical_samples','new_neural_amplitude_calls','new_H_calls','new_JVP_calls','training_steps'))
arms=geometry['arms']; colors={'diagonal':'#D55E00','control':'#0072B2'}
labels={'diagonal':'3Hz endpoint','control':'HC-control endpoint'}
corpora=('fresh_calibration','known_development','combined_development','viewed_audit')
corpus_labels=('Fresh calibration\n128 records','Seen development\n256 records',
               'Combined metric\n384 records','Viewed audit\n128 records')
probe_labels=('Joint 1','Joint 2','Base 1','Base 2','Attention 1','Attention 2')
fig,axes=plt.subplots(2,2,figsize=(12,8.2),constrained_layout=True)

# (a) Direct saved pooled values and chain-deletion extrema; not bootstrap CIs.
for index,arm in enumerate(arms):
    offset=(-.10,.10)[index]
    for location,corpus in enumerate(corpora):
        record=geometry['corpora'][corpus]
        value=record['pooled']['arms'][index]['reference']['projection_fraction']
        leave=[row['geometry']['arms'][index]['reference']['projection_fraction']
               for row in record['leave_one_group_out']]
        x=location+offset
        if leave:
            lo,hi=min(leave),max(leave)
            axes[0,0].vlines(x,lo,hi,color=colors[arm],lw=1.5,alpha=.75)
            axes[0,0].hlines((lo,hi),x-.04,x+.04,color=colors[arm],lw=1.2)
        axes[0,0].scatter(x,value,color=colors[arm],marker='D' if not leave else ('o','s')[index],
                          s=38,label=labels[arm] if location==0 else None,zorder=4)
axes[0,0].set(xticks=range(4),xticklabels=corpus_labels,ylim=(0,1.035),xlim=(-.4,3.4),
              ylabel=r'Fraction of $\Vert R\Vert_Q^2$ along $P_C^\perp D_0$',
              title='(a) Reference alignment: pooled + LOCO range (not CI)')
axes[0,0].tick_params(axis='x',labelsize=8)
axes[0,0].legend(fontsize=8,frameon=False,loc='lower left')

# (b) These are saved normalized row masses, not new wavefunction values.
styles=(('carrier','Carrier norm C','#555555','-'),
        ('reference',r'Reference $P_C^\perp D_0$','#CC79A7',':'),
        ('force',r'Carrier $(H-E_C)C$','#009E73','-.'),
        ('diagonal','3Hz residual','#D55E00','-'),
        ('control','HC-control residual','#0072B2','--'))
for key,label,color,linestyle in styles:
    mass=profiles[key]
    assert mass.shape==(128,) and np.all(mass>=0)
    np.testing.assert_allclose(mass.sum(),1.,atol=3e-14)
    axes[0,1].plot(np.arange(1,len(mass)+1),np.cumsum(np.sort(mass)[::-1]),
                  color=color,ls=linestyle,lw=1.6,label=label)
axes[0,1].set(xscale='log',xlim=(1,128),ylim=(0,1.035),xticks=(1,2,4,8,16,32,64,128),
              xticklabels=('1','2','4','8','16','32','64','128'),
              xlabel='Largest k rows (ranked separately for each profile)',
              ylabel='Cumulative observed weighted norm fraction',
              title='(b) Viewed audit: residual concentration differs from C')
axes[0,1].legend(fontsize=7.5,frameon=False,loc='lower right')

# (c) Fraction is relative to each probe after removing the carrier ray.
for key,label,color,marker in (('prior_calibration','Prior cal64','#555555','o'),
                              ('known_development','Seen256','#009E73','s')):
    values=geometry['pretraining_probes'][key]['innovation_norm_fraction']
    axes[1,0].plot(range(6),values,marker=marker,color=color,ms=4,lw=1.5,label=label)
axes[1,0].set(xticks=range(6),xticklabels=probe_labels,ylim=(-.02,1.035),
              ylabel='Fraction of carrier-perpendicular JVP norm remaining',
              title=r'(c) Six saved probes: remove C and $D_0$ within each corpus')
axes[1,0].legend(fontsize=8,frameon=False,loc='lower right')

# (d) Unlike panel c, the subtraction coefficients are frozen on prior cal64.
transfer=geometry['chart']['prior64_fitted_JVP_transfer']
for key,label,color,marker in (('prior_calibration','On fitting cal64','#555555','o'),
                              ('known_development','Transferred to seen256','#009E73','s')):
    fraction=[row['anchor_plane_fraction'] for row in transfer[key]]
    axes[1,1].plot(range(6),fraction,marker=marker,color=color,ms=4,lw=1.5,label=label)
for index in (4,5):
    value=transfer['known_development'][index]['anchor_plane_fraction']
    axes[1,1].annotate(f'{100*value:.1f}%',(index,value),xytext=(0,8),
                     textcoords='offset points',ha='center',fontsize=9,color='#007B57')
axes[1,1].set(xticks=range(6),xticklabels=probe_labels,ylim=(-.025,.55),
              ylabel=r'Fraction of fixed-chart innovation in local span(C,$D_0$)',
              title='(d) Fixed cal64 chart: orthogonality does not transfer exactly')
axes[1,1].legend(fontsize=8,frameon=False,loc='upper right')
for ax in axes[1]:
    ax.axvspan(3.6,5.4,color='#0072B2',alpha=.045,zorder=0)
    ax.tick_params(axis='x',labelrotation=20,labelsize=8)
for ax in axes.flat: ax.grid(alpha=.15)
fig.suptitle('Saved-array structural diagnosis — post hoc, no new states or sampling',fontsize=12)
out=project/'figures/exploration';out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/'fig114_carrier_residual_geometry.png',dpi=180)
fig.savefig(out/'fig114_carrier_residual_geometry.pdf');plt.show()

table=['| Corpus | Re normalized Gram (3Hz, control) | LOCO range, not CI |',
       '| --- | ---: | --- |']
for corpus,label in zip(corpora,corpus_labels):
    record=geometry['corpora'][corpus]
    value=record['pooled']['pairs'][0]['real_overlap']
    leave=[row['geometry']['pairs'][0]['real_overlap'] for row in record['leave_one_group_out']]
    interval=f'[{min(leave):+.4f}, {max(leave):+.4f}]' if leave else 'Not inferred for mixed metric'
    table.append(f"| {label.replace(chr(10),' ')} | {value:+.6f} | {interval} |")
display(Markdown('\n'.join(table)))
table=['| Viewed-audit endpoint | Left H interference / site | Right H interference / site | Left minus right / site |',
       '| --- | ---: | ---: | ---: |']
for index,arm in enumerate(arms):
    h=geometry['corpora']['viewed_audit']['pooled']['arms'][index]['hamiltonian']
    left=h['interference_left_per_site'];right=h['interference_right_per_site']
    table.append(f'| {labels[arm]} | {left:+.7f} | {right:+.7f} | {left-right:+.7f} |')
display(Markdown('\n'.join(table)))
print('The two finite-Q H cross terms are retained, not forced equal by Hermitization.')
print('Structure manifest/result hashes:',manifest['sha256'],geometry['sha256'])
print('Saved endpoint chart replay:',geometry['chart']['endpoints'])
print('Six-probe span only (NOT full parameter tangent rank):',geometry['six_probe_span'])
for arm in arms:
    event=geometry['events'][arm]
    print(arm,'dominant row/chain=',event['row'],event['chain'],
          'weighted fraction=',event['weighted_fraction'],'raw squared fraction=',event['raw_squared_fraction'],
          'proposal percentile=',event['proposal_midrank_percent'],
          'configuration/orbit grouped fractions=',event['configuration']['max_fraction'],event['orbit']['max_fraction'])
print('No extra resources:',geometry['resources'])
print('Source training gate remains',geometry['source_gate_passed'],
      '; new variational win:',geometry['new_variational_win_claim'],
      '; production authorized:',geometry['production_authorized'])
"""),markdown(r"""
### What the geometry does and does not establish

On the viewed audit, about 96% of each endpoint's carrier-orthogonal change
lies along the reference direction, and their signed normalized overlap is
approximately -0.965: these are almost **opposite directions**, not merely
the same direction with a smaller positive step. But the same observed row
contributes 92.1%/93.3% of residual norm, and deleting its chain changes the
reference-alignment estimate substantially. Neither global dominance nor
sampling coverage is thereby certified. The carrier-force norm is far less
concentrated than the added-state norm; these are different objects.

Attention is not shown to be algebraically dead. Its two saved random JVPs
retain substantial innovation outside C and D0, but these **six probes are
not the full 15315-parameter tangent space** and their existence proves no
energy advantage. A chart fitted once on cal64 is orthogonal there; on
seen256 the two attention innovations have **8.9% and 24.9%** of their norm
back in the local anchor plane. Finite-metric orthogonality is not a global
Hilbert-space statement, and refitting on each evaluation batch is forbidden.

The new global coordinate adapter reproduces both saved endpoint rays to
at most $8.45\times10^{-17}$ relative amplitude error, including evaluation
outside its fitting corpus. This is an algebraic replay, **not an energy
improvement**. Tied replay coordinates preserve the saved states; independently
training mixing and innovation gain would be a separate, explicit choice and
does not automatically preserve the old nonlinear state family. This section
selects no new endpoint, releases no larger-size/GPU budget and stays only in
the exploration notebook, not the PRL result notebook.
""")]
    for cell in cells: cell.metadata['experiment_section']='carrier_residual_geometry_20260930'
    return cells


def schur_control_training_cells():
    """Read sealed fixed-endpoint estimator training and its untouched audit."""
    cells=[markdown(r"""
## Matched-state-distance training: diagonal 3Hz versus known-carrier HC

Triangular $6\times6$ PBC, $J_2/J_1=0.125$. The exact singlet/full-$A_1$
three-determinant pi/0-flux carrier, six Schur-label branches and attention
architecture are unchanged. **All base, node-direction and attention
parameters train**; this is an estimator comparison, not a new ansatz.

Both arms make six preregistered attempted updates on the same 256-record
training corpus. Each direction gets one maximum-step evaluation and **14
fixed bisection evaluations** to match the measured FS step to 0.01. The
matching tolerance is 2.5%, with a between-arm step ratio at most 1.05.
Either both updates are accepted or neither is; rejected attempts remain
visible, with no extra steps or energy-based checkpoint selection.

The common **development metric** combines 128 fresh calibration records
and all 256 previously viewed force-audit records. Each corpus contributes
half the empirical carrier-normalized Gram. These seen 256 records are now
development data, not independent validation. They enter neither training
energy nor the terminal audit. The metric is not a certified global FS norm.

Only after both endpoints are sealed, **128 untouched audit records from 16
independent persistent chains** are generated. Reference and endpoint full-H
actions give physical energy and Var(H); neither the diagonal objective nor
the carrier-control surrogate supplies the reported variance. The displayed
95% intervals resample paired whole chains, not independent configurations.
No sampling, training, H action or new diagonalization occurs in this notebook.
"""),code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs'
folder=project/'results/spin_schur_control_training_v1'
case=folder/'n36'
training=json.loads((case/'train/results.json').read_text())
audit=json.loads((case/'audit/results.json').read_text())
metric=dict(np.load(case/'prepare/metric.npz'))
endpoints=dict(np.load(case/'train/metric_values.npz'))
registered=json.loads((folder/'registration.json').read_text())
spec=registered['config']; protocol=next(row for row in spec['cases'] if row['length']==6)
assert len(training['history'])==protocol['steps']==6
assert spec['bisection_steps']==14
assert audit['sample_count']==128 and audit['independent_chains']==16
assert np.count_nonzero(metric['corpus']==0)==128
assert np.count_nonzero(metric['corpus']==1)==256
for report in (training,audit):
    assert report['identity']['registration_sha256']==registered['registration_sha256']
names=('diagonal','control')
labels={'diagonal':'3Hz estimator','control':'Known-HC control'}
colors={'diagonal':'#D55E00','control':'#0072B2'}
markers={'diagonal':'o','control':'s'}
history=training['history']; steps=np.asarray([row['step'] for row in history])
accepted=np.asarray([row['accepted_pair'] for row in history],dtype=bool)

fig,axes=plt.subplots(2,2,figsize=(11,7.4),constrained_layout=True)
for index,arm in enumerate(names):
    rows=[row['arms'][arm] for row in history]
    values=np.asarray([row['training_objective_before'] for row in rows])
    values=values-values[0]
    radii=[row['match']['distance']['fs'] for row in rows]
    assert all(row['match']['evaluation_count']==15 for row in rows)
    axes[0,0].plot(steps,values,marker=markers[arm],color=colors[arm],
                    ls='-' if index==0 else '--',ms=4,label=labels[arm])
    axes[0,1].plot(steps,radii,marker=markers[arm],color=colors[arm],
                    ls='-' if index==0 else '--',ms=4,label=labels[arm])
    if np.any(~accepted):
        axes[0,1].scatter(steps[~accepted],np.asarray(radii)[~accepted],
                         marker='x',s=65,color=colors[arm],zorder=5)
    for ax,key in zip(axes[1],('energy_per_site','variance_per_site')):
        row=audit['paired'][arm+'-minus-carrier'][key]
        value=row['value'];lo,hi=row['chain_bootstrap_95']
        # Percentile intervals need not contain their point estimate. Draw
        # interval endpoints directly rather than clip negative yerr values.
        ax.vlines(index,lo,hi,color=colors[arm],lw=1.8)
        ax.hlines((lo,hi),index-.055,index+.055,color=colors[arm],lw=1.3)
        ax.scatter(index,value,marker=markers[arm],color=colors[arm],s=40,zorder=4)
axes[0,0].set(xlabel='Update (objective measured before step)',
              ylabel='Own surrogate change / site',
              title='Different training surrogates; not physical E',xticks=steps)
axes[0,0].legend(fontsize=8,frameon=False)
axes[0,1].axhline(protocol['step_fs'],color='black',ls=':',lw=1,label='Registered FS target')
axes[0,1].axhspan(protocol['step_fs']*(1-spec['relative_matching_tolerance']),
                  protocol['step_fs'],color='#009E73',alpha=.08)
axes[0,1].set(xlabel='Attempted update; x = pair rejected',ylabel='Measured FS step',
              title='Matched development-state steps',xticks=steps,
              ylim=(protocol['step_fs']*.974,protocol['step_fs']*1.005))
axes[0,1].legend(fontsize=8,frameon=False)
for ax,ylabel in zip(axes[1],(r'Paired $\Delta E/N$ versus carrier',
                            r'Paired $\Delta\mathrm{Var}(H)/N$ versus carrier')):
    ax.axhline(0,color='black',lw=.8,ls=':')
    ax.set(xticks=range(2),xticklabels=[labels[a] for a in names],ylabel=ylabel,
           title='Untouched full-H; paired whole-chain 95% CI',xlim=(-.45,1.45))
for ax in axes.flat: ax.grid(alpha=.15)
for ax in axes.flat: ax.title.set_fontsize(10)
fig.suptitle('Matched-step training, same SU(2)/A1 ansatz',fontsize=12)
out=project/'figures/exploration';out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/'fig113_schur_control_training.png',dpi=180)
fig.savefig(out/'fig113_schur_control_training.pdf');plt.show()

def saved_metric_distance(c,psi,q):
    before=c/np.sqrt(q);after=psi/np.sqrt(q)
    before_norm=np.vdot(before,before).real
    overlap=np.vdot(before,after)
    perpendicular=after-before*(overlap/before_norm)
    return float(np.arctan2(np.linalg.norm(perpendicular),abs(overlap)/np.sqrt(before_norm)))

print('Registration:',registered['registration_sha256'])
print('Accepted paired attempts:',int(accepted.sum()),'of',len(history))
for arm in names:
    combined=saved_metric_distance(metric['carrier'],endpoints[arm],metric['q'])
    np.testing.assert_allclose(combined,training['final_metric_distance'][arm]['fs'],atol=3e-12,rtol=1e-10)
    constituent={}
    for corpus,label in ((0,'fresh128'),(1,'previously_seen256')):
        select=metric['corpus']==corpus
        constituent[label]=saved_metric_distance(metric['carrier'][select],endpoints[arm][select],metric['q'][select])
    print(arm,'FS metric combined / constituent / untouched=',combined,constituent,audit['distances'][arm]['fs'])
    print('  Parameter changes by block:',training['parameter_changes'][arm])
    print('  Own measured CPU arm seconds:',training['arm_seconds'][arm])
    print('  Paired full-H changes:',audit['paired'][arm+'-minus-carrier'])
    print('  Norm / tail diagnostics:',{key:audit['arms'][arm][key] for key in
        ('weight_ess_fraction','max_weight_fraction','max_residual_chain_fraction','exact_node_samples','node_H2_contribution_fraction')})
print('Control minus diagonal:',audit['paired']['control-minus-diagonal'])
print('Carrier full-H audit:',audit['arms']['carrier'])
print('Shared gradient compile seconds:',training.get('shared_gradient_compile_seconds'))
print('CPU train / audit seconds:',training['seconds'],audit['seconds'])
print('Training / audit work ledgers:',training['ledger'],audit['ledger'])
print('N64 operational release gate:',audit['gate'])

# Retrospective explanation only: no new model/H calls and no gate changes.
# Project out the carrier ray in EACH resampled quadrature. At finite Q,
# <C|Hr> and conjugate(<r|HC>) need not agree; retain both terms.
raw=dict(np.load(case/'audit/values.npz'))
def residual_energy_parts(c,hc,v,hv,q,n_sites):
    inner=lambda a,b: np.sum(a.conj()*b/q)
    zc=inner(c,c).real;zv=inner(v,v).real
    b=inner(c,v)/zc;r=v-b*c;hr=hv-b*hc
    ec=inner(c,hc).real/zc;zr=inner(r,r).real
    left=(b.conjugate()*inner(c,hr)).real/zv/n_sites
    right=(b*inner(r,hc)).real/zv/n_sites
    quadratic=(inner(r,hr).real-ec*zr)/zv/n_sites
    return np.asarray([left,right,left+right,quadratic,zr/zv])
fields=('C-Hr cross/site','r-HC cross/site','interference sum/site',
        'quadratic residual/site','weight outside carrier ray')
chain_labels=np.unique(raw['groups']);rng=np.random.default_rng(spec['bootstrap_seed'])
draws=rng.choice(chain_labels,size=(spec['bootstrap_replicates'],len(chain_labels)),replace=True)
for k,arm in enumerate(names,1):
    def parts(indices):
        return residual_energy_parts(raw['psi'][indices,0],raw['hpsi'][indices,0],
            raw['psi'][indices,k],raw['hpsi'][indices,k],raw['q'][indices],36)
    point=parts(np.arange(len(raw['q'])))
    np.testing.assert_allclose(point[2]+point[3],
        audit['paired'][arm+'-minus-carrier']['energy_per_site']['value'],atol=3e-13)
    np.testing.assert_allclose(point[4],np.sin(audit['distances'][arm]['fs'])**2,atol=3e-13)
    replicas=np.asarray([parts(np.concatenate([np.flatnonzero(raw['groups']==g) for g in row]))
                         for row in draws])
    intervals=np.quantile(replicas,[.025,.975],axis=0)
    print('POST-HOC saved-array carrier/residual decomposition:',arm)
    for j,field in enumerate(fields):
        print(f'  {field}: {point[j]:+.7g}; whole-chain95% {intervals[:,j].tolist()}')
"""),code(r"""
# Conditional N64 data, if any, are reported without inventing a missing run.
from IPython.display import Markdown,display
n64=folder/'n64'
if (n64/'audit/results.json').exists():
    audit64=json.loads((n64/'audit/results.json').read_text())
    lines=['| N64 endpoint | Paired delta E/N [chain 95% CI] | Paired delta Var(H)/N [chain 95% CI] |',
           '| --- | --- | --- |']
    for arm in names:
        row=audit64['paired'][arm+'-minus-carrier']
        fields=[]
        for key in ('energy_per_site','variance_per_site'):
            item=row[key];lo,hi=item['chain_bootstrap_95']
            fields.append(f"{item['value']:+.6g} [{lo:+.6g}, {hi:+.6g}]")
        lines.append('| '+labels[arm]+' | '+' | '.join(fields)+' |')
    display(Markdown('\n'.join(lines)))
    print('N64 audit samples/chains:',audit64['sample_count'],audit64['independent_chains'])
    print('N64 scope: conditional size-transfer test only; 8x8 does not contain K.')
elif (n64/'train/results.json').exists():
    print('N64 fixed endpoints exist but no completed full-H audit. No N64 physical result is inferred.')
else:
    print('No completed N64 training/audit result. N36 gate decision:',audit['gate']['passed'])
"""),markdown(r"""
### Interpretation boundary

The upper-left curves are changes from initialization in **different finite-Q surrogate objectives**.
Their vertical ordering is not a comparison of physical variational energies;
only within-arm optimization is shown. Values are recorded before each
attempt, not a selected best checkpoint or an extra final training evaluation.
The bottom panels use the same fresh configurations and complete physical H
for both fixed endpoints and the carrier, preserving paired covariance.

Shared reference-H and carrier-H work is counted, not silently omitted.
Finite-sample normalized ratios can be biased. Whole-chain confidence
intervals, norm concentration and observed tail contributions do not prove
equilibration or unseen-tail coverage. Development-to-audit FS transfer and
the two constituent development metrics remain explicit diagnostics.

The retrospective carrier/residual decomposition uses only the saved full-H
arrays. It separates interference with the carrier from the residual's
quadratic contribution, retaining BOTH finite-Q cross terms. These terms
need not agree individually and the quadratic term need not be positive.
This is a finite-state decomposition, not a Taylor expansion; confidence
intervals of separate contributions must not be added.
The weight outside the carrier ray is $\sin^2 d_{\rm FS}$, not evidence of
lost configuration support or a QSL mechanism. It does not change any gate.

Even a resolved carrier improvement would test this training mechanism, not
an advantage over optimized two-Lanczos, projected VMC or fixed-Sz ViT.
The N64 release gate is operational; its variance allowance is not a proof
of a variance win. No result in this section automatically authorizes GPU
production, a QSL identification or promotion to the PRL figure notebook.
""")]
    for cell in cells: cell.metadata['experiment_section']='schur_control_training_20260930'
    return cells


def schur_carrier_control_cells():
    """Frozen-state force/noise experiment, not an optimized-energy figure."""
    cells=[markdown(r"""
## Same SU(2)/A1 manifold: does known carrier HC improve training forces?

Triangular J1-J2 model, J2/J1=.125, periodic 6x6 and 8x8. Keep the exact
singlet/full-A1 three-determinant pi/0 carrier and the same six-branch local
Schur-label attention. No training, added heads, q cutoff or residual H calls.
The compact CG contraction is algebraically equivalent; it keeps every group
image (432/768). N64 has its own certified AP/PA/AA carrier, not copied N36
boundary signs. 8x8 omits K and is a cost/force-transfer check, not phase evidence.

At the frozen carrier, compare the pure D=3Hz force with the homogeneous
Hermitian control using HC. The latter equals the usual complex-centered
full-H VMC force at C. The SU(2) identity permits cheap D treatment of later
singlet residuals, but a finite-Q surrogate is NOT a variational bound.

Six parameter-space directions are fixed before inspecting outcomes: joint,
base-only and attention-only, two seeds each. These probe a fully trainable
15315/57107-parameter model; they are not six learned parameters. Independent
64-row calibration normalizes each physical tangent once; 256 untouched
records from16 independent persistent chains compare both estimators with
common Q. Whole-chain paired resampling, not IID snapshot error bars. Keep
all configurations, especially carrier-small/tangent-large events.
"""),code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from su2_nqs.frontier.artifacts import canonical_sha256
root=next(p for p in (Path.cwd(),*Path.cwd().parents)if(p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs'
folder=project/'results/spin_schur_carrier_control_v1'
records={n:json.loads((folder/f'n{n}'/'results.json').read_text()) for n in (36,64)}
labels=['Joint 1','Joint 2','Base 1','Base 2','Attention 1','Attention 2']
fig,axes=plt.subplots(2,2,figsize=(11,7.6),constrained_layout=True)
for column,(n,r) in enumerate(records.items()):
    assert r['result_sha256']==canonical_sha256({k:v for k,v in r.items()if k!='result_sha256'})
    s=r['summary'];ax=axes[0,column]
    for name,label,marker in [('diagonal','3Hz','o'),('control','Known-HC control','s')]:
        chain=np.asarray(s['forces'][name]['chain_forces'])
        ax.plot(range(6),np.var(chain,axis=0,ddof=1),marker=marker,label=label,ms=4)
    ax.set(yscale='log',xticks=range(6),xticklabels=labels,ylabel='Across-chain directional-force variance',
           title=f'N={n}: fixed state, same samples and directions')
    ax.tick_params(axis='x',labelrotation=30);ax.legend(fontsize=8)
    value=s['variance_reduction'];lo,hi=s['variance_reduction_bootstrap95']
    axes[1,0].errorbar([n],[value],yerr=[[value-lo],[hi-value]],fmt='o',capsize=5,color=f'C{column}')
    axes[1,1].plot(range(6),s['audit_projective_norm_squared'],marker=['o','s'][column],label=f'N={n}',color=f'C{column}')
    print('N=',n,'parameters=',r['parameter_count'],'CPU process-wall s=',r['seconds'],
          'peak GiB=',r['peak_rss_bytes']/2**30,'variance reduction [paired bootstrap95]=',value,[lo,hi])
    print('No new state: carrier full-H E/N (not an optimized result)=',s['forces']['control']['energy_per_site'])
    print('Metric max single-row fractions=',s['tangent_max_sample_norm_fraction'])
    print('Force max sample/chain influence=',[(name,s['forces'][name]['max_sample_squared_influence_fraction'],
          s['forces'][name]['max_chain_squared_influence_fraction'])for name in ('diagonal','control')])
    print('Sampling/cost ledger=',r['sampling'],'JVP seconds=',r['timings'])
axes[1,0].axhline(1,color='k',ls=':',lw=1)
axes[1,0].set(xticks=[36,64],xlabel='N (different carriers / tangent spaces)',yscale='log',
              ylabel='Trace covariance ratio: 3Hz / control',title='Paired whole-chain bootstrap 95% interval')
axes[1,1].axhline(1,color='k',ls=':',lw=1,label='Independent calibration = 1')
axes[1,1].set(xticks=range(6),xticklabels=labels,yscale='log',ylabel='Audit physical tangent norm squared',
              title='Remaining issue: calibration-to-audit metric drift')
axes[1,1].tick_params(axis='x',labelrotation=30);axes[1,1].legend(fontsize=8)
for ax in axes.flat:ax.grid(alpha=.2)
fig.suptitle('Carrier control reduces sampled force noise; no variational win measured')
out=project/'figures/exploration';out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/'fig112_schur_carrier_control.png',dpi=180)
fig.savefig(out/'fig112_schur_carrier_control.pdf');plt.show()
"""),markdown(r"""
### Interpretation and next action

The noise ratio applies to the six preregistered directions under this common
defensive-Q protocol. HC reuses the carrier connections already needed for
Q; it costs no extra determinant queries HERE. Neither that ratio nor the
compact forward-JVP timing is an end-to-end training/GPU acceleration claim.
Unresolved random directional mean forces do not imply a zero full gradient.

Tangent support is less well covered than carrier norm: substantial metric
drift and large individual contributions remain despite benign norm ESS.
Do not discard these events, enlarge a physical trust radius based on the
noise ratio, or interpret short-chain bootstrap coverage as equilibrium proof.
A positive result justifies one separately registered, fixed-endpoint N36
full-gradient estimator comparison with independent full-H energy/variance
auditing. N64 does NOT proceed to training: audit/calibration tangent norm
squared differs by29--963x, and one sample accounts for78% of control-force
squared influence. Its80x point estimate is not a scaling advantage. Neither
result justifies new architecture growth or production GPU jobs.
This figure belongs to exploration, not the PRL result notebook.
""")]
    for cell in cells:cell.metadata['experiment_section']='schur_carrier_control_20260930'
    return cells


def schur_cutset_training_cells():
    """Fixed-endpoint joint training; raw data only, no optimizer in notebook."""
    cells=[markdown(r"""
## N36 joint Schur-conditioned training on a strong singlet carrier

Triangle6x6 PBC, J2/J1=.125. Attention, rank-free coefficients and spatial
scramble all start at the same analytic three-determinant pi/0-flux carrier.
All base tensors, node directions and coefficient generators are trainable.
The six-branch architecture is unchanged; full SU2/A1 is retained.

Six attempted Adam updates on a common128-row quadrature;64 independent
calibration rows set one common residual scale and cap finite FS distances.
The objective is the exact-singlet3Hz identity, with its correctly weighted
amplitude gradient. Reusing a finite training corpus allows overfitting: a
falling training curve is NOT evidence of lower variational energy.

Only after all three endpoints are sealed are32 new audit configurations
generated from16 independent chains. A positive carrier/uniform target plus
fixed exchange pushforward covers H-connected configurations. Its unknown
normalization cancels; chain resampling preserves paired covariance. The
small audit does not certify equilibrium, unseen tails, or convergence.
Physical Var(H) below comes from direct Hpsi, never from Var(3Hz).
"""),code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from su2_nqs.vmc.cutset_training import weighted_state_distance
root=next(p for p in (Path.cwd(),*Path.cwd().parents)if(p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs'
folder=project/'results/spin_schur_cutset_training_v1'
names=['attention','rank_free','scrambled']
labels=['Attention','Rank-free','Scrambled']
training={name:json.loads((folder/name/'results.json').read_text()) for name in names}
audit=json.loads((folder/'audit/results.json').read_text())
raw=dict(np.load(folder/'audit/values.npz'))
fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for i,(name,label) in enumerate(zip(names,labels)):
    r=training[name];h=r['history'];color=f'C{i}'
    y=[row['training_energy_3hz_per_site_before'] for row in h]+[r['final_training_energy_3hz_per_site']]
    axes[0,0].plot(range(7),y,marker=['o','s','^'][i],ls=['-','--',':'][i],ms=3,label=label,color=color)
    d=[0]+[row['calibration_parent_distance']['fs'] for row in h]
    axes[0,1].plot(range(7),d,marker=['o','s','^'][i],ls=['-','--',':'][i],ms=3,label=label,color=color)
    ad=weighted_state_distance(raw['psi'][:,0],raw['psi'][:,i+1],raw['q'])
    axes[0,1].scatter([6.3],[ad['fs']],marker='x',color=color)
    for ax,metric in zip(axes[1],['energy_per_site','variance_per_site']):
        item=audit['paired'][name+'-minus-carrier'][metric]
        lo,hi=item['chain_bootstrap_95'];value=item['value']
        ax.errorbar([i],[value],yerr=[[value-lo],[hi-value]],
                    fmt='o',capsize=4,color=color)
    print(name,'params=',r['parameters'],'seconds=',r['seconds'],
          'accepted updates=',sum(x['accepted_factor']>0 for x in h),
          'audit FS=',ad['fs'],'block updates=',r['parameter_changes_by_block'])
    print('full H:',audit['arms'][name])
    print('paired to carrier:',audit['paired'][name+'-minus-carrier'])
axes[0,0].set(xlabel='Attempted update',ylabel='Fixed-Q training 3Hz / N',title='Optimization only, not a physical audit')
axes[0,0].legend(fontsize=8)
axes[0,1].axhline(.075,color='k',ls=':',lw=1,label='Parent-distance cap on calibration')
axes[0,1].set(xlabel='Attempted update (x: untouched audit)',ylabel='FS angle to carrier',title='Calibration vs. independent state distance')
axes[0,1].legend(fontsize=7)
for ax,ylab in zip(axes[1],[r'Paired delta E / N',r'Paired delta Var(H) / N']):
    ax.axhline(0,color='k',lw=.8);ax.set(xticks=range(3),xticklabels=labels,ylabel=ylab)
    ax.set_title('Paired full-H change: whole-chain bootstrap 95% CI')
for ax in axes.flat:ax.grid(alpha=.15)
fig.suptitle('Bounded joint-training comparison: separate optimization from physical improvement')
out=project/'figures/exploration';out.mkdir(exist_ok=True,parents=True)
fig.savefig(out/'fig111_schur_cutset_joint_training.png',dpi=180)
fig.savefig(out/'fig111_schur_cutset_joint_training.pdf');plt.show()
print('Carrier:',audit['arms']['carrier'])
print('All paired comparisons:',audit['paired'])
print('Audit ledger:',audit['ledger'])
"""),markdown(r"""
### What this experiment can decide

Completed: all three arms show a nominal energy/variance decrease, but paired
uncertainties do not resolve it. Attention deltaE/N=-.001145+/- .000830
(whole-chain jackknife SE); bootstrap95% interval[-.002717,+.000958].
Attention versus scramble differs by only-2.40e-6+/-5.34e-6. No independent
geometric-attention advantage is established. Untouched FS=.09256/.08713/.09241
exceeds the calibration cap.075; finite calibration is not a global certificate.
All endpoints are retained without post-audit rescaling or step selection.

The free rank-matched table has15244 parameters versus15315 for attention
and scramble, and contains the attention output family. A neural-capacity
claim is therefore inappropriate. A useful result would be reproducible
geometric optimization bias or physically lower energy at tractable cost.
Negative/uncertain paired gains are retained. Do not choose another endpoint
or extend audit samples after looking at these results. Single-seed, six-step
results cannot establish asymptotic optimization capacity or a PRL-level win.

Shared carrier/reference H actions are evaluated once by exact linearity,
not omitted from the cost. Checkpoints retain Adam moments and the fixed
training corpus identity. Square strong-carrier benchmarking remains separate.
""")]
    for cell in cells:cell.metadata['experiment_section']='schur_cutset_training_20260930'
    return cells


def schur_cutset_cells():
    """Six-branch interface/cost evidence; never label frozen probes as energies."""
    cells=[markdown(r"""
## Six-branch Schur-conditioned attention: strict physical amplitudes and cost

One selected four-spin cell has six reduced labels (q,a):
(0,0),(0,1),(2,0),(2,1),(2,2),(4,0). Each branch retains the **complete**
magnetic multiplet. Attention changes node-specific reduced kernels; the
branches are coherently summed before the full A1 projection. Magnetic
indices and physical bitstrings never enter the network.

This is partially configuration-dependent attention, not a complete all-label
ViT or a demonstrated long-range-entanglement improvement. Same-rank free
conditional coefficients contain every attention output; the full free bank
is a larger expressivity upper bound. Actual parameter counts are reported,
not padded to manufacture equality. Spatial scramble changes only cell-pair
features, not physical tensors, Hamiltonian, or A1. Features here use axial-grid
coordinate differences, not periodic minimum-image triangular distances.

The triangular N36 test uses the existing exact-A1 three-determinant pi/0-flux
carrier and the centered residual D(theta)-D(reference), both contractions
counted. The square test is **residual-only**: no validated competitive square
N36 carrier callback is claimed. No training, new samples, ED or remote jobs.
Frozen H ratios are NOT expectation values or a variational-energy comparison.
"""),code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from su2_nqs.frontier.artifacts import canonical_sha256
root=next(p for p in (Path.cwd(),*Path.cwd().parents)if(p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs';folder=project/'results/spin_schur_cutset_v1'
results={};raw={}
for geometry in ('triangular','square'):
    path=folder/geometry/'results.json'
    if not path.exists():
        failure=folder/geometry/'failure.json'
        print(geometry, 'BOUNDED FAILURE' if failure.exists() else 'NOT COMPLETE',
              json.loads(failure.read_text()) if failure.exists() else '')
        continue
    r=json.loads(path.read_text())
    assert r['result_sha256']==canonical_sha256({k:v for k,v in r.items()if k!='result_sha256'})
    assert r['training_steps']==0 and r['new_samples']==0
    results[geometry]=r;raw[geometry]=dict(np.load(folder/geometry/'values.npz'))
    print(geometry,'passed=',r['passed'],'strong_carrier_ready=',r['strong_carrier_ready'],
          'wall seconds=',r['wall_seconds'],'peak GiB=',r['peak_rss_bytes']/2**30)
    for arm,a in r['arms'].items():
        m=r['measurements']
        print(arm,'parameters=',a['parameter_count'],'bank=',a['generated_bank_parameter_count'],
              'warm amp/H/JVP seconds=',[np.median(m[arm+'_'+k]['execution_seconds'])
                                        for k in ('amplitude','H','end_to_end_JVP')],
              'JVP relative FD error=',a['JVP_finite_difference_relative_error'],
              'A1 error=',a['candidate_A1_probe_relative_error'])
    print('physical-versus-scramble probes:',r['validation'])
if results:
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),constrained_layout=True)
    geometry=next(iter(results));r=results[geometry];d=raw[geometry]
    node=r['validation']['physical_remote_node_ids'][-1]
    for mode,style in (('physical','-'),('scrambled','--')):
        coefficients=d[mode+'_conditional_coefficients'][:,node]
        for rank in range(coefficients.shape[1]):
            axes[0].plot(np.arange(6),coefficients[:,rank],style,marker='o',ms=4,
                         label=f'{mode}, direction {rank+1}')
    axes[0].set(xticks=range(6),xticklabels=['0,0','0,1','2,0','2,1','2,2','4,0'],
                xlabel='Selected cell label (doubled q, a)',ylabel='Reduced-kernel coefficient',
                title=f'Remote node {node}, not containing selected cell')
    arms=('attention','scrambled','rank_free','full_bank')
    for g,r in results.items():
        times=[np.median(r['measurements'][a+'_H']['execution_seconds']) for a in arms]
        label='Triangle: carrier + centered residual' if g=='triangular' else 'Square: residual only'
        axes[1].plot(range(4),times,'o-',label=label)
    axes[1].set(xticks=range(4),xticklabels=['Attention','Scramble','Rank-free','Full bank'],
                ylabel='Warm seconds / 2 full-H centers',yscale='log',
                title='Single-process CPU probes, not time-to-accuracy')
    for ax in axes:ax.grid(alpha=.15);ax.legend(fontsize=7)
    fig.suptitle('Schur-conditioned remote tensors: interface evidence, no variational win yet')
    out=project/'figures/exploration';out.mkdir(exist_ok=True,parents=True)
    fig.savefig(out/'fig110_schur_cutset_interface_cost.png',dpi=180)
    fig.savefig(out/'fig110_schur_cutset_interface_cost.pdf');plt.show()
"""),markdown(r"""
### Physical interpretation and next decision

SU2 fixes magnetic degeneracies, not the remaining entanglement. For a singlet,
the reduced density matrix of a region has blocks
rho_A = direct-sum_j [p_j I_(2j+1)/(2j+1) tensor sigma_j]. A useful neural
correlator should learn the weights p_j and multiplicity correlations sigma_j,
not just produce attractive attention maps. Future physical diagnostics include
Schur-resolved entropy and conditional remote bond correlations
<P_q S_i.S_j>/<P_q> - <S_i.S_j>; q labels are basis-independent, individual
multiplicity labels a depend on the chosen Schur basis.

The six-label free coefficient table contains the attention family. Only a
bounded, equal-physical-distance/full-H-work training comparison can establish
a useful attention bias. If the table is equally good and cheaper, keep it:
that supports the conditional SU2 manifold, not a mandatory Transformer.
Do not expand to 6^9 branches or restart direct fusion-path calculations.

The measured H cost motivates reusing strict-singlet diagonal training:
P0 (3Hz) P0 = P0 H P0 makes both energy and the score-function gradient exact.
The new N8 attention-direction regression has energy/gradient errors
1.39e-17 /1.73e-18. It also verifies that Var(3Hz)=Var(H)+||(3Hz-H)psi||²:
the fixture gives .60334529=.25455890+.34878639, with the extra term in S=2.
Thus diagonal training variance is NOT the physical variance, and diagonal
sampling noise need not be smaller. Keep a fresh, fixed-endpoint node-safe
full-H audit; do not perform expensive full-H evaluation at every update.

Relevant primary literature already includes [non-Abelian and lattice-symmetric
NQS](https://arxiv.org/abs/2104.14869), [symmetry-constrained autoregressive
Transformer/RNN states](https://arxiv.org/abs/2101.07243), and
[neural backflow tensor networks](https://arxiv.org/html/2603.14425v1).
The candidate contribution must be a tractable symmetry-preserving interface
with physical energy/correlation gains, not simply the words SU2 + ViT.
""")]
    for cell in cells:cell.metadata['experiment_section']='schur_cutset_20260930'
    return cells


def decoder_importance_cells():
    """Independent amplitude-level second-moment audit, not a state improvement."""
    cells=[markdown(r"""
## Fixed N36 parent: independent core / H-connected importance audit

Same4492-parameter strict-SU2/full-A1 physical initialization as the preceding
proposal comparison. **This decoder is not a ViT.** No parameter update, new
ED or remote job.512 IID draws per geometry (16 independently keyed groups of32),
plus8 parent-action replay points. Prior Born arms have256 H measurements each;
their estimates below are context, not an equal-budget efficiency comparison.

Qbase=.9 Qcore+.1 U; Qpush(x)=sum_b p_b Qbase(T_b x), with p_b proportional
to |J_b| and unconditional swaps including nulls. Q=(Qbase+Qpush)/2 is exactly
normalized and has full fixed-m=0 support. Density uses the entire mixture.
At every point compute psi and Hpsi directly, including possible nodes:
z=|psi|²/Q, e=conj(psi) Hpsi/Q, m=|Hpsi|²/Q. Reuse the2x2 Gram of(psi,Hpsi).
E=sum(e)/sum(z), M2=sum(m)/sum(z); variance uses the stable residual
sum(|Hpsi-Epsi|²/Q)/sum(z). No ratio at psi=0, no second H application.

Uncertainty retains covariance: IID influence SE,16-group jackknife, and
2000 whole-group bootstrap replicates. Ratio estimates are finite-sample
biased; no finite dataset proves unseen-tail coverage. Core/H-only/outside-both
are **physical-bit support sets**, not SU2 spin channels or entanglement phases.
"""),code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from su2_nqs.frontier.artifacts import canonical_sha256
root=next(p for p in (Path.cwd(),*Path.cwd().parents)if(p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs';folder=project/'results/spin_decoder_importance_audit_v1'
geometries=('triangular','square')
results={g:json.loads((folder/g/'results.json').read_text())for g in geometries}
raw={g:dict(np.load(folder/g/'samples.npz'))for g in geometries}
old={(g,a):json.loads((project/'results/spin_decoder_sampler_comparison_v1'/g/a/'results.json').read_text())
     for g in geometries for a in ('nearest','global','frame_mix')}
for r in results.values():
    assert r['result_sha256']==canonical_sha256({k:v for k,v in r.items()if k!='result_sha256'})
    assert r['initial_parameters_sha256']==r['final_parameters_sha256']
    assert r['training_steps']==0 and r['measurement_H_centers']==512 and not r['parent_is_ViT']
out=project/'figures/exploration';out.mkdir(parents=True,exist_ok=True)
fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
for col,g in enumerate(geometries):
    d=raw[g];psi=d['psi'];hp=d['hpsi'];q=np.exp(d['log_proposal'])
    z=abs(psi)**2/q;e=psi.conj()*hp/q;m=abs(hp)**2/q
    stops=np.arange(32,513,32);energy=[];variance=[]
    for stop in stops:
        E=e[:stop].sum()/z[:stop].sum();M=m[:stop].sum()/z[:stop].sum()
        energy.append(E.real/36);variance.append((M-abs(E)**2)/36)
    for row,(series,field) in enumerate(((energy,'energy_per_site'),(variance,'variance_per_site'))):
        ax=axes[row,col];metric=results[g]['audit']['metrics'][field]
        ax.plot(stops,series,'o-',ms=3,color='#009E73',label='IID bridge cumulative estimate')
        ax.errorbar([512],[metric['value']],yerr=[metric['group_jackknife_se']],fmt='o',capsize=4,
                    color='#009E73',label='Final: 1 group-jackknife SE')
        for a,style,color in (('nearest',':','#777777'),('global','--','#0072B2'),('frame_mix','-.','#D55E00')):
            ax.axhline(old[g,a]['summary'][field],ls=style,color=color,lw=1,label='Previous Born '+a)
        ax.set(xlabel='Independent H centers (cumulative points are correlated)',ylabel='E / N' if row==0 else 'Variance / N')
        ax.grid(alpha=.15);ax.legend(fontsize=7)
    axes[0,col].set_title(g.capitalize())
fig.suptitle('Same parent: second-moment correction is not variational improvement')
fig.savefig(out/'fig108_decoder_importance_moments.png',dpi=180)
fig.savefig(out/'fig108_decoder_importance_moments.pdf');plt.show()
for g,r in results.items():
    print(g,'metrics (IID SE / jackknife / bootstrap95):',r['audit']['metrics'])
    print('norm:',r['audit']['norm_estimate'],'+/-',r['audit']['norm_iid_se'],
          'seconds:',r['active_seconds'],'H replay error:',r['replay_errors'])
"""),code(r"""
fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
for col,g in enumerate(geometries):
    parts=results[g]['support'];names=('core','H_only','outside_both')
    for field,marker,label in (('normalization_fraction','o','Norm fraction'),('second_moment_fraction','s','H-squared fraction'),
                              ('residual_fraction','^','Residual-variance fraction')):
        axes[0,col].plot(range(3),[parts[k][field]for k in names],marker=marker,label=label)
    axes[0,col].set(title=g.capitalize(),yscale='log',ylabel='Importance-weighted fraction',
                    xticks=range(3),xticklabels=['Core support','H-only support','Outside both'])
    d=raw[g];q=np.exp(d['log_proposal']);psi=d['psi'];hp=d['hpsi'];z=abs(psi)**2/q
    E=(psi.conj()*hp/q).sum()/z.sum();residual=abs(hp-E*psi)**2/q
    for values,label,color in ((z,'Norm','#0072B2'),(abs(hp)**2/q,'H squared','#777777'),(residual,'Residual variance','#D55E00')):
        ordered=np.sort(values)[::-1]
        axes[1,col].plot(np.arange(1,len(values)+1)/len(values),np.cumsum(ordered)/ordered.sum(),label=label,color=color)
    axes[1,col].set(xscale='log',xlabel='Fraction of IID samples (largest contribution first)',ylabel='Cumulative contribution')
    for ax in axes[:,col]:ax.grid(alpha=.15);ax.legend(fontsize=8)
    print(g,'support decomposition:',parts)
    print('Concentrations:',{k:results[g]['audit'][k]for k in
                            ('normalization_concentration','second_moment_concentration','residual_concentration')})
fig.suptitle('Tiny norm does not imply negligible residual; support labels are NOT spin-q sectors')
fig.savefig(out/'fig109_decoder_importance_support.png',dpi=180)
fig.savefig(out/'fig109_decoder_importance_support.pdf');plt.show()
"""),markdown(r"""
### Relation to the SU2 + long-range-neural-correlation goal

This is a fixed-parent reliability audit, not a new SU2-ViT ansatz or an energy
improvement. Attention has not yet entered this4492-parameter decoder.
The audit should close the repeated Born-sampling question at its registered
budget and support a bounded, independent validation of an actual state update.
The core-support classification alone cannot identify triplet fluctuations;
that would require physical block-spin projectors or correlation measurements.

For energy scale only, Fan et al. report CNN-MPS E/N=-0.50380737(5) on the
same square6x6 PBC J2/J1=.5 problem. This is a fully trained literature reference,
not a work-matched arm or a new ED calculation. Its reported advantage over
T-MPS also cautions against presuming attention superiority. See
[Fan et al., 2026, main text / Fig.2](https://arxiv.org/html/2603.14425v1).
Our initialized parent is not yet competitive. The eventual test must combine
strict SU2, genuinely configuration-dependent reduced correlations, controlled
contraction cost and physical energy/long-range-correlation improvements.
""")]
    for cell in cells:cell.metadata['experiment_section']='decoder_importance_20260930'
    return cells


def decoder_reuse_cells():
    """Exact same-state N36 query reuse and costs of training components."""
    cells=[markdown(r"""
## N36: exact bit-string reuse, then score-gradient and proposal costs

Same saved parameters, width2/all-q decoder, same eight probes and all432/288
A1 images as the preceding cost test. Merge repeated bonds, skip like-spin
offdiagonal terms, encode exact physical spin strings as uint64 and deduplicate
raw-amplitude queries across centers/connectors/images. The inverse map
retains every original projection weight. No model-invariance quotient or
physical-state change. Encoding is currently limited to63 sites.

End-to-end H includes rebuilding the device dedup plan. Cached-plan timing is
separate and must not be advertised as the ordinary moving-chain cost.
Chunked/rematerialized full A1 gives differentiable amplitudes; the covariance
force differentiates all4492 real parameters, **not the local energies**.
On these frozen non-Born probes it is only a cost/replay diagnostic.
Sampler blocks refresh current amplitudes and run1/16 proposals per chain,
not1/16 lattice sweeps. Every timing repeat restarts the same8 chains/key;
there is no equilibrium claim or parameter update. CPU only, no GPU budget.
The initial64-query tile was slower despite fewer queries. One registered
followup uses1024-query tiles; **both attempts are shown**, not only the best.
"""),code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from su2_nqs.frontier.artifacts import canonical_sha256
root=next(p for p in (Path.cwd(),*Path.cwd().parents)if(p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs'
results={(g,tile):json.loads((project/f'results/spin_decoder_reuse_v{v}'/g/'results.json').read_text())
         for g in ('triangular','square')for v,tile in ((1,64),(2,1024))}
for r in results.values():
    assert r['result_sha256']==canonical_sha256({k:v for k,v in r.items()if k!='result_sha256'})
    assert r['passed'] and r['training_updates']==0 and not r['equilibrium_sampling_claim']
out=project/'figures/exploration';out.mkdir(parents=True,exist_ok=True)
fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
colors={64:'#777777',1024:'#0072B2'}
for col,geometry in enumerate(('triangular','square')):
    for tile in (64,1024):
        r=results[geometry,tile];measures=[r['measurements'][k]for k in ('old_H','end_to_end_H')]
        y=np.array([np.median(m['execution_seconds'])for m in measures])
        low=np.array([min(m['execution_seconds'])for m in measures])
        high=np.array([max(m['execution_seconds'])for m in measures])
        axes[0,col].errorbar([0,1],y,yerr=[y-low,high-y],marker='o',capsize=3,color=colors[tile],label=f'Tile {tile}')
        axes[1,col].plot([0,1],[r['raw_query_count'],r['evaluated_count']],marker='o',color=colors[tile],label=f'Tile {tile}')
    axes[0,col].set_title(geometry.capitalize());axes[0,col].set_ylabel('CPU seconds / 8 centers')
    axes[1,col].set_ylabel('Raw queries (including padding)')
for ax in axes.flat:
    ax.set_xticks([0,1],['Original full A1','Exact query reuse']);ax.set_yscale('log');ax.grid(alpha=.15);ax.legend()
fig.suptitle('Same physical state and full H; end-to-end timing includes dedup planning')
fig.savefig(out/'fig96_decoder_n36_reuse.png',dpi=180);fig.savefig(out/'fig96_decoder_n36_reuse.pdf');plt.show()
"""),code(r"""
keys=('plan','cached_H','end_to_end_H','chunked_amplitude','score_force','sample_1','sample_16')
labels=('Plan only','H with cached plan','End-to-end H','A1 amplitude','Covariance force','Refresh + 1 proposal','Refresh + 16 proposals')
fig,axes=plt.subplots(2,1,figsize=(11,8),constrained_layout=True)
for ax,geometry in zip(axes,('triangular','square')):
    for offset,tile in ((-.12,64),(.12,1024)):
        r=results[geometry,tile];measurements=[r['measurements'][k]for k in keys]
        y=np.array([np.median(m['execution_seconds'])for m in measurements])
        low=np.array([min(m['execution_seconds'])for m in measurements]);high=np.array([max(m['execution_seconds'])for m in measurements])
        ax.errorbar(np.arange(len(keys))+offset,y,yerr=[y-low,high-y],fmt='o',capsize=3,color=colors[tile],label=f'Tile {tile}')
    ax.set_xticks(np.arange(len(keys)),labels,rotation=20,ha='right');ax.set_yscale('log')
    ax.set_ylabel('CPU seconds / 8 centers or chains');ax.grid(axis='y',alpha=.15);ax.legend();ax.set_title(geometry.capitalize())
fig.suptitle('Warm component costs; three-call ranges, not statistical uncertainty')
fig.savefig(out/'fig97_decoder_n36_training_cost.png',dpi=180);fig.savefig(out/'fig97_decoder_n36_training_cost.pdf');plt.show()
for (geometry,tile),r in results.items():
    print(geometry,'tile',tile,'unique queries',r['unique_count'],'padded evaluations',r['evaluated_count'],
          'peak CPU RSS GiB',r['peak_rss_bytes']/2**30)
    for name,m in r['measurements'].items():
        print(name,'median seconds',float(np.median(m['execution_seconds'])),
              'compile/lower',m['compile_seconds'],m['lowering_seconds'])
    print('validation',r['validation'])
"""),markdown(r"""
### What this does and does not establish

This experiment checks unchanged amplitudes/H, force directional derivatives,
unchanged same-key proposal decisions and measured implementation cost.
Exact duplicate elimination need not be a speedup: sorting, batch sizes and
memory traffic also matter. Compare end-to-end costs, not raw-query count
alone. These are frozen-probe CPU results; no trained energy, independent-chain
variance, mixing time, GPU efficiency or PRL-level physical win follows.

See `results/spin_decoder_reuse_v1/REPORT.md`. Article figures remain unchanged.
""")]
    for cell in cells:cell.metadata['experiment_section']='decoder_reuse_20260929'
    return cells


def decoder_cost_cells():
    """Frozen N36 amplitude/projection/H-ratio costs, without energy estimates."""
    cells=[markdown(r"""
## N36 strict-decoder cost preflight: amplitudes are not the full-H workload

Both 6x6 tori, unchanged reduced width2, complete local/inter-cell spin
channels, fresh fixed-seed **4492-real-parameter** decoder. The nine-cell tree
does not have the N16 parameter layout: no weight-transfer or retained-energy
claim is made. First1/8 sealed configurations per geometry, no filtering or
new sampling. Each geometry runs in a fresh process, serial CPU execution.

The exact A1 sum uses **all432 triangular /288 square images**. No ViT patch-
translation quotient is assumed. Existing full-H code evaluates all216/144
bonds in connector batches of4, including like-spin proposals. All numbers
are forward-inference costs, not gradients, Monte Carlo equilibration,
training progress or GPU throughput. Three warmed blocked calls after
separate lowering, compilation and first execution. See
`results/spin_decoder_cost_v1/REPORT.md`.
"""),code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from su2_nqs.frontier.artifacts import canonical_sha256
root=next(p for p in (Path.cwd(),*Path.cwd().parents)if(p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs';folder=project/'results/spin_decoder_cost_v1'
results={g:json.loads((folder/g/'results.json').read_text())for g in ('triangular','square')}
for r in results.values():
    assert r['result_sha256']==canonical_sha256({k:v for k,v in r.items()if k!='result_sha256'})
    assert r['passed'] and r['new_samples']==r['training_updates']==0
out=project/'figures/exploration';out.mkdir(parents=True,exist_ok=True)
fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
colors={'raw':'#0072B2','a1':'#D55E00'}
for row,(geometry,r) in enumerate(results.items()):
    for col,kind in enumerate(('amplitude','H')):
        for mode in ('raw','a1'):
            points=[r['measurements'][f'{mode}_{kind}_b{b}']for b in (1,8)]
            y=np.array([np.median(m['execution_seconds'])*1000/m['batch']for m in points])
            low=np.array([min(m['execution_seconds'])*1000/m['batch']for m in points])
            high=np.array([max(m['execution_seconds'])*1000/m['batch']for m in points])
            axes[row,col].errorbar([1,8],y,yerr=[y-low,high-y],color=colors[mode],marker='o',
                label='raw exact SU2'if mode=='raw'else'exact SU2 + full A1',capsize=3)
        axes[row,col].set_yscale('log');axes[row,col].set_xticks([1,8]);axes[row,col].grid(alpha=.15)
        axes[row,col].set_xlabel('Center batch size');axes[row,col].set_ylabel('CPU milliseconds / center')
        axes[row,col].set_title(f"{geometry.capitalize()}: {'amplitude'if kind=='amplitude'else'full H-ratio'}")
        axes[row,col].legend(fontsize=8)
fig.suptitle('Frozen N36 forward cost; whiskers are three-call ranges, not confidence intervals',fontsize=11)
fig.savefig(out/'fig94_decoder_n36_cost.png',dpi=180);fig.savefig(out/'fig94_decoder_n36_cost.pdf');plt.show()
"""),code(r"""
fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
for col,(geometry,r) in enumerate(results.items()):
    keys=list(r['measurements']);measurements=list(r['measurements'].values());x=np.arange(len(keys))
    labels=[k.replace('amplitude','amp').replace('_','\n')for k in keys]
    for field,label in [('lowering_seconds','lowering'),('compile_seconds','compilation')]:
        axes[0,col].plot(x,[m[field]for m in measurements],marker='o',label=label)
    axes[0,col].set_ylabel('Seconds');axes[0,col].legend(fontsize=8);axes[0,col].set_title(geometry.capitalize())
    axes[1,col].plot(x,[m['peak_rss_after_bytes']/2**30 for m in measurements],marker='o')
    axes[1,col].set_ylabel('Cumulative process peak RSS (GiB)')
    for ax in axes[:,col]: ax.set_xticks(x,labels,fontsize=8);ax.grid(alpha=.15)
fig.suptitle('Compilation and CPU-process high-water memory; NOT GPU VRAM or per-kernel peaks',fontsize=11)
fig.savefig(out/'fig95_decoder_n36_compile_memory.png',dpi=180)
fig.savefig(out/'fig95_decoder_n36_compile_memory.pdf');plt.show()
for geometry,r in results.items():
    print('\n',geometry,'group images',r['images'],'bonds',r['bonds'],'parameters',r['parameter_count'])
    for key,m in r['measurements'].items():
        print(key,'median batch seconds',float(np.median(m['execution_seconds'])),
              'compiler temp bytes',m['compiler_memory_estimates'].get('temp_size_in_bytes'))
    print('validation',r['validation'])
    print('CG preparation seconds',sum(t['seconds']for t in r['cg_tables']),
          'dense table bytes',sum(t['bytes']for t in r['cg_tables']))
"""),markdown(r"""
### Interpretation boundary

The full spatial projector is numerically correct, but its raw tree is not
invariant under all nine patch translations. The ViT quotient cannot simply
be transplanted. Projection overhead must be measured in H-ratios as well as
isolated amplitudes: batching and intermediate memory differ substantially.

Compiler temporary-buffer estimates exclude significant constants, allocator
and compiler state. Process peak RSS is cumulative across executables and
cannot be called GPU VRAM or memory required by one production kernel.
The full-H implementation is unchanged; evaluating like-spin connectors and
repeated symmetry images can leave optimization opportunities. These are
measured implementation costs, not mathematical lower bounds.

No statistical physical energy follows from these untrained-model pointwise
ratios. No new N16 optimization, N36 training, MCMC, ED or remote submission.
The article notebook is unchanged.
""")]
    for cell in cells:cell.metadata['experiment_section']='decoder_cost_20260929'
    return cells


def vap_decoder_cells():
    """Read-only strict local-Schur compression and subsequent energy refinement."""
    cells=[markdown(r"""
## Can a strict physical Schur/CG decoder carry the good VAP wavefunction?

Use the **saved VAP amplitudes**, not new ED eigenvectors, as teachers.
Two fixed widths, two teacher/decoder-seed pairs on each N16 lattice; all
eight outcomes retained. The existing complete four-spin Schur front end
and balanced CG contractions have free layer-shared reduced tensors. Two
real decoders form a complex state, with **152 / 364 real parameters**.
No teacher is added to the output, no spin projector is in the decoder, and
no attention block is introduced. A1 remains imposed by exact physical
orbit sums; these exhaustive N16 sums are not a scalable projection claim.

Stage 1 fits $1-|\langle\psi_{\rm VAP}|\psi_{\rm decoder}\rangle|^2$ for
256 attempts. Stage 2 resets optimizer moments and minimizes physical H for
128 attempts, without changing the model. The fit objective is not energy.
This is teacher-assisted compression, not a matched-work competition with
the 3992-parameter ViT; teacher-generation cost must not be discarded in an
efficiency claim. Reference: `results/spin_vap_decoder_v1/REPORT.md`.
"""),code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from su2_nqs.frontier.artifacts import canonical_sha256
root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs';folder=project/'results/spin_vap_decoder_v1'
trials=[json.loads(p.read_text()) for p in sorted(folder.glob('*/results.json'))]
assert len(trials)==8
for t in trials:
    assert t['result_sha256']==canonical_sha256({k:v for k,v in t.items() if k!='result_sha256'})
out=project/'figures/exploration';out.mkdir(parents=True,exist_ok=True)
colors={2026093001:'#0072B2',2026093002:'#D55E00'}
fig,axes=plt.subplots(1,2,figsize=(11,4.3),constrained_layout=True)
for col,geometry in enumerate(('triangular','square')):
    for t in trials:
        if t['case']['geometry']!=geometry: continue
        points=[dict(step=0,**t['initial'])]+[v for v in t['trajectory'] if v['stage']=='fit']
        axes[col].plot([v['step'] for v in points],[max(v['teacher_infidelity'],1e-14) for v in points],
            c=colors[t['decoder_seed']],ls='-' if t['width']==1 else '--',
            label=f"D={t['width']}, pair {str(t['decoder_seed'])[-2:]}")
    axes[col].set_yscale('log');axes[col].set_xlabel('Fitting attempts');axes[col].grid(alpha=.15)
    axes[col].set_title(f'{geometry.capitalize()} 4x4 PBC');axes[col].legend(fontsize=8)
axes[0].set_ylabel('Teacher infidelity (display floor 1e-14)')
fig.suptitle('Small strict-SU2 decoders fitted to saved VAP states; all eight runs',fontsize=12)
fig.savefig(out/'fig92_vap_decoder_fidelity.png',dpi=180);fig.savefig(out/'fig92_vap_decoder_fidelity.pdf');plt.show()
"""),code(r"""
fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
keys=('energy_error_per_site','variance_per_site')
floors=(1e-12,1e-13)
for col,geometry in enumerate(('triangular','square')):
    for t in trials:
        if t['case']['geometry']!=geometry: continue
        points=[t['teacher_metrics'],t['stages']['fit']['metrics'],t['stages']['energy']['metrics']]
        offset=(-.06 if t['decoder_seed']==2026093001 else .06)+(-.02 if t['width']==1 else .02)
        for row,key in enumerate(keys):
            ys=[max(p[key],floors[row]) for p in points]
            axes[row,col].scatter(np.arange(3)+offset,ys,c=colors[t['decoder_seed']],s=28,
                marker='o' if t['width']==1 else 's',label=f"D={t['width']}, pair {str(t['decoder_seed'])[-2:]}")
            axes[row,col].plot(np.arange(1,3)+offset,ys[1:],c=colors[t['decoder_seed']],alpha=.6,
                ls='-' if t['width']==1 else '--')
    axes[0,col].set_title(f'{geometry.capitalize()} 4x4 PBC');axes[0,col].legend(fontsize=8)
    for row in range(2):
        axes[row,col].set_yscale('log');axes[row,col].grid(alpha=.15)
        axes[row,col].set_xticks(range(3),['VAP teacher','Decoder fit','Energy refined'])
        axes[row,col].axhline(floors[row],ls=':',lw=.7,c='0.5')
axes[0,0].set_ylabel('Energy error / site (floor 1e-12)')
axes[1,0].set_ylabel('Var(H) / site (floor 1e-13)')
fig.suptitle('Teacher-assisted refinement; values below display floors are not ranked',fontsize=12)
fig.savefig(out/'fig93_vap_decoder_energy.png',dpi=180);fig.savefig(out/'fig93_vap_decoder_energy.pdf');plt.show()
print('Raw endpoint values (no floors, no sampling uncertainties):')
for t in trials:
    f,e=t['stages']['fit']['metrics'],t['stages']['energy']['metrics']
    print(t['teacher'],t['width'],'parameters',t['parameter_count'],
          'fit infidelity',f['teacher_infidelity'],'refined error/site',e['energy_error_per_site'],
          'refined variance/site',e['variance_per_site'],'S2 residual',e['singlet_residual_norm'])
print('\nRandom-direction projective tangent rank LOWER BOUNDS:')
for t in trials:
    print(t['teacher'],t['width'],{stage:rank['rank_lower_bounds'] for stage,rank in t['tangent_ranks'].items()},
          'ambient dimension',t['tangent_ranks']['initial']['physical_projective_dimension'])
print('\nWork is not matched to the earlier ViT:')
for t in trials:
    print(t['teacher'],t['width'],
          'fit loss calls',t['stages']['fit']['training']['full_H_calls'],
          'refinement H calls',t['stages']['energy']['training']['full_H_calls'],
          'wall s',t['wall_seconds'])
print('\nPhysical 2x2-block spin weights and entropy (all refined states):')
for t in trials:
    cut=t['stages']['energy']['metrics']['cuts']['plaquette']
    print(t['teacher'],t['width'],'P(q=0,2,4)',cut['q_probabilities'],'entropy nats',cut['entropy_nats'])
"""),markdown(r"""
### What this check can and cannot establish

High fidelity demonstrates that the **particular teacher** can be compressed
at this finite size. Full local tangent rank, when observed, means access to
all local directions of this tiny symmetry sector, not a global universal-
approximation or large-size claim. Conversely, a nonzero fitting residual
under a fixed optimizer budget is not a rigorous capacity lower bound.

SU2 is exact even before the outer A1 projection. The tensors never depend
on physical magnetic configurations. The two seed pairs vary both the saved
teacher and decoder initialization; this is not two initializations fitted
to the very same teacher. The complete local spin sectors are retained.

A physical reminder: the global singlet has large local **S_cell=1** weight
(about 0.491 on the triangular rhombus, 0.377 on the square plaquette).
Its local singlet weights are about 0.471 and 0.601. Exact global SU2 does
not mean restricting the physical state to local singlets on one fixed
partition. Coherent sums over different partitions are a separate possibility.
These are two different finite Hamiltonians, not
a phase diagnosis or a controlled QSL-versus-AFM comparison.

The next scaling question is physical contraction, spatial projection and
sampling cost, not whether to add parameters to an already saturated N16
sector. No N36 training, new ED, GPU submission or article-notebook promotion.
""")]
    for cell in cells: cell.metadata['experiment_section']='vap_decoder_20260929'
    return cells


def full_vit_constraint_cells():
    """Sealed soft-spin-constraint outcomes, never run training in the notebook."""
    cells = [markdown(r"""
## Can a scalable spin constraint replace exact VAP? A bounded negative result

Same 3992-parameter physical-input ViT, both N16 lattices, both inherited
initialization seeds; 256 attempts per arm. The new loss is the **total**
$\langle H\rangle+\lambda\langle S^2\rangle$, with preregistered
$\lambda=0.25,1$. The $\lambda=0$ raw and VAP controls are sealed earlier
runs with the same parents and optimizer policy, not retrained controls.
Finite penalty is **not exact SU(2)**; lattice A1 stays exact.

Plot physical H, not the penalized `training.history.energy`. All N16
expectations are exhaustive, with no Monte Carlo error bars. Equal attempts
do not match actual backtracks/work or realized state distances. All eight
outcomes are retained. See `results/spin_full_vit_constraint_v1/REPORT.md`.
"""), code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from su2_nqs.frontier.artifacts import canonical_sha256
root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs'
folder=project/'results/spin_full_vit_constraint_v1'
reference=project/'results/spin_full_vit_vap_v1'
def sealed_result(path):
    r=json.loads(path.read_text())
    assert r['result_sha256']==canonical_sha256({k:v for k,v in r.items() if k!='result_sha256'})
    return r
trials=[sealed_result(p) for p in sorted(folder.glob('*/results.json'))]
assert len(trials)==8
names=sorted({t['trial'] for t in trials})
controls={name:{arm:sealed_result(reference/f'{name}_{arm}'/'results.json')
                 for arm in ('raw','vap')} for name in names}
out=project/'figures/exploration';out.mkdir(parents=True,exist_ok=True)
colors={2026092901:'#0072B2',2026092902:'#D55E00'}
fig,axes=plt.subplots(3,2,figsize=(10,8),constrained_layout=True)
keys=('energy_error_per_site','variance_per_site','spin_squared')
labels=('Energy error / raw-control error','Var(H) / raw-control variance',r'$\langle S^2\rangle$ / raw-control value')
for col,geometry in enumerate(('triangular','square')):
    for name in names:
        raw=controls[name]['raw']
        if raw['case']['geometry']!=geometry: continue
        group=sorted((t for t in trials if t['trial']==name),key=lambda t:t['penalty'])
        points=[raw['final']]+[t['final'] for t in group]
        x=[0]+[t['penalty'] for t in group]
        for row,key in enumerate(keys):
            axes[row,col].plot(x,[p[key]/raw['final'][key] for p in points],
                c=colors[raw['seed']],marker='o' if raw['seed']==2026092901 else 's',
                label=f"seed {str(raw['seed'])[-2:]}")
    axes[0,col].set_title(f'{geometry.capitalize()} 4x4 PBC')
    axes[0,col].legend(fontsize=8)
    axes[2,col].set_xlabel(r'Penalty $\lambda$ (total-energy objective)')
    for ax in axes[:,col]:
        ax.axhline(1,c='0.45',lw=.8,ls=':');ax.set_xticks([0,.25,1]);ax.grid(alpha=.15)
for row,label in enumerate(labels): axes[row,0].set_ylabel(label)
fig.suptitle('Spin purity improves in 8/8 arms; physical energy worsens in 8/8',fontsize=12)
fig.savefig(out/'fig90_full_vit_spin_constraint.png',dpi=180)
fig.savefig(out/'fig90_full_vit_spin_constraint.pdf');plt.show()
"""), code(r"""
fig,axes=plt.subplots(1,2,figsize=(10,4.4),constrained_layout=True)
for col,geometry in enumerate(('triangular','square')):
    for name in names:
        raw=controls[name]['raw'];vap=controls[name]['vap']
        if raw['case']['geometry']!=geometry: continue
        group=sorted((t for t in trials if t['trial']==name),key=lambda t:t['penalty'])
        x=[0]+[t['penalty'] for t in group]
        errors=[raw['pav']['energy_error_per_site']]+[t['postprojection_diagnostic']['energy_error_per_site'] for t in group]
        # Ratios expose within-sector deterioration without hiding seed dependence.
        axes[col].plot(x,np.array(errors)/errors[0],c=colors[raw['seed']],
            marker='o' if raw['seed']==2026092901 else 's',label=f"seed {str(raw['seed'])[-2:]}")
        print(name,'projected errors/site (raw PAV, lambda025, lambda1):',errors,
              'earlier VAP:',vap['final']['energy_error_per_site'])
    axes[col].axhline(1,c='0.45',ls=':',lw=.8);axes[col].grid(alpha=.15)
    axes[col].set_title(f'{geometry.capitalize()} 4x4 PBC')
    axes[col].set_xticks([0,.25,1]);axes[col].set_xlabel(r'Penalty $\lambda$');axes[col].legend(fontsize=8)
axes[0].set_ylabel('Projected energy error / ordinary PAV error')
fig.suptitle('The singlet component also worsens: not merely residual spin contamination',fontsize=11)
fig.savefig(out/'fig91_full_vit_constraint_singlet.png',dpi=180)
fig.savefig(out/'fig91_full_vit_constraint_singlet.pdf');plt.show()
print('\nEvery endpoint: physical energy/site, variance/site, <S2>, singlet weight')
for t in trials:
    f=t['final']
    print(t['trial'],t['arm'],*[f[k] for k in ('energy_per_site','variance_per_site','spin_squared','singlet_weight')])
print('\nN36 FROZEN probes: CPU median seconds per 8 centers; NOT energy estimates')
preflight=sealed_result(folder/'ratio_preflight.json')
assert preflight['n36_training_updates']==preflight['new_samples']==0
for row in preflight['entries']:
    times={k:float(np.median(v['execution_seconds'])) for k,v in row['timings'].items()}
    print(row['geometry'],times,'(H+packed S2)/H',1+times['S2_packed']/times['H'],
          'absolute ratio discrepancy',row['pointwise_absolute_error'])
"""), markdown(r"""
### Decision and physical interpretation

The normalized singlet component is worse in all eight constrained endpoints,
even though total singlet weight rises. Thus this particular finite-penalty
policy sacrifices quality **inside** the desired sector while suppressing
other sectors; it does not reproduce variation after exact projection.
This is not a no-go theorem for all penalties, optimizers or ViTs.

The new matrix-free S2 ratio kernel is correct on independent N16/full-action
and N36/all-pair checks. At N36 it needs 324 unlike-spin swaps, not 630 pairs;
measured packed-S2 runtime is about half the all-pair implementation. Separate
H+S2 execution still costs 2.42x/3.17x H alone (triangular/square CPU probes).
Joint ratio reuse may reduce that overhead. No GPU or sampled-convergence
advantage follows from eight frozen configurations.

Next: test whether an existing strict physical CG/local-Schur decoder can
compress the **saved VAP amplitudes**, then refine their energy, under a
fixed capacity/work budget. This separates expressivity from initialization
and optimization. No new decoder or decoder-training result is claimed here;
no lambda sweep, N36 training or remote production is automatically opened.
The article notebook remains unchanged.
""")]
    for cell in cells: cell.metadata['experiment_section']='full_vit_constraint_20260929'
    return cells


def full_vit_vap_cells():
    """Saved-data-only same-backbone singlet VAP/PAV comparison."""
    cells = [markdown(r"""
## Same backbone, different symmetry objective: exact singlet VAP (N=16)

The physical-input factored Spatial-ViT and its 3992 trainable parameters are
unchanged. Each saved checkpoint forks into ordinary fixed-$S^z$/A1 training
and variation **after** exact singlet projection (VAP). Read-only projection
of the ordinary endpoint is PAV. Separate the instantaneous projection gain
from subsequent training. Both lattices and both inherited initialization
seeds are retained; 256 attempted updates per arm, common exact sine-FS cap.

These are exhaustive-support deterministic energies, not Monte Carlo means:
no sampling error bars or two-seed significance claim. Equal update caps do
not equalize actual H calls, CPU time, or realized state distances. Dense P0
is a diagnostic and is **not** the proposed large-system implementation.
No new Hamiltonian diagonalization or remote job. See
`results/spin_full_vit_vap_v1/REPORT.md` for interpretation and limits.
"""), code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from su2_nqs.frontier.artifacts import canonical_sha256
root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs';folder=project/'results/spin_full_vit_vap_v1'
trials=[json.loads(p.read_text()) for p in sorted(folder.glob('*/results.json'))]
assert len(trials)==8
for t in trials:
    assert t['result_sha256']==canonical_sha256({k:v for k,v in t.items() if k!='result_sha256'})
pairs={name:{t['arm']:t for t in trials if t['trial']==name} for name in sorted({t['trial'] for t in trials})}
out=project/'figures/exploration';out.mkdir(parents=True,exist_ok=True)
colors={2026092901:'#0072B2',2026092902:'#D55E00'}
fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
labels=['Parent','P0 parent','Raw trained','PAV','VAP']
for col,geometry in enumerate(('triangular','square')):
    for name,pair in pairs.items():
        raw,vap=pair['raw'],pair['vap']
        if raw['case']['geometry']!=geometry: continue
        points=[raw['parent'],raw['frozen_projection'],raw['final'],raw['pav'],vap['final']]
        for row,key in enumerate(('energy_error_per_site','variance_per_site')):
            # Categories are distinct branches, not a sequential training path.
            x=np.arange(5)+(-.055 if raw['seed']==2026092901 else .055)
            axes[row,col].scatter(x,[p[key] for p in points],c=colors[raw['seed']],s=34,
                marker='o' if raw['seed']==2026092901 else 's',label=f"seed {str(raw['seed'])[-2:]}")
            axes[row,col].set_yscale('log');axes[row,col].set_xticks(range(5),labels,rotation=15)
            axes[row,col].grid(alpha=.15)
        axes[0,col].set_title(f'{geometry.capitalize()} 4x4 PBC')
    axes[0,col].legend(fontsize=8)
axes[0,0].set_ylabel('Energy error / site');axes[1,0].set_ylabel('Var(H) / site')
fig.suptitle('Projection benefit versus projected-loss training: all four parents retained',fontsize=12)
fig.savefig(out/'fig88_full_vit_vap_pav.png',dpi=180);fig.savefig(out/'fig88_full_vit_vap_pav.pdf');plt.show()
"""), code(r"""
fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for col,geometry in enumerate(('triangular','square')):
    for t in trials:
        if t['case']['geometry']!=geometry:continue
        initial=t['frozen_projection'] if t['arm']=='vap' else t['parent']
        trajectory=[dict(step=0,**initial)]+t['trajectory']
        for row,key in enumerate(('energy_error_per_site','variance_per_site')):
            axes[row,col].plot([x['step'] for x in trajectory],[x[key] for x in trajectory],
                c=colors[t['seed']],ls='-' if t['arm']=='vap' else '--',
                label=f"{t['arm'].upper()}, seed {str(t['seed'])[-2:]}")
            axes[row,col].set_yscale('log');axes[row,col].grid(alpha=.15)
    axes[0,col].set_title(geometry.capitalize());axes[1,col].set_xlabel('Attempted joint-backbone updates')
    axes[0,col].legend(fontsize=8)
axes[0,0].set_ylabel('Energy error / site');axes[1,0].set_ylabel('Var(H) / site')
fig.suptitle('Same FS cap and optimizer policy; realized physical distances/work differ',fontsize=12)
fig.savefig(out/'fig89_full_vit_vap_training.png',dpi=180);fig.savefig(out/'fig89_full_vit_vap_training.pdf');plt.show()
print('Exact-support endpoint and work ledger (no statistical errors):')
for t in trials:
    print(t['trial'],t['arm'],'E/N',t['final']['energy_per_site'],
          'Var/N',t['final']['variance_per_site'],'steps',t['training']['completed'],
          'H calls',t['training']['full_H_calls'],'wall s',round(t['wall_seconds'],2))
print('\nPhysical 2x2 cut: local doubled-q probabilities and entropy (nats)')
for name,pair in pairs.items():
    for label,state in [('P0 parent',pair['raw']['frozen_projection']),('PAV',pair['raw']['pav']),('VAP',pair['vap']['final'])]:
        cut=state['cuts']['plaquette']
        print(name,label,'p(q=0,2,4)',cut['q_probabilities'],'S',cut['entropy_nats'])
controls=json.loads((project/'results/spin_strong_carrier_innovation_v1/results.json').read_text())
assert controls['result_sha256']==canonical_sha256({k:v for k,v in controls.items() if k!='result_sha256'})
k4=controls['cases']['original/j0.125/K4']['states']
print('\nCached triangular Krylov references: different protocol/space, NOT equal-work controls')
for label,state in [('K2',controls['cases']['original/j0.125/K2']['states']['carrier']),
                    ('K4',k4['carrier']),('K5',k4['rank_matched_direct_krylov']),('K6',k4['rank_matched_conditional_krylov'])]:
    print(label,state['energy_error_per_site'],state['variance_per_site'])
"""), markdown(r"""
### Interpretation of this bounded test

VAP beats the frozen projected parent **and** the PAV endpoint in energy and
variance for all four inherited seeds. Triangular errors/site reach
$2.40\times10^{-9}$ and $1.58\times10^{-8}$; square reaches
$2.38\times10^{-5}$ and $1.03\times10^{-4}$. No parameters were added.
This supports training **inside the physical singlet image**, not merely
projecting an ordinary trained network at the end.

The essential limitation is not hidden: singlet-A1 dimensions are only 14
and 31, and P0 is not a scalable black-box operation. The square VAP networks
have only about 0.939 singlet weight *before* projection; discarding P0
destroys their energetic advantage. The physical projected states themselves
remain singlets. Local block probabilities/entropies change modestly and do
not establish a new phase. This is a positive N16 mechanism result, not yet
a large-system or PRL-level claim. No N36/GPU production is released.
""")]
    for cell in cells: cell.metadata['experiment_section']='full_vit_vap_20260929'
    return cells


def full_vit_matched_distance_cells():
    """Calibration-only radius matching with a separately frozen energy audit."""
    cells=[markdown(r"""
## Equal measured distance: damping direction or merely a smaller step?

Square $6\times6$, $J_2/J_1=0.5$; unchanged **4032-parameter fixed-Sz/A1**
step-1064 parent. Reuse the sealed $\lambda=0.1,1$ SR directions from the
two previous 64-state training batches. These are not new training seeds.
Each ray is independently scaled to sine-FS distance 0.01 and 0.02 on a
**fresh 512-record calibration pool**, using only amplitude overlap. All
eight checkpoints freeze before a distinct **1024-record, 64-chain audit**.
No candidate is adopted, no audit retuning, no long training or new ED.

The state distance is $d=\sqrt{1-|\langle r\rangle_0|^2/
\langle |r|^2\rangle_0}$, with $r=\psi_{\mathrm{candidate}}/\psi_0$.
Equality is empirical on the calibration pool, **not global**. On the audit
pool both candidate energies/variances are separately reweighted from the
same parent; neither candidate is incorrectly used as an unweighted parent.
Whole-chain bootstrap retains time dependence, pairs both directions, and
uses a 24-test energy/variance family (99.7917% per interval). Intervals do
not include missing-support or mixing bias. Distance intervals are descriptive.

The audit has tail-sensitive parent variance: the top 1% of records account
for 76.8% of observed variance (calibration: 95.2%). All directions are shown;
high ESS or Rhat near one is not proof that rare contributions are resolved.
"""),code(r"""
from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib.pyplot as plt
from su2_nqs.frontier.artifacts import canonical_sha256
root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs';folder=project/'results/spin_full_vit_matched_distance_v1'
out=project/'figures/exploration';out.mkdir(parents=True,exist_ok=True)
def sealed(path):
    value=json.loads(path.read_text());assert value['result_sha256']==canonical_sha256({k:v for k,v in value.items() if k!='result_sha256'});return value
result=sealed(folder/'results.json');registration=sealed(folder/'registration.json')
assert result['registration_sha256']==registration['result_sha256']
for row in result['candidates']:
    f=folder/row['match']['name']
    assert hashlib.sha256((f/'audit.npz').read_bytes()).hexdigest()==row['raw_sha256']
    assert hashlib.sha256((f/'candidate.pkl').read_bytes()).hexdigest()==row['match']['checkpoint_sha256']
spec=registration['config'];colors={'sr_original':'#b34a39','sr_damped':'#276ca0'}
labels={'sr_original':r'$\lambda=0.1$', 'sr_damped':r'$\lambda=1$'}
fig,axes=plt.subplots(2,3,figsize=(12,7.6),layout='constrained')
for i,seed in enumerate(spec['training_seeds']):
    ax=axes[i,0];ax.plot([0,.022],[0,.022],':',color='.55',label='calibration target')
    for arm in spec['directions']:
        rows=sorted([r for r in result['candidates'] if r['match']['seed']==seed and r['match']['direction']==arm],key=lambda r:r['match']['target'])
        target=np.array([r['match']['target'] for r in rows]);distance=np.array([r['audit']['sampled_fs_sine'] for r in rows])
        ax.plot(target,distance,'o-',color=colors[arm],label=labels[arm])
        for j,key in enumerate(('energy_per_site','variance_per_site'),1):
            point=np.array([r['uncertainty'][key]['difference'] for r in rows])*1000
            bounds=np.array([r['uncertainty'][key]['family_adjusted_interval'] for r in rows])*1000
            axes[i,j].errorbar(distance,point,yerr=np.stack((point-bounds[:,0],bounds[:,1]-point)),
                               fmt='o-',color=colors[arm],label=labels[arm],capsize=3)
    ax.set(xlabel='calibration sine-FS',ylabel='independent-audit sine-FS',title=f'Batch {seed%1000}: distance transfer')
    ax.legend(fontsize=8)
    for j,name in enumerate(('energy','variance'),1):
        axes[i,j].axhline(0,color='.6',lw=1);axes[i,j].set(xlabel='actual audit sine-FS',ylabel=rf'$10^3\,\Delta$ {name} / site',title=f'Candidate minus parent: {name}')
    for a in axes[i]:a.grid(alpha=.18)
fig.suptitle('Equal calibration distance, independent audit — all eight candidates\nErrors: paired-chain family-adjusted intervals; conditional reweighting, not converged energies',fontsize=12)
fig.savefig(out/'fig86_full_vit_matched_distance.png',dpi=180);fig.savefig(out/'fig86_full_vit_matched_distance.pdf');plt.show()
print('Full-H centers:',result['full_H_calls'],'Local wall seconds:',round(result['wall_seconds'],2))
"""),code(r"""
fig,axes=plt.subplots(1,3,figsize=(12,4.4),layout='constrained')
rows=result['comparisons'];x=np.arange(len(rows))
for j,key in enumerate(('energy_per_site','variance_per_site','sampled_fs_sine')):
    point=np.array([r['uncertainty'][key]['difference'] for r in rows])*1000
    bounds=np.array([r['uncertainty'][key]['family_adjusted_interval'] for r in rows])*1000
    for i,r in enumerate(rows):
        color='#794b9a' if r['independent_matching_warning'] else '#276ca0'
        axes[j].errorbar(i,point[i],yerr=[[point[i]-bounds[i,0]],[bounds[i,1]-point[i]]],
                         fmt='o' if not r['independent_matching_warning'] else 's',color=color,capsize=4)
    axes[j].axhline(0,color='.5',lw=1);axes[j].set_xticks(x,[f"{r['seed']%1000}\nr={r['target']:.2f}" for r in rows])
    axes[j].set(xlabel='old training batch / calibration radius',ylabel=r'$10^3$ (damped minus original)',title=key.replace('_per_site',' / site').replace('_',' '));axes[j].grid(axis='y',alpha=.2)
fig.suptitle('Is the more damped direction better? All adjusted energy/variance intervals include zero\nPurple squares: >10% independent distance mismatch; no audit-based rescaling',fontsize=11)
fig.savefig(out/'fig87_full_vit_damping_contrasts.png',dpi=180);fig.savefig(out/'fig87_full_vit_damping_contrasts.pdf');plt.show()
for r in rows:
    print(r['seed'],r['target'],'relative audit distance mismatch:',f"{100*r['audit_relative_distance_mismatch']:.1f}%")
print('\nPost-hoc description only: finite-state chord direction / metric concentration')
for r in rows:
    changes=[];concentrations=[]
    for arm in spec['directions']:
        name=f"batch_{r['seed']}_{arm}_fs{r['target']:.3f}"
        with np.load(folder/name/'audit.npz') as data:
            lr=data['log_ratios']
        z=np.exp(lr-lr.real.max());z*=np.exp(-1j*np.angle(z.mean()));z-=z.mean()
        changes.append(z)
        chain=(abs(z)**2).sum(axis=1).ravel();concentrations.append(chain.max()/chain.sum())
    cosine=np.vdot(changes[0],changes[1]).real/np.sqrt(np.vdot(changes[0],changes[0]).real*np.vdot(changes[1],changes[1]).real)
    print(r['seed'],r['target'],'orthogonal-chord cosine',round(cosine,6),
          'largest-chain FS numerator shares',np.round(concentrations,6))
"""),markdown(r"""
### Conclusion and boundary

Calibration matching succeeds to $10^{-6}$, but audit matching is not
guaranteed: batch 301 differs by 9.4%/9.2%, batch 302 by 35.9%/35.2% using
the pair-mean denominator. Stronger damping is actually **farther** from the
parent on the audit pool in all four pairs, so its previously smaller step
is no longer an explanation of the point estimates. Nevertheless, all four
paired energy **and** variance family-adjusted intervals include zero.
Batch 302 fails the independent distance-transfer reference. Thus **no
reproducible damping-direction advantage is established**; this does not
prove equality or that all earlier damping effects were merely shrinkage.
The post-hoc chord cosines are 0.72--0.83: damping changes the represented
direction on these configurations. In batch 301, one audit chain supplies
80--83% of the weak-damping distance numerator. No chain is discarded; these
descriptive diagnostics do not add a new selection criterion or prove mixing.

Do not retune on this audit, select a checkpoint, launch square long
training, or promote these diagnostic figures to the article notebook.
See `results/spin_full_vit_matched_distance_v1/REPORT.md` and the current
architecture discussion in `PRL_PRODUCTION_OUTLINE.md`. The next useful
SU2 question is a same-backbone projected-loss comparison at exact-small
support, not another round of scalar damping scans.
""")]
    for cell in cells:cell.metadata['experiment_section']='full_vit_matched_distance_20260929'
    return cells


def full_vit_step_audit_cells():
    """Single-step generalization, not a new training or candidate-energy record."""
    cells=[markdown(r"""
## Frozen square parent: does a single update generalize outside its batch?

Same square $6\times6$, $J_2/J_1=0.5$, step-1064 **4032-parameter fixed-Sz,
exact lattice A1** parent; no architecture change or exact spin-SU2 claim.
Two new 64-chain training batches each generate four **one-step** candidates:
inherited Adam, original ridge SR, SR with step size 0.003 (was 0.03), and SR
with damping 1 (was 0.1). Every candidate starts from the same parent, never
from a preceding candidate. None is adopted for continued training.

Two separate new parent-target streams (ordinary and tempered exchange) have
32 chains ×16 snapshots each. They do not inform the update. Candidate
full-H ratios on those configurations are **self-normalized, correlated
sampling estimates**, not independent candidate-Born measurements. Paired
bootstrap resamples whole chains within each stream; 16 comparisons
(eight candidates × energy/variance) use 99.6875% family-adjusted intervals.
Missing support and mixing bias are not covered by these intervals.

Keep the 24 previously fixed diagnostic centers and every unique single-bond
exchange neighbor. The `high` label means configurations from previous SR
high-local-energy chains, not an independently defined physical sector or
an eigenenergy. These probes are **not a Born ensemble**. Log-weight changes
relative to low diagnostic centers are gauge invariant but do not determine
region probability. Large changes near near-zero amplitudes can be physically
negligible. The 0.05 distance, 0.8 ESS and log(2) probe flags are reference
warnings, not certified global safety criteria.
"""),code(r"""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib.pyplot as plt
from su2_nqs.frontier.artifacts import canonical_sha256
root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs';folder=project/'results/spin_full_vit_step_audit_v1'
out=project/'figures/exploration';out.mkdir(parents=True,exist_ok=True)
def sealed(path):
    r=json.loads(path.read_text());assert r['result_sha256']==canonical_sha256({k:v for k,v in r.items() if k!='result_sha256'});return r
reg=sealed(folder/'registration.json');result=sealed(folder/'results.json');cfg=reg['config']
assert result['registration_sha256']==reg['result_sha256'] and result['adopted_candidates']==0
arms=[a['id'] for a in cfg['arms']];names=['Adam','Original SR','SR step/10','SR damping x10']
seeds=cfg['training_seeds'];colors=['#0072B2','#009E73','#D55E00','#CC79A7']
for r in result['candidates']:
    job=folder/f"batch_{r['seed']}"/r['arm']['id']
    assert hashlib.sha256((job/'candidate.pkl').read_bytes()).hexdigest()==r['checkpoint_sha256']
    assert hashlib.sha256((job/'audit.npz').read_bytes()).hexdigest()==r['raw_sha256']
fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
for row,seed in enumerate(seeds):
    rs=[next(r for r in result['candidates'] if r['seed']==seed and r['arm']['id']==a) for a in arms]
    for source,style,label in (('training','o-','Training'),('validation','s--','Independent validation')):
        for col,metric in enumerate(('sampled_fs_sine','reweighting_ess_fraction')):
            axes[row,col].plot(range(4),[r[source][metric] for r in rs],style,label=label)
    axes[row,0].axhline(.05,c='0.4',ls=':');axes[row,0].set_ylabel('Sampled FS sine distance')
    axes[row,1].axhline(.8,c='0.4',ls=':');axes[row,1].set_ylabel('Importance-weight ESS fraction')
    axes[row,2].plot(range(4),[r['probes']['central_high_minus_low_log_weight_change'] for r in rs],'o-',c='#D55E00')
    axes[row,2].axhline(np.log(2),c='0.4',ls=':',label='log(2) reference')
    axes[row,2].axhline(0,c='0.7',lw=.8);axes[row,2].set_ylabel('Change in mean log weight: high - low')
    for col in range(3):
        axes[row,col].set_xticks(range(4),names,rotation=15,fontsize=8);axes[row,col].grid(alpha=.15)
        axes[row,col].set_title(f'Training batch seed {seed}',fontsize=10)
axes[0,0].legend(fontsize=8);axes[0,2].legend(fontsize=8)
fig.suptitle('Same-parent, one-step candidates: typical-state distances versus fixed-region weight changes\nDiagnostics only; no candidate adopted and no global safety certificate',fontsize=11)
fig.savefig(out/'fig84_full_vit_step_generalization.png',dpi=180);fig.savefig(out/'fig84_full_vit_step_generalization.pdf');plt.show()
print('Parent independent validation:',result['parent_validation'])
for r in result['candidates']:
    print(r['seed'],r['arm']['id'],r['reference_flags'])
    print('Validation streams:',[(s['sampled_fs_sine'],s['delta_energy_per_site'],s['delta_variance_per_site']) for s in r['validation_by_stream']])
"""),code(r"""
fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for col,seed in enumerate(seeds):
    rs=[next(r for r in result['candidates'] if r['seed']==seed and r['arm']['id']==a) for a in arms]
    for row,quantity in enumerate(('energy_per_site','variance_per_site')):
        ax=axes[row,col]
        for i,r in enumerate(rs):
            d=r['uncertainty'][quantity];v=d['difference'];ci=d['family_adjusted_interval']
            ax.errorbar(i+.06,v,yerr=[[max(v-ci[0],0)],[max(ci[1]-v,0)]],fmt='o',color=colors[i],capsize=4)
            ax.scatter(i-.06,r['training']['delta_'+quantity],marker='x',c=colors[i])
        ax.axhline(0,c='0.4',lw=1);ax.set_xticks(range(4),names,rotation=15,fontsize=8);ax.grid(alpha=.15)
        ax.set_ylabel('Candidate - parent: '+('E/N' if row==0 else 'Var(H)/N'))
        ax.set_title(f'Training batch seed {seed}',fontsize=10)
fig.suptitle('One-step correlated-sampling changes, not equilibrated candidate energies\nCircles: independent parent pools, paired-chain 99.6875% intervals; crosses: fitting batch',fontsize=11)
fig.savefig(out/'fig85_full_vit_step_correlated_changes.png',dpi=180);fig.savefig(out/'fig85_full_vit_step_correlated_changes.pdf');plt.show()

with np.load(folder/'probes.npz') as p:
    centers=p['central_indices'];old_center_weights=2*p['parent_logs'][centers].real
    probe_parent_weights=2*p['parent_logs'].real
    old_contrast=old_center_weights.reshape(3,8).mean(axis=1)[2]-old_center_weights.reshape(3,8).mean(axis=1)[1]
print('Parent selected high-minus-low mean log weight:',old_contrast)
for r in result['candidates']:
    shift=r['probes']['central_high_minus_low_log_weight_change']
    print(r['seed'],r['arm']['id'],'candidate high-minus-low:',old_contrast+shift,
        'relative contrast multiplier:',np.exp(shift),'probe tails:',r['probes']['outside_training_centered_quantiles'])
    with np.load(folder/f"batch_{r['seed']}"/r['arm']['id']/'audit.npz') as raw:
        delta=2*raw['probe_log_ratios'].real;centered=delta-delta[centers[8:16]].mean()
        idx=np.argmax(abs(centered));before=probe_parent_weights[idx]-probe_parent_weights[centers[8:16]].mean()
        print('Largest-change point, log weight relative to low-center mean, before/after:',before,before+centered[idx])
    print('Paired inference:',r['uncertainty'])
print('Full-H centers:',result['full_H_calls'],'wall seconds:',result['wall_seconds'])
"""),markdown(r"""
### Interpretation and scope

**Completed result:** original SR passes the fitting and independent-parent
FS/ESS references in both batches, yet the old high/low probe contrast grows
10.0x/2.1x. Batch 302's fitting energy improvement becomes statistically
unresolved in validation, and variance improvement is not retained. No
significant global energy increase is established. Smaller steps reduce both
probe drift and gain; damping changes both direction and actual state
distance. Adam and small-step SR show negative conditional energy/variance
intervals in both batches, but this is not a long-run policy comparison.
The parent's observed variance is much lower than the older tail-dominated
audit, underscoring that the new pool can miss extreme events.

Passing a typical-state overlap test does not constrain all amplitude ratios.
Conversely a large ratio on a very low-weight configuration is not proof of
large global probability or higher energy. A one-step audit can locate an
early warning, but cannot by itself explain a failure after 128 evolving
updates. No candidate was selected or used to continue training. The complete
numerical interpretation and bounded next decision are maintained in
`results/spin_full_vit_step_audit_v1/REPORT.md`; these are exploration figures,
not paper evidence for an SU(2) or QSL improvement.
""")]
    for cell in cells:cell.metadata['experiment_section']='full_vit_step_audit_20260928'
    return cells


def full_vit_recheck_cells():
    """Training-randomness replication and frozen-square interventions only."""
    cells=[markdown(r"""
## N36 unchanged full-ViT: seed replication and frozen sampling diagnosis

The **4032-parameter, fixed-$S^z=0$, exact lattice $\Gamma-A_1$** architecture
is unchanged. This is **not** an exact spin-SU(2) result. Triangular
$J_2/J_1=0.125$: two new training-sampling seeds fork the **same step-1064
checkpoint**, each into original Adam and the previously registered ridge SR
policy. Each arm has 128 updates × 64 full-H centers. These are independent
training-noise replications, **not independent weight initializations**.
After freezing each endpoint, three disjoint audit streams supply 48 chains
and 1536 records (half the previous audit size). Independent endpoint
differences use **unpaired whole-chain bootstrap**, with 98.75% intervals for
the four primary tests (two seeds × energy/variance). Do not pool away a failed
seed, tune on these audits, or interpret intervals as equilibration guarantees.

Square $J_2/J_1=0.5$: **zero optimizer updates**. Freeze the old parent and SR
endpoint. Under each, evolve ordinary exchange and replica exchange from the
same 24 cold starts: eight uniform fixed-M states, eight prior low-energy
chains, eight prior high-energy chains. The last two cohorts are deliberately
biased diagnostics, **not Born samples**. Two independent sampler seeds, six
temperatures $\beta=0,0.1,0.25,0.5,0.75,1$. Ordinary 384 proposals/block versus
64/replica/block matches proposed-amplitude work; refreshes and swap overhead
are additional. Only the cold replica enters full-H measurements. Keep all
24 blocks or the fixed emergency time cap; no adaptive extension. Agreement
between starts is necessary but not sufficient for equilibrium. No averaging
of biased cohorts into a claimed ground-state energy.
Here beta is a power of Born weights, not the physical inverse temperature
of the spin Hamiltonian.
"""),code(r"""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib.pyplot as plt
from su2_nqs.frontier.artifacts import canonical_sha256
root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs';folder=project/'results/spin_full_vit_recheck_v1'
out=project/'figures/exploration';out.mkdir(parents=True,exist_ok=True)
def sealed(path):
    r=json.loads(path.read_text())
    assert r['result_sha256']==canonical_sha256({k:v for k,v in r.items() if k!='result_sha256'})
    return r
reg=sealed(folder/'registration.json');cfg=reg['config']
tri=[sealed(folder/'triangular'/f'seed_{seed}'/'results.json') for seed in cfg['triangular']['training_seeds']]
sq=[]
for state in cfg['square']['parameter_states']:
    for kernel in cfg['square']['kernels']:
        for seed in cfg['square']['seeds']:
            job=folder/'square'/f'{state}_{kernel}_{seed}';r=sealed(job/'results.json')
            assert r['registration_sha256']==reg['result_sha256']
            assert hashlib.sha256((job/'trajectories.npz').read_bytes()).hexdigest()==r['raw_sha256']
            assert r['optimizer_updates']==0 and not r['equilibrium_energy_claimed']
            sq.append(r)
for r in tri:
    assert r['registration_sha256']==reg['result_sha256'] and not r['independent_weight_initialization']
    for arm,t in r['training'].items():
        assert hashlib.sha256((folder/'triangular'/f"seed_{r['seed']}"/arm/'checkpoint.pkl').read_bytes()).hexdigest()==t['checkpoint_sha256']
    for arm,a in r['audits'].items():
        for stream in a['streams']:
            path=folder/'triangular'/f"seed_{r['seed']}"/f"audit_{arm}_{stream['seed']}.npz"
            assert hashlib.sha256(path.read_bytes()).hexdigest()==stream['raw_sha256']
colors={'adam':'#0072B2','ridge_sr':'#009E73'}
old=sealed(project/'results/spin_full_vit_optimization_v1/triangular/results.json')
fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
for row,q in enumerate(('energy','variance')):
    for i,r in enumerate(tri):
        for arm,offset in (('adam',-.08),('ridge_sr',.08)):
            a=r['audits'][arm];v=a[q+'_per_site'];ci=a[q+'_chain_bootstrap_95']
            axes[row,0].errorbar(i+offset,v,yerr=[[max(v-ci[0],0)],[max(ci[1]-v,0)]],
                fmt='o',capsize=4,color=colors[arm],label=arm if i==0 else None)
        d=r['comparison'][q+'_per_site'];v=d['difference'];ci=d['primary_family_bonferroni_98_75']
        axes[row,1].errorbar(i,v,yerr=[[max(v-ci[0],0)],[max(ci[1]-v,0)]],fmt='o',capsize=5,c='#009E73')
    for arm,a in old['audits'].items():
        if arm in colors:axes[row,0].axhline(a[q+'_per_site'],ls=':',lw=1,c=colors[arm])
    axes[row,1].axhline(0,c='0.4',lw=1)
    for col in range(2):
        axes[row,col].set_xticks(range(2),[str(r['seed']) for r in tri]);axes[row,col].set_xlabel('New training-sampling seed')
        axes[row,col].set_ylabel(('Energy / site' if row==0 else 'Var(H) / site')+(' difference: SR - Adam' if col else ''))
        axes[row,col].grid(alpha=.15)
axes[0,0].legend();axes[0,0].set_title('Frozen endpoint audits: 95% conditional intervals')
axes[0,1].set_title('Primary comparison: 98.75% conditional intervals')
fig.suptitle('Triangular 6x6: repeatability at the same inherited checkpoint\nDotted: earlier single-seed result, not included in primary inference',fontsize=11)
fig.savefig(out/'fig81_full_vit_seed_replication.png',dpi=180);fig.savefig(out/'fig81_full_vit_seed_replication.pdf');plt.show()
for r in tri:
    print('Training seed:',r['seed'],'conditional joint pass:',r['conditional_joint_pass'])
    print(r['comparison'])
    for arm,a in r['audits'].items():
        print(arm,{k:a[k] for k in ('energy_per_site','variance_per_site','split_chain_energy_rhat',
            'second_minus_first_half_per_site','half_drift_chain_standard_error','top_one_percent_variance_share')})
"""),code(r"""
fig,axes=plt.subplots(2,4,figsize=(15,7),sharex=True,sharey=True,constrained_layout=True)
cohorts=cfg['square']['cohorts'];cohort_colors=['#0072B2','#009E73','#D55E00']
for row,seed in enumerate(cfg['square']['seeds']):
    for col,(state,kernel) in enumerate((('parent','ordinary'),('parent','tempered'),('ridge_sr','ordinary'),('ridge_sr','tempered'))):
        r=next(r for r in sq if (r['seed'],r['state'],r['kernel'])==(seed,state,kernel))
        raw=np.load(folder/'square'/f'{state}_{kernel}_{seed}'/'trajectories.npz')
        e=raw['local_energies'].real/36;ax=axes[row,col]
        for i,(name,color) in enumerate(zip(cohorts,cohort_colors)):
            # Deliberately non-Born block-zero ratios can be huge; retain and
            # print them, but show post-transition trajectories on this axis.
            ax.plot(np.arange(1,len(e)),e[1:,i*8:(i+1)*8].mean(axis=1),'.-',lw=1,ms=3,c=color,label=name)
        ax.set_title(f"{state} / {kernel}\nseed {seed}, Rhat={r['summary']['descriptive_split_rhat']:.2f}",fontsize=9)
        ax.axvspan(max(0,len(e)-8),len(e)-1,color='0.5',alpha=.08)
        ax.set_xlabel('Block (matched proposal work)');ax.set_ylabel('Cohort local energy / site');ax.grid(alpha=.15)
        print('FROZEN',state,kernel,seed,r['summary'],'seconds',r['active_seconds'])
        print('Block-zero cohort local-energy means (non-Born):',e[0].reshape(3,8).mean(axis=1))
axes[0,0].legend(fontsize=7)
fig.suptitle('Square 6x6: frozen parameters, start-cohort memory under two kernels\nBlocks 1 onward; non-Born block 0 printed separately. NOT equilibrium energy estimates.',fontsize=12)
fig.savefig(out/'fig82_full_vit_square_frozen_cohorts.png',dpi=180);fig.savefig(out/'fig82_full_vit_square_frozen_cohorts.pdf');plt.show()

fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for r in sq:
    if r['kernel']!='tempered':continue
    h=r['history'];label=f"{r['state']}, seed {r['seed']}"
    axes[0].plot(cfg['square']['betas'],np.mean([s['local_move_fraction'] for s in h],axis=0),'o-',label=label)
    beta=np.array(cfg['square']['betas']);mid=(beta[1:]+beta[:-1])/2
    axes[1].plot(mid,np.mean([s['swap_acceptance'] for s in h],axis=0),'o-',label=label)
axes[0].set(xlabel='Sampling inverse temperature beta',ylabel='Nontrivial local-move fraction')
axes[1].set(xlabel='Midpoint of adjacent sampling betas',ylabel='Replica-swap acceptance')
for ax in axes:ax.grid(alpha=.15);ax.legend(fontsize=7)
fig.suptitle('Sampling diagnostic only: acceptance is not a certificate of equilibration')
fig.savefig(out/'fig83_full_vit_square_tempering.png',dpi=180);fig.savefig(out/'fig83_full_vit_square_tempering.pdf');plt.show()
reference=sealed(folder/'square/reference_check.json')
for label,d in reference['diagnostics'].items():
    print('Independent frozen full-H check',label,d['H_replay_max_absolute'],d['H_replay_max_relative'])
    print('Selected high minus low mean log Born weight (NOT basin mass):',d['high_minus_low_mean_log_weight'])
print('Total full-H centers:',sum(r['full_H_configuration_calls'] for r in tri+sq)+reference['full_H_configuration_calls'])
"""),markdown(r"""
### Interpretation boundary

**Completed outcome:** both new triangular seeds pass the registered joint
energy/variance comparison: SR E/N=-0.48117/-0.48836 versus Adam
-0.44570/-0.44356. This replicates the optimizer direction, not an identical
endpoint or a new variational energy record. On square, parent cohort
differences are 0.006--0.009/site; SR differences remain 0.496/0.544 under
ordinary exchange and 0.215/0.204 under tempering. SR Rhat remains 1.59/1.68
even with tempering. Its equilibrium energy is unresolved. Fixed high/low
configuration probes also show a +26.99 shift in relative log Born weight
after the old SR updates: deterministic state redistribution, not basin
probabilities. The evidence points to update-amplified sampling barriers,
without establishing the true equilibrium-energy change.

Only a repeatable triangular frozen-audit advantage supports an optimization
policy improvement for this checkpoint family. It does not establish an SU(2)
or QSL advantage, independent-initialization robustness, or competitive
literature energy. For square, ordinary and tempered kernels target the same
frozen Born state; a change in measured energy without changing parameters is
evidence about **sampling**, not variational improvement. Persistent start
memory leaves equilibrium energy unresolved. Even if tempering removes it,
rare-event/tail bias remains possible. Frozen sampler interventions cannot
prove that every earlier optimization step was sound. Final results and the
bounded next decision are in `results/spin_full_vit_recheck_v1/REPORT.md`.
""")]
    for cell in cells:cell.metadata['experiment_section']='full_vit_recheck_20260928'
    return cells


def full_vit_optimization_cells():
    """Same-model bounded optimizer comparison; retain every arm and audit."""
    cells=[markdown(r"""
## N36 full-ViT: bounded optimization-policy comparison (2026-09-28)

**No architecture change:** fixed $S^z=0$, exact lattice $\Gamma-A_1$, 4032
real parameters, the same full Spatial-ViT. This is **not exact spin SU(2)**.
From the frozen step-1064 model on each lattice, fork (1) inherited Adam,
$\eta=3\times10^{-4}$; (2) inherited Adam, $\eta=10^{-4}$; (3) sample-space
ridge SR, damping 0.1 and bounded physical-state update. All three begin with
the identical freshly thermalized 64-chain pool and RNG. Their later physical
samples diverge. Both Adam arms retain the parent's moments; SR has no Adam
moments. The batch-size increase relative to the older 32-chain run is common
to all arms and is **not isolated** as an experimental factor.

Fixed budget: **128 attempted updates × 64 full-H centers per arm**; emergency
cap 600 active training seconds per arm. No new carrier/head, checkpoint
selection, hyperparameter scan, remote job or ED. SR uses an explicit
$2M\times P$ score matrix and a $2M\times2M$ ridge solve, NOT a
score-matrix-free method. Its trust tests use sampled overlap/ESS with no
candidate-energy evaluations. This compares complete optimizer policies,
including trust control, rather than isolating preconditioning alone.

After all arms on a lattice are frozen, measure the parent and every endpoint
using disjoint audit seeds: 3 streams × 32 chains × 32 snapshots, burn 4608
exchange proposals and thin 72. All 24 audit seeds were fixed before training.
Bootstrap whole chains; independent endpoint differences are **unpaired**.
Primary family: SR minus original Adam, energy AND variance, on two lattices;
98.75% Bonferroni intervals are reported alongside marginal 95% intervals.
These are conditional sampling intervals, not proof of complete equilibration.
"""),code(r"""
from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib.pyplot as plt
from su2_nqs.frontier.artifacts import canonical_sha256
root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs'
out=project/'figures/exploration';out.mkdir(parents=True,exist_ok=True)
folder=project/'results/spin_full_vit_optimization_v1'
def sealed(path):
    r=json.loads(path.read_text())
    assert r['result_sha256']==canonical_sha256({k:v for k,v in r.items() if k!='result_sha256'})
    return r
registration=sealed(folder/'registration.json')
policies=[sealed(folder/g/'results.json') for g in ('triangular','square')]
refs=json.loads((project/'references/finite_torus_6x6_targets.json').read_text())
labels={'adam':'Adam 3e-4','adam_low_lr':'Adam 1e-4','ridge_sr':'Ridge SR'}
colors={'adam':'#0072B2','adam_low_lr':'#D55E00','ridge_sr':'#009E73'}
for r in policies:
    assert r['registration_sha256']==registration['result_sha256']
    for arm,t in r['training'].items():
        assert hashlib.sha256((folder/r['geometry']/arm/'checkpoint.pkl').read_bytes()).hexdigest()==t['checkpoint_sha256']
    for arm,a in r['audits'].items():
        for s in a['streams']:
            assert hashlib.sha256((folder/r['geometry']/f"audit_{arm}_{s['seed']}.npz").read_bytes()).hexdigest()==s['raw_sha256']

fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for col,r in enumerate(policies):
    for arm,t in r['training'].items():
        h=t['history'];x=np.arange(1,len(h)+1)*64
        for row,(key,quantity) in enumerate((('energy_before_update_per_site','energy'),('variance_before_update_per_site','variance'))):
            ax=axes[row,col];y=np.array([s[key] for s in h]);color=colors[arm]
            ax.plot(x,y,lw=.5,alpha=.2,c=color)
            # Fixed non-overlapping 16-update means; not confidence bands.
            blocks=[slice(i,min(i+16,len(h))) for i in range(0,len(h),16)]
            ax.plot([x[b].mean() for b in blocks],[y[b].mean() for b in blocks],c=color,lw=1.5,label=labels[arm])
            a=r['audits'][arm];v=a[quantity+'_per_site'];ci=a[quantity+'_chain_bootstrap_95']
            ax.errorbar(x[-1]+(list(labels).index(arm)-1)*95,v,
                yerr=[[max(v-ci[0],0)],[max(ci[1]-v,0)]],fmt='o',ms=4,c=color,capsize=3)
    for row,quantity in enumerate(('energy','variance')):
        ax=axes[row,col];a=r['audits']['parent'];v=a[quantity+'_per_site'];ci=a[quantity+'_chain_bootstrap_95']
        ax.errorbar(0,v,yerr=[[max(v-ci[0],0)],[max(ci[1]-v,0)]],fmt='D',c='black',ms=4,capsize=3,label='Parent audit')
        ax.set_xlabel('Training full-H center calls (audit endpoints slightly offset)')
        ax.set_ylabel('Energy / site' if row==0 else 'Var(H) / site');ax.grid(alpha=.15)
    ref=refs['triangular_6x6_torus' if r['geometry']=='triangular' else 'square_6x6_pbc']
    axes[0,col].axhline(ref['exact_energy_per_site'],ls='--',lw=1,c='0.3',label='Cached same-torus ED target')
    axes[0,col].set_title(f"{r['geometry'].capitalize()} 6x6, J2={r['j2_over_j1']}"+
        ('\nSR audit not equilibrated (Rhat=3.15)' if r['geometry']=='square' else ''),fontsize=11)
    axes[0,col].legend(fontsize=7);axes[1,col].set_yscale('log')
    print(r['geometry'],'matched full budget:',r['matched_training_work'],
        'primary energy+variance conditional pass:',r['primary_conditional_joint_improvement'])
    print('Primary SR - Adam:',r['comparisons']['ridge_sr_minus_adam'])
fig.suptitle('Same full-ViT, fixed budget: optimizer policy, not ansatz capacity\nThin: noisy training batches; thick: 16-update means; points: independent frozen audits',fontsize=11)
fig.savefig(out/'fig77_full_vit_bounded_optimization.png',dpi=180)
fig.savefig(out/'fig77_full_vit_bounded_optimization.pdf');plt.show()
"""),code(r"""
fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for col,r in enumerate(policies):
    for idx,arm in enumerate(('parent',*labels)):
        a=r['audits'][arm]
        for row,quantity in enumerate(('energy','variance')):
            ax=axes[row,col];v=a[quantity+'_per_site'];ci=a[quantity+'_chain_bootstrap_95']
            stream=a['stream_energies_per_site' if row==0 else 'stream_variances_per_site']
            ax.scatter(idx+np.array([-.08,0,.08]),stream,c=['#0072B2','#D55E00','#CC79A7'],s=32)
            ax.errorbar(idx+.16,v,yerr=[[max(v-ci[0],0)],[max(ci[1]-v,0)]],fmt='D',c='black',ms=4,capsize=3)
            ax.set_xticks(range(4),['Parent','Adam\n3e-4','Adam\n1e-4','Ridge SR'])
            ax.grid(alpha=.15);ax.set_ylabel('Energy / site' if row==0 else 'Var(H) / site')
        print(r['geometry'],arm,{k:a[k] for k in ('energy_per_site','chain_mean_standard_error',
            'variance_per_site','variance_chain_standard_error','split_chain_energy_rhat',
            'second_minus_first_half_per_site','half_drift_chain_standard_error','top_one_percent_variance_share')})
    axes[0,col].set_title(r['geometry'].capitalize()+
        ('\nSR audit not equilibrated' if r['geometry']=='square' else ''),fontsize=11)
    axes[1,col].set_yscale('log')
fig.suptitle('All independently audited states and streams (no selected winner)\nColored: three stream estimates; black: conditional chain-bootstrap 95% interval',fontsize=11)
fig.savefig(out/'fig78_full_vit_optimizer_audits.png',dpi=180)
fig.savefig(out/'fig78_full_vit_optimizer_audits.pdf');plt.show()

fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
for ax,r in zip(axes,policies):
    for arm,t in r['training'].items():
        h=t['history'];x=np.array([s['active_seconds'] for s in h]);y=np.array([s['energy_before_update_per_site'] for s in h])
        blocks=[slice(i,min(i+16,len(h))) for i in range(0,len(h),16)]
        ax.plot([x[b].mean() for b in blocks],[y[b].mean() for b in blocks],c=colors[arm],label=labels[arm])
        a=r['audits'][arm];v=a['energy_per_site'];ci=a['energy_chain_bootstrap_95']
        ax.errorbar(x[-1],v,yerr=[[max(v-ci[0],0)],[max(ci[1]-v,0)]],fmt='o',c=colors[arm],capsize=3)
        print(r['geometry'],arm,'seconds=',t['active_seconds'],'compile=',t['optimizer_compile_seconds'],
            'steps=',t['completed_steps'],'rejected=',sum(not s['accepted'] for s in h))
    ax.set_xlabel('Active CPU training time / s\n(excludes compilation and audit)',fontsize=9)
    ax.set_ylabel('Energy / site');ax.set_title(r['geometry'].capitalize()+
        ('\nSR audit not equilibrated' if r['geometry']=='square' else ''),fontsize=11)
    ax.grid(alpha=.15);ax.legend(fontsize=8)
fig.suptitle('Actual CPU cost; not a GPU-hours or equal-wall-time comparison',fontsize=11)
fig.savefig(out/'fig79_full_vit_optimizer_cpu_cost.png',dpi=180)
fig.savefig(out/'fig79_full_vit_optimizer_cpu_cost.pdf');plt.show()
"""),code(r"""
# Post-hoc diagnosis using every saved chain; no new samples or H calls.
fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for ax,r in zip(axes,policies):
    for arm in ('parent',*labels):
        a=r['audits'][arm]
        values=np.array([np.load(folder/r['geometry']/f"audit_{arm}_{s['seed']}.npz")['local_energies'] for s in a['streams']])
        chain_means=values.transpose(0,2,1).reshape(96,32).real.mean(axis=1)/36
        color='0.5' if arm=='parent' else colors[arm]
        label='Parent' if arm=='parent' else labels[arm]
        ax.plot((np.arange(96)+.5)/96,np.sort(chain_means),'.-',ms=2,lw=1,c=color,
            label=f"{label}, Rhat={a['split_chain_energy_rhat']:.3f}")
        print('Post-hoc',r['geometry'],arm,'chains with positive mean E/N:',int(sum(chain_means>0)))
    ax.set_xlabel('Empirical quantile across all 96 audit chains')
    ax.set_ylabel('Chain-mean local energy / site');ax.grid(alpha=.15)
    ax.set_title(r['geometry'].capitalize());ax.legend(fontsize=7)
fig.suptitle('Independent-chain audit exposes the square SR failure\nPost-hoc diagnosis; these are not thermodynamic phases or equilibrium-energy bounds',fontsize=11)
fig.savefig(out/'fig80_full_vit_optimizer_chain_diagnosis.png',dpi=180)
fig.savefig(out/'fig80_full_vit_optimizer_chain_diagnosis.pdf');plt.show()
"""),markdown(r"""
### Limits and decision

The completed result is asymmetric. **Triangular:** SR passes the conditional
energy-and-variance comparison with Adam, E/N=-0.488492(570) versus
-0.449013(1090), with variance/site 0.038803 versus 0.154476. It is still
far above the cached ED target -0.515564/site. **Square:** SR fails; independent
audit streams give -0.38469, -0.40839 and -0.35265/site, unlike its low training
batches. Pooled split-chain energy Rhat=3.155, and 13/96 chain means are positive.
The nominal pooled -0.38191(1988)/site is **not a trustworthy equilibrated
energy**. No confidence interval here accounts for this strong mixing failure.
The square parent/Adam/low-lr observed variances are also tail-sensitive: the
largest 1% of residuals supplies roughly 88%/80%/77%, respectively. Do not
interpret changing audit variance as a change to a frozen model.

The post-hoc chain plot uses all saved records without generating samples or
changing the predeclared endpoint comparison. It diagnoses disagreement among
chains, not a magnetic phase or a spin-liquid mechanism. Sampled trust bounds
passing on the 64 training configurations did not guarantee a safe global
wavefunction update on the square lattice. Do not silently choose an earlier
SR checkpoint or discard high-energy chains to manufacture a win.

Only one inherited training seed per geometry is tested. Three independent
audit streams do **not** equal three independently trained models. Finite
sampling error, incomplete mixing, unseen rare configurations and parameter
undertraining are distinct limitations. Both primary observables must have
negative upper 98.75% bounds before calling the predeclared conditional joint
test positive. A noisy low training energy, apparent variance reduction, or
sampled trust check alone is insufficient. Keep time-capped arms as failures
of matched-work completion rather than extending them selectively.

The same-parent fork answers whether a bounded change in **optimization
policy** helps this existing manifold. It cannot establish that adding SU(2)
symmetry or parameters improves physics, nor a QSL interpretation. Any
promising endpoint must still be judged against finite-torus reference
accuracy and actual cost; no automatic production budget is released.

Method context: [Chen and Heyl, Nature Physics 20, 1476 (2024)](https://www.nature.com/articles/s41567-024-02566-1)
motivates the sample-space SR identity. This real-parameter ridge/trust policy
does not reproduce their complete MinSR training protocol. Final numerical
interpretation, including negative results and tail warnings, is maintained
in `results/spin_full_vit_optimization_v1/REPORT.md`.
""")]
    for cell in cells:
        cell.metadata['experiment_section']='full_vit_optimization_20260928'
    return cells


def full_vit_continuation_cells():
    """Frozen 1000-step continuation and independent audits; exploration only."""
    cells = [markdown(r"""
## N36 full-ViT: preregistered 1000-step checkpoint continuation (2026-09-28)

Two **4032-parameter** models continue from their own saved step-64 pilot:
triangular $6\times6$, $J_2/J_1=0.125$, and square $6\times6$, $J_2/J_1=0.5$.
All layers, Adam moments, chains and PRNG are restored. Exactly 1000 **additional**
updates: terminal step **1064**, not a best-energy checkpoint. The model is
fixed $S^z=0$ and exact lattice $\Gamma-A_1$, **not exact spin SU(2)**. There
is no carrier, new head, extra capacity, learning-rate search or audit-based
early stopping. This is a stronger baseline-training test, not a symmetry win.

Training uses 32 chains, 36 all-site exchange proposals per update, Adam
$\eta=3\times10^{-4}$. After training, both the frozen parent and terminal
state are remeasured on **distinct** preregistered random streams: three
streams, 32 chains each, 2304 burn-in proposals and 32 snapshots separated by
72 proposals. Each state has 3072 records, but only **96 independent chains**.
The old 256-record pilot audits are not reused in the endpoint comparison.
Bars below resample whole chains; burn-in bias and missing basins remain
possible. Independent endpoint differences are **unpaired**, not paired SE.
"""), code(r"""
from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib.pyplot as plt
from su2_nqs.frontier.artifacts import canonical_sha256
root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs'
out=project/'figures/exploration';out.mkdir(parents=True,exist_ok=True)
folder=project/'results/spin_full_vit_n36_continuation_v1'
references=json.loads((project/'references/finite_torus_6x6_targets.json').read_text())
def sealed(path):
    r=json.loads(path.read_text())
    assert r['result_sha256']==canonical_sha256({k:v for k,v in r.items() if k!='result_sha256'})
    return r
continued=[sealed(folder/g/'results.json') for g in ('triangular','square')]
for r in continued:
    reg=sealed(folder/r['geometry']/'registration.json')
    assert r['registration_sha256']==reg['result_sha256']
    assert r['completed_additional_steps']==1000 and r['total_steps']==1064
    assert hashlib.sha256((folder/r['geometry']/'checkpoint.pkl').read_bytes()).hexdigest()==r['checkpoint_sha256']
    for label,audit in r['audits'].items():
        for record in audit['streams']:
            path=folder/r['geometry']/f"audit_{label}_{record['seed']}.npz"
            assert hashlib.sha256(path.read_bytes()).hexdigest()==record['raw_sha256']

fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for col,r in enumerate(continued):
    h=r['history'];steps=np.array([v['step'] for v in h])
    for row,key,audit_key in [(0,'energy_before_update_per_site','energy_per_site'),
                            (1,'variance_before_update_per_site','variance_per_site')]:
        ax=axes[row,col];y=np.array([v[key] for v in h])
        ax.plot(steps,y,c='#0072B2',lw=.45,alpha=.25,label='Training batch (32 chains)')
        ax.plot(steps.reshape(-1,50).mean(axis=1),y.reshape(-1,50).mean(axis=1),
                c='#0072B2',lw=1.4,label='50-update mean (not uncertainty)')
        for label,x in [('parent',64),('terminal',1064)]:
            a=r['audits'][label];ci=a['energy_chain_bootstrap_95' if row==0 else 'variance_chain_bootstrap_95']
            v=a[audit_key];color='#D55E00' if label=='parent' else '#009E73'
            ax.errorbar(x,v,yerr=[[max(v-ci[0],0)],[max(ci[1]-v,0)]],fmt='o',ms=5,
                        c=color,capsize=4,label=f'Fresh {label} audit, chain bootstrap 95%')
        ax.grid(alpha=.15);ax.set_xlabel('Cumulative update (parent = 64; fixed terminal = 1064)')
    ref=references['triangular_6x6_torus' if r['geometry']=='triangular' else 'square_6x6_pbc']
    axes[0,col].axhline(ref['exact_energy_per_site'],c='0.3',ls='--',lw=1,label='Cached same-torus ED target')
    axes[0,col].set_title(f"{r['geometry'].capitalize()} 6x6, J2={r['j2_over_j1']}")
    axes[0,col].set_ylabel('Energy / site');axes[1,col].set_ylabel('Var(H) / site')
    axes[1,col].set_yscale('log')
    axes[0,col].legend(fontsize=6.8,loc='upper right',framealpha=.95)
    print(r['geometry'],'training seconds',r['training_seconds'],
          'wall seconds',r['wall_seconds_this_invocation'],'full-H calls',r['full_H_configuration_calls'])
    print('Independent terminal-minus-parent:',r['independent_differences'])
fig.suptitle('Full-ViT continuation: fixed budget, all layers trained, no checkpoint selection\nFixed Sz=0 + exact lattice A1; energy accuracy is not an SU(2) or frontier win',fontsize=11)
fig.savefig(out/'fig75_full_vit_n36_continuation.png',dpi=180)
fig.savefig(out/'fig75_full_vit_n36_continuation.pdf');plt.show()
"""), code(r"""
fig,axes=plt.subplots(2,2,figsize=(10,6.5),constrained_layout=True)
for col,r in enumerate(continued):
    for row,quantity in enumerate(('energy','variance')):
        ax=axes[row,col]
        for idx,label in enumerate(('parent','terminal')):
            a=r['audits'][label]
            samples=a['stream_energies_per_site' if quantity=='energy' else 'stream_variances_per_site']
            x=idx+np.array([-.065,0,.065])
            ax.scatter(x,samples,c=['#0072B2','#D55E00','#CC79A7'],s=38,zorder=4)
            value=a[f'{quantity}_per_site'];ci=a[f'{quantity}_chain_bootstrap_95']
            ax.errorbar(idx+.14,value,yerr=[[max(value-ci[0],0)],[max(ci[1]-value,0)]],
                        fmt='D',ms=4,c='black',capsize=4)
        ax.set_xticks([0,1],['Step 64\nfresh audit','Step 1064\nfresh audit'])
        ax.set_ylabel('Energy / site' if quantity=='energy' else 'Var(H) / site');ax.grid(alpha=.15)
        if row==0:ax.set_title(r['geometry'].capitalize())
    for label,a in r['audits'].items():
        print(r['geometry'],label,{k:a[k] for k in ['energy_per_site','chain_mean_standard_error',
            'variance_per_site','variance_chain_standard_error','split_chain_energy_rhat',
            'second_minus_first_half_per_site','half_drift_chain_standard_error',
            'energy_imaginary_per_site','top_one_percent_variance_share']})
        print('Stream nontrivial acceptance:',[s['mean_nontrivial_acceptance'] for s in a['streams']])
fig.suptitle('Three preregistered independent streams per state\nColored dots: stream estimates. Black: pooled estimate with whole-chain bootstrap 95%.',fontsize=11)
fig.savefig(out/'fig76_full_vit_independent_audit.png',dpi=180)
fig.savefig(out/'fig76_full_vit_independent_audit.pdf');plt.show()
"""), markdown(r"""
### Interpretation boundary

Both frozen terminal states have lower audited energy and observed variance
than their own parents, but their same-torus energy gaps remain approximately
0.07477/site (triangular) and 0.04792/site (square). This is an undertraining
result, not a competitive ansatz advantage. Square's largest 1% of observed
residuals accounts for 61.28% of its terminal variance; its parent audit also
has a roughly 2.8-chain-SE half-run drift. Retain these warnings.

An individual square **training batch** reaches -0.54304/site, below the ED
target. A finite local-energy sample mean is not itself a variational upper
bound. This noisy batch is neither a claimed win nor a selected checkpoint.

Compare energy **and** variance to the low-energy step-64 parent, not to the
near-maximal-spin random initialization. A decrease tests undertraining; it
does not establish asymptotic convergence, a scalable exact-SU(2) advantage,
a quantum spin liquid, or competitive accuracy. Colored stream agreement,
split-chain energy Rhat, half-run drift, imaginary energy and tail contributions
are diagnostics; none certifies that all basins have been sampled. Training
has only one inherited seed per lattice. No extra audit, hyperparameter scan,
longer trajectory or GPU campaign is automatically released by this section.

Same-torus target numbers and provenance are cached in
`references/finite_torus_6x6_targets.json`: triangular $J_2=0.125$,
$E_0/N=-0.515564$ (Iqbal et al., PRB 93, 144411, 2016); square $J_2=0.5$,
$E_0/N=-0.503810$ (Wang et al., PRX 14, 031006, 2024). These are accuracy
targets, not same-work baselines; verify wrapping conventions before a close
triangular comparison. No new diagonalization is performed. The fixed final
results and next bounded decision are recorded in the accompanying REPORT.md.
""")]
    for cell in cells:
        cell.metadata['experiment_section']='full_vit_continuation_20260928'
    return cells


def strong_innovation_cells():
    """Carrier reoptimization and independent usefulness, not another sample audit."""
    cells=[markdown(r"""
## Strong-carrier innovation after ordinary H updates (2026-09-27)

**New question:** the old strong-space test kept $OF\psi_{K2}$ fixed. Here
the same physical $P_{A_1}OF P_{A_1}$ acts on the **reoptimized current
carrier** in K2, K4, or K2 + all three complete-exchange images. No new
operator, neural parameter, epsilon scan, sampling or full-sector ED is used.
The cached triangular 4x4 singlet/A1 block has dimension 14. All 18 cells
(three couplings, two placements, three carriers) are retained.

For an anchor B, find its Ritz state $\psi_B$, then
$B_H=\mathrm{span}(B,H\psi_B)$ and
$r_\perp=(1-P_{B_H})OF\psi_B$. The direct test compares B+OF with B+H at
rank r+1. The innovation test compares $B_H+r_\perp$ against a **second H
step** at rank r+2, also checking global rank-matched Krylov. Rank never
exceeds 7. Five nonlocal scrambles fix every vector of $B_H$; they are not
lattice-local controls. Rank matching does not establish GPU-work matching.

The first energy derivative along $r_\perp$ at $\psi_B$ is zero **by
construction**, not evidence that this vector is physically useless. Its
variational usefulness is evaluated after the H step. Exact references
are cached states, used for diagnostics only, not to select directions.

**Outcome:** all 18 directions retain a nonzero independent component, but
0/18 beat same-rank Krylov in energy/variance. At the primary K4, J2=0.125
point, 17.97% of the OF-image norm squared survives removal of B and H;
it captures only 0.021% of the original residual norm squared. After H,
its capture rises to 28.56%, and adding it reduces variance and two-cell
RDM error, but another ordinary H step does substantially better.
Winning 5/5 scrambles here is not a competitive variational win.

These are finite-cluster diagnostics, not a phase diagram or evidence
against all SU(2)/ViT architectures. This fixed OF family is not promoted
to 6x6 production. All plots remain in the exploration notebook.
"""),code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from su2_nqs.frontier.artifacts import canonical_sha256
root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs'
innovation=json.loads((project/'results/spin_strong_carrier_innovation_v1/results.json').read_text())
assert innovation['result_sha256']==canonical_sha256({k:v for k,v in innovation.items() if k!='result_sha256'})
out=project/'figures/exploration';out.mkdir(parents=True,exist_ok=True)
couplings=innovation['config']['couplings']
anchors=innovation['config']['anchors']
fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
for col,anchor in enumerate(anchors):
    for partition,linestyle in (('original','-'),('spectator_row_shift','--')):
        rows=[innovation['cases'][f'{partition}/j{j}/{anchor}'] for j in couplings]
        for arm,control,color,label in (
            ('direct','rank_matched_direct_krylov','#D55E00','B+OF (rank r+1)'),
            ('conditional','rank_matched_conditional_krylov','#0072B2','After H + innovation (rank r+2)'),
        ):
            delta=[r['states'][arm]['energy_per_site']-r['states'][control]['energy_per_site'] for r in rows]
            ratio=[r['states'][arm]['variance_total']/r['states'][control]['variance_total'] for r in rows]
            axes[0,col].plot(couplings,delta,'o',ls=linestyle,c=color,label=label+(' / shifted' if partition!='original' else ''))
            axes[1,col].plot(couplings,ratio,'o',ls=linestyle,c=color)
    axes[0,col].axhline(0,color='black',lw=.8)
    axes[0,col].set(title=anchor,yscale='symlog');axes[0,col].set_yscale('symlog',linthresh=1e-8)
    axes[1,col].axhline(1,color='black',lw=.8)
    axes[1,col].set(yscale='log',xlabel=r'$J_2/J_1$')
    for ax in axes[:,col]:ax.set_xticks(couplings);ax.grid(alpha=.15)
axes[0,0].set_ylabel('Energy/site minus same-rank Krylov')
axes[1,0].set_ylabel('Variance / same-rank Krylov variance')
axes[0,2].legend(frameon=False,fontsize=6.5,loc='best')
fig.suptitle('Independent directions do not beat the matched variational controls')
fig.savefig(out/'fig70_strong_carrier_innovation_rank_controls.png',dpi=180)
fig.savefig(out/'fig70_strong_carrier_innovation_rank_controls.pdf');plt.show()
print('Summary:',innovation['summary'],'result seal:',innovation['result_sha256'])
"""),code(r"""
rows=[innovation['cases'][f'original/j{j}/K4'] for j in couplings]
fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
for key,label,color in (
    ('outside_anchor_and_h_fraction','Independent norm fraction','#777777'),
    ('original_gradient_capture_fraction','Original H residual captured','#D55E00'),
    ('next_gradient_capture_fraction','After-H residual captured','#0072B2'),
):
    axes[0].plot(couplings,[r['diagnostics'][key] for r in rows],'o-',label=label,c=color)
axes[0].set(yscale='log',ylabel='Squared-norm fraction',title='K4: geometry versus useful overlap')
axes[0].legend(frameon=False,fontsize=7)
for arm,label,color in (
    ('carrier','K4','#777777'),('h_once','K5','#D55E00'),
    ('conditional','K5 + independent OF','#0072B2'),('h_twice','K6','#009E73'),
):
    axes[1].plot(couplings,[r['states'][arm]['two_cell_rdm_trace_distance'] for r in rows],'o-',label=label,c=color)
    axes[2].plot(couplings,[abs(r['states'][arm]['connected_OF']-r['exact_connected_OF']) for r in rows],'o-',label=label,c=color)
axes[1].set(yscale='log',ylabel='Two-cell RDM trace-distance error',title='Local state accuracy')
axes[2].set(yscale='log',ylabel='Absolute connected OF error',title='Not every physical observable improves')
axes[1].legend(frameon=False,fontsize=7,loc='center',bbox_to_anchor=(.47,.63))
for ax in axes:ax.set(xlabel=r'$J_2/J_1$',xticks=couplings);ax.grid(alpha=.15)
fig.suptitle('Original placement: finite 4x4 diagnostics, not a QSL phase claim')
fig.savefig(out/'fig71_strong_carrier_innovation_physics.png',dpi=180)
fig.savefig(out/'fig71_strong_carrier_innovation_physics.pdf');plt.show()
primary=innovation['cases'][innovation['primary_case']]
for arm in ('carrier','direct','h_once','conditional','h_twice'):
    s=primary['states'][arm]
    print(arm,'rank',s['rank'],'E/site',s['energy_per_site'],'variance',s['variance_total'],
          'RDM error',s['two_cell_rdm_trace_distance'],'connected OF error',abs(s['connected_OF']-primary['exact_connected_OF']))
print('Scramble wins:',primary['scramble_joint_wins'],'/5; same-rank gates:',primary['gates'])
print('No Monte Carlo samples, remote job, new NN, or full-sector Hamiltonian ED in this follow-up.')
""")]
    for cell in cells:cell.metadata['experiment_section']='strong_carrier_innovation_20260927'
    return cells


def stratified_audit_cells():
    """Frozen-state two-phase audit: all raw data retained, weights explicit."""
    cells=[markdown(r"""
## Frozen-state stratified tail audit (2026-09-27)

The triangular $6\times6$, $J_2/J_1=0.125$ state is unchanged:
$\phi=\psi_{\pi/0}+0.5\bar A\psi_{\pi/0}$, exactly $S=0,\Gamma-A_1$.
Eight new independent pools contain 64 chains and 4096 records each.
Old residual thresholds and the proposal $q=|\psi|^2+\kappa$ are frozen.
There is no fitting, new operator, or epsilon selection.

All 32768 carrier H-images are evaluated. Within each pool, expensive
dressed H-images are selected by simple random sampling without replacement
in four residual strata: 32,24,32 records, then **every observed record in
the extreme stratum**. For stratum size $M_h$ and selected count $n_h$,
the correction raw moment is estimated as
$\sum_h(M_h/M)\overline{\delta X}_h$, equivalent to inverse-inclusion weights
$M_h/n_h$. Carrier raw moments use the complete pool. Only after combining
the eight pools do we form normalization, energy and variance ratios.
Treating these deliberately oversampled tails as equally weighted samples
would be wrong. No old data enter the new physical estimate.

Reported paired SEs come from between-pool raw-moment covariance. The new
95% intervals use Student-t with 7 degrees of freedom and a delta method;
these are approximate intervals, not equilibration or tail-coverage proofs.
Conditional phase-two design SE is shown separately and is **not added
again** to the between-pool SE. Census of the observed extreme stratum does
not cover extreme states absent from the finite MCMC pool. Final ratios can
still have finite-sample bias. This remains an exploratory test against a
zero-Lanczos carrier, not a matched Krylov/ViT comparison or a PRL result.

**Completed outcome:** all eight pool energy changes are negative, with
combined $\Delta E/N=-3.20\times10^{-5}$. The preregistered between-pool SE
is $0.61\times10^{-5}$, but the conditional selection SE is larger,
$1.45\times10^{-5}$. The eight realized pool means are unusually consistent
relative to this estimated within-pool noise; do not advertise a robust
five-sigma result. The variance change is $-4.90\times10^{-5}$ with a
between-pool SE $2.39\times10^{-5}$, and its t(7) interval still includes
zero. This strengthens the small lowering-direction lead, not a joint
variational win. All 18 observed extreme records are retained; their net
variance contribution is negative, not the anticipated automatic penalty.
"""),code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from su2_nqs.frontier.artifacts import canonical_sha256
root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs'
directory=project/'results/spin_stratified_environment_audit_v1'
strat=json.loads((directory/'results.json').read_text())
assert canonical_sha256({k:v for k,v in strat.items() if k!='result_sha256'})==strat['result_sha256']
ordinary=json.loads((root/strat['config']['source_directory']/'results.json').read_text())
out=project/'figures/exploration';out.mkdir(parents=True,exist_ok=True)
agg=strat['aggregate']
fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for ax,key,label in zip(axes,('energy_change_total','variance_change_total'),(r'$\Delta E/N$',r'$\Delta\mathrm{Var}(H)/N$')):
    vals=np.array([ordinary['aggregate'][key],agg[key]])/36
    errors=np.array([ordinary['aggregate']['paired_chain_bootstrap_se'][key],agg['independent_pool_se'][key]])/36
    ax.errorbar((0,1),vals,yerr=errors,fmt='o',capsize=5,color='#0072B2',label='One paired SE')
    ax.errorbar(1.10,vals[1],yerr=agg['conditional_design_se'][key]/36,fmt='o',mfc='white',color='#D55E00',capsize=4,label='New: conditional design SE only')
    ci=np.array(agg['student_t_95'][key])/36
    ax.vlines(1,*ci,color='#0072B2',alpha=.4,lw=2,label='New: nominal 95% t(7) interval')
    ax.axhline(0,c='black',lw=.8);ax.set(xticks=(0,1),xticklabels=('Prior ordinary\n1024 dressed records',f"New stratified\n{strat['sample_count']} dressed records"),ylabel=label)
    ax.set_xlim(-.5,1.5);ax.grid(axis='y',alpha=.15)
axes[0].legend(frameon=False,fontsize=8)
fig.suptitle('Same physical state and epsilon=0.5; independent datasets, not pooled')
fig.savefig(out/'fig67_finite_A1_stratified_comparison.png',dpi=180)
fig.savefig(out/'fig67_finite_A1_stratified_comparison.pdf');plt.show()
print('Frozen source:',strat['config']['source_result_sha256'])
print('New result seal:',strat['result_sha256'])
print('Paired changes:',{k:(agg[k]/36,agg['independent_pool_se'][k]/36,np.array(agg['student_t_95'][k])/36) for k in ('energy_change_total','variance_change_total')})
print('Absolute energy:',agg['energy_total']/36,'+/-',agg['independent_pool_se']['energy_total']/36)
"""),code(r"""
pop=np.array(strat['stratum_population_counts']);chosen=np.array(strat['stratum_sample_counts'])
labels=['Below old p90','p90--p99','p99--p99.9','Above old p99.9']
fig,grid=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
axes=grid.ravel()
axes[0].plot(range(4),pop/pop.sum(),'o-',label='Fraction of new carrier pool',color='#0072B2')
axes[0].plot(range(4),chosen/chosen.sum(),'s--',label='Fraction of dressed evaluations',color='#D55E00')
axes[0].set(yscale='log',ylabel='Fraction (not a physical sector weight)',title='Tails intentionally oversampled')
axes[0].legend(frameon=False,fontsize=8)
for row in strat['replicates']:
    p=np.array(row['population_counts']);n=np.array(row['selected_counts'])
    prob=np.divide(n,p,out=np.full(4,np.nan),where=p>0)
    axes[1].plot(range(4),prob,'o-',alpha=.45,lw=.8,color='#009E73')
axes[1].set(yscale='log',ylabel='Inclusion probability n_h / M_h',title='Known weights; extreme stratum is a census')
axes[1].axhline(1,color='gray',lw=.7,ls=':')
raw_strata=np.mean([[s['weighted_moment_contribution'] for s in r['summary']['strata']] for r in strat['replicates']],axis=0)
z,b,c,dz,db,dc=agg['mean_moments'];z1=z+dz;e0=b/z;e1=(b+db)/z1;v0=c/z-e0**2
# Exact additive decomposition, including the normalization change; no tail deletion.
contributions={
    'energy_change_total':raw_strata@np.array([-e0,1.,0.])/z1,
    'variance_change_total':raw_strata@np.array([e0*e1-v0,-(e0+e1),1.])/z1,
}
for ax,key,label in zip(axes[2:],contributions,(r'Contribution to $\Delta E/N$',r'Contribution to $\Delta\mathrm{Var}(H)/N$')):
    np.testing.assert_allclose(contributions[key].sum(),agg[key],atol=2e-13)
    ax.plot(range(4),contributions[key]/36,'o-',color='#CC79A7')
    ax.axhline(0,color='black',lw=.8);ax.set(ylabel=label,title='Weighted contributions, including normalization')
for ax in axes:
    ax.set(xticks=range(4),xticklabels=labels);ax.tick_params(axis='x',labelrotation=20);ax.grid(alpha=.15)
fig.suptitle(f"{strat['pool_sample_count']} carrier records; {strat['sample_count']} dressed records; {pop[3]} extreme records included")
fig.savefig(out/'fig68_finite_A1_stratified_coverage.png',dpi=180)
fig.savefig(out/'fig68_finite_A1_stratified_coverage.pdf');plt.show()
print('Population counts:',pop,'selected counts:',chosen)
print('Additive energy/variance contributions per site:',{k:v/36 for k,v in contributions.items()})
print('These historical residual bins are sampling strata, not spin or lattice symmetry sectors.')
print('Stratum contributions are an exact estimator decomposition, not local energy densities; all strata are retained.')
"""),code(r"""
fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for ax,key,label in zip(axes,('energy_change_total','variance_change_total'),(r'$\Delta E/N$',r'$\Delta\mathrm{Var}(H)/N$')):
    vals=np.array([r['summary']['physical'][key] for r in strat['replicates']])/36
    design=np.array([r['summary']['conditional_design_se'][key] for r in strat['replicates']])/36
    ax.errorbar(range(1,9),vals,yerr=design,fmt='o',capsize=3,color='#0072B2',label='Pool estimate; conditional design SE only')
    lo,hi=np.array(agg['student_t_95'][key])/36
    ax.axhspan(lo,hi,color='#009E73',alpha=.12,label='Combined: approximate 95% t(7) interval')
    ax.axhline(agg[key]/36,color='#009E73',lw=1.2)
    ax.axhline(0,c='black',lw=.8);ax.set(xlabel='Independent pool',ylabel=label,xticks=range(1,9));ax.grid(alpha=.15)
axes[0].legend(frameon=False,fontsize=7,loc='best')
fig.suptitle('No seed selection: all eight preregistered pools')
fig.savefig(out/'fig69_finite_A1_stratified_pools.png',dpi=180)
fig.savefig(out/'fig69_finite_A1_stratified_pools.pdf');plt.show()
for key in ('energy_change_total','variance_change_total'):
    print(key,'between-pool SE/site',agg['independent_pool_se'][key]/36,
          'conditional design SE/site',agg['conditional_design_se'][key]/36,
          'delete-one-pool range/site',np.array(agg['leave_one_pool_out'][key])/36)
print('Conditional design bars omit phase-one MCMC uncertainty; do not interpret them as total pool errors.')
print('See the new REPORT.md for the decision. No remote budget is released by this diagnostic.')
""")]
    for cell in cells:cell.metadata['experiment_section']='stratified_environment_audit_20260927'
    return cells


def finite_audit_cells():
    """Larger untouched fixed-step comparison with explicit residual tails."""
    cells=[markdown(r"""
## Enlarged frozen-step audit: independent chains and rare configurations

This confirms or challenges the previous 128-sample pilot **without changing
the carrier, scalar operator, exact A1 projection, or primary epsilon=0.5**.
No fitting or step selection is performed. Four fresh streams contain 64
independent chains each, with 8192 burn-in exchange proposals. Each stream
saves a cheap 4096-configuration carrier pool, then evaluates the expensive
full-H dressed action on four blocks selected in advance (0,16,32,48).
Consequently, the finite-state comparison has 1024 configurations from 256
chains, while the carrier-tail diagnostic has 16384 configurations from those
same chains. The pools are **not** extra dressed-state samples.

All operator terms and all Hamiltonian connections remain deterministic.
The new scanned evaluator only changes batching of the old determinant
formula, with sealed-amplitude/H replay before every stream. Tail thresholds
are the 90%,99%,99.9% quantiles of the old baseline residual contribution;
new energies never enter the selection rule. Every sample is retained in the
physical estimate. Deleting a tail or a chain is a sensitivity check only.
"""),code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from su2_nqs.frontier.artifacts import canonical_sha256
from su2_nqs.benchmarks.finite_audit_statistics import FIELDS, paired_chain_summary
root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs';directory=project/'results/spin_finite_environment_audit_v1'
audit=json.loads((directory/'results.json').read_text())
assert canonical_sha256({k:v for k,v in audit.items() if k!='result_sha256'})==audit['result_sha256']
pilot=json.loads((root/audit['config']['source_directory']/'results.json').read_text())
old=json.loads((root/audit['config']['tail_reference_directory']/'results.json').read_text())
out=project/'figures/exploration';out.mkdir(parents=True,exist_ok=True)
streams=[]
for i in range(4):
    with np.load(directory/f'replicate_{i}.npz') as b:streams.append({k:b[k] for k in FIELDS})
data={k:np.concatenate([s[k] for s in streams]) for k in FIELDS}
chains=np.concatenate([np.tile(np.arange(64)+i*64,4) for i in range(4)])
prefix=[]
for blocks in range(1,5):
    ix=np.concatenate([np.arange(i*256,i*256+blocks*64) for i in range(4)])
    prefix.append(paired_chain_summary({k:v[ix] for k,v in data.items()},chains[ix],.5,replicates=400,seed=2026092799+20+blocks))
prefix[-1]=audit['aggregate'] # final error bars use the sealed 1000-resample estimate
pilot_primary=next(r for r in pilot['aggregate'] if r['epsilon']==.5)
fig,axes=plt.subplots(1,2,figsize=(10,4.2),constrained_layout=True)
for ax,key,label in zip(axes,('energy_change_total','variance_change_total'),(r'$\Delta E/N$',r'$\Delta\mathrm{Var}(H)/N$')):
    ax.errorbar([r['sample_count'] for r in prefix],[r[key]/36 for r in prefix],yerr=[r['paired_chain_bootstrap_se'][key]/36 for r in prefix],fmt='o-',color='#0072B2',capsize=3,label='New fixed-step audit')
    ax.errorbar(128,pilot_primary[key]/36,yerr=pilot_primary['paired_chain_bootstrap_se'][key]/36,fmt='s',color='gray',capsize=3,label='Old pilot (not pooled)')
    ax.axhline(0,c='black',lw=.8);ax.set(xlabel='Finite-state configurations',ylabel=label,xticks=(128,256,512,768,1024));ax.grid(alpha=.15)
axes[0].legend(frameon=False,fontsize=8)
fig.suptitle('Frozen strict-SU(2)/A1 state, epsilon=0.5: cumulative paired changes')
fig.savefig(out/'fig64_finite_A1_enlarged_convergence.png',dpi=180);fig.savefig(out/'fig64_finite_A1_enlarged_convergence.pdf');plt.show()
print('Primary full audit:',audit['aggregate'])
print('1024 configurations are 256 persistent chains, not 1024 independent observations.')
print('Result seal:',audit['result_sha256'])
"""),code(r"""
fig,axes=plt.subplots(1,3,figsize=(13,4.3),constrained_layout=True)
subset_summary=paired_chain_summary(data,chains,0.,replicates=400,seed=2026092799+30)
from su2_nqs.benchmarks.finite_scalar_dressing import finite_step_moments
pool_blocks=[]
for i in range(4):
    with np.load(directory/f'pool_{i}.npz') as b:pool_blocks.append({k:b[k] for k in ('psi','hpsi','proposal_squared')})
pool={k:np.concatenate([s[k] for s in pool_blocks]) for k in pool_blocks[0]}
pool['direction']=np.zeros_like(pool['psi']);pool['hdirection']=np.zeros_like(pool['psi'])
subset_chain_indices=np.arange(1024).reshape(4,4,64).transpose(0,2,1).reshape(256,4)
pool_chain_indices=np.arange(16384).reshape(4,64,64).transpose(0,2,1).reshape(256,64)
rng=np.random.default_rng(2026092799+31);coverage_boot=[]
for _ in range(1000):
    picked=rng.integers(0,256,256)
    si=subset_chain_indices[picked].ravel();pi=pool_chain_indices[picked].ravel()
    sm=finite_step_moments(*(data[k][si] for k in FIELDS),0.)
    pm=finite_step_moments(*(pool[k][pi] for k in FIELDS),0.)
    coverage_boot.append(sm['variance_total']-pm['variance_total'])
print('Carrier variance subset-minus-pool (shared-chain paired diagnostic):',
      subset_summary['variance_total']-audit['baseline_pool']['variance_total'],
      'SE',np.std(coverage_boot,ddof=1),'bootstrap percentile interval',np.quantile(coverage_boot,[.025,.975]))
values=[old['aggregate']['variance_total']/36,subset_summary['variance_total']/36,audit['baseline_pool']['variance_total']/36]
errors=[old['aggregate']['chain_bootstrap_se']['variance_total']/36,subset_summary['paired_chain_bootstrap_se']['variance_total']/36,audit['baseline_pool']['paired_chain_bootstrap_se']['variance_total']/36]
axes[0].errorbar(range(3),values,yerr=errors,fmt='o',capsize=4,color='#0072B2')
axes[0].set(xticks=range(3),xticklabels=['Old\n16384','New subset\n1024','New pool\n16384'],ylabel=r'Carrier $\mathrm{Var}(H)/N$',title='Baseline tail coverage')
labels=['below old p90','p90--p99','p99--p99.9','above p99.9']
for ax,key,label in zip(axes[1:],('energy_change_contribution','variance_change_contribution'),(r'Contribution to $\Delta E/N$',r'Contribution to $\Delta\mathrm{Var}(H)/N$')):
    vals=[s[key]/36 if s['count'] else np.nan for s in audit['tail']['strata']]
    ax.plot(range(4),vals,'o-',color='#009E73');ax.axhline(0,c='black',lw=.8)
    ax.set(xticks=range(4),xticklabels=labels,ylabel=label,title='All strata retained')
    ax.tick_params(axis='x',labelrotation=30)
    ax.margins(y=.2);ax.set_xlim(-.25,3.35)
    for i,s in enumerate(audit['tail']['strata']):
        if s['count']:ax.annotate(f"n={s['count']}",(i,vals[i]),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
        else:ax.text(i+.15,.55,'Not sampled\n(n=0)',transform=ax.get_xaxis_transform(),ha='right',fontsize=8,color='gray')
fig.suptitle('Residual-tail diagnostics: historical thresholds, no energy-based selection')
fig.savefig(out/'fig65_finite_A1_enlarged_tails.png',dpi=180);fig.savefig(out/'fig65_finite_A1_enlarged_tails.pdf');plt.show()
print('Finite subset tail decomposition:',audit['tail'])
print('Full baseline pool tail decomposition:',audit['baseline_pool_tail'])
print('Tail deletion is NOT the physical estimate; all samples remain in reported energies.')

fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for ax,key,label in zip(axes,('energy_change_total','variance_change_total'),(r'$\Delta E/N$',r'$\Delta\mathrm{Var}(H)/N$')):
    ax.errorbar(range(1,5),[r[key]/36 for r in audit['per_stream']],yerr=[r['paired_chain_bootstrap_se'][key]/36 for r in audit['per_stream']],fmt='o',capsize=4,color='#0072B2')
    mean=audit['aggregate'][key]/36;se=audit['aggregate']['paired_chain_bootstrap_se'][key]/36
    ax.axhline(0,c='black',lw=.8);ax.axhline(mean,c='#009E73');ax.axhspan(mean-se,mean+se,color='#009E73',alpha=.12)
    ax.set(xlabel='Independent audit stream',xticks=range(1,5),ylabel=label)
fig.suptitle('Four new independent streams; whole-chain paired uncertainty')
fig.savefig(out/'fig66_finite_A1_enlarged_streams.png',dpi=180);fig.savefig(out/'fig66_finite_A1_enlarged_streams.pdf');plt.show()
print('Leave-one-chain-out ranges:',audit['aggregate']['leave_one_chain_out'])
print('Stream replay errors:',[r['replay_errors'] for r in audit['replicates']])
print('Total worker seconds:',audit['total_worker_seconds'])
"""),markdown(r"""
### Interpretation boundary

The complete state and Hamiltonian are unchanged; any change relative to
the pilot reflects sampling rather than an expanded variational manifold.
Uncertainty is paired and clustered by persistent chain. Tail frequencies,
the cheaper larger-pool baseline, and delete-one-chain sensitivity supplement
but cannot prove equilibration. The cumulative points are correlated and
do not constitute repeated independent confirmations.
Tail-bin contributions are estimator accounting, including normalization
effects, not local energy densities or evidence for a physical order parameter.

The reference carrier remains zero-Lanczos pi/0 flux. Even a reliable gain
over this carrier is not a win over optimized two-Lanczos, a strong three-mode
space, or ViT, and does not identify a QSL. See the audit REPORT.md for the
actual outcome and next decision. No remote production is released.

**Outcome:** the full new audit gives $\Delta E/N=-2.03(1.16)\times10^{-5}$
and $\Delta\mathrm{Var}(H)/N=-3.00(11.04)\times10^{-5}$. Both percentile
intervals include zero. Energy decreases in four streams, variance in three;
the latter changes sign under deletion of one chain. The new large carrier
pool reproduces the old variance, but the expensive subset contains no
configurations above the old p99.9 cutoff (the large pool contains nine).
Thus the pilot's simultaneous-improvement suggestion is **not confirmed**.
There remains a small unresolved energy lead, not a proved absence of gain.
No additional tail points were appended post hoc to the registered result.
""")]
    for cell in cells:cell.metadata['experiment_section']='finite_environment_audit_20260927'
    return cells


def symmetry_transfer_cells():
    """Saved-only crossfold, physical Schur readout, and bounded transfer."""
    cells = [markdown(r"""
## Symmetry, optimization transfer, and a physical local readout

This section separates three questions instead of interpreting every energy
change as an architectural improvement. **No cell here trains a model,
queries neural amplitudes or Hamiltonian connectivity, samples, or repeats
ED.** A labelled post-hoc projection contracts saved small matrices and states.

1. On the already viewed $N=36$ records, does a direction fitted to 12 chains
   also lower energy on four omitted chains? Four fixed folds and three fixed
   warmstarts are shown individually. They are correlated diagnostics, not
   independent confirmation. Fit-set Fubini–Study distance is matched, while
   the held-set distance is measured rather than assumed equal.
2. For an actual four-site region, what distinguishes globally singlet
   states? We read $\rho_A=\bigoplus_q[I_{q+1}/(q+1)]\otimes R_q$,
   $p_q=\mathrm{tr}R_q$, and $\sigma_q=R_q/p_q$ directly from saved physical
   amplitudes. The entropy identity is
   $S_A=H(p_q)+\sum_qp_q\log(q+1)+\sum_qp_qS(\sigma_q)$.
   The input density matrix is **not twirled** to make this identity true.
3. A separately bounded $N=16$ coupling-transfer calculation compares
   shared and pointwise optimization, with and without exact singlet VAP.
   Its figure is produced only if a sealed complete result exists.

Readout cut: sites $(0,1,4,5)$, row-major $4\times4$ indexing: a square
plaquette or triangular rhombus. Triangular $4\times4$ misses the K point;
neither smooth spectra nor these coupling labels establish a QSL or phase
boundary. Square $J_2=0.5$ states are two saved variational VAP endpoints,
not exact eigenvectors; square $J_2=0,0.8$ reference vectors were not found.
"""), code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from su2_nqs.frontier.artifacts import canonical_sha256
from su2_nqs.frontier.provenance import file_sha256

root = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p/'su2_nqs').is_dir())
project = root/'projects/clebsch_tree_nqs'
out = project/'figures/exploration'; out.mkdir(parents=True, exist_ok=True)

def st_load(folder):
    path = project/'results'/folder
    payload = json.loads((path/'results.json').read_text())
    unsigned = dict(payload); sha = unsigned.pop('result_sha256')
    assert canonical_sha256(unsigned) == sha and payload['status'] == 'complete'
    if 'registration_sha256' in payload:
        assert file_sha256(path/'registration.json') == payload['registration_sha256']
    for name, digest in payload.get('artifacts', {}).items():
        assert file_sha256(path/name) == digest, name
    return path, payload

st_cross_path, st_cross = st_load('spin_resonance_crossfold_v1')
st_read_path, st_read = st_load('spin_symmetry_readout_v1')
assert file_sha256(st_read_path/'readout.npz') == st_read['arrays_sha256']
assert st_cross['new_H_queries'] == st_cross['new_samples'] == st_cross['new_ED_calls'] == 0
assert st_read['new_ED'] == st_read['new_H_actions'] == st_read['new_samples'] == 0
assert not st_cross['independent_evidence']
print('Crossfold result:', st_cross['result_sha256'])
print('Readout result:', st_read['result_sha256'])
"""), code(r"""
# Fig122: all 24 frozen-parent one-step trials; no crossfold SEM bars.
fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), constrained_layout=True)
st_colors = {'matched_free': '#0072B2', 'attention': '#D55E00'}
st_markers = ['o', 's', '^']
st_radius = st_cross['config']['target_fs']
for family in ('matched_free', 'attention'):
    for start in range(3):
        records = [st_cross['records'][f'{family}{start}__fold{fold}'] for fold in range(4)]
        xs = np.arange(1, 5) + (start-1)*.08 + (-.02 if family == 'matched_free' else .02)
        fit_errors = [abs(r['fit']['distance']['fs']/st_radius-1) for r in records]
        measurements = (np.maximum(fit_errors, 1e-10),
                        [1e6*r['held_out']['delta_energy_per_site'] for r in records],
                        [1e6*r['held_out']['delta_variance_per_site'] for r in records],
                        [r['held_out']['distance']['fs']/st_radius for r in records])
        label = ('Free360' if family == 'matched_free' else 'Attention730') + f'; start {start+1}'
        for ax, ys in zip(axes.flat, measurements):
            ax.plot(xs, ys, marker=st_markers[start], color=st_colors[family],
                    ls='--' if family == 'matched_free' else '-', lw=.8, ms=5, alpha=.85, label=label)
axes[0,0].set(yscale='log', ylabel='|FS(fit) / target - 1|', title='Fit distance: fixed-grid problem repaired')
axes[0,0].axhline(st_cross['config']['relative_fs_tolerance'], color='0.5', ls=':', label='Matching tolerance')
axes[0,0].legend(fontsize=6.8, frameon=False, ncol=2)
axes[0,1].set(ylabel=r'Held $\Delta E/N\;[10^{-6}]$', title='Energy transfer; negative is downhill')
axes[1,0].set(ylabel=r'Held $\Delta\mathrm{Var}(H)/N\;[10^{-6}]$', title='Variance is not guaranteed to decrease')
axes[1,1].set(ylabel='FS(held) / target', title='Held physical distance is not matched')
for ax in (axes[0,1], axes[1,0]): ax.axhline(0, color='0.5', ls=':', lw=1)
axes[1,1].axhline(1, color='0.5', ls=':', lw=1)
for ax in axes.flat:
    ax.set(xticks=range(1,5), xlabel='Omitted chain fold'); ax.grid(alpha=.15)
fig.suptitle('N36 cached directional diagnostic — all data/parents previously viewed', fontsize=12)
fig.savefig(out/'fig122_cached_crossfold_direction_transfer.png', dpi=180)
fig.savefig(out/'fig122_cached_crossfold_direction_transfer.pdf')
plt.show()
for family, row in st_cross['summaries'].items():
    print(family, 'fit matched:', row['all_fit_FS_matched'],
          'registered downhill mean folds:', row['downhill_held_folds'], '/ 4')
display(Markdown('The 3/4-fold criterion is a prespecified diagnostic, **not a significance test**. '
    'Do not infer an attention win by comparing small fold-averaged changes or treating the three starts as independent samples.'))
"""), code(r"""
# Fig123: measured physical region, not an attention heat map.
st_states = st_read['states']
st_tri = sorted([(a['provenance']['coupling'], name, a) for name,a in st_states.items()
                 if name.startswith('triangular_')])
st_j = np.array([j for j,_,_ in st_tri])
st_qcolors = {0:'#0072B2', 2:'#D55E00', 4:'#009E73'}
fig, axes = plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
for q in (0,2,4):
    axes[0,0].plot(st_j, [max(0.,a['sectors'][str(q)]['p']) for _,_,a in st_tri],
        'o-', color=st_qcolors[q], label=f'q={q}')
for index in range(2):
    axes[0,1].plot(st_j, [a['sectors']['0']['conditional_multiplicity_spectrum'][index] for _,_,a in st_tri],
        'o-', color=('#0072B2','#CC79A7')[index], label=f'Singlet multiplicity eigenvalue {index+1}')
axes[0,2].plot(st_j, [a['multiplicity_entropy'] for _,_,a in st_tri], 'o-', color='#0072B2')
axes[0,0].set(ylabel=r'$p_A(q)$', title='Triangular N16: local spin weights')
axes[0,1].set(ylabel=r'Eigenvalues of $\sigma_{q=0}$', title='Same q, different multiplicity spectrum')
axes[0,2].set(ylabel=r'$\sum_q p_q S(\sigma_q)$ [nats]', title='Multiplicity contribution changes')
for ax in axes[0]: ax.set_xlabel(r'$J_2/J_1$')
axes[0,0].legend(frameon=False,fontsize=8); axes[0,1].legend(frameon=False,fontsize=7)
st_square = [(n,a) for n,a in st_states.items() if n.startswith('square_')]
for index,(name,a) in enumerate(st_square):
    axes[1,0].plot([0,2,4], [a['sectors'][str(q)]['p'] for q in (0,2,4)],
        marker=st_markers[index], ls='-' if index==0 else '--', ms=5,
        label=f"VAP seed {a['provenance']['source'].split('_s')[-1].split('_')[0]}")
axes[1,0].set(xticks=[0,2,4],xlabel='Doubled local spin q',ylabel=r'$p_A(q)$',
              title='Square J2=0.5: two saved VAP states')
axes[1,0].legend(frameon=False,fontsize=7)
st_controls = [('dimer_internal','Internal dimers','#0072B2'),
               ('dimer_four_crossing','Four crossing dimers','#D55E00'),
               ('multiplicity_bell_q0','Multiplicity Bell','#009E73')]
for index,(name,label,color) in enumerate(st_controls):
    a=st_states[name]
    axes[1,1].plot([0,2,4], [max(0.,a['sectors'][str(q)]['p']) for q in (0,2,4)],
        marker=st_markers[index], color=color, ls='--' if index==2 else '-', label=label, ms=6)
    axes[1,2].plot(a['entropy_nats'], index, 'o', color=color, ms=7,
        label='Total entropy' if index==0 else None)
    axes[1,2].plot(a['multiplicity_entropy'], index+.12, 's', mfc='none', mec=color, ms=7,
        label='Multiplicity entropy' if index==0 else None)
axes[1,1].set(xticks=[0,2,4],xlabel='Doubled local spin q',ylabel=r'$p_A(q)$',
              title='Analytic singlets, identical physical cut')
axes[1,1].legend(frameon=False,fontsize=7)
axes[1,2].set(yticks=np.arange(3)+.06,yticklabels=[label for _,label,_ in st_controls],
              xlabel='Entropy [nats]',title='Spin weights do not fix entanglement')
axes[1,2].legend(frameon=False,fontsize=7,loc='lower right')
axes[1,2].set_ylim(2.5,-.45)
for ax in axes.flat: ax.grid(alpha=.15)
fig.suptitle('SU(2) physical readout on A={0,1,4,5}; finite-region diagnostics, not phase identification',fontsize=12)
fig.savefig(out/'fig123_physical_su2_multiplicity_readout.png',dpi=180)
fig.savefig(out/'fig123_physical_su2_multiplicity_readout.pdf')
plt.show()

st_rows=['| State | p(q=0,2,4) | q=0 conditional spectrum | Total entropy | Multiplicity entropy |',
         '| --- | --- | --- | ---: | ---: |']
for name,a in st_states.items():
    pq=', '.join(f"{max(0.,a['sectors'][str(q)]['p']):.6f}" for q in (0,2,4))
    spectrum=', '.join(f'{x:.6f}' for x in a['sectors']['0']['conditional_multiplicity_spectrum'])
    st_rows.append(f"| {name} | {pq} | {spectrum} | {a['entropy_nats']:.6f} | {a['multiplicity_entropy']:.6f} |")
display(Markdown('\n'.join(st_rows)))
display(Markdown('Internal dimers and the multiplicity Bell state both have $p(q=0)=1$, '
    r'but $S_\mathrm{mult}=0$ versus $\log2$: **global singlet symmetry and even identical local q weights do not fix the physical state**. '
    'The small smooth triangular trends are not QSL evidence. J2=0, 0.125, 0.175 references are exact within the saved 14-dimensional Gamma-A1 singlet sector; '
    'J2=0.075 is the saved full-M0 reference. Tiny roundoff probabilities are clipped only for display; inactive-q conditional spectra have no interpretation.'))
"""), markdown(r"""
### Bounded coupling transfer: read only when the result is complete

The shared models see $J_2/J_1=0.2,0.5,0.8$; open markers denote the withheld
$0.35,0.65$ couplings. Pointwise controls intentionally optimize at all five
couplings. Both representations use the same backbone and parameters at
initialization, **not the same physical state**, because singlet projection
changes the initializer. The reported aggregate H-objective count is matched;
steps per coupling, wall time, and projection cost are not.

Only $J_2=0.5$ has a saved exact energy anchor. The Hellmann–Feynman defect is
a diagnostic of the learned branch, not a validated susceptibility or phase
boundary. Two historical warmstarts do not define statistical error bars.
"""), code(r"""
st_transfer_file = project/'results/spin_coupling_transfer_v1/results.json'
if not st_transfer_file.exists():
    display(Markdown('**Coupling-transfer calculation not complete: Fig124 is pending. No energies or response data are inferred.**'))
else:
    st_pending_payload = json.loads(st_transfer_file.read_text())
    if st_pending_payload.get('status') != 'complete':
        display(Markdown('**Coupling-transfer calculation remains incomplete; Fig124 is not generated.**'))
    else:
        st_tpath, st_transfer = st_load('spin_coupling_transfer_v1')
        st_tcfg = st_transfer['config']; st_js=np.asarray(st_tcfg['all_couplings'])
        assert not st_tcfg['heldout_in_objective'] and st_transfer['new_ED_calls']==0
        fig, axes = plt.subplots(3,2,figsize=(12,11),constrained_layout=True)
        st_repcolors={'fixed_sz':'#D55E00','singlet_vap':'#0072B2'}
        st_replabels={'fixed_sz':'Fixed Sz + A1','singlet_vap':'Singlet VAP + A1'}
        for col,parent in enumerate(st_tcfg['parents']):
            for rep in st_tcfg['representations']:
                color=st_repcolors[rep]; prefix=parent+'__'+rep+'__'
                shared=st_transfer['audit'][prefix+'shared']
                for kind,style in (('shared','-'),('pointwise','--')):
                    rows=[shared[str(j)] if kind=='shared' else
                          st_transfer['audit'][prefix+f'point_{j:.2f}'][str(j)] for j in st_js]
                    for ax,metric in zip(axes[:2,col],('energy_per_site','variance_per_site')):
                        ys=np.array([row[metric] for row in rows])
                        ax.plot(st_js,ys,style,color=color,lw=1.3,label=st_replabels[rep]+'; '+kind)
                        for j,y in zip(st_js,ys):
                            held=j in st_tcfg['held_out_couplings']
                            ax.plot(j,y,'o' if kind=='shared' else 's',color=color,ms=5,
                                    mfc='white' if held else color)
                hf=[shared[str(j)]['response']['hellmann_feynman_defect'] for j in st_js]
                axes[2,col].plot(st_js,hf,'-',color=color,label=st_replabels[rep])
                for j,y in zip(st_js,hf):
                    axes[2,col].plot(j,y,'o',color=color,mfc='white' if j in st_tcfg['held_out_couplings'] else color)
                print(parent,rep,'max derivative/FD mismatch',
                      max(shared[str(j)]['response']['energy_derivative_fd_error'] for j in st_js),
                      'max curvature/FD mismatch',
                      max(shared[str(j)]['response']['energy_curvature_fd_error'] for j in st_js))
            axes[0,col].set_title('Saved parent '+parent.rsplit('_s',1)[-1])
            axes[0,col].legend(frameon=False,fontsize=7)
            axes[1,col].set_yscale('log')
            axes[2,col].axhline(0,color='0.5',ls=':',lw=1)
            for ax in axes[:,col]:
                ax.set_xlabel(r'$J_2/J_1$'); ax.grid(alpha=.15)
                ax.ticklabel_format(axis='x',style='plain',useOffset=False)
        axes[0,0].set_ylabel('Full-support variational E/N')
        axes[1,0].set_ylabel('Full-support Var(H)/N')
        axes[2,0].set_ylabel(r'$d(E/N)/dJ_2 - \langle H_2\rangle/N$')
        fig.suptitle('Square N16 coupling transfer — open markers withheld only from shared fits',fontsize=12)
        fig.savefig(out/'fig124_coupling_transfer_energy_response.png',dpi=180)
        fig.savefig(out/'fig124_coupling_transfer_energy_response.pdf')
        plt.show()
        print('Transfer result hash:',st_transfer['result_sha256'])
        print('Measured seconds:',st_transfer['seconds'],'aggregate training H objectives:',st_transfer['total_training_H_objectives'])
        display(Markdown('Complete exact-support energies are variational, but this is not a converged-ground-state or scaling certificate. '
            'Report both historical parents and all pointwise controls; do not select an initialization after reading the held couplings.'))
"""), code(r"""
if 'st_transfer' in globals():
    # Fig125: shared-versus-pointwise agreement, not ground-state accuracy.
    fig, axes = plt.subplots(3,2,figsize=(12,9),constrained_layout=True)
    for col,parent in enumerate(st_tcfg['parents']):
        prefix=parent+'__singlet_vap__'
        for kind,style,color in (('shared','-','#0072B2'),('pointwise','--','#D55E00')):
            rows=[st_transfer['audit'][prefix+'shared'][str(j)] if kind=='shared' else
                  st_transfer['audit'][prefix+f'point_{j:.2f}'][str(j)] for j in st_js]
            readouts=[row['multiplicity_readout']['sectors']['0'] for row in rows]
            for ax,ys in zip(axes[:2,col],([r['p'] for r in readouts],
                    [r['conditional_multiplicity_spectrum'][0] for r in readouts])):
                ax.plot(st_js,ys,style,color=color,label=kind)
                for j,y in zip(st_js,ys):
                    ax.plot(j,y,'o',color=color,mfc='white' if j in st_tcfg['held_out_couplings'] else color)
        comparisons=[next(r for r in st_transfer['comparisons']
                          if r['parent']==parent and r['coupling']==j)['arms']['singlet_vap'] for j in st_js]
        for key,label,color in (('plaquette_trace_distance_to_pointwise',r'$D(\rho_A^{shared},\rho_A^{point})$','#0072B2'),
                                ('p_q_TV_to_pointwise',r'TV of $p_A(q)$','#D55E00')):
            axes[2,col].plot(st_js,[r[key] for r in comparisons],'o-',color=color,label=label)
        axes[0,col].set_title('Saved parent '+parent.rsplit('_s',1)[-1])
        for ax in axes[:,col]:
            ax.set_xlabel(r'$J_2/J_1$');ax.legend(frameon=False,fontsize=8);ax.grid(alpha=.15)
    axes[0,0].set_ylabel(r'Local singlet weight $p_A(q=0)$')
    axes[1,0].set_ylabel(r'Largest eigenvalue of $\sigma_{q=0}$')
    axes[2,0].set_ylabel('Distance to separately trained state')
    fig.suptitle('Square N16: local physical structure transfers imperfectly — neither curve is an exact reference',fontsize=11)
    fig.savefig(out/'fig125_coupling_transfer_multiplicity.png',dpi=180)
    fig.savefig(out/'fig125_coupling_transfer_multiplicity.pdf')
    plt.show()

    # Post-hoc PAV decomposition: no retraining or endpoint selection.
    with np.load(st_tpath/'operators.npz',allow_pickle=False) as archive:
        st_ops={key:archive[key] for key in archive.files}
    st_pav=[]
    with np.load(st_tpath/'states.npz',allow_pickle=False) as archive:
        for parent in st_tcfg['parents']:
            for j in st_js:
                physical=archive[parent+'__fixed_sz__shared__'+str(j)]
                orbit=st_ops['orbit_id'];sizes=st_ops['orbit_sizes']
                v=np.zeros(len(sizes),dtype=complex)
                np.add.at(v,orbit,physical/np.sqrt(sizes[orbit]))
                v=st_ops['p0']@v;v/=np.linalg.norm(v)
                hv=(st_ops['h0']+j*st_ops['h2'])@v;e=float(np.vdot(v,hv).real)
                vap=st_transfer['audit'][parent+'__singlet_vap__shared'][str(j)]
                st_pav.append(dict(parent=parent,j=j,delta_E=vap['energy_per_site']-e/16,
                                  delta_V=vap['variance_per_site']-np.linalg.norm(hv-e*v)**2/16))
    st_rows=['| Parent | Withheld J2 | SU2 minus fixed-Sz E/N | SU2 minus post-hoc PAV E/N | Shared minus pointwise SU2 E/N |',
             '| --- | ---: | ---: | ---: | ---: |']
    for row in st_transfer['comparisons']:
        if row['coupling'] not in st_tcfg['held_out_couplings']:continue
        pav=next(r for r in st_pav if r['parent']==row['parent'] and r['j']==row['coupling'])
        st_rows.append(f"| {row['parent'].rsplit('_s',1)[-1]} | {row['coupling']} | "
            f"{row['shared_SU2_minus_fixedSz_energy']:+.7f} | {pav['delta_E']:+.7f} | "
            f"{row['arms']['singlet_vap']['shared_minus_pointwise_energy']:+.7f} |")
    display(Markdown('\n'.join(st_rows)))
    print('Post-hoc VAP better than saved-state PAV in BOTH E and variance:',
          sum(r['delta_E']<0 and r['delta_V']<0 for r in st_pav),'/',len(st_pav))
    display(Markdown('PAV is **projection after variation** of the saved raw shared state. '
        'This descriptive, post-hoc decomposition was not a registered winning criterion. '
        'It suggests the advantage is not solely a cheap projection at the endpoint; it is not an independent replication.'))
"""), code(r"""
# Fig126: saved-endpoint stationarity and coupling-channel capacity.
st_spath, st_stationarity = st_load('spin_coupling_stationarity_v1')
assert st_stationarity['training_updates_committed'] == 0
assert st_stationarity['new_ED_calls'] == st_stationarity['new_samples'] == 0
fig, axes = plt.subplots(2,3,figsize=(14,8.2),constrained_layout=True)
st_parentcolors=['#0072B2','#D55E00']
for color,parent in zip(st_parentcolors,st_tcfg['parents']):
    name=parent+'__singlet_vap__shared'; label=parent.rsplit('_s',1)[-1]
    rows=[st_stationarity['records'][name+'__'+str(j)] for j in st_js]
    probes=[r['subspaces']['training_J_only'] for r in rows]
    axes[0,0].plot(st_js,1e6*np.array([p['delta_energy_per_site'] for p in probes]),'o-',color=color,label=label)
    axes[0,1].plot(st_js,1e5*np.array([p['delta_variance_per_site'] for p in probes]),'o-',color=color,label=label)
    learned=[r['conditioning_capacity']['learned_coupling_tangent']['captured_squared_fraction'] for r in rows]
    available=[r['conditioning_capacity']['all_patch_bias_tangents']['captured_squared_fraction'] for r in rows]
    axes[0,2].plot(st_js,available,'o-',color=color,label=label+'; all 15 bias tangents')
    axes[0,2].plot(st_js,learned,'o--',color=color,mfc='white',label=label+'; learned J tangent')
for ax in axes[0]:
    ax.axhline(0,color='0.5',ls=':',lw=1);ax.set_xlabel(r'$J_2/J_1$');ax.grid(alpha=.15)
axes[0,0].set(ylabel=r'Finite-probe $\Delta E/N\;[10^{-6}]$',title=r'FS=$10^{-3}$: every shared SU2 energy falls')
axes[0,1].set(ylabel=r'Finite-probe $\Delta\mathrm{Var}(H)/N\;[10^{-5}]$',title='Variance: one high-J exception')
axes[0,2].set(ylabel='Squared residual captured',yscale='log',title='Learned J line underuses bias tangent space')
for ax in axes[0]:ax.legend(frameon=False,fontsize=7)
for col,parent in enumerate(st_tcfg['parents']):
    name=parent+'__singlet_vap__shared'
    matrix=np.array(st_stationarity['gradient_conflicts'][name]['training_J']['cosines'])
    heat=axes[1,col].imshow(matrix,vmin=-1,vmax=1,cmap='coolwarm')
    axes[1,col].set(xticks=range(3),yticks=range(3),xticklabels=['.2','.5','.8'],yticklabels=['.2','.5','.8'],
        xlabel=r'Gradient coupling $J_2/J_1$',ylabel=r'Gradient coupling $J_2/J_1$',
        title='Gradient cosines; '+parent.rsplit('_s',1)[-1])
    for i in range(3):
        for k in range(3):axes[1,col].text(k,i,f'{matrix[i,k]:.2f}',ha='center',va='center',fontsize=8)
fig.colorbar(heat,ax=axes[1,:2],shrink=.75,label='Cosine (coordinate dependent)')
for color,parent in zip(st_parentcolors,st_tcfg['parents']):
    secs=[st_stationarity['saved_state_projector_secants'][parent+'__'+str(j)] for j in st_js[1:-1]]
    label=parent.rsplit('_s',1)[-1]
    axes[1,2].plot(st_js[1:-1],[s['projector_secant_cosine'] for s in secs],'o-',color=color,label=label+'; direction cosine')
    axes[1,2].plot(st_js[1:-1],[s['first_over_second_speed'] for s in secs],'s--',color=color,mfc='white',label=label+'; speed ratio')
axes[1,2].axhline(1,color='0.5',ls=':',lw=1)
axes[1,2].set(xlabel=r'$J_2/J_1$',ylabel='Shared versus pointwise',
    title='Coarse paths align; response uncertified')
axes[1,2].grid(alpha=.15);axes[1,2].legend(frameon=False,fontsize=7)
fig.suptitle('Square N16 diagnosis: nonstationarity precedes a conditioning-capacity claim',fontsize=12)
fig.savefig(out/'fig126_coupling_stationarity_capacity.png',dpi=180)
fig.savefig(out/'fig126_coupling_stationarity_capacity.pdf')
plt.show()
for parent in st_tcfg['parents']:
    joint=st_stationarity['gradient_conflicts'][parent+'__singlet_vap__shared']['mean_training_objective_product_FS_training_gradient_span']
    print(parent,'training-product FS unit slope',joint['unit_FS_descent_slope_per_site'],
          'declared-span residual fraction',joint['projected_residual_fraction'])
print('All finite probes matched:',sum(
    p['physical_step']['matched'] for row in st_stationarity['records'].values()
    for p in row['subspaces'].values()),'/',st_stationarity['finite_state_probes'])
"""), markdown(r"""
### What is established, and what is not

* **Cached N36:** all 24 fit-set FS distances now match the requested radius
  (worst relative error $2.99\times10^{-5}$). Omitted-chain energy goes down
  in 8/12 attention and 9/12 free-table cases; variance goes down in 6/12 each.
  Equal fit distance does not give equal held distance or robust generalization.
  These parents already saw every record: there is no untouched-audit claim.
* **Physical readout:** identical global singlet symmetry does not fix local
  spin weights or multiplicity entanglement. Even $p_A(0)=1$ allows zero or
  $\log 2$ multiplicity entropy in the explicit controls. The triangular
  small-cluster trend is a measurable rearrangement, not a phase diagnosis.
* **Bounded transfer:** strict-SU2 shared VAP lowers both E and variance
  relative to the fixed-Sz shared network in all four withheld parent/coupling
  cases, and also beats its post-hoc PAV. However it loses to the independently
  trained SU2 control at all four withheld points. The two old parents are
  design cases, not enough independent seeds for significance or scaling claims.
* **Response is not reliable:** at J2=0.65 the strict shared branch has HF
  defects about -0.0525 and -0.0508 per site. At J2=0.8 the second parent's
  learned energy curvature is +6.20 per site. The ground energy of
  $H_0+J_2H_2$ is concave, so this is not the ground-state response.
  AD/FD and the HF residual identity agree: this is not a derivative bug.
  A vanishing HF defect alone would still not certify susceptibility.
* **The immediate limitation is nonstationarity:** in the fixed span generated
  by only the three training-coupling gradients, a separately FS-matched
  $10^{-3}$ probe lowers shared-SU2 energy at all ten parent/coupling points
  by $7.44\times10^{-6}$ to $3.61\times10^{-5}$ per site. Nine also lower
  variance. A common product-state training direction has nonzero unit-FS
  slopes -0.01579 and -0.00454 per site. These are strict lower bounds from
  a declared three-direction span, not full tangent-space stationarity.
* **The conditioning channel is underused, not disproved:** LayerNorm removes
  the uniform component, so the 16 stored slope weights supply at most 15
  physical bias directions; all ten measured Jacobians have rank15. The
  learned J tangent captures only 0.0366%–4.45% of the squared energy residual,
  whereas all bias tangents capture 2.54%–67.1%. Yet shared-versus-pointwise
  coarse projector secants have cosines0.877–0.979 and speed ratios0.846–1.013.
  The coarse path is not simply wrong, while fine response is uncontrolled.

The existing local attention has positional input, but just two collinear,
equal-distance environments and a one-layer query readout. A future spatial
test needs inequivalent environments/message propagation, not merely another
radial bias. It has **not** been added here. The separate transfer experiment
reuses the existing full factored-Spatial-ViT, with only a 16-weight coupling
input, exact small-support VAP and no magnetic/dimer order targets. It is not
a scalable SU2 decoder or a demonstrated foundation neural quantum state.
At one fixed coupling its state manifold is exactly the old ViT manifold:
the slope can be absorbed into the freely trainable patch bias. Its actual
restriction is the relation shared across couplings, not single-point capacity.

All numerical checkpoints and endpoints are retained. No N36 training/sampling,
ED recomputation, architecture expansion or GPU production is released by this
diagnostic. Figure126 uses saved operators and fixed virtual probes; no probe
is committed as a checkpoint. These figures belong to exploration, not the
final article notebook.
""")]
    cells.extend(full_tangent_single_J_cells())
    cells.extend(finite_schur_cost_cells())
    if (ROOT/'projects/clebsch_tree_nqs/results/spin_finite_schur_optimize_v1/audit.json').exists():
        cells.extend(finite_schur_optimization_cells())
    if (ROOT/'projects/clebsch_tree_nqs/results/spin_finite_schur_protected_v1/audit.json').exists():
        cells.extend(finite_schur_protected_cells())
    if (ROOT/'projects/clebsch_tree_nqs/results/spin_finite_schur_direction_v1/audit.json').exists():
        cells.extend(finite_schur_direction_cells())
    if (ROOT/'projects/clebsch_tree_nqs/results/spin_finite_schur_jvp_v1/results.json').exists():
        cells.extend(finite_schur_jvp_cells())
    if (ROOT/'projects/clebsch_tree_nqs/results/spin_finite_schur_rayleigh_v1/results.json').exists():
        cells.extend(finite_schur_rayleigh_cells())
    if (ROOT/'projects/clebsch_tree_nqs/results/spin_finite_schur_moments_v1/results.json').exists():
        cells.extend(finite_schur_moment_cells())
    if (ROOT/'projects/clebsch_tree_nqs/results/spin_finite_schur_curvature_v1/results.json').exists():
        cells.extend(finite_schur_curvature_cells())
    if (ROOT/'projects/clebsch_tree_nqs/results/spin_finite_schur_reliability_v1/results.json').exists():
        cells.extend(finite_schur_reliability_cells())
    if (ROOT/'projects/clebsch_tree_nqs/results/spin_finite_schur_generalization_v1/results.json').exists():
        cells.extend(finite_schur_generalization_cells())
    if (ROOT/'projects/clebsch_tree_nqs/results/spin_finite_schur_population_v2/results.json').exists():
        cells.extend(finite_schur_population_cells())
    if (ROOT/'projects/clebsch_tree_nqs/results/spin_finite_schur_reanchor_v1/results.json').exists():
        cells.extend(finite_schur_reanchor_cells())
    for cell in cells:
        cell.metadata['experiment_section'] = 'symmetry_transfer_readout'
    return cells


def full_tangent_single_J_cells():
    """Saved complete tangents and the single candidate; no new training/ED."""
    return [markdown(r"""
### Complete common tangent and single-J SU(2) tensor decoder (2026-10-01)

All four original/SR endpoints have full shared product rank180/180 at
cutoffs1e-6,1e-8,1e-10, even with w held fixed. Every individual ray has rank60.
The residual is fully accessible locally; the failed48-step optimization is
not evidence for affine capacity failure. Neither local rank nor monotone
training guarantees correct held-coupling response. The independent rays have
independent phase gauges; a shared optimum can in general retain pointwise
gradients, although these full-rank endpoints are not stationary.

The sole new candidate computes complex coefficients from ALL local(q,a) and
inter-cell q labels, using local convolution plus global attention. Fixed
Schur/CG tensors linearly decode them; full A1 is imposed by the fixed orbit
map. There is no nonlinear function of magnetic indices or physical bits.
The carrier is additive and starts at exactly the saved strong state, so its
nodes are not frozen. All6002 neural weights are trained jointly.

Two preset parents at square4x4,J2/J1=.5 receive32 natural-gradient steps.
Continue original VAP as a strong control at the same initial physical ray,
step count and trust cap. Both reach near-saturated energy; CNN/attention does
NOT beat the continued control in energy. Variance is slightly smaller, but
counts/cost are not matched: no architecture, SU2 efficiency or PRL win.

This COMPLETE-BASIS oracle is exponential, not a scalable N36 tensor network.
v1 stops on the control's1.09e-8 dual residual versus1e-8 tolerance; v2 raises
common damping .001->.01 without relaxing that tolerance. v2 is authoritative;
no v1 neural outcome is selected. Next: a finite-bond, symmetry-preserving
contraction of this candidate at a SINGLE coupling, with a free-reduced-tensor
control. Do not add heads, more coupling inputs or small-size tail iterations.
Figures127/128 are exploratory; the paper notebook remains unchanged.
"""), code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
root = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p/'su2_nqs').is_dir())
project = root/'projects/clebsch_tree_nqs'
ft = json.loads((project/'results/spin_coupling_full_tangent_v1/results.json').read_text())
sc = json.loads((project/'results/spin_reduced_conv_attention_v2/results.json').read_text())
out = project/'figures/exploration'
out.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
for i,parent in enumerate(sc['protocol']['parents']):
    color = ('#0072B2','#D55E00')[i]
    slopes=[]
    for stage,style in (('original',':'),('product_sr','-')):
        row=ft['records'][parent+'__'+stage]['training_product']['all']['spectra']
        s=np.array(row['parameter_coordinates']['singular_values'][:180])
        ax[0].semilogy(np.arange(1,181),s/s[0],style,color=color,label=f'Parent{i+1}: {stage}')
        slopes.append(-row['column_equilibrated']['cutoffs']['1e-10']['unit_product_FS_descent_slope_per_site'])
    ax[1].plot([0,1],slopes,'o-',color=color,label=f'Parent{i+1}')
ax[0].axhline(1e-6,color='0.6',ls='--',lw=.7)
ax[0].set(xlabel='Physical tangent singular-value index',ylabel=r'$s_k/s_1$',title='Full-rank common tangent (180)')
ax[1].set_xticks([0,1],['Original','48-step SR'])
ax[1].set(ylabel='Descent per unit FS (energy/site)',title='Remaining physical descent')
for a in ax:a.legend(frameon=False,fontsize=8)
fig.savefig(out/'fig127_complete_common_tangent.png',dpi=180)
fig.savefig(out/'fig127_complete_common_tangent.pdf')
plt.show()
fig,ax=plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
for i,parent in enumerate(sc['protocol']['parents']):
    for arm,color,style in (('reduced_CNN_attention','#0072B2','-'),('continued_original_VAP','#D55E00','--')):
        row=sc['records'][parent+'__'+arm]
        steps=[0]+[h['step'] for h in row['training']['history']]
        energies=[row['initial']['energy_per_site']]+[h['energy']/16 for h in row['training']['history']]
        ax[i].plot(steps,energies,style,color=color,label=arm.replace('_',' '))
        print(parent,arm,'parameters=',row['parameter_count'],'seconds=',round(row['seconds'],3),
              'final energy/variance=',row['final'])
    ax[i].set(xlabel='Fixed-budget SR step (not matched H work)',ylabel=r'$E/N$',title=f'Square 4x4, J2/J1=0.5, parent {i+1}')
    ax[i].ticklabel_format(axis='y',style='plain',useOffset=False)
    ax[i].legend(frameon=False,fontsize=8)
fig.savefig(out/'fig128_single_J_tensor_decoder.png',dpi=180)
fig.savefig(out/'fig128_single_J_tensor_decoder.pdf')
plt.show()
print('Full-tangent validation:',{k:v for k,v in ft.items() if k.startswith('max_')})
print('Decoder validation:',sc['exact_decoder_errors'])
print('Seals:',ft['result_sha256'],sc['result_sha256'])
""")]


def finite_schur_cost_cells():
    """Saved finite-bond decoding/H cost; no new samples or variational claim."""
    return [markdown(r"""
### Finite-bond Schur decoder: measured N36 work, free reduced-tensor control

The sole CNN/attention candidate now has a **specified finite factorization**:
complete four-spin local Schur tensors, whole internal multiplets q=0,2,4
(doubled spin), chi=2 multiplicities, and a fixed singlet root. The internal
virtual dimension is chi*(1+3+5)=18. No physical-bit-dependent truncation.
All magnetic legs enter only fixed linear CG contractions. Full Gamma-A1
means all432 triangular /288 square images, not a positional-bias claim.

The neural network generates the entire bank once per update, outside every
physical/image/H query. A device-only plan coalesces identical bit strings,
retaining full group multiplicities and all216/144 bonds. It is forward-only;
ordinary VMC differentiates projected scores separately, with H ratios fixed.
Exact-key coalescing currently supports N<=63; the separate key-free projected
amplitude supports N64. An N64 full-H work path still needs its own validation.

**Critical boundary:** this is NOT a lossless factorization of the previous
complete-configuration Transformer. A static neural tensor bank is contained
in the untied free reduced-tensor manifold at the SAME bond budget. The latter
has1760 nominal real parameters versus6132 neural parameters; gauge/unused
embedding directions mean neither count is an effective physical dimension.
Both start at the same physical state. There is no expressivity or efficiency
win from the timing test, and no N36 energy optimization or ED is performed.

Only the first1/4 already frozen probes are used. Warm times are medians of
three synchronized CPU executions; lowering/compilation are stored separately.
The4-probe score force is a timing/AD check, NOT an unbiased energy gradient;
the1-probe covariance force is identically zero and is not used as a gradient
cost benchmark. No GPU speed or training-hour extrapolation is made.
The triangular centered additive pi/0-flux form replays the independently
certified three-determinant carrier and retains nonzero residual gradients.
The second reference contraction is included in its quoted H cost.
Its v1 timing uses nonzero residual probes. For production, the new
`vmc/projected_spin_products.py` returns Hphi directly, including at phi=0;
combine carrier/residual/reference products before dividing by the FINAL
state. Do not reconstruct reference Hphi as (Hphi/phi)*phi at its nodes.
Independent node and saved N36 product replays check this separately.
"""), code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from su2_nqs.frontier.artifacts import canonical_sha256
root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs';folder=project/'results/spin_finite_schur_cost_v1'
reports={g:json.loads((folder/g/'results.json').read_text()) for g in ('triangular','square')}
out=project/'figures/exploration';out.mkdir(parents=True,exist_ok=True)
fig,axes=plt.subplots(1,2,figsize=(10,3.9),constrained_layout=True)
for ax,(geometry,r) in zip(axes,reports.items()):
    assert canonical_sha256({k:v for k,v in r.items() if k!='result_sha256'})==r['result_sha256']
    assert r['passed'] and r['training_steps']==r['new_samples']==0
    for key,label,color,style in (
        ('network_full_H','CNN/attention: full H','#0072B2','-'),
        ('free_full_H','Untied free tensors: full H','#D55E00','--'),
        ('network_A1','CNN/attention: full A1 amplitude','#009E73',':')):
        y=[np.median(r['measurements'][f'{key}_b{b}']['execution_seconds']) for b in (1,4)]
        ax.plot([1,4],y,'o'+style,color=color,label=label)
    grad=np.median(r['measurements']['network_score_force_b4']['execution_seconds'])
    ax.plot(4,grad,'s',color='#CC79A7',label='4-probe score AD (cost only)')
    ax.set(title=geometry.capitalize()+f" 6x6, J2/J1={r['case']['j2_over_j1']}",
        xlabel='Frozen physical centers / call',ylabel='Warm CPU seconds / call',yscale='log',xticks=[1,4])
    ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=7)
    print(geometry,'state budget q<=4, chi=2, D=18','neural/free nominal parameters=',
          r['neural_real_parameters'],r['free_real_parameters'],'query counts=',r['query_counts'])
    print('4-probe median milliseconds:',{k:round(1e3*np.median(v['execution_seconds']),3)
        for k,v in r['measurements'].items() if k.endswith('_b4')})
    print('Checks:',r['checks'],'process seconds/GiB:',r['wall_seconds'],r['peak_rss_bytes']/2**30)
fig.suptitle('Finite-bond decoder is tractable; full projected H remains the main cost')
fig.savefig(out/'fig129_finite_schur_decoder_cost.png',dpi=180)
fig.savefig(out/'fig129_finite_schur_decoder_cost.pdf');plt.show()
print('No energy/variance estimate, neural-versus-free win, or GPU production release.')
""")]


def finite_schur_optimization_cells():
    """Fixed-endpoint full-H optimization, with untouched paired uncertainty."""
    return [markdown(r"""
### N36 finite-Schur energy optimization: fixed budget, independent endpoints

Triangular periodic 6x6, J2/J1=0.125. The **same** q<=4, chi=2 decoder,
complete A1 projection and analytic singlet pi/0-flux carrier are retained.
The physical state is C/cscale + s[D(theta)-D(initial)]; normalization units
are calibrated without fitting energy. The two arms start at exactly C.
All CNN/attention weights or all untied complex reduced tensors are trained.
The free manifold contains the neural bank: no greater expressivity claim.

Each arm receives eight sample-space SR attempts, 32 Born centers per attempt,
identical sample/H budgets, the same sampled FS/ESS caps and initial proposal
stream. These are **caps**, not an exact equality of realized physical steps.
The training curves are noisy old-state estimates, not endpoint audits or
best-step selection. No energy/score clipping or residual-node division.

After both final checkpoints are sealed, 128 **fresh** defensive-Q records
from eight independent chains are shared by carrier and both endpoints.
Full Hpsi is evaluated directly; Var(H)/N includes H-squared contributions
at wavefunction nodes. Error bars are paired whole-chain bootstrap standard
errors, not IID sample errors. They do not certify equilibration, rare-tail
coverage or a PRL win. The carrier is the fixed analytic pi/0-flux state,
NOT a newly optimized Jastrow/two-Lanczos baseline. No new ED or GPU jobs.
This figure belongs only to exploration; the article notebook is unchanged.

**Physical failure mechanism:** resolve the final state into its carrier ray
and a quadrature-orthogonal component, psi=alpha*C+t. The added t is still an
exact singlet/A1 state; it is not sector leakage. Its much higher Rayleigh
energy dominates the unfavorable final energy change. The readout below uses
ONLY saved amplitudes/H products. It neither optimizes a linear mixture nor
authorizes another response head, longer run or larger lattice.
"""), code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from su2_nqs.frontier.provenance import file_sha256
root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs';folder=project/'results/spin_finite_schur_optimize_v1'
audit=json.loads((folder/'audit.json').read_text());m=audit['metrics']
reg=json.loads((folder/'registration.json').read_text())
training={a:json.loads((folder/a/'results.json').read_text()) for a in ('network','free')}
assert audit['registration_sha256']==reg['registration_sha256']
assert all(r['fixed_attempts']==8 and r['full_H_centers']==256 for r in training.values())
assert all(file_sha256(folder/a/'endpoint.json')==audit['endpoint_seals'][a] for a in training)
assert file_sha256(folder/'audit.npz')==audit['audit_sha256']
assert not audit['audit_used_for_training'] and m['sample_count']==128
labels={'network':'CNN/attention','free':'Untied free tensors'}
colors={'network':'#0072B2','free':'#D55E00'}
fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
for arm,r in training.items():
    steps=[row['step']-1 for row in r['history']]
    for ax,field in zip(axes[0],('energy_per_site','variance_per_site')):
        ax.plot(steps,[row[field] for row in r['history']],'o-',lw=1,color=colors[arm],label=labels[arm])
for ax,title,ylabel in zip(axes[0],('Noisy training estimates (not audit)','Training variance (not audit)'),
                         ('E/N','Var(H)/N')):
    ax.set(title=title,xlabel='Completed updates at sampled state',ylabel=ylabel)
    ax.grid(alpha=.18);ax.legend(frameon=False,fontsize=8)
axes[0,1].set_yscale('log')
for ax,metric,title,ylabel in zip(axes[1],('energy_per_site','variance_per_site'),
    ('Untouched paired energy change','Untouched paired variance change'),('Delta E/N','Delta Var(H)/N')):
    ax.axhline(0,color='gray',lw=.8)
    for index,arm in enumerate(('network','free')):
        v=m['paired'][arm+'-minus-carrier'][metric]
        ax.errorbar(index,v['value'],yerr=v['chain_bootstrap_se'],fmt='o',capsize=4,color=colors[arm])
    ax.set(title=title,ylabel=ylabel,xticks=[0,1],xticklabels=[labels[a] for a in ('network','free')])
    ax.grid(axis='y',alpha=.18)
fig.suptitle('Same finite-SU(2) manifold: bounded neural/free optimization and fresh audit')
out=project/'figures/exploration';out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/'fig130_finite_schur_energy_optimization.png',dpi=180)
fig.savefig(out/'fig130_finite_schur_energy_optimization.pdf');plt.show()
for a,v in m['arms'].items():
    print(a,'energy',v['energy_per_site'],'variance',v['variance_per_site'],
          'imaginary energy',v['imaginary_energy_per_site'],'importance ESS fraction',v['weight_ess_fraction'],
          'largest residual sample/chain fractions',v['max_residual_sample_fraction'],v['max_residual_chain_fraction'])
print('Paired changes:',json.dumps(m['paired'],indent=2))
print('Independent endpoint distances:',audit['distances'])
with np.load(folder/'audit.npz') as data:
    p,h,q,groups=(data[k] for k in ('psi','hpsi','q','groups'))
    def residual_readout(index,mask):
        c,hc,z,hz,w=p[mask,0],h[mask,0],p[mask,index],h[mask,index],1/q[mask]
        nc=np.sum(abs(c)**2*w);nz=np.sum(abs(z)**2*w)
        alpha=np.sum(c.conj()*z*w)/nc;t=z-alpha*c;ht=hz-alpha*hc
        nt=np.sum(abs(t)**2*w);ec=np.sum(c.conj()*hc*w).real/nc/36
        et=np.sum(t.conj()*ht*w).real/nt/36
        cross=np.sum((alpha.conjugate()*c.conj()*ht+alpha*t.conj()*hc)*w).real/nz/36
        diagonal=nt/nz*(et-ec)
        return np.array([nt/nz,et,cross,diagonal,cross+diagonal])
    for index,arm in enumerate(('network','free'),start=1):
        point=residual_readout(index,np.ones(len(q),bool))
        leave=np.array([residual_readout(index,groups!=chain) for chain in np.unique(groups)])
        count=len(leave);se=np.sqrt((count-1)/count*np.sum((leave-leave.mean(axis=0))**2,axis=0))
        assert np.isclose(point[-1],m['paired'][arm+'-minus-carrier']['energy_per_site']['value'],atol=2e-13)
        print(arm,'[orthogonal weight, E_t/N, coherent-cross DeltaE/N, diagonal DeltaE/N, sum] =',point)
        print('Leave-one-chain jackknife SE =',se)
for a,r in training.items():
    print(a,'process cost',json.loads((folder/f'train_{a}_execution.json').read_text()),
          'FS range',[min(x['diagnostics']['sampled_fs_sine'] for x in r['history']),
                      max(x['diagnostics']['sampled_fs_sine'] for x in r['history'])],
          'largest SR residual',max(x['diagnostics']['relative_linear_residual'] for x in r['history']))
print('Finite pilot only; no best-step selection, optimized-VMC/SOTA comparison or GPU release.')
""")]


def finite_schur_protected_cells():
    """Saved protected optimization versus prior failure; no new calculations."""
    return [markdown(r"""
### Off-training physical-update control and protection of the same carrier

Same triangular6x6,J2/J1=.125, same finite strict-SU2/A1 decoder, same
CNN/attention/free parameters and same centered additive analytic carrier.
No new head, q/chi increase, lattice or ED. Raw SR still uses32 Born centers.
The previous64-record positive-Q normalization cache is now explicitly an
**optimizer control set**, not validation. Known HC and immutable reference
H products/connection plans are reused exactly, never by removing images.

Each attempted update has13 amplitude-only probes for FS=.01 on these
separate controls, then ONE complete-H candidate evaluation. Acceptance
requires nonzero physical motion, cumulative carrier FS<=.04, no control
energy increase versus carrier OR parent, Var(H)/N<=1.1 carrier, and
importance-weight ESS>=.6. Rejecting preserves the entire parent parameter
tree. Failed distance matching is flagged, not called matched. These are
finite-quadrature constraints, NOT global quality certificates.

Eight fixed attempts per arm;256 training-H centers plus512 candidate-control
centers each. Control reference H is prepared once for64 centers. Both arms
have identical budgets, but this costs MORE than the previous unprotected
protocol. No efficiency win from a cost-unmatched comparison. Final
checkpoints precede a NEW128-record/eight-chain audit, with a fresh seed and
full node-safe H/H-squared moments. No old audit is called untouched.

The plot reports acceptance, actual physical motion, and fresh paired energy
and variance changes. Rejection-all/collapse-to-carrier is NOT improvement.
Small, unresolved energy differences do not constitute a PRL/SOTA win;
the carrier is fixed analytic pi/0-flux, not reoptimized two-Lanczos VMC.
Old/new training AND audit seeds differ: comparisons are descriptive, not a
matched-seed causal optimizer ablation. The raw carrier-energy change between
audits is noise/finite-sampling fluctuation of the SAME physical state, not
improvement. New orthogonal singlet/A1 weights are only about4–5e-5; their
energy is still high. The weak coherent energy term, not a new low-energy
residual sector, explains the small net point change printed below.
Use this figure for exploration only; the article notebook remains unchanged.
"""),code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from su2_nqs.frontier.provenance import file_sha256
root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs';folder=project/'results/spin_finite_schur_protected_v1'
reg=json.loads((folder/'registration.json').read_text());prep=json.loads((folder/'prepared.json').read_text())
audit=json.loads((folder/'audit.json').read_text());m=audit['metrics']
old=json.loads((project/'results/spin_finite_schur_optimize_v1/audit.json').read_text())
runs={a:json.loads((folder/a/'results.json').read_text()) for a in ('network','free')}
assert audit['registration_sha256']==reg['registration_sha256']
assert file_sha256(folder/'audit.npz')==audit['audit_sha256']
assert file_sha256(folder/'control.npz')==prep['control_sha256']
assert all(r['fixed_attempts']==8 and r['training_H_centers']==256 and r['control_H_centers']==512 for r in runs.values())
assert all(file_sha256(folder/a/'endpoint.json')==audit['endpoint_seals'][a] for a in runs)
assert prep['control_used_for_training'] and not audit['audit_used_for_training']
colors={'network':'#0072B2','free':'#D55E00'};labels={'network':'CNN/attention','free':'Untied free tensors'}
fig,axes=plt.subplots(2,2,figsize=(10.5,7),constrained_layout=True)
for arm,r in runs.items():
    committed=0.;radius=0.;energies=[];radii=[]
    for row in r['history']:
        if row['accepted']:
            committed=row['quality_gate']['delta_energy_vs_carrier']
            radius=row['quality_gate']['carrier_distance']['fs']
        energies.append(committed);radii.append(radius)
    steps=np.arange(1,9);accepted=np.array([row['accepted'] for row in r['history']])
    axes[0,0].plot(steps,energies,'-',color=colors[arm],label=labels[arm]+f" ({r['accepted_steps']}/8 accepted)")
    axes[0,0].plot(steps[accepted],np.array(energies)[accepted],'o',color=colors[arm])
    axes[0,0].plot(steps[~accepted],np.array(energies)[~accepted],'x',color=colors[arm])
    axes[0,1].plot(steps,radii,'o-',color=colors[arm],label=labels[arm]+' control')
    axes[0,1].plot(9,audit['distances'][arm]['fs'],'s',color=colors[arm],label=labels[arm]+' fresh audit')
axes[0,0].axhline(0,color='gray',lw=.8)
axes[0,0].set(title='Committed control energy (training data)',xlabel='Attempt',ylabel='Delta E/N versus carrier')
axes[0,1].axhline(.04,color='gray',ls='--',lw=.8,label='Control carrier-radius cap')
axes[0,1].set(title='Control and independent endpoint geometry',xlabel='Attempt; 9 = fresh audit',ylabel='Carrier FS angle')
for ax in axes[0]:ax.legend(frameon=False,fontsize=7);ax.grid(alpha=.18)
for ax,metric,title,ylabel in zip(axes[1],('energy_per_site','variance_per_site'),
    ('Fresh paired energy change','Fresh paired variance change'),('Delta E/N','Delta Var(H)/N')):
    ax.axhline(0,color='gray',lw=.8)
    for i,arm in enumerate(('network','free')):
        v=m['paired'][arm+'-minus-carrier'][metric]
        ax.errorbar(i,v['value'],yerr=v['chain_bootstrap_se'],fmt='o',capsize=4,
                    color=colors[arm])
    ax.set(title=title,ylabel=ylabel,xticks=[0,1],xticklabels=[labels[a] for a in ('network','free')])
    ax.grid(axis='y',alpha=.18);ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
fig.suptitle('Unchanged finite-SU(2) ansatz: carrier quality retained, energy gain unresolved')
out=project/'figures/exploration';out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/'fig131_finite_schur_carrier_protection.png',dpi=180)
fig.savefig(out/'fig131_finite_schur_carrier_protection.pdf');plt.show()
print('Prior failed endpoint distances on reused controls:',prep['prior_failed_geometry'])
for arm,r in runs.items():
    print(arm,'accepted attempts',r['accepted_steps'],'fixed training/control H centers',r['training_H_centers'],r['control_H_centers'])
    print('Rejected by:',[x['quality_gate']['rejected_by'] for x in r['history'] if not x['accepted']])
    print('Matched radii/alpha:',[(x['matching']['distance']['fs'],x['matching']['alpha'],x['matching']['matched']) for x in r['history']])
    print('Cost:',json.loads((folder/f'train_{arm}_execution.json').read_text()))
    print('Fresh endpoint distance:',audit['distances'][arm])
for arm,r in m['arms'].items():
    print(arm,'fresh energy',r['energy_per_site'],'fresh variance',r['variance_per_site'],
          'weight ESS fraction',r['weight_ess_fraction'],'residual max-sample/chain fractions',
          r['max_residual_sample_fraction'],r['max_residual_chain_fraction'])
print('Fresh paired changes:',json.dumps(m['paired'],indent=2))
print('Prior unprotected paired changes (different seeds and costs, not a matched ablation):',old['metrics']['paired'])
print('Cached reference-H cost:',prep['reference_work'])
with np.load(folder/'audit.npz') as data:
    p,h,q=data['psi'],data['hpsi'],data['q'];w=1/q;c,hc=p[:,0],h[:,0]
    nc=np.sum(abs(c)**2*w);ec=np.sum(c.conj()*hc*w).real/nc/36
    for index,arm in enumerate(('network','free'),start=1):
        z,hz=p[:,index],h[:,index];nz=np.sum(abs(z)**2*w)
        alpha=np.sum(c.conj()*z*w)/nc;t=z-alpha*c;ht=hz-alpha*hc
        nt=np.sum(abs(t)**2*w);et=np.sum(t.conj()*ht*w).real/nt/36
        cross=np.sum((alpha.conjugate()*c.conj()*ht+alpha*t.conj()*hc)*w).real/nz/36
        diagonal=nt/nz*(et-ec)
        assert np.isclose(cross+diagonal,m['paired'][arm+'-minus-carrier']['energy_per_site']['value'],atol=2e-13)
        print(arm,'orthogonal singlet weight',nt/nz,'E_t/N',et,
              'coherent/diagonal DeltaE/N',cross,diagonal)
print('Control acceptance is not validation; no post-audit continuation or production release.')
""")]


def finite_schur_direction_cells():
    """Saved imaginary-time direction/cross-chain test; no new H or gradients."""
    return [markdown(r"""
### What makes a useful residual? Imaginary-time alignment, not its own energy

For $\psi=\alpha C+t$, with $\langle C|t\rangle=0$, the energy change is

\[
\Delta E=\frac{2\operatorname{Re}[\alpha^*\langle C|H|t\rangle]
                  +\|t\|^2(E_t-E_C)}{\|\psi\|^2}.
\]

An orthogonal component can have HIGH energy and still lower the complete
state's energy through favorable coherence. The first-order target is
$-r=-(H-E_C)C$, not an independently low-energy residual. At finite Q
quadrature the two sampled coherences $\langle t|HC\rangle$ and
$\langle C|Ht\rangle$ must be kept separately: their finite-sample estimates
need not agree even though the independently checked H is Hermitian.
The printed decomposition uses both, not the population identity twice.

**Bounded unchanged-architecture experiment:** triangular6x6,J2/J1=.125;
strict local Schur decoder,q<=4,chi=2; complete432-image A1 and216-bond H;
same centered additive analytic carrier and full neural/free parameters.
Two methods from the SAME initial carrier: original32-record Born-score SR
versus node-safe amplitude-space SR using the KNOWN64-record positive-Q
control cache. Same damping .1, no sweep. Amplitude derivatives include
carrier nodes; positive-Q normalization and complex parent-ray gauge cancel.
Same13 amplitude-only probes to control FS=.01 and ONE full-H control
candidate per method. This is an optimizer comparison, not a new ansatz.
Record counts differ by design; no sample-efficiency claim from that contrast.

Two predetermined four-chain fitting/held-chain splits diagnose alignment
transfer; they do not select hyperparameters or endpoints. All64 controls
are used for the final importance direction, so control fitting/energy gates
are NOT validation. No new training H calls or training samples. Retain every
rejected candidate and rollback its committed endpoint to the carrier.
Seal all four proposals before ONE fresh common128-record/eight-chain audit;
audit raw proposals even when rejected. Rejected raw candidates are NOT
trained checkpoints. The figure shows actual acceptance and paired energy
and variance with whole-chain percentile95% intervals, without comparing raw
carrier energies across different audits. Printed SEs and intervals need not
be symmetric or imply identical conclusions from a point/SE ratio.

The fixed carrier is not optimized VMC/two-Lanczos. Weight ESS is not chain
equilibration or unseen-tail coverage. No SOTA/SU2/attention/PRL/scaling claim,
post-audit continuation, extra heads, ED or GPU production is licensed.
"""),code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from su2_nqs.frontier.provenance import file_sha256
root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs';folder=project/'results/spin_finite_schur_direction_v1'
reg=json.loads((folder/'registration.json').read_text());audit=json.loads((folder/'audit.json').read_text())
runs={a:json.loads((folder/a/'results.json').read_text()) for a in ('network','free')}
assert not audit['audit_used_for_training'] and audit['raw_candidates_include_rejections']
assert file_sha256(folder/'audit.npz')==audit['audit_sha256']
assert all(r['control_H_centers']==128 and r['new_training_H_centers']==0 for r in runs.values())
names=['network_born','network_importance','free_born','free_importance']
labels=['Neural\nBorn','Neural\nImportance','Free\nBorn','Free\nImportance']
colors={'network':'#0072B2','free':'#D55E00'}
fig,axes=plt.subplots(2,2,figsize=(11,7.4),constrained_layout=True)
for arm,r in runs.items():
    for fold,record in enumerate(r['crossfold']):
        axes[0,0].plot([0,1],[record[s]['descent_alignment_cosine'] for s in ('fit','held')],
            marker=('o','s')[fold],ls=('-','--')[fold],color=colors[arm],
            label=f'{arm}, split {fold+1}')
axes[0,0].axhline(0,color='gray',lw=.8)
axes[0,0].set(title='Importance direction: cross-chain alignment',ylabel='Cosine with -(H-E)C',
              xticks=[0,1],xticklabels=['Fitting chains','Other chains'])
axes[0,0].legend(frameon=False,fontsize=7)
for i,name in enumerate(names):
    arm,method=name.split('_');row=runs[arm]['methods'][method]
    pred=row['diagnostics']['common_control_prediction']['predicted_one_sided_energy_slope_total']
    pred*=reg['config']['direction_step_size']*row['matching']['alpha']/36
    actual=row['quality_gate']['delta_energy_vs_carrier']
    axes[0,1].plot([i-.12,i+.12],[pred,actual],'-',color=colors[arm],alpha=.7)
    axes[0,1].plot(i-.12,pred,'^',color=colors[arm])
    axes[0,1].plot(i+.12,actual,'o',color=colors[arm])
axes[0,1].axhline(0,color='gray',lw=.8)
axes[0,1].set(title='Known controls: one-sided prediction vs actual H',ylabel='Delta E/N',
              xticks=range(4),xticklabels=labels)
axes[0,1].plot([],[],'^',color='gray',label='One-sided linear prediction')
axes[0,1].plot([],[],'o',color='gray',label='Actual nonlinear full-H state')
axes[0,1].legend(frameon=False,fontsize=7)
for ax,field,title,ylabel in zip(axes[1],('energy_per_site','variance_per_site'),
    ('Fresh paired energy (95% CI): ALL proposals','Fresh paired variance (95% CI): ALL proposals'),('Delta E/N','Delta Var(H)/N')):
    ax.axhline(0,color='gray',lw=.8)
    for i,name in enumerate(names):
        arm,method=name.split('_');row=audit['metrics']['paired'][name+'-minus-carrier'][field]
        marker='o' if audit['accepted_before_audit'][name] else 'x'
        low,high=row['chain_bootstrap_95']
        ax.errorbar(i,row['value'],yerr=np.array([[row['value']-low],[high-row['value']]]),
                    fmt=marker,capsize=4,color=colors[arm])
    ax.set(title=title,ylabel=ylabel,xticks=range(4),xticklabels=labels)
for ax in axes.ravel():
    ax.grid(alpha=.18)
    if ax is not axes[0,0]:ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
fig.suptitle('Fixed carrier and finite SU(2) ansatz: fitting alignment is not reliable descent\n'
             'Fresh audit: circles = accepted on controls; crosses = rejected raw candidates')
out=project/'figures/exploration';out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/'fig132_finite_schur_direction_alignment.png',dpi=180)
fig.savefig(out/'fig132_finite_schur_direction_alignment.pdf');plt.show()
for arm,r in runs.items():
    print(arm,'cross-chain diagnostics',r['crossfold'])
    for method,row in r['methods'].items():
        print(arm,method,'control decision',row['accepted'],row['quality_gate']['rejected_by'],
              'actual matching',row['matching']['distance'],row['matching']['matched'],
              'full-H work',row['control_H'],'direction fit',row['diagnostics'])
    print('Process cost',json.loads((folder/f'train_{arm}_execution.json').read_text()))
print('Fresh paired audit:',json.dumps(audit['metrics']['paired'],indent=2))
print('Fresh physical distances',audit['distances'])
with np.load(folder/'audit.npz') as d:
    p,h,w=d['psi'],d['hpsi'],1/d['q'];c,hc=p[:,0],h[:,0]
    nc=np.sum(abs(c)**2*w);ec=np.sum(c.conj()*hc*w).real/nc/36
    for index,name in enumerate(names,start=1):
        z,hz=p[:,index],h[:,index];nz=np.sum(abs(z)**2*w)
        alpha=np.sum(c.conj()*z*w)/nc;t=z-alpha*c;ht=hz-alpha*hc
        nt=np.sum(abs(t)**2*w);et=np.sum(t.conj()*ht*w).real/nt/36
        forward=np.sum(alpha*t.conj()*hc*w)/nz/36
        reverse=np.sum(alpha.conjugate()*c.conj()*ht*w)/nz/36
        diagonal=nt/nz*(et-ec)
        de=audit['metrics']['paired'][name+'-minus-carrier']['energy_per_site']['value']
        assert np.isclose(forward.real+reverse.real+diagonal,de,atol=2e-13)
        print(name,'orthogonal weight',nt/nz,'E_t/N',et,
              'forward/reverse coherence',forward,reverse,'diagonal cost',diagonal,
              'Hermiticity quadrature defect',abs(reverse-forward.conjugate()),
              'accepted BEFORE audit',audit['accepted_before_audit'][name])
for name,row in audit['metrics']['arms'].items():
    print(name,'E/N',row['energy_per_site'],'Var/N',row['variance_per_site'],
          'weight ESS',row['weight_ess_fraction'],'largest residual record/chain fractions',
          row['max_residual_sample_fraction'],row['max_residual_chain_fraction'])
print('No endpoint selection from this audit, no continuation/size/GPU release; article notebook unchanged.')
""")]


def finite_schur_jvp_cells():
    """Saved full-H JVP/two-coherence gate; no H, training or new samples here."""
    return [markdown(r"""
### Full-H JVP: test BOTH coherences before changing the carrier

The unchanged triangular6x6,J2/J1=.125 finite-SU2/A1 decoder now evaluates
$\delta\psi$ and $H\delta\psi$ by FORWARD JVP of the existing division-free
full-H kernel, with the exact-key connection plans held constant. All432
spatial images and216 bonds remain. First JVP the complete neural/free
parameter-to-bank map, then JVP the nonlinear tensor contraction; evaluating
a tangent bank as a wavefunction would be incorrect. No residual ratios,
node masking, Hamiltonian truncation, new head or architecture are introduced.

For $\beta=\langle C|\delta\psi\rangle/\|C\|^2$, use
$t=\delta\psi-\beta C$ and $Ht=H\delta\psi-\beta HC$ together. The ACTUAL
finite-Q Rayleigh derivative is

\[
\frac{dE}{ds}=\frac{\operatorname{Re}[\langle t|HC\rangle+
                                   \langle C|Ht\rangle]}{\|C\|^2}.
\]

It is not generally twice the first term on an incomplete quadrature. The
variance derivative also retains complex sampled E. Validate through an
independent dense graph, node cases, actual-parameter finite differences and
N36 graph connections. The finite-difference probe uses only the first FOUR
control centers; its slope must not be compared to the whole64-control slope.

Reuse the SEALED Born/importance directions and old64 optimizer controls;
do not read the consumed audit. Reconstruct two predeclared fold directions
at the same damping for DIAGNOSIS ONLY, never choose/refit/flip a direction
from JVP outcomes. Four64-center primal+tangent actions/arm, plus16 forward
centers/arm for fixed finite differences. A JVP action is not counted as one
ordinary forward call or an efficiency win. All parameters remain eligible;
only the permission to take a fixed step changes.

Before any proposal, the conservative POINT screen requires favorable
forward AND reverse coherences on EACH of the two32-record chain halves,
nonzero motion and max/min half FS speed<=2. This is a declared optimizer
safety policy, NOT a theorem that a rejected direction is globally uphill.
A negative TOTAL slope can be rejected when individual coherences disagree.
Known controls are training/acceptance data, NOT untouched validation.

Only an eligible full-fit direction can attempt a single13-probe radius-
calibrated step, followed by full-H carrier protection on all64 controls AND
each half. Raw per-half FS must stay<=.01; no energy increase and variance
<=1.1C remain. Fold directions cannot update. Save the original carrier if
any permission/quality check fails. Only accepted updates authorize ONE new
128-record common audit; otherwise no new sampling/audit H is performed.
No old endpoint is altered, no new capacity/size/GPU, no PRL win from safety.

The first plot predicts the PREVIOUS finite candidates on the SAME controls
using old sealed step coefficients; its circles are old candidate values,
not newly optimized energies. Points have no statistical error bars because
these are exact functions of a FIXED cache, not population confidence bounds.
This is an exploration-only numerical/physical-coherence diagnosis.
"""),code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from su2_nqs.frontier.provenance import file_sha256
root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs';folder=project/'results/spin_finite_schur_jvp_v1'
reg=json.loads((folder/'registration.json').read_text());decision=json.loads((folder/'results.json').read_text())
runs={a:json.loads((folder/a/'results.json').read_text()) for a in ('network','free')}
source=root/reg['config']['direction_source']
previous={a:json.loads((source/a/'results.json').read_text()) for a in ('network','free')}
assert decision['registration_sha256']==reg['registration_sha256']
assert all(r['JVP_H_centers']==256 and r['finite_difference_H_centers']==16 for r in runs.values())
assert all(file_sha256(folder/a/'products.npz')==r['products_sha256'] for a,r in runs.items())
names=['network_born','network_importance','free_born','free_importance']
labels=['Neural\nBorn','Neural\nImportance','Free\nBorn','Free\nImportance']
colors={'network':'#0072B2','free':'#D55E00'}
fig,axes=plt.subplots(1,3,figsize=(14,4.4),constrained_layout=True)
for i,name in enumerate(names):
    arm,method=name.split('_');r=runs[arm]['records'][method];d=r['diagnostic']['full']
    old=previous[arm]['methods'][method];step=reg['config']['direction_step_size']*old['matching']['alpha']
    values=[step*d['one_sided_energy_slope_per_site'],step*d['energy_slope_per_site'],
            old['quality_gate']['delta_energy_vs_carrier']]
    axes[0].plot(np.array([i-.16,i,i+.16]),values,'-',color=colors[arm],alpha=.6)
    for x,value,marker in zip([i-.16,i,i+.16],values,['^','s','o']):
        axes[0].plot(x,value,marker,color=colors[arm])
    halves=r['diagnostic']['halves']
    axes[1].plot([0,1],[x['energy_slope_per_unit_fs'] for x in halves],
                 marker='o' if method=='born' else 's',ls='-' if method=='born' else '--',
                 color=colors[arm],label=arm+' '+method)
axes[0].axhline(0,color='gray',lw=.8)
axes[0].set(title='Same controls: old candidate energy change',ylabel='Delta E/N',xticks=range(4),xticklabels=labels)
for marker,label in zip(['^','s','o'],['One-sided prediction','Two-sided JVP prediction','Prior finite candidate']):
    axes[0].plot([],[],marker,color='gray',label=label)
axes[0].legend(frameon=False,fontsize=7)
axes[0].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
axes[1].axhline(0,color='gray',lw=.8)
axes[1].set(title='Full two-sided slope at unit physical step',ylabel='d(E/N) / d(FS)',
            xticks=[0,1],xticklabels=['Even-chain half','Odd-chain half'])
axes[1].legend(frameon=False,fontsize=7)
for arm,r in runs.items():
    values=[r['records'][m]['eligibility']['speed_ratio'] for m in ('born','importance','fit_0','fit_1')]
    axes[2].semilogy(range(4),values,'o-',color=colors[arm],label=arm)
axes[2].axhline(2,color='gray',ls='--',lw=.8,label='Declared screening bound')
axes[2].set(title='Cross-chain FS-speed mismatch',ylabel='Max / min half tangent speed',
            xticks=range(4),xticklabels=['Born','Importance','Even fit\n(diag. only)','Odd fit\n(diag. only)'])
axes[2].legend(frameon=False,fontsize=7)
for ax in axes:ax.grid(alpha=.18)
fig.suptitle(f"Full-H JVP before updating: {decision['accepted_updates']} accepted changes; "
             f"{decision['new_audit_records']} new audit records")
out=project/'figures/exploration';out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/'fig133_full_H_JVP_coherence_gate.png',dpi=180)
fig.savefig(out/'fig133_full_H_JVP_coherence_gate.pdf');plt.show()
for arm,r in runs.items():
    print(arm,'actual H budgets',r['JVP_H_centers'],r['finite_difference_H_centers'],r['candidate_H_centers'])
    for name,row in r['records'].items():
        print(arm,name,'FULL',row['diagnostic']['full'],'HALVES',row['diagnostic']['halves'],
              'pre-update permission',row['eligibility'],'diagnostic ONLY',row['diagnostic_only'],
              'FD on first FOUR records',row.get('finite_difference'),'JVP cost',row['JVP_H'])
    print('Updates/endpoints',r['updates'],'process cost',json.loads((folder/f'probe_{arm}_execution.json').read_text()))
print('Final decision',decision)
print('Control screen is not variational improvement or a globally certified gradient; article notebook unchanged.')
""")]


def finite_schur_rayleigh_cells():
    """Saved complete-gradient and actual-chain calibration; no numerical jobs."""
    return [markdown(r"""
### Complete two-sided gradient and ACTUAL worst-chain physical step

Same triangular $6\times6$, $J_2/J_1=0.125$, analytic projected pi/0-flux
carrier, centered additive residual, $q\leq4,\chi=2$ strict SU2 decoder and
complete432-image A1. ALL neural/free parameters remain jointly eligible.
There is no new head or architecture, and this static reduced-bank generator
is not an arbitrary configuration-dependent full ViT.

The missing reverse term is now a full parameter gradient: transpose the
EXISTING full-H connection/image map, accumulate raw contraction pullbacks
in bounded JAX chunks, then VJP the generator. The normalization derivative
and complex parent-ray gauge are included. No presumed Hermitian identity
on incomplete quadrature, H/psi division, or parameter-by-parameter H JVP.
The forward H implementation is unchanged. Gradient dot products reproduce
all old sealed JVP slopes; the new direction also passes actual-parameter FD.

Use the same $M=2\mathrm{FS}+0.1I$ metric. Predeclare a common-descent rule
for the full cache and both parity-chain halves: minimize
$\|\sum_iw_i g_i\|^2_{M^{-1}}$ on the simplex, then
$d=M^{-1}\sum_iw_i g_i$. KKT checks require all three $-g_i\cdot d<0$.
Weights use gradients only, NOT trial energies or an audit. Pooled SR and
half-fitted directions are diagnostic controls, not selectable alternatives.
Only the SUM of coherences must descend; individual terms need not be negative.

Linear FS speed sets a bracket only. Exactly13 amplitude-only probes match
the largest ACTUAL nonlinear FS distance over the full cache, two halves and
all8 individual chains to0.01. A single step does not make every chain's
distance identical; all measured values are shown. All64 records are KNOWN
optimizer data, so half-cache descent is not fresh generalization.

Both directions now descend on both halves and both finite candidates lower
the SAME-cache energy. Neural $\Delta E/N=-0.0022896$, free $-0.0079034$,
but variance ratios are1.312 and2.182 (declared ceiling1.1). Both are rejected;
all final checkpoints remain the EXACT original carrier. No new training or
audit samples, no ED/size/GPU release. These point energies are NOT validated
variational bounds or a superiority claim. The stronger reverse term can
optimize an incomplete quadrature, so cache descent is not enough.
The free gradient has31.6% outside the sampled central-J row space versus
1.81% neural; this is a diagnostic of possible off-cache sensitivity, not a
proved explanation of the physical population variance.

Figure134 is exploration only. Do not tune the radius/variance envelope or
rerun this sealed experiment after seeing the rejection.
"""),code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from su2_nqs.frontier.provenance import file_sha256
root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs';folder=project/'results/spin_finite_schur_rayleigh_v1'
reg=json.loads((folder/'registration.json').read_text());decision=json.loads((folder/'results.json').read_text())
runs={a:json.loads((folder/a/'results.json').read_text()) for a in ('network','free')}
prior_folder=root/reg['config']['jvp_source']
prior={a:json.loads((prior_folder/a/'results.json').read_text()) for a in runs}
assert all(file_sha256(folder/a/'products.npz')==r['product_sha256'] for a,r in runs.items())
assert decision['registration_sha256']==reg['registration_sha256']
colors={'network':'#0072B2','free':'#D55E00'}
fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
for arm,row in runs.items():
    axes[0].plot([0,1],[r['energy_slope_per_unit_fs'] for r in row['tangent']['halves']],
        'o-',color=colors[arm],label=arm+' complete-gradient')
    axes[0].plot([0,1],[r['energy_slope_per_unit_fs'] for r in prior[arm]['records']['importance']['diagnostic']['halves']],
        's--',color=colors[arm],alpha=.5,label=arm+' old importance')
    distances=row['trial'].get('matching',{}).get('distances',{})
    if distances:
        axes[1].plot(range(8),[distances[f'chain_{i}']['fs'] for i in range(8)],'o-',color=colors[arm],label=arm)
    if row['trial']['raw_candidate_constructed']:
        g=row['trial']['quality_gates'][0]
        x,y=g['delta_energy_vs_carrier'],g['delta_variance_vs_carrier']
        axes[2].plot(x,y,'o',color=colors[arm],label=arm)
        ratio=g['candidate']['variance_per_site']/g['carrier']['variance_per_site']
        axes[2].annotate(f"{arm}\nvariance {ratio:.3f}x",(x,y),xytext=(8,-24),textcoords='offset points',fontsize=8)
axes[0].axhline(0,color='gray',lw=.8)
axes[0].set(title='Total derivative at unit FS (known cache)',ylabel='d(E/N) / d(FS)',
    xticks=[0,1],xticklabels=['Even-chain half','Odd-chain half'])
axes[0].legend(frameon=False,fontsize=7)
axes[1].axhline(.01,color='gray',ls='--',label='Actual FS cap')
axes[1].set(title='Nonlinear physical step: every chain',xlabel='Known optimizer chain',ylabel='Actual FS distance',xticks=range(8),ylim=(0,.012))
axes[1].legend(frameon=False,fontsize=7)
first=next(r for r in runs.values() if r['trial']['raw_candidate_constructed'])['trial']['quality_gates'][0]
allowance=(reg['config']['protection']['maximum_variance_ratio']-1)*first['carrier']['variance_per_site']
axes[2].axhline(0,color='gray',lw=.8);axes[2].axvline(0,color='gray',lw=.8)
axes[2].axhline(allowance,color='gray',ls='--',label='Declared variance allowance')
axes[2].set(title='Finite candidates: energy falls, variance rises',xlabel='Delta E/N (known cache)',ylabel='Delta Var(H)/N (known cache)')
axes[2].margins(x=.2,y=.2);axes[2].legend(frameon=False,fontsize=7)
for ax in axes:ax.grid(alpha=.18)
fig.suptitle('Complete two-sided optimizer + worst-chain FS: both candidates rejected; carrier preserved')
out=project/'figures/exploration';out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/'fig134_complete_Rayleigh_actual_chain_FS.png',dpi=180)
fig.savefig(out/'fig134_complete_Rayleigh_actual_chain_FS.pdf');plt.show()
for arm,row in runs.items():
    print(arm,'SR common-gradient/KKT',row['SR'],'pooled control',row['pooled_SR'],'crossfold',row['crossfold'])
    print('Actual physical matching/quality',row['trial'])
    print('Derivative costs',row['pullback_work'],'JVP',row['JVP_seconds'],'FD errors',row['finite_difference_relative_errors'])
    print('Process',json.loads((folder/f'train_{arm}_execution.json').read_text()))
print('Final decision',decision)
print('No new audit, no validated energy win; figure remains exploration-only, article notebook unchanged.')
""")]


def finite_schur_moment_cells():
    """Saved joint energy/residual test, not new training, H calls or ED."""
    return [markdown(r"""
### Joint energy and centered residual second-moment direction

Keep the same triangular $6\times6$, $J_2/J_1=0.125$, strict SU2 decoder,
complete432-image $A_1$, analytic projected pi/0-flux carrier, neural/free
tensor banks and complete216-bond Hamiltonian. No new head, capacity or size.
The residual objective is
$R=\|(H-E)\psi\|_Q^2/(N\|\psi\|_Q^2)$, **not** bare $\langle H^2\rangle$.
Finite-Q $E$ can be complex; both coherent terms and the normalization
derivative are retained. The new cotangents reuse the same raw-product
pullback. Half and full residual gradients require TWO cotangent right-hand
sides per plan; their extra backward work is not called a free forward call.

Six objectives are declared: energy and $R$ on the full64-record cache and
each fixed parity-chain half. Normalize gradients by their inverse common
$M=2\mathrm{FS}+0.1I$ metric norms, then minimize the normalized convex
combination's metric norm. This removes arbitrary energy/residual units.
Weights use gradients only, never candidate energies or old audits. Normalize
the resulting direction to unit full-cache infinitesimal FS. Independent
full-H JVP and actual-parameter finite differences check all derivatives.

Both arms have strictly negative energy and residual derivatives on all three
declared regions. But the single finite proposal is not jointly robust:
the neural full cache lowers both $E/N$ and $R$, while its even-chain half
raises $R$ by3.98%; the free full-cache $R$ rises37.5%. Both are rejected,
so zero fresh audit records or commits. This is **not** a first-order
incompatibility or a SU2/attention no-go, and not an audited energy win.

At most one candidate per arm: match the maximum ACTUAL nonlinear FS across
all8 chains, both halves and the full cache to0.01 with13 amplitude probes.
The predeclared finite-step rule requires **no increase of either energy or
$R$** on full cache and both halves. There is no radius retry or damping
sweep. If neither proposal passes, preserve the exact original carrier and
draw no audit. If one passes, perform one fresh shared128-record full-H audit
of ALL constructed candidates, reporting paired chain uncertainty even for
cache-rejected controls. No post-audit refit.

Known-cache slopes and finite points are optimizer diagnostics, not population
variational bounds. A negative infinitesimal slope need not survive finite
curvature. Fresh point acceptance is also weaker than a resolved paired
energy/variance improvement. This finite static reduced-bank generator is
not a complete configuration-dependent foundation ViT. Figure135 remains
exploration-only; the article notebook is unchanged.
"""),code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.vmc.carrier_protection import physical_moments
root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs';folder=project/'results/spin_finite_schur_moments_v1'
reg=json.loads((folder/'registration.json').read_text());decision=json.loads((folder/'results.json').read_text())
runs={a:json.loads((folder/a/'results.json').read_text()) for a in ('network','free')}
old={a:json.loads((root/reg['config']['moment_source']/a/'results.json').read_text()) for a in runs}
assert decision['registration_sha256']==reg['registration_sha256']
assert all(file_sha256(folder/a/'products.npz')==r['product_sha256'] for a,r in runs.items())
colors={'network':'#0072B2','free':'#D55E00'}
fig,axes=plt.subplots(1,3,figsize=(14,4.7),constrained_layout=True)
for arm,row in runs.items():
    records=[row['tangent']['full'],*row['tangent']['halves']]
    for key,style in [('energy_slope_per_site','o-'),('variance_slope_per_site','s--')]:
        axes[0].plot(range(3),[r[key] for r in records],style,color=colors[arm],
            label=arm+(' dE/N' if key.startswith('energy') else ' dR'))
    trial=row['trial']
    if trial['raw_candidate_constructed']:
        distances=trial['matching']['distances']
        axes[1].plot(range(8),[distances[f'chain_{i}']['fs'] for i in range(8)],'o-',color=colors[arm],label=arm)
        gate=trial['quality_gates'][0]
        x,y=gate['delta_energy_vs_carrier'],gate['delta_variance_vs_carrier']
        axes[2].plot(x,y,'o',color=colors[arm],label=arm+' joint full (known)')
        axes[2].annotate(arm,(x,y),xytext=(7,7),textcoords='offset points',fontsize=8)
        for name,marker,half in zip(('even','odd'),('^','v'),trial['quality_gates'][1:]):
            axes[2].plot(half['delta_energy_vs_carrier'],half['delta_variance_vs_carrier'],
                marker,color=colors[arm],ms=5,label=arm+' '+name+' half (known)')
    previous=old[arm]['trial']
    if previous['raw_candidate_constructed']:
        gate=previous['quality_gates'][0]
        axes[2].plot(gate['delta_energy_vs_carrier'],gate['delta_variance_vs_carrier'],
            'x',color=colors[arm],alpha=.6,label=arm+' energy-only (known)')
audit=None
if (folder/'audit.json').exists():
    audit=json.loads((folder/'audit.json').read_text())
    for arm in runs:
        metrics=audit['metrics']['paired'].get(arm+'-minus-carrier')
        if metrics:
            e,r=metrics['energy_per_site'],metrics['variance_per_site']
            x,y=e['value'],r['value'];ei,ri=e['chain_bootstrap_95'],r['chain_bootstrap_95']
            axes[2].plot(ei,[y,y],'-',color=colors[arm],lw=1)
            axes[2].plot([x,x],ri,'-',color=colors[arm],lw=1)
            axes[2].plot(x,y,'D',color=colors[arm],ms=6,label=arm+' fresh paired95%')
axes[0].axhline(0,color='gray',lw=.8)
axes[0].set(title='Six exact derivatives: known controls',ylabel='Derivative per unit full-cache FS',
    xticks=range(3),xticklabels=['Full','Even chains','Odd chains'])
axes[0].legend(frameon=False,fontsize=8)
axes[1].axhline(.01,color='gray',ls='--',label='Actual FS cap')
axes[1].set(title='Finite physical update: each chain',xlabel='Known optimizer chain',ylabel='Actual FS distance',xticks=range(8),ylim=(0,.012))
axes[1].legend(frameon=False,fontsize=8)
axes[2].axhline(0,color='gray',lw=.8);axes[2].axvline(0,color='gray',lw=.8)
axes[2].set(title='Finite candidates: not population bounds',xlabel='Delta E/N versus same carrier',ylabel='Delta centered residual R')
axes[2].legend(frameon=False,fontsize=7);axes[2].margins(x=.2,y=.2)
for ax in axes:ax.grid(alpha=.18)
fig.suptitle(f"Joint descent: {decision['accepted_updates']} commits; {decision['new_audit_records']} fresh audit records")
out=project/'figures/exploration';out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/'fig135_joint_energy_residual_moments.png',dpi=180)
fig.savefig(out/'fig135_joint_energy_residual_moments.pdf');plt.show()
for arm,row in runs.items():
    print(arm,'Common direction/KKT',{k:row['common'][k] for k in
        ('common_descent','objective_weights','predicted_region_slopes','simplex_kkt_scaled_violation')})
    trial=row['trial']
    print('Fixed finite candidate decision',{k:trial[k] for k in
        ('raw_candidate_constructed','cache_accepted','eligibility')})
    if trial['raw_candidate_constructed']:
        print('Cache full/even/odd Delta E/R and rejection',[
            (g['delta_energy_vs_carrier'],g['delta_variance_vs_carrier'],g['rejected_by']) for g in trial['quality_gates']])
    print('Derivative replay',row['old_JVP_residual_replay_errors'],row['new_JVP_slope_abs_error'])
    print('Actual parameter FD',row['finite_difference_relative_errors'],row['finite_difference_slopes_first_four'])
    print('Derivative cost',row['pullback_work'],'JVP seconds',row['JVP_seconds'])
    print('Process',json.loads((folder/f'train_{arm}_execution.json').read_text()))
    if row['trial']['raw_candidate_constructed']:
        # Algebraic replay of SAVED products only. No H call, new proposal,
        # state-distance retry, audit read or parameter update.
        with np.load(root/reg['config']['previous']/'control.npz') as d:
            c,hc,q,groups=d['carrier'],36*d['hcarrier'],d['q'],d['groups']
        with np.load(folder/arm/'products.npz') as p:delta,hdelta=p['delta'],p['hdelta']
        alpha=row['trial']['matching']['alpha']
        for name,mask,record in zip(['full','even','odd'],[np.ones(len(q),bool),groups%2==0,groups%2==1],
                                   [row['tangent']['full'],*row['tangent']['halves']]):
            parent=physical_moments(c[mask],hc[mask],q[mask],36)
            ray=physical_moments(c[mask]+alpha*delta[mask],hc[mask]+alpha*hdelta[mask],q[mask],36)
            print(arm,name,'first-order predicted DeltaR',alpha*record['variance_slope_per_site'],
                'linear tangent-ray finite DeltaR',ray['variance_per_site']-parent['variance_per_site'])
print('Final decision',decision)
if audit:print('Fresh paired audit, no refit',audit)
else:print('No cache-passing proposal: no fresh audit drawn. Carrier and checkpoints protected.')
print('No PRL/SOTA/scale win inferred; article notebook unchanged.')
""")]


def finite_schur_curvature_cells():
    """Saved curvature direction and fresh audit; no train, sampling or H job."""
    return [markdown(r"""
### Residual-response trust: cache improvement does not transfer

Same triangular $6\times6$, $J_2/J_1=0.125$, exact SU2/full $A_1$ decoder,
analytic projected pi/0-flux carrier, neural/free tensor banks and full H.
No new head or variational capacity. Build a declared six-direction optimizer
span from SAVED Born/importance/half-fit/energy/joint directions and full-H
tangents. Add a centered residual-response metric to the same restricted
$2\mathrm{FS}+0.1I$ geometry. With $t=\delta\psi-\beta\psi$ and
$Ht=H\delta\psi-\beta H\psi$, use
$k=(H-E)t-\delta E\,\psi$, retaining complex finite-Q $E$ and BOTH
coherences in $\delta E$. The positive Gram of $k$ is a Gauss--Newton trust
proxy, **not the full Hessian**. Normalize by each full/half parent $R$ and
average the three regions, fixed weight1. No parameter/weight/radius sweep.

This combines a restricted saved span and response-aware geometry. Without
a preregistered same-span/no-response finite control it does **not** isolate
the response metric's causal effect. It is an optimizer test, not a new
SU2/attention architecture or a demonstration of full-manifold stationarity.
All six energy/residual slopes descend and a NEW full-H JVP verifies the
saved-span combination. Exact tangent-ray moments and actual nonlinear
moments must both pass at the unchanged worst-chain FS cap0.01.

Both single proposals lower E and R on full/even/odd known controls. Full
cache DeltaE/N is -0.0008992/-0.0009566; R falls8.13%/11.57%. Physical
direction cosines with the old joint update are0.868/0.523, so this is not
merely rescaling it. This licenses ONE fresh common128-record audit, not
additional training. The audit reverses the apparent gain: neural/free
DeltaE/N=+0.000339/+0.000183 and R rises31.9%/5.07%. Energy paired95%
intervals include zero; neither shows an energy advantage. Residual changes
have positive paired intervals on these eight chains. Max audit FS becomes
0.0550/0.0221 instead of the known-control0.01 cap. Both proposals rejected;
endpoints remain EXACT original carrier. No audit-driven refit or GPU release.

**Interpretation:** insufficient optimization remains possible, since no
converged trajectory has been demonstrated. But these data do not show
"lower energy plus higher variance" on independent samples: even the energy
gain fails transfer. More steps on the same controls are not a demonstrated
remedy for sample-outside physical geometry. Quantum Var(H) is a state
property, distinct from Monte Carlo error; more samples alone do not make
the state more eigenlike. A genuine independently resolved lower energy
would still be a variational gain even if variance temporarily increased;
the strict joint gate here is a diagnostic/protection choice, not a theorem
that every useful optimization step must lower variance.

Figure136 is exploration only; the article notebook is unchanged. No
population-tail/equilibration guarantee follows from128 records/eight chains.
"""),code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from su2_nqs.frontier.provenance import file_sha256
root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs';folder=project/'results/spin_finite_schur_curvature_v1'
reg=json.loads((folder/'registration.json').read_text());decision=json.loads((folder/'results.json').read_text())
runs={a:json.loads((folder/a/'results.json').read_text()) for a in ('network','free')}
previous={a:json.loads((root/reg['config']['curvature_source']/a/'results.json').read_text()) for a in runs}
audit=json.loads((folder/'audit.json').read_text()) if (folder/'audit.json').exists() else None
assert decision['registration_sha256']==reg['registration_sha256']
assert all(file_sha256(folder/a/'products.npz')==r['product_sha256'] for a,r in runs.items())
colors={'network':'#0072B2','free':'#D55E00'}
fig,axes=plt.subplots(1,3,figsize=(14.5,4.7),constrained_layout=True)
for arm,row in runs.items():
    trial=row['trial']
    if not trial['raw_candidate_constructed']:continue
    for ax,key,metric in zip(axes[:2],('delta_energy_vs_carrier','delta_variance_vs_carrier'),('energy_per_site','variance_per_site')):
        values=[g[key] for g in trial['quality_gates']]
        ax.plot(range(3),values,'o-',color=colors[arm],label=arm+' response-aware, known')
        if previous[arm]['trial']['raw_candidate_constructed']:
            ax.plot(range(3),[g[key] for g in previous[arm]['trial']['quality_gates']],
                'x:',color=colors[arm],alpha=.45,label=arm+' old joint, known')
        if audit:
            summary=audit['metrics']['paired'][arm+'-minus-carrier'][metric]
            ax.plot([3,3],summary['chain_bootstrap_95'],'-',color=colors[arm],lw=1.3)
            ax.plot(3,summary['value'],'D',color=colors[arm],label=arm+' fresh paired95%')
    axes[2].plot(range(8),[trial['matching']['distances'][f'chain_{i}']['fs'] for i in range(8)],
        'o-',color=colors[arm],label=arm+' known controls')
    if audit:
        axes[2].plot(range(8),[audit['distances'][arm][f'chain_{i}']['fs'] for i in range(8)],
            's--',color=colors[arm],label=arm+' fresh audit')
axes[0].set(title='Energy: apparent cache gain fails transfer',ylabel='Delta E/N versus same carrier')
axes[1].set(title='Centered residual moment R',ylabel='Delta R versus same carrier')
for ax in axes[:2]:
    ax.axhline(0,color='gray',lw=.8);ax.axvline(2.5,color='gray',ls='--',alpha=.6)
    ax.set(xticks=range(4),xticklabels=['Full cache','Even cache','Odd cache','Fresh audit'])
    ax.legend(frameon=False,fontsize=7);ax.grid(alpha=.18)
axes[2].axhline(.01,color='gray',ls='--',label='Declared actual FS cap')
axes[2].set(title='Physical step does not generalize',xlabel='Independent chain label',ylabel='Actual FS distance',xticks=range(8),ylim=(0,.06))
axes[2].legend(frameon=False,fontsize=7);axes[2].grid(alpha=.18)
fig.suptitle('Same architecture + residual-response trust: no independent energy/variance win')
out=project/'figures/exploration';out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/'fig136_residual_response_trust_fresh_audit.png',dpi=180)
fig.savefig(out/'fig136_residual_response_trust_fresh_audit.pdf');plt.show()
for arm,row in runs.items():
    print(arm,'Saved span rank',row['common']['saved_span_rank'],'metric coefficient',reg['config']['response_weight'])
    print('Common slopes',row['common']['predicted_region_slopes'],'KKT',row['common']['simplex_kkt_scaled_violation'])
    print('Known full/even/odd candidate E/R',[(g['delta_energy_vs_carrier'],g['delta_variance_vs_carrier']) for g in row['trial'].get('quality_gates',[])])
    print('Exact tangent-ray prediction',row['tangent_ray_at_matched_step'])
    print('Derivative replay/FD',row['saved_span_JVP_relative_errors'],row['finite_difference_relative_errors'])
    print('Process',json.loads((folder/f'train_{arm}_execution.json').read_text()))
    if audit:
        print('Fresh paired E/R',audit['metrics']['paired'][arm+'-minus-carrier'])
        print('Fresh rejection',audit['gates'][arm]['rejected_by'])
        print('Known/full audit FS',row['trial']['matching']['distances']['full']['fs'],audit['distances'][arm]['full']['fs'])
print('Final decisions/checkpoints',decision)
if audit:print('Fresh cost',audit['full_H_work'],audit['sampling'])
print('No post-audit refit; not a resolved variational or PRL result, article notebook unchanged.')
""")]


def finite_schur_reanchor_cells():
    """Retained physical carrier, full-parameter increments and frozen filter control."""
    return [markdown(r"""
### Retained stronger carrier: does the neural increment exceed filtering?

Same triangular $6\times6$, $J_2/J_1=.125$, strict $S=0$ and ALL432
$\Gamma A_1$ images, finite $q\le4/\chi=2$ Schur decoder and216-bond H.
No new head, network capacity, GPU job or continued old trajectory.

Freeze the retained carrier $C_1=aC_0+bHC_0$ and reconstruct
$HC_1=aHC_0+bH^2C_0$ using the SAME TOTAL H. The new state is
$C_1+s[\Phi(\theta)-\Phi(\theta_\mathrm{reference})]$; both the physical
state and H product are centered at C1 throughout optimization/evaluation.
Keep the original network/free-bank initialization, residual units and
resolved192-record geometry. One FIT64 population-E/complete-finite-Q-R
direction per arm, not another force-weighting ladder. Every H-JVP/FD uses
both coherent energy terms. Seal directions before new128 calibration and
a separate shared128 untouched audit; no direction/phase refitting there.

The filter control continues the SAME degree-one polynomial of C0. It is
**not** a second Lanczos step on C1, optimized two-Lanczos or a SOTA VMC state.
Its positive spectral ratio to C1 certifies true energy nonincrease but NOT
variance descent. Actual chain FS<=.01 and strict energy/variance checks
remain; response-limited neural underfill is reported, not equal-distance win.
Sampling retains cheaper positive C0-Q with an exchange bridge/uniform floor;
exact Q reweights C1/candidates. No C1 Born-sampling or mixing certificate.

Read out the new tangent outside span(C0,HC0) without removing/refitting it.
Linear independence does not imply lower energy. Compare actual independent
energy/variance with C1 AND the frozen filter candidate. This existing bank
generator uses local convolution/attention but is not a complete
configuration-dependent foundation ViT. Figure141 is exploration only.

If a starred neural probe is present, it is explicitly a **post-calibration,
pre-audit diagnostic extension**. Original strict half-chain refusal stays
unchanged. Only a direction with downward energy in every fold and downward
full residual gets one full-response-capped finite probe; do not refit it.
An energy-uphill free direction is skipped logically. The probe uses the
SAME future audit, no additional samples, and does not commit the original
neural endpoint or release production. Its adaptive status is never hidden.
"""),code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs';folder=project/'results/spin_finite_schur_reanchor_v1'
result=json.loads((folder/'results.json').read_text());audit=json.loads((folder/'audit.json').read_text())
proposals={a:json.loads((folder/a/'proposal.json').read_text()) for a in ('network','free')}
filt=json.loads((folder/'filter.json').read_text());names=['network','free','filter']
labels=['Neural','Free tensors','Continued filter']
diagnostic=None
if (folder/'diagnostic/results.json').exists():
    diagnostic=json.loads((folder/'diagnostic/results.json').read_text())
    diagnostic_proposals=json.loads((folder/'diagnostic/proposals.json').read_text())
    if 'network' in diagnostic['arms']:labels[0]='Neural probe*'
fig,axes=plt.subplots(1,2,figsize=(10.4,4),constrained_layout=True)
for col,(key,sekey,title) in enumerate([('delta_energy_vs_carrier','energy_per_site','Energy relative to retained C1'),
        ('delta_variance_vs_carrier','variance_per_site','Residual variance relative to C1')]):
    for index,name in enumerate(names):
        gates=filt['quality_gates'] if name=='filter' else proposals[name]['trial'].get('quality_gates',[])
        audit_gate=audit['gates'].get(name)
        if diagnostic and name in diagnostic['arms']:
            gates=diagnostic_proposals['arms'][name]['quality_gates']
            audit_gate=diagnostic['arms'][name]['C1_gate']
        if gates:
            g=gates[0];axes[col].errorbar(index-.08,g[key],yerr=g['paired_chain_jackknife_se'][sekey],fmt='o',capsize=4,color='#E69F00',label='Calibration' if index==0 else None)
        if audit_gate:
            g=audit_gate;axes[col].errorbar(index+.08,g[key],yerr=g['paired_chain_jackknife_se'][sekey],fmt='s',capsize=4,color='#0072B2',label='Untouched audit' if index==0 else None)
        else:axes[col].text(index,.97,'No finite\ncandidate',ha='center',va='top',transform=axes[col].get_xaxis_transform(),fontsize=8)
    axes[col].axhline(0,color='gray',lw=.8);axes[col].set(title=title,xticks=range(3),xticklabels=labels);axes[col].grid(alpha=.18)
axes[0].set_ylabel(r'$\Delta E/N$');axes[1].set_ylabel(r'$\Delta R$')
# Explicit handles also cover the case that no neural candidate exists.
from matplotlib.lines import Line2D
axes[0].legend(handles=[Line2D([],[],color='#E69F00',marker='o',ls='',label='Calibration'),
    Line2D([],[],color='#0072B2',marker='s',ls='',label='Untouched audit')],frameon=False,fontsize=8)
fig.suptitle('Retained C1: neural increment versus continued carrier filtering')
fig.supxlabel('Neural probe*: post-calibration, pre-audit diagnostic; actual distances and method costs differ.',fontsize=9)
out=project/'figures/exploration';out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/'fig141_retained_carrier_neural_increment.png',dpi=180)
fig.savefig(out/'fig141_retained_carrier_neural_increment.pdf');plt.show()
rows=['| Arm | Constructed | C1 retained | New tangent outside filter span | Max audit chain FS |',
      '| --- | --- | --- | ---: | ---: |']
for name in names:
    proposal=filt if name=='filter' else proposals[name]
    fraction='—' if name=='filter' else f"{proposal['outside_filter_span']['outside_carrier_span_fraction']:.6f}"
    distance_rows=audit['distances'].get(name,{})
    constructed=proposal['trial']['raw_candidate_constructed']
    if diagnostic and name in diagnostic['arms']:
        constructed='Diagnostic only';distance_rows=diagnostic['arms'][name]['distances']
    distance=max((v['fs'] for k,v in distance_rows.items() if k.startswith('chain')),default=float('nan'))
    rows.append(f"| {name} | {constructed} | {result['accepted'].get(name,False)} | {fraction} | {distance:.5g} |")
display(Markdown('\n'.join(rows)))
print('Actual paired comparisons with frozen filter',audit['comparisons_with_filter'])
print('Registration/results',result)
if diagnostic:print('Calibration-adaptive diagnostic; original endpoints unchanged',diagnostic)
if diagnostic and 'network' in diagnostic['arms']:
    from su2_nqs.vmc.carrier_reanchoring import carrier_span_diagnostics
    with np.load(folder/'audit.npz') as z:d={k:z[k] for k in z.files}
    with np.load(folder/'diagnostic/network/audit.npz') as z:probe={k:z[k] for k in z.files}
    finite_span=carrier_span_diagnostics(d['carrier'],36*d['hcarrier'],probe['psi']-d['carrier'],
        probe['hpsi']-36*d['hcarrier'],d['base_products'][:,:2],d['base_products'][:,1:3],d['q'],36)
    print('Saved-only actual finite correction outside filter space',finite_span['outside_carrier_span_fraction'],
          '; independence is not an energy win, no gate/phase refit.')
print('Do not turn a continued-carrier improvement into an SU2+ViT claim; paper notebook unchanged.')
""")]


def finite_schur_population_cells():
    """Saved H² population forces and a physically certified spectral witness."""
    return [markdown(r"""
### Population residual forces, cross-block consensus and a spectral sanity check

Same $6\times6$ triangular system, $J_2/J_1=.125$, strict SU2/full $A_1$,
CNN/attention/free-tensor decoder, analytic carrier and original full-H kernel.
No new head, capacity, state manifold, GPU task or old-trajectory extension.
The prior study amended only E force, leaving a finite-Q residual force.
Here compute the analytic carrier's $H^2C$ without ED and test

$$g_R=\frac{2\operatorname{Re}\langle J_\perp|(H^2-2E H)C\rangle}{NZ}.$$

This is the Hermitian POPULATION variance-force estimator, not generally the
derivative of an incomplete empirical ratio. Actual JVP/FD/candidate energy
still keeps BOTH coherent terms and all nodes. All320 previous FIT/GEO/CAL
records are now explicitly KNOWN FORCE data; none are called untouched.
The192-record geometry is unchanged. Compare previous primary, replacement
of R force on FIT64, balanced320 forces, and consensus E/R forces across
the three known blocks. All original parameters participate and all directions
use the same FIT-unit FS. Independent new calibration can deny/bound an
orientation but never refit it. Neural acceptance remains strict E/R protection.

**Do not confuse a numerical screen with a physical theorem.** The separate
carrier-only control is $[1+\beta(E_\mathrm{anchor}-H)]C$. With
$B=\frac34\sum_b|J_b|$ and $\beta\le .5/\max(B-E_\mathrm{anchor},B)$,
the spectral amplitude is positive and its squared weight decreases with
energy. It cannot raise the true normalized energy, unless stationary; this
does NOT prove variance descent. The two-sided finite-Q slope can disagree.
This control has a separately preregistered5% variance-risk cap and two
paired-SE energy compatibility screen. That is not the neural policy, nor a
two-SE superiority claim. Its success is ordinary imaginary-time/Krylov
physics, NOT SU2+ViT novelty. Never choose/refit after the untouched audit.

The first execution failed only while serializing a duplicate array field,
before new calibration data. Preserve that failure and frozen source, reuse
the sealed320 H² products, fix the writer and register v2; no H² repetition.
Figures139/140 are exploration, not article energy/phase figures.
"""),code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from su2_nqs.frontier.provenance import file_sha256
root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs';folder=project/'results/spin_finite_schur_population_v2'
reg=json.loads((folder/'registration.json').read_text());decision=json.loads((folder/'results.json').read_text())
methods=reg['config']['methods'];arms=('network','free')
runs={(a,m):json.loads((folder/a/m/'results.json').read_text()) for a in arms for m in methods}
fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
labels=['Previous','Pop-R\nFIT64','Pop-E/R\nKNOWN320','Consensus\n(primary)']
for row,arm in enumerate(arms):
    for region,marker,style,color in [('full','o','-','#0072B2'),('half_0','^','--','#D55E00'),('half_1','s',':','#009E73')]:
        records=[runs[arm,m]['calibration_guard']['regions'][region] for m in methods]
        for col,key in enumerate(('energy_slope_per_site','variance_slope_per_site','tangent_fs_speed')):
            axes[row,col].plot(range(4),[r[key] for r in records],marker+style,color=color,label=region,ms=5)
            if region=='full' and col<2:
                axes[row,col].errorbar(range(4),[r[key] for r in records],
                    yerr=[runs[arm,m]['calibration_guard']['chain_jackknife_se'][key] for m in methods],fmt='none',capsize=3,color=color)
    for col,title,y in [(0,'new finite-Q energy slope',r'$d(E/N)/d\alpha$'),
            (1,'new finite-Q residual slope',r'$dR/d\alpha$'),(2,'new physical-step sensitivity','FS speed / same FIT speed')]:
        axes[row,col].set(title=arm+': '+title,ylabel=y,xticks=range(4),xticklabels=labels)
        axes[row,col].grid(alpha=.18);axes[row,col].axhline(1 if col==2 else 0,color='gray',lw=.8)
    axes[row,2].legend(frameon=False,fontsize=8)
fig.suptitle(f"Population-force controls: {decision['accepted_proposals']} neural/free accepted proposals; architecture unchanged")
out=project/'figures/exploration';out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/'fig139_population_force_crossfit.png',dpi=180);fig.savefig(out/'fig139_population_force_crossfit.pdf');plt.show()
for arm in arms:
    geometry=json.loads((folder/arm/'geometry.json').read_text())
    print(arm,'population H² consistency diagnostics',geometry['population_diagnostics'])
    for m in methods:
        r=runs[arm,m];print(m,reg['config']['method_meanings'][m],r['calibration_guard'],r['trial'])
print('Full registration',reg['registration_sha256'],'decisions',decision)
"""),code(r"""
spectral=json.loads((folder/'spectral.json').read_text())
print('Physical spectral certificate (NOT variance or neural advantage)',spectral['energy_certificate'])
print('Finite-Q full/two-forward derivatives',spectral['calibration_guard'])
candidate=spectral['candidate']
if candidate:
    gates=candidate['gates'];names=['Cal full','Cal even','Cal odd']
    if 'audit' in spectral:gates=gates+[spectral['audit']['gate']];names+=['Untouched']
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for col,(key,sekey,title) in enumerate([('delta_energy_vs_carrier','energy_per_site','Paired energy change'),
            ('delta_variance_vs_carrier','variance_per_site','Paired residual change')]):
        axes[col].errorbar(range(len(gates)),[g[key] for g in gates],
            yerr=[g['paired_chain_jackknife_se'][sekey] for g in gates],fmt='o-',capsize=4,color='#0072B2')
        axes[col].axhline(0,color='gray',lw=.8);axes[col].set(title=title,xticks=range(len(gates)),xticklabels=names)
        axes[col].grid(alpha=.18)
    axes[0].set_ylabel(r'$\Delta E/N$');axes[1].set_ylabel(r'$\Delta R$')
    fig.suptitle(f"Carrier-only bounded imaginary-time step; retained={spectral['accepted']} (not a neural win)")
    fig.savefig(out/'fig140_spectral_carrier_sanity.png',dpi=180);fig.savefig(out/'fig140_spectral_carrier_sanity.pdf');plt.show()
    print('Finite proposal and original zero-tolerance point gates',candidate)
    if 'audit' in spectral:print('Untouched paired full-H audit',spectral['audit']['metrics'])
else:print('No finite spectral proposal; inspect saved directional/cost failure without calling it a no-go.')
print('No actual architectural accuracy/resource claim. Paper notebook unchanged; known samples not untouched.')
"""),markdown(r"""
#### Saved-only estimator self-check; no gate override or new samples

Moving H across carrier cross terms gives fixed-coefficient population
control variates: E numerator $\langle C|HC\rangle+2\mathrm{Re}\langle D|HC\rangle
+\langle D|HD\rangle$, and H² numerator $\|HC\|^2+2\mathrm{Re}\langle D|H^2C\rangle+\|HD\|^2$.
These corrections need not vanish on incomplete quadrature. Report them on
the SAME saved audit, keeping the saved candidate phase fixed; do not pick
the favorable estimator or refit/accept rejected candidates post hoc.
The corrected variance is NOT candidate-ray gauge invariant on finite Q:
even a pure phase change of the parent can change its correction. Therefore
skip using it as an unconstrained training loss. Exact support removes this
defect. The table is exploratory, not a newly preregistered energy claim.
"""),code(r"""
from su2_nqs.vmc.anchored_moments import anchored_moments
if (folder/'audit.npz').exists() and (folder/'spectral_audit.npz').exists():
    with np.load(folder/'audit.npz') as z:data={k:z[k] for k in z.files}
    with np.load(folder/'spectral_audit.npz') as z:h2=z['products'][:,2]
    info=json.loads((folder/'audit.json').read_text());names=['carrier',*info['gates']]
    c,hc,q,g=data['psi'][:,0],data['hpsi'][:,0],data['q'],data['groups']
    def changes(index,mask):
        before=anchored_moments(c[mask],hc[mask],h2[mask],c[mask],hc[mask],q[mask],36)
        after=anchored_moments(c[mask],hc[mask],h2[mask],data['psi'][mask,index],data['hpsi'][mask,index],q[mask],36)
        return np.array([after['raw']['energy_per_site']-before['raw']['energy_per_site'],
            after['energy_per_site']-before['energy_per_site'],
            after['raw']['variance_per_site']-before['raw']['variance_per_site'],
            after['variance_per_site']-before['variance_per_site']])
    rows=['| Saved candidate | Raw ΔE/N ± SE | Anchored ΔE/N ± SE | Raw ΔR ± SE | Anchored ΔR ± SE |',
        '| --- | ---: | ---: | ---: | ---: |']
    for i,name in enumerate(names[1:],1):
        point=changes(i,np.ones(len(q),bool));leave=np.stack([changes(i,g!=label) for label in np.unique(g)])
        se=np.sqrt(7/8*np.sum((leave-leave.mean(axis=0))**2,axis=0))
        rows.append('| '+name+' | '+' | '.join(f'{v:+.3g} ± {s:.3g}' for v,s in zip(point,se))+' |')
    display(Markdown('\n'.join(rows)))
    parent=anchored_moments(c,hc,h2,c,hc,q,36)
    phase_only=anchored_moments(c,hc,h2,np.exp(.7j)*c,np.exp(.7j)*hc,q,36)
    print('Pure phase: identical physical parent; raw ΔR',phase_only['raw']['variance_per_site']-parent['raw']['variance_per_site'],
          'but finite-Q anchored ΔR',phase_only['variance_per_site']-parent['variance_per_site'])
    print('Do not train through this phase-sensitive empirical readout or reinterpret old acceptance decisions.')
else:print('No common saved audit/H² products; no extra sampling for this readout.')
""")]


def finite_schur_generalization_cells():
    """Fixed-parent force/geometry controls, saved analysis only."""
    return [markdown(r"""
### Direction and update geometry: out-of-sample generalization, not longer training

Keep the same triangular $6\times6$, $J_2/J_1=.125$, exact SU2, all432 $A_1$
images,216-bond H, centered additive analytic carrier, CNN/attention and free
reduced tensors. Four FULL-PARAMETER directions are fixed before new data:

1. **Fit:** replay the original six-objective full/even/odd energy/residual SR.
2. **Pool:** identical forces; equal parent-norm mass FIT64 + seen GEOMETRY128,
   with ONE global gauge retaining relative block phase/amplitude changes.
3. **Resolve:** same forces/geometry; discard metric eigenmodes below
   $10^{-6}\lambda_{\max}$ as an explicit optimizer regularizer, not an ansatz truncation.
4. **Hermitian (primary):** same resolved geometry; replace ONLY energy forces
   with the population Hermitian $2\mathrm{Re}\langle J_\perp|(H-E)C\rangle/N$
   estimator. Residual forces remain complete finite-Q derivatives.

This is an ordered ladder, not a full factorial. Known geometry records
provide Jacobians/norms, NOT HC, energies, slopes, or fitted force labels.
All directions use the SAME fit-full unit FS speed. New128 calibration
records measure true two-sided full-H JVPs, actual parameter FD and step
reliability. Both coherent terms remain in every finite-Q derivative/test;
the population force is NOT generally the incomplete empirical derivative.
All records, nodes and all24 fit/geometry/calibration chains are retained.

Only negative full/even/odd E/R slopes earn bounded actual-amplitude probes
and one full-H candidate. Maximum measured chain FS=.01; response-limited
underfill is not called equal-distance success. A separate conditional128
untouched audit assesses ALL constructed candidates, with no post-audit
refit or best-of-eight selection. Eight chains do not certify equilibration;
point acceptance is not a two-SE variational win. No trajectory extension,
new head, ED, GPU release, or article figure promotion. Figure138 is exploration.
"""),code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from su2_nqs.frontier.provenance import file_sha256
root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs';folder=project/'results/spin_finite_schur_generalization_v1'
reg=json.loads((folder/'registration.json').read_text());final=json.loads((folder/'results.json').read_text())
methods=reg['config']['methods'];arms=('network','free')
runs={(a,m):json.loads((folder/a/m/'results.json').read_text()) for a in arms for m in methods}
assert all(file_sha256(folder/a/m/'products.npz')==r['product_sha256'] for (a,m),r in runs.items())
fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
labels=['Fit','Pool','Resolve','Hermitian\n(primary)']
for row,arm in enumerate(arms):
    for region,marker,style,offset,color in [('full','o','-',0,'#0072B2'),
            ('half_0','^','--',-.10,'#D55E00'),('half_1','s',':',.10,'#009E73')]:
        records=[runs[arm,m]['calibration_guard']['regions'][region] for m in methods]
        x=np.arange(4)+offset
        for col,key in enumerate(('energy_slope_per_site','variance_slope_per_site','tangent_fs_speed')):
            ax=axes[row,col];ax.plot(x,[r[key] for r in records],marker+style,color=color,label=region,ms=5)
            if region=='full' and col<2:
                ax.errorbar(x,[r[key] for r in records],yerr=[runs[arm,m]['calibration_guard']['chain_jackknife_se'][key]
                    for m in methods],fmt='none',capsize=3,color=color)
            ax.set(xticks=range(4),xticklabels=labels);ax.grid(alpha=.18)
    axes[row,0].set(title=arm+': new finite-Q energy slope',ylabel=r'$d(E/N)/d\alpha$')
    axes[row,1].set(title=arm+': new finite-Q residual slope',ylabel=r'$dR/d\alpha$')
    axes[row,2].set(title=arm+': new physical-step sensitivity',ylabel='FS speed / same fit-full speed')
    for col in (0,1):axes[row,col].axhline(0,color='gray',lw=.8)
    axes[row,2].axhline(1,color='gray',lw=.8)
    axes[row,2].legend(frameon=False,fontsize=8)
fig.suptitle(f"Fixed parent: {final['accepted_proposals']} independently accepted proposals; no longer trajectory")
out=project/'figures/exploration';out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/'fig138_force_geometry_generalization.png',dpi=180)
fig.savefig(out/'fig138_force_geometry_generalization.pdf');plt.show()
table=['| Arm/method | New full dE ± chain SE | New full dR ± chain SE | New full FS speed | Eligible | Finite/calibration/confirmed |',
       '| --- | ---: | ---: | ---: | --- | --- |']
for (arm,method),r in runs.items():
    g=r['calibration_guard'];v=g['regions']['full'];se=g['chain_jackknife_se'];t=r['trial']
    table.append(f"| {arm}/{method} | {v['energy_slope_per_site']:+.5f} ± {se['energy_slope_per_site']:.5f} | "
        f"{v['variance_slope_per_site']:+.5f} ± {se['variance_slope_per_site']:.5f} | {v['tangent_fs_speed']:.3f} | "
        f"{g['eligible']} | {t['raw_candidate_constructed']}/{t['cache_accepted']}/{final['accepted'][arm+'_'+method]} |")
display(Markdown('\n'.join(table)))
for arm in arms:
    geometry=json.loads((folder/arm/'geometry.json').read_text())
    print(arm,'resolved ranks / discarded force fractions',[(m,v['solve'].get('resolved_rank'),
        v['solve'].get('gradient_discarded_fractions')) for m,v in geometry['methods'].items()])
    print('Measured 2-RHS full-H JVP work (not isolated per-method throughput)',
        json.loads((folder/arm/'results.json').read_text())['JVP_work'])
print('Registration',reg['registration_sha256'],'final decisions',final)
if (folder/'audit.json').exists():print('Untouched actual E/R paired audit',json.loads((folder/'audit.json').read_text()))
else:print('No calibration-passing candidate: untouched audit seed unspent.')
print('Population force vs finite-Q derivative distinguished; point sign failures are not a population no-go.')
print('Architecture/carrier unchanged; no undertraining, competitive-energy or PRL claim. Article notebook untouched.')
""")]


def finite_schur_reliability_cells():
    """Saved independent-direction calibration, not H work or more training."""
    return [markdown(r"""
### Independent update calibration before diagnosing undertraining

The parent, exact SU2/full432-image $A_1$, full216-bond H, analytic carrier,
CNN/attention and free-tensor architectures remain unchanged. Preserve the
previous curvature direction with its original orientation. No coefficient
refit, direction flip, new head, ED or GPU job. Known64 FIT records have
already influenced the direction; they cannot independently validate it.

NEW128 CALIBRATION records (8 chains, seed2026100281) measure full-H JVPs
on different configurations. They may bound the step or deny the update,
but do not fit parameters. A separate conditional CONFIRMATION uses another
new seed2026100282; it may accept/reject, never change direction or step.
Thus calibration is **not** called an untouched audit. A point pass alone
is not a statistically established variational win or an equilibration proof.

Before any finite proposal, both energy and centered residual slopes must
be negative on full/even/odd calibration folds. Both coherent energy terms
are kept; the population-Hermitian $2\mathrm{Re}\langle\delta\psi|H\psi\rangle$
estimate is reported separately. Alpha is bounded by half the predicted
linear residual decrease divided by the positive residual-response power.
This is a Gauss--Newton proxy, **not the full Hessian**. Only an eligible
orientation earns13 amplitude probes on the union of FIT and CALIBRATION,
with all16 individual-chain actual FS distances capped at0.01; a deliberately
smaller response-limited step is not an equal-distance comparison. Full-H
finite moments must then pass on both datasets. Nodes are never discarded.

Only independent confirmation of a nonzero update releases the reserved
eight-attempt fixed-endpoint full-gradient trial. If the screen fails, the
trial is withheld, not extended or labeled converged. A noisy sign failure
does not prove global uphill physics or a SU2/attention no-go. It means
undertraining cannot yet be diagnosed using a reliable convergence curve.
Figure137 is optimizer exploration, not a paper energy/phase claim.
"""),code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from su2_nqs.frontier.provenance import file_sha256
root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs';folder=project/'results/spin_finite_schur_reliability_v1'
reg=json.loads((folder/'registration.json').read_text())
decision=json.loads((folder/'results.json').read_text())
release=json.loads((folder/'convergence_release.json').read_text())
runs={a:json.loads((folder/a/'results.json').read_text()) for a in ('network','free')}
assert all(file_sha256(folder/a/'products.npz')==r['product_sha256'] for a,r in runs.items())
colors={'network':'#0072B2','free':'#D55E00'}
fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
for arm,row in runs.items():
    for stage,guard,marker,style,x in [('Fit',row['fit_guard'],'o','-',np.arange(3)),
                                     ('Calibration',row['calibration_guard'],'s','--',np.arange(3)+3)]:
        records=[guard['regions'][k] for k in ('full','half_0','half_1')]
        for ax,key in zip(axes[:2],('energy_slope_per_site','variance_slope_per_site')):
            ax.plot(x,[r[key] for r in records],marker+style,color=colors[arm],ms=5,
                    label=arm+' '+stage)
            ax.errorbar(x[0],records[0][key],yerr=guard['chain_jackknife_se'][key],
                        fmt='none',capsize=3,color=colors[arm])
        axes[2].plot(x,[r['tangent_fs_speed']/row['fit_guard']['regions']['full']['tangent_fs_speed'] for r in records],
                     marker+style,color=colors[arm],ms=5,label=arm+' '+stage)
        if stage=='Calibration':
            axes[0].plot(x[0]+.14,records[0]['one_sided_energy_slope_per_site'],'x',color=colors[arm],ms=7,
                label=arm+' 2forward (not full derivative)')
axes[0].set(title='Same direction: energy derivative',ylabel=r'$d(E/N)/d\alpha$')
axes[1].set(title='Residual-moment direction transfer',ylabel=r'$dR/d\alpha$')
axes[2].set(title='Infinitesimal physical-step amplification',ylabel='FS speed / fit-full speed')
axes[2].axhline(1,color='gray',lw=.8)
for ax in axes:
    if ax is not axes[2]:ax.axhline(0,color='gray',lw=.8)
    ax.axvline(2.5,color='gray',ls=':',lw=.8)
    ax.set(xticks=range(6),xticklabels=['Fit\nfull','Fit\neven','Fit\nodd','New\nfull','New\neven','New\nodd'])
    ax.grid(alpha=.18);ax.legend(frameon=False,fontsize=7)
fig.suptitle(f"Frozen update: {decision['accepted_updates']} confirmed commits; bounded trajectory released={release['released']}")
out=project/'figures/exploration';out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/'fig137_independent_update_calibration.png',dpi=180)
fig.savefig(out/'fig137_independent_update_calibration.pdf');plt.show()
for arm,row in runs.items():
    print(arm,'Fit slopes/curvature bounds',row['fit_guard'])
    print('Independent calibration (not untouched audit)',row['calibration_guard'])
    print('New actual-parameter FD',row['finite_difference_relative_errors'])
    print('Decision',row['trial'],'JVP H seconds',row['JVP_seconds'])
    print('Process',json.loads((folder/f'proposal_{arm}_execution.json').read_text()))
print('Calibration sampling',json.loads((folder/'calibration.json').read_text())['sampling'])
print('Independent confirmation result',decision)
print('Conditional fixed-budget reservation',release)
if (folder/'audit.json').exists():print('Separate untouched confirmation',json.loads((folder/'audit.json').read_text()))
else:print('No eligible independently calibrated candidate: confirmation seed unused.')
print('No undertraining diagnosis from a withheld trajectory, no PRL/SOTA win; paper notebook unchanged.')
"""),markdown(r"""
#### Post-hoc leverage readout: all records retained

On the SAME saved calibration records, inspect the positive contributions
to centered tangent FS norm and residual-response power, alongside carrier
norm. No extra sampling, clipping, tail deletion, refit or gate change.
This diagnoses estimator leverage, **not an order parameter or QSL feature**.
Residual-response power is not the parent's actual variance. Ranked finite
records do not establish their frequency in the true Hilbert-space measure.
"""),code(r"""
from IPython.display import display, Markdown
with np.load(folder/'calibration.npz') as a:
    c,hc,q,groups=a['carrier'],36*a['hcarrier'],a['q'],a['groups']
rows=['| Arm / positive contribution | Largest record | Largest four records | Largest chain |',
      '| --- | ---: | ---: | ---: |']
for arm,row in runs.items():
    with np.load(folder/arm/'products.npz') as a:delta,hdelta=a['delta'],a['hdelta']
    norm=np.sum(abs(c)**2/q);beta=np.sum(c.conj()*delta/q)/norm
    t,ht=delta-beta*c,hdelta-beta*hc
    e=np.sum(c.conj()*hc/q)/norm
    de=np.sum((t.conj()*hc+c.conj()*ht)/q)/norm
    k=ht-e*t-de*c
    powers={'Tangent FS':abs(t)**2/q,'Residual response':abs(k)**2/q,'Carrier norm':abs(c)**2/q}
    guard=row['calibration_guard']['regions']['full']
    assert np.isclose(powers['Tangent FS'].sum()/norm,guard['tangent_fs_speed']**2,rtol=2e-12)
    assert np.isclose(powers['Residual response'].sum()/(36*norm),guard['residual_response_per_site'],rtol=2e-12)
    for name,power in powers.items():
        weights=power/power.sum();ranked=np.sort(weights)[::-1]
        chain=max(weights[groups==g].sum() for g in np.unique(groups))
        rows.append(f'| {arm} / {name} | {ranked[0]:.2%} | {ranked[:4].sum():.2%} | {chain:.2%} |')
display(Markdown('\n'.join(rows)))
print('All128 calibration records remain in every reported slope/power; leverage readout changes no decision.')
""")]


def krylov_carrier_bridge_cells():
    """Saved-only baseline bridge: no coefficient fitting, sampling or H."""
    cells=[markdown(r"""
## Frozen two-Lanczos bridge: stronger parent, not a neural win (2026-10-05)

Triangular 6x6 PBC, J2/J1=0.125. Reuse the genuine sealed G1 two-Lanczos
state, NOT the present degree-one C2 filter. Convert standardized ORDERED
operator coefficients algebraically to P(H)C0. This is exactly the same
physical-amplitude representation; no global fusion-coefficient conversion.
H commutes with SU(2) and the lattice group, so P(H) preserves the analytic
carrier's strict S=0/Gamma-A1. No network architecture/head/update is added.

Both old512-record streams are **KNOWN REANALYSIS**, never a new untouched
audit. Whole-chain errors retain eight correlated records per64 chains, but
do not certify equilibration or unseen tails. The C0-Born proposal cannot
resolve newly opened exact C0 nodes. Development was used to fit K2; its
result is not out-of-training validation. All amplitude/H products and nodes
are kept; no ratio or coefficient clipping. Published energies are context,
not a statistically matched win; costs/variance are not available for those
published states.

Reference lines: [Roth et al., PRB108054410, Table III](https://arxiv.org/pdf/2211.07749):
N=36,J2=1/8, GCNN -0.515386(9), published Gutzwiller+Lanczos -0.512503(3),
ED -0.515564. No ED is repeated here. The much newer
[spatial-attention study](https://arxiv.org/abs/2602.02665) concerns triangular
J2=0 and square J2=.5 scale-up; do not substitute those energies for this
frustrated triangular point or claim that attention alone is new.
"""),code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs'
bridge=json.loads((project/'results/spin_krylov_carrier_bridge_v1/results.json').read_text())
out=project/'figures/exploration';out.mkdir(exist_ok=True)
fig,axes=plt.subplots(1,3,figsize=(12.5,3.8),constrained_layout=True)
streams=('development','audit')
for arm,offset,color in (('C2',-.12,'#D55E00'),('frozen_K2',.12,'#0072B2')):
    for axis,metric in zip(axes[:2],('energy_per_site','variance_per_site')):
        points=[bridge['known_comparison'][s]['arms'][arm][metric] for s in streams]
        axis.errorbar(np.arange(2)+offset,[p['value'] for p in points],
            yerr=[p['chain_jackknife_se'] for p in points],fmt='o-',capsize=3,color=color,
            label=arm.replace('_',' '),ms=4,lw=1)
for a in axes[:2]:
    a.set(xticks=[0,1],xticklabels=['old development','old audit'],xlabel='KNOWN common-chain reanalysis')
    a.legend(frameon=False,fontsize=8)
axes[0].axhline(-.515564,color='black',ls=':',lw=.8,label='published ED')
axes[0].axhline(-.515386,color='#009E73',ls='--',lw=.8,label='published GCNN')
axes[0].axhline(-.512503,color='#CC79A7',ls='-.',lw=.8,label='published VMC+LS')
axes[0].legend(frameon=False,fontsize=7)
axes[0].set(ylabel='Energy / site (J1)',title='Strong reference, not neural gain')
axes[1].set(ylabel='Full-H variance / site (J1 squared)',title='One whole-chain jackknife SE')
for i,r in enumerate(bridge['preflight']):
    counts=r['work']['terminals_per_center_and_power'][0]
    axes[2].plot(range(4),counts,'o-',lw=1,label=f"center{i}: {r['wall_seconds']:.2f}s CPU")
axes[2].set(yscale='log',xticks=range(4),xlabel='Power p in H^p C0',ylabel='Nonzero aggregated terminals',
            title='Actual N=36 query growth')
axes[2].legend(frameon=False,fontsize=8)
fig.suptitle('Frozen carrier bridge; exploratory, no new independent energy validation')
for ext in ('png','pdf'):fig.savefig(out/f'fig149_frozen_krylov_carrier_bridge.{ext}',dpi=180,bbox_inches='tight')
plt.show()
print('New H applications:',bridge['new_full_H_equivalent_applications'],'; samples/updates:',bridge['new_samples'],bridge['parameter_updates'])
print('Power replay max relative defect:',bridge['maximum_power_relative_defect'])
"""),code(r"""
from IPython.display import Markdown, display
lines=['| Known stream | K2-C2 DeltaE/site +/- chain SE | K2-C2 DeltaVar/site +/- chain SE | FS angle | K2 importance ESS fraction |',
       '| --- | ---: | ---: | ---: | ---: |']
for s in streams:
    v=bridge['known_comparison'][s];p=v['paired']['frozen_K2-minus-C2']
    e,r=p['energy_per_site'],p['variance_per_site']
    lines.append(f"| {s} | {e['value']:+.6f} +/- {e['chain_jackknife_se']:.6f} | {r['value']:+.6f} +/- {r['chain_jackknife_se']:.6f} | {v['physical_distance']['fs']:.3f} | {v['arms']['frozen_K2']['weight_ess_fraction']:.3f} |")
display(Markdown('\n'.join(lines)))
print('Both streams suggest a materially stronger baseline, not neural novelty or an independent population result.')
print('Two-center cost does NOT benchmark neural gradients, MCMC equilibration, GPU-hours or large-size scaling.')
print('Next: separately register K2 + Phi(theta)-Phi(theta0), preserving the current strict SU2/full-A1 decoder;')
print('freeze a baseline anchor before independent data, account for all carrier/H3 costs, and test outside the carrier span.')
""")]
    for cell in cells:cell.metadata['experiment_section']='krylov_carrier_bridge'
    return cells


def krylov_residual_training_cells():
    """Saved-only bounded K2/neural/free training and independent audit."""
    cells=[markdown(r"""
## Strong K2 + full neural residual versus free reduced tensors (2026-10-05)

Triangular **6x6 PBC, J2/J1=0.125**. The decoder, q<=4, multiplicity width2,
embedding16/depth2, exact S=0 and all432 Gamma-A1 images are unchanged.
Both arms start at precisely the SAME sealed two-Lanczos carrier:
`psi=K2+s[Phi(theta)-Phi(theta0)]`, including the corresponding full Hpsi.
All6132 neural /1760 free-tensor real parameters are eligible. This is a
full-parameter trial, **not equal-parameter or equal-runtime arms**. The
attention generator still produces a static reduced-tensor bank; this is
not already a full configuration-dependent foundation ViT.

Registration seals at most8 attempts per arm,1800s CPU/8GiB,64 KNOWN optimizer
records, fixed FS cap.01, both energy coherences and the full residual-moment
gradient. Four chains FIT the gradient; four known chains CONTROL it. Full-H
JVP checks the two slopes. A rejected fixed-cache update is not repeated:
the safe checkpoint stays at K2 and the last rejected raw candidate is saved.
The **128 new common audit records** are drawn only AFTER both states are
sealed. No audit-dependent learning, step selection or additional samples.

The plots distinguish rejected raw candidates from the retained K2 endpoints.
Known-data energy/variance point changes have no statistical error bars;
fresh error bars are one whole-chain jackknife SE (16 chains x8 records).
Small positive-mixture audits do not certify equilibration or unseen tails;
finite-Q energies need not obey a variational bound. Rare records and carrier
nodes are retained, never clipped. This is an exploratory optimization result,
not an SU2-specific/ViT-specific variational, phase or PRL claim.
"""),code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs';folder=project/'results/spin_krylov_residual_training_v1'
audit=json.loads((folder/'audit.json').read_text())
training={a:json.loads((folder/a/'training.json').read_text()) for a in ('network','free')}
budget=json.loads((folder/'budget.json').read_text())
out=project/'figures/exploration';out.mkdir(exist_ok=True)
fig,axes=plt.subplots(1,2,figsize=(10.5,4),constrained_layout=True)
for arm,offset,color in (('network',-.045,'#0072B2'),('free',.045,'#D55E00')):
    name=audit['arm_roles'][arm]['evaluated_state']
    pair=audit['metrics']['paired'][name+'-minus-K2']
    row=training[arm]['history'][-1]
    for axis,metric in zip(axes,('energy_per_site','variance_per_site')):
        values=[row['gate']['metrics'][b]['after'][metric]-row['gate']['metrics'][b]['K2'][metric]
                for b in ('fit','control')]
        axis.plot(np.arange(2)+offset,values,'o',color=color,ms=5,
                  label=arm+' raw candidate (rejected)')
        point=pair[metric]
        axis.errorbar(2+offset,point['value'],yerr=point['chain_jackknife_se'],fmt='s',
                      color=color,capsize=4,ms=5)
        axis.axhline(0,color='gray',lw=.8)
        axis.set(xticks=[0,1,2],xticklabels=['known FIT32','known CONTROL32','fresh audit128'])
        axis.ticklabel_format(axis='y',style='sci',scilimits=(-3,3))
axes[0].set(ylabel='Delta energy / site vs K2 (J1)',title='Full parameters, fixed strong carrier')
axes[1].set(ylabel='Delta full-H variance / site (J1 squared)',title='Safe endpoints remain exactly K2')
axes[0].legend(frameon=False,fontsize=8)
fig.suptitle('Candidate changes on three quadratures, NOT a convergence trajectory')
for ext in ('png','pdf'):fig.savefig(out/f'fig150_krylov_residual_training.{ext}',dpi=180,bbox_inches='tight')
plt.show()
print('Accepted / attempted updates:',{a:(t['accepted_updates'],t['attempts']) for a,t in training.items()})
print('Cumulative measured workflow CPU wall seconds:',budget['seconds'])
print('Resolved retained-endpoint energy+variance wins:',audit['resolved_endpoint_energy_variance_wins'])
print('No new ED or GPU-hours; article notebook unchanged.')
"""),code(r"""
from IPython.display import Markdown, display
lines=['| Frozen audit state | E/site +/- chain SE | Var/site +/- chain SE | max chain FS from K2 |',
       '| --- | ---: | ---: | ---: |']
for name,metrics in audit['metrics']['arms'].items():
    e,r=metrics['energy_per_site'],metrics['variance_per_site']
    fs=0. if name=='K2' else max(v['fs'] for v in audit['distances'][name]['per_chain'].values())
    lines.append(f"| {name} | {e['value']:.7f} +/- {e['chain_jackknife_se']:.7f} | {r['value']:.7f} +/- {r['chain_jackknife_se']:.7f} | {fs:.5f} |")
display(Markdown('\n'.join(lines)))
with np.load(folder/'audit.npz',allow_pickle=False) as saved:
    p,h,q=saved['psi'],saved['hpsi'],saved['q']
fig,axes=plt.subplots(1,2,figsize=(10.5,3.8),constrained_layout=True)
for i,name in enumerate(audit['metrics']['arms']):
    norm=np.sum(abs(p[:,i])**2/q);energy=np.sum(p[:,i].conj()*h[:,i]/q)/norm
    contributions=(abs(p[:,i])**2/q,abs(h[:,i]-energy*p[:,i])**2/q)
    for axis,w in zip(axes,contributions):
        ranked=np.sort(w)[::-1]
        axis.plot(np.arange(1,len(q)+1)/len(q),np.cumsum(ranked)/ranked.sum(),label=name,lw=1.2)
for a,title in zip(axes,('Normalization weight concentration','Residual second-moment concentration')):
    a.plot([0,1],[0,1],':',color='gray',lw=.8)
    a.set(xlabel='Fraction of records, descending contribution',ylabel='Cumulative contribution',title=title,
          xlim=(0,1),ylim=(0,1.01))
axes[0].legend(frameon=False,fontsize=7)
for ext in ('png','pdf'):fig.savefig(out/f'fig151_krylov_residual_audit_concentration.{ext}',dpi=180,bbox_inches='tight')
plt.show()
for a,t in training.items():
    row=t['history'][0]
    print(a,'parameters',t['initial_parameter_count'],'resolved sampled FS rank',row['SR']['resolved_rank'],
          'gradient/H-JVP relative discrepancy',row['gradient_H_JVP_replay_relative_defect'])
    print('Known infinitesimal E/R update slopes:',row['SR']['original_predicted_region_slopes'])
    print('Finite candidate gates:',row['gate']['checks'])
print('H3 terminal queries, reference-H and each residual-H cost are all retained in audit.json.')
print('Concentration is a diagnostic of THESE records, not a certificate for unobserved configurations.')
print('Independent or outside-Krylov directions do not guarantee low H-residual curvature or neural gain.')
""")]
    for cell in cells:cell.metadata['experiment_section']='krylov_residual_training'
    return cells


def schur_force_calibration_cells():
    """Read-only fixed-parent force diagnostic; no repeated H or fitting."""
    cells=[markdown(r"""
## One fixed-parent cross-chain calibration; narrowed QSL-candidate target (2026-10-05)

Triangular 6x6, J2/J1=0.125, the frozen corrected chi4 parent, SAME C2,
strict S=0, all432 Gamma-A1 images and216 total-H bonds. No parameter
updates, new chains, architecture changes, ED or GPU. On each of three
predeclared edges, freeze four REAL transported columns built on the original
known192/32 records. Use the first chronological record from each of64 saved
chain IDs; parity folds are balanced within both saved streams. Fit32 chains,
test the other32, reverse once. Both old audits are now KNOWN diagnostics.

Slopes contain both energy coherences and full-H residual derivatives. Error
bars are one held-chain jackknife SE CONDITIONAL on frozen fit coefficients;
they exclude training/column-selection uncertainty. No step or best edge is
selected. Unit-FS slopes distinguish direction from physical-speed transfer.
This is exploration, not an untouched variational/PRL result. The current goal
is a reproducible advantage in the QSL-candidate region; cross-model/coupling
universality is optional, independent validation and strong controls are not.
"""),code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs'
result=json.loads((project/'results/spin_schur_force_calibration_v1/results.json').read_text())
out=project/'figures/exploration';out.mkdir(exist_ok=True)
fig,axes=plt.subplots(1,3,figsize=(12,3.7),constrained_layout=True)
colors=('#0072B2','#D55E00')
for index,row in enumerate(result['history']):
    for fold in row['calibration']['folds']:
        k=fold['training_fold'];color=colors[k];x=index+(-.16 if k==0 else .16)
        label=f'fold{k} -> fold{1-k}' if index==0 else None
        for axis,name in zip(axes[:2],('energy','variance')):
            key=name+'_per_unit_fs'
            a,b=fold['training'][key],fold['held'][key]
            if a is not None and b is not None:
                axis.plot([x-.045,x+.045],[a,b],color=color,lw=.7)
                axis.scatter(x-.045,a,facecolors='none',edgecolors=color,s=24)
                axis.errorbar(x+.045,b,yerr=fold['conditional_chain_jackknife_se'][key],
                              fmt='o',color=color,capsize=2,label=label,ms=4)
        axes[2].errorbar(x,fold['fs_speed_transfer'],
            yerr=fold['conditional_chain_jackknife_se']['fs_speed'],fmt='o',color=color,capsize=2,label=label,ms=4)
for axis in axes:
    axis.set(xticks=range(3),xticklabels=[str(tuple(r['pair'])) for r in result['history']],xlabel='Predeclared internal edge')
    axis.legend(frameon=False,fontsize=8)
for axis in axes[:2]:
    axis.axhline(0,color='black',ls='--',lw=.8)
    axis.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
axes[0].set(ylabel='d(E/N) / unit measured FS',title='Energy: hollow fit, filled held chains')
axes[1].set(ylabel='d(Var/N) / unit measured FS',title='Residual second moment')
axes[2].axhline(1,color='black',ls='--',lw=.8)
axes[2].set(ylabel='Held FS speed / fit FS speed',title='Geometry transfer; same coefficients')
fig.suptitle('Fixed parent; known-cache diagnostic; conditional one-chain-SE bars')
for ext in ('png','pdf'):fig.savefig(out/f'fig148_fixed_parent_cross_chain_force.{ext}',dpi=180,bbox_inches='tight')
plt.show()
print('Point joint-descent edges:',result['useful_point_edges'])
print('Conditional two-SE edges:',result['conditional_two_SE_edges'])
print('New samples / updates / GPU-hours:',result['new_samples'],result['parameter_updates'],result['gpu_hours'])
print('New tangent-H column-centres / CPU seconds:',result['new_H_equivalent_columns'],result['seconds'])
print('STOP:',result['route_decision'])
"""),code(r"""
from IPython.display import Markdown, display
lines=['| Edge / train->held | Held E slope +/- conditional SE | Held Var slope +/- conditional SE | FS transfer | Top4 absolute E / Var contributions |',
       '| --- | ---: | ---: | ---: | ---: |']
for row in result['history']:
    for fold in row['calibration']['folds']:
        h=fold['held'];se=fold['conditional_chain_jackknife_se'];lev=fold['held_contribution_concentration']
        lines.append(f"| {tuple(row['pair'])} / {fold['training_fold']}->{fold['held_fold']} | {h['energy']:+.5g} +/- {se['energy']:.3g} | {h['variance']:+.5g} +/- {se['variance']:.3g} | {fold['fs_speed_transfer']:.3g} | {lev['energy']['top_four_absolute_fraction']:.1%} / {lev['variance']['top_four_absolute_fraction']:.1%} |")
display(Markdown('\n'.join(lines)))
print('No rare record is removed/clipped. These conditional errors are NOT a population significance certificate.')
print('No checkpoint state changed and no finite physical step is chosen, even if a diagnostic is positive.')
""")]
    for cell in cells:cell.metadata['experiment_section']='schur_force_calibration'
    return cells


def schur_transport_cells():
    """Saved-only three-edge retraction-aware tangent canary, no new queries."""
    cells=[markdown(r"""
## Three-edge retraction-aware route-1 canary (2026-10-04)

Same triangular6x6,J2=.125, chi4/q<=4, strict S0/full432-image A1 and C2.
Start at the saved corrected chi4 endpoint. Only three internal edges;
no new head, capacity, architecture, old-audit fitting, ED or GPU job.
Transport the two existing complex joined directions through whole-q
physical truncation BEFORE joint E/R selection. Four REAL columns maximum;
full/half centered probes reject non-smooth directions. Complete H acts on
the actual columns. Known192 amplitudes/32 H records are not held-out data.

At most nine finite-step probes and TWO actual candidate H evaluations per
edge; preserve all original projected-distance and actual E/R safeguards.
Only a safe changed endpoint triggers the preregistered two new streams.
The projection is exact A1, but tree-SVD is not globally projected-FS optimal.
Neither three attempts nor failed transport proves capacity exhaustion.
"""),code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs';folder=project/'results/spin_schur_transport_v1'
result=json.loads((folder/'results.json').read_text())
training=json.loads((folder/'training.json').read_text())
out=project/'figures/exploration';out.mkdir(exist_ok=True)
fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
colors=('#0072B2','#D55E00','#009E73')
for history,color in zip(training['history'],colors):
    pair=str(tuple(history['pair']));slopes=history.get('slopes',{})
    for block,marker in ((0,'o'),(1,'^')):
        if not slopes:continue
        a,b=slopes['unretracted'][block],slopes['same_ray_after_transport'][block]
        axes[0].plot([a['energy'],b['energy']],[a['variance'],b['variance']],'-',color=color,lw=.8)
        axes[0].scatter(a['energy'],a['variance'],marker=marker,facecolors='none',edgecolors=color)
        axes[0].scatter(b['energy'],b['variance'],marker=marker,color=color,label=pair if block==0 else None)
    print(pair,'retained',history['transport']['retained_real_columns'],
          'max probe defect',max(max(c.get('half_step_relative_defect',0),c.get('two_sided_relative_defect',0)) for c in history['transport']['columns']),
          'accepted',history['accepted'])
axes[0].set(xlabel='E/N slope, same original coefficient ray',ylabel='Var/N slope',
            title='Known cache: hollow raw, filled transported')
axes[0].legend(frameon=False,title='Internal edge; circle FIT, triangle control',fontsize=8,
               loc='upper center',bbox_to_anchor=(.5,-.22),ncol=3)
if result['fresh_audit_performed']:
    audit=result['combined_audit']
    for pair,color,label in (('transport4-minus-carrier','#0072B2','relative to C2'),
                             ('transport4-minus-previous4','#D55E00','relative to previous4')):
        d=audit['paired'][pair];e,v=d['energy_per_site'],d['variance_per_site']
        axes[1].errorbar(e['value'],v['value'],xerr=e['chain_jackknife_se'],yerr=v['chain_jackknife_se'],
                         fmt='o',capsize=4,color=color,label=label)
    axes[1].set(xlabel='Paired delta E/N',ylabel='Paired delta Var/N',title='Fresh common audit: one whole-chain SE')
else:
    for history,color in zip(training['history'],colors):
        for candidate in history['actual_candidates']:
            gate=candidate['gate'];axes[1].scatter(gate['delta_energy'],gate['delta_variance'],color=color,
                marker='o' if gate['accepted'] else 'x')
    axes[1].set(xlabel='Actual candidate delta E/N',ylabel='Actual candidate delta Var/N',title='Known cache ONLY; no fresh audit allowed')
for axis in axes:
    axis.axhline(0,color='.5',lw=.8);axis.axvline(0,color='.5',lw=.8)
    axis.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
if result['fresh_audit_performed']:axes[1].legend(frameon=False)
fig.suptitle('Retraction-aware direction diagnostic; not a neural/literature win')
for ext in ('png','pdf'):fig.savefig(out/f'fig146_schur_retraction_transport.{ext}',dpi=180,bbox_inches='tight')
plt.show()
print('Fixed attempt count:',training['attempts'],'accepted:',training['accepted'],'decision:',result['route_decision'])
if result['fresh_audit_performed']:
    for pair in ('transport4-minus-carrier','transport4-minus-previous4'):
        print(pair,{m:(audit['paired'][pair][m]['value'],audit['paired'][pair][m]['chain_jackknife_se'])
                    for m in ('energy_per_site','variance_per_site')})
    print('Finite-Q ratios and rare-tail/equilibration limitations remain; no post-audit refit.')
""")]
    cells.append(code(r"""
# POST-AUDIT diagnosis only: never select/refit a state using these values.
# The coherent chord need not have the original fixed chi4 factorization.
from su2_nqs.vmc.joint_linear_step import linear_moment_slopes
from su2_nqs.vmc.carrier_protection import physical_moments
from su2_nqs.frontier.artifacts import atomic_json,canonical_sha256
with np.load(folder/'known.npz') as a:known={k:a[k] for k in a.files}
with np.load(folder/'parent_known.npz') as a:old={k:a[k] for k in a.files}
with np.load(folder/'current_known.npz') as a:new={k:a[k] for k in a.files}
hi=known['H_indices'];diagnostics={}
for name,sl in (('known FIT',slice(0,16)),('known control',slice(16,None))):
    p,h=old['psi'][hi[sl]],old['hpsi'][sl]
    t,ht=new['psi'][hi[sl]]-p,new['hpsi'][sl]-h
    diagnostics[name]=(p,h,t,ht,known['q'][hi[sl]],None)
if result['fresh_audit_performed']:
    streams=[]
    for i in (0,1):
        with np.load(folder/f'audit{i}.npz') as a:streams.append({k:a[k] for k in a.files})
    merged={k:np.concatenate([a[k] for a in streams]) for k in ('psi','hpsi','q','groups')}
    merged['groups'][128:]+=32
    for name,a in [*(('fresh stream'+str(i),a) for i,a in enumerate(streams)),('fresh combined',merged)]:
        diagnostics[name]=(a['psi'][:,1],a['hpsi'][:,1],a['psi'][:,2]-a['psi'][:,1],
                           a['hpsi'][:,2]-a['hpsi'][:,1],a['q'],a['groups'])
summary={};alphas=np.array([-1.,0.,.25,.5,1.]);fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for name,(p,h,t,ht,q,groups) in diagnostics.items():
    slopes=linear_moment_slopes(p,h,t[:,None],ht[:,None],q,36)
    base_m=physical_moments(p,h,q,36)
    curve=[physical_moments(p+alpha*t,h+alpha*ht,q,36) for alpha in alphas]
    row={m:float(slopes[m][0]) for m in ('energy','variance')}
    if groups is not None:
        leave=[];leave_chords=[]
        for group in np.unique(groups):
            m=groups!=group;s=linear_moment_slopes(p[m],h[m],t[m,None],ht[m,None],q[m],36)
            leave.append([s['energy'][0],s['variance'][0]])
            bm=physical_moments(p[m],h[m],q[m],36)
            lm=[physical_moments(p[m]+alpha*t[m],h[m]+alpha*ht[m],q[m],36) for alpha in alphas]
            leave_chords.append([[v[key]-bm[key] for key in ('energy_per_site','variance_per_site')] for v in lm])
        leave=np.asarray(leave);se=np.sqrt((len(leave)-1)/len(leave)*np.sum((leave-leave.mean(axis=0))**2,axis=0))
        row.update(energy_chain_se=float(se[0]),variance_chain_se=float(se[1]))
        lc=np.asarray(leave_chords);cs=np.sqrt((len(lc)-1)/len(lc)*np.sum((lc-lc.mean(axis=0))**2,axis=0))
        row['chord_chain_se']=cs.tolist()
        for column,(axis,metric) in enumerate(zip(axes,('energy_per_site','variance_per_site'))):
            vals=np.array([v[metric]-base_m[metric] for v in curve])
            line,=axis.plot(alphas,vals,'o-',label=name)
            axis.fill_between(alphas,vals-cs[:,column],vals+cs[:,column],color=line.get_color(),alpha=.12)
    row['chord']=[dict(alpha=float(a),delta_energy=m['energy_per_site']-base_m['energy_per_site'],
                      delta_variance=m['variance_per_site']-base_m['variance_per_site']) for a,m in zip(alphas,curve)]
    summary[name]=row;print(name,{k:v for k,v in row.items() if k!='chord'})
for axis,metric in zip(axes,('E/N','Var/N')):
    axis.axhline(0,color='.5',lw=.8);axis.axvline(0,color='.5',lw=.8)
    axis.set(xlabel='Diagnostic chord alpha; endpoint alpha=1',ylabel='Delta '+metric)
    axis.ticklabel_format(axis='y',style='sci',scilimits=(0,0));axis.legend(frameon=False)
axes[0].set_title('Fresh streams disagree on energy direction')
axes[1].set_title('Variance-slope point estimates are positive')
fig.suptitle('Post-audit chord ONLY; bands=one chain SE; NOT step selection')
for ext in ('png','pdf'):fig.savefig(out/f'fig147_schur_transport_chord.{ext}',dpi=180,bbox_inches='tight')
plt.show()
diagnosis=dict(parent_result_sha256=result.get('result_sha256'),endpoint_seals=result.get('endpoint_seals'),
               moments=summary,post_audit_exploration=True,not_a_registered_gate=True,
               new_H_calls=0,new_samples=0,endpoint_refitted=False,
               chord_may_leave_fixed_chi4_manifold=True)
diagnosis['diagnosis_sha256']=canonical_sha256(diagnosis)
atomic_json(folder/'post_audit_diagnosis.json',diagnosis)
print('No chord alpha is selected or saved as a candidate; original gate/endpoint/results remain sealed.')
"""))
    for cell in cells:cell.metadata['experiment_section']='schur_transport'
    return cells


def schur_metric_capacity_cells():
    """Saved-only metric/actual-step correction, paired against frozen originals."""
    cells=[markdown(r"""
## Route 1: physical-metric retraction and joint E/variance steps (2026-10-04)

The SAME triangular6x6,J2=.125, strict S=0/full432-image A1, q<=4,
chi2/4 free banks and C2 carrier remain. No new neural head or route2/3.
The old192 known amplitude/32 H rows are reused. Exact UNPROJECTED tree Grams
whiten whole-q two-node matrices; this is NOT the full projected/carrier FS
metric. Actual full-A1/carrier amplitudes and complete H still decide updates.

Retain the existing two complex search columns. Their complete H products
allow joint E/R coefficients and finite-step predictions without more tangent
H calls. At most18 amplitude probes and TWO actual candidate H evaluations
per attempt; both actual known-block variances must not increase. Fixed14
attempts/width, all checkpoints retained, no best-iteration/audit selection.
This prospective stricter policy does not change any old result or gate.

Two NEW32-chain x4-record streams compare C2, old2/new2 and old4/new4 on the
SAME256 records. Error bars are one whole-chain jackknife SE. Finite-sample
ratios, equilibrium, rare-tail coverage, and multiplicity of comparisons
remain limitations. Old-vs-new is a compound optimizer correction, NOT an
isolated SVD ablation or a neural/SU2 superiority claim. Training costs and
physical update distances are not matched to the previous experiment.
"""),code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs';folder=project/'results/spin_schur_metric_capacity_v1'
result=json.loads((folder/'results.json').read_text());audit=result['combined_audit']
rows={w:json.loads((folder/f'chi{w}/results.json').read_text()) for w in (2,4)}
out=project/'figures/exploration';out.mkdir(exist_ok=True)
fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for axis,title,pairs in ((axes[0],'Relative to retained C2',('new2-minus-carrier','new4-minus-carrier')),
                         (axes[1],'Relative to OLD endpoint, same width',('new2-minus-old2','new4-minus-old4'))):
    for width,pair,color in zip((2,4),pairs,('#0072B2','#D55E00')):
        d=audit['paired'][pair];e,r=d['energy_per_site'],d['variance_per_site']
        axis.errorbar(e['value'],r['value'],xerr=e['chain_jackknife_se'],yerr=r['chain_jackknife_se'],fmt='o',capsize=4,color=color,label=f'chi={width}')
    axis.axhline(0,color='.5',lw=.8);axis.axvline(0,color='.5',lw=.8)
    axis.set(xlabel='Paired delta E/N',ylabel='Paired delta Var/N',title=title)
    axis.ticklabel_format(axis='both',style='sci',scilimits=(0,0));axis.legend(frameon=False)
fig.suptitle('Fresh common audit: lower-left means both improve (not a PRL win)')
for ext in ('png','pdf'):fig.savefig(out/f'fig144_schur_metric_joint_audit.{ext}',dpi=180,bbox_inches='tight')
plt.show()
for pair in ('new2-minus-carrier','new4-minus-carrier','new2-minus-old2','new4-minus-old4'):
    print(pair,{m:(audit['paired'][pair][m]['value'],audit['paired'][pair][m]['chain_jackknife_se']) for m in ('energy_per_site','variance_per_site')})
print('Accepted attempts:',{w:r['accepted'] for w,r in rows.items()},'Decision:',result['route_decision'])
print('No equilibrium/tail or full-manifold stationarity certificate; see RESULTS.md.')
"""),code(r"""
fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
losses=[];steps=[]
for width,color in ((2,'#0072B2'),(4,'#D55E00')):
    label_used=False
    for history in rows[width]['history']:
        for index,candidate in enumerate(history['actual_candidates']):
            accepted=history['accepted'] and history['selected_candidate']==index
            t=candidate['truncation'];norm=t['physical_norm_squared']
            x=t['comparison_discarded_physical_norm_squared']/norm;y=t['discarded_physical_norm_squared']/norm
            losses.append((width,x,y));step=max(d['fs'] for d in candidate['distances'])
            delta=candidate['physical_truncation_distance']['fs'];steps.append((width,step,delta))
            axes[0].loglog(max(x,1e-16),max(y,1e-16),'o',color=color,ms=4,label=f'chi={width}' if not label_used else None)
            axes[1].scatter(step,delta,edgecolors=color,facecolors=color if accepted else 'none',s=24)
            label_used=True
if losses:
    lower=max(min(min(x,y) for _,x,y in losses),1e-16);upper=max(max(max(x,y) for _,x,y in losses),lower*10)
    axes[0].plot([lower,upper],[lower,upper],':',color='.5',label='Equal physical norm loss')
    axes[0].legend(frameon=False)
if steps:
    upper=max(s for _,s,d in steps)
    axes[1].plot([0,upper],[0,.5*upper],':',color='.5',label='Registered retraction/step cap')
    axes[1].legend(frameon=False)
axes[0].set(xlabel='Coordinate-SVD physical error / norm',ylabel='Whitened-SVD physical error / norm',title='Same target: exact UNPROJECTED tree norm')
axes[1].set(xlabel='Actual largest known-block step FS',ylabel='Retraction FS from intended update',title='Full A1 + carrier; filled = committed')
fig.suptitle('Mechanism check ONLY; known optimization records, not phase evidence')
for ext in ('png','pdf'):fig.savefig(out/f'fig145_schur_metric_retraction_mechanism.{ext}',dpi=180,bbox_inches='tight')
plt.show()
print('Relative Gram-error display floor1e-16 groups numerical roundoff, not a new Schmidt channel or physical rank loss.')
print('Actual full-H evaluations:',{w:sum(len(h['actual_candidates']) for h in r['history']) for w,r in rows.items()})
""")]
    for cell in cells:cell.metadata['experiment_section']='schur_metric_capacity'
    return cells


def schur_capacity_cells():
    """Saved-only two-width result and scoped physical tree Schmidt spectra."""
    cells = [markdown(r"""
## Fixed-q multiplicity capacity: chi=2 versus chi=4 (2026-10-03)

Only the number of copies of each doubled-spin irrep changes. Local Schur,
q<=4, all432 triangular Gamma-A1 images, total216-bond H, C2 carrier and
residual units remain fixed. Both free-tensor models receive14 two-node
Ritz/SVD attempts. Two search columns per joined block are a bounded optimizer,
NOT full-block/global stationarity. Every endpoint is frozen before two NEW
32-chain x4-record audit streams. No neural model/route2/route3 is trained.

Training curves are on KNOWN quadratures, never independent evidence. Audit
differences use common records, full H, node-safe second moments and whole-chain
errors; finite-Q bias, equilibrium and rare-tail coverage remain unproved.
No endpoint selection or extra optimization follows inspection of the audit.

The second figure is an EXACT physical Schmidt spectrum of the **UNPROJECTED
decoder tree only**. Subtree/environment Gram matrices correct nonorthogonal
tensor coordinates. It is not the spectrum of the full A1/carrier+dressing
wavefunction, and is not a QSL/phase diagnosis. Coordinate SVD values used
for retraction are not substituted for physical Schmidt values.
"""), code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import replace
from su2_nqs.frontier.parameter_artifact import load_parameter_artifact
from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.models.schur_two_site import embed_reduced_bank
from su2_nqs.benchmarks.schur_tree_spectrum import internal_schmidt_spectra
from projects.clebsch_tree_nqs.workflows.spin_schur_capacity import parent_setup
root=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'su2_nqs').is_dir())
project=root/'projects/clebsch_tree_nqs';folder=project/'results/spin_schur_capacity_v1'
reg=json.loads((folder/'registration.json').read_text());result=json.loads((folder/'results.json').read_text())
audit=result['combined_audit'];out=project/'figures/exploration';out.mkdir(exist_ok=True)
records={w:json.loads((folder/f'chi{w}/results.json').read_text()) for w in (2,4)}
fig,axes=plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
for w,color in ((2,'#0072B2'),(4,'#D55E00')):
    history=records[w]['history'];current=history[0]['before'][0]['energy_per_site'];values=[current]
    for row in history:
        if row['accepted']:current=row['after'][0]['energy_per_site']
        values.append(current)
    axes[0].plot(np.arange(len(values)),values,'o-',ms=3,color=color,label=f'chi={w}')
axes[0].set(xlabel='Fixed two-node attempt',ylabel='Known FIT E/N',title='Optimization data only')
axes[0].legend(frameon=False)
pairs=('chi2-minus-carrier','chi4-minus-carrier','chi4-minus-chi2')
for axis,metric,label in ((axes[1],'energy_per_site','Paired delta E/N'),(axes[2],'variance_per_site','Paired delta Var/N')):
    for i,pair in enumerate(pairs):
        r=audit['paired'][pair][metric]
        axis.errorbar(i,r['value'],yerr=r['chain_jackknife_se'],fmt='o',capsize=4,color=('#0072B2','#D55E00','#009E73')[i])
    axis.axhline(0,color='.5',lw=.8);axis.set_xticks(range(3),('chi2 - C2','chi4 - C2','chi4 - chi2'),rotation=20)
    axis.set(ylabel=label,title='Untouched common audit');axis.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
fig.suptitle('Fixed-q free-tensor capacity: bounded optimization, not a neural win')
for extension in ('png','pdf'):fig.savefig(out/f'fig142_schur_multiplicity_capacity.{extension}',dpi=180,bbox_inches='tight')
plt.show()
for pair in pairs:
    print(pair,{m:(audit['paired'][pair][m]['value'],audit['paired'][pair][m]['chain_jackknife_se']) for m in ('energy_per_site','variance_per_site')})
print('Serial route decision:',result['route_decision'])
"""), code(r"""
small,initial,*_=parent_setup(reg['config'])
spectra={'reference':internal_schmidt_spectra(initial,small)}
for w in (2,4):
    cfg=replace(small,tree=replace(small.tree,feature_dim=w));template=embed_reduced_bank(initial,small,cfg)
    bank=load_parameter_artifact(folder/f'chi{w}/endpoint.json',template)
    spectra[f'chi{w}']=internal_schmidt_spectra(bank,cfg)
analysis=dict(spectra=spectra,endpoint_seals=result['endpoint_seals'],
    analysis_source_sha256=file_sha256(root/'su2_nqs/benchmarks/schur_tree_spectrum.py'),
    scope='UNPROJECTED decoder tree; not full A1/additive state; no phase claim')
atomic_json(folder/'spectrum_analysis.json',analysis)
fig,axes=plt.subplots(1,2,figsize=(8.8,3.8),constrained_layout=True)
for name,color,style in (('reference','.4',':'),('chi2','#0072B2','--'),('chi4','#D55E00','-')):
    # Fixed cut0: the first eight-spin subtree, not a selected favorable cut.
    block=spectra[name]['cuts']['0']['q_blocks'];qs=np.array(sorted(map(int,block)))
    axes[0].plot(qs,[block[str(q)]['p'] for q in qs],'o'+style,color=color,label=name)
    weights=np.array(block['2']['reduced_weights'])
    axes[1].semilogy(np.arange(1,len(weights)+1),np.maximum(weights,1e-16),'o'+style,color=color,label=name)
axes[0].set(xlabel='Doubled regional spin q=2j',ylabel='Exact p(q)',title='Fixed eight-spin cut, raw tree')
axes[1].set(xlabel='Multiplicity Schmidt index in q=2',ylabel='Reduced Schmidt weight',title='Not coordinate SVD values')
axes[0].legend(frameon=False);axes[1].legend(frameon=False)
fig.suptitle('Unprojected decoder capacity ONLY; not a QSL diagnosis')
for extension in ('png','pdf'):fig.savefig(out/f'fig143_schur_physical_tree_spectrum.{extension}',dpi=180,bbox_inches='tight')
plt.show()
for name,s in spectra.items():
    print(name,'largest cut trace error',max(v['trace_error'] for v in s['cuts'].values()))
""")]
    for cell in cells:cell.metadata['experiment_section']='schur_capacity'
    return cells


def update_exploration_section(cells) -> Path:
    """Execute one tagged section and preserve all other notebook content."""
    tags = {c.metadata['experiment_section'] for c in cells}
    if len(tags) != 1:
        raise ValueError('incremental update needs exactly one section tag')
    tag = tags.pop()
    old = nbf.read(OUTPUT, as_version=4)
    section = nbf.v4.new_notebook(cells=cells, metadata=old.metadata)
    executed = NotebookClient(section, timeout=180, kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}}).execute()
    old.cells = [c for c in old.cells if c.metadata.get("experiment_section") != tag] + executed.cells
    nbf.write(old, OUTPUT)
    return OUTPUT


def update_channel_attention_section() -> Path:
    """Backward-compatible incremental entry point."""
    return update_exploration_section(channel_attention_cells())


def build() -> Path:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3 (nqs)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
        "symmetry_ladder_schema": 1,
    }
    notebook["cells"] = [
        markdown(
            r"""
# SU(2) + lattice symmetry as an algorithmic physics prior

This is the **exploration notebook** for the route started on 2026-09-16.
It retains numerical diagnostics, rejected gates, and mechanism studies so
that the compact paper notebook does not mix publication claims with route
selection evidence.

The falsifiable question is whether, on the same physical-input tree and a
frozen optimization protocol, the cumulative ladder

\[
\text{full Hilbert}\rightarrow M=0\rightarrow S=0
\rightarrow S=0+\text{lattice irrep}
\]

improves energy, variance, and updates to an energy target.  Small exact-support
systems establish the mechanism.  They do not establish a thermodynamic phase
diagram or scalable speedup.

"""
        ),
        code(
            r"""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import numpy as np
import matplotlib.pyplot as plt
from projects.clebsch_tree_nqs import analysis as prl_analysis_module
from projects.clebsch_tree_nqs.workflows import spin as spin_workflow
from projects.clebsch_tree_nqs.workflows import spin_sampled_preflight
from projects.clebsch_tree_nqs.workflows import spin_sampled_runtime
from projects.clebsch_tree_nqs.workflows import spin_observable_validation
from projects.clebsch_tree_nqs.workflows import spin_physics_interpretation
from projects.clebsch_tree_nqs.workflows import spin_sampled_observable_probe
from projects.clebsch_tree_nqs.workflows import spin_sampled_training
from projects.clebsch_tree_nqs.workflows import spin_sampled_training_post_analysis
from projects.clebsch_tree_nqs.workflows import spin_sampled_training_v2
from projects.clebsch_tree_nqs.workflows import spin_sampled_training_v2_selector
from projects.clebsch_tree_nqs.workflows import spin_sampled_training_v2_metric_correction_selector
from projects.clebsch_tree_nqs.workflows import spin_singlet_force_efficiency
from projects.clebsch_tree_nqs.workflows import spin_spatial_cg
from projects.clebsch_tree_nqs.workflows import spin_spatial_cg_capacity
from projects.clebsch_tree_nqs.workflows import spin_spatial_vit
from projects.clebsch_tree_nqs.workflows import spin_su2_route_vit
from projects.clebsch_tree_nqs.workflows import spin_su2_route_vit_causal_scan
from projects.clebsch_tree_nqs.workflows import spin_su2_route_vit_n36_audit
from projects.clebsch_tree_nqs.workflows import spin_su2_route_vit_n36_smoke
from projects.clebsch_tree_nqs.workflows import spin_su2_route_vit_physical_routes
from projects.clebsch_tree_nqs.workflows import spin_triangle_patch_n36_low_spin_resonance
from projects.clebsch_tree_nqs.workflows import spin_two_cluster_virtual_processes
from projects.clebsch_tree_nqs.workflows.common import validate_complete_payload
from su2_nqs.frontier.artifacts import (
    canonical_sha256,
    embedded_trial_manifest,
    trial_payload_sha256,
)
from su2_nqs.frontier.provenance import (
    file_sha256,
    numerical_runtime_identity,
    transitive_local_source_manifest,
)

ROOT = Path.cwd()
PROJECT = ROOT / "projects/clebsch_tree_nqs"
FIGURES = PROJECT / "figures/exploration"
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 220,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "font.size": 10,
})

def load_json(path):
    return json.loads(Path(path).read_text())

def load_complete(path):
    path = Path(path)
    payload = load_json(path)
    if payload.get("status") != "complete":
        raise RuntimeError(f"paper-facing input is not complete: {path}")
    return payload

spin_path = PROJECT / "results/spin_validation_v1/results.json"
spin = load_complete(spin_path)
full = load_complete(PROJECT / "results/spin_true_full_v1/results.json")
brackets = load_complete(PROJECT / "results/spin_parameter_brackets_v1/results.json")
prl_pareto = load_complete(PROJECT / "results/spin_prl_local_pareto_v1/results.json")
prl_u1_long = load_complete(PROJECT / "results/spin_prl_local_u1_long_v1/results.json")
prl_su2_long_path = PROJECT / "results/spin_prl_local_su2_long_v1/results.json"
prl_su2_long = load_complete(prl_su2_long_path) if prl_su2_long_path.exists() else None
prl_analysis = load_json(PROJECT / "results/spin_prl_local_analysis_v1/summary.json")
sampled_preflight = load_complete(
    PROJECT / "results/spin_sampled_estimator_preflight_v1/results.json"
)
singlet_force = load_complete(
    PROJECT / "results/spin_singlet_force_efficiency_v1/results.json"
)
spatial_cg = load_complete(
    PROJECT / "results/spin_spatial_cg_local_pilot_v1/results.json"
)
spatial_cg_capacity = load_complete(
    PROJECT / "results/spin_spatial_cg_capacity_v1/results.json"
)
spatial_vit = load_complete(
    PROJECT / "results/spin_fixed_sz_spatial_vit_local_frontier_v1/results.json"
)
su2_route_vit = load_complete(
    PROJECT / "results/spin_su2_route_vit_local_pilot_v1/results.json"
)
su2_route_vit_long = load_complete(
    PROJECT / "results/spin_su2_route_vit_triangular_long_canary_v1/results.json"
)
su2_route_vit_n36 = load_complete(
    PROJECT / "results/spin_su2_route_vit_n36_smoke_v1/results.json"
)
su2_route_vit_scan = load_complete(
    PROJECT / "results/spin_su2_route_vit_coupling_scan_v1/results.json"
)
su2_route_vit_boundary = load_complete(
    PROJECT / "results/spin_su2_route_vit_boundary_replication_v1/results.json"
)
su2_route_vit_physical_routes = load_complete(
    PROJECT / "results/spin_su2_route_vit_physical_routes_v1/results.json"
)
su2_route_vit_physical_routes_replication = load_complete(
    PROJECT
    / "results/spin_su2_route_vit_physical_routes_replication_v1/results.json"
)
su2_route_vit_n36_audit = load_complete(
    PROJECT / "results/spin_su2_route_vit_n36_audit_longer_v1/results.json"
)
su2_route_vit_n36_row_replication = load_complete(
    PROJECT / "results/spin_su2_route_vit_n36_row_replication_v1/results.json"
)
two_cluster_mechanism = load_complete(
    PROJECT / "results/spin_two_cluster_virtual_processes_v1/results.json"
)
physics_interpretation = load_complete(
    PROJECT / "results/spin_physics_interpretation_v1/results.json"
)
e2_resonance = load_complete(
    PROJECT / "results/spin_e2_resonance_audit_v1/results.json"
)
n36_basin_gate = load_complete(
    PROJECT / "results/spin_su2_route_vit_n36_basin_gate_global_v1/results.json"
)
carrier_four_arm = load_complete(
    PROJECT / "results/spin_symmetry_carrier_four_arm_v1/results.json"
)
n36_carrier_vit = load_complete(
    PROJECT / "results/spin_n36_carrier_vit_matched_work_v7/results.json"
)
n36_multiplicity_spatial = load_complete(
    PROJECT / "results/spin_n36_multiplicity_spatial_attention_v1/results.json"
)
n36_joint_multiplicity_spatial = load_complete(
    PROJECT
    / "results/spin_n36_joint_multiplicity_spatial_attention_v1/results.json"
)
n36_pi_flux = load_complete(
    PROJECT / "results/spin_n36_pi_flux_benchmark_v1/results.json"
)
n36_physical_three_arm = load_complete(
    PROJECT / "results/spin_n36_physical_three_arm_pilot_v1/results.json"
)
n36_coherent_su2_raw = load_complete(
    PROJECT / "results/spin_n36_su2_coherent_carrier_pilot_v1/results.json"
)
n36_coherent_su2_born = load_complete(
    PROJECT / "results/spin_n36_su2_coherent_carrier_pilot_v2/results.json"
)
scalar_operator_frontier = load_complete(
    PROJECT / "results/spin_scalar_operator_frontier_v1/results.json"
)
qsl_three_arm_frontier = load_complete(
    PROJECT / "results/spin_qsl_pi_flux_three_arm_frontier_v1/results.json"
)
n36_operator_transfer = load_complete(
    PROJECT / "results/spin_n36_operator_tangent_transfer_v1/results.json"
)
n36_operator_transfer_l1 = load_complete(
    PROJECT / "results/spin_n36_operator_tangent_transfer_l1_v1/results.json"
)
n36_preregistered_state = load_complete(
    PROJECT / "results/spin_n36_preregistered_channel_state_v1/results.json"
)
n64_operator_capacity = load_complete(
    PROJECT / "results/spin_n64_operator_capacity_v1/results.json"
)
square_n36_operator_capacity = load_complete(
    PROJECT / "results/spin_square_n36_operator_capacity_v1/results.json"
)
g0_cpu_triangular_n36 = load_complete(
    PROJECT / "results/remote_g0_cpu_reference_v1/triangular_n36/results.json"
)
g0_cpu_square_n64 = load_complete(
    PROJECT / "results/remote_g0_cpu_reference_v1/square_n64/results.json"
)
g0_gpu_triangular_n36 = load_complete(
    PROJECT / "results/remote_g0_a100_606800/triangular_n36/results.json"
)
g0_gpu_square_n64_single = load_complete(
    PROJECT
    / "results/remote_g0_n64_single_batch_a100_608378/square_n64/results.json"
)
symmetry_response_development = load_complete(
    PROJECT
    / "results/spin_symmetry_response_operator_development_v1/results.json"
)
square_b2_response_audit = load_complete(
    PROJECT / "results/spin_square_b2_response_audit_v1/results.json"
)
conditioned_response_development = load_complete(
    PROJECT
    / "results/spin_conditioned_response_gate_development_v1/results.json"
)
symmetry_labelled_loop_screen = load_complete(
    PROJECT / "results/spin_symmetry_labelled_loop_screen_v1/results.json"
)
symmetry_loop_coupling_scan = load_complete(
    PROJECT / "results/spin_symmetry_loop_coupling_scan_v1/results.json"
)
triangular_n16_state_manifold = load_complete(
    PROJECT / "results/spin_triangular_n16_state_manifold_v1/results.json"
)
many_cell_matrix_free = load_complete(
    PROJECT / "results/spin_many_cell_matrix_free_v1/results.json"
)
sequential_cell_gate = load_complete(
    PROJECT / "results/spin_sequential_cell_gate_v1/results.json"
)
carrier_cell_scalar_circuit = load_complete(
    PROJECT / "results/spin_carrier_cell_scalar_circuit_v1/results.json"
)
rvb_covering = load_complete(
    PROJECT / "results/spin_rvb_covering_frontier_v1/results.json"
)
rvb_matrix_free = load_complete(
    PROJECT / "results/spin_rvb_covering_matrix_free_gate_v1/results.json"
)
flux_loop_interference = load_complete(
    PROJECT
    / "results/spin_flux_loop_interference_confirmation_v1/results.json"
)
flux_loop_n36_preflight = load_complete(
    PROJECT / "results/spin_n36_flux_loop_preflight_v1/results.json"
)
relative_flux_crossfit = load_complete(
    PROJECT / "results/spin_n36_relative_flux_crossfit_v1/results.json"
)
neural_backflow_pfaffian = load_complete(
    PROJECT / "results/spin_neural_backflow_pfaffian_v1/results.json"
)
residual_neural_backflow = load_complete(
    PROJECT / "results/spin_residual_neural_backflow_v1/results.json"
)
resonating_neural_pfaffian = load_complete(
    PROJECT / "results/spin_resonating_neural_pfaffian_v1/results.json"
)
long_range_loop_rvb = load_complete(
    PROJECT / "results/spin_long_range_loop_rvb_v1/results.json"
)
g0_a100_accounting = load_json(
    PROJECT / "results/remote_g0_a100_campaign_v1/accounting.json"
)
low_spin_local_pilots = {
    geometry: load_complete(
        PROJECT
        / (
            "results/spin_triangle_patch_n36_low_spin_resonance_"
            f"{geometry}_local_pilot_v1/results.json"
        )
    )
    for geometry in ("square", "triangular")
}
observable_validation = load_complete(
    PROJECT / "results/spin_observable_validation_v1/results.json"
)
observable_probe = load_complete(
    PROJECT / "results/spin_sampled_observable_probe_v1/results.json"
)
sampled_training_path = (
    PROJECT / "results/spin_6x6_sampled_training_pilot_v1/results.json"
)
sampled_training = load_complete(sampled_training_path)
sampled_post_analysis_path = (
    PROJECT / "results/spin_6x6_sampled_training_post_analysis_v1/results.json"
)
sampled_post_analysis = load_complete(sampled_post_analysis_path)
literal_lr_selector_config_path = (
    PROJECT / "configs/sampled/spin_6x6_center_only_lr_selector.json"
)
literal_lr_selector_path = (
    PROJECT / "results/spin_6x6_center_only_lr_selector_v1/results.json"
)
metric_lr_selector_config_path = (
    PROJECT
    / "configs/sampled/spin_6x6_center_only_lr_metric_correction_selector.json"
)
metric_lr_selector_path = (
    PROJECT
    / "results/spin_6x6_center_only_lr_metric_correction_selector_v2/results.json"
)
literal_lr_selector = load_complete(literal_lr_selector_path)
metric_lr_selector = load_complete(metric_lr_selector_path)
runtime_smokes = {
    geometry: load_complete(
        PROJECT / f"results/spin_{geometry}_6x6_runtime_smoke_v2/results.json"
    )
    for geometry in ("square", "triangular")
}

singlet_force_without_seal = deepcopy(singlet_force)
singlet_force_seal = singlet_force_without_seal.pop("result_sha256")
if singlet_force_seal != canonical_sha256(singlet_force_without_seal):
    raise RuntimeError("singlet-force benchmark result seal failed")
singlet_force_source = transitive_local_source_manifest(
    (Path(spin_singlet_force_efficiency.__file__).resolve(),),
    root=ROOT,
    package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
)
if singlet_force["source_sha256"] != singlet_force_source:
    raise RuntimeError("singlet-force benchmark source seal is stale")
for payload, workflow_module, label in (
    (spatial_cg, spin_spatial_cg, "Spatial-CG pilot"),
    (spatial_cg_capacity, spin_spatial_cg_capacity, "Spatial-CG capacity"),
):
    payload_without_seal = deepcopy(payload)
    result_seal = payload_without_seal.pop("result_sha256", None)
    if result_seal != canonical_sha256(payload_without_seal):
        raise RuntimeError(f"{label} result seal failed")
    expected_source = transitive_local_source_manifest(
        (Path(workflow_module.__file__).resolve(),),
        root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
    )
    if payload["source_manifest"] != expected_source:
        raise RuntimeError(f"{label} source seal is stale")
    if payload["source_manifest_sha256"] != canonical_sha256(expected_source):
        raise RuntimeError(f"{label} source-manifest hash is broken")
if spatial_cg_capacity.get("pilot_result_sha256") != spatial_cg.get(
    "result_sha256"
):
    raise RuntimeError("Spatial-CG capacity result is not bound to this pilot")
spatial_vit_without_seal = deepcopy(spatial_vit)
spatial_vit_seal = spatial_vit_without_seal.pop("result_sha256", None)
if spatial_vit_seal != canonical_sha256(spatial_vit_without_seal):
    raise RuntimeError("fixed-Sz Spatial-ViT result seal failed")
if spatial_vit["config_sha256"] != canonical_sha256(spatial_vit["config"]):
    raise RuntimeError("fixed-Sz Spatial-ViT embedded-config seal is broken")
spatial_vit_source = transitive_local_source_manifest(
    (Path(spin_spatial_vit.__file__).resolve(),),
    root=ROOT,
    package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
)
if spatial_vit["source_manifest"] != spatial_vit_source:
    raise RuntimeError("fixed-Sz Spatial-ViT source seal is stale")
if spatial_vit["source_manifest_sha256"] != canonical_sha256(
    spatial_vit_source
):
    raise RuntimeError("fixed-Sz Spatial-ViT source-manifest hash is broken")
if spatial_vit["runtime_identity_sha256"] != canonical_sha256(
    spatial_vit["runtime_identity"]
):
    raise RuntimeError("fixed-Sz Spatial-ViT runtime-identity seal is broken")
if spatial_vit.get("hamiltonian_diagonalization_performed") is not False:
    raise RuntimeError("fixed-Sz Spatial-ViT unexpectedly performed an eigensolve")
for payload, workflow_module, label in (
    (su2_route_vit, spin_su2_route_vit, "SU2 Route-ViT pilot"),
    (su2_route_vit_long, spin_su2_route_vit, "SU2 Route-ViT long canary"),
    (su2_route_vit_n36, spin_su2_route_vit_n36_smoke, "SU2 Route-ViT N36 smoke"),
):
    unsealed = deepcopy(payload)
    seal = unsealed.pop("result_sha256", None)
    if seal != canonical_sha256(unsealed):
        raise RuntimeError(f"{label} result seal failed")
    source = transitive_local_source_manifest(
        (Path(workflow_module.__file__).resolve(),),
        root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
    )
    if payload["source_manifest"] != source:
        raise RuntimeError(f"{label} source seal is stale")
    if payload["source_manifest_sha256"] != canonical_sha256(source):
        raise RuntimeError(f"{label} source-manifest hash is broken")
    if payload.get("hamiltonian_diagonalization_performed") is not False:
        raise RuntimeError(f"{label} unexpectedly performed an eigensolve")
for payload, workflow_module, label in (
    (su2_route_vit_scan, spin_su2_route_vit_causal_scan, "Route-ViT coupling scan"),
    (su2_route_vit_boundary, spin_su2_route_vit_causal_scan, "Route-ViT boundary replication"),
    (
        su2_route_vit_physical_routes,
        spin_su2_route_vit_physical_routes,
        "Route-ViT physical-route scan",
    ),
    (
        su2_route_vit_physical_routes_replication,
        spin_su2_route_vit_physical_routes,
        "Route-ViT physical-route replication",
    ),
    (
        su2_route_vit_n36_audit,
        spin_su2_route_vit_n36_audit,
        "Route-ViT N36 sampler audit",
    ),
    (
        su2_route_vit_n36_row_replication,
        spin_su2_route_vit_n36_audit,
        "Route-ViT N36 row-chain replication",
    ),
):
    unsealed = deepcopy(payload)
    seal = unsealed.pop("result_sha256", None)
    if seal != canonical_sha256(unsealed):
        raise RuntimeError(f"{label} result seal failed")
    source = transitive_local_source_manifest(
        (Path(workflow_module.__file__).resolve(),),
        root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
    )
    if payload["source_manifest"] != source:
        raise RuntimeError(f"{label} source seal is stale")
    if payload["source_manifest_sha256"] != canonical_sha256(source):
        raise RuntimeError(f"{label} source-manifest hash is broken")
    if payload.get("hamiltonian_diagonalization_performed") is not False:
        raise RuntimeError(f"{label} unexpectedly performed an eigensolve")
two_cluster_without_seal = deepcopy(two_cluster_mechanism)
two_cluster_seal = two_cluster_without_seal.pop("result_sha256", None)
if two_cluster_seal != canonical_sha256(two_cluster_without_seal):
    raise RuntimeError("two-cluster mechanism result seal failed")
if two_cluster_mechanism["config_sha256"] != canonical_sha256(
    two_cluster_mechanism["config"]
):
    raise RuntimeError("two-cluster mechanism embedded-config seal is broken")
two_cluster_source = transitive_local_source_manifest(
    (Path(spin_two_cluster_virtual_processes.__file__).resolve(),),
    root=ROOT,
    package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
)
if two_cluster_mechanism["source_manifest"] != two_cluster_source:
    raise RuntimeError("two-cluster mechanism source seal is stale")
if two_cluster_mechanism["source_manifest_sha256"] != canonical_sha256(
    two_cluster_source
):
    raise RuntimeError("two-cluster mechanism source-manifest hash is broken")
if two_cluster_mechanism["runtime_identity_sha256"] != canonical_sha256(
    two_cluster_mechanism["runtime_identity"]
):
    raise RuntimeError("two-cluster mechanism runtime-identity seal is broken")
physics_without_seal = deepcopy(physics_interpretation)
physics_seal = physics_without_seal.pop("result_sha256", None)
if physics_seal != canonical_sha256(physics_without_seal):
    raise RuntimeError("spin-physics interpretation result seal failed")
physics_source = transitive_local_source_manifest(
    (Path(spin_physics_interpretation.__file__).resolve(),),
    root=ROOT,
    package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
)
if physics_interpretation["source_manifest"] != physics_source:
    raise RuntimeError("spin-physics interpretation source seal is stale")
if physics_interpretation["source_manifest_sha256"] != canonical_sha256(
    physics_source
):
    raise RuntimeError("spin-physics interpretation source-manifest hash is broken")
if two_cluster_mechanism["claim_boundary"] != (
    spin_two_cluster_virtual_processes.CLAIM_BOUNDARY
):
    raise RuntimeError("two-cluster mechanism claim boundary changed")
if two_cluster_mechanism["hamiltonian_diagonalization"] != {
    "open_cluster_sites": 6,
    "periodic_torus_sites": 12,
    "n36_ed_performed": False,
}:
    raise RuntimeError("two-cluster mechanism ED boundary changed")
expected_orientation_readout = {
    "square": {
        "route_k1_percent": 92.26,
        "direction_shared": (3, 5, 5),
        "physical_pair_shared": (4, 5, 5),
    },
    "triangular": {
        "route_k1_percent": 92.50,
        "direction_shared": (4, 5, 5),
        "physical_pair_shared": (4, 5, 5),
    },
}
orientation_threshold_keys = (
    "90_percent_second_order_energy",
    "99_percent_second_order_energy",
    "99p9_percent_second_order_energy",
)
for case in two_cluster_mechanism["cases"]:
    geometry = case["geometry"]
    support = case.get("support_orientation", {})
    if (
        support.get("lattice_shape") != [6, 6]
        or support.get("cluster_shape") != [2, 3]
        or support.get("translation_group_order") != 36
        or support.get("distinct_tiling_count") != 6
    ):
        raise RuntimeError(f"{geometry} support-orientation scope changed")
    direction_union = support["direction_resolved"]["union"]
    pair_union = support["physical_pair_total"]["union"]
    if (
        direction_union["route_k_union_ranks"][0] != 9
        or direction_union["union_rank"] != 9
        or pair_union["union_rank"] != 9
    ):
        raise RuntimeError(f"{geometry} support-orientation rank changed")
    expected = expected_orientation_readout[geometry]
    route_percent = 100.0 * direction_union[
        "route_conditioned_second_order_energy_fractions"
    ][0]
    direction_thresholds = tuple(
        direction_union["minimum_shared_carriers"][key]
        for key in orientation_threshold_keys
    )
    pair_thresholds = tuple(
        pair_union["minimum_shared_carriers"][key]
        for key in orientation_threshold_keys
    )
    if (
        abs(route_percent - expected["route_k1_percent"]) > 0.005
        or direction_thresholds != expected["direction_shared"]
        or pair_thresholds != expected["physical_pair_shared"]
    ):
        raise RuntimeError(f"{geometry} support-orientation readout changed")
low_spin_local_pilot_source = transitive_local_source_manifest(
    (Path(spin_triangle_patch_n36_low_spin_resonance.__file__).resolve(),),
    root=ROOT,
    package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
)
expected_low_spin_cases = {
    "square": "square_6x6_j2_0p5",
    "triangular": "triangular_6x6_j2_0p125",
}
for geometry, payload in low_spin_local_pilots.items():
    body = deepcopy(payload)
    result_seal = body.pop("result_sha256", None)
    if result_seal != canonical_sha256(body):
        raise RuntimeError(f"{geometry} low-spin local-pilot result seal failed")
    if payload["config_sha256"] != canonical_sha256(payload["config"]):
        raise RuntimeError(f"{geometry} low-spin embedded-config seal is broken")
    if payload["execution_spec_sha256"] != canonical_sha256(
        payload["execution_spec"]
    ):
        raise RuntimeError(f"{geometry} low-spin execution-spec seal is broken")
    if payload["source_manifest"] != low_spin_local_pilot_source:
        raise RuntimeError(f"{geometry} low-spin local pilot is stale")
    if payload["source_manifest_sha256"] != canonical_sha256(
        low_spin_local_pilot_source
    ):
        raise RuntimeError(f"{geometry} low-spin source-manifest seal is broken")
    if payload["runtime_identity_sha256"] != canonical_sha256(
        payload["runtime_identity"]
    ):
        raise RuntimeError(f"{geometry} low-spin runtime-identity seal is broken")
    if (
        payload.get("task")
        != spin_triangle_patch_n36_low_spin_resonance.TASK
        or payload.get("claim_boundary")
        != spin_triangle_patch_n36_low_spin_resonance.CLAIM_BOUNDARY
        or payload.get("execution_tier") != "local_pilot"
        or payload.get("local_pilot_mode") is not True
        or payload.get("smoke_mode") is not False
    ):
        raise RuntimeError(f"{geometry} low-spin pilot identity changed")
    if (
        payload.get("hamiltonian_diagonalization_on_n36_performed") is not False
        or payload.get("local_six_site_diagonalization_only") is not True
    ):
        raise RuntimeError(f"{geometry} low-spin ED boundary changed")
    if payload.get("selected_case_id") != expected_low_spin_cases[geometry]:
        raise RuntimeError(f"{geometry} low-spin selected case changed")
    if len(payload.get("cases", ())) != 1:
        raise RuntimeError(f"{geometry} low-spin pilot must contain one case")
    case = payload["cases"][0]
    if case.get("geometry") != geometry or case.get("case_id") != expected_low_spin_cases[geometry]:
        raise RuntimeError(f"{geometry} low-spin case identity changed")
    if [arm.get("arm_id") for arm in case.get("arms", ())] != [
        "A0_frozen",
        "A1_q0",
        "A2_q2_increment",
        "A12_q0_q2",
    ]:
        raise RuntimeError(f"{geometry} low-spin arm ordering changed")
for payload, workflow, label in [
    (spin, spin_workflow.WORKFLOW, "spin validation"),
    (full, spin_workflow.WORKFLOW, "true-full calibration"),
    (brackets, spin_workflow.WORKFLOW, "spin parameter brackets"),
    (prl_pareto, spin_workflow.WORKFLOW, "spin PRL parameter Pareto"),
    (prl_u1_long, spin_workflow.WORKFLOW, "spin PRL horizon extension"),
]:
    validate_complete_payload(payload, workflow)
if prl_su2_long is not None:
    validate_complete_payload(prl_su2_long, spin_workflow.WORKFLOW)

# The exact-Born estimator preflight has a deliberately narrower workflow than
# the exact-optimization campaigns, so verify both its current source seal and
# its embedded trial seal directly.
sampled_source = spin_sampled_preflight._source_manifest()
if sampled_preflight["source_sha256"] != sampled_source:
    raise RuntimeError("sampled-estimator result is stale relative to current source")
if sampled_preflight["source_manifest_sha256"] != canonical_sha256(sampled_source):
    raise RuntimeError("sampled-estimator source-manifest hash is broken")
if sampled_preflight["trial_manifest_sha256"] != canonical_sha256(
    embedded_trial_manifest(sampled_preflight)
):
    raise RuntimeError("sampled-estimator embedded-trial seal is broken")
observable_source = spin_observable_validation._source_manifest()
if observable_validation["config_sha256"] != canonical_sha256(
    observable_validation["config"]
):
    raise RuntimeError("spin-observable embedded-config seal is broken")
if observable_validation["source_sha256"] != observable_source:
    raise RuntimeError("spin-observable result is stale relative to current source")
if observable_validation["source_manifest_sha256"] != canonical_sha256(
    observable_source
):
    raise RuntimeError("spin-observable source-manifest seal is broken")
if observable_validation["runtime_identity_sha256"] != canonical_sha256(
    observable_validation["runtime_identity"]
):
    raise RuntimeError("spin-observable runtime-identity seal is broken")
if observable_validation["trial_manifest_sha256"] != canonical_sha256(
    embedded_trial_manifest(observable_validation)
):
    raise RuntimeError("spin-observable embedded-trial seal is broken")
runtime_source = spin_sampled_runtime._source_manifest()
for geometry, payload in runtime_smokes.items():
    if payload["config_sha256"] != canonical_sha256(payload["config"]):
        raise RuntimeError(f"{geometry} 6x6 runtime embedded-config seal is broken")
    if payload["source_sha256"] != runtime_source:
        raise RuntimeError(f"{geometry} 6x6 runtime result is stale relative to current source")
    if payload["source_manifest_sha256"] != canonical_sha256(runtime_source):
        raise RuntimeError(f"{geometry} 6x6 runtime source-manifest hash is broken")
    if payload["runtime_identity_sha256"] != canonical_sha256(payload["runtime_identity"]):
        raise RuntimeError(f"{geometry} 6x6 runtime-identity seal is broken")
    if payload["trial_manifest_sha256"] != canonical_sha256(embedded_trial_manifest(payload)):
        raise RuntimeError(f"{geometry} 6x6 runtime embedded-trial seal is broken")
    result_directory = PROJECT / f"results/spin_{geometry}_6x6_runtime_smoke_v2"
    runtime_case = payload["cases"][0]
    for trial in runtime_case["trials"]:
        checkpoint_path = result_directory / trial["checkpoint"]["path"]
        checkpoint = load_json(checkpoint_path)
        checkpoint_seal = checkpoint.pop("checkpoint_sha256", None)
        if checkpoint_seal != canonical_sha256(checkpoint):
            raise RuntimeError(f"{geometry} checkpoint seal is broken: {checkpoint_path.name}")
        if checkpoint_seal != trial["checkpoint"]["checkpoint_sha256"]:
            raise RuntimeError(f"{geometry} trial/checkpoint seal mismatch")
        expected_identity = {
            "config_sha256": payload["config_sha256"],
            "source_manifest_sha256": payload["source_manifest_sha256"],
            "runtime_identity_sha256": payload["runtime_identity_sha256"],
            "case_id": runtime_case["case_id"],
            "arm_id": trial["arm_id"],
            "backbone": trial["backbone"],
            "initial_params_sha256": trial["initial_params_sha256"],
        }
        if checkpoint["identity"] != expected_identity:
            raise RuntimeError(f"{geometry} checkpoint identity does not match its trial")
        if trial["sampler"]["resume_count"] != 0:
            raise RuntimeError(
                f"{geometry} resumed runtime smoke is ineligible for the timing panel"
            )
probe_source = spin_sampled_observable_probe._source_manifest()
if observable_probe["config_sha256"] != canonical_sha256(observable_probe["config"]):
    raise RuntimeError("sampled-observable embedded-config seal is broken")
if observable_probe["source_sha256"] != probe_source:
    raise RuntimeError("sampled-observable probe is stale relative to current source")
if observable_probe["source_manifest_sha256"] != canonical_sha256(probe_source):
    raise RuntimeError("sampled-observable source-manifest seal is broken")
if observable_probe["runtime_identity_sha256"] != canonical_sha256(
    observable_probe["runtime_identity"]
):
    raise RuntimeError("sampled-observable runtime-identity seal is broken")
if observable_probe["trial_manifest_sha256"] != canonical_sha256(
    embedded_trial_manifest(observable_probe)
):
    raise RuntimeError("sampled-observable embedded-trial seal is broken")
probe_directory = PROJECT / "results/spin_sampled_observable_probe_v1"
for probe_case in observable_probe["cases"]:
    geometry = probe_case["geometry"]
    upstream = runtime_smokes[geometry]
    if probe_case["upstream_config_sha256"] != upstream["config_sha256"]:
        raise RuntimeError(f"{geometry} sampled-observable upstream config mismatch")
    if probe_case["upstream_source_manifest_sha256"] != upstream["source_manifest_sha256"]:
        raise RuntimeError(f"{geometry} sampled-observable upstream source mismatch")
    if probe_case["upstream_runtime_identity_sha256"] != upstream["runtime_identity_sha256"]:
        raise RuntimeError(f"{geometry} sampled-observable upstream runtime mismatch")
    if probe_case["upstream_trial_manifest_sha256"] != upstream["trial_manifest_sha256"]:
        raise RuntimeError(f"{geometry} sampled-observable upstream trial mismatch")
    for trial in probe_case["trials"]:
        checkpoint_path = probe_directory / trial["checkpoint"]["path"]
        checkpoint = load_json(checkpoint_path)
        checkpoint_seal = checkpoint.pop("checkpoint_sha256", None)
        if checkpoint_seal != canonical_sha256(checkpoint):
            raise RuntimeError(f"sampled-observable checkpoint seal is broken: {checkpoint_path.name}")
        if checkpoint_seal != trial["checkpoint"]["checkpoint_sha256"]:
            raise RuntimeError("sampled-observable trial/checkpoint seal mismatch")
        expected_identity = {
            "config_sha256": observable_probe["config_sha256"],
            "source_manifest_sha256": observable_probe["source_manifest_sha256"],
            "runtime_identity_sha256": observable_probe["runtime_identity_sha256"],
            "case_id": probe_case["case_id"],
            "arm_id": trial["arm_id"],
            "backbone": trial["backbone"],
            "initial_params_sha256": trial["initial_params_sha256"],
            "upstream_trial_payload_sha256": trial["upstream"]["trial_payload_sha256"],
            "upstream_checkpoint_sha256": trial["upstream"]["checkpoint_sha256"],
        }
        if checkpoint["identity"] != expected_identity:
            raise RuntimeError("sampled-observable checkpoint identity mismatch")

# The first parameter-updating 6x6 pilot and its deterministic replay are
# paper-facing only if the current code, numerical runtime, input files,
# embedded trial records, and every referenced checkpoint generation remain
# exactly the objects that produced the sealed analysis.
training_source = spin_sampled_training._source_manifest()
if sampled_training.get("task") != "physical_sampled_training":
    raise RuntimeError("6x6 pilot has the wrong task identity")
if sampled_training["config_sha256"] != canonical_sha256(sampled_training["config"]):
    raise RuntimeError("6x6 pilot embedded-config seal is broken")
if sampled_training["source_sha256"] != training_source:
    raise RuntimeError("6x6 pilot is stale relative to current training source")
if sampled_training["source_manifest_sha256"] != canonical_sha256(training_source):
    raise RuntimeError("6x6 pilot source-manifest seal is broken")
if sampled_training["runtime_identity_sha256"] != canonical_sha256(
    sampled_training["runtime_identity"]
):
    raise RuntimeError("6x6 pilot runtime-identity seal is broken")
if sampled_training["trial_manifest_sha256"] != canonical_sha256(
    embedded_trial_manifest(sampled_training)
):
    raise RuntimeError("6x6 pilot embedded-trial seal is broken")
if sampled_training["route_boundary"].get("direct_fusion_path_used") is not False:
    raise RuntimeError("6x6 pilot violates the active route boundary")

post_source = spin_sampled_training_post_analysis._source_manifest()
if sampled_post_analysis.get("task") != "physical_sampled_training_post_analysis":
    raise RuntimeError("6x6 replay analysis has the wrong task identity")
if sampled_post_analysis["config_sha256"] != canonical_sha256(
    sampled_post_analysis["config"]
):
    raise RuntimeError("6x6 replay embedded-config seal is broken")
if sampled_post_analysis["source_sha256"] != post_source:
    raise RuntimeError("6x6 replay analysis is stale relative to current source")
if sampled_post_analysis["source_manifest_sha256"] != canonical_sha256(post_source):
    raise RuntimeError("6x6 replay source-manifest seal is broken")
if sampled_post_analysis["runtime_identity_sha256"] != canonical_sha256(
    sampled_post_analysis["runtime_identity"]
):
    raise RuntimeError("6x6 replay runtime-identity seal is broken")
if sampled_post_analysis["input_results_manifest_sha256"] != canonical_sha256(
    sampled_post_analysis["input_results_manifest"]
):
    raise RuntimeError("6x6 replay input-manifest seal is broken")
expected_post_inputs = {
    "sampled_training_pilot": file_sha256(sampled_training_path),
    "runtime_square": file_sha256(
        PROJECT / "results/spin_square_6x6_runtime_smoke_v2/results.json"
    ),
    "runtime_triangular": file_sha256(
        PROJECT / "results/spin_triangular_6x6_runtime_smoke_v2/results.json"
    ),
}
if sampled_post_analysis["input_results_manifest"] != expected_post_inputs:
    raise RuntimeError("6x6 replay does not identify the loaded immutable inputs")
if sampled_post_analysis["trial_manifest_sha256"] != canonical_sha256(
    embedded_trial_manifest(sampled_post_analysis)
):
    raise RuntimeError("6x6 replay embedded-trial seal is broken")
if not sampled_post_analysis["all_post_analysis_gates_passed"]:
    raise RuntimeError("6x6 replay analysis did not pass all audit gates")
if sampled_post_analysis["route_boundary"] != {
    "direct_fusion_path_used": False,
    "parameter_update_performed": False,
    "training_chain_mutated": False,
    "large_ED_performed": False,
    "sealed_checkpoint_replay_only": True,
    "energy_or_physics_win_claimed": False,
}:
    raise RuntimeError("6x6 replay route boundary changed")

pilot_cases = {case["case_id"]: case for case in sampled_training["cases"]}
pilot_directory = sampled_training_path.parent
for post_case in sampled_post_analysis["cases"]:
    pilot_case = pilot_cases[post_case["case_id"]]
    pilot_trials = {trial["arm_id"]: trial for trial in pilot_case["trials"]}
    for post_trial in post_case["trials"]:
        if post_trial["trial_payload_sha256"] != trial_payload_sha256(post_trial):
            raise RuntimeError("6x6 replay trial payload seal is broken")
        if not post_trial["all_post_analysis_gates_passed"]:
            raise RuntimeError("6x6 replay trial failed an audit gate")
        pilot_trial = pilot_trials[post_trial["arm_id"]]
        if pilot_trial["trial_payload_sha256"] != trial_payload_sha256(pilot_trial):
            raise RuntimeError("6x6 pilot trial payload seal is broken")
        if post_trial["initial_params_sha256"] != pilot_trial["initial_params_sha256"]:
            raise RuntimeError("6x6 replay/pilot initial-parameter lineage mismatch")
        if post_trial["checkpoint_run_identity_sha256"] != pilot_trial["checkpoint"][
            "run_identity_sha256"
        ]:
            raise RuntimeError("6x6 replay/pilot checkpoint identity mismatch")
        series_directory = pilot_directory / pilot_trial["checkpoint"]["series_directory"]
        for generation in post_trial["generation_manifest"]:
            generation_path = series_directory / generation["file"]
            if file_sha256(generation_path) != generation["file_sha256"]:
                raise RuntimeError(f"6x6 replay generation seal is broken: {generation_path.name}")
            if generation["file_sha256"] != generation["registered_file_sha256"]:
                raise RuntimeError("6x6 replay generation/registry hash mismatch")

# The 50-update center-only LR scan is loaded through the same fail-closed
# auditors that produced its selectors.  This validates all three embedded
# configs, current source manifests, runtime/common profiles, trial/array
# seals, fixed measurement streams, and the selected step-50 checkpoint
# generation plus params/Adam/chains/key hashes.  The literal accepted-move
# selector remains the immutable primary record; the conditional-acceptance
# selector is loaded only as a separately sealed semantic sensitivity.
literal_lr_spec = load_json(literal_lr_selector_config_path)
spin_sampled_training_v2_selector._validate_spec(literal_lr_spec)
metric_lr_spec = load_json(metric_lr_selector_config_path)
spin_sampled_training_v2_metric_correction_selector._validate_spec(
    metric_lr_spec
)
if literal_lr_selector["config"] != literal_lr_spec:
    raise RuntimeError("literal LR selector embedded config is stale")
if metric_lr_selector["config"] != metric_lr_spec:
    raise RuntimeError("metric-correction LR selector embedded config is stale")
metric_selector_source = (
    spin_sampled_training_v2_metric_correction_selector._source_manifest()
)
if metric_lr_selector["source_sha256"] != metric_selector_source:
    raise RuntimeError("metric-correction LR selector is stale relative to source")
if metric_lr_selector["source_manifest_sha256"] != canonical_sha256(
    metric_selector_source
):
    raise RuntimeError("metric-correction LR selector source seal is broken")
if metric_lr_selector["config_sha256"] != canonical_sha256(metric_lr_spec):
    raise RuntimeError("metric-correction LR selector config seal is broken")
if metric_lr_selector["result_payload_sha256"] != (
    spin_sampled_training_v2_metric_correction_selector._result_payload_sha256(
        metric_lr_selector
    )
):
    raise RuntimeError("metric-correction LR selector payload seal is broken")

literal_lr_bundle = (
    spin_sampled_training_v2_metric_correction_selector._load_literal_primary(
        metric_lr_spec
    )
)
current_lr_scan_source = spin_sampled_training_v2._source_manifest()
lr_scan_candidates = [
    spin_sampled_training_v2_selector._audit_candidate(
        dict(entry),
        selector_spec=literal_lr_spec,
        current_scan_source=current_lr_scan_source,
    )
    for entry in metric_lr_spec["input_candidates"]
]
literal_primary_binding = (
    spin_sampled_training_v2_metric_correction_selector._audit_literal_decision(
        literal_lr_bundle, lr_scan_candidates
    )
)
if metric_lr_selector["literal_primary_binding"] != literal_primary_binding:
    raise RuntimeError("metric correction does not bind the audited literal selector")

expected_lr_input_manifest = [
    {
        "candidate_id": candidate["candidate_id"],
        "learning_rate": candidate["learning_rate"],
        "config_path": spin_sampled_training_v2_metric_correction_selector._portable_path(
            candidate["config_path"]
        ),
        "config_file_sha256": candidate["config_file_sha256"],
        "config_sha256": candidate["config_sha256"],
        "result_path": spin_sampled_training_v2_metric_correction_selector._portable_path(
            candidate["result_path"]
        ),
        "result_file_sha256": candidate["result_file_sha256"],
        "source_manifest_sha256": candidate["source_manifest_sha256"],
        "runtime_identity_sha256": candidate["runtime_identity_sha256"],
        "common_scan_profile_sha256": candidate["common_scan_profile_sha256"],
    }
    for candidate in sorted(
        lr_scan_candidates, key=lambda item: float(item["learning_rate"])
    )
]
if metric_lr_selector["input_manifest"] != expected_lr_input_manifest:
    raise RuntimeError("metric-correction LR selector input manifest is broken")
if literal_lr_selector["input_manifest"] != expected_lr_input_manifest:
    raise RuntimeError("literal LR selector input manifest is broken")

corrected_lr_candidates = deepcopy(lr_scan_candidates)
for candidate in corrected_lr_candidates:
    spin_sampled_training_v2_metric_correction_selector._apply_corrected_metric(
        candidate, metric_lr_spec
    )
spin_sampled_training_v2_selector._audit_cross_candidate_matching(
    corrected_lr_candidates
)
expected_metric_geometry_results = (
    spin_sampled_training_v2_metric_correction_selector._corrected_geometry_results(
        corrected_lr_candidates, metric_lr_spec
    )
)
if metric_lr_selector["geometry_results"] != expected_metric_geometry_results:
    raise RuntimeError("metric-correction decisions or checkpoint bindings changed")
metric_lr_decisions = {
    row["geometry"]: row["decision"] for row in metric_lr_selector["geometry_results"]
}
if {
    geometry: decision["selected_candidate_id"]
    for geometry, decision in metric_lr_decisions.items()
} != {"square": "lr_2e-3", "triangular": "lr_1e-3"}:
    raise RuntimeError("metric-correction LR selections are not the registered result")
if any(
    action != "no_adam_promotion"
    for action in literal_primary_binding["actions_by_case"].values()
):
    raise RuntimeError("literal LR primary decision is not no-Adam-promotion")
lr_scan_payloads = {
    candidate["candidate_id"]: candidate["payload"]
    for candidate in lr_scan_candidates
}
if not prl_analysis["validation"]["passed"]:
    raise RuntimeError("strict PRL Pareto analysis did not pass validation")
summary_seal = prl_analysis.get("summary_payload_sha256")
summary_body = {
    key: value for key, value in prl_analysis.items()
    if key != "summary_payload_sha256"
}
if summary_seal != canonical_sha256(summary_body):
    raise RuntimeError("strict PRL summary payload seal is broken")
analysis_source = prl_analysis["analysis_provenance"]["source_sha256"]
if prl_analysis["analysis_provenance"]["source_manifest_sha256"] != canonical_sha256(analysis_source):
    raise RuntimeError("strict PRL analysis-source manifest seal is broken")
expected_analysis_source = {
    "projects/clebsch_tree_nqs/analysis.py": hashlib.sha256(
        Path(prl_analysis_module.__file__).read_bytes()
    ).hexdigest()
}
if analysis_source != expected_analysis_source:
    raise RuntimeError("strict PRL analysis is stale relative to current source")
if prl_analysis["inputs"]["base_trial_manifest_sha256"] != prl_pareto["trial_manifest_sha256"]:
    raise RuntimeError("strict PRL analysis does not identify the loaded base campaign")
if prl_analysis["inputs"]["long_trial_manifest_sha256"] != prl_u1_long["trial_manifest_sha256"]:
    raise RuntimeError("strict PRL analysis does not identify the loaded U1-long campaign")
analysis_su2_long_seal = prl_analysis["inputs"].get("su2_long_trial_manifest_sha256")
if prl_su2_long is None and analysis_su2_long_seal is not None:
    raise RuntimeError("strict PRL analysis identifies a missing SU2-long campaign")
if prl_su2_long is not None and analysis_su2_long_seal != prl_su2_long["trial_manifest_sha256"]:
    raise RuntimeError("strict PRL analysis does not identify the loaded SU2-long campaign")
vn = load_json(PROJECT / "references/vieijra_nys_prb104_045123.json")
tri_refs = load_json(PROJECT / "references/triangular_nqs_targets.json")
transformer_refs = load_json(PROJECT / "references/spin_transformer_targets.json")
route_vit_coupling_refs = load_json(
    PROJECT / "references/spin_j1j2_coupling_scan.json"
)

print("spin source:", spin_path.name)
print("Spin production route-boundary flags:", spin["route_boundary"])
validated_trial_counts = [
    section["trial_count"]
    for name in ("base", "long", "su2_long")
    if isinstance((section := prl_analysis["validation"].get(name)), dict)
]
print("PRL local campaign trial counts:", validated_trial_counts)
print("Spatial-CG pilot/capacity:", spatial_cg["status"], spatial_cg_capacity["status"])

def trial_wall_seconds(trial):
    return trial["trial_wall_seconds_excluding_case_preparation"]

def save_figure(fig, stem):
    fig.savefig(FIGURES / f"{stem}.png")
    fig.savefig(FIGURES / f"{stem}.pdf")
"""
        ),
        markdown(
            r"""
## 1. Published square-lattice reference: what Vieijra--Nys did and did not win

Their physical-input ClebschTree imposes exact SU(2), then projects to
\(k=(0,0),A_1\).  The `j_max=4` used in the paper is the physical angular
momentum and therefore corresponds to repository cutoff `q_max=8`.

The panel below uses published **8x8 total energies**.  It subtracts the lowest
listed value separately at each coupling.  Missing parameter,
seed, and wall-time metadata mean this is an accuracy context plot—not a fair
efficiency ranking.  In particular, ClebschTree is competitive but is not the
lowest listed method at every coupling.
"""
        ),
        code(
            r"""
rows = vn["table_I_8x8_total_energies"]
counts = {j2: sum(row["j2_over_j1"] == j2 for row in rows) for j2 in {row["j2_over_j1"] for row in rows}}
couplings = sorted(j2 for j2, count in counts.items() if count > 1)
best = {j2: min(row["energy"] for row in rows if row["j2_over_j1"] == j2) for j2 in couplings}
methods = sorted({row["method"] for row in rows})

fig, ax = plt.subplots(figsize=(7.4, 4.2))
markers = ["o", "s", "^", "D", "v", "P", "X"]
for marker, method in zip(markers, methods):
    selected = sorted((row for row in rows if row["method"] == method and row["j2_over_j1"] in couplings), key=lambda row: row["j2_over_j1"])
    x = [row["j2_over_j1"] for row in selected]
    y = [1e3 * (row["energy"] - best[row["j2_over_j1"]]) / 64 for row in selected]
    yerr = [1e3 * row["uncertainty"] / 64 for row in selected]
    ax.errorbar(x, y, yerr=yerr, marker=marker, ms=5, linestyle="none",
                capsize=2, elinewidth=.8, label=method)
ax.axhline(0, color="black", lw=0.8)
ax.set(xlabel=r"$J_2/J_1$", ylabel=r"$(E-E_{\rm best\ listed})/N\;(10^{-3}J_1)$")
ax.set_title("Published square 8x8 accuracy context (not compute matched)")
ax.legend(ncol=2, fontsize=8, frameon=False)
fig.tight_layout()
save_figure(fig, "fig01_vieijra_nys_8x8_residual")
plt.show()
"""
        ),
        markdown(
            r"""
## 2. Square 4x4: the controlled symmetry ladder

The solid curves share the same fixed-\(M=0\) Hamiltonian support.  Thus
`m0_unconstrained_none` is an unconstrained **parametrization** on an externally
restricted support, not a true no-symmetry calculation.  The dashed full-space
curve is the separate, genuinely unrestricted \(2^{16}\) calibration.

The main physics test is factorial.  U(1) versus SU(2) tests the combined
effect of exact SU(2) channel algebra under equal width (their raw parameter
counts still differ); applying the same translation/A1 projector to both
backbones separates that effect from lattice-sector selection.
"""
        ),
        code(
            r"""
def case_by_id(payload, case_id):
    return next(case for case in payload["cases"] if case["case_id"] == case_id)

def assert_variational_and_floor(values, tolerance=1e-9, floor=1e-12):
    values = np.asarray(values, dtype=float)
    if np.min(values) < -tolerance:
        raise AssertionError(f"variational error below anchor: {np.min(values)}")
    return np.maximum(values, floor)

def trial_trajectory(case, trial, total_steps):
    history = trial["training"]["history"]
    steps = np.asarray([0] + [int(row["step"]) - 1 for row in history] + [total_steps])
    energy = np.asarray(
        [trial["training"]["initial_loss"]]
        + [row["loss_before_update"] for row in history]
        + [trial["metrics"]["energy"]]
    )
    return steps, (energy - case["exact_energy_anchor"]) / case["n_sites"]

def median_history(case, arm_id, total_steps):
    trials = [trial for trial in case["trials"] if trial["arm_id"] == arm_id]
    trajectories = [trial_trajectory(case, trial, total_steps) for trial in trials]
    steps = trajectories[0][0]
    values = np.asarray([trajectory[1] for trajectory in trajectories])
    return steps, np.median(values, axis=0), np.min(values, axis=0), np.max(values, axis=0)

square = case_by_id(spin, "square_4x4_j2_0p5_m0")
arm_order = [
    "m0_unconstrained_none", "u1_none", "u1_translation", "u1_gamma_a1",
    "su2_none", "su2_translation", "su2_gamma_a1",
]
labels = {
    "m0_unconstrained_none": "M0-restricted unconstrained",
    "u1_none": "U(1)",
    "u1_translation": "U(1)+translations",
    "u1_gamma_a1": "U(1)+Gamma.A1",
    "su2_none": "SU(2)",
    "su2_translation": "SU(2)+translations",
    "su2_gamma_a1": "SU(2)+Gamma.A1",
}
colors = dict(zip(arm_order, plt.cm.viridis(np.linspace(0.05, 0.9, len(arm_order)))))

fig, ax = plt.subplots(figsize=(7.4, 4.4))
spin_steps = int(spin["config"]["optimization"]["steps"])
for arm in arm_order:
    if not any(trial["arm_id"] == arm for trial in square["trials"]):
        continue
    steps, median, low, high = median_history(square, arm, spin_steps)
    positive = assert_variational_and_floor(median)
    ax.semilogy(steps, positive, marker="o", ms=3, lw=1.5, color=colors[arm], label=labels[arm])
    if len([t for t in square["trials"] if t["arm_id"] == arm]) > 1:
        ax.fill_between(steps, assert_variational_and_floor(low), assert_variational_and_floor(high), color=colors[arm], alpha=.12)
ax.set(xlabel="Adam updates", ylabel=r"energy error per site $\Delta E/N$")
ax.set_title(r"Square 4x4, $J_2/J_1=0.5$: median and seed range ($n=3$)")
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
save_figure(fig, "fig02_square_symmetry_ladder_convergence")
plt.show()
"""
        ),
        code(
            r"""
# The true-full calibration has a different width and budget, so it is not
# overlaid on the controlled ladder.
full_case = full["cases"][0]
full_trial = full_case["trials"][0]
full_steps_total = int(full["config"]["optimization"]["steps"])
full_steps, full_error = trial_trajectory(full_case, full_trial, full_steps_total)
fig, ax = plt.subplots(figsize=(6.4, 3.5))
ax.semilogy(full_steps, assert_variational_and_floor(full_error), "o--", color="0.35")
ax.set(xlabel="Adam updates", ylabel=r"energy error per site $\Delta E/N$",
       title="True full-Hilbert calibration (one seed; separate protocol)")
ax.text(.98, .96, f"final $M=0$ weight = {full_trial['metrics']['m0_weight']:.3f}",
        transform=ax.transAxes, ha="right", va="top")
fig.tight_layout()
save_figure(fig, "fig02c_true_full_sector_calibration")
plt.show()
"""
        ),
        code(
            r"""
# Factorial attribution: compare the same projector on U(1) and SU(2).
def median_final_error(case, arm_id):
    return np.median([
        trial["metrics"]["energy_error_per_site"]
        for trial in case["trials"] if trial["arm_id"] == arm_id
    ])

def paired_error_ratios(case, control_arm, exact_spin_arm):
    control = {
        trial["seed"]: trial["metrics"]["energy_error_per_site"]
        for trial in case["trials"] if trial["arm_id"] == control_arm
    }
    exact_spin = {
        trial["seed"]: trial["metrics"]["energy_error_per_site"]
        for trial in case["trials"] if trial["arm_id"] == exact_spin_arm
    }
    if set(control) != set(exact_spin):
        raise AssertionError("paired arms do not have identical seed sets")
    return {seed: control[seed] / exact_spin[seed] for seed in sorted(control)}

projectors = ["none", "translation", "gamma_a1"]
x_projectors = np.arange(len(projectors))
spin_projector_labels = ["none", "translations", r"$\Gamma A_1$"]
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
for backbone, marker in [("u1", "s"), ("su2", "o")]:
    arm_ids = [f"{backbone}_{projection}" for projection in projectors]
    values = [median_final_error(square, arm_id) for arm_id in arm_ids]
    spin_squared = [np.median([
        trial["metrics"]["spin_squared_expectation"]
        for trial in square["trials"] if trial["arm_id"] == arm_id
    ]) for arm_id in arm_ids]
    label = labels[f"{backbone}_none"].split("+")[0]
    axes[0].semilogy(x_projectors, assert_variational_and_floor(values), marker=marker, ms=6, lw=1.6, label=label)
    if backbone != "su2":
        axes[1].semilogy(x_projectors, np.abs(spin_squared), marker=marker, ms=6, lw=1.6, label=label)
for ax in axes:
    ax.set_xticks(x_projectors, spin_projector_labels)
axes[0].set(xlabel="lattice projector", ylabel=r"median $\Delta E/N$", title="Energy accuracy")
axes[1].set(xlabel="lattice projector", ylabel=r"median $|\langle S^2\rangle|$", title="Spin-sector purity")
axes[1].text(.04, .08, r"SU(2): $S=0$ exactly by construction",
             transform=axes[1].transAxes, fontsize=8)
axes[0].legend(frameon=False)
fig.suptitle("Square 4x4 factorial attribution: spin sector x lattice irrep")
fig.tight_layout()
save_figure(fig, "fig02b_square_spin_lattice_factorial")
plt.show()

print("SU(2) gain with no lattice projection:",
      median_final_error(square, "u1_none") / median_final_error(square, "su2_none"))
print("SU(2) gain after the same A1 projection:",
      median_final_error(square, "u1_gamma_a1") / median_final_error(square, "su2_gamma_a1"))
print("A1 paired energy-error ratios by seed (U1/SU2):",
      paired_error_ratios(square, "u1_gamma_a1", "su2_gamma_a1"))
"""
        ),
        code(
            r"""
# Accuracy versus raw parameter count uses only unprojected arms, avoiding a
# vertical stack of identical parameter counts from lattice projection.  The
# timing panel separately shows same-projector endpoint pairs.  Individual
# seeds are faint; large symbols are medians.
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.5))
capacity_arms = ["m0_unconstrained_none", "u1_none", "su2_none"]
short_labels = {
    "m0_unconstrained_none": r"$M=0$ control",
    "u1_none": "U(1)",
    "u1_translation": "U(1)+T",
    "u1_gamma_a1": r"U(1)+$\Gamma A_1$",
    "su2_none": "SU(2)",
    "su2_translation": "SU(2)+T",
    "su2_gamma_a1": r"SU(2)+$\Gamma A_1$",
}
for arm in capacity_arms:
    trials = [trial for trial in square["trials"] if trial["arm_id"] == arm]
    if not trials:
        continue
    trial_errors = np.asarray([trial["metrics"]["energy_error_per_site"] for trial in trials])
    error = np.median(trial_errors)
    params = trials[0]["parameter_count"]
    axes[0].loglog(np.full(len(trials), params), assert_variational_and_floor(trial_errors), ".", color=colors[arm], alpha=.35)
    axes[0].loglog(params, error, "o", color=colors[arm], label=short_labels[arm])

# Pre-specified raw-parameter brackets are hollow markers.  They are selected
# from parameter counts, not from observed energies.
bracket_case = brackets["cases"][0]
for bracket_index, arm in enumerate(sorted({trial["arm_id"] for trial in bracket_case["trials"]})):
    trials = [trial for trial in bracket_case["trials"] if trial["arm_id"] == arm]
    error = np.median([trial["metrics"]["energy_error_per_site"] for trial in trials])
    params = trials[0]["parameter_count"]
    bracket_label = "preselected raw-P brackets" if bracket_index == 0 else None
    axes[0].loglog(params, error, "o", mfc="none", mec="0.15", ms=7, label=bracket_label)

# Local endpoint timing is descriptive: it includes cold JIT and is neither a
# matched-work trajectory nor a sampled-VMC benchmark.
timing_projectors = ["none", "translation", "gamma_a1"]
timing_markers = {"none": "o", "translation": "s", "gamma_a1": "D"}
timing_labels = {"none": "none", "translation": "translations", "gamma_a1": r"$\Gamma A_1$"}
timing_colors = {"u1": "#35618d", "su2": "#2aa876"}
for projection in timing_projectors:
    pair = []
    for backbone in ["u1", "su2"]:
        arm = f"{backbone}_{projection}"
        trials = [trial for trial in square["trials"] if trial["arm_id"] == arm]
        trial_errors = np.asarray([trial["metrics"]["energy_error_per_site"] for trial in trials])
        trial_walls = np.asarray([trial_wall_seconds(trial) for trial in trials])
        wall = np.median(trial_walls)
        error = np.median(trial_errors)
        axes[1].loglog(trial_walls, assert_variational_and_floor(trial_errors), ".",
                       color=timing_colors[backbone], alpha=.28)
        axes[1].loglog(wall, error, marker=timing_markers[projection], linestyle="none",
                       ms=7, color=timing_colors[backbone])
        pair.append((wall, error))
    axes[1].loglog([point[0] for point in pair], [point[1] for point in pair],
                   color="0.55", lw=.9, alpha=.8)
axes[0].set(xlabel="raw trainable parameters", ylabel=r"median $\Delta E/N$")
axes[1].set(xlabel="local CPU endpoint seconds (cold JIT included)", ylabel=r"median $\Delta E/N$")
axes[0].set_title("Accuracy--capacity")
axes[1].set_title("Exact-support endpoint diagnostic")
handles, legend_labels = axes[0].get_legend_handles_labels()
from matplotlib.lines import Line2D
timing_handles = [
    Line2D([0], [0], color=timing_colors["u1"], marker="o", lw=0, label="U(1) timing"),
    Line2D([0], [0], color=timing_colors["su2"], marker="o", lw=0, label="SU(2) timing"),
] + [
    Line2D([0], [0], color="0.35", marker=timing_markers[key], lw=0, label=f"projector: {timing_labels[key]}")
    for key in timing_projectors
]
fig.legend(handles + timing_handles, legend_labels + [handle.get_label() for handle in timing_handles],
           loc="lower center", ncol=5, fontsize=7.2, frameon=False)
fig.suptitle("Fixed-update local diagnostic; sampled-VMC work audit pending")
fig.tight_layout(rect=(0, .16, 1, .94))
save_figure(fig, "fig03_square_accuracy_capacity_time")
plt.show()

print("True-full M=0 probability after training:", full_trial["metrics"]["m0_weight"])
print("Compiled raw-P brackets around SU(2):", square["parameter_brackets"])
"""
        ),
        markdown(
            r"""
## 3. Triangular 4x4: finite-cluster J2 scan, not a phase diagram

This torus is clean (no first/second-shell aliasing) and contains all three M
points, but it does **not** contain the 120-degree K momentum.  Consequently we
show energy/variance and the allowed M response only.  We do not plot a nominal
\(S(K)\), label a QSL window from these points, or infer a transition.

The correct production geometry for simultaneous K/M competition is 6x6.  A
3x6 torus contains K but aliases second-neighbor periodic images and is only a
shape stress control at nonzero \(J_2\).
"""
        ),
        code(
            r"""
tri_cases = sorted(
    [case for case in spin["cases"] if case["geometry"] == "triangular"],
    key=lambda case: case["j2_over_j1"],
)
tri_arms = ["u1_none", "u1_gamma_a1", "su2_none", "su2_gamma_a1"]
tri_colors = {arm: colors[arm] for arm in tri_arms}
fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8))
for arm in tri_arms:
    xs, energy, variance, m_response = [], [], [], []
    for case in tri_cases:
        trials = [trial for trial in case["trials"] if trial["arm_id"] == arm]
        if not trials:
            continue
        xs.append(case["j2_over_j1"])
        energy.append(np.median([trial["metrics"]["energy_error_per_site"] for trial in trials]))
        variance.append(np.median([trial["metrics"]["variance_per_site"] for trial in trials]))
        m_response.append(np.median([
            trial["metrics"]["physical_observables"]["stripe_M_structure_factor_max"] / case["n_sites"]
            for trial in trials
        ]))
        for trial in trials:
            axes[0].semilogy(case["j2_over_j1"], assert_variational_and_floor(trial["metrics"]["energy_error_per_site"]),
                            ".", color=tri_colors[arm], alpha=.28, ms=4)
    axes[0].semilogy(xs, assert_variational_and_floor(energy), marker="o", lw=1.5, color=tri_colors[arm], label=labels.get(arm, arm))
    axes[1].semilogy(xs, assert_variational_and_floor(variance), marker="o", lw=1.5, color=tri_colors[arm], label=labels.get(arm, arm))
    axes[2].plot(xs, m_response, marker="o", lw=1.5, color=tri_colors[arm], label=labels.get(arm, arm))
axes[0].set(xlabel=r"$J_2/J_1$", ylabel=r"median $\Delta E/N$")
axes[1].set(xlabel=r"$J_2/J_1$", ylabel=r"median variance/site")
axes[2].set(xlabel=r"$J_2/J_1$", ylabel=r"allowed-$M$ response $S(M)/N$")
axes[0].set_title("Finite-graph accuracy")
axes[1].set_title("Wavefunction residual")
axes[2].set_title("Allowed M-point response")
axes[2].legend(fontsize=7, frameon=False)
fig.suptitle("Triangular 4x4 finite-cluster J2 scan: no K point, no QSL claim")
fig.tight_layout()
save_figure(fig, "fig04_triangular_finite_cluster_scan")
plt.show()

print("Triangular unprojected paired wins (ratio U1/SU2 > 1 favors SU2):")
for case in tri_cases:
    ratios = paired_error_ratios(case, "u1_none", "su2_none")
    print(case["j2_over_j1"], ratios, "wins", sum(value > 1 for value in ratios.values()), "/", len(ratios))
"""
        ),
        code(
            r"""
def median_trial(case, arm_id):
    trials = [trial for trial in case["trials"] if trial["arm_id"] == arm_id]
    errors = np.asarray([trial["metrics"]["energy_error_per_site"] for trial in trials])
    return trials[int(np.argsort(errors)[len(errors)//2])]

tri_center = min(tri_cases, key=lambda case: abs(case["j2_over_j1"] - 0.125))
chosen_cases = [square, tri_center]
titles = [r"square 4x4, $J_2/J_1=0.5$", r"triangular 4x4, $J_2/J_1=0.125$"]
grids = []
for case in chosen_cases:
    trials = [trial for trial in case["trials"] if trial["arm_id"] == "su2_gamma_a1"]
    grids.append(np.median([
        np.asarray(trial["metrics"]["physical_observables"]["spin_structure_factor_grid"])
        for trial in trials
    ], axis=0))
common_vmax = max(float(np.max(grid)) for grid in grids)
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), layout="constrained")
for index, (ax, grid, title) in enumerate(zip(axes, grids, titles)):
    image = ax.imshow(grid, origin="lower", cmap="magma", vmin=0, vmax=common_vmax)
    ax.set(title=title, xlabel="reciprocal index $n_1$", ylabel="reciprocal index $n_2$")
    if index == 1:
        ax.text(.03, .96, "K unavailable on 4x4", transform=ax.transAxes,
                color="white", ha="left", va="top", fontsize=8)
fig.colorbar(image, ax=axes, fraction=.035, pad=.03, label=r"$S(\mathbf{q})$")
fig.suptitle("Seed-median SU(2)+A1 physical spin structure factors (shared scale)")
save_figure(fig, "fig05_spin_structure_factors")
plt.show()
"""
        ),
        markdown(
            r"""
## 4. HubbardCG: 1D calibration and 2x4 ladder bridge

The open chains compare the same physical-token tree with fixed
\((N_e,S_z)\) against exact \((N_e,S=0)\).  The 2x4 periodic ladder then adds
the identical signed translation/retained-space-group projector to both
backbones.  The displayed trajectories are the sealed three-seed validation.
The 2x4 geometry is not a bulk 2D result, and its order-16 group is not the
square-lattice \(C_{4v}\) group.
"""
        ),
        code(
            r"""
hubbard_labels = {
    "ne_sz_none": r"fixed $(N_e,S_z)$",
    "ne_sz_translation": r"fixed $(N_e,S_z)$+translations",
    "ne_sz_gamma_a1": r"fixed $(N_e,S_z)$+signed rectangular-G16",
    "su2_none": r"exact $(N_e,S)$",
    "su2_translation": r"SU(2)+translations",
    "su2_gamma_a1": r"SU(2)+signed rectangular-G16",
}
hubbard_arms = list(hubbard_labels)
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))
for arm in hubbard_arms:
    x, errors, doublons = [], [], []
    for index, case in enumerate(hubbard["cases"]):
        trials = [trial for trial in case["trials"] if trial["arm_id"] == arm]
        if not trials:
            continue
        x.append(index)
        errors.append(np.median([trial["metrics"]["energy_error_per_site"] for trial in trials]))
        doublons.append(np.median([trial["metrics"]["double_occupancy_per_site"] for trial in trials]))
    axes[0].semilogy(x, assert_variational_and_floor(errors), marker="o", linestyle="none", label=hubbard_labels[arm])
    axes[1].plot(x, doublons, marker="o", linestyle="none", label=hubbard_labels[arm])
tick_labels = [case["case_id"].replace("hubbard_", "") for case in hubbard["cases"]]
for ax in axes:
    ax.set_xticks(range(len(tick_labels)), tick_labels, rotation=22, ha="right")
axes[0].set(ylabel=r"median $\Delta E/N$", title="Ground-state accuracy")
axes[1].set(ylabel=r"double occupancy $D/N$", title="Physical charge fluctuation")
axes[1].legend(fontsize=8, frameon=False)
fig.suptitle("Hubbard symmetry ladder: chains to a periodic 2x4 ladder")
fig.tight_layout()
save_figure(fig, "fig06_hubbard_1d_2d_bridge")
plt.show()
"""
        ),
        code(
            r"""
ladder = case_by_id(hubbard, "hubbard_ladder_2x4_pbc_U4")
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
hubbard_projector_labels = ["none", "translations", "rectangular-G16"]
for backbone, marker, label in [
    ("ne_sz", "s", r"fixed $(N_e,S_z)$"),
    ("su2", "o", r"exact $(N_e,S)$"),
]:
    arm_ids = [f"{backbone}_{projection}" for projection in projectors]
    values = [median_final_error(ladder, arm_id) for arm_id in arm_ids]
    spin_squared = [np.median([
        trial["metrics"]["spin_squared_expectation"]
        for trial in ladder["trials"] if trial["arm_id"] == arm_id
    ]) for arm_id in arm_ids]
    axes[0].semilogy(x_projectors, assert_variational_and_floor(values), marker=marker, ms=6, lw=1.6, label=label)
    if backbone != "su2":
        axes[1].semilogy(x_projectors, np.abs(spin_squared), marker=marker, ms=6, lw=1.6, label=label)
for ax in axes:
    ax.set_xticks(x_projectors, hubbard_projector_labels)
axes[0].set(xlabel="signed lattice projector", ylabel=r"median $\Delta E/N$", title="Energy accuracy")
axes[1].set(xlabel="signed lattice projector", ylabel=r"median $|\langle S^2\rangle|$", title="Spin-sector purity")
axes[1].text(.04, .08, r"SU(2): $S=0$ exactly by construction",
             transform=axes[1].transAxes, fontsize=8)
axes[0].legend(frameon=False)
fig.suptitle("Hubbard 2x4 factorial attribution: spin sector x lattice irrep")
fig.tight_layout()
save_figure(fig, "fig06b_hubbard_spin_lattice_factorial")
plt.show()

print("Hubbard G16 paired energy-error ratios by seed (Abelian/SU2):",
      paired_error_ratios(ladder, "ne_sz_gamma_a1", "su2_gamma_a1"))
"""
        ),
        code(
            r"""
fig, ax = plt.subplots(figsize=(7.2, 4.2))
hubbard_steps = int(hubbard["config"]["optimization"]["steps"])
for arm in hubbard_arms:
    trials = [trial for trial in ladder["trials"] if trial["arm_id"] == arm]
    if not trials:
        continue
    trajectories = [trial_trajectory(ladder, trial, hubbard_steps) for trial in trials]
    steps = trajectories[0][0]
    curves = np.asarray([trajectory[1] for trajectory in trajectories])
    ax.semilogy(steps, assert_variational_and_floor(np.median(curves, axis=0)), marker="o", ms=3, label=hubbard_labels[arm])
    if len(trials) > 1:
        ax.fill_between(steps, assert_variational_and_floor(curves.min(0)), assert_variational_and_floor(curves.max(0)), alpha=.12)
ax.set(xlabel="Adam updates", ylabel=r"energy error per site $\Delta E/N$")
ax.set_title("Hubbard 2x4 PBC: median and seed range ($n=3$)")
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
save_figure(fig, "fig07_hubbard_ladder_convergence")
plt.show()
"""
        ),
        markdown(
            r"""
## 5. PRL local gate I: parameter--accuracy compression survives geometry and coupling

The primary local claim uses only the translation sector, where the retained
spaces remain nontrivial.  It compares five widths on both backbones, eight
independent seed labels per point, and four frustrated 4x4 Hamiltonians.  Two
couplings were held out from the initial central-point diagnostic.  Seed
labels across U(1) and SU(2) are **not** matched initial functions, so the
strict analysis uses independent seed-cluster bootstrap and an exact 8+8
permutation test rather than paired-seed error bars.

Across all four cases the integrated parameter frontier favors SU(2) by
**3.38--10.05x** in geometric energy error. This is an established
finite-graph result on the raw-parameter axis. Its microscopic optimization
mechanism and matched-work scalability remain hypotheses.
"""
        ),
        code(
            r"""
prl_case_order = [
    "square_4x4_j2_0p5_prl_pareto",
    "square_4x4_j2_0p55_heldout",
    "triangular_4x4_j2_0p125_prl_pareto",
    "triangular_4x4_j2_0p175_heldout",
]
prl_case_titles = {
    "square_4x4_j2_0p5_prl_pareto": r"square 4x4, $J_2/J_1=0.50$",
    "square_4x4_j2_0p55_heldout": r"square 4x4, $J_2/J_1=0.55$ (held out)",
    "triangular_4x4_j2_0p125_prl_pareto": r"triangular 4x4, $J_2/J_1=0.125$",
    "triangular_4x4_j2_0p175_heldout": r"triangular 4x4, $J_2/J_1=0.175$ (held out)",
}
prl_backbone_colors = {"u1": "#35618d", "su2": "#d55e00"}

fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.4), sharex=True, sharey=True)
for panel, (ax, case_id) in enumerate(zip(axes.flat, prl_case_order)):
    case = case_by_id(prl_pareto, case_id)
    for backbone, marker in [("u1", "s"), ("su2", "o")]:
        medians = []
        parameter_counts = []
        for width in [2, 3, 4, 5, 6]:
            arm_id = f"{backbone}_D{width}_translation"
            trials = [trial for trial in case["trials"] if trial["arm_id"] == arm_id]
            errors = np.asarray([trial["metrics"]["energy_error_per_site"] for trial in trials])
            params = int(trials[0]["parameter_count"])
            parameter_counts.append(params)
            medians.append(float(np.median(errors)))
            ax.loglog(
                np.full(errors.size, params), assert_variational_and_floor(errors),
                marker, ms=3.2, linestyle="none", color=prl_backbone_colors[backbone],
                alpha=.25,
            )
        ax.loglog(
            parameter_counts, assert_variational_and_floor(medians), marker=marker,
            ms=6, lw=1.8, color=prl_backbone_colors[backbone],
            label="U(1), fixed $M=0$" if backbone == "u1" else "exact SU(2), $S=0$",
        )
    factor = prl_analysis["cases"][case_id]["parameter_pareto"][
        "geometric_error_advantage_factor"
    ]
    common_parameter_interval = prl_analysis["cases"][case_id]["parameter_pareto"][
        "common_budget_interval"
    ]
    for boundary in common_parameter_interval:
        ax.axvline(boundary, color="0.45", lw=.7, linestyle=":")
    lower = prl_analysis["cases"][case_id]["parameter_pareto"][
        "independent_cluster_bootstrap"
    ]["average_log10_advantage"]["one_sided_95_lower"]
    ax.set_title(prl_case_titles[case_id], fontsize=10)
    ax.text(
        .04, .06, rf"integrated SU(2) gain $={factor:.2f}\times$" + "\n" +
        rf"95% one-sided lower $={lower:.2f}$ dex",
        transform=ax.transAxes, fontsize=8, va="bottom",
        bbox={"facecolor": "white", "alpha": .78, "edgecolor": "none"},
    )
    if panel >= 2:
        ax.set_xlabel("raw trainable parameters")
    if panel % 2 == 0:
        ax.set_ylabel(r"final energy error per site $\Delta E/N$")
handles, legend_labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=False)
fig.suptitle("Translation-sector parameter--accuracy Pareto (dotted: common integration range)")
fig.tight_layout(rect=(0, .06, 1, .96))
save_figure(fig, "fig08_spin_prl_parameter_pareto")
plt.show()

parameter_factors = {
    case_id: prl_analysis["cases"][case_id]["parameter_pareto"][
        "geometric_error_advantage_factor"
    ]
    for case_id in prl_case_order
}
print("Integrated parameter-frontier SU(2) gains:", parameter_factors)
print("Range:", min(parameter_factors.values()), "to", max(parameter_factors.values()))
"""
        ),
        markdown(
            r"""
## 6. Exploratory long-horizon CPU timing: not publication-eligible

The recorded-time diagnostic uses anytime lower envelopes and excludes compilation
from the warm clock.  Its U(1) frontier includes the targeted 1200-update D=6
continuation; the horizon-completed SU(2) trajectory is included whenever it
is present in the sealed strict-analysis summary.  Curves are compared only on
their common **measured** interval: no extrapolation beyond a completed run.

The frontiers cross and provide no evidence for a uniform local CPU speedup.
However, the input artifacts do not record verified machine-load isolation,
and part of the U(1)-long run may have overlapped another process. The
contamination is conservative for an SU(2) speed claim, but the exact factors
and p-values are **exploratory**, not publication-grade. No optimization-
landscape mechanism is inferred from these curves.
"""
        ),
        code(
            r"""
fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.4), sharex=False, sharey=True)
for panel, (ax, case_id) in enumerate(zip(axes.flat, prl_case_order)):
    frontier = prl_analysis["cases"][case_id]["matched_time_pareto"]["warm"]
    seconds = np.asarray(frontier["budget_axis"], dtype=float)
    u1_log = np.asarray(frontier["u1_frontier_log10_error_per_site"], dtype=float)
    su2_log = np.asarray(frontier["su2_frontier_log10_error_per_site"], dtype=float)
    u1_error, su2_error = 10.0**u1_log, 10.0**su2_log
    ax.loglog(seconds, u1_error, "s-", markevery=4, ms=4, lw=1.7,
              color=prl_backbone_colors["u1"], label="U(1) anytime frontier")
    ax.loglog(seconds, su2_error, "o-", markevery=4, ms=4, lw=1.7,
              color=prl_backbone_colors["su2"], label="SU(2) anytime frontier")
    ax.fill_between(seconds, u1_error, su2_error, where=u1_error > su2_error,
                    color=prl_backbone_colors["su2"], alpha=.10,
                    interpolate=True)
    ax.fill_between(seconds, u1_error, su2_error, where=u1_error < su2_error,
                    color=prl_backbone_colors["u1"], alpha=.10,
                    interpolate=True)
    difference = u1_log - su2_log
    crossing_indices = np.flatnonzero(np.signbit(difference[1:]) != np.signbit(difference[:-1]))
    for index in crossing_indices:
        cross_time = np.sqrt(seconds[index] * seconds[index + 1])
        cross_error = 10.0 ** np.mean([
            u1_log[index], u1_log[index + 1], su2_log[index], su2_log[index + 1]
        ])
        ax.plot(cross_time, cross_error, "x", color="black", ms=6, mew=1.1)
    factor = frontier["geometric_error_advantage_factor"]
    worst = frontier["worst_log10_advantage"]
    ax.set_title(prl_case_titles[case_id], fontsize=10)
    ax.text(
        .04, .06,
        rf"integrated factor $={factor:.2f}\times$" + "\n" +
        rf"worst advantage $={worst:.2f}$ dex" +
        "\nnumerical checks: FAIL\ntiming claim: INELIGIBLE",
        transform=ax.transAxes, fontsize=8, va="bottom", color="#8b1a1a",
        bbox={"facecolor": "white", "alpha": .80, "edgecolor": "none"},
    )
    ax.set_xlabel("measured warm training time (s)")
    if panel % 2 == 0:
        ax.set_ylabel(r"anytime energy error per site $\Delta E/N$")
handles, legend_labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=False)
fig.suptitle("Exploratory CPU timing: crossing frontiers; machine isolation not audited")
fig.tight_layout(rect=(0, .06, 1, .96))
save_figure(fig, "fig09_spin_prl_warm_anytime_diagnostic")
plt.show()

for case_id in prl_case_order:
    warm = prl_analysis["cases"][case_id]["matched_time_pareto"]["warm"]
    print(case_id, {
        "common_seconds": warm["common_observed_time_interval_seconds"],
        "integrated_factor": warm["geometric_error_advantage_factor"],
        "worst_advantage_dex": warm["worst_log10_advantage"],
        "u1_source": warm.get("u1_D6_source"),
        "su2_source": warm.get("su2_D6_source", "base 300-update trajectory"),
    })
"""
        ),
        markdown(
            r"""
## 7. Sampled-estimator preflight: the exhaustive-to-VMC bridge

Before leaving exact support, an N=8 oracle samples i.i.d. from the exact Born
distribution and compares finite-sample energy and gradient estimators with
their exhaustive values.  This isolates estimator correctness from Markov
chain mixing.  The expected energy-error decay and improving gradient cosine
are visible for both backbones and both projector choices.

This is a necessary implementation gate, not evidence of scalable VMC: the
draws are independent, the state is a random initialization, and no 6x6
training result is inferred from this panel.
"""
        ),
        code(
            r"""
sampled_case = sampled_preflight["cases"][0]
sampled_labels = {
    "u1_none": "U(1)",
    "u1_translation": "U(1)+translations",
    "su2_none": "SU(2)",
    "su2_translation": "SU(2)+translations",
}
sampled_styles = {
    "u1_none": ("#35618d", "--", "s"),
    "u1_translation": ("#35618d", "-", "s"),
    "su2_none": ("#d55e00", "--", "o"),
    "su2_translation": ("#d55e00", "-", "o"),
}
fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
for trial in sampled_case["trials"]:
    arm_id = trial["arm_id"]
    color, linestyle, marker = sampled_styles[arm_id]
    summaries = trial["metrics"]["sample_summaries"]
    counts = np.asarray([row["sample_count"] for row in summaries], dtype=float)
    rmse = np.asarray([row["energy_rmse_per_site"] for row in summaries], dtype=float)
    cosine = np.asarray([row["gradient_cosine_mean"] for row in summaries], dtype=float)
    axes[0].loglog(counts, rmse, marker=marker, linestyle=linestyle,
                   color=color, lw=1.6, ms=5, label=sampled_labels[arm_id])
    axes[1].semilogx(counts, cosine, marker=marker, linestyle=linestyle,
                     color=color, lw=1.6, ms=5, label=sampled_labels[arm_id])

# A slope guide is visual only; its normalization is set by the displayed data.
all_rmse_at_first = [
    trial["metrics"]["sample_summaries"][0]["energy_rmse_per_site"]
    for trial in sampled_case["trials"]
]
guide_counts = np.asarray(sampled_preflight["config"]["sampling"]["sample_counts"], dtype=float)
guide = np.median(all_rmse_at_first) * np.sqrt(guide_counts[0] / guide_counts)
axes[0].loglog(guide_counts, guide, color="0.35", lw=1.0, linestyle=":",
               label=r"$M^{-1/2}$ guide")
axes[0].set(xlabel="independent Born samples $M$", ylabel="energy-estimator RMSE per site",
            title="Energy estimator")
axes[1].set(xlabel="independent Born samples $M$", ylabel="mean gradient cosine",
            title="VMC gradient direction", ylim=(0.55, 1.01))
axes[1].axhline(1.0, color="0.35", lw=.8, linestyle=":")
axes[0].legend(fontsize=7.5, frameon=False)
fig.suptitle("Exact-Born oracle preflight (16 replicates; not an MCMC benchmark)")
fig.tight_layout()
save_figure(fig, "fig10_spin_sampled_estimator_preflight")
plt.show()

print("Largest-sample preflight metrics:")
for trial in sampled_case["trials"]:
    endpoint = trial["metrics"]["sample_summaries"][-1]
    print(trial["arm_id"], {
        "M": endpoint["sample_count"],
        "energy_RMSE_per_site": endpoint["energy_rmse_per_site"],
        "gradient_cosine_mean": endpoint["gradient_cosine_mean"],
        "gradient_cosine_minimum": endpoint["gradient_cosine_minimum"],
    })
"""
        ),
        markdown(
            r"""
### Full-spin observable algebra gate

The paper-facing convention is now fixed to

\[
S(\mathbf q)=\frac{1}{N}\sum_{ij}e^{i\mathbf q\cdot
(\mathbf r_i-\mathbf r_j)}\langle\mathbf S_i\cdot\mathbf S_j\rangle .
\]

The order-parameter convention is stored separately as
\(m^2(\mathbf q)=S(\mathbf q)/N\).  A sealed exact-small artifact compares the
JAX action, including transverse exchange, with an independent fixed-\(S_z\)
sparse operator.  It also checks singlet \(S(0)=0\),
\(\sum_q S(q)=3N/4\), and \(\sum_q m^2(q)=3/4\).  Multiple momenta share one
set of pair-swapped amplitude calls.  This closes the **operator-action** gate,
not the end-to-end trained-6x6 MCMC measurement gate.
"""
        ),
        code(
            r"""
observable_trial = observable_validation["cases"][0]["trials"][0]
if not observable_trial["all_gates_passed"]:
    raise RuntimeError("full-spin observable algebra gate did not pass")
observable_metrics = observable_trial["metrics"]
print("Full-spin observable validation:")
print({
    "normalization": observable_trial["normalization_contract"],
    "N8_action_max_abs_error": observable_metrics[
        "n8_complex_state_action_max_abs_discrepancy"
    ],
    "singlet_q0_action_max_abs": observable_metrics[
        "singlet_q0_action_max_abs"
    ],
    "sum_Sq": observable_metrics["structure_factor_sum"],
    "sum_Sq_target": observable_metrics["structure_factor_sum_target"],
    "sum_m2": observable_metrics["m2_sum"],
    "sum_m2_target": observable_metrics["m2_sum_target"],
})
"""
        ),
        markdown(
            r"""
## 8. Appendix-style 6x6 runtime readiness: no energy or phase claim

These two physical-input smokes exercise the same translation-projected U(1)
and SU(2) backbones on square and triangular 6x6 tori without ED.  All
configured batch kernels are finite, all chains move, and the minimum
effective sample sizes clear the predeclared threshold.  The curves also make
the present CPU cost explicit: the SU(2) realization is slower per state in
this unoptimized kernel.

The sampled states are random initializations.  Their energies and diagonal
(S^{zz}) probes are used only for mixing diagnostics; this panel establishes
runtime readiness, **not** energy accuracy, optimization efficiency, magnetic
order, or a QSL.
"""
        ),
        code(
            r"""
runtime_titles = {
    "square": r"square 6x6, $J_2/J_1=0.5$",
    "triangular": r"triangular 6x6, $J_2/J_1=0.125$",
}
fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.1))
for row, geometry in enumerate(("square", "triangular")):
    runtime_case = runtime_smokes[geometry]["cases"][0]
    for trial in runtime_case["trials"]:
        backbone = trial["backbone"]
        color = prl_backbone_colors[backbone]
        marker = "s" if backbone == "u1" else "o"
        label = "U(1)+translations" if backbone == "u1" else "SU(2)+translations"
        profile = trial["performance_profile"]
        batches = np.asarray([point["batch_size"] for point in profile])
        hpsi = np.asarray([
            point["hpsi_warm_throughput_states_per_second"] for point in profile
        ])
        gradient = np.asarray([
            point["gradient_warm_throughput_states_per_second"] for point in profile
        ])
        axes[row, 0].plot(batches, hpsi, marker=marker, color=color, lw=1.6,
                          ms=5, label=label)
        axes[row, 1].plot(batches, gradient, marker=marker, color=color, lw=1.6,
                          ms=5, label=label)

        measurement = trial["sampler"]["measurement_diagnostics"]
        observable_ess = [
            value["ess"]
            for value in trial["observables"].values()
            if isinstance(value, dict) and "ess" in value
        ]
        axes[row, 2].scatter(
            measurement["conditional_move_acceptance_mean"], min(observable_ess),
            marker=marker, color=color, s=42, label=label,
        )
    for column in (0, 1):
        axes[row, column].set_xscale("log", base=2)
        axes[row, column].set_yscale("log")
        axes[row, column].set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
        axes[row, column].set_xlabel("profile batch size")
    axes[row, 0].set_ylabel(
        runtime_titles[geometry] + "\n" + "warm states/s"
    )
    axes[row, 2].axhline(50, color="0.35", lw=.8, linestyle=":")
    axes[row, 2].axvline(.1, color="0.35", lw=.8, linestyle=":")
    axes[row, 2].axvline(.9, color="0.35", lw=.8, linestyle=":")
    axes[row, 2].set(
        xlabel="conditional move acceptance",
        ylabel="minimum ESS over diagnostics",
        xlim=(0, .95),
    )
axes[0, 0].set_title(r"warm $H\psi$ throughput")
axes[0, 1].set_title("warm value+gradient throughput")
axes[0, 2].set_title("MCMC readiness")
handles, legend_labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=False)
fig.suptitle("6x6 random-state runtime readiness only (CPU; no accuracy or phase claim)")
fig.tight_layout(rect=(0, .06, 1, .95))
save_figure(fig, "fig11_spin_6x6_runtime_readiness")
plt.show()

for geometry, payload in runtime_smokes.items():
    print("\n", geometry, "6x6 random-state readiness")
    for trial in payload["cases"][0]["trials"]:
        measurement = trial["sampler"]["measurement_diagnostics"]
        observable_ess = [
            value["ess"]
            for value in trial["observables"].values()
            if isinstance(value, dict) and "ess" in value
        ]
        print(trial["arm_id"], {
            "all_gates_passed": trial["all_readiness_gates_passed"],
            "conditional_acceptance": measurement["conditional_move_acceptance_mean"],
            "frozen_chain_fraction": measurement["frozen_chain_fraction"],
            "minimum_diagnostic_ESS": min(observable_ess),
        })
"""
        ),
        markdown(
            r"""
### End-to-end 6x6 full-\(S(\mathbf q)\) bridge

The sealed follow-up reconstructs each runtime arm from its source-checked
checkpoint, verifies the cached amplitudes, and evaluates all square
((0,0)), Neel and stripe modes or triangular ((0,0)), K and three M modes
with shared pair-amplitude work.  Each mode has 144 autocorrelated local
samples.  Exact SU(2) gives (S(0)=0) configuration by configuration to
machine precision on both geometries, while U(1) retains total-spin
contamination.

This is a strong end-to-end correctness check and a cost measurement at random
initialization.  The nonzero-momentum values are deliberately not shown as
magnetic-order estimates; trained-state trajectories are the next gate.
"""
        ),
        code(
            r"""
print("6x6 full-spin observable aggregation probe:")
for case in observable_probe["cases"]:
    for trial in case["trials"]:
        q0 = trial["observables"]["mode_summaries"]["q0"]
        print(case["geometry"], trial["arm_id"], {
            "samples_per_mode": trial["sampling"]["total_local_samples_per_mode"],
            "q0_mean": q0["mean_real"],
            "q0_blocked_chain_mcse": q0["blocked_chain_mcse"],
            "q0_maximum_abs_local": trial["observables"]["q0_maximum_abs_local"],
            "warm_multi_q_seconds_per_16_states": trial["cost"][
                "warm_observable_median_seconds"
            ],
            "all_gates_passed": trial["all_readiness_gates_passed"],
})
"""
        ),
        markdown(
            r"""
## 9. First parameter-updating 6x6 pilot: a useful negative result

The first actual sampled-Adam run used one seed, D4, 32 persistent chains,
20 updates, learning rate \(5\times10^{-4}\), and the same translation
projector on square \(J_2/J_1=0.5\) and triangular \(J_2/J_1=0.125\).  The
four per-arm registered measurement blocks were replayed from sealed checkpoints;
all summaries reproduce exactly and all replay gates pass.

This deliberately short pilot does **not** enter a useful optimization regime.
For every arm, the update-0 to update-20 energy change is below one combined
MCSE.  The large SU(2)--U(1) energy offset is already present at update 0, so
it is a consequence of the exact singlet prior at random initialization—not
an optimization win.  The square SU(2) trajectory was resumed once, making
its timing ineligible; no wall-clock comparison is made here.
"""
        ),
        code(
            r"""
pilot_colors = {"u1": "#35618d", "su2": "#d55e00"}
pilot_markers = {"u1": "s", "su2": "o"}
fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0), sharex="col")
for row_index, post_case in enumerate(sampled_post_analysis["cases"]):
    geometry = post_case["geometry"]
    for trial in post_case["trials"]:
        backbone = trial["backbone"]
        color = pilot_colors[backbone]
        marker = pilot_markers[backbone]
        label = "U(1)+translations" if backbone == "u1" else "SU(2)+translations"
        updates = np.asarray([point["update"] for point in trial["replay_updates"]])
        energies = np.asarray([
            point["energy_E_per_site"]["mean_real"]
            for point in trial["replay_updates"]
        ])
        energy_mcse = np.asarray([
            point["energy_E_per_site"]["chain_mean_mcse"]
            for point in trial["replay_updates"]
        ])
        variances = np.asarray([
            point["variance_H_per_site"]["variance_H_per_site"]
            for point in trial["replay_updates"]
        ])
        variance_mcse = np.asarray([
            point["variance_H_per_site"]["variance_H_per_site_chain_mcse"]
            for point in trial["replay_updates"]
        ])
        axes[row_index, 0].errorbar(
            updates, energies, yerr=energy_mcse, color=color, marker=marker,
            lw=1.5, ms=5, capsize=2, elinewidth=.8, label=label,
        )
        update0 = trial["runtime_random_state_update0"]
        axes[row_index, 0].errorbar(
            [0], [update0["energy_E_per_site"]],
            yerr=[update0["energy_E_per_site_mcse"]], color=color,
            marker=marker, mfc="white", linestyle="none", ms=6,
            capsize=2, elinewidth=.8,
        )
        axes[row_index, 1].errorbar(
            updates, variances, yerr=variance_mcse, color=color, marker=marker,
            lw=1.5, ms=5, capsize=2, elinewidth=.8, label=label,
        )
    axes[row_index, 0].axhline(0, color="0.4", lw=.7, linestyle=":")
    axes[row_index, 0].set_ylabel(runtime_titles[geometry] + "\n" + r"$E/N$")
    axes[row_index, 1].set_ylabel(r"$\mathrm{Var}(H)/N$")
    axes[row_index, 1].set_ylim(bottom=0)
for ax in axes[-1]:
    ax.set_xlabel("Adam update")
axes[0, 0].set_title("Energy (open marker: independent update-0 block)")
axes[0, 1].set_title("Local-energy variance")
handles, pilot_legend_labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, pilot_legend_labels, loc="lower center", ncol=2, frameon=False)
fig.suptitle("Negative 6x6 optimizer pilot: no MC-resolved energy lowering")
fig.tight_layout(rect=(0, .06, 1, .95))
save_figure(fig, "fig12_spin_6x6_negative_optimizer_pilot")
plt.show()

for post_case in sampled_post_analysis["cases"]:
    by_backbone = {trial["backbone"]: trial for trial in post_case["trials"]}
    initial_gap = (
        by_backbone["su2"]["runtime_random_state_update0"]["energy_E_per_site"]
        - by_backbone["u1"]["runtime_random_state_update0"]["energy_E_per_site"]
    )
    final_gap = post_case["same_schedule_SU2_vs_U1_step20"]["energy"]["difference"]
    print(post_case["geometry"], {
        "SU2_minus_U1_E_over_N_at_update0": initial_gap,
        "SU2_minus_U1_E_over_N_at_update20": final_gap,
        "per_arm_update0_to_20_signed_lowering_z": {
            name: trial["runtime_update0_to_pilot_step20_energy_comparison"][
                "signed_energy_lowering_z"
            ]
            for name, trial in by_backbone.items()
        },
    })
"""
        ),
        markdown(
            r"""
The full rotationally invariant structure factors also show no robust trend.
Every primary endpoint change is below two paired-chain MCSE; one of several
single-seed slope diagnostics barely crosses two MCSE, which is neither a
replica-level significance test nor evidence for order.  The measured
nonzero-momentum values remain \(O(1)\), near the local-moment scale \(3/4\),
whereas a magnetically ordered peak would grow as \(O(N)\).  Exact SU(2)
continues to enforce \(S(0)=0\) to about \(4.2\times10^{-17}\) in the block
means.  These data support no magnetic-order, QSL, or efficiency claim.
"""
        ),
        code(
            r"""
primary_values = []
for post_case in sampled_post_analysis["cases"]:
    for trial in post_case["trials"]:
        primary_names = trial["observable_contract"]["primary"]
        endpoint_ratios = {
            name: trial["trends"][name]["paired_chain_primary"][
                "paired_endpoint_abs_delta_over_mcse"
            ]
            for name in primary_names
        }
        for point in trial["replay_updates"]:
            primary_values.extend(
                point["physics_observables"][name]["mean_real"]
                for name in primary_names
            )
        q0_max = max(
            abs(point["full_SU2_structure_factor_mode_summaries"]["q0"]["mean_real"])
            for point in trial["replay_updates"]
        )
        print(post_case["geometry"], trial["backbone"], {
            "primary_endpoint_abs_delta_over_paired_MCSE": endpoint_ratios,
            "maximum_abs_q0_block_mean": q0_max,
        })
print("primary nonzero-momentum S(q) range:", min(primary_values), max(primary_values))
"""
        ),
        markdown(
            r"""
## 10. Center-only 50-update scan: primary decision and semantic sensitivity

The center-only backward pass makes the next optimizer test affordable without
changing the U(1)/SU(2) physical backbones.  Three learning rates
\(10^{-3},2\times10^{-3},5\times10^{-3}\) share the same initial parameters,
post-burn chains, proposal streams, update schedule, and fixed-measurement
streams.  The panels below use lines and points—not parameter-count or
wall-time bars—to show total energy per site, \({\rm Var}(H)/N\), and
conditional Metropolis acceptance.

Two decisions must remain visible side by side:

1. The immutable literal primary selector used
   `accepted_move_fraction >= 0.15`, namely accepted nontrivial moves divided
   by *all* proposals.  With roughly 48% null proposals it returned
   `no_adam_promotion` on both geometries.
2. The intended preregistration semantics were
   `conditional_move_acceptance >= 0.15`, namely acceptance conditioned on a
   nontrivial proposal.  We do **not** rewrite the primary artifact.  A new,
   separately sealed secondary sensitivity changes only that field and selects
   \(2\times10^{-3}\) for square and \(10^{-3}\) for triangular.

The largest learning rate often lowers the sampled energy most, but its U(1)
arms fall below the corrected mixing threshold.  Even the lowest endpoints
remain far above the ground-state scale near \(-0.5J_1\) per site, so these
trajectories are optimizer/mixer diagnostics—not converged states.  They
support no efficiency, magnetic-order, QSL, or phase claim.
"""
        ),
        code(
            r"""
lr_colors = {
    "lr_1e-3": "#0072b2",
    "lr_2e-3": "#009e73",
    "lr_5e-3": "#d55e00",
}
lr_labels = {
    "lr_1e-3": r"$10^{-3}$",
    "lr_2e-3": r"$2\times10^{-3}$",
    "lr_5e-3": r"$5\times10^{-3}$",
}
backbone_style = {
    "u1": {"linestyle": "--", "marker": "s", "label": "U(1)"},
    "su2": {"linestyle": "-", "marker": "o", "label": "SU(2)"},
}
geometry_cases = {
    "square": "square_6x6_j2_0p5",
    "triangular": "triangular_6x6_j2_0p125",
}
fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.0), sharex="col")
for row_index, (geometry, case_id) in enumerate(geometry_cases.items()):
    for candidate_id, payload in sorted(
        lr_scan_payloads.items(), key=lambda item: float(item[1]["learning_rate"])
    ):
        case = next(item for item in payload["cases"] if item["case_id"] == case_id)
        for trial in case["trials"]:
            backbone = trial["backbone"]
            style = backbone_style[backbone]
            label = f"{lr_labels[candidate_id]}  {style['label']}"
            measurements = sorted(
                trial["fixed_parameter_measurements"], key=lambda item: item["update"]
            )
            updates = np.asarray([item["update"] for item in measurements])
            energies = np.asarray([item["energy_per_site"] for item in measurements])
            energy_mcse = np.asarray(
                [item["energy_per_site_chain_mcse"] for item in measurements]
            )
            variances = np.asarray(
                [item["variance_h_per_site"] for item in measurements]
            )
            variance_mcse = np.asarray(
                [item["variance_h_per_site_chain_mcse"] for item in measurements]
            )
            common = dict(
                color=lr_colors[candidate_id],
                linestyle=style["linestyle"],
                marker=style["marker"],
                lw=1.45,
                ms=4.5,
                label=label,
            )
            axes[row_index, 0].errorbar(
                updates, energies, yerr=energy_mcse, capsize=2,
                elinewidth=.7, **common,
            )
            axes[row_index, 1].errorbar(
                updates, variances, yerr=variance_mcse, capsize=2,
                elinewidth=.7, **common,
            )
            history_updates = np.asarray([item["update"] for item in trial["history"]])
            conditional = np.asarray([
                item["sampler"]["conditional_move_acceptance"]
                for item in trial["history"]
            ])
            axes[row_index, 2].plot(
                history_updates, conditional, color=lr_colors[candidate_id],
                linestyle=style["linestyle"], lw=1.25, alpha=.9, label=label,
            )
    axes[row_index, 0].set_ylabel(
        ("square $J_2/J_1=0.5$" if geometry == "square" else
         "triangular $J_2/J_1=0.125$") + "\n" + r"$E/N$"
    )
    axes[row_index, 1].set_ylabel(r"$\mathrm{Var}(H)/N$")
    axes[row_index, 1].set_ylim(bottom=0)
    axes[row_index, 2].set_ylabel("conditional acceptance")
    axes[row_index, 2].axhline(.15, color="0.25", lw=.9, linestyle=":")
for column, title in enumerate(
    ("fixed-parameter energy", "fixed-parameter variance", "training-chain mixing")
):
    axes[0, column].set_title(title)
    axes[-1, column].set_xlabel("Adam update")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
fig.suptitle(
    "Center-only 6x6 LR scan: descent remains far from a ground-state benchmark"
)
fig.tight_layout(rect=(0, .10, 1, .95))
save_figure(fig, "fig13_spin_6x6_center_only_lr_metric_correction")
plt.show()

print("immutable literal primary:", literal_primary_binding["actions_by_case"])
print("secondary conditional-acceptance sensitivity:", {
    geometry: {
        "candidate": decision["selected_candidate_id"],
        "learning_rate": decision["selected_learning_rate"],
        "training_authorized": decision["training_authorized_by_secondary_selector"],
    }
    for geometry, decision in metric_lr_decisions.items()
})
for geometry_result in metric_lr_selector["geometry_results"]:
    print("\n", geometry_result["geometry"])
    for candidate in geometry_result["candidate_diagnostics"]:
        print(candidate["candidate_id"], {
            backbone: {
                "conditional_last10": arm[
                    "last10_conditional_move_acceptance_median"
                ],
                "accepted_all_proposals_last10": arm[
                    "last10_accepted_move_fraction_median"
                ],
                "nontrivial_last10": arm[
                    "last10_nontrivial_proposal_fraction_median"
                ],
            }
            for backbone, arm in candidate["arms"].items()
        })
"""
        ),
        markdown(
            r"""
## 11. Exact SU(2) as a computational resource: force work--variance crossover

Inside an exact singlet manifold, `H` and diagonal `3Hz` give the same energy
and variational force in expectation.  They do not have the same sampling
noise: full `H` becomes zero-variance at an eigenstate, while `3Hz` avoids the
exchange-connected neural-network evaluations.  The left panel reports the
exact asymptotic force-MSE times a projected-amplitude work proxy on `N=8`;
the right panel is a separate non-ED `6x6` local CPU kernel profile.

The result supports a staged protocol—diagonal early/mid training and full-H
late refinement—not permanent replacement of the physical local energy.
"""
        ),
        code(
            r"""
colors = {"square": "#0072b2", "triangular": "#d55e00"}
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))

for case in singlet_force["cases"]:
    geometry = case["geometry"]
    rows = [
        row for row in case["report"]["rows"]
        if row["sample_count"] == 64
        and row["diagonal_over_full_asymptotic_work_adjusted_mse"] is not None
    ]
    rows = sorted(rows, key=lambda row: row["energy_error"], reverse=True)
    x = np.asarray([row["energy_error"] for row in rows])
    y = np.asarray([
        row["diagonal_over_full_asymptotic_work_adjusted_mse"] for row in rows
    ])
    axes[0].loglog(x, y, marker="o", lw=1.7, color=colors[geometry], label=geometry)
    for row, xx, yy in zip(rows, x, y):
        axes[0].annotate(
            f"{row['ground_state_overlap']:.2g}", (xx, yy),
            xytext=(4, 4), textcoords="offset points", fontsize=8,
        )
axes[0].axhline(1.0, color="0.25", linestyle=":", lw=1.0)
axes[0].set_xlabel(r"energy error $E-E_0$")
axes[0].set_ylabel(r"$(\mathrm{MSE}\times\mathrm{work})_{3H_z}/(\mathrm{MSE}\times\mathrm{work})_H$")
axes[0].set_title("N=8 exact-Born force frontier")
axes[0].legend(frameon=False)

for profile in singlet_force["runtime_profiles"]:
    geometry = profile["geometry"]
    values = [
        profile["full"]["warm_median_seconds"],
        profile["diagonal"]["warm_median_seconds"],
    ]
    axes[1].plot(
        (0, 1), values, marker="o", lw=1.7, color=colors[geometry],
        label=f"{geometry}: {profile['measured_diagonal_speedup']:.2f}x",
    )
axes[1].set_xticks((0, 1), ("full H", r"diagonal $3H_z$"))
axes[1].set_yscale("log")
axes[1].set_ylabel("warm force-kernel seconds")
axes[1].set_title("6x6 D4 + translations, local CPU")
axes[1].legend(frameon=False)

fig.suptitle("Exact-singlet symmetry removes exchange-connected force evaluations")
fig.tight_layout()
save_figure(fig, "fig14_spin_singlet_force_efficiency")
plt.show()

print("Timing profiles are exploratory and not claim eligible:")
for profile in singlet_force["runtime_profiles"]:
    print(profile["profile_id"], profile["measured_diagonal_speedup"], "x")
"""
        ),
        markdown(
            r"""
## 12. Minimal support-aware Spatial-CG: exact architecture, negative efficiency gate

The minimal candidate coherently resonates complete balanced fusion routes.
Every route covers every physical spin once, all routes share one set of
reduced CG tensors, and only two trainable query coefficients over fixed
shell-locality descriptors are added.  The state therefore has a concrete
block-spin interpretation: competing (J_1) and (J_2) bonds prefer incompatible
hierarchical clusterings.  In the `4x4` pilot, `qmax=8` retains every
intermediate-spin channel of the balanced `N=16` tree, so this is not a
channel-truncation test.

The exact gate succeeds.  On both square and triangular `2x4`, the full
coefficient state and random parameter tangents have (S^2) and global-
rotation residuals at roughly (10^{-15}).  Seven distinct raw routes retain
rank five after translation projection.  This establishes a valid exact-SU(2)
architecture, not an energy or efficiency win.

The paired `4x4` test is negative.  At 100 updates the physical route model
wins only `1/3` square and `3/3` triangular same-step pairs, with median paired
error ratios `1.0348` and `0.9936`, while costing about eight route contractions
per configuration.  An 800-update single tree uses the same nominal
contraction count and is orders of magnitude more accurate.  Against the one
preregistered descriptor-mismatched bank, physical routes win only `1/3`
square and `2/3` triangular pairs; one bank is sufficient to fail promotion,
not to characterize random-route statistics.  A frozen-backbone Ritz audit
finds only small coherent headroom, and the two locality features recover a
minority of it.  The global route-mixture architecture is therefore frozen
rather than scaled to GPU.
"""
        ),
        code(
            r"""
arm_style = {
    "single_tree": ("o", "single tree, 100"),
    "spatial_cg": ("s", "physical Spatial-CG, 100"),
    "scrambled_spatial_cg": ("x", "scrambled Spatial-CG, 100"),
    "single_tree_work_control": ("D", "single tree, 800"),
}
geometry_color = {"square": "#0072b2", "triangular": "#d55e00"}
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.4))

for case in spatial_cg["cases"]:
    geometry = case["geometry"]
    for arm_id, (marker, label) in arm_style.items():
        trials = [row for row in case["trials"] if row["arm_id"] == arm_id]
        x = np.asarray([
            row["work_ledger"]["total_route_tree_contractions_over_updates"]
            for row in trials
        ], dtype=float)
        y = np.asarray([
            row["metrics"]["energy_error_per_site"] for row in trials
        ], dtype=float)
        axes[0].scatter(
            x, y, marker=marker, s=34, alpha=0.35,
            color=geometry_color[geometry],
        )
        axes[0].scatter(
            np.median(x), np.median(y), marker=marker, s=82,
            edgecolor="black" if marker not in ("x", "+") else None,
            linewidth=0.7, color=geometry_color[geometry],
            label=f"{geometry}: {label}",
        )
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("nominal raw tree-configuration contractions")
axes[0].set_ylabel(r"energy error per site")
axes[0].set_title("4x4 nominal-work negative gate")
axes[0].legend(frameon=False, fontsize=7, ncol=2)

categories = (
    "2-feature\nquery",
    "full-span\nRitz",
    "physical-query\nadvantage",
    "physical-Ritz\nadvantage",
)
for offset, case in zip((-0.08, 0.08), spatial_cg_capacity["cases"]):
    geometry = case["geometry"]
    n_sites = 16
    values = []
    for row in case["trials"]:
        physical = row["physical_routes"]
        values.append([
            physical["two_feature_gain_below_reference"] / n_sites,
            physical["unconstrained_ritz_gain_below_reference"] / n_sites,
            -row["physical_minus_scrambled_two_feature_energy"] / n_sites,
            -row["physical_minus_scrambled_ritz_energy"] / n_sites,
        ])
    values = np.asarray(values)
    for row in values:
        axes[1].plot(
            np.arange(4) + offset, row, color=geometry_color[geometry],
            alpha=0.25, lw=0.9, marker="o", ms=3,
        )
    axes[1].plot(
        np.arange(4) + offset, np.median(values, axis=0),
        color=geometry_color[geometry], lw=2.0, marker="o", ms=7,
        label=geometry,
    )
axes[1].axhline(0.0, color="0.25", linestyle=":", lw=1.0)
axes[1].set_xticks(np.arange(4), categories)
axes[1].set_ylabel("energy gain per site (positive = physical wins)")
axes[1].set_title("Frozen-backbone route capacity")
axes[1].legend(frameon=False)

fig.suptitle("Support-aware global route resonance is exact but not compute competitive")
fig.tight_layout()
save_figure(fig, "fig15_spin_spatial_cg_kill_gate")
plt.show()

print("N=8 exact gates:")
for geometry, report in spatial_cg["exact_small_oracles"].items():
    print(geometry, {
        "S2": report["spin_squared_residual_ratio"],
        "tangent_S2": report["maximum_tangent_spin_squared_residual_ratio"],
        "projected_rank": report["translation_projected_component_span"]["rank_at_1e_minus_10"],
    })
print("\n4x4 pilot summary:")
for case_id, summary in spatial_cg["summary"].items():
    print(case_id, summary)
print("\nFrozen-backbone capacity summary:")
for case_id, summary in spatial_cg_capacity["summary"].items():
    print(case_id, summary)
"""
        ),
        markdown(
            r"""
## 13. Fixed-$S^z$ factored Spatial-ViT: a validation frontier, not a production claim

This is the first conventional spatial-attention control on the same frustrated
`4x4` translation sectors used above.  It uses `2x2` patches (only four
tokens), one layer, widths `d=6--12`, three seeds, and exact-support Adam.  It
is therefore a **validation-scale fixed-$S^z$ baseline**: it is not a
reproduction of the published `20x20` ViT, a scaling result, or evidence for a
spin-liquid phase.

The left column places every seed and the median capacity trend beside the
`D=2--4` portion of the already sealed translation-sector ClebschTree
campaigns, which brackets the ViT parameter range without letting near-floor
large-tree outliers set the visual scale.  The latter use the same 300-update
learning rate protocol but different seed ensembles and topologies, so they
are frontier context rather than a matched-backbone causal comparison.  At
`d=8`, the distance envelope improves the square median over the equal-capacity
flat kernel (`2/3` seed wins), but not the triangular median (`1/3` seed wins).
Increasing width is not monotone: the `d=12` runs collapse under this optimizer
on both geometries.

The right column diagnoses that collapse without appealing to an algorithmic
surrogate.  It plots energy error against the physical spin contamination
`1-w_{S=0}`.  The best `d=8` states have singlet weights near `0.99`, whereas
the unstable large-width states retain only about half their weight in the
singlet sector.  Singlet projection explains only part of the remaining
energy error, however, so exact SU(2) is a useful prior but not by itself the
missing spatial representation or optimizer.
"""
        ),
        code(
            r"""
vit_case_ids = {
    "square": "square_4x4_j2_0p5_fixed_sz_spatial_vit",
    "triangular": "triangular_4x4_j2_0p125_fixed_sz_spatial_vit",
}
tree_case_ids = {
    "square": "square_4x4_j2_0p5_prl_pareto",
    "triangular": "triangular_4x4_j2_0p125_prl_pareto",
}
geometry_titles = {
    "square": r"square 4x4, $J_2/J_1=0.50$",
    "triangular": r"triangular 4x4, $J_2/J_1=0.125$",
}
spatial_vit_arms = [
    "spatial_vit_d6_l1",
    "spatial_vit_d8_l1",
    "spatial_vit_d10_l1",
    "spatial_vit_d12_l1_saturation",
]
spatial_width_colors = {
    6: "#56b4e9",
    8: "#0072b2",
    10: "#009e73",
    12: "#cc79a7",
}
spatial_label_offsets = {
    6: (5, 3),
    8: (-23, -13),
    10: (7, 4),
    12: (5, 3),
}
frontier_styles = {
    "u1": {"color": "#777777", "marker": "^", "linestyle": "--",
           "label": r"ClebschTree, fixed $M=0$ (sealed context)"},
    "su2": {"color": "#d55e00", "marker": "o", "linestyle": "-",
            "label": r"ClebschTree, exact $S=0$ (sealed context)"},
}

fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.1))
for row_index, geometry in enumerate(("square", "triangular")):
    frontier_ax, spin_ax = axes[row_index]
    tree_case = case_by_id(prl_pareto, tree_case_ids[geometry])
    vit_case = case_by_id(spatial_vit, vit_case_ids[geometry])

    # Sealed ClebschTree translation-sector context: eight seeds per point.
    # D2--D4 brackets the ViT range; D5--D6 near-floor outliers are already
    # shown in Fig. 8 and would obscure this validation-scale comparison.
    for backbone, style in frontier_styles.items():
        counts, medians = [], []
        for width in range(2, 5):
            arm_id = f"{backbone}_D{width}_translation"
            trials = [row for row in tree_case["trials"] if row["arm_id"] == arm_id]
            errors = assert_variational_and_floor(
                [row["metrics"]["energy_error_per_site"] for row in trials]
            )
            count = int(trials[0]["parameter_count"])
            seed_x = count * np.exp(np.linspace(-0.018, 0.018, len(errors)))
            frontier_ax.scatter(
                seed_x, errors, s=14, marker=style["marker"],
                color=style["color"], alpha=0.16, linewidths=0.5,
            )
            counts.append(count)
            medians.append(float(np.median(errors)))
        frontier_ax.plot(
            counts, assert_variational_and_floor(medians),
            marker=style["marker"], ms=5.8, lw=1.7,
            linestyle=style["linestyle"], color=style["color"],
            label=style["label"],
        )

    # Fixed-Sz Spatial-ViT capacity line and same-capacity flat-kernel control.
    vit_counts, vit_medians = [], []
    spin_median_x, spin_median_y = [], []
    for arm_id in spatial_vit_arms:
        trials = [row for row in vit_case["trials"] if row["arm_id"] == arm_id]
        width = int(trials[0]["embedding_dim"])
        count = int(trials[0]["parameter_count"])
        errors = assert_variational_and_floor(
            [row["metrics"]["energy_error_per_site"] for row in trials]
        )
        contamination = np.maximum(
            1.0 - np.asarray([
                row["spin_sector_diagnostics"]["singlet_weight"] for row in trials
            ], dtype=float),
            1.0e-12,
        )
        seed_x = count * np.exp(np.linspace(-0.012, 0.012, len(errors)))
        frontier_ax.scatter(
            seed_x, errors, s=25, marker="s",
            color=spatial_width_colors[width], alpha=0.42, linewidths=0,
        )
        spin_ax.scatter(
            contamination, errors, s=27, marker="s",
            color=spatial_width_colors[width], alpha=0.42, linewidths=0,
        )
        median_error = float(np.median(errors))
        median_contamination = float(np.median(contamination))
        vit_counts.append(count)
        vit_medians.append(median_error)
        spin_median_x.append(median_contamination)
        spin_median_y.append(median_error)
        spin_ax.scatter(
            median_contamination, median_error, s=72, marker="s",
            color=spatial_width_colors[width], edgecolor="black", linewidth=0.65,
            zorder=4,
        )
        spin_ax.annotate(
            f"d{width}", (median_contamination, median_error),
            xytext=spatial_label_offsets[width],
            textcoords="offset points", fontsize=8,
        )
    frontier_ax.plot(
        vit_counts, assert_variational_and_floor(vit_medians),
        color="#0072b2", marker="s", ms=6, lw=1.9,
        label=r"fixed-$S^z$ Spatial-ViT (3-seed median)",
    )
    spin_ax.plot(
        spin_median_x, assert_variational_and_floor(spin_median_y),
        color="#0072b2", lw=1.25, alpha=0.75,
    )

    flat_trials = [
        row for row in vit_case["trials"] if row["arm_id"] == "flat_vit_d8_l1"
    ]
    flat_errors = assert_variational_and_floor(
        [row["metrics"]["energy_error_per_site"] for row in flat_trials]
    )
    flat_count = int(flat_trials[0]["parameter_count"])
    flat_contamination = np.maximum(
        1.0 - np.asarray([
            row["spin_sector_diagnostics"]["singlet_weight"] for row in flat_trials
        ], dtype=float),
        1.0e-12,
    )
    frontier_ax.scatter(
        flat_count * np.exp(np.linspace(-0.012, 0.012, len(flat_errors))),
        flat_errors, s=30, marker="X", color="#000000", alpha=0.4,
    )
    frontier_ax.scatter(
        flat_count, np.median(flat_errors), s=74, marker="X",
        color="#000000", label=r"fixed-$S^z$ flat-kernel ViT, d8",
    )
    spin_ax.scatter(
        flat_contamination, flat_errors, s=33, marker="X",
        color="#000000", alpha=0.38,
    )
    spin_ax.scatter(
        np.median(flat_contamination), np.median(flat_errors),
        s=78, marker="X", color="#000000", zorder=4,
    )
    spin_ax.annotate(
        "flat d8",
        (np.median(flat_contamination), np.median(flat_errors)),
        xytext=(8, -15), textcoords="offset points", fontsize=8,
    )

    paired = spatial_vit["summary"][vit_case["case_id"]][
        "paired_spatial_vs_flat"
    ][0]
    frontier_ax.text(
        0.97, 0.96,
        f"spatial d8 wins {paired['spatial_energy_win_count']}/"
        f"{paired['seed_count']} seeds\n"
        f"median spatial/flat = {paired['median_paired_energy_error_ratio']:.3f}",
        transform=frontier_ax.transAxes, ha="right", va="top", fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
    )
    frontier_ax.set_xscale("log")
    frontier_ax.set_yscale("log")
    frontier_ax.set_title(geometry_titles[geometry] + ": parameter frontier")
    frontier_ax.set_ylabel(r"final energy error per site $\Delta E/N$")
    spin_ax.set_xscale("log")
    spin_ax.set_yscale("log")
    spin_ax.set_title(geometry_titles[geometry] + ": spin-sector diagnosis")
    spin_ax.set_ylabel(r"final energy error per site $\Delta E/N$")
    if row_index == 1:
        frontier_ax.set_xlabel("raw trainable parameters")
        spin_ax.set_xlabel(r"spin contamination $1-w_{S=0}$")

handles, labels = axes[0, 0].get_legend_handles_labels()
unique_legend = dict(zip(labels, handles))
fig.legend(
    unique_legend.values(), unique_legend.keys(),
    loc="lower center", ncol=2, frameon=False, fontsize=8,
)
fig.suptitle(
    "Validation-scale fixed-Sz Spatial-ViT: local capacity and physical spin diagnosis"
)
fig.tight_layout(rect=(0, 0.085, 1, 0.96))
save_figure(fig, "fig16_fixed_sz_spatial_vit_frontier")
plt.show()

print("Fixed-Sz Spatial-ViT claim boundary:", spatial_vit["claim_boundary"])
for geometry in ("square", "triangular"):
    case_id = vit_case_ids[geometry]
    paired = spatial_vit["summary"][case_id]["paired_spatial_vs_flat"][0]
    spatial_d8 = spatial_vit["summary"][case_id]["arms"]["spatial_vit_d8_l1"]
    print(geometry, {
        "spatial_d8_median_error_per_site": spatial_d8["median_energy_error_per_site"],
        "spatial_d8_median_singlet_weight": spatial_d8["median_singlet_weight"],
        "paired_spatial_over_flat": paired["median_paired_energy_error_ratio"],
        "paired_seed_wins": paired["spatial_energy_win_count"],
})
"""
        ),
        markdown(
            r"""
## 14. Six-site singlet core: exact two-cluster mechanism diagnostic

This section asks a narrower physical question than the 6x6 optimization
campaign: **is the isolated \(2\times3\) singlet a sound local core, and which
cluster-spin sectors repair the energy when two cores are coupled?**  The
answer is obtained by exact diagonalization of one **open \(2\times6\)**
two-cluster problem.  A separate translated-covering calculation on a periodic
\(2\times6\) thin torus diagnoses \(q=0\) resonance.  These finite-system
mechanisms are not an N=36 result, a thermodynamic statement, or a variational
energy/scaling win.

The isolated six-site ground singlet is used without approximation.  At full
inter-cluster coupling the exact ground state keeps about 79--80% weight in
that local core, while almost all of the missing sector weight is an
equal-\(q=2\) (\(S=1\) on each cluster) virtual pair.  The \(q\ge4\) weight is
tiny.  Crucially, the energy-lowest triplet is not the best compressed
carrier: boundary-response and resolvent-dressed carriers recover much more
of the coupling gap at the same rank.  This motivates support-aware low-spin
carriers rather than a generic enlargement of the network.  A separate
periodic \(6\times6\) bond-ledger audit asks whether the same carrier can be
shared across the six translations of one fixed \(2\times3\) orientation.  It
does not construct or diagonalize an N=36 state.
"""
        ),
        code(
            r"""
from IPython.display import Markdown, display

mechanism_cases = {
    case["geometry"]: case for case in two_cluster_mechanism["cases"]
}
geometry_labels = {"square": "square J2/J1=0.5", "triangular": "triangular J2/J1=0.125"}

# A compact table of quantities that answer the physical question directly.
table = [
    "| geometry | local core probability | exact q=2 pair weight | q>=4 weight | "
    "2nd-order gap recovery | dressed k=1 | dressed k=2 | complete q=2 | q=0 covering resonance |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for geometry in ("square", "triangular"):
    case = mechanism_cases[geometry]
    virtual = case["virtual_pair"]
    coupled = virtual["open_full_coupling"]
    weights = dict(zip(
        coupled["cluster_pair_sector_q_values"],
        coupled["equal_q_sector_weights"],
    ))
    carriers = {
        row["carrier"]: row for row in coupled["carrier_results"]
    }
    resonance = case["covering_resonance"]["all_distinct_translations"]
    table.append(
        "| {label} | {core:.2f}% | {q2:.2f}% | {high:.3g}% | {e2:.2f}% | "
        "{d1:.2f}% | {d2:.2f}% | {allq2:.2f}% | {q0:.2f}% |".format(
            label=geometry_labels[geometry],
            core=100.0 * 0.5 * (
                coupled["left_cluster_core_probability"]
                + coupled["right_cluster_core_probability"]
            ),
            q2=100.0 * weights[2],
            high=100.0 * sum(weights[q] for q in (4, 6)),
            e2=100.0 * coupled["second_order_fraction_of_exact_coupling_gain"],
            d1=100.0 * carriers[
                "resolvent_dressed_schmidt_k1"
            ]["recovered_exact_gap_fraction"],
            d2=100.0 * carriers[
                "resolvent_dressed_schmidt_k2"
            ]["recovered_exact_gap_fraction"],
            allq2=100.0 * carriers[
                "complete_q2_pair_space"
            ]["recovered_exact_gap_fraction"],
            q0=100.0 * resonance["recovered_exact_gap_fraction"],
        )
    )
display(Markdown("\n".join(table)))

# Translation/interface audit.  The direction-separated route number omits
# cross-direction interference; the physical-pair ledger first sums every
# bond joining the same two blocks.  Shared-bank counts are prefix lengths in
# the reported joint-covariance ordering, not global subspace optima.
orientation_table = [
    "| geometry | per-route k=1 E2 | first-mode union rank | "
    "direction-shared rank @ 90/99/99.9% | "
    "physical-pair rank @ 90/99/99.9% |",
    "|---|---:|---:|:---:|:---:|",
]
threshold_keys = (
    "90_percent_second_order_energy",
    "99_percent_second_order_energy",
    "99p9_percent_second_order_energy",
)
for geometry in ("square", "triangular"):
    support = mechanism_cases[geometry]["support_orientation"]
    direction_union = support["direction_resolved"]["union"]
    pair_union = support["physical_pair_total"]["union"]

    def threshold_triplet(union):
        return "/".join(
            str(union["minimum_shared_carriers"][key])
            for key in threshold_keys
        )

    orientation_table.append(
        "| {label} | {route:.2f}% | {rank} | {direction} | {pair} |".format(
            label=geometry_labels[geometry],
            route=100.0
            * direction_union["route_conditioned_second_order_energy_fractions"][0],
            rank=direction_union["route_k_union_ranks"][0],
            direction=threshold_triplet(direction_union),
            pair=threshold_triplet(pair_union),
        )
    )
display(Markdown("\n".join(orientation_table)))

family_style = {
    "lowest": {"marker": "X", "linestyle": "None", "label": "lowest-energy triplet"},
    "raw": {"marker": "o", "linestyle": "-", "label": "boundary-response SVD"},
    "dressed": {"marker": "s", "linestyle": "-", "label": "resolvent-dressed SVD"},
    "complete": {"marker": "*", "linestyle": "None", "label": "complete q=2 space"},
}
carrier_families = {
    "lowest": ["lowest_triplet_pair"],
    "raw": ["raw_response_schmidt_k1", "raw_response_schmidt_k2"],
    "dressed": ["resolvent_dressed_schmidt_k1", "resolvent_dressed_schmidt_k2"],
    "complete": ["complete_q2_pair_space"],
}

fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2))
for ax, geometry in zip(axes[0], ("square", "triangular")):
    coupled = mechanism_cases[geometry]["virtual_pair"]["open_full_coupling"]
    carriers = {row["carrier"]: row for row in coupled["carrier_results"]}
    for family, names in carrier_families.items():
        rows = [carriers[name] for name in names]
        ax.plot(
            [row["left_carrier_count"] for row in rows],
            [100.0 * row["recovered_exact_gap_fraction"] for row in rows],
            color={"lowest": "0.45", "raw": "tab:blue", "dressed": "tab:orange", "complete": "tab:green"}[family],
            markersize=7 if family != "complete" else 10,
            linewidth=1.6,
            **family_style[family],
        )
    ax.axhline(100.0, color="0.35", linestyle=":", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 9], labels=["1", "2", "9 (all)"])
    ax.set_ylim(0.0, 104.0)
    ax.set_xlabel("q=2 carrier rank per cluster")
    ax.set_title(geometry_labels[geometry])
axes[0, 0].set_ylabel("full-coupling gap recovered (%)")

q_ax = axes[1, 0]
for geometry, marker in (("square", "o"), ("triangular", "s")):
    coupled = mechanism_cases[geometry]["virtual_pair"]["open_full_coupling"]
    q_ax.plot(
        coupled["cluster_pair_sector_q_values"],
        coupled["equal_q_sector_weights"],
        marker=marker,
        linewidth=1.7,
        label=geometry_labels[geometry],
    )
q_ax.set_yscale("log")
q_ax.set_xticks([0, 2, 4, 6])
q_ax.set_xlabel("cluster doubled spin q")
q_ax.set_ylabel("exact equal-q pair weight")
q_ax.set_title("open 2x6 exact ground state")
q_ax.legend(frameon=False, fontsize=8)

orientation_ax = axes[1, 1]
orientation_styles = {
    "direction_resolved": ("-", "o", "direction-separated shared bank"),
    "physical_pair_total": ("--", "s", "physical-pair shared bank"),
}
for geometry, color in (("square", "tab:blue"), ("triangular", "tab:orange")):
    support = mechanism_cases[geometry]["support_orientation"]
    for family, (linestyle, marker, label) in orientation_styles.items():
        union = support[family]["union"]
        fractions = 100.0 * np.asarray(
            union["shared_bank_second_order_energy_fractions"], dtype=float
        )
        orientation_ax.plot(
            np.arange(1, len(fractions) + 1), fractions,
            color=color, linestyle=linestyle, marker=marker, markersize=4,
            linewidth=1.5, label=f"{geometry}: {label}",
        )
    route_k1 = 100.0 * support["direction_resolved"]["union"][
        "route_conditioned_second_order_energy_fractions"
    ][0]
    orientation_ax.scatter(
        [1], [route_k1], marker="*", s=90, color=color, edgecolor="white",
        linewidth=0.6, zorder=5, label=f"{geometry}: per-route k=1",
    )
orientation_ax.axhline(90.0, color="0.45", linestyle=":", linewidth=1.0)
orientation_ax.axhline(99.0, color="0.65", linestyle=":", linewidth=0.8)
orientation_ax.set_xlim(0.75, 9.25)
orientation_ax.set_ylim(45.0, 101.0)
orientation_ax.set_xticks([1, 3, 5, 7, 9])
orientation_ax.set_xlabel("shared q=2 carrier count")
orientation_ax.set_ylabel(r"perturbative $E^{(2)}$ recovered (%)")
orientation_ax.set_title(r"fixed-$2\times3$ translation/interface audit")
orientation_ax.legend(
    frameon=True, framealpha=0.92, edgecolor="none",
    fontsize=6.5, ncol=1, loc="lower right",
)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8
)
fig.suptitle("Six-site singlet core: low-spin virtual carriers and orientation cost")
fig.tight_layout(rect=(0, 0.07, 1, 0.95))
save_figure(fig, "fig17_two_cluster_singlet_core_mechanism")
plt.show()

print("Finite-system claim boundary:", two_cluster_mechanism["claim_boundary"])
"""
        ),
        markdown(
            r"""
The open-cut result is a positive mechanism test: the complete \(q=2\)-pair
space recovers 97.48% (square) and 98.29% (triangular) of the exact
full-coupling energy gain, while \(q\ge4\) carries less than \(7\times10^{-5}\)
of the norm.  A rank-one resolvent carrier already recovers 84.25% and 75.37%,
far above the 20.82% and 12.37% obtained from the lowest-energy triplet alone.
The periodic translated-core span separately recovers 24.20% and 29.54% of its
gap, establishing a distinct \(q=0\) resonance channel.  These percentages
must not be added: they come from different exact diagnostics.

The orientation audit sharpens the architectural conclusion.  A separately
chosen leading carrier for each direction captures 92.26% (square) and 92.50%
(triangular) of the direction-separated second-order energy, but the union of
those first modes already has rank nine.  One shared direction bank needs
3/5/5 carriers on the square lattice and 4/5/5 on the triangular lattice to
cross 90/99/99.9%; after all bonds joining each physical block pair are summed,
both need 4/5/5.  Thus a single energy-lowest or globally shared carrier is not
the physical compression: routing by interface/support is essential.  The
audit covers only translations of a fixed \(2\times3\) orientation, omits the
rotated \(3\times2\) covering and point-group projection, and performs no N=36
ED or trained-state test.
"""
        ),
        markdown(
            r"""
## 15. N=36 low-spin directional pilot — not a mechanism proof

The exact two-cluster result motivates a deliberately small \(6\times6\)
directional test.  The same frozen local multiplet backbone is probed by four
one-step arms: A0 is the support-floor baseline, A1 releases \(q=0\), A2
releases the disjoint \(q=2\) increment, and A12 releases both.  Each active
arm is proposed from the same training sample, checked with a paired held-out
full-\(H\) estimator, and measured once more on an independent proposal-Born
audit stream.

The dotted lines below are only visual guides.  In the upper panels, different
arms use independent audit streams, and even rejected proposals are plotted;
their error bars are across-chain MCSEs.  The lower panels show the paired
held-out energy shift divided by its reported standard error, with a one-SE
error bar.  A filled green point means the preregistered paired gate accepted
the proposal; a red cross means rollback.  Neither marker is a convergence or
mechanism verdict.
"""
        ),
        code(
            r"""
from IPython.display import Markdown, display
from matplotlib.lines import Line2D

low_spin_cases = {
    geometry: low_spin_local_pilots[geometry]["cases"][0]
    for geometry in ("square", "triangular")
}
arm_order = ("A0_frozen", "A1_q0", "A2_q2_increment", "A12_q0_q2")
arm_labels = ("A0", r"A1: $q=0$", r"A2: $q=2$", r"A12: $q=0,2$")
geometry_colors = {"square": "tab:blue", "triangular": "tab:orange"}
geometry_titles = {
    "square": r"square $6\times6$, $J_2/J_1=0.5$",
    "triangular": r"triangular $6\times6$, $J_2/J_1=0.125$",
}

fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0), sharex="col")
for column, geometry in enumerate(("square", "triangular")):
    case = low_spin_cases[geometry]
    arms = {row["arm_id"]: row for row in case["arms"]}
    rows = [arms[name] for name in arm_order]
    x = np.arange(len(rows), dtype=float)
    color = geometry_colors[geometry]

    energy = np.asarray(
        [row["physical_audit"]["energy_per_site"] for row in rows], dtype=float
    )
    energy_mcse = np.asarray(
        [row["physical_audit"]["energy_per_site_chain_mcse"] for row in rows],
        dtype=float,
    )
    energy_ax = axes[0, column]
    energy_ax.plot(x, energy, linestyle=":", linewidth=1.3, color=color, alpha=0.8)
    energy_ax.errorbar(
        x, energy, yerr=energy_mcse, fmt="none", color=color,
        capsize=3.0, elinewidth=1.2,
    )
    energy_ax.scatter(
        [x[0]], [energy[0]], marker="D", s=46, color="0.35", zorder=4
    )
    for position, row, value in zip(x[1:], rows[1:], energy[1:], strict=True):
        if row["strictly_accepted"]:
            energy_ax.scatter(
                [position], [value], marker="o", s=70, facecolor="#2ca02c",
                edgecolor="white", linewidth=0.8, zorder=5,
            )
        else:
            energy_ax.scatter(
                [position], [value], marker="X", s=70, color="#c23b22", zorder=5
            )
    energy_ax.set_title(geometry_titles[geometry])
    energy_ax.set_ylabel(r"independent proposal audit $E/N$")

    active_rows = rows[1:]
    active_x = x[1:]
    paired_z = np.asarray([
        row["independent_full_h_confirmation"]["comparison"]["energy_difference"]
        / row["independent_full_h_confirmation"]["comparison"]["standard_error"]
        for row in active_rows
    ])
    paired_ax = axes[1, column]
    paired_ax.axhspan(-1.0, 1.0, color="0.92", zorder=0)
    paired_ax.axhline(0.0, color="0.35", linewidth=1.0)
    paired_ax.plot(
        active_x, paired_z, linestyle=":", linewidth=1.3, color=color, alpha=0.8
    )
    paired_ax.errorbar(
        active_x, paired_z, yerr=np.ones_like(paired_z), fmt="none",
        color=color, capsize=3.0, elinewidth=1.2,
    )
    for position, row, value in zip(active_x, active_rows, paired_z, strict=True):
        if row["strictly_accepted"]:
            paired_ax.scatter(
                [position], [value], marker="o", s=70, facecolor="#2ca02c",
                edgecolor="white", linewidth=0.8, zorder=5,
            )
        else:
            paired_ax.scatter(
                [position], [value], marker="X", s=70, color="#c23b22", zorder=5
            )
    paired_ax.set_ylabel(r"paired held-out $\Delta E/\mathrm{SE}$")
    paired_ax.set_xticks(x, arm_labels)
    paired_ax.tick_params(axis="x", rotation=18)
    paired_ax.text(
        0.02, 0.04, "negative favors proposal", transform=paired_ax.transAxes,
        fontsize=8, color="0.35",
    )

    if geometry == "square":
        a2 = arms["A2_q2_increment"]
        cosine = a2["split_half"]["gradient_cosine"]
        a2_z = paired_z[1]
        paired_ax.annotate(
            f"split-gradient cosine = {cosine:.3f}\nreversal: directional only",
            xy=(2.0, a2_z), xytext=(0.34, 0.30),
            textcoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": "0.3", "lw": 0.8},
            bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "none"},
            fontsize=8, ha="left", va="bottom",
        )

legend_handles = [
    Line2D([0], [0], marker="D", linestyle="None", color="0.35", label="A0 baseline"),
    Line2D(
        [0], [0], marker="o", linestyle="None", markerfacecolor="#2ca02c",
        markeredgecolor="white", color="#2ca02c", label="paired gate accepted",
    ),
    Line2D(
        [0], [0], marker="X", linestyle="None", color="#c23b22",
        label="paired gate rejected / rollback",
    ),
]
fig.legend(
    handles=legend_handles, loc="lower center", ncol=3, frameon=False, fontsize=8
)
fig.suptitle("N=36 low-spin directional pilot — not a mechanism proof")
fig.tight_layout(rect=(0, 0.08, 1, 0.95))
save_figure(fig, "fig18_n36_low_spin_directional_pilot")
plt.show()

# The q=2 arm is the only arm accepted by the paired held-out gate in both
# geometries.  Its physical channel changes remain unresolved on this local
# sample budget, so report estimates and MCSEs rather than signs alone.
pilot_table = [
    "| geometry | paired ΔE/SE | decision | split ∇E cosine | "
    "ΔP(S=1) | ΔEintra/N | ΔEinter/N |",
    "|---|---:|:---:|---:|---:|---:|---:|",
]
for geometry in ("square", "triangular"):
    a2 = {
        row["arm_id"]: row for row in low_spin_cases[geometry]["arms"]
    }["A2_q2_increment"]
    comparison = a2["independent_full_h_confirmation"]["comparison"]
    change = a2["physical_change_from_A0_with_independent_chain_uncertainty"]

    def estimate(name):
        row = change[name]
        return (
            f"{row['difference']:+.4f} ± "
            f"{row['independent_stream_combined_mcse']:.4f}"
        )

    pilot_table.append(
        "| {geometry} | {paired:+.2f} | {decision} | {cosine:+.3f} | "
        "{p1} | {intra} | {inter} |".format(
            geometry=geometry,
            paired=comparison["energy_difference"] / comparison["standard_error"],
            decision="accepted" if a2["strictly_accepted"] else "rollback",
            cosine=a2["split_half"]["gradient_cosine"],
            p1=estimate("P_S1"),
            intra=estimate("intra_energy_per_site"),
            inter=estimate("inter_energy_per_site"),
        )
    )
display(Markdown("\n".join(pilot_table)))

print("Local-pilot claim boundaries:")
for geometry in ("square", "triangular"):
    print(geometry, low_spin_local_pilots[geometry]["claim_boundary"])
"""
        ),
        markdown(
            r"""
The only common directional signal is that A2 passes the paired held-out
energy gate in both geometries.  That is not yet a physical mechanism result.
On the square lattice its two half-sample gradients are nearly antiparallel,
and the independent audit does not reproduce the paired energy decrease.  On
the triangular lattice the independent energy shift is more favorable, but
the changes in triplet weight and intra/inter-cluster energy remain unresolved
and do not isolate the perturbative virtual-triplet signature.  A1 and A12
roll back under the paired gate.  These local pilots therefore motivate the
support-aware response-carrier experiment; they establish neither an energy
win nor a QSL, phase boundary, or phase diagram.
"""
        ),
        markdown(
            r"""
## 16. Exact-SU(2) Route-ViT: frustration-selective reduced-tensor resonance

The new hybrid does not apply an ordinary nonlinear ViT to magnetic CG
indices.  A route-token Transformer reads multiscale J1/J2 locality
histograms and generates coherent route weights plus low-rank reduced-tensor
modulations.  Every component and tangent therefore remains an exact
singlet; translations are projected coherently in the physical basis.

The paired plot starts all D=4 arms from the same 100-update warm state and
then spends 120 additional updates.  The triangular conditioned model lowers
the energy error in all three seeds, whereas coefficient-only mixing wins no
seed.  Square is a negative control.  The result identifies a candidate
frustration-selective mechanism, not a converged or large-system win.
"""
        ),
        code(
            r"""
route_arm_order = [
    "single_tree_d4",
    "route_vit_coefficients_d4",
    "route_vit_conditioned_d4",
    "single_tree_matched_width",
]
route_arm_labels = ["D4 tree", "ViT weights", "ViT reduced", "D6 tree"]
route_case_ids = {
    "square": "square_4x4_j2_0p5_su2_route_vit",
    "triangular": "triangular_4x4_j2_0p125_su2_route_vit",
}

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
for ax, geometry in zip(axes, ("square", "triangular"), strict=True):
    case = next(row for row in su2_route_vit["cases"] if row["case_id"] == route_case_ids[geometry])
    for trial in case["trials"]:
        by_arm = {row["arm_id"]: row for row in trial["arms"]}
        errors = [by_arm[arm]["metrics"]["energy_error_per_site"] for arm in route_arm_order]
        ax.semilogy(range(len(errors)), errors, "o-", lw=1.25, ms=4.5,
                    alpha=.78, label=f"seed {trial['seed']}")
    ax.set_xticks(range(len(route_arm_labels)), route_arm_labels, rotation=25, ha="right")
    ax.set_title(f"{geometry} 4x4")
    ax.set_ylabel(r"$(E-E_0)/N$")
    ax.legend(frameon=False, fontsize=8)
fig.suptitle("Exact-SU(2) Route-ViT: paired finite-work variational gate")
fig.tight_layout()
save_figure(fig, "fig19_su2_route_vit_variational_gate")
plt.show()

for geometry, case_id in route_case_ids.items():
    paired = su2_route_vit["summary"][case_id]["paired_vs_d4"]
    print(geometry, "conditioned:", paired["route_vit_conditioned_d4"])
print("Long triangular energies:", {
    arm: values["median_energy"]
    for arm, values in su2_route_vit_long["summary"][
        "triangular_4x4_j2_0p125_su2_route_vit_long"
    ]["arms"].items()
})
print("N36 readiness:", su2_route_vit_n36["runtime"])
print("N36 untrained energy:", su2_route_vit_n36["singlet_diagonal_energy"])
"""
        ),
        markdown(
            r"""
At 800 updates the conditioned state remains below its nested D4 tree but
above the parameter-matched D6 tree.  Thus the 4x4 final-energy gate is not a
win.  The triangular 6x6 smoke establishes only sampling/force readiness:
accepted nontrivial moves are 0.69--0.81 with the dimer/global proposal, while
the untrained energy is far above the ground-state scale.  The next decisive
panel must come from resumable 6x6 training with held-out full-H confirmation.
"""
        ),
        markdown(
            r"""
### Coupling-resolved causal controls and phase-boundary context

The original (J_2/J_1=0.125) pilot did not separate useful global route
communication from extra nonlinear parameters.  The following scan adds two
causal controls at exactly the same warmed backbone and update count:

- global attention with the physical route-token association deliberately
  scrambled; and
- the same attention parameters used route-locally, with no cross-route read.

The finite (4\times4) exact anchors are used only for the vertical energy
error.  Shaded coupling intervals are thermodynamic literature context, not
phase labels for this small torus.  The triangular interval combines the
intermediate windows reported by Zhu--White and Iqbal *et al.*; the square
interval follows the QSL/VBS window reported by Wang *et al.* and should be
read with the known method dependence of the square-lattice phase diagram.
The (D=5) triangular iPEPS series in panel (d) was supplied by the user; its
thermodynamic energy density is shown separately and is never substituted for
a finite-torus target.
"""
        ),
        code(
            r"""
def case_by_coupling(payload, geometry):
    return sorted(
        (case for case in payload["cases"] if case["geometry"] == geometry),
        key=lambda case: case["j2_over_j1"],
    )

scan_arm_order = [
    "single_tree_d4",
    "route_vit_conditioned_d4",
    "route_vit_scrambled_tokens_d4",
    "route_local_conditioned_d4",
    "single_tree_matched_width",
]
scan_arm_style = {
    "single_tree_d4": ("D4 tree", "#4c78a8", "o"),
    "route_vit_conditioned_d4": ("physical global", "#e45756", "s"),
    "route_vit_scrambled_tokens_d4": ("scrambled global", "#72b7b2", "^"),
    "route_local_conditioned_d4": ("physical local", "#f2cf5b", "D"),
    "single_tree_matched_width": ("D6 tree", "#b279a2", "v"),
}

fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.8))
for ax, geometry, interval in (
    (axes[0, 0], "triangular", (0.06, 0.17)),
    (axes[0, 1], "square", (0.45, 0.61)),
):
    cases = case_by_coupling(su2_route_vit_scan, geometry)
    x = np.asarray([case["j2_over_j1"] for case in cases])
    ax.axvspan(*interval, color="0.88", zorder=0,
               label="literature boundary region")
    for arm_id in scan_arm_order:
        label, color, marker = scan_arm_style[arm_id]
        y = []
        for case in cases:
            by_arm = {row["arm_id"]: row for row in case["trials"][0]["arms"]}
            y.append(by_arm[arm_id]["metrics"]["energy_error_per_site"])
        ax.semilogy(x, y, marker=marker, lw=1.45, ms=4.8,
                    color=color, label=label)
    ax.set_xlabel(r"$J_2/J_1$")
    ax.set_ylabel(r"finite-$4\times4$ $(E-E_0)/N$")
    ax.set_title(f"({ 'a' if geometry == 'triangular' else 'b' }) {geometry}: coupling response")
    ax.legend(frameon=False, fontsize=7.2, ncol=2)

tri_scan = next(
    case for case in su2_route_vit_scan["cases"]
    if case["geometry"] == "triangular"
    and np.isclose(case["j2_over_j1"], 0.075)
)
tri_rep = next(
    case for case in su2_route_vit_boundary["cases"]
    if case["geometry"] == "triangular"
)
paired_arms = [
    "single_tree_d4",
    "route_vit_conditioned_d4",
    "route_vit_scrambled_tokens_d4",
    "route_local_conditioned_d4",
    "single_tree_matched_width",
]
paired_labels = ["D4", "physical\nglobal", "scrambled\nglobal", "physical\nlocal", "D6"]
for trial in tri_scan["trials"] + tri_rep["trials"]:
    by_arm = {row["arm_id"]: row for row in trial["arms"]}
    y = [by_arm[arm]["metrics"]["energy_error_per_site"] for arm in paired_arms]
    axes[1, 0].semilogy(range(len(y)), y, "o-", lw=1.25, ms=4.5,
                        label=f"seed {trial['seed']}")
axes[1, 0].set_xticks(range(len(paired_labels)), paired_labels)
axes[1, 0].set_ylabel(r"finite-$4\times4$ $(E-E_0)/N$")
axes[1, 0].set_title(r"(c) triangular $J_2/J_1=0.075$: paired seeds")
axes[1, 0].legend(frameon=False, fontsize=8)

ipeps = route_vit_coupling_refs["triangular_user_ipeps_d5"]["points"]
axes[1, 1].plot(
    [row["j2_over_j1"] for row in ipeps],
    [row["energy_per_site"] for row in ipeps],
    "o-",
                color="#59a14f", lw=1.6, ms=4.8, label=r"$D=5$ iPEPS")
axes[1, 1].axvspan(0.06, 0.17, color="0.88", zorder=0,
                   label="literature intermediate window")
axes[1, 1].set_xlabel(r"$J_2/J_1$")
axes[1, 1].set_ylabel(r"thermodynamic $E/N$")
axes[1, 1].set_title("(d) triangular iPEPS context (not N=16 target)")
axes[1, 1].legend(frameon=False, fontsize=8)

fig.suptitle("Exact-SU(2) Route-ViT across frustrated couplings")
fig.tight_layout()
save_figure(fig, "fig20_su2_route_vit_coupling_response")
plt.show()
"""
        ),
        markdown(
            r"""
At triangular (J_2/J_1=0.075), the physical global model beats D4, D6,
scrambled-global, and route-local controls in all three seeds at the fixed
120-update comparison budget.  This is the cleanest local causal win: it lies
near the 120-AFM/intermediate boundary and is not explained by width alone or
by merely adding attention parameters.  At square (J_2/J_1=0.4), global
multi-route mixing beats D4 and D6 in all three seeds, but the scrambled token
control is lower in two seeds.  It therefore does not demonstrate that the
current square tokens encode the relevant Néel-to-nonmagnetic physics.  At
larger square (J_2), especially 0.6, the scrambled and wider-tree controls
can be much better, warning that the present route bank is poorly adapted to
stripe correlations.

Adding the missing third triangular nearest-neighbor dimer orientation costs
zero trainable parameters, but wins only (2/3,1/3,2/3) seeds at
(J_2/J_1=0.075,0.125,0.175).  The physically complete route bank is therefore
useful architecture evidence, not yet a robust variational result.
"""
        ),
        code(
            r"""
physical_points = {}
for payload in (su2_route_vit_physical_routes,
                su2_route_vit_physical_routes_replication):
    for case in payload["cases"]:
        comparison = payload["physical_route_comparisons"][case["case_id"]]
        j2 = float(case["j2_over_j1"])
        for row in comparison["per_seed"]:
            physical_points[(int(row["seed"]), j2)] = float(
                row["complete_minus_base_energy"]
            )

fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.7))
for seed, color in zip((31, 32, 33), ("#4c78a8", "#e45756", "#59a14f"), strict=True):
    points = sorted((j2, delta) for (s, j2), delta in physical_points.items() if s == seed)
    axes[0].plot([p[0] for p in points], [p[1] for p in points], "o-",
                 color=color, lw=1.25, ms=4.5, label=f"seed {seed}")
axes[0].axhline(0.0, color="black", lw=.8)
axes[0].set_xlabel(r"$J_2/J_1$")
axes[0].set_ylabel(r"$E_{9\ routes}-E_{8\ routes}$")
axes[0].set_title("(a) zero-parameter dimer completion")
axes[0].legend(frameon=False, fontsize=8)

row_arms = su2_route_vit_n36_row_replication["sampler_arms"]
complete_arm = next(
    arm for arm in su2_route_vit_n36_audit["sampler_arms"]
    if arm["edge_bank"] == "dimer_complete"
)
sampler_rows = row_arms + [complete_arm]
x = np.arange(len(sampler_rows))
y = np.asarray([row["energy_per_site"] for row in sampler_rows])
yerr = np.asarray([row["chain_mean_standard_error_total"] / 36.0 for row in sampler_rows])
axes[1].errorbar(x, y, yerr=yerr, fmt="o", capsize=3, color="#4c78a8")
axes[1].set_xticks(x, ["row A", "row B", "row C", "3-dir."])
axes[1].set_ylabel(r"untrained $6\times6$ $E/N$")
axes[1].set_title("(b) independent-chain basin audit")
for index, row in enumerate(sampler_rows):
    axes[1].annotate(
        f"ESS {row['effective_sample_size_fraction']:.2f}",
        (index, y[index]), xytext=(5, 8), textcoords="offset points",
        ha="left", fontsize=7,
    )

attempts = su2_route_vit_n36_audit["one_step_line_search"]["attempts"]
scales = np.asarray([row["step_scale"] for row in attempts])
deltas = np.asarray([row["candidate_minus_reference"] for row in attempts])
two_se = 2.0 * np.asarray([row["paired_chain_standard_error"] for row in attempts])
axes[2].errorbar(scales, deltas, yerr=two_se, fmt="o-", capsize=3,
                 color="#e45756", label=r"candidate-reference $\pm2$ SE")
axes[2].axhline(0.0, color="black", lw=.8)
axes[2].set_xscale("log")
axes[2].set_xticks(scales, ["3", "10", "30"])
axes[2].tick_params(axis="x", which="minor", labelbottom=False)
axes[2].set_xlabel("force step scale")
axes[2].set_ylabel(r"held-out $\Delta E$")
axes[2].set_title("(c) N=36 one-step rejection gate")
axes[2].legend(frameon=False, fontsize=8)

fig.suptitle("Physical route extension and the N=36 sampling blocker")
fig.tight_layout()
save_figure(fig, "fig21_su2_route_vit_physical_and_n36_audit")
plt.show()
"""
        ),
        markdown(
            r"""
The (6\times6) calculation now passes forward evaluation, exact-singlet
diagonal-force evaluation, a tiny full-(H) audit, and checksummed checkpoint
resume.  It does **not** pass a production sampling gate.  Three nominally
well-mixed row-proposal replicas have temporal ESS fractions (0.90,0.99,1.00)
but total energies spanning (0.766), while the three-orientation proposal
lands in another basin.  Temporal autocorrelation alone therefore misses
between-basin nonergodicity.  None of the three held-out force steps clears a
two-standard-error improvement gate.  Long local training would currently
turn sampler bias into an apparent variational gain, so the next computation
must first add replica exchange, independent proposals, or another explicit
global basin-mixing mechanism.
"""
        ),
        markdown(
            r"""
## 17. Where the 4x4 variational error lives

The common-state comparison below uses one Hamiltonian, one exact state, the
same translation projection in the objective, and three unrejected seeds.  It
does more than rank energies: each translation-trained state is decomposed
into its exact space-group-singlet component and the orthogonal lattice
sectors.  We then compare spin structure factors and reduced density matrices
on supports of 2, 3, 4, 6, and 8 sites.

This changes the interpretation of the Route-ViT result.  The physical global
route is the lowest-energy arm in all three seeds and has the best median
structure factor, but the decisive variable is the weight in the correct
\(\Gamma\)-\(A_1\) sector.  After normalizing that component, every arm is
already extremely close to the exact state.  On this torus the retained
\(S=0,\Gamma\)-\(A_1\) space has only 14 states, so direct Gamma-A1 training
saturates a tiny sector and cannot discriminate ansatz quality.
"""
        ),
        code(
            r"""
p1 = physics_interpretation["P1_P2_P5"]
translation_trials = [
    trial for trial in p1["trials"]
    if trial["training_projection"] == "translation"
]
physics_arm_order = [
    "single_tree_d4",
    "route_vit_coefficients_d4",
    "route_vit_conditioned_d4",
    "route_vit_scrambled_tokens_d4",
    "route_local_conditioned_d4",
    "single_tree_matched_width",
]
physics_arm_labels = [
    "D4", "coeff.", "physical\nglobal", "scrambled\nglobal",
    "physical\nlocal", "D6",
]
physics_arm_colors = {
    arm: color for arm, color in zip(
        physics_arm_order,
        ("#4c78a8", "#9d755d", "#e45756", "#72b7b2", "#f2cf5b", "#b279a2"),
        strict=True,
    )
}

fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.8))
for trial in translation_trials:
    by_arm = {row["arm_id"]: row for row in trial["arms"]}
    energy = [by_arm[arm]["metrics"]["energy_error_per_site"] for arm in physics_arm_order]
    sq_error = [by_arm[arm]["structure_factor_error"]["rms_error"] for arm in physics_arm_order]
    axes[0, 0].semilogy(range(len(energy)), energy, "o-", lw=1.25, ms=4.5,
                         label=f"seed {trial['seed']}")
    axes[0, 1].semilogy(range(len(sq_error)), sq_error, "o-", lw=1.25, ms=4.5,
                         label=f"seed {trial['seed']}")
axes[0, 0].set_ylabel(r"$(E-E_0)/N$")
axes[0, 1].set_ylabel(r"RMS error of $S(\mathbf{q})$")
for ax, title in zip(axes[0], ("(a) variational energy", "(b) magnetic correlations"), strict=True):
    ax.set_xticks(range(len(physics_arm_labels)), physics_arm_labels)
    ax.set_title(title)
axes[0, 0].legend(frameon=False, fontsize=8)

missing_weight, energy_error = [], []
for trial in translation_trials:
    for row in trial["arms"]:
        arm = row["arm_id"]
        x_value = 1.0 - row["gamma_a1_decomposition"]["weight"]
        y_value = row["metrics"]["energy_error_per_site"]
        missing_weight.append(x_value)
        energy_error.append(y_value)
        axes[1, 0].scatter(
            x_value, y_value, color=physics_arm_colors[arm], s=31,
            edgecolor="white", linewidth=.35,
            label=physics_arm_labels[physics_arm_order.index(arm)].replace("\n", " ")
            if trial is translation_trials[0] else None,
        )
correlation = float(np.corrcoef(missing_weight, energy_error)[0, 1])
axes[1, 0].set_xscale("log")
axes[1, 0].set_yscale("log")
axes[1, 0].set_xlabel(r"missing $\Gamma$-$A_1$ weight $1-w_\Gamma$")
axes[1, 0].set_ylabel(r"$(E-E_0)/N$")
axes[1, 0].set_title(f"(c) error is sector leakage ($r={correlation:.3f}$)")
axes[1, 0].legend(frameon=False, fontsize=6.7, ncol=2)

support_keys = ("bond_2", "triangle_3", "rhombus_4", "block_2x3_6", "half_torus_8")
support_sizes = np.asarray([2, 3, 4, 6, 8])
for arm in ("single_tree_d4", "route_vit_conditioned_d4",
            "route_vit_scrambled_tokens_d4", "single_tree_matched_width"):
    values = np.asarray([
        [
            next(row for row in trial["arms"] if row["arm_id"] == arm)
            ["gamma_a1_decomposition"]["local_rdm_error"][key]["trace_distance_mean"]
            for key in support_keys
        ]
        for trial in translation_trials
    ])
    median = np.median(values, axis=0)
    axes[1, 1].semilogy(
        support_sizes, median, "o-", lw=1.35, ms=4.2,
        color=physics_arm_colors[arm],
        label=physics_arm_labels[physics_arm_order.index(arm)].replace("\n", " "),
    )
    axes[1, 1].fill_between(
        support_sizes, np.min(values, axis=0), np.max(values, axis=0),
        color=physics_arm_colors[arm], alpha=.09,
    )
axes[1, 1].set_xlabel("support size")
axes[1, 1].set_ylabel(r"trace distance after $\Gamma$-$A_1$ selection")
axes[1, 1].set_title("(d) sector-internal error grows with scale")
axes[1, 1].legend(frameon=False, fontsize=7)

fig.suptitle(r"Triangular $4\times4$, $J_2/J_1=0.075$: symmetry organization, not a phase claim")
fig.tight_layout()
save_figure(fig, "fig22_route_vit_physical_error_decomposition")
plt.show()

print("exact Gamma-A1 singlet dimension:", p1["exact_gamma_a1_singlet_dimension"])
print("energy-error / missing-sector-weight correlation:", correlation)
print("translation summary:")
for arm, row in physics_interpretation["summary"]["p1_p2_p5_by_projection_and_arm"]["translation"].items():
    print(arm, row)
"""
        ),
        markdown(
            r"""
The energy--sector-weight correlation is not merely cosmetic.  Across the 18
translation-trained states, 80--99% of the variational error is assigned to
the component orthogonal to \(\Gamma\)-\(A_1\); normalized sector components
have exact-state fidelities between 0.99976 and 0.99999.  The physical global
tokens therefore supply a useful **symmetry-organizing prior** at finite
optimization time.  They do not yet show superior expressivity within the
correct irrep.  The 6- and 8-site RDM errors are the first place where the
remaining sector-internal differences become visible.
"""
        ),
        markdown(
            r"""
## 18. Finite-cluster mechanism: a soft bond-nematic doublet

To recover momenta absent on 4x4, the second calculation uses the 12-site HNF
cell \(((6,4),(0,2))\), which preserves full \(D_6\) and the K/M/X stars.  It
diagonalizes only 132 singlet and 297 triplet multiplicity states.  The price
is a thin-torus geometry: 36 periodic \(J_2\) images alias to 18 distinct
edges of weight two.

The dominant observation is a rapidly softening \(\Gamma\)-\(E_2\) singlet
doublet reached by bond-nematic sources.  The source overlaps the lowest
doublet at 95--96%, and its response grows much faster than the uniform
chirality response.  Simultaneously the lowest magnetic excitation transfers
from K to M between \(J_2/J_1=0.10\) and 0.125.  An X-even bond branch also
has a 92--97% lowest-pole fraction; it remains a monopole *candidate* until an
independent PSG quantum-number match is supplied.
"""
        ),
        code(
            r"""
spectral_cases = physics_interpretation["P3_P4"]["cases"]
j2_grid = np.asarray([row["j2_over_j1"] for row in spectral_cases])

fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.6))
gap_series = {
    r"$\Gamma$-$E_2$ bond": [row["Gamma_E2_bond_nematic"]["lowest_gap"] for row in spectral_cases],
    r"$X^+$ bond": [row["X_plus_bond_branch"]["lowest_gap"] for row in spectral_cases],
    r"K triplet": [row["spin_response"]["K"]["lowest_gap"] for row in spectral_cases],
    r"M triplet": [row["spin_response"]["M"]["lowest_gap"] for row in spectral_cases],
}
for (label, values), marker in zip(gap_series.items(), ("o", "s", "^", "v"), strict=True):
    axes[0, 0].plot(j2_grid, values, marker=marker, lw=1.45, ms=4.6, label=label)
axes[0, 0].axvspan(.10, .125, color="0.9", zorder=0)
axes[0, 0].set_ylabel(r"gap $\Delta/J_1$")
axes[0, 0].set_title("(a) symmetry-resolved soft modes")
axes[0, 0].legend(frameon=False, fontsize=7.5)

for label, key, color, marker in (
    (r"bond-nematic $\Gamma$-$E_2$", "Gamma_E2_bond_nematic", "#e45756", "o"),
    (r"uniform chirality $\Gamma$-$A_2$", "Gamma_A2_uniform_chirality_response", "#4c78a8", "s"),
):
    axes[0, 1].semilogy(
        j2_grid, [row[key]["static_response"] for row in spectral_cases],
        marker=marker, lw=1.45, ms=4.6, color=color, label=label,
    )
axes[0, 1].set_ylabel("static response (source normalization fixed)")
axes[0, 1].set_title("(b) nematic response grows fastest")
axes[0, 1].legend(frameon=False, fontsize=7.5)

axes[0, 2].semilogy(
    j2_grid, [row["Gamma_E2_bond_nematic"]["fidelity_response"] for row in spectral_cases],
    "o-", lw=1.45, ms=4.6, color="#e45756", label=r"$\Gamma$-$E_2$ source",
)
axes[0, 2].semilogy(
    j2_grid, [row["J2_fidelity_response"]["fidelity_response"] for row in spectral_cases],
    "s-", lw=1.45, ms=4.6, color="#59a14f", label=r"global $J_2$ perturbation",
)
axes[0, 2].set_ylabel("fidelity response")
axes[0, 2].set_title("(c) wavefunction sensitivity")
axes[0, 2].legend(frameon=False, fontsize=7.5)

axes[1, 0].plot(
    j2_grid, [row["Gamma_E2_bond_nematic"]["lowest_doublet_fraction"] for row in spectral_cases],
    "o-", lw=1.45, ms=4.6, label=r"$\Gamma$-$E_2$ lowest doublet",
)
axes[1, 0].plot(
    j2_grid, [row["X_plus_bond_branch"]["lowest_pole_fraction"] for row in spectral_cases],
    "s-", lw=1.45, ms=4.6, label=r"$X^+$ lowest pole",
)
axes[1, 0].set_ylim(.88, .985)
axes[1, 0].set_ylabel("source spectral weight")
axes[1, 0].set_title("(d) nearly single-mode bond response")
axes[1, 0].legend(frameon=False, fontsize=7.5)

for momentum, color, marker in (("K", "#4c78a8", "o"), ("M", "#f28e2b", "s")):
    axes[1, 1].plot(
        j2_grid,
        [row["spin_response"][momentum]["spin_structure_factor_isotropic"] for row in spectral_cases],
        marker=marker, lw=1.45, ms=4.6, color=color, label=momentum,
    )
axes[1, 1].set_ylabel(r"$S(\mathbf{q})$")
axes[1, 1].set_title(r"(e) magnetic weight shifts K $\to$ M")
axes[1, 1].legend(frameon=False, fontsize=8)

for momentum, color, marker in (("K", "#4c78a8", "o"), ("M", "#f28e2b", "s")):
    axes[1, 2].plot(
        j2_grid,
        [row["spin_response"][momentum]["static_response_isotropic_per_site"] for row in spectral_cases],
        marker=marker, lw=1.45, ms=4.6, color=color, label=momentum,
    )
axes[1, 2].set_ylabel("isotropic magnetic response / site")
axes[1, 2].set_title("(f) competing magnetic response")
axes[1, 2].legend(frameon=False, fontsize=8)

for ax in axes.flat:
    ax.set_xlabel(r"$J_2/J_1$")
fig.suptitle("Triangular N=12: symmetry-resolved response, not a thermodynamic phase diagram")
fig.tight_layout()
save_figure(fig, "fig23_triangular_symmetry_resolved_soft_modes")
plt.show()

print("geometry caveat:", physics_interpretation["P3_P4"]["geometry_warning"])
print("projector audit:", physics_interpretation["P3_P4"]["projector_audit"])
"""
        ),
        markdown(
            r"""
The local physics hypothesis to test at larger size is now concrete: the
frustrated regime is organized by a low-lying, almost single-mode
\(\Gamma\)-\(E_2\) bond resonance while magnetic spectral weight moves from K
toward M.  This suggests feeding symmetry-labelled bond carriers to the
SU(2)-ViT and asking whether it lowers both the energy and the nematic response
error at fixed parameters.  The finite cluster does **not** prove spontaneous
nematic order, a chiral spin liquid, a deconfined monopole, or a phase
boundary.  Those require size scaling and, for the X branch, a PSG matching
calculation.
"""
        ),
        markdown(
            r"""
## 19. \(E_2\) resonance carrier: a positive mechanism, not an energy frontier

The symmetry-labelled nematic doublet is fixed as

\[
N_a=N^{-1/2}\sum_{r,\delta}E_{a\delta}\,
\mathbf S_r\!\cdot\!\mathbf S_{r+\delta},\qquad
K_{ss'}=\frac{1}{\sqrt2}\sum_a
\frac{N_a^{(s)}N_a^{(s')}+N_a^{(s')}N_a^{(s)}}{2}.
\]

Thus every ground-state carrier is explicitly
\(E_2\otimes E_2\to A_1\); the linear states \(N_a|\psi\rangle\) are retained
separately for the response moments.  The three-carrier shell bank uses
\(K_{11},K_{12},K_{22}\).  It is compared with a dimension-matched ordinary
Hamiltonian Krylov space on the same sealed states, without new training or
diagonalization.  Here \(m_0\) is an equal-time sum-rule weight and the plotted
\(m_1/m_0\) is an SMA gap—not a static susceptibility.
"""
        ),
        code(
            r"""
carrier_variants = (
    "baseline",
    "e2_shell_bank",
    "a1_shell_bank_same_work",
    "joint_e2_a1_shell_banks",
    "hamiltonian_krylov_matched",
)
carrier_labels = (
    "base",
    r"$E_2$ bank",
    r"$A_1$ same work",
    r"$E_2+A_1$",
    "Krylov ceiling",
)
colors = ("#4c78a8", "#f28e2b", "#59a14f")

fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), sharex=True)
for column, projection in enumerate(("translation", "gamma_a1")):
    trials = [row for row in e2_resonance["trials"] if row["projection"] == projection]
    for color, row in zip(colors, trials, strict=True):
        energy = [max(row["variants"][name]["energy_error"], 1e-14) for name in carrier_variants]
        m0 = [max(row["variants"][name]["e2_m0_abs_error"], 1e-14) for name in carrier_variants]
        gap = [max(row["variants"][name]["e2_sma_gap_abs_error"], 1e-14) for name in carrier_variants]
        axes[0, column].plot(range(len(carrier_variants)), energy, "o-", color=color,
                             lw=1.35, ms=4.5, label=f"seed {row['seed']}")
        axes[1, column].plot(range(len(carrier_variants)), m0, "o-", color=color,
                             lw=1.35, ms=4.5)
        axes[1, column].plot(range(len(carrier_variants)), gap, "s--", color=color,
                             lw=1.05, ms=3.8, alpha=.78)
    axes[0, column].set_yscale("log")
    axes[1, column].set_yscale("log")
    axes[0, column].set_title(
        "(a) translation-trained" if column == 0 else r"(b) trained in $\Gamma$-$A_1$"
    )
    axes[1, column].set_title(
        "(c) response moments" if column == 0 else r"(d) response inside $\Gamma$-$A_1$"
    )
    axes[1, column].set_xticks(range(len(carrier_variants)), carrier_labels, rotation=22, ha="right")
axes[0, 0].set_ylabel(r"total energy error $E-E_0$")
axes[1, 0].set_ylabel(r"absolute response error ($m_0$: circle; SMA: square)")
axes[0, 0].legend(frameon=False, fontsize=8)
fig.suptitle(r"Same-work control: $A_1$ beats $E_2$, while the channels remain complementary")
fig.tight_layout()
save_figure(fig, "fig24_e2_resonance_carrier_audit")
plt.show()

print("exact reference:", e2_resonance["exact_reference"])
print("Gamma-A1 dimensions:", {
    key: value["dimension"] for key, value in e2_resonance["gamma_a1_dimensions"].items()
})
print("summary:", e2_resonance["summary"])
print("N12 symmetry certificate:", e2_resonance["n12_exact_symmetry_certificate"])
"""
        ),
        markdown(
            r"""
The strict same-work comparison changes the verdict.  The isotropic
\(A_1\)-shell-pair bank has lower energy than the \(E_2\) bank in all three
translation-trained states and all three \(\Gamma\)-\(A_1\) states.  Thus a
soft nematic response does not imply that the missing ground-state energy is
dominated by an \(E_2\)-pair tangent.  Nevertheless, adding \(E_2\) after
\(A_1\), and adding \(A_1\) after \(E_2\), lowers the energy in 3/3 seeds in
both sectors.  The channels are complementary, but \(E_2\) is not the primary
variational carrier.  A dimension-matched Hamiltonian Krylov space remains a
useful ceiling rather than the same-work physical control.

The construction is algebraically portable to the 1,109,658-dimensional N=36
\(\Gamma\)-\(A_1\) singlet sector.  The next section replaces the Krylov ceiling
by Hamiltonian-blind generic two-bond carriers and records the repaired N=36
mixing gate.  Neither change by itself is an N=36 carrier-energy result.
"""
        ),
        markdown(
            r"""
## 20. Four-arm causal control and repaired N=36 basin mixing

The frozen-backbone comparison now uses four scientific arms: baseline,
three isotropic \(A_1\) shell-pair carriers, \(A_1\) plus three
\(E_2\otimes E_2\to A_1\) carriers, and generic two-bond carriers.  Generic
sources are drawn from a larger relative-bond geometry pool, made Hermitian,
and Reynolds-averaged over the full space group without access to the
Hamiltonian or energy.  The nested comparisons are dimension matched:
\(A_1\) versus generic-3 and \(A_1+E_2\) versus generic-6.

Separately, the N=36 sampler now mixes exchange moves, D6 orbit moves, and a
full-support dimer/uniform independence proposal.  The diagnostic remains in
this notebook: passing it admits a later variational test, but is not a lower
energy or QSL claim.
"""
        ),
        code(
            r"""
direction_count = np.asarray([0, 3, 6])
fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))

for trial_index, trial in enumerate(carrier_four_arm["trials"]):
    color = colors[trial_index]
    physical = [
        trial["arms"][name]["energy_error"]
        for name in ("baseline", "A1", "A1_plus_E2")
    ]
    axes[0].plot(direction_count, physical, "o-", color=color, lw=1.6,
                 label=f"physical, seed {trial['state_seed']}")
    for bank_index, bank in enumerate(trial["generic_bank_robustness"]):
        generic = [
            trial["arms"]["baseline"]["energy_error"],
            bank["generic3"]["energy_error"],
            bank["generic6"]["energy_error"],
        ]
        axes[0].plot(direction_count, generic, "--", color=color, lw=.9,
                     alpha=.38, marker="x", ms=3.4,
                     label="generic banks" if trial_index == 0 and bank_index == 0 else None)

    physical_m0 = [
        trial["arms"][name]["e2_m0_abs_error"]
        for name in ("baseline", "A1", "A1_plus_E2")
    ]
    primary_generic = [
        trial["arms"][name]["e2_m0_abs_error"]
        for name in ("baseline", "generic3", "generic6")
    ]
    axes[1].plot(direction_count, physical_m0, "o-", color=color, lw=1.6,
                 label=f"physical, seed {trial['state_seed']}")
    axes[1].plot(direction_count, primary_generic, "x--", color=color,
                 lw=1.0, alpha=.55)

observable_order = ("diagonal_3hz_energy", "log_born_weight", "nematic_q1",
                    "nematic_q2", "nematic_q_squared", "diagonal_Szz_K_star",
                    "diagonal_Szz_M_star")
observable_labels = (r"$E$", r"$\log p$", r"$Q_1$", r"$Q_2$", r"$Q^2$", r"$S^{zz}(K)$", r"$S^{zz}(M)$")
rhat = [n36_basin_gate["correlations"][name]["split_rhat"] for name in observable_order]
axes[2].plot(range(len(rhat)), rhat, "o-", color="#4c78a8", lw=1.5)
axes[2].axhline(n36_basin_gate["config"]["gates"]["split_rhat_maximum"],
                color="#d62728", ls="--", lw=1.0, label="admission threshold")
axes[2].set_xticks(range(len(rhat)), observable_labels, rotation=25, ha="right")

axes[0].set(yscale="log", xlabel="added carrier directions",
            ylabel=r"total energy error $E-E_0$", title="(a) matched carrier subspaces")
axes[1].set(yscale="log", xlabel="added carrier directions",
            ylabel=r"$|m_0-m_0^{\rm exact}|$", title=r"(b) nematic-response repair")
axes[2].set(ylabel=r"split $\hat R$", title="(c) N=36 cross-basin gate")
axes[0].legend(frameon=False, fontsize=7.4, ncol=2)
axes[2].legend(frameon=False, fontsize=8)
fig.tight_layout()
save_figure(fig, "fig25_four_arm_carriers_and_n36_basin_gate")
plt.show()

print("four-arm summary:", carrier_four_arm["summary"])
print("N36 gates:", n36_basin_gate["gates"])
print("N36 split-Rhat:", {name: n36_basin_gate["correlations"][name]["split_rhat"]
                             for name in observable_order})
"""
        ),
        markdown(
            r"""
The physical organization beats the generic controls robustly: \(A_1\) has
lower energy than generic-3 in all 9 state/bank comparisons, and
\(A_1+E_2\) beats generic-6 in all 9.  All resulting states remain singlets,
the maximum generic \(\Gamma\)-\(A_1\) leakage is below \(6\times10^{-11}\),
and each generic bank contributes six directions outside the physical
six-dimensional span.  The result is therefore not explained by a rotated
copy of the same carrier space.

The interpretation is narrower than “E2 wins.”  The isotropic \(A_1\) bank
produces nearly all of the energy improvement.  Adding \(E_2\) repairs the
nematic equal-time response in 3/3 states, but its marginal energy gain is
smaller than the second generic triplet in 0/9 comparisons.  The next N=36
experiment should therefore make \(A_1\) the primary trainable physical
carrier and retain \(E_2\) as a preregistered response-repair channel.

The repaired sampler passes every admission gate.  Across energy, log Born
weight, both nematic components, \(Q^2\), and K/M diagonal structure probes,
split \(\hat R\) lies between 1.0086 and 1.0189; stratum ESS fractions are
0.402--0.522.  Exchange, D6-orbit, and independence proposals all accept
nontrivial moves, four initialization strata agree in a 32-sample full-H
audit within two combined standard errors, and checkpoint replay is exact.
This clears the sampling blocker only; the state remains untrained and only
translation projected.
"""
        ),
        markdown(
            r"""
## 21. N=36 common-base carrier versus ViT tangent frontier

This is the first trainable \(6\times6\) comparison in which all six arms
start from the same wavefunction, see the exact same known-density bridge
samples, take the same two updates, and receive 56 full-H configuration
actions.  The four exact-SU(2) arms compare \(A_1\), \(A_1+E_2\), and
dimension-matched generic route carriers.  The two controls are six-dimensional
ViT correlator tangents on the same SU(2) base: a translation-projected
fixed-\(S^z\) spatial ViT and an absolute-position, symmetry-unaware ViT.

The result is a route-selection pilot, not a competitive \(6\times6\) energy:
the common audit has only 16 configurations and the base is still the
near-dimer Route-ViT checkpoint.
"""
        ),
        code(
            r"""
arm_order = [
    "su2_a1_3", "su2_a1_e2_6", "su2_generic_3", "su2_generic_6",
    "fixed_sz_spatial_vit_6", "symmetry_unaware_vit_6",
]
arm_labels = [r"$A_1$", r"$A_1+E_2$", "generic-3", "generic-6",
              r"fixed-$S^z$ spatial ViT", "position ViT"]
rows = {row["arm_id"]: row for row in n36_carrier_vit["arms"]}
x = np.arange(len(arm_order))
delta_e = np.asarray([
    rows[name]["final_audit"]["energy_per_site"]
    - rows[name]["initial_audit"]["energy_per_site"]
    for name in arm_order
])
delta_q2 = np.asarray([
    rows[name]["final_audit"]["diagonal_nematic_q_squared"]
    - rows[name]["initial_audit"]["diagonal_nematic_q_squared"]
    for name in arm_order
])
wall = np.asarray([rows[name]["wall_seconds"] for name in arm_order])
arm_colors = ["#4c78a8", "#2f4b7c", "#9c9c9c", "#6b6b6b", "#f58518", "#54a24b"]

fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0))
axes[0].plot(x, delta_e, "-", color="0.72", lw=1.0)
for index, color in enumerate(arm_colors):
    axes[0].plot(x[index], delta_e[index], "o", color=color, ms=6)
axes[0].axhline(0.0, color="0.35", ls="--", lw=.9)
axes[0].set_ylabel(r"$E_{\rm final}/N-E_{0}/N$")
axes[0].set_title("(a) common-base energy response")
axes[1].plot(x, delta_q2, "-", color="0.72", lw=1.0)
for index, color in enumerate(arm_colors):
    axes[1].plot(x[index], delta_q2[index], "s", color=color, ms=5.8)
axes[1].axhline(0.0, color="0.35", ls="--", lw=.9)
axes[1].set_ylabel(r"$\Delta\langle Q_1^2+Q_2^2\rangle_{S^z}$")
axes[1].set_title("(b) diagonal nematic response")
for index, (label, color) in enumerate(zip(arm_labels, arm_colors, strict=True)):
    axes[2].plot(wall[index], delta_e[index], "o", color=color, ms=6, label=label)
axes[2].axhline(0.0, color="0.35", ls="--", lw=.9)
axes[2].set(xlabel="local CPU wall time (s)",
            ylabel=r"$\Delta E/N$", title="(c) energy response versus time")
axes[2].legend(frameon=False, fontsize=7.2, ncol=2, loc="lower left")
for axis in axes[:2]:
    axis.set_xticks(x, arm_labels, rotation=27, ha="right")
fig.tight_layout()
save_figure(fig, "fig26_n36_carrier_vit_common_base_frontier")
plt.show()

print("matched-work gates:", n36_carrier_vit["gates"])
print("energy comparisons:", n36_carrier_vit["comparisons"])
print("ViT tangent registrations:", n36_carrier_vit["vit_residual_controls"])
"""
        ),
        markdown(
            r"""
The physical carrier result is directionally consistent but too small to
claim: \(A_1+E_2\) lies \(3.75\times10^{-8}\) per site below generic-6 and
adding the registered E2-response triplet lowers \(A_1\) by
\(1.91\times10^{-7}\) per site.  The 16-sample audit cannot resolve either
difference.

The largest directional response instead comes from the factorized spatial
inductive bias.  The real-valued fixed-\(S^z\), translation-projected
spatial-ViT correlator lowers the common-base energy by
\(8.19\times10^{-5}\) per site, while the absolute-position ViT does not.
Both controls have six full-rank, Hamiltonian-blind whitened tangents, exactly
nest the same base, and keep every sampled local energy real; all arms retain
final importance ESS above 0.812.  The 16-state audit does **not** resolve this
as an energy win.  It only nominates spatial organization, rather than a
generic increase in tangent dimension, for the next test: lift the correlator
into an SU(2)-scalar reduced-tensor operation and repeat the common-base
frontier with multiple sample sets and paired uncertainty.
"""
        ),
        markdown(
            r"""
## 22. Repeated SU(2)-scalar multiplicity-attention audit

The model is the spin-½ triangular-lattice (J_1-J_2) Heisenberg model on
the periodic (6\times6) torus at (J_2/J_1=1/8).  The same-torus references
are (-0.515564) per site from ED and (-0.515386(9)) from the raw
space-group GCNN of Roth, Szabo, and MacDonald (PRB 108, 054410).  The common
Route-ViT state is only a sampler-admitted near-dimer checkpoint, not a
converged competitor.

The new exact-SU(2) attention reads static support and pair descriptors for
all 35 fusion nodes in each route and generates node-resolved reduced maps.
It never acts on a magnetic index.  A parameter-identical control scrambles
the descriptor-to-node assignment while retaining all descriptor marginals.
Three independent training/audit samples now permit paired uncertainties.
"""
        ),
        code(
            r"""
literature = n36_multiplicity_spatial["literature_targets"]
replicates = n36_multiplicity_spatial["replicates"]
aggregate = n36_multiplicity_spatial["paired_aggregate"]
energy_aggregate = n36_multiplicity_spatial["arm_energy_aggregate"]

absolute_labels = ["ED", "space-group\nGCNN", "SU2 spatial", "scrambled", "fixed-Sz\nspatial"]
absolute_energy = [
    literature["exact_36_site_energy_per_site"],
    literature["raw_space_group_gcnn_energy_per_site"],
    energy_aggregate["su2_multiplicity_spatial"]["mean_energy_per_site"],
    energy_aggregate["su2_scrambled_geometry"]["mean_energy_per_site"],
    energy_aggregate["fixed_sz_spatial_vit"]["mean_energy_per_site"],
]
absolute_error = [
    0.0,
    literature["raw_space_group_gcnn_uncertainty"],
    energy_aggregate["su2_multiplicity_spatial"]["replicate_standard_error_per_site"],
    energy_aggregate["su2_scrambled_geometry"]["replicate_standard_error_per_site"],
    energy_aggregate["fixed_sz_spatial_vit"]["replicate_standard_error_per_site"],
]

fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.9))
x = np.arange(len(absolute_labels))
axes[0].errorbar(x, absolute_energy, yerr=absolute_error, fmt="o", capsize=3,
                 color="#2f4b7c")
axes[0].set_xticks(x, absolute_labels, rotation=24, ha="right")
axes[0].set_ylabel(r"$E/N$")
axes[0].set_title("(a) absolute same-torus target")

paired_small = ["spatial_minus_base", "scrambled_minus_base"]
small_labels = ["SU2 spatial", "scrambled"]
small_colors = ["#4c78a8", "#9c9c9c"]
for index, rep in enumerate(replicates):
    for position, (name, color) in enumerate(zip(paired_small, small_colors, strict=True)):
        row = rep["paired_comparisons"][name]
        axes[1].errorbar(position + 0.06*(index-1), row["difference_per_site"]*1e7,
                         yerr=row["standard_error_per_site"]*1e7,
                         fmt="o", capsize=2, color=color, alpha=.82)
axes[1].axhline(0, color="0.4", ls="--", lw=.8)
axes[1].set_xticks(range(2), small_labels, rotation=20, ha="right")
axes[1].set_ylabel(r"paired $\Delta E/N\;(10^{-7})$")
axes[1].set_title("(b) exact-SU2 micro-response")

fixed_rows = [rep["paired_comparisons"]["fixed_sz_minus_base"] for rep in replicates]
axes[2].errorbar(np.arange(3), [row["difference_per_site"]*1e4 for row in fixed_rows],
                 yerr=[row["standard_error_per_site"]*1e4 for row in fixed_rows],
                 fmt="o", capsize=3, color="#f58518")
axes[2].axhline(0, color="0.4", ls="--", lw=.8)
axes[2].set_xticks(range(3), ["rep 0", "rep 1", "rep 2"])
axes[2].set_ylabel(r"paired $\Delta E/N\;(10^{-4})$")
axes[2].set_title(r"(c) fixed-$S^z$ response")

causal_rows = [rep["paired_comparisons"]["spatial_minus_scrambled"] for rep in replicates]
axes[3].errorbar(np.arange(3), [row["difference_per_site"]*1e8 for row in causal_rows],
                 yerr=[row["standard_error_per_site"]*1e8 for row in causal_rows],
                 fmt="o", capsize=3, color="#54a24b")
mean = aggregate["spatial_minus_scrambled"]["mean_difference_per_site"]*1e8
axes[3].axhline(mean, color="#54a24b", lw=1.1, label="replicate mean")
axes[3].axhline(0, color="0.4", ls="--", lw=.8)
axes[3].set_xticks(range(3), ["rep 0", "rep 1", "rep 2"])
axes[3].set_ylabel(r"spatial-scrambled $\Delta E/N\;(10^{-8})$")
axes[3].set_title("(d) geometry-causal effect")
axes[3].legend(frameon=False, fontsize=7.5)
fig.tight_layout()
save_figure(fig, "fig27_n36_su2_multiplicity_spatial_paired")
plt.show()

print("paired aggregate:", aggregate)
print("admission gates:", n36_multiplicity_spatial["gates"])
"""
        ),
        markdown(
            r"""
Correct support geometry beats the scrambled control in all three replicates:
(Delta E/N=-7.54(1.33)\times10^{-8}).  This is a reproducible causal
effect, but far too small to matter physically.  The fixed-(S^z) direction
is roughly three orders larger, yet changes sign in one replicate and is not
resolved in the replicate mean.  Thus the attention boundary is
mathematically sound, while the variational bottleneck remains the frozen,
untrained reduced backbone.  Promotion requires joint backbone-attention
training and an energy-to-work comparison against the immutable ED/GCNN
targets; increasing a residual scale alone is not a physics result.
"""
        ),
        markdown(
            r"""
## 23. Joint reduced-backbone training and a physical pi-flux baseline

The frozen-head experiment isolated a causal response, but it could not test
whether the reduced multiplicity state and spatial attention cooperate during
optimization.  The next audit therefore jointly trains the 1278-parameter
reduced Route-ViT backbone and the 1614-parameter SU(2)-scalar attention head.
The spatial and scrambled arms start from identical parameters and see the
same configurations, four updates, and 96 full-H configuration calls per
replicate.

This short common-proposal audit is accompanied by a genuinely independent
variational benchmark.  The projected pi/0-flux state is a spin-independent
fermionic Dirac sea on the same periodic 6x6 triangular torus.  Gutzwiller
projection fixes one fermion per site, three closed-shell boundary sectors are
combined in the exact Gamma-A1 character, and no Hamiltonian diagonalization
or neural residual enters its energy estimate.  Its energy is measured by
direct fixed-Sz exchange-chain VMC in three independent 2048-sample runs.
"""
        ),
        code(
            r"""
pi_lit = n36_pi_flux["literature_targets"]
pi_agg = n36_pi_flux["aggregate"]
joint_reps = n36_joint_multiplicity_spatial["replicates"]
joint_agg = n36_joint_multiplicity_spatial["paired_aggregate"]

method_labels = [
    "ED", "space-group\nGCNN", "DSL\n0-LS", "projected pi-flux\n(this work)",
    "DSL\n1-LS", "DSL\n2-LS",
]
method_energies = np.asarray([
    pi_lit["exact_36_site_energy_per_site"],
    pi_lit["space_group_gcnn_energy_per_site"],
    pi_lit["dsl_zero_lanczos_energy_per_site"],
    pi_agg["mean_energy_per_site"],
    pi_lit["dsl_one_lanczos_energy_per_site"],
    pi_lit["dsl_two_lanczos_energy_per_site"],
])
method_errors = np.asarray([
    0.0, 9e-6, 1e-6, pi_agg["replicate_standard_error_per_site"], 2e-6, 3e-6,
])

fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.0))
x = np.arange(len(method_labels))
colors = ["#222222", "#54a24b", "#9c755f", "#e45756", "#9c755f", "#9c755f"]
for index, color in enumerate(colors):
    axes[0].errorbar(
        x[index], method_energies[index], yerr=method_errors[index], fmt="o",
        capsize=3, color=color, ms=6.3, zorder=3,
    )
axes[0].plot([2, 4, 5], method_energies[[2, 4, 5]], "--", color="#9c755f", lw=1.0)
axes[0].set_xticks(x, method_labels, rotation=25, ha="right")
axes[0].set_ylabel(r"$E/N$")
axes[0].set_title(r"(a) $6\times6$, $J_2/J_1=1/8$")

joint_names = ["spatial_minus_base", "scrambled_minus_base"]
joint_labels = ["joint spatial", "joint scrambled"]
for rep_index, rep in enumerate(joint_reps):
    values = np.asarray([
        rep["paired_comparisons"][name]["difference_per_site"] for name in joint_names
    ]) * 1e4
    axes[1].plot([0, 1], values, "o-", color=f"C{rep_index}", alpha=.78,
                 label=f"rep {rep_index}")
axes[1].axhline(0.0, color="0.4", ls="--", lw=.8)
axes[1].set_xticks([0, 1], joint_labels, rotation=18, ha="right")
axes[1].set_ylabel(r"paired $\Delta E/N\;(10^{-4})$")
axes[1].set_title("(b) joint backbone + head descent")
axes[1].legend(frameon=False, fontsize=7.5)

causal_rows = [
    rep["paired_comparisons"]["spatial_minus_scrambled"] for rep in joint_reps
]
causal_x = np.arange(len(causal_rows))
axes[2].errorbar(
    causal_x,
    [row["difference_per_site"] * 1e5 for row in causal_rows],
    yerr=[row["standard_error_per_site"] * 1e5 for row in causal_rows],
    fmt="o", capsize=3, color="#4c78a8",
)
causal_mean = joint_agg["spatial_minus_scrambled"]["mean_difference_per_site"] * 1e5
causal_se = joint_agg["spatial_minus_scrambled"]["replicate_standard_error_per_site"] * 1e5
axes[2].axhspan(causal_mean-causal_se, causal_mean+causal_se,
                color="#4c78a8", alpha=.14)
axes[2].axhline(causal_mean, color="#4c78a8", lw=1.1, label="replicate mean")
axes[2].axhline(0.0, color="0.4", ls="--", lw=.8)
axes[2].set_xticks(causal_x, [f"rep {index}" for index in causal_x])
axes[2].set_ylabel(r"spatial-scrambled $\Delta E/N\;(10^{-5})$")
axes[2].set_title("(c) geometry-causal increment")
axes[2].legend(frameon=False, fontsize=7.5)

fig.tight_layout()
save_figure(fig, "fig28_n36_pi_flux_joint_backbone")
plt.show()

print("pi-flux aggregate:", pi_agg)
print("joint paired aggregate:", joint_agg)
print("joint admission gates:", n36_joint_multiplicity_spatial["gates"])
"""
        ),
        markdown(
            r"""
The projected pi-flux state reaches (E/N=-0.508186(81)), where the
parenthesized uncertainty is the standard error across independent runs.
It is substantially below the unoptimized (0)-Lanczos U(1) Dirac state of
Iqbal *et al.* ((-0.501788(1))), but above their one- and two-Lanczos-step
states ((-0.510344(2)), (-0.512503(3))); the constructions differ, so this
ordering is a physical benchmark rather than an architecture victory.  Its
gap to ED is (0.0073782) per site (1.43%), showing that a sensible frustrated
sign structure removes most of the catastrophic gap of the near-dimer state.

Joint training lowers the paired common-proposal estimate by
(-8.44(1.10)\times10^{-4}) per site in the spatial arm, but the scrambled arm
already gains (-8.19(1.07)\times10^{-4}).  Correct geometry contributes the
smaller, sign-consistent increment
(-2.55(0.34)\times10^{-5}).  Thus joint training amplifies the geometry
response by roughly two orders of magnitude, yet about 97% of the short-run
descent comes from the shared reduced backbone.  The next credible neural
test should start from or distill the pi-flux sign structure; more attention
optimization around the near-dimer basin is not competitive.
"""
        ),
        markdown(
            r"""
## 24. Four-by-four provenance audit and the physical-carrier three-arm pilot

The historical six-scalar N=36 ``fixed_sz_spatial_vit_6`` object was not
obtained by transferring a trained 4x4 state.  It instantiated a fresh 6x6
Spatial-ViT with random Hamiltonian-blind weights, selected six adapter
directions, whitened their tangent vectors, and multiplied that local tangent
onto the same near-dimer SU(2) base.  It was a directional response test, not
an independent ViT variational state.

The actual 4x4 fixed-Sz Spatial-ViT is a separate exact-support calculation.
On the triangular torus at $J_2/J_1=1/8$, its matched $d=8$ spatial model has
median error $0.024883/site$ and beats the flat control in only one of three
seeds.  The failure cannot be blamed on the finite graph alone: exact-SU(2)
Route-ViT and matched-width ClebschTree states on the same graph reach
$1.42\times10^{-7}$ and $1.34\times10^{-9}$ error per site.

The N=36 follow-up therefore compares actual wavefunctions: a from-scratch
fixed-Sz Spatial-ViT, a projected pi-flux state times a positive
translation-scalar Spatial-ViT correlator, and an exact-SU(2) multiplicity
spatial model trained first by parton-Born ratio-fidelity distillation.  Every
arm then receives the same proposal samples, two VMC updates, and 72 full-H
configuration calls per replicate.
"""
        ),
        code(
            r"""
tri_vit_case = next(
    case for case in spatial_vit["cases"]
    if case["case_id"] == "triangular_4x4_j2_0p125_fixed_sz_spatial_vit"
)
four_by_four = {
    arm: [
        trial["metrics"]["energy_error_per_site"]
        for trial in tri_vit_case["trials"] if trial["arm_id"] == arm
    ]
    for arm in ("flat_vit_d8_l1", "spatial_vit_d8_l1")
}
su2_four = su2_route_vit_long["summary"][
    "triangular_4x4_j2_0p125_su2_route_vit_long"
]["arms"]
three_reps = n36_physical_three_arm["replicates"]
three_agg = n36_physical_three_arm["aggregate"]
three_ids = [
    "from_scratch_fixed_sz_spatial_vit",
    "pi_flux_spatial_vit_correlator",
    "pi_flux_distilled_su2_multiplicity_spatial",
]

fig, axes = plt.subplots(1, 3, figsize=(14.7, 4.05))
labels_4 = ["flat ViT\nd=8", "spatial ViT\nd=8", "SU2 route-ViT", "SU2 tree\nmatched width"]
for index, arm in enumerate(("flat_vit_d8_l1", "spatial_vit_d8_l1")):
    values = np.asarray(four_by_four[arm])
    axes[0].plot(index + np.linspace(-.07, .07, len(values)), values, "o",
                 color=("#9c9c9c", "#4c78a8")[index], alpha=.82)
    axes[0].plot(index, np.median(values), "_", color="black", ms=16, mew=1.4)
axes[0].plot(2, su2_four["route_vit_conditioned_d4"]["median_energy_error_per_site"],
             "D", color="#54a24b", ms=6)
axes[0].plot(3, su2_four["single_tree_matched_width"]["median_energy_error_per_site"],
             "D", color="#2f4b7c", ms=6)
axes[0].set_yscale("log")
axes[0].set_xticks(range(4), labels_4, rotation=19, ha="right")
axes[0].set_ylabel(r"$E/N-E_0/N$")
axes[0].set_title(r"(a) trained $4\times4$ states")

rep_x = np.arange(len(three_reps))
anchored_rows = [
    next(row for row in rep["arms"] if row["arm_id"] == three_ids[1])[
        "paired_final_minus_initial"
    ]
    for rep in three_reps
]
axes[1].errorbar(
    rep_x,
    [row["difference_per_site"] * 1e3 for row in anchored_rows],
    yerr=[row["standard_error_per_site"] * 1e3 for row in anchored_rows],
    fmt="o", capsize=3, color="#e45756", label="pi-flux + ViT",
)
axes[1].axhline(0.0, color="0.4", ls="--", lw=.8)
axes[1].set_xticks(rep_x, [f"rep {index}" for index in rep_x])
axes[1].set_ylabel(r"paired $(E_f-E_i)/N\;(10^{-3})$")
axes[1].set_title("(b) admitted low-energy response")
axes[1].legend(frameon=False, fontsize=7.5)
axes[1].text(.02, .04, "pure ViT excluded: positive energy / ESS failure",
             transform=axes[1].transAxes, fontsize=7.2, color="0.38")

short_labels = ["pure\nViT", "pi-flux\n+ ViT", "distilled\nSU2"]
arm_colors = ["#9c9c9c", "#e45756", "#4c78a8"]
for arm_index, (arm_id, color) in enumerate(zip(three_ids, arm_colors, strict=True)):
    ess = [
        next(row for row in rep["arms"] if row["arm_id"] == arm_id)[
            "final_audit"
        ]["importance_ess_fraction"]
        for rep in three_reps
    ]
    axes[2].plot(arm_index + np.linspace(-.04, .04, len(ess)), ess, "o",
                 color=color, label="importance ESS" if arm_index == 0 else None)
heldout_fidelity = [
    next(row for row in rep["arms"] if row["arm_id"] == three_ids[2])[
        "carrier_distillation"
    ]["heldout_sample_fidelity"]
    for rep in three_reps
]
axes[2].plot(2 + np.linspace(.10, .18, len(heldout_fidelity)), heldout_fidelity,
             "D", color="#f58518", ms=5, label="held-out carrier fidelity")
axes[2].axhline(
    n36_physical_three_arm["gates"]["importance_ess_fraction_minimum"],
    color="0.35", ls="--", lw=.9, label="ESS gate",
)
axes[2].set_yscale("log")
axes[2].set_ylim(.025, 1.15)
axes[2].set_xticks(range(3), short_labels)
axes[2].set_ylabel("fraction")
axes[2].set_title("(c) overlap/admission audit")
axes[2].legend(frameon=False, fontsize=7.2)

fig.tight_layout()
save_figure(fig, "fig29_vit_provenance_physical_three_arm")
plt.show()

print("three-arm aggregate:", three_agg)
print("three-arm gates:", n36_physical_three_arm["gates"])
"""
        ),
        markdown(
            r"""
The from-scratch ViT stays near the positive, high-spin fixed-Sz basin: its
short-run energy is positive and one replicate fails the ESS gate.  It is not
a valid low-energy comparator.  The pi-flux correlator passes ESS in both
replicates (minimum 0.934) and lowers the paired energy by
$-0.00432(66)/site$ relative to the exactly nested bare carrier.  Its two
absolute 24-sample estimates straddle the longer-run pi-flux value, so the
paired response—not their raw mean—is the present evidence.

Ratio-fidelity distillation of the near-dimer exact-SU2 model reaches only
0.048 and 0.042 held-out fidelity.  Its ESS fails in both replicates, so its
raw ratio-estimator energies are discarded.  Exact symmetry by itself is not
the carrier.  The promoted architecture must contain the parton state as an
explicit coherent channel and let an SU(2)-scalar multiplicity correlator act
around that state; asking the old backbone to relearn the sign structure is
both less accurate and roughly an order of magnitude slower per matched
full-H ledger in this pilot.
"""
        ),
        markdown(
            r"""
### Explicit singlet addition fails; scalar operators reveal the physical boundary

The next test makes the projected pi-flux carrier an exact coherent channel,

\[
|\psi\rangle=|\psi_{\pi}\rangle+\lambda|\psi_{\rm CG-attn}\rangle .
\]

This is algebraically safer than a pointwise ViT factor: both terms are
singlets, so every mixture remains in $S=0$.  It is nevertheless not a useful
variational neighbourhood.  Raw amplitude-RMS calibration collapses the
importance ESS to $1/24$ in both replicates.  Carrier-Born Hilbert-norm
calibration still fails one ESS gate; the surviving replicate develops an
enormous local-energy variance and no significant paired result.  The two
singlets have incompatible nodal/support structures.

The replacement acts with exact SU(2)-scalar, full-$A_1$ operators *on* the
physical carrier.  The 4x4 exact-small scan below compares every proposed
operator bank with a same-dimensional polynomial Krylov space.  It is route
selection evidence and remains outside the paper notebook.
"""
        ),
        code(
            r"""
scalar_scan = scalar_operator_frontier["scan"]
couplings = np.asarray([row["j2_over_j1"] for row in scalar_scan])

def scalar_energy(label):
    return np.asarray([
        next(item for item in row["rows"] if item["label"] == label)[
            "energy_per_site"
        ]
        for row in scalar_scan
    ])

fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.1))
for label, title, color, marker in (
    ("bare_pi0", "bare pi-flux", "#9c9c9c", "o"),
    ("one_lanczos_H", "one Lanczos", "#f58518", "s"),
    ("shell_resolved_J1_J2", "shell J1/J2", "#4c78a8", "^"),
    ("two_lanczos_H", "two Lanczos", "#54a24b", "D"),
):
    axes[0].plot(couplings, scalar_energy(label), marker=marker, ms=4,
                 color=color, label=title)
axes[0].plot(
    couplings,
    [row["exact_energy_per_site"] for row in scalar_scan],
    color="black", lw=1.2, label="exact",
)
axes[0].set_xlabel(r"$J_2/J_1$")
axes[0].set_ylabel(r"$E/N$")
axes[0].set_title("(a) scalar response on the physical carrier")
axes[0].legend(frameon=False, fontsize=7.5, ncol=2)

axes[1].plot(
    couplings,
    [row["same_dimension_gate"]["shell_minus_two_lanczos_energy_per_site"]
     for row in scalar_scan],
    "o-", color="#4c78a8", label="shell - matched Krylov",
)
axes[1].plot(
    couplings,
    [row["e2_scalar_gate"]["e2_bank_minus_degree3_krylov_energy_per_site"]
     for row in scalar_scan],
    "s-", color="#e45756", label=r"$E_2$ bank - matched Krylov",
)
axes[1].plot(
    couplings,
    [row["e2_scalar_gate"]["joint_minus_degree5_krylov_energy_per_site"]
     for row in scalar_scan],
    "^-", color="#b279a2", label=r"Krylov+$E_2$ - matched Krylov",
)
axes[1].axhline(0.0, color="0.35", ls="--", lw=.9)
axes[1].axvline(0.125, color="0.65", ls=":", lw=.9)
axes[1].set_xlabel(r"$J_2/J_1$")
axes[1].set_ylabel(r"matched $\Delta E/N$")
axes[1].set_title("(b) only stripe-side shell response crosses")
axes[1].legend(frameon=False, fontsize=7.2)

fig.tight_layout()
save_figure(fig, "fig30_scalar_operator_frontier")
plt.show()

print("raw coherent-SU2 ESS:",
      n36_coherent_su2_raw["aggregate"]["minimum_final_importance_ess_fraction"])
print("Born-calibrated coherent-SU2 gate:", n36_coherent_su2_born["aggregate"])
print("scalar-operator decision:", scalar_operator_frontier["decision"])
"""
        ),
        markdown(
            r"""
At $J_2/J_1=1/8$, shell-resolved $J_1/J_2$ response improves on one
Lanczos step but remains $1.184\times10^{-4}/site$ above the equal-dimensional
two-Lanczos state.  The $E_2\otimes E_2\to A_1$ bank also loses both matched
Krylov gates.  Shell resolution first wins at the registered $J_2/J_1=0.175$
point, so it is interpreted as a stripe-side response.  For the QSL region,
the next scalar bank must instead target gauge flux, valence-bond resonance,
or long-range singlet rearrangement.  $E_2$ remains a response observable, not
the main energy head.
"""
        ),
        markdown(
            r"""
## 25. Projected-parton three-arm frontier: a variational win, not yet a QSL-specific win

All arms below start from the same projected pi/0-flux carrier on the
triangular periodic $4\times4$ cluster at $J_2/J_1=0.125$.  The conventional
control optimizes every full-$D_6$ pair-Jastrow orbit jointly with up to two
Lanczos steps.  The fixed-$S^z$ arm uses a positive full-$A_1$ Spatial-ViT
correlator.  The exact-SU(2) arm uses two residual layers of five scalar
operator tokens: $J_1,J_2$, rhombus ring exchange, rhombus dimer resonance,
and a six-site loop rearrangement.
"""
        ),
        code(
            r"""
qsl_frontier = qsl_three_arm_frontier["frontier"]
qsl_scan = qsl_three_arm_frontier["phase_scan"]
qsl_n36_preflight = qsl_three_arm_frontier["n36_connectivity_preflight"]
jastrow2 = qsl_frontier["optimized_pi_flux_jastrow_lanczos"][-1]
fixed = qsl_frontier["aggregates"]["fixed_sz_vit"]
operator = qsl_frontier["aggregates"]["exact_su2_a1_operator_vit"]

fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
labels = ["bare\n$\pi$-flux", "fixed-$S^z$\nViT", "Jastrow\n+ 2 LS", "SU(2)/$A_1$\nOperator-ViT"]
values = [
    qsl_frontier["bare_carrier"]["energy_per_site"],
    fixed["mean_energy_per_site"],
    jastrow2["energy_per_site"],
    operator["mean_energy_per_site"],
]
errors = [0.0, fixed["replicate_standard_error_per_site"], 0.0,
          operator["replicate_standard_error_per_site"]]
colors = ["#9c9c9c", "#e45756", "#f58518", "#4c78a8"]
for index, (value, error, color) in enumerate(zip(values, errors, colors)):
    axes[0].errorbar(index, value, yerr=error, fmt="o", ms=7, capsize=3,
                     color=color)
axes[0].plot(range(len(values)), values, color="0.75", lw=0.8, zorder=0)
axes[0].axhline(qsl_frontier["exact_energy_anchor_per_site"], color="black",
                lw=1.1, ls="--", label="exact anchor")
axes[0].set_xticks(range(len(labels)), labels)
axes[0].set_ylabel(r"$E/N$")
axes[0].set_title("(a) common-carrier variational frontier")
axes[0].legend(frameon=False, fontsize=8)

scan_rows = qsl_scan["rows"]
couplings = [row["j2_over_j1"] for row in scan_rows]
gain = [row["operator_minus_jastrow_two_lanczos_per_site"] for row in scan_rows]
axes[1].axvspan(0.075, 0.15, color="#72b7b2", alpha=0.16,
                label="declared frustrated window")
axes[1].plot(couplings, gain, "o-", color="#4c78a8",
             label="Operator-ViT - Jastrow+2LS")
axes[1].axhline(0.0, color="0.35", ls="--", lw=.9)
axes[1].set_xlabel(r"$J_2/J_1$")
axes[1].set_ylabel(r"$\Delta E/N$")
axes[1].set_title("(b) the finite-size gain is not QSL-specific")
axes[1].legend(frameon=False, fontsize=7.7)

fig.tight_layout()
save_figure(fig, "fig31_pi_flux_operator_vit_frontier")
plt.show()

print("comparisons:", qsl_frontier["comparisons"])
print("symmetry/work ledger:", qsl_frontier["operator_tokens"])
print("gates:", qsl_frontier["gates"])
print("phase specificity:", {
    key: qsl_scan[key] for key in (
        "mean_gain_per_site_in_declared_qsl_window",
        "mean_gain_per_site_outside_declared_qsl_window",
        "qsl_specific_enhancement_gate",
    )
})
print("N=36 connectivity ratios:",
      qsl_n36_preflight["mean_ratios_operator_over_h"])
"""
        ),
        markdown(
            r"""
At matched 160-step neural optimization, the 528-parameter exact-SU(2)/$A_1$
Operator-ViT reaches $E/N=-0.53457369(71)$, while the 650-parameter
fixed-$S^z$ ViT reaches $-0.53163(22)$.  It also lies
$1.71\times10^{-4}$ per site below the jointly optimized full-$D_6$
pair-Jastrow plus two-Lanczos baseline and only $8.25\times10^{-6}$ per site
above the stored exact anchor.  All 31 response columns remain singlets and
$\Gamma$-$A_1$ to relative residuals below $6.7\times10^{-14}$.

This is a real **finite-cluster variational win**.  It is not yet a QSL claim:
the operator arm wins throughout $J_2/J_1=0\ldots0.175$, and its mean gain is
not enhanced inside the declared frustrated window.  The sparse union of all
five scalar tokens has 1.80 times the nonzero count of one full-H action per
layer.  The matrix-free $N=36$ preflight is finite but less favorable after
two layers: it visits 3.26 times as many unique configurations and 3.18 times
as many raw second-layer entries as two-Lanczos, averaged over three fixed
half-filled configurations.  The asymptotic operator depth is still two and
the branches are batchable, but GPU wall-time and stochastic variance must be
measured directly.  Until then, the exact-small Gram-cache wall time is
deliberately excluded from any efficiency claim.
"""
        ),
        markdown(
            r"""
## 26. Physical conclusions and next promotion gates

What the enlarged local result now supports:

- in the translation sector, exact SU(2) yields a reproducible **3.38--10.05x
  parameter--accuracy compression** across square/triangular geometries and
  central/held-out frustrated couplings;
- on one open two-cluster cut, the exact six-site core retains about 80% local
  weight and the missing coupling energy is overwhelmingly repaired by
  support-aware \(q=2\) virtual pairs; translated coverings identify a
  separate finite-\(2\times6\) \(q=0\) resonance channel;
- at the same raw-parameter budget SU(2) reaches lower finite-graph energy
  errors; the mechanism and matched-work scaling are not yet established;
- a true-full-Hilbert control illustrates, but does not isolate or quantify,
  the additional burden of discovering the correct magnetization sector rather
  than assuming it;

What it does **not** mean:

- the 4x4 triangular torus has no K point, so it cannot establish 120-degree
  order, a QSL, or a phase boundary;
- exact-vector orbit gathers hide the scalable cost of character projection;
- lower energy at equal width is not automatically a parameter- or
  compute-matched win;
- the long-horizon anytime frontiers cross, but missing machine-load isolation
  makes the CPU timing panel exploratory and ineligible for a paper claim;
- the triangular A1 retained singlet space has dimension 14 and reaches the
  numerical floor in several seeds; A1 is therefore a mechanism control, not
  the primary accuracy claim;
- exact-Born i.i.d. sampling validates estimators but says nothing about MCMC
  mixing or 6x6 throughput;
- the 6x6 CPU smoke clears finite-kernel and mixing-readiness gates, but its
  random-state values are not trained energies or phase observables and the
  current SU(2) kernel has lower per-state throughput;
- a full SU(2)-invariant sampled `S(q)` action, including transverse exchange,
  now passes independent exact-small action, singlet, and reciprocal sum-rule
  tests, a sealed random-state 6x6 aggregation/cost bridge, and deterministic
  replay on the first parameter-updating pilot;
- the 20-update 6x6 pilot produces no MC-resolved energy descent or robust
  structure-factor evolution; its large SU(2)--U(1) offset is already present
  at initialization and is not an optimization win;
- the 50-update center-only scan gives more descent, but its immutable literal
  accepted-move gate returns no promotion; a separately sealed correction of
  the preregistered conditional-acceptance semantics selects square
  \(2\times10^{-3}\) and triangular \(10^{-3}\) only as a secondary
  sensitivity, not as an efficiency or physics result;

The exact-singlet estimator now adds one positive local result: on the tested
N=8 states the diagonal force lies below the work-adjusted frontier away from
the exact eigenstate, and its compiled 6x6 CPU kernel is `3.99x` (square) and
`8.16x` (triangular) faster in a non-isolated random-state profile.  At the
exact eigenstate full `H` regains zero variance.  This is a staging mechanism,
not yet a time-to-energy result.

The minimal global support-aware Spatial-CG route mixture has now passed its
wavefunction/tangent exactness gate and failed its matched-work promotion
gate.  It is frozen as an architecture control rather than promoted to 6x6.
A failure-motivated spatial candidate worth falsifying would route **locally
within multiplicity space** using support- and scale-resolved reduced tensors,
but the present negative data do not support that model yet.  It must remain
strictly multilinear in every physical spin leg and, before any GPU scale-up,
beat both the single-tree work control and multiple scrambled-support controls
on this same `4x4` protocol.  The conventional fixed-`Sz` Spatial-Attention
control is now implemented, but its nonmonotone width scaling and
geometry-dependent distance-kernel gain mean that it has not yet established
an optimizer-stable modern energy frontier.  On this validation protocol the
sealed exact-SU(2) ClebschTree context remains more parameter-efficient than
the fixed-`Sz` ViT, which motivates a strict reduced-tensor SU(2)-spatial model
after the conventional baseline is stabilized.

Promotion then requires a matched-work size transfer on square 6x6 at
\(J_2/J_1=0.5\) and triangular 6x6 at \(J_2/J_1=0.125\), using the staged
diagonal/full-H estimator only after it is validated on trained checkpoints.
Accuracy is measured by raw energy, physical full-H variance/V-score, the full
\(S(q)\), and connected dimer correlations. Triangular \(J_2=0\) is the first
ordered control. Its external
raw-energy targets at \(J_2=0\) and 0.125 are
\(-0.560313(3)\) and \(-0.515386(9)\) per site, respectively.  The 2026
compute-scaling benchmark makes \(J_2=0.125\) the stringent test: its reported
V-score exponent is \(0.40(1)\), versus \(0.54(1)\) at \(J_2=0\).

"""
        ),
        code(
            r"""
print("Triangular production targets:")
for row in tri_refs["roth_szabo_macdonald_2023"]["targets"][:2]:
    print(row)
print("\nScaling exponents:")
for row in tri_refs["rende_et_al_2026"]["triangular_scaling_exponents"]:
    print(row)
print("\nModern Spatial-Attention raw targets:")
print("square 20x20 J2/J1=0.5:",
      transformer_refs["spatial_attention_2026"]["square_20x20_j2_0p5"])
print("triangular J2=0:",
      transformer_refs["spatial_attention_2026"]["triangular_j2_0_raw_energy_per_site"])
"""
        ),
    ]
    notebook["cells"].extend(
        [
            markdown(
                r"""
## 27. N=36 physical-operator transfer: independent directions, unresolved energy gain

The exact-tested on-demand scalar operators were applied to three independent
projected-parton Born samples on the triangular periodic (6\times6) torus.
This is beyond the size at which this project repeats ED.  A crucial null check
is exact: because (H) itself is one token direction, the instantaneous
natural-gradient response of the full token space equals the Hamiltonian
direction.  A claim must therefore come from a finite-depth state, not tangent
language.

We compare the finite spaces (\mathrm{span}\{\psi,H\psi\}) and
(\mathrm{span}\{\psi,O_a\psi\}).  Coefficients are fitted on two independent
replicates and evaluated on the held-out third.  With 96 configurations per
replicate, neither the full five-token space nor the single resonance channel
pre-registered from exact (N=16) gives a resolved (N=36) energy win.  This
is a statistical **non-promotion**, not evidence that the finite-depth ansatz
fails.  By contrast, ring and six-site-loop responses remain strongly
orthogonal to the Hamiltonian tangent, so physically distinct capacity
survives and in several cases grows from (N=16) to (N=36).

Each replicate evaluates about (1.6\times10^6) unique terminal carrier
amplitudes for the ten required depth-two words in roughly one CPU minute.
The local route is therefore adequate for mechanism tests, but resolving an
energy difference of order (10^{-4}) per site requires substantially more
independent samples or accelerator training.  This panel is exploratory and
must not enter the paper notebook as an energy claim.
"""
            ),
            code(
                r"""
rows = n36_operator_transfer["aggregate"]
x = np.asarray([row["j2_over_j1"] for row in rows])
full = np.asarray([row["mean_operator_minus_one_lanczos_energy_per_site"] for row in rows])
full_se = np.asarray([row["replicate_standard_error_operator_minus_one_lanczos_per_site"] for row in rows])
single = np.asarray([row["n36_held_out_preregistered_channel_gain_per_site"] for row in rows])
single_se = np.asarray([row["n36_held_out_preregistered_channel_standard_error"] for row in rows])

fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.7))
axes[0].axhline(0.0, color="black", lw=0.9)
axes[0].errorbar(x, full, yerr=full_se, marker="o", capsize=3,
                 label="five-token, held out")
axes[0].errorbar(x, single, yerr=single_se, marker="s", capsize=3,
                 label="N=16-selected channel, held out")
axes[0].set_xlabel(r"$J_2/J_1$")
axes[0].set_ylabel(r"$(E_{\rm op}-E_{1\rm LS})/N$")
axes[0].legend(frameon=False, fontsize=8)
axes[0].set_title("No resolved N=36 energy promotion")

labels = ["rhombus_ring_exchange", "rhombus_dimer_resonance",
          "six_site_loop_rearrangement"]
pretty = ["rhombus ring", "dimer resonance", "six-site loop"]
for label, name in zip(labels, pretty):
    n16 = [row["resonance_tangent_independence"][label][
        "n16_exact_fraction_orthogonal_to_hamiltonian_tangent"] for row in rows]
    n36 = [row["resonance_tangent_independence"][label][
        "n36_mean_fraction_orthogonal_to_hamiltonian_tangent"] for row in rows]
    n36_se = [row["resonance_tangent_independence"][label][
        "n36_replicate_standard_error"] for row in rows]
    line = axes[1].plot(x, n16, marker="x", ls="--", label=f"{name}, N=16")[0]
    axes[1].errorbar(x, n36, yerr=n36_se, marker="o", ls="-",
                     color=line.get_color(), capsize=3, label=f"{name}, N=36")
axes[1].set_xlabel(r"$J_2/J_1$")
axes[1].set_ylabel(r"fraction of $O_a\psi$ orthogonal to $H\psi$")
axes[1].set_ylim(0.0, 1.05)
axes[1].set_title("Distinct resonance capacity survives scaling")
axes[1].legend(frameon=False, fontsize=7, ncol=2)
fig.tight_layout()
save_figure(fig, "fig32_n36_operator_transfer_audit")
plt.show()
"""
            ),
        ]
    )

    notebook["cells"].extend(
        [
            markdown(
                r"""
### Cross-fitted relative-flux repair: stable modes, but no new state manifold

Development chains alone fix column normalization, the spectral cutoff, and
all projection coefficients.  Three carrier modes survive with a modest
condition number, and the same frozen modes remain nonsingular on a completely
independent audit seed.  Thus the weak raw carrier direction can be removed
without destroying the relative-flux coordinates.

That repair does not create the missing physics.  After removing carrier,
one-Hamiltonian, and (H^2|pi) directions using development coefficients, only
2.15% of the two relative-mode dimer response remains on audit chains.  The
registered threshold was 5%.  Independently scrambled configuration/response
pairings are much less transferable, so the physical response is structured;
it is simply almost entirely contained in the existing low-order manifold.
The current carrier bank is therefore frozen.  This figure belongs in the
exploration ledger, not the PRL result notebook.
"""
            ),
            code(
                r"""
crossfit = relative_flux_crossfit["crossfit"]
basis = crossfit["development_frozen_basis"]
audit_gram = crossfit["audit_frozen_mode_gram"]
response = crossfit["cross_fitted_dimer_response"]
control = crossfit["scrambled_configuration_controls"]

fig, axes = plt.subplots(1, 3, figsize=(14.4, 3.8))
eigenvalues = np.asarray(basis["eigenvalues_descending"])
indices = np.arange(1, len(eigenvalues) + 1)
axes[0].semilogy(indices, eigenvalues, "o-", label="development spectrum")
axes[0].axhline(.005 * eigenvalues[0], color="black", ls="--", lw=.9,
                label="fixed cutoff")
axes[0].set(xlabel="carrier mode", ylabel="overlap eigenvalue",
            title="(a) Three modes retained")
axes[0].legend(frameon=False, fontsize=7)

audit_eigenvalues = np.asarray(audit_gram["eigenvalues"])
axes[1].plot(np.arange(1, len(audit_eigenvalues) + 1), audit_eigenvalues,
             "o-", label="audit, frozen modes")
axes[1].axhline(1.0, color="black", ls=":", lw=.9)
axes[1].set(xlabel="retained mode", ylabel="audit Gram eigenvalue",
            title="(b) Modes transfer across seeds")
axes[1].legend(frameon=False, fontsize=7)

mode_residuals = np.asarray(response["audit_residual_fractions"])
axes[2].plot(np.arange(1, len(mode_residuals) + 1), mode_residuals,
             "o-", label="physical relative modes")
axes[2].axhline(.05, color="black", ls="--", lw=.9,
                label="registered non-Krylov gate")
axes[2].axhline(control["median_audit_joint_residual_fraction"],
                color="#e45756", ls=":", lw=1.1,
                label="median scrambled control")
axes[2].set_yscale("log")
axes[2].set(xlabel="relative mode", ylabel="held-out residual fraction",
            title="(c) Structured, but 97.85% redundant")
axes[2].legend(frameon=False, fontsize=7)

fig.tight_layout()
save_figure(fig, "fig46_n36_relative_flux_crossfit_stop")
plt.show()

print("decision", relative_flux_crossfit["promotion_decision"])
print("gates", crossfit["gates"])
print("joint audit residual", response["audit_joint_residual_fraction"])
print("physical/scrambled", control["physical_to_median_scrambled_residual_ratio"])
print("sealed hash", relative_flux_crossfit["result_sha256"])
"""
            ),
        ]
    )

    notebook["cells"].extend(
        [
            markdown(
                r"""
### Triangular (6\times6) support/action preflight: sampling passes, one mode is Krylov-redundant

Fresh projected pi/0-flux Born chains show that all three full-432-image
paired-(Z_2) carriers have excellent proposal support: no sampled nodes,
ESS fractions (0.57)--(0.96), and mild projection cancellation.  The
four-carrier sampled overlap is full rank, although its condition number
(2.30\times10^3) exposes a weak relative-flux mode.

The common dimer-resonance ratio is configuration dependent for every
carrier.  Its component orthogonal to the carrier's Hamiltonian tangent is
about 23.0%, 0.76%, and 7.33%, respectively.  The preregistered gate required
at least 5% for every carrier, so it fails because the middle paired carrier's
dimer response is almost entirely an ordinary Krylov fluctuation.  This is a
representation diagnosis, not an energy result.  A matched-energy pilot and
GPU production both remain closed; the next bounded experiment should form
well-conditioned relative-flux modes and demand held-out non-Krylov response.
"""
            ),
            code(
                r"""
preflight = flux_loop_n36_preflight["preflight"]
carriers = preflight["paired_carriers"]
actions = preflight["dimer_action"]
short = ["delta=.35", "delta=.70", "J2+E2"]

fig, axes = plt.subplots(1, 3, figsize=(14.4, 3.8))
x = np.arange(3)
axes[0].plot(x, [row["importance_ess_fraction"] for row in carriers], "o-",
             label="importance ESS")
axes[0].plot(x, [row["median_projection_cancellation_ratio"] for row in carriers],
             "s-", label="median cancellation")
axes[0].set_xticks(x, short)
axes[0].set(ylim=(0, 1.05), title="(a) Projected-carrier support")
axes[0].legend(frameon=False, fontsize=7)

eigenvalues = np.asarray(preflight["sampled_carrier_bank"]["eigenvalues"])
axes[1].semilogy(np.arange(1, 5), eigenvalues, "o-")
axes[1].set(xlabel="sampled flux-mode index", ylabel="overlap eigenvalue",
            title="(b) One weak relative-flux mode")

orthogonal = [row["dimer_fraction_orthogonal_to_hamiltonian_tangent"]
              for row in actions]
axes[2].plot(x, orthogonal, "o-")
axes[2].axhline(.05, color="black", ls="--", lw=.9, label="registered gate")
axes[2].set_xticks(x, short)
axes[2].set(ylabel="non-Krylov dimer fraction",
            title="(c) Middle response is Krylov-redundant")
axes[2].legend(frameon=False, fontsize=7)

fig.tight_layout()
save_figure(fig, "fig45_n36_flux_loop_support_action_preflight")
plt.show()

print("decision", flux_loop_n36_preflight["promotion_decision"])
print("gates", preflight["gates"])
print("timing", preflight["timing"])
print("sealed hash", flux_loop_n36_preflight["result_sha256"])
"""
            ),
        ]
    )

    notebook["cells"].extend(
        [
            markdown(
                r"""
### Phase-aware carrier re-entry: a reproducible mechanism, not yet the frontier

The equal-amplitude triangular covering route failed because its overlap
average sign is only (0.018), while the square Marshall control has sign
one.  The replacement experiment therefore returns to physical spin
configurations and changes the state manifold: the pi/0-flux two-Lanczos
space is augmented by three exact-(S=0,Gamma-A_1) paired-(Z_2) carriers,
then by the same full-(A_1) local dimer-resonance action on each carrier.

Triangular (N=12) is excluded as a mechanism test: four projected flux
carriers already span its entire four-dimensional singlet (A_1) sector.
On the nontrivial triangular and square (N=16) spaces, the physical dimer
columns beat all 16 newly registered geometry scrambles in both energy and
variance.  The directional all-win probability is (2^{-16}) per geometry.
The mechanism nevertheless loses to a nine-dimensional Hamiltonian Krylov
space on both lattices.  It licenses only a bounded (6\times6) sampled
preflight, not GPU production or a variational-win claim.
"""
            ),
            code(
                r"""
suite = flux_loop_interference["suite"]
cases = (suite["triangular_n16"], suite["square_n16"])
labels = ("triangular 4x4", "square 4x4")
arms = (
    ("two_lanczos", "two-Lanczos", "o"),
    ("cross_flux_phase_bank", "cross-flux bank", "s"),
    ("phase_bank_plus_dimer_resonance", "+ dimer resonance", "D"),
    ("dimension_matched_krylov", "matched Krylov", "*"),
)

fig, axes = plt.subplots(1, 3, figsize=(14.4, 3.9))
for case, label, color in zip(cases, labels, ("#4c78a8", "#e45756"), strict=True):
    energies = [case[key]["energy_per_site"] for key, _, _ in arms]
    axes[0].plot(np.arange(len(arms)), energies, "o-", color=color, label=label)
axes[0].set_xticks(np.arange(len(arms)), [name for _, name, _ in arms], rotation=18)
axes[0].set(ylabel=r"$E/N$", title="(a) Physical descent, but Krylov remains lower")
axes[0].legend(frameon=False, fontsize=7)

for case, label, marker in zip(cases, labels, ("o", "s"), strict=True):
    physical = case["phase_bank_plus_dimer_resonance"]
    delta = np.asarray([
        physical["energy_per_site"] - row["energy_per_site"]
        for row in case["scrambled_controls"]
    ])
    axes[1].plot(np.arange(len(delta)), delta, marker=marker, label=label)
axes[1].axhline(0.0, color="black", lw=.8)
axes[1].set(xlabel="untouched scramble seed",
            ylabel=r"$(E_{phys}-E_{scr})/N$",
            title="(b) Physical geometry wins 16/16")
axes[1].legend(frameon=False, fontsize=7)

for case, label, color in zip(cases, labels, ("#4c78a8", "#e45756"), strict=True):
    for key, arm_label, marker in arms:
        row = case[key]
        axes[2].scatter(row["variance_per_site"], row["energy_per_site"],
                        marker=marker, color=color,
                        label=f"{label}: {arm_label}")
axes[2].set_xscale("log")
axes[2].set(xlabel=r"$\mathrm{Var}(H)/N$", ylabel=r"$E/N$",
            title="(c) Energy--variance frontier")
axes[2].legend(frameon=False, fontsize=5.7, ncol=2)

fig.tight_layout()
save_figure(fig, "fig44_flux_loop_interference_confirmation")
plt.show()

print("N12 saturation control", suite["triangular_n12_saturation_control"])
print("decision", suite["decision"])
print("sealed hash", flux_loop_interference["result_sha256"])
"""
            ),
        ]
    )

    notebook["cells"].extend(
        [
            markdown(
                r"""
### Representation follow-up: cell-sequential factorization is not yet a carrier solution

Replacing 15 root-first internal-node decisions by four physical
((q_i,a_i,Q_i)) cell decisions improves both fidelity and energy at nearly
matched parameter count. The complete 1430-dimensional singlet space and the
matrix-free Hamiltonian are unchanged, so this isolates the neural
factorization itself.

The wider probe shows continued capacity gain, but the exact projected
(\pi/0)-flux carrier is still not reproduced accurately when it is compressed
into the same reduced ViT. This is the decisive boundary: preserve the carrier
as a directly evaluable channel and learn a residual; do not spend GPU time on
a larger pure carrier-distillation network. The (6\times6) sampler passes,
while energy production remains blocked by carrier fidelity and scalable full
(\Gamma\!-
A_1) projection.
"""
            ),
            code(
                r"""
cell_gate = sequential_cell_gate
root = cell_gate["root_first_reference"]
matched = cell_gate["matched_cell_trials"]
capacity = cell_gate["capacity_trial"]
carrier = cell_gate["physical_carrier_trial"]
target_energy = cell_gate["target_replay"]["energy_total"]

fig, axes = plt.subplots(1, 3, figsize=(14.4, 3.8))

axes[0].plot(
    [root["parameter_count"]], [root["fidelity"]], "s",
    color="#9d755d", ms=7, label="root-first",
)
axes[0].plot(
    [row["parameter_count"] for row in matched],
    [row["fidelity"] for row in matched],
    "o", color="#4c78a8", alpha=.8, label="cell sequential, matched",
)
axes[0].plot(
    [capacity["parameter_count"]], [capacity["fidelity"]], "D",
    color="#e45756", ms=6, label="cell sequential, wider",
)
axes[0].axhline(1.0, color="black", ls="--", lw=.9, label="sealed target")
axes[0].set(xlabel="trainable parameters", ylabel="coefficient fidelity",
            title="(a) Physical factorization improves capacity")
axes[0].legend(frameon=False, fontsize=6.5)

for row, marker, color, label in (
    (root, "s", "#9d755d", "root-first"),
    (capacity, "D", "#e45756", "cell sequential, wider"),
    (carrier, "^", "#59a14f", "distilled projected carrier"),
):
    axes[1].plot(row["fidelity"], row["energy_total"] / 16, marker,
                 color=color, ms=7, label=label)
axes[1].plot(
    [row["fidelity"] for row in matched],
    [row["energy_total"] / 16 for row in matched],
    "o", color="#4c78a8", alpha=.8, label="matched seeds",
)
axes[1].axhline(target_energy / 16, color="black", ls="--", lw=.9,
                label="sealed target")
axes[1].set(xlabel="coefficient fidelity", ylabel=r"$E/N$",
            title="(b) Improvement remains far from the frontier")
axes[1].legend(frameon=False, fontsize=6.2)

gate_names = list(cell_gate["gate_results"])
gate_values = [float(cell_gate["gate_results"][name]) for name in gate_names]
axes[2].plot(gate_values, range(len(gate_names)), "o", color="#4c78a8")
axes[2].set_yticks(range(len(gate_names)),
                   [name.replace("_", " ") for name in gate_names])
axes[2].set_xticks((0, 1), ("fail", "pass"))
axes[2].set_xlim(-.15, 1.15)
axes[2].grid(axis="x", alpha=.2)
axes[2].set_title("(c) 6x6 remains fail-closed", fontsize=10)

fig.tight_layout()
save_figure(fig, "fig40_sequential_cell_factorization_gate")
plt.show()

print("matched summary", cell_gate["matched_summary"])
print("capacity", {key: capacity[key] for key in
      ("fidelity", "energy_total", "variance_total")})
print("carrier", {key: carrier[key] for key in
      ("fidelity", "energy_total", "variance_total")})
print("blocking reasons", cell_gate["blocking_reasons"])
"""
            ),
        ]
    )
    notebook["cells"].extend(
        [
            markdown(
                r"""
## 30. Kim A100 G0: warm kernels pass, short jobs do not

The complete bounded campaign consumed 0.4747 A100-hours.  Every measured
common batch passes the joint fivefold warm `Hpsi` and gradient gate.  Exact
SU(2) reaches about 10--11x acceleration at both triangular 6x6 and square
8x8, while the matched fixed-$S^z$ control accelerates still more strongly.

This does **not** establish an end-to-end efficiency claim.  GPU cold
compilation is 4--7x slower than CPU compilation, so each short GPU arm takes
2.8--6.0 times the complete CPU wall time.  The first square 8x8 multi-batch
scan also exceeded 32 GiB host memory; fixing a single production batch
reduced MaxRSS to about 22.3 GiB and completed in 7:47, with 14.99 GiB sampled
peak GPU memory.

All registered numerical outputs agree across devices, but JAX initialized
the random-normal parameters independently on CPU and GPU, giving different
bitwise parameter hashes.  The fail-closed decision therefore leaves G1
unlicensed.  A common serialized initialization and a one-batch checkpointed
training trajectory are required before any further accelerator allocation.
"""
            ),
            code(
                r"""
def trial_map(payload):
    return {row["arm_id"]: row for row in payload["cases"][0]["trials"]}

tri_cpu = trial_map(g0_cpu_triangular_n36)
tri_gpu = trial_map(g0_gpu_triangular_n36)
sq_cpu = trial_map(load_complete(
    PROJECT / "results/remote_g0_n64_single_batch_cpu_v1/square_n64/results.json"
))
sq_gpu = trial_map(g0_gpu_square_n64_single)
arms = ["fixed_sz_translation", "exact_su2_translation"]
names = {"fixed_sz_translation": "fixed-$S^z$", "exact_su2_translation": "exact SU(2)"}

fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.7))
for arm in arms:
    cpu_rows = tri_cpu[arm]["performance_profile"]
    gpu_rows = tri_gpu[arm]["performance_profile"]
    batch = [row["batch_size"] for row in cpu_rows]
    h_speed = [g["hpsi_warm_throughput_states_per_second"] /
               c["hpsi_warm_throughput_states_per_second"]
               for c, g in zip(cpu_rows, gpu_rows)]
    grad_speed = [g["gradient_warm_throughput_states_per_second"] /
                  c["gradient_warm_throughput_states_per_second"]
                  for c, g in zip(cpu_rows, gpu_rows)]
    line = axes[0].plot(batch, h_speed, marker="o", label=f"{names[arm]}, $H\\psi$")[0]
    axes[0].plot(batch, grad_speed, marker="s", ls="--", color=line.get_color(),
                 label=f"{names[arm]}, gradient")
axes[0].axhline(5.0, color="black", lw=0.9, ls=":")
axes[0].set_xlabel("batch size")
axes[0].set_ylabel("A100 / CPU warm throughput")
axes[0].set_title(r"triangular $6\times6$")
axes[0].legend(frameon=False, fontsize=7)

for arm in arms:
    c = sq_cpu[arm]["performance_profile"][0]
    g = sq_gpu[arm]["performance_profile"][0]
    axes[1].plot(
        [0, 1],
        [g["hpsi_warm_throughput_states_per_second"] / c["hpsi_warm_throughput_states_per_second"],
         g["gradient_warm_throughput_states_per_second"] / c["gradient_warm_throughput_states_per_second"]],
        marker="o", label=names[arm],
    )
axes[1].axhline(5.0, color="black", lw=0.9, ls=":")
axes[1].set_xticks([0, 1], [r"$H\psi$", "gradient"])
axes[1].set_ylabel("A100 / CPU warm throughput")
axes[1].set_title(r"square $8\times8$, batch 16")
axes[1].legend(frameon=False, fontsize=8)

for arm in arms:
    ratios = [
        tri_gpu[arm]["wall_seconds_including_compile_profile_and_sampling"] /
        tri_cpu[arm]["wall_seconds_including_compile_profile_and_sampling"],
        sq_gpu[arm]["wall_seconds_including_compile_profile_and_sampling"] /
        sq_cpu[arm]["wall_seconds_including_compile_profile_and_sampling"],
    ]
    axes[2].plot([0, 1], ratios, marker="o", label=names[arm])
axes[2].axhline(1.0, color="black", lw=0.9, ls=":")
axes[2].set_xticks([0, 1], [r"tri. $6\times6$", r"sq. $8\times8$"])
axes[2].set_ylabel("GPU / CPU complete-arm wall time")
axes[2].set_title("Cold compile dominates short canaries")
axes[2].legend(frameon=False, fontsize=8)
fig.tight_layout()
save_figure(fig, "fig35_kim_a100_g0_gate")
plt.show()
"""
            ),
        ]
    )
    notebook["cells"].extend(
        [
            markdown(
                r"""
## 28. G0 CPU denominator: the ED-intractable JAX size bridge

The remote canary now measures only accelerator-relevant JAX kernels.  Its
local denominator contains triangular $6\times6$ and square $8\times8$,
with identical fixed-$S^z$ and exact-SU(2) physical-input backbones.  The
$8\times8$ fixed-$S^z=0$ dimension is $1.83\times10^{18}$; no ED is
performed.  Both `Hpsi` and the energy gradient complete through batch 16 on
the laptop CPU.  Exact SU(2) is substantially more expensive in this
unoptimized kernel, so the GPU question is quantitative rather than assumed.

This is an engineering gate, not an article energy panel.  A GPU result is
promoted only after identical config/parameter/state hashes, numerical
agreement, and simultaneous fivefold `Hpsi` and gradient speedups at the same
batch.  Until such a result exists, this panel stays in the exploration
notebook and remote energy training remains unauthorized.
"""
            ),
            code(
                r"""
cpu_cases = {
    r"triangular $6\times6$": g0_cpu_triangular_n36,
    r"square $8\times8$": g0_cpu_square_n64,
}
styles = {
    "fixed_sz_translation": ("fixed-$S^z$", "o"),
    "exact_su2_translation": ("exact SU(2)", "s"),
}
fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.7), sharex=False)
for case_name, payload in cpu_cases.items():
    for trial in payload["cases"][0]["trials"]:
        label, marker = styles[trial["arm_id"]]
        profiles = trial["performance_profile"]
        batches = [row["batch_size"] for row in profiles]
        common = f"{case_name}, {label}"
        axes[0].plot(
            batches,
            [row["hpsi_warm_throughput_states_per_second"] for row in profiles],
            marker=marker,
            label=common,
        )
        axes[1].plot(
            batches,
            [row["gradient_warm_throughput_states_per_second"] for row in profiles],
            marker=marker,
            label=common,
        )
for axis, title in zip(axes, [r"warm $H\psi$", "warm energy gradient"]):
    axis.set_xlabel("batch size")
    axis.set_ylabel("states / second")
    axis.set_yscale("log")
    axis.set_title(title)
    axis.grid(alpha=0.2)
axes[1].legend(frameon=False, fontsize=7)
fig.tight_layout()
save_figure(fig, "fig34_g0_cpu_kernel_denominator")
plt.show()
"""
            ),
        ]
    )
    notebook["cells"].extend(
        [
            markdown(
                r"""
## 29. Preregistered-state, N=64, and square-universality gates

The first N=36 carrier-Born audit was expanded from 3 to 12 independent
replicates (1152 measurements).  It did not justify further sample-only
scaling: no five-token energy difference is resolved, and the exact-N=16
rhombus-ring choice at (J_2/J_1=0.125) trends upward in the carrier-Born
diagnostic.  We therefore fitted coefficients on that separate dataset and
sampled the **actual** one-Lanczos and preregistered-resonance states from a
common mixture.  The importance ESS is 0.998--1.000, but the frustrated-point
difference remains below one standard error.  The stripe-side dimer channel
is worse in all three replicates.  This is a clean non-promotion of a single
fixed resonance head.

The triangular 8x8 bridge is far beyond this project's ED workflow: the
fixed-Sz-zero dimension is 1.83e18.  Its exact S=0, Gamma-A1 projected carrier
uses three determinants rather than 768 Reynolds images.  Ring, dimer, and
loop responses retain components outside the Hamiltonian tangent, while one
layer evaluates about 9.0e3 unique connected amplitudes/s on the
local CPU.  These are capacity and runtime facts, not trained energies; the
8x8 torus also lacks the triangular K point.

For square 6x6, an independently validated projected Fermi-sea
carrier uses two determinants rather than 288 images.  The six-site rectangle
direction is about 95% orthogonal to the Hamiltonian tangent throughout the
ordered/frustrated/stripe scan.  However, the bare carrier is quantitatively
poor, so a remote calculation must jointly optimize a strong Jastrow/ViT
backbone and exact-scalar operator layers.  Observables are not computed for
these rejected states.
"""
            ),
            code(
                r"""
state_rows = n36_preregistered_state["couplings"]
tri64_rows = n64_operator_capacity["aggregate"]
square_rows = square_n36_operator_capacity["aggregate"]

fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.7))
x = np.asarray([row["j2_over_j1"] for row in state_rows])
y = np.asarray([row["mean_resonance_minus_lanczos_energy_per_site"] for row in state_rows])
e = np.asarray([row["replicate_standard_error_difference_per_site"] for row in state_rows])
axes[0].axhline(0.0, color="black", lw=0.9)
axes[0].errorbar(x, y, yerr=e, marker="o", capsize=3, color="tab:red")
axes[0].set_xlabel(r"$J_2/J_1$")
axes[0].set_ylabel(r"$(E_{\rm res}-E_{1\rm LS})/N$")
axes[0].set_title(r"$6\times6$: actual-state non-promotion")

tri_labels = ["rhombus_ring_exchange", "rhombus_dimer_resonance",
              "six_site_loop_rearrangement"]
square_labels = ["plaquette_ring_exchange", "plaquette_dimer_resonance",
                 "six_site_rectangle_rearrangement"]
for label in tri_labels:
    axes[1].errorbar(
        [row["j2_over_j1"] for row in tri64_rows],
        [row["resonance_tangent_independence"][label]["mean_fraction_orthogonal_to_hamiltonian_tangent"] for row in tri64_rows],
        yerr=[row["resonance_tangent_independence"][label]["replicate_standard_error"] for row in tri64_rows],
        marker="o", capsize=3, label=label.replace("_", " "),
    )
for label in square_labels:
    axes[2].errorbar(
        [row["j2_over_j1"] for row in square_rows],
        [row["resonance_tangent_independence"][label]["mean_fraction_orthogonal_to_hamiltonian_tangent"] for row in square_rows],
        yerr=[row["resonance_tangent_independence"][label]["replicate_standard_error"] for row in square_rows],
        marker="o", capsize=3, label=label.replace("_", " "),
    )
for axis, title in zip(axes[1:], [r"triangular $8\times8$ capacity", r"square $6\times6$ capacity"]):
    axis.set_xlabel(r"$J_2/J_1$")
    axis.set_ylabel(r"fraction orthogonal to $H\psi$")
    axis.set_ylim(0.0, 1.05)
    axis.set_title(title)
    axis.legend(frameon=False, fontsize=7)
fig.tight_layout()
save_figure(fig, "fig33_operator_scaling_and_square_gate")
plt.show()
"""
            ),
        ]
    )
    # Keep the late production-gate sections in physical-to-engineering order
    # even though they are assembled by independent result blocks above.
    late_titles = (
        "## 28. G0 CPU denominator",
        "## 29. Preregistered-state",
        "## 30. Kim A100 G0",
    )
    late_blocks = {}
    late_starts = []
    for title in late_titles:
        index = next(
            i
            for i, cell in enumerate(notebook["cells"])
            if cell.cell_type == "markdown" and cell.source.lstrip().startswith(title)
        )
        late_starts.append(index)
        late_blocks[title] = notebook["cells"][index : index + 2]
    late_start, late_end = min(late_starts), max(late_starts) + 2
    if late_end - late_start != 6:
        raise RuntimeError("late production-gate notebook blocks are not contiguous")
    notebook["cells"][late_start:late_end] = [
        cell for title in late_titles for cell in late_blocks[title]
    ]

    notebook["cells"].extend(
        [
            markdown(
                r"""
## 31. Symmetry-labelled response loops: exact mechanism, failed seed cure

A nontrivial point-group operator cannot be mixed linearly into an (A_1)
ground state.  The tested mechanism instead closes a virtual response loop,

\[
K_\Gamma=\sum_\mu O_{\Gamma,\mu}^\dagger O_{\Gamma,\mu},
\qquad A_1\rightarrow\Gamma\rightarrow A_1.
\]

The triangular (E_2/A_2) and square (B_1/B_2/A_2) implementations pass
singlet, irrep-covariance, scalar-return, Hermiticity, and matched-spectrum
checks below (1.2\times10^{-13}).  The five-seed joint-channel development
does not beat the original overlapping bank or the equal-dimension isospectral
scramble.  A post-hoc (B_2) ablation improves four of five development seeds,
so it was frozen and tested on five untouched seeds.

The audit rejects the stronger claim.  (B_2) reduces the baseline worst
error from (8.75\times10^{-5}) to (2.50\times10^{-5}) and improves the
median response-moment error relative to baseline, but wins only 3/5 baseline
pairs and 2/5 scrambled pairs; the scramble also has lower median variance.
An exact post-hoc SMA diagnosis gives gaps (0.547,3.857,2.968) for
(B_1,B_2,A_2), respectively.  Thus (B_2) is a small-weight high-energy
error filter, not a soft mode, and fixed response loops do not remove basin
dependence.  This panel remains exploration-only and does not license (N=36).
"""
            ),
            code(
                r"""
development = symmetry_response_development["suite"]["square_n16_j2_05"]
audit = square_b2_response_audit["suite"]

fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))

development_arms = {
    "physical_overlapping_baseline": ("overlap baseline", "o", "#4c78a8"),
    "B2_J2_response_augmented": (r"+$B_2$ return", "s", "#e45756"),
    "all_labelled_response_augmented": ("all labelled", "^", "#72b7b2"),
    "all_scrambled_response_augmented": ("all scrambled", "D", "#777777"),
}
for arm, (label, marker, color) in development_arms.items():
    errors = development["summary"][arm]["all_seed_energy_errors_total"]
    axes[0, 0].semilogy(
        range(1, 6), np.maximum(errors, 1e-14),
        marker=marker, color=color, label=label,
    )
axes[0, 0].set(xlabel="development seed index", ylabel="total-energy error",
               title="(a) Joint response bank does not stabilize seeds")
axes[0, 0].legend(frameon=False, fontsize=8)

audit_arms = {
    "physical_overlapping_baseline": ("overlap baseline", "o", "#4c78a8"),
    "B2_J2_response_augmented": (r"+$B_2$ return", "s", "#e45756"),
    "scrambled_B2_response_augmented": (r"+scrambled $B_2$", "D", "#777777"),
}
for arm, (label, marker, color) in audit_arms.items():
    errors = audit["summary"][arm]["all_seed_energy_errors_total"]
    axes[0, 1].semilogy(
        range(1, 6), np.maximum(errors, 1e-14),
        marker=marker, color=color, label=label,
    )
axes[0, 1].axhline(1e-6, color="black", lw=0.8, ls="--")
axes[0, 1].set(xlabel="untouched audit seed index", ylabel="total-energy error",
               title=r"(b) $B_2$ shrinks the tail but loses the control")
axes[0, 1].legend(frameon=False, fontsize=8)

for arm, (label, marker, color) in audit_arms.items():
    values = [row["response_moments"]["median_m0_relative_error"]
              for row in audit["runs"][arm]]
    axes[1, 0].semilogy(range(1, 6), values, marker=marker, color=color, label=label)
axes[1, 0].set(xlabel="untouched audit seed index",
               ylabel=r"median relative $m_0$ error",
               title="(c) Response repair is not label-specific")

summary_metrics = ("maximum_energy_error_total", "seed_range_energy_error_total")
for arm, (label, marker, color) in audit_arms.items():
    values = [audit["summary"][arm][metric] for metric in summary_metrics]
    axes[1, 1].semilogy((0, 1), values, marker=marker, color=color, label=label)
axes[1, 1].set_xticks((0, 1), ("worst seed", "seed range"))
axes[1, 1].set_ylabel("total-energy error")
axes[1, 1].set_title("(d) Tail reduction alone is insufficient")

for axis in axes.flat:
    axis.grid(alpha=0.2)
fig.tight_layout()
save_figure(fig, "fig35_symmetry_response_seed_falsification")
plt.show()
"""
            ),
        ]
    )

    notebook["cells"].extend(
        [
            markdown(
                r"""
## 32. Configuration dependence is not the cure; virtual loop sectors are diagnostic

The next frozen test allowed the response amplitude to depend on the complete
reduced local-Schur configuration.  The gate reads only causal (q,a) and
spatial tokens, is projected into exact singlet-(A_1) coordinates, and is
coupled to each response return through a Hermitian anticommutator.  Its
control is orthogonally conjugated at every parameter point, so instantaneous
operator spectra, parameter count, and gate values are exactly matched.

This added freedom does not cure the optimization basins.  The physical arm
wins only 3/5 paired seeds against the pointwise-isospectral control; its
median total-energy error is (2.82\times10^{-6}), compared with
(7.26\times10^{-8}) for the fixed response bank and
(1.41\times10^{-7}) for the original overlapping baseline.  This closes the
interpretation that the fixed channels merely lacked configuration-dependent
weights.

A distinct triangular (4\times4) screen then resolves rhombus ring and dimer
resonances into (E_2) doublets and the directed ring current into an (A_2)
pseudoscalar before returning them to (A_1).  On all three previously frozen
SU(2)-Operator-ViT states, this labelled bank beats the existing uniform
(A_1) loop bank and captures (7.7\%-13.0\%) of the instantaneous energy
residual, versus only (0.23\%-0.58\%) for the uniform loops.  Nevertheless a
same-dimension degree-4 Krylov space remains decisively lower in energy and
variance.  The emerging picture is therefore physical but not yet a
variational win: virtual nontrivial spatial sectors diagnose the missing
fluctuations, while this shallow return mechanism does not organize them as
efficiently as the Hamiltonian residual itself.  Both panels remain
exploration-only.
"""
            ),
            code(
                r"""
conditioned = conditioned_response_development["suite"]["square_n16_j2_05"]
loop = symmetry_labelled_loop_screen["suite"]

fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))

conditioned_arms = {
    "physical_overlapping_baseline": ("overlap baseline", "o", "#4c78a8"),
    "fixed_labelled_response": ("fixed response", "s", "#72b7b2"),
    "conditioned_labelled_response": ("conditioned response", "^", "#e45756"),
    "conditioned_isospectral_scramble": ("pointwise scramble", "D", "#777777"),
}
for arm, (label, marker, color) in conditioned_arms.items():
    errors = conditioned["summary"][arm]["all_seed_energy_errors_total"]
    axes[0].semilogy(range(1, 6), np.maximum(errors, 1e-14), marker=marker,
                    color=color, label=label)
axes[0].set(xlabel="fresh development seed", ylabel="total-energy error",
            title="(a) Learned response gate is less robust")
axes[0].legend(frameon=False, fontsize=7)

seed_index = [row["frozen_seed_index"] + 1 for row in loop["rows"]]
axes[1].plot(seed_index,
             [row["labelled_residual_capture_fraction"] for row in loop["rows"]],
             marker="o", label=r"$E_2/A_2$ virtual loops")
axes[1].plot(seed_index,
             [row["existing_a1_residual_capture_fraction"] for row in loop["rows"]],
             marker="s", label=r"uniform $A_1$ loops")
axes[1].set(xlabel="frozen SU(2)-ViT seed", ylabel="residual capture fraction",
            title="(b) Nontrivial sectors carry missing physics")
axes[1].legend(frameon=False, fontsize=8)

labelled_gap = []
combined_gap = []
for row in loop["rows"]:
    ritz = row["ritz"]
    labelled_gap.append(
        ritz["labelled_E2_A2_loop_bank"]["energy_per_site"]
        - ritz["matched_degree4_krylov"]["energy_per_site"]
    )
    combined_gap.append(
        ritz["combined_A1_and_labelled_loops"]["energy_per_site"]
        - ritz["matched_degree7_krylov"]["energy_per_site"]
    )
axes[2].plot(seed_index, labelled_gap, marker="o", label="labelled vs degree-4")
axes[2].plot(seed_index, combined_gap, marker="s", label="combined vs degree-7")
axes[2].axhline(0.0, color="black", lw=0.8, ls="--")
axes[2].set(xlabel="frozen SU(2)-ViT seed",
            ylabel=r"$E_{physical}-E_{Krylov}$ per site",
            title="(c) Krylov remains the variational frontier")
axes[2].legend(frameon=False, fontsize=8)

for axis in axes:
    axis.grid(alpha=0.2)
fig.tight_layout()
save_figure(fig, "fig36_conditioned_and_loop_response_verdict")
plt.show()
"""
            ),
        ]
    )

    notebook["cells"].extend(
        [
            markdown(
                r"""
## 33. Coupling validation: the (A_2) loop signal is generic, not QSL-selective

The (J_2/J_1=0.125) observation above generated a frozen hypothesis.  The
same exact-SU(2)/(A_1) Operator-ViT was then trained with three entirely new
seeds at seven couplings.  The discovery point is shown but excluded from the
phase-contrast gate; the held-out frustrated set is
(0.075,0.10,0.15), and the ordered-side control is (0,0.05,0.175).

The channel itself is reproducible: (A_2) is largest in all three new
discovery-point runs and 16/18 held-out runs.  It is not phase-selective.
The frustrated/ordered ratios are (0.899) for residual capture, (0.972) for
the SMA gap, and only (1.024) for the SMA susceptibility proxy.  The smooth
increase of the proxy follows a nearly monotonic decrease of the SMA energy
scale rather than a peak in the putative QSL window.  Moreover the (A_2)
return loses to matched degree-2 Krylov in all 18 held-out comparisons, and
the full labelled bank loses to degree-4 Krylov in all 18.

Thus virtual loop-current fluctuations are a robust short-range property of
the triangular model, not a finite-size signature that distinguishes the QSL
in this ansatz.  This closes the response-return route as both a central
physical claim and an energy architecture.  The plotted quantity
(2m_0/Delta_SMA) is explicitly an SMA proxy, not an exact thermodynamic
susceptibility.
"""
            ),
            code(
                r"""
scan = symmetry_loop_coupling_scan["suite"]
summary = scan["coupling_summary"]
x = np.asarray([row["j2_over_j1"] for row in summary])

fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
for axis in axes:
    axis.axvspan(0.075, 0.15, color="#f2cf5b", alpha=0.18,
                 label="declared frustrated window")
    axis.axvline(0.125, color="#555555", lw=0.9, ls="--")

axes[0].plot(x, [row["mean_A2_residual_capture"] for row in summary],
             marker="o", label=r"$A_2$ directed ring")
axes[0].plot(x, [row["mean_E2_ring_residual_capture"] for row in summary],
             marker="s", label=r"$E_2$ ring")
axes[0].plot(x, [row["mean_E2_dimer_residual_capture"] for row in summary],
             marker="^", label=r"$E_2$ dimer")
axes[0].set(xlabel=r"$J_2/J_1$", ylabel="mean residual capture",
            title=r"(a) $A_2$ dominates but does not peak in QSL")
axes[0].legend(frameon=False, fontsize=7)

axes[1].plot(x, [row["mean_A2_sma_gap"] for row in summary],
             marker="o", color="#e45756")
axes[1].set(xlabel=r"$J_2/J_1$", ylabel=r"$A_2$ SMA gap",
            title="(b) Smooth response-scale softening")

axes[2].plot(
    x,
    [row["mean_A2_sma_susceptibility_proxy_per_site"] for row in summary],
    marker="o", color="#4c78a8",
)
axes[2].set(xlabel=r"$J_2/J_1$", ylabel=r"$2m_0/(N\Delta_{SMA})$",
            title="(c) No phase-selective response peak")

for axis in axes:
    axis.grid(alpha=0.2)
fig.tight_layout()
save_figure(fig, "fig37_A2_loop_coupling_falsification")
plt.show()
"""
            ),
        ]
    )

    notebook["cells"].extend(
        [
            markdown(
                r"""
## 34. State-manifold change: complete triangular-4x4 multiplicity Spatial-ViT

The response-head route is closed above.  The next experiment changes the
entire variational manifold.  A physical-basis tomography of the frozen
states first shows that optimized two-Lanczos and exact-SU(2)/(A_1)
Operator-ViT already have the exact ground-state sign pattern and no nodes;
their residual error is in configuration-dependent magnitudes.

Four complete (2\times2) rhombus Schur cells span all 1430 (N=16)
singlets.  A causal Spatial-ViT acts only on reduced ((q,a)) and inter-cell
spin labels, followed by the exact 192-element (Gamma)-(A_1) projector.
The anchored arm adds an orthogonal learned state to the frozen Operator-ViT,
so the carrier is an optimization origin rather than a nodal constraint.

All three preregistered fresh seeds beat the frozen Operator-ViT by 160
updates.  At 1000 updates the anchored arm reaches the exact finite-cluster
state for every seed, whereas one standalone seed remains at a
(2.8\times10^{-6}) total-energy error.  This identifies the physical carrier
as a basin stabilizer and the full multiplicity field as the missing
variational degree of freedom.  It does not yet establish sampled scaling.
"""
            ),
            code(
                r"""
manifold = triangular_n16_state_manifold["suite"]
exact_energy = manifold["exact_ground"]["energy_total"]
operator = manifold["frozen_operator_vit_comparator"]

fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.7))
colors = {
    "standalone_complete_multiplicity_vit": "#4c78a8",
    "operator_anchored_complete_multiplicity_vit": "#e45756",
}
for arm, color in colors.items():
    for run in manifold["runs"][arm]:
        history = run["optimizer"]["history"]
        axes[0].semilogy(
            [row["step"] for row in history],
            np.maximum(
                [row["loss_before_update"] - exact_energy for row in history],
                1e-14,
            ),
            "o-", ms=2.7, lw=.9, alpha=.65, color=color,
            label=arm.replace("_", " ") if run["seed"] == 2111 else None,
        )
axes[0].axhline(operator["energy_error_total"], color="#9d755d", ls="--",
                lw=1.2, label="frozen Operator-ViT")
axes[0].set(xlabel="updates", ylabel=r"total $E-E_0$",
            title="(a) independent seed trajectories")
axes[0].legend(frameon=False, fontsize=6.5)

audit = manifold["frozen_state_bottleneck_audit"]
names = (
    "optimized_pi_flux_jastrow_two_lanczos",
    "pi_flux_fixed_sz_gamma_a1_spatial_vit",
    "pi_flux_exact_su2_a1_operator_vit",
)
for index, name in enumerate(names):
    row = audit[name]
    axes[1].plot(
        (0, 1, 2),
        np.maximum(
            (row["energy_error_total"],
             row["exact_phase_candidate_magnitude_energy_error_total"],
             row["candidate_phase_exact_magnitude_energy_error_total"]),
            1e-14,
        ),
        "o-", label=name.replace("_", " "),
    )
axes[1].set_yscale("log")
axes[1].set_xticks((0, 1, 2), ("actual", "amplitude-only", "phase-only"))
axes[1].set(ylabel=r"total $E-E_0$", title="(b) phase--magnitude tomography")
axes[1].legend(frameon=False, fontsize=5.8)

anchored_runs = manifold["runs"]["operator_anchored_complete_multiplicity_vit"]
axes[2].plot(
    [row["seed"] for row in anchored_runs],
    [row["final_residual_scale"] for row in anchored_runs],
    "o-", color="#e45756", label="learned residual scale",
)
axes[2].axhline(0.0, color="black", lw=.8)
axes[2].set(xlabel="fresh seed", ylabel="additive residual scale",
            title="(c) carrier is left in all basins")
axes[2].legend(frameon=False, fontsize=7)
fig.tight_layout()
save_figure(fig, "fig38_complete_multiplicity_state_manifold_audit")
plt.show()

print("promotion gates", manifold["promotion_gates"])
print("summary", manifold["summary"])
"""
            ),
        ]
    )

    notebook["cells"].extend(
        [
            markdown(
                r"""
## 35. Many-cell matrix-free replay: operator pass, neural scale gate fail

The sealed triangular (4\times4) state is changed exactly from the local
(D_4)-labelled Schur basis to a physical many-cell fusion tree. The tree
labels are local ((q,a)) plus inter-cell doubled spins. Local F-move/
recoupling columns reproduce every amplitude, the energy, and the variance
without constructing a fusion-space Hamiltonian.

This must be separated from the learned scaling question. A shared-parameter
Spatial-ViT can sample the nine-cell (6\times6) tree, but at the registered
local budget it does not distill the sealed coefficient field. Full scalable
(Gamma)-(A_1) projection is also missing. Consequently the GPU production
gate remains closed; the negative result belongs in exploration, not in the
paper notebook.
"""
            ),
            code(
                r"""
matrix_free = many_cell_matrix_free
replay = matrix_free["n16_exact_replay"]
distilled = matrix_free["transferable_vit_distillation"]

fig, axes = plt.subplots(1, 3, figsize=(14.5, 3.8))

names = ("max |delta psi|", "relative L2", "|delta E|", "variance")
values = (
    replay["physical_amplitude_max_abs_error"],
    replay["physical_amplitude_relative_l2_error"],
    replay["matrix_free_vs_sealed_energy_abs_error"],
    replay["matrix_free_variance_total"],
)
axes[0].plot(range(4), values, "o-", color="#4c78a8")
axes[0].set_yscale("log")
axes[0].set_xticks(range(4), names, rotation=25, ha="right")
axes[0].set(title="(a) Exact representation/operator replay")

sealed_per_site = replay["sealed_energy_total"] / 16
distilled_per_site = distilled["energy_total"] / 16
axes[1].axhline(sealed_per_site, color="black", ls="--", lw=1,
                label="sealed target")
axes[1].plot((0, 1), (sealed_per_site, distilled_per_site), "o-",
             color="#e45756")
axes[1].set_xticks((0, 1), ("exact carrier", "shared ViT"))
axes[1].set(ylabel=r"$E/N$", title="(b) Neural coefficient bottleneck")
axes[1].legend(frameon=False, fontsize=7)

gate_names = list(matrix_free["gate_results"])
gate_values = [float(matrix_free["gate_results"][name]) for name in gate_names]
axes[2].plot(gate_values, range(len(gate_names)), "o", color="#59a14f")
axes[2].set_yticks(
    range(len(gate_names)), [name.replace("_", " ") for name in gate_names]
)
axes[2].set_xticks((0, 1), ("fail", "pass"))
axes[2].set_xlim(-0.15, 1.15)
axes[2].set_title("(c) Fail-closed 6x6 authorization", fontsize=10)
axes[2].grid(axis="x", alpha=.2)

fig.tight_layout()
save_figure(fig, "fig39_many_cell_matrix_free_gate")
plt.show()

print("N36 production authorized:", matrix_free["n36_gpu_production_authorized"])
print("blocking reasons:", matrix_free["blocking_reasons"])
"""
            ),
        ]
    )

    notebook["cells"].extend(
        [
            markdown(
                r"""
## 36. Overlapping-cell locality calibration: rejected by its matched control

This calibration preserves the projected (\pi/0)-flux carrier and acts only
inside the exact (S=0,\Gamma-A_1) space. Its physical operators average four
translated local-Schur frames; the control applies a common Haar conjugation,
so the two banks have identical spectra, parameter count, initialization, and
optimization work.

The physical circuit beats optimized Jastrow plus two Lanczos, but loses to
both the sealed Operator-ViT and the paired isospectral scramble. The reduced
space has dimension only 14 while the circuit has 16 parameters, and the
scramble reaches the exact state. Therefore this is a small-space universality
artifact, not evidence that scrambled nonlocal operators are physical. The
full coupling scan and (6\times6) promotion are stopped.
"""
            ),
            code(
                r"""
cell_scalar = carrier_cell_scalar_circuit["frontier"]
summaries = cell_scalar["coupling_summaries"]
couplings = np.asarray(sorted(float(value) for value in summaries))

def summary(coupling):
    return summaries[str(coupling)]

physical = np.asarray([
    summary(value)["arms"]["physical_overlapping_cell_circuit"]["median_energy_per_site"]
    for value in couplings
])
scrambled = np.asarray([
    summary(value)["arms"]["isospectral_scrambled_cell_circuit"]["median_energy_per_site"]
    for value in couplings
])
exact = np.asarray([summary(value)["exact_anchor_energy_per_site"] for value in couplings])
operator = np.asarray([
    summary(value)["references"]["sealed_operator_vit_energy_per_site"]
    for value in couplings
])
jastrow = np.asarray([
    summary(value)["references"]["optimized_jastrow_two_lanczos_energy_per_site"]
    for value in couplings
])

fig, axes = plt.subplots(1, 3, figsize=(14.4, 3.8))
for values, marker, label in (
    (exact, "_", "exact anchor"),
    (scrambled, "x", "isospectral scramble"),
    (operator, "s", "sealed Operator-ViT"),
    (physical, "o", "physical Schur-locality"),
    (jastrow, "^", "Jastrow + two Lanczos"),
):
    axes[0].plot(couplings, values, marker=marker, label=label)
axes[0].set(xlabel=r"$J_2/J_1$", ylabel=r"$E/N$",
            title="(a) Physical arm does not win")
axes[0].legend(frameon=False, fontsize=6.3)

for coupling in couplings:
    differences = summary(coupling)["paired_physical_minus_scrambled_energy_per_site"]
    axes[1].plot([coupling] * len(differences), differences, "o", alpha=.8)
axes[1].axhline(0.0, color="black", lw=.8)
axes[1].set(xlabel=r"$J_2/J_1$", ylabel=r"$(E_{phys}-E_{scr})/N$",
            title="(b) Scramble wins every paired seed")

runs = cell_scalar["runs"]
for coupling, color in zip(couplings, ("#4c78a8", "#e45756"), strict=True):
    candidates = [row for row in runs if row["j2_over_j1"] == coupling and
                  row["arm_id"] == "physical_overlapping_cell_circuit"]
    best = min(candidates, key=lambda row: row["metrics"]["energy_per_site"])
    channels = best["metrics"]["channel_diagnostics"]["local_cell_q_weights"]
    q = np.asarray(sorted(int(label[1:]) for label in channels))
    values = np.asarray([channels[f"q{value}"] for value in q])
    axes[2].plot(q / 2, values, "o-", color=color, label=fr"$J_2/J_1={coupling:g}$")
axes[2].set(xlabel="local cell spin j", ylabel="probability weight",
            title="(c) Physical channel readout (diagnostic only)")
axes[2].legend(frameon=False, fontsize=7)

fig.tight_layout()
save_figure(fig, "fig41_carrier_cell_scalar_route_falsification")
plt.show()

print("decision", carrier_cell_scalar_circuit["decision"])
print("gates", cell_scalar["gates"])
"""
            ),
        ]
    )

    notebook["cells"].extend(
        [
            markdown(
                r"""
## 37. Symmetry-built-in RVB covering attention: local promotion gate

This is the first result from the post-Reynolds representation.  Every dimer
covering is an exact singlet and signed covering-orbit columns lie exactly in
the physical \\(\Gamma-A_1\\) sector.  The learned model sees only
space-group labels of bonds and bond pairs, followed by invariant pooling.

The left panel is a state-manifold falsifier: the complete nearest-neighbour
RVB orbit bank adds low-energy directions beyond the carrier two-Lanczos
space on both geometries.  The center and right panels are a causal
compression test at identical parameters, seeds, and 500-update work.  The
control reassigns whole relation-feature rows among the same physical orbit
columns, so it preserves the exact variational span and both symmetries.

The square paired-energy result passes; the variance improvement is a trend,
not an all-seed result.  This licenses a matrix-free \\(6\times6\\)
covering-sampling throughput canary, not a QSL or large-size energy claim.
"""
            ),
            code(
                r"""
rvb_oracle = rvb_covering["oracle"]
rvb_attention = rvb_covering["attention"]
case_keys = ("triangular_n12", "square_n16")
case_labels = ("triangular N=12", "square N=16")
x = np.arange(2)

fig, axes = plt.subplots(1, 3, figsize=(14.4, 3.9))
for field, marker, label in (
    ("carrier", "o", "carrier"),
    ("two_lanczos", "s", "two-Lanczos"),
    ("rvb_a1_oracle", "D", "NN-RVB oracle"),
    ("two_lanczos_plus_rvb_a1_oracle", "*", "two-Lanczos + RVB"),
):
    values = [rvb_oracle[key][field]["energy_per_site"] for key in case_keys]
    axes[0].plot(x, values, marker=marker, label=label)
axes[0].set_xticks(x, case_labels, rotation=12)
axes[0].set(ylabel=r"$E/N$", title="(a) RVB adds non-Krylov directions")
axes[0].legend(frameon=False, fontsize=7)

for key, marker, label in zip(case_keys, ("o", "s"), case_labels, strict=True):
    differences = np.asarray(
        rvb_attention[key]["paired_physical_minus_scrambled_per_site"]
    )
    axes[1].plot(np.arange(len(differences)), differences, marker=marker,
                 label=label)
axes[1].axhline(0.0, color="black", lw=.8)
axes[1].set(xlabel="paired seed", ylabel=r"$(E_{phys}-E_{scr})/N$",
            title="(b) Physical relation labels optimize faster")
axes[1].legend(frameon=False, fontsize=7)

square = rvb_attention["square_n16"]
for arm, marker, label in (
    ("physical_relation_attention", "o", "physical relations"),
    ("scrambled_relation_attention", "x", "feature reassignment"),
):
    rows = square["runs"][arm]
    axes[2].scatter(
        [row["metrics"]["variance_total"] for row in rows],
        [row["metrics"]["energy_per_site"] for row in rows],
        marker=marker, label=label,
    )
axes[2].scatter(
    [square["two_lanczos"]["variance_total"]],
    [square["two_lanczos"]["energy_per_site"]],
    marker="*", s=95, label="two-Lanczos",
)
axes[2].set(xlabel=r"$\mathrm{Var}(H)$", ylabel=r"$E/N$",
            title="(c) Square energy--variance frontier")
axes[2].legend(frameon=False, fontsize=7)

fig.tight_layout()
save_figure(fig, "fig42_rvb_covering_attention_frontier")
plt.show()

print("decision", rvb_covering["decision"])
print("maximum spin residual", rvb_attention["maximum_spin_raising_relative_norm"])
"""
            ),
        ]
    )

    notebook["cells"].extend(
        [
            markdown(
                r"""
### Configuration-dependent neural Pfaffian: geometry is real, but the manifold loses

The complete triangular `N=16`, `S=0,Gamma-A1` support provides a strict stop
gate for a configuration-dependent singlet Pfaffian with neural backflow.
Physical J1/J2 relation labels outperform site-scrambled controls, so the
network does detect genuine lattice correlations.  That signal survives when
the projected-VMC/two-Lanczos carrier is nested as an additive channel and
when several independently optimized Pfaffian states are coherently mixed.

It is not a variational win.  The strongest five-state physical bank remains
above both the sealed Operator-ViT and rank-five Krylov in energy and
variance.  The comparison therefore identifies a state-manifold mismatch,
not a symmetry leak or a single bad optimization basin.  This is a negative
exploration result; it is intentionally excluded from the article notebook
and does not authorize triangular `6x6`.
"""
            ),
            code(
                r"""
direct = neural_backflow_pfaffian["suite"]
residual = residual_neural_backflow["suite"]
resonating = resonating_neural_pfaffian["suite"]

labels = ["direct J1", "direct J1+J2", "residual J1",
          "residual J1+J2", "coherent bank"]
energies = np.asarray([
    direct["models"]["compact_j1"]["physical_summary"]["median_energy_per_site"],
    direct["models"]["extended_j1j2"]["physical_summary"]["median_energy_per_site"],
    residual["models"]["compact_j1"]["physical_summary"]["median_energy_per_site"],
    residual["models"]["extended_j1j2"]["physical_summary"]["median_energy_per_site"],
    resonating["banks"]["combined"]["physical"]["energy_per_site"],
])
variances = np.asarray([
    direct["models"]["compact_j1"]["physical_summary"]["median_variance_per_site"],
    direct["models"]["extended_j1j2"]["physical_summary"]["median_variance_per_site"],
    residual["models"]["compact_j1"]["physical_summary"]["median_variance_per_site"],
    residual["models"]["extended_j1j2"]["physical_summary"]["median_variance_per_site"],
    resonating["banks"]["combined"]["physical"]["variance_per_site"],
])
baselines = direct["frozen_baselines"]
baseline_labels = ["projected VMC + 2LS", "fixed-Sz ViT", "Operator-ViT"]
baseline_keys = ["projected_vmc_two_lanczos", "fixed_sz_spatial_vit",
                 "exact_su2_operator_vit"]
baseline_energies = np.asarray([baselines[key]["energy_per_site"]
                                for key in baseline_keys])
baseline_variances = np.asarray([baselines[key]["variance_per_site"]
                                 for key in baseline_keys])

fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.2))
x = np.arange(len(labels))
axes[0].plot(x, energies, "o-", color="#4c78a8", label="neural Pfaffian")
for value, name, color, linestyle in zip(
    baseline_energies, baseline_labels,
    ("#f58518", "#54a24b", "#e45756"), ("--", "-.", ":"), strict=True
):
    axes[0].axhline(value, color=color, ls=linestyle, lw=1.05, label=name)
axes[0].set_xticks(x, labels, rotation=25, ha="right")
axes[0].set(ylabel=r"$E/N$", title="(a) Better carrier handling is not enough")
axes[0].legend(frameon=False, fontsize=6.8)

axes[1].plot(variances, energies, "o-", color="#4c78a8",
             label="neural Pfaffian sequence")
annotation_offsets = ((4, 3), (4, 3), (-10, 12), (5, -12), (-42, 8))
for index, (name, offset) in enumerate(zip(labels, annotation_offsets, strict=True)):
    axes[1].annotate(name, (variances[index], energies[index]),
                     xytext=offset, textcoords="offset points", fontsize=6.6)
for variance, energy, label, marker, color in zip(
    baseline_variances, baseline_energies, baseline_labels,
    ("s", "^", "*"), ("#f58518", "#54a24b", "#e45756"), strict=True
):
    axes[1].scatter(variance, energy, marker=marker, color=color,
                    s=75, label=label)
axes[1].set_xscale("log")
axes[1].set(xlabel=r"$\mathrm{Var}(H)/N$", ylabel=r"$E/N$",
            title="(b) Energy--variance frontier")
axes[1].legend(frameon=False, fontsize=6.8)

comparison_labels = ["direct J1+J2", "residual J1+J2", "coherent bank"]
physical = np.asarray([
    energies[1], energies[3], energies[4],
])
scrambled = np.asarray([
    direct["models"]["extended_j1j2"]["scrambled_summary"]["median_energy_per_site"],
    residual["models"]["extended_j1j2"]["scrambled_summary"]["median_energy_per_site"],
    resonating["banks"]["combined"]["scrambled"]["energy_per_site"],
])
krylov = np.asarray([
    direct["models"]["extended_j1j2"]["same_carrier_dimension_matched_krylov"]["energy_per_site"],
    residual["models"]["extended_j1j2"]["same_anchor_dimension_matched_krylov"]["energy_per_site"],
    resonating["banks"]["combined"]["same_anchor_rank_matched_krylov"]["energy_per_site"],
])
xc = np.arange(len(comparison_labels))
axes[2].axhline(0.0, color="black", lw=.8)
axes[2].plot(xc, physical - scrambled, "o-", label="physical - scramble")
axes[2].plot(xc, physical - krylov, "s-", label="physical - matched Krylov")
axes[2].set_xticks(xc, comparison_labels, rotation=20, ha="right")
axes[2].set(ylabel=r"energy difference per site",
            title="(c) Geometry helps; Krylov still wins")
axes[2].legend(frameon=False, fontsize=7)

fig.tight_layout()
save_figure(fig, "fig47_n16_neural_backflow_pfaffian_stop")
plt.show()

print("direct decision", neural_backflow_pfaffian["promotion_decision"])
print("residual decision", residual_neural_backflow["promotion_decision"])
print("resonating decision", resonating_neural_pfaffian["promotion_decision"])
print("result hashes", neural_backflow_pfaffian["result_sha256"],
      residual_neural_backflow["result_sha256"],
      resonating_neural_pfaffian["result_sha256"])
"""
            ),
        ]
    )

    notebook["cells"].extend(
        [
            markdown(
                r"""
### Phase-aware long-range loop-RVB: physical mechanism, final frontier stop

The final spin-only carrier contingency replaces pair-kernel backflow by a
valence-bond loop gas.  It contains all nearest-neighbour coverings and those
with one J2 bond, projected pi/0-flux or paired-Z2 signs, and trainable
four-/six-site alternating-resonance weights.  The projected-VMC/two-Lanczos
state remains an additive anchor, and both registered manifolds stay well
below the complete 14-dimensional target sector.

Physical loop labels improve median energy and variance relative to matched
orbit-feature scrambles, with three of four paired joint wins.  The mechanism
is therefore real.  It still loses to Operator-ViT and equal-dimension Krylov
in both observables.  This activates the preregistered stopping rule: the
result remains exploration-only, triangular `6x6` is closed, and no further
spin-only carrier variant is licensed.
"""
            ),
            code(
                r"""
loop_suite = long_range_loop_rvb["suite"]
single = loop_suite["models"]["single_psg_loop_jastrow"]
two_psg = loop_suite["models"]["two_psg_loop_resonance_bank"]
loop_baselines = loop_suite["frozen_baselines"]

progression_labels = ["projected anchor", "factorized PSG",
                      "single-PSG loop", "two-PSG loop"]
progression_energy = np.asarray([
    loop_suite["anchor"]["energy_per_site"],
    loop_suite["factorized_psg_zero_loop"]["energy_per_site"],
    single["physical_summary"]["median_energy_per_site"],
    two_psg["physical_summary"]["median_energy_per_site"],
])

fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.2))
xp = np.arange(len(progression_labels))
axes[0].plot(xp, progression_energy, "o-", color="#4c78a8",
             label="physical loop-RVB")
axes[0].axhline(
    loop_baselines["exact_su2_operator_vit"]["energy_per_site"],
    color="#e45756", ls=":", lw=1.1, label="Operator-ViT",
)
axes[0].axhline(
    two_psg["matched_dimension_krylov"]["energy_per_site"],
    color="#f58518", ls="--", lw=1.1, label="rank-7 Krylov",
)
axes[0].set_xticks(xp, progression_labels, rotation=22, ha="right")
axes[0].set(ylabel=r"$E/N$",
            title="(a) Loops help, but do not reach the frontier")
axes[0].legend(frameon=False, fontsize=7)

for model, marker, label in (
    (single, "o", "single PSG"),
    (two_psg, "s", "two PSG + parity"),
):
    physical = np.asarray([
        row["metrics"]["energy_per_site"] for row in model["physical_runs"]
    ])
    scrambled = np.asarray([
        row["metrics"]["energy_per_site"] for row in model["scrambled_runs"]
    ])
    axes[1].plot(np.arange(len(physical)), physical - scrambled,
                 marker=marker, label=label)
axes[1].axhline(0.0, color="black", lw=.8)
axes[1].set(xlabel="fresh paired seed",
            ylabel=r"$(E_{physical}-E_{scramble})/N$",
            title="(b) Geometry wins in 3/4 paired trials")
axes[1].legend(frameon=False, fontsize=7)

for model, color, marker, label in (
    (single, "#4c78a8", "o", "single PSG, physical"),
    (two_psg, "#72b7b2", "s", "two PSG, physical"),
):
    axes[2].scatter(
        [row["metrics"]["variance_per_site"] for row in model["physical_runs"]],
        [row["metrics"]["energy_per_site"] for row in model["physical_runs"]],
        color=color, marker=marker, label=label,
    )
axes[2].scatter(
    [row["metrics"]["variance_per_site"] for row in two_psg["scrambled_runs"]],
    [row["metrics"]["energy_per_site"] for row in two_psg["scrambled_runs"]],
    color="#bab0ac", marker="x", label="two PSG, scramble",
)
axes[2].scatter(
    [loop_baselines["exact_su2_operator_vit"]["variance_per_site"]],
    [loop_baselines["exact_su2_operator_vit"]["energy_per_site"]],
    color="#e45756", marker="*", s=95, label="Operator-ViT",
)
axes[2].scatter(
    [two_psg["matched_dimension_krylov"]["variance_per_site"]],
    [two_psg["matched_dimension_krylov"]["energy_per_site"]],
    color="#f58518", marker="D", s=48, label="rank-7 Krylov",
)
axes[2].set_xscale("log")
axes[2].set(xlabel=r"$\mathrm{Var}(H)/N$", ylabel=r"$E/N$",
            title="(c) Energy--variance stop gate")
axes[2].legend(frameon=False, fontsize=6.8)

fig.tight_layout()
save_figure(fig, "fig48_n16_long_range_loop_rvb_stop")
plt.show()

print("decision", long_range_loop_rvb["promotion_decision"])
print("gates", two_psg["gates"])
print("result hash", long_range_loop_rvb["result_sha256"])
"""
            ),
        ]
    )

    # The two latest cells are authored near the shared N36 diagnostics above,
    # but the exact-small causal gate must precede its licensed scale preflight
    # in the rendered narrative.
    n36_heading = "support/action preflight: sampling passes"
    exact_heading = "Phase-aware carrier re-entry: a reproducible mechanism"
    n36_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if cell.cell_type == "markdown" and n36_heading in cell.source
    )
    n36_block = notebook["cells"][n36_index : n36_index + 2]
    del notebook["cells"][n36_index : n36_index + 2]
    exact_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if cell.cell_type == "markdown" and exact_heading in cell.source
    )
    notebook["cells"][exact_index + 2 : exact_index + 2] = n36_block

    # The cross-fitted stop test is the direct consequence of the N36
    # preflight and must follow it even though its cells are authored beside
    # the shared sampled-response diagnostics.
    crossfit_heading = "Cross-fitted relative-flux repair"
    crossfit_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if cell.cell_type == "markdown" and crossfit_heading in cell.source
    )
    crossfit_block = notebook["cells"][crossfit_index : crossfit_index + 2]
    del notebook["cells"][crossfit_index : crossfit_index + 2]
    n36_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if cell.cell_type == "markdown" and n36_heading in cell.source
    )
    notebook["cells"][n36_index + 2 : n36_index + 2] = crossfit_block

    # The neural-Pfaffian stop gate is the direct successor to the failed
    # relative-flux carrier bank. Keep the negative result in exploration and
    # adjacent to the decision that licensed it.
    backflow_heading = "Configuration-dependent neural Pfaffian"
    backflow_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if cell.cell_type == "markdown" and backflow_heading in cell.source
    )
    backflow_block = notebook["cells"][backflow_index : backflow_index + 2]
    del notebook["cells"][backflow_index : backflow_index + 2]
    crossfit_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if cell.cell_type == "markdown" and crossfit_heading in cell.source
    )
    notebook["cells"][crossfit_index + 2 : crossfit_index + 2] = backflow_block

    # The long-range loop-RVB gate is the declared final contingency after the
    # neural-Pfaffian rejection and must immediately follow it.
    loop_rvb_heading = "Phase-aware long-range loop-RVB"
    loop_rvb_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if cell.cell_type == "markdown" and loop_rvb_heading in cell.source
    )
    loop_rvb_block = notebook["cells"][loop_rvb_index : loop_rvb_index + 2]
    del notebook["cells"][loop_rvb_index : loop_rvb_index + 2]
    backflow_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if cell.cell_type == "markdown" and backflow_heading in cell.source
    )
    notebook["cells"][backflow_index + 2 : backflow_index + 2] = loop_rvb_block

    # HubbardCG is a separate downstream project. Keep its frozen artifact
    # untouched, but do not carry the historical Hubbard figure block into
    # the spin-only paper notebook. Renumber subsequent top-level sections.
    filtered = []
    skipping_hubbard = False
    section_map = {
        5: 4,
        6: 5,
        7: 6,
        8: 7,
        9: 8,
        10: 9,
        11: 10,
        12: 11,
        13: 12,
        14: 13,
        15: 14,
        16: 15,
        17: 16,
        18: 17,
        19: 18,
        20: 19,
        21: 20,
        22: 21,
        23: 22,
        24: 23,
        25: 24,
        26: 25,
        27: 26,
        28: 27,
        29: 28,
        30: 29,
        31: 30,
        32: 31,
        33: 32,
        34: 33,
    }
    for cell in notebook["cells"]:
        source = cell.source.lstrip()
        if cell.cell_type == "markdown" and source.startswith("## 4. HubbardCG"):
            skipping_hubbard = True
            continue
        if skipping_hubbard:
            if cell.cell_type == "markdown" and source.startswith("## 5."):
                skipping_hubbard = False
            else:
                continue
        if cell.cell_type == "markdown":
            for old, new in section_map.items():
                prefix = f"## {old}."
                if cell.source.lstrip().startswith(prefix):
                    leading = cell.source[: len(cell.source) - len(cell.source.lstrip())]
                    cell.source = leading + cell.source.lstrip().replace(
                        prefix, f"## {new}.", 1
                    )
                    break
        filtered.append(cell)
    notebook["cells"] = filtered

    notebook["cells"].extend(
        [
            nbf.v4.new_markdown_cell(
                r"""
## Scale-first analytic-carrier stop gate (2026-09-25)

This exploratory-only panel records the negative (6\times6) production
gate.  All curves use sealed checkpoints and four untouched 512-sample
streams.  The exact-SU(2) arm lowers neither lattice's energy, while both
physical-bit controls do.  A separate reduced-label audit shows why the
single-frame plaquette carrier is restrictive: local (S=1) weight grows,
but inter-cell nonzero-spin support remains about two percent or decreases.
This figure must not be moved to the paper notebook unless a successor state
manifold passes the registered variational gate.
"""
            ),
            nbf.v4.new_code_cell(
                r"""
carrier_audit = json.loads((
    PROJECT / "results/remote_scale_first_carrier_checkpoint_audit_484555/summary.json"
).read_text())
schur_channels = json.loads((
    PROJECT / "results/spin_scale_first_carrier_schur_channel_audit_v1/results.json"
).read_text())

arm_style = {
    "exact_su2_balanced_schur_spatial_vit": ("exact SU(2)", "#4c78a8", "o"),
    "fixed_sz_shared_autoregressive_spatial_vit": (r"fixed $S^z$", "#f58518", "s"),
    "symmetry_unaware_autoregressive_spatial_vit": ("no spin mask", "#54a24b", "^"),
}
fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.35))
for axis, (case_id, title) in zip(
    axes[:2],
    [
        ("triangular_6x6_j2_0.125", r"triangular $J_2/J_1=0.125$"),
        ("square_6x6_j2_0.5", r"square $J_2/J_1=0.5$"),
    ],
):
    for arm_id, (label, color, marker) in arm_style.items():
        rows = sorted(
            [
                row for row in carrier_audit["summaries"]
                if row["case_id"] == case_id and row["arm_id"] == arm_id
            ],
            key=lambda row: row["checkpoint_step"],
        )
        axis.errorbar(
            [row["checkpoint_step"] for row in rows],
            [row["energy_mean"] / 36.0 for row in rows],
            yerr=[row["energy_between_seed_standard_error"] / 36.0 for row in rows],
            color=color,
            marker=marker,
            lw=1.5,
            capsize=2,
            label=label,
        )
    axis.set(xlabel="VMC step", ylabel=r"untouched $E/N$", title=title)
    axis.legend(frameon=False, fontsize=7)

for summary, color, marker in zip(
    schur_channels["summaries"], ["#4c78a8", "#e45756"], ["o", "s"]
):
    geometry = summary["case_id"].split("_")[0]
    steps = [0, 64]
    axes[2].plot(
        steps,
        [100.0 * summary[f"step_{step}"]["cell_q2_fraction"]["mean"] for step in steps],
        color=color,
        marker=marker,
        label=f"{geometry}: local S=1",
    )
    axes[2].plot(
        steps,
        [100.0 * summary[f"step_{step}"]["intercell_nonzero_fraction"]["mean"] for step in steps],
        color=color,
        marker=marker,
        ls="--",
        label=f"{geometry}: inter-cell q>0",
    )
axes[2].set(xlabel="VMC step", ylabel="sampled support (%)",
            title="Schur-channel support")
axes[2].legend(frameon=False, fontsize=6.8)
fig.tight_layout()
save_figure(fig, "fig49_scale_first_carrier_negative_gate")
plt.show()

print("audit gate:", carrier_audit["production_release_allowed"])
print("audit hash:", carrier_audit["result_sha256"])
print("channel hash:", schur_channels["result_sha256"])
"""
            ),
            nbf.v4.new_markdown_cell(
                r"""
## Coherent multi-frame support without an energy win (2026-09-25)

Translated and rotated plaquette frames remove the artificial boundary of a
single Schur partition: every nearest-neighbour bond is internal in at least
one frame, and the canonical inter-cell Casimir becomes nonzero on (6\times6).
The final panel is the decisive untouched audit.  A configuration-dependent,
exact-singlet pair-(B_1) resonance is restricted by finite sampled
Fubini--Study and symmetric-KL radii, but its energy change is unresolved on
both lattices.  This is a physical mechanism/stop figure for exploration, not
a PRL variational result.
"""
            ),
            nbf.v4.new_code_cell(
                r"""
frame_exact = json.loads((
    PROJECT / "results/spin_coherent_plaquette_frame_gate_v1/results.json"
).read_text())
frame_support = json.loads((
    PROJECT / "results/spin_n36_coherent_frame_support_v1/results.json"
).read_text())
frame_canary = json.loads((
    PROJECT / "results/spin_n36_coherent_frame_canary_v1/results.json"
).read_text())

geometry_style = {
    "square": ("square", "#4c78a8", "o"),
    "triangular": ("triangular", "#e45756", "s"),
}
fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.25))

for case in frame_exact["cases"]:
    geometry = case["case_id"].split("_")[0]
    label, color, marker = geometry_style[geometry]
    axes[0].plot(
        [0, 1],
        [
            case["primary_j1_bond_coverage"]["coverage_fraction"],
            case["coherent_frame_j1_bond_coverage"]["coverage_fraction"],
        ],
        color=color, marker=marker, lw=1.6, label=label,
    )
axes[0].set(
    xticks=[0, 1], xticklabels=["single frame", "all frames"],
    ylabel=r"fraction of $J_1$ bonds internal", ylim=(0.35, 1.05),
    title="geometry coverage",
)
axes[0].legend(frameon=False, fontsize=8)

for case in frame_support["cases"]:
    geometry = case["case_id"].split("_")[0]
    label, color, marker = geometry_style[geometry]
    summary = case["summary"]
    axes[1].errorbar(
        [0, 1],
        [summary["aligned_mean_intercell_casimir"],
         summary["coherent_mean_intercell_casimir"]],
        yerr=[0.0, summary["paired_casimir_standard_error"]],
        color=color, marker=marker, lw=1.6, capsize=2, label=label,
    )
axes[1].set(
    xticks=[0, 1], xticklabels=["aligned", "coherent"],
    ylabel=r"mean non-root $\langle S_A^2\rangle$",
    title=r"inter-cell $q>0$ witness",
)
axes[1].legend(frameon=False, fontsize=8)

for index, case in enumerate(frame_canary["cases"]):
    geometry = case["case_id"].split("_")[0]
    label, color, marker = geometry_style[geometry]
    summary = case["audit_summary"]
    axes[2].errorbar(
        index,
        1e5 * summary["paired_energy_change_per_site_mean"],
        yerr=1e5 * summary["paired_energy_change_per_site_standard_error"],
        color=color, marker=marker, capsize=3, label=label,
    )
axes[2].axhline(0.0, color="black", lw=0.8, ls="--")
axes[2].set(
    xticks=[0, 1], xticklabels=["square", "triangular"],
    ylabel=r"untouched $\Delta E/N\;(10^{-5})$",
    title="bounded resonance canary",
)
axes[2].legend(frameon=False, fontsize=8)
fig.tight_layout()
save_figure(fig, "fig50_coherent_frame_support_stop")
plt.show()

print("N16 gate:", frame_exact["all_exact_n16_gates_passed"])
print("N36 support gate:", frame_support["all_support_gates_passed"])
print("untouched energy gate:", frame_canary["all_untouched_energy_gates_passed"])
print("canary decision:", frame_canary["decision"])
"""
            ),
            nbf.v4.new_markdown_cell(
                r"""
## Complete local-singlet resonance bank: square descent, triangular instability

The final coherent-frame falsifier expands each frame into every local
singlet-number channel (k=0,ldots,9).  It is not another neural head: every
column is separately exact (S=0,Gamma-A_1), and the development stream fixes
one direction before four untouched streams are opened.  The square response
is a reproducible descent but is only marginally better than matched frame
scrambles.  On the triangular QSL target, a rare (k=8) component produces a
large cross-stream physical-distance failure and raises the mean energy.

This closes the coherent local-resonance route.  The figure is diagnostic and
belongs only in the exploration notebook.
"""
            ),
            nbf.v4.new_code_cell(
                r"""
multibody = json.loads((
    PROJECT / "results/spin_n36_multibody_frame_resonance_v1/results.json"
).read_text())

fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.35))
for case in multibody["cases"]:
    geometry = case["case_id"].split("_")[0]
    label, color, marker = geometry_style[geometry]
    pair = case["pair_b1_control"]["audit_summary"]
    full = case["physical_multibody_bank"]["audit_summary"]
    axes[0].errorbar(
        [0, 1],
        [1e3 * pair["paired_energy_change_per_site_mean"],
         1e3 * full["paired_energy_change_per_site_mean"]],
        yerr=[1e3 * pair["paired_energy_change_per_site_standard_error"],
              1e3 * full["paired_energy_change_per_site_standard_error"]],
        color=color, marker=marker, lw=1.5, capsize=2, label=label,
    )
axes[0].axhline(0.0, color="black", lw=0.8, ls="--")
axes[0].set(
    xticks=[0, 1], xticklabels=[r"$k=0,2$", r"$k=0\ldots9$"],
    ylabel=r"untouched $\Delta E/N\;(10^{-3})$",
    title="collective resonance",
)
axes[0].legend(frameon=False, fontsize=8)

for case_index, case in enumerate(multibody["cases"]):
    geometry = case["case_id"].split("_")[0]
    label, color, marker = geometry_style[geometry]
    physical = case["physical_multibody_bank"]["audit_summary"][
        "paired_energy_change_per_site_mean"
    ]
    controls = [
        row["audit_summary"]["paired_energy_change_per_site_mean"]
        for row in case["scrambled_multibody_banks"]
    ]
    x0 = 2 * case_index
    axes[1].scatter(
        np.full(len(controls), x0 + 0.18), 1e3 * np.asarray(controls),
        color="#888888", marker="x", s=28,
        label="matched scrambles" if case_index == 0 else None,
    )
    axes[1].scatter(
        [x0 - 0.18], [1e3 * physical], color=color, marker=marker, s=38,
        label=f"{label}: physical",
    )
    axes[1].plot(
        [x0 - 0.28, x0 + 0.28],
        [1e3 * np.median(controls)] * 2,
        color="#777777", lw=1.0,
    )
axes[1].axhline(0.0, color="black", lw=0.8, ls="--")
axes[1].set(
    xticks=[0, 2], xticklabels=["square", "triangular"],
    ylabel=r"mean $\Delta E/N\;(10^{-3})$",
    title="geometry vs scrambles",
)
axes[1].legend(frameon=False, fontsize=7)

for case in multibody["cases"]:
    geometry = case["case_id"].split("_")[0]
    label, color, marker = geometry_style[geometry]
    rows = case["physical_multibody_bank"]["audit_rows"]
    axes[2].plot(
        range(1, len(rows) + 1),
        [row["physical_distance"]["fubini_study_distance"] for row in rows],
        color=color, marker=marker, lw=1.5, label=label,
    )
axes[2].axhline(0.03, color="black", lw=0.9, ls="--", label="dev radius")
axes[2].set(
    xlabel="untouched stream", ylabel="realized FS distance",
    xticks=[1, 2, 3, 4], title="triangular support shift",
)
axes[2].legend(frameon=False, fontsize=8)
fig.tight_layout()
save_figure(fig, "fig51_multibody_frame_resonance_closure")
plt.show()

print("decision:", multibody["decision"])
print("hash:", multibody["result_sha256"])
"""
            ),
            nbf.v4.new_markdown_cell(
                r"""
## Sealed full-ViT spectral error: better than K4 observables, not a new direction

The final exact-SU(2) Spatial-ViT is decomposed in the complete fourteen-state
``S=0,Gamma-A1`` spectrum.  We also compare ``K4+ViT`` with K5 and
``K4+ViT+HViT`` with K6 at equal subspace rank.  This is a route-selection
diagnostic: the small sector is too saturated for a paper scaling claim.
"""
            ),
            nbf.v4.new_code_cell(
                r"""
tomography = json.loads((
    PROJECT / "results/spin_n16_spectral_error_tomography_v1/results.json"
).read_text())

fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.35))
profile_names = (
    "full_su2_spatial_vit",
    "degree_four_krylov",
    "degree_five_krylov",
)
styles = {
    "full_su2_spatial_vit": ("SU(2)-ViT", "#0072B2", "o"),
    "degree_four_krylov": ("K4", "#D55E00", "s"),
    "degree_five_krylov": ("K5", "#009E73", "^"),
}
for name in profile_names:
    label, color, marker = styles[name]
    weights = np.asarray(tomography["spectral_profiles"][name]["spectral_weights"])
    axes[0].semilogy(
        np.arange(len(weights)), np.maximum(weights, 1e-18),
        color=color, marker=marker, ms=4, lw=1.2, label=label,
    )
axes[0].set(
    xlabel="exact A1 eigenstate index",
    ylabel="spectral weight",
    title="small high-energy ViT tail",
)
axes[0].legend(frameon=False, fontsize=8)

display_names = (
    "full_su2_spatial_vit",
    "degree_four_krylov",
    "degree_five_krylov",
    "k4_plus_vit",
)
for name in display_names:
    energy_error = (
        tomography["metrics"][name]["energy_per_site"]
        - tomography["spectral_profiles"][name]["ground_energy_total"] / 16.0
    )
    observable_error = tomography[
        "median_symmetry_observable_relative_errors"
    ][name]
    label = styles.get(name, ("K4+ViT", "#CC79A7", "D"))[0]
    color = styles.get(name, (None, "#CC79A7", "D"))[1]
    marker = styles.get(name, (None, "#CC79A7", "D"))[2]
    axes[1].scatter(energy_error, observable_error, color=color, marker=marker, s=42)
    axes[1].annotate(label, (energy_error, observable_error), xytext=(4, 3),
                     textcoords="offset points", fontsize=8)
axes[1].set(
    xlabel=r"energy error per site",
    ylabel="median observable relative error",
    title="K5 already supersedes ViT",
)
axes[1].set_xscale("log")
axes[1].set_yscale("log")
fig.tight_layout()
save_figure(fig, "fig52_full_vit_spectral_error_tomography")
plt.show()

print("gates:", tomography["gates"])
print("decision:", tomography["decision"])
print("hash:", tomography["result_sha256"])
"""
            ),
        ]
    )

    notebook.cells.extend(channel_attention_cells())
    notebook.cells.extend(channel_mechanism_cells())
    notebook.cells.extend(exchange_environment_cells())
    notebook.cells.extend(environment_complement_cells())
    notebook.cells.extend(finite_environment_cells())
    notebook.cells.extend(finite_audit_cells())
    notebook.cells.extend(stratified_audit_cells())
    notebook.cells.extend(strong_innovation_cells())
    notebook.cells.extend(full_vit_budget_cells())
    notebook.cells.extend(full_vit_continuation_cells())
    notebook.cells.extend(full_vit_optimization_cells())
    notebook.cells.extend(full_vit_recheck_cells())
    notebook.cells.extend(full_vit_step_audit_cells())
    notebook.cells.extend(full_vit_matched_distance_cells())
    notebook.cells.extend(full_vit_vap_cells())
    notebook.cells.extend(full_vit_constraint_cells())
    notebook.cells.extend(vap_decoder_cells())
    notebook.cells.extend(decoder_cost_cells())
    notebook.cells.extend(decoder_reuse_cells())
    notebook.cells.extend(decoder_pilot_cells())
    notebook.cells.extend(decoder_pilot_cells(physical_initialization=True))
    notebook.cells.extend(decoder_sampler_cells())
    notebook.cells.extend(decoder_importance_cells())
    notebook.cells.extend(schur_cutset_cells())
    notebook.cells.extend(schur_cutset_training_cells())
    notebook.cells.extend(schur_carrier_control_cells())
    notebook.cells.extend(schur_control_training_cells())
    notebook.cells.extend(carrier_residual_geometry_cells())
    notebook.cells.extend(carrier_mixture_comparison_cells())
    notebook.cells.extend(conditional_carrier_resonance_cells())
    notebook.cells.extend(overlap_attention_cost_cells())
    notebook.cells.extend(resonance_reuse_cells())
    notebook.cells.extend(resonance_training_cells())
    notebook.cells.extend(resonance_distillation_cells())
    notebook.cells.extend(resonance_continuation_cells())
    notebook.cells.extend(symmetry_transfer_cells())
    if (PROJECT/'results/spin_schur_capacity_v1/results.json').exists():
        notebook.cells.extend(schur_capacity_cells())
    if (PROJECT/'results/spin_schur_metric_capacity_v1/results.json').exists():
        notebook.cells.extend(schur_metric_capacity_cells())
    if (PROJECT/'results/spin_schur_transport_v1/results.json').exists():
        notebook.cells.extend(schur_transport_cells())
    if (PROJECT/'results/spin_schur_force_calibration_v1/results.json').exists():
        notebook.cells.extend(schur_force_calibration_cells())
    if (PROJECT/'results/spin_krylov_carrier_bridge_v1/results.json').exists():
        notebook.cells.extend(krylov_carrier_bridge_cells())
    if (PROJECT/'results/spin_krylov_residual_training_v1/audit.json').exists():
        notebook.cells.extend(krylov_residual_training_cells())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    executed = client.execute()
    nbf.write(executed, OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--channel-attention-only", action="store_true")
    selection.add_argument("--krylov-carrier-bridge-only", action="store_true")
    selection.add_argument("--krylov-residual-training-only", action="store_true")
    selection.add_argument("--section", choices=("channel-attention", "channel-mechanism", "exchange-environment", "environment-complement", "finite-environment", "finite-audit", "stratified-audit", "strong-innovation", "full-vit-budget", "full-vit-continuation", "full-vit-optimization", "full-vit-recheck", "full-vit-step-audit", "full-vit-matched-distance", "full-vit-vap", "full-vit-constraint", "vap-decoder", "decoder-cost", "decoder-reuse", "decoder-pilot", "decoder-physical", "decoder-sampler", "decoder-importance", "schur-cutset", "schur-cutset-training", "schur-carrier-control", "schur-control-training", "carrier-residual-geometry", "carrier-mixture-comparison", "conditional-carrier-resonance", "overlap-attention-cost", "resonance-reuse", "resonance-training", "resonance-distillation", "resonance-continuation", "symmetry-transfer", "schur-capacity", "schur-metric-capacity", "schur-transport", "schur-force-calibration"))
    args = parser.parse_args()
    if args.krylov_residual_training_only:
        print(update_exploration_section(krylov_residual_training_cells()))
    elif args.krylov_carrier_bridge_only:
        print(update_exploration_section(krylov_carrier_bridge_cells()))
    elif args.section == "schur-force-calibration":
        print(update_exploration_section(schur_force_calibration_cells()))
    elif args.section == "schur-transport":
        print(update_exploration_section(schur_transport_cells()))
    elif args.section == "schur-metric-capacity":
        print(update_exploration_section(schur_metric_capacity_cells()))
    elif args.section == "schur-capacity":
        print(update_exploration_section(schur_capacity_cells()))
    elif args.section == "symmetry-transfer":
        print(update_exploration_section(symmetry_transfer_cells()))
    elif args.section == "resonance-continuation":
        print(update_exploration_section(resonance_continuation_cells()))
    elif args.section == "resonance-distillation":
        print(update_exploration_section(resonance_distillation_cells()))
    elif args.section == "resonance-training":
        print(update_exploration_section(resonance_training_cells()))
    elif args.section == "resonance-reuse":
        print(update_exploration_section(resonance_reuse_cells()))
    elif args.section == "overlap-attention-cost":
        print(update_exploration_section(overlap_attention_cost_cells()))
    elif args.section == "conditional-carrier-resonance":
        print(update_exploration_section(conditional_carrier_resonance_cells()))
    elif args.section == "carrier-mixture-comparison":
        print(update_exploration_section(carrier_mixture_comparison_cells()))
    elif args.section == "carrier-residual-geometry":
        print(update_exploration_section(carrier_residual_geometry_cells()))
    elif args.section == "schur-control-training":
        print(update_exploration_section(schur_control_training_cells()))
    elif args.section == "schur-carrier-control":
        print(update_exploration_section(schur_carrier_control_cells()))
    elif args.section == "schur-cutset-training":
        print(update_exploration_section(schur_cutset_training_cells()))
    elif args.section == "schur-cutset":
        print(update_exploration_section(schur_cutset_cells()))
    elif args.section == "decoder-importance":
        print(update_exploration_section(decoder_importance_cells()))
    elif args.section == "decoder-sampler":
        print(update_exploration_section(decoder_sampler_cells()))
    elif args.section == "decoder-physical":
        print(update_exploration_section(decoder_pilot_cells(physical_initialization=True)))
    elif args.section == "decoder-pilot":
        print(update_exploration_section(decoder_pilot_cells()))
    elif args.section == "decoder-reuse":
        print(update_exploration_section(decoder_reuse_cells()))
    elif args.section == "decoder-cost":
        print(update_exploration_section(decoder_cost_cells()))
    elif args.section == "vap-decoder":
        print(update_exploration_section(vap_decoder_cells()))
    elif args.section == "full-vit-constraint":
        print(update_exploration_section(full_vit_constraint_cells()))
    elif args.section == "full-vit-vap":
        print(update_exploration_section(full_vit_vap_cells()))
    elif args.section == "full-vit-matched-distance":
        print(update_exploration_section(full_vit_matched_distance_cells()))
    elif args.section == "full-vit-step-audit":
        print(update_exploration_section(full_vit_step_audit_cells()))
    elif args.section == "full-vit-recheck":
        print(update_exploration_section(full_vit_recheck_cells()))
    elif args.section == "full-vit-optimization":
        print(update_exploration_section(full_vit_optimization_cells()))
    elif args.section == "full-vit-continuation":
        print(update_exploration_section(full_vit_continuation_cells()))
    elif args.section == "full-vit-budget":
        print(update_exploration_section(full_vit_budget_cells()))
    elif args.section == "strong-innovation":
        print(update_exploration_section(strong_innovation_cells()))
    elif args.section == "stratified-audit":
        print(update_exploration_section(stratified_audit_cells()))
    elif args.section == "finite-audit":
        print(update_exploration_section(finite_audit_cells()))
    elif args.section == "finite-environment":
        print(update_exploration_section(finite_environment_cells()))
    elif args.section == "environment-complement":
        print(update_exploration_section(environment_complement_cells()))
    elif args.section == "exchange-environment":
        print(update_exploration_section(exchange_environment_cells()))
    elif args.section == "channel-mechanism":
        print(update_exploration_section(channel_mechanism_cells()))
    elif args.channel_attention_only or args.section == "channel-attention":
        print(update_channel_attention_section())
    else:
        print(build())
