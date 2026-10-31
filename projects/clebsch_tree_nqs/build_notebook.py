#!/usr/bin/env python3
"""Build the compact, paper-facing ClebschTree spin-physics notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).resolve().parent
OUTPUT = PROJECT / "notebooks/clebsch_tree_physics.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build() -> Path:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3 (nqs)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
        "paper_notebook_schema": 1,
    }
    notebook["cells"] = [
        markdown(
            r"""
# Exact SU(2) and lattice symmetry as physics priors

This is the compact **paper-facing** notebook.  It contains only six figures
that currently support the spin-only PRL narrative:

1. factorial separation of spin and lattice symmetry;
2. parameter--accuracy compression on square and triangular clusters;
3. where the remaining triangular-state error lives; and
4. the symmetry-resolved finite-cluster mechanism motivating larger systems;
   and
5. the local energy-and-physics corroboration gate for Operator-ViT; and
6. the complete configuration-dependent multiplicity-space variational win.

Negative gates, sampler audits, optimizer studies, and unpromoted carrier
tests belong in `clebsch_tree_exploration.ipynb`.  Small clusters establish a
mechanism and a controlled variational comparison; they do not establish a
thermodynamic phase diagram.
"""
        ),
        code(
            r"""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path.cwd()
PROJECT = ROOT / "projects/clebsch_tree_nqs"
FIGURES = PROJECT / "figures/paper"
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

def load_complete(relative_path):
    path = PROJECT / relative_path
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete":
        raise RuntimeError(f"paper input is incomplete: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"{relative_path}: sha256={digest[:16]}...")
    return payload

def load_json(relative_path):
    path = PROJECT / relative_path
    payload = json.loads(path.read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"{relative_path}: sha256={digest[:16]}...")
    return payload

def save_figure(fig, stem):
    fig.savefig(FIGURES / f"{stem}.png", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")

def case_by_id(payload, case_id):
    return next(case for case in payload["cases"] if case["case_id"] == case_id)

def positive(values, floor=1e-13):
    values = np.asarray(values, dtype=float)
    if np.min(values) < -1e-9:
        raise AssertionError(f"variational error below anchor: {np.min(values)}")
    return np.maximum(values, floor)

spin = load_complete("results/spin_validation_v1/results.json")
prl_pareto = load_complete("results/spin_prl_local_pareto_v1/results.json")
prl_analysis = load_json("results/spin_prl_local_analysis_v1/summary.json")
physics = load_complete("results/spin_physics_interpretation_v1/results.json")
qsl_symmetry = load_complete("results/spin_qsl_symmetry_diagnostics_v1/results.json")
state_manifold = load_complete("results/spin_triangular_n16_state_manifold_v1/results.json")
"""
        ),
        markdown(
            r"""
## Figure 1 — factorial attribution of symmetry

The same lattice projector is applied to fixed-\(S_z\) U(1) and exact-singlet
SU(2) backbones.  This separates the benefit of eliminating unphysical total-
spin sectors from the benefit of selecting the correct spatial irrep.  The
comparison is at equal feature width, not equal raw parameter count.
"""
        ),
        code(
            r"""
square = case_by_id(spin, "square_4x4_j2_0p5_m0")

def median_metric(arm_id, key):
    return np.median([
        trial["metrics"][key] for trial in square["trials"]
        if trial["arm_id"] == arm_id
    ])

projectors = ("none", "translation", "gamma_a1")
x = np.arange(3)
tick_labels = ("none", "translations", r"$\Gamma A_1$")
colors = {"u1": "#35618d", "su2": "#d55e00"}

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
for backbone, marker in (("u1", "s"), ("su2", "o")):
    arms = [f"{backbone}_{projection}" for projection in projectors]
    energy = [median_metric(arm, "energy_error_per_site") for arm in arms]
    spin2 = [abs(median_metric(arm, "spin_squared_expectation")) for arm in arms]
    label = "fixed $S_z$" if backbone == "u1" else "exact $S=0$"
    axes[0].semilogy(x, positive(energy), marker=marker, ms=6, lw=1.7,
                    color=colors[backbone], label=label)
    if backbone == "u1":
        axes[1].semilogy(x, positive(spin2), marker=marker, ms=6, lw=1.7,
                        color=colors[backbone], label=label)
for ax in axes:
    ax.set_xticks(x, tick_labels)
    ax.set_xlabel("lattice projector")
axes[0].set_ylabel(r"median energy error/site $\Delta E/N$")
axes[0].set_title("(a) energy accuracy")
axes[1].set_ylabel(r"median $|\langle S^2\rangle|$")
axes[1].set_title("(b) spin-sector purity")
axes[1].text(.04, .10, r"SU(2): $S=0$ by construction", transform=axes[1].transAxes)
axes[0].legend(frameon=False)
fig.suptitle(r"Square $4\times4$, $J_2/J_1=0.5$: spin symmetry $\times$ lattice symmetry")
fig.tight_layout()
save_figure(fig, "fig01_square_spin_lattice_factorial")
plt.show()
"""
        ),
        markdown(
            r"""
## Figure 2 — parameter--accuracy compression

The translation-sector comparison retains a nontrivial variational space and
uses eight independent seed labels at each width.  The integrated frontier is
computed only over the common raw-parameter interval.  It supports a finite-
graph parameter-efficiency result, not yet a wall-time or asymptotic scaling
claim.
"""
        ),
        code(
            r"""
case_order = (
    "square_4x4_j2_0p5_prl_pareto",
    "square_4x4_j2_0p55_heldout",
    "triangular_4x4_j2_0p125_prl_pareto",
    "triangular_4x4_j2_0p175_heldout",
)
titles = {
    case_order[0]: r"square $4\times4$, $J_2/J_1=0.50$",
    case_order[1]: r"square $4\times4$, $J_2/J_1=0.55$ (held out)",
    case_order[2]: r"triangular $4\times4$, $J_2/J_1=0.125$",
    case_order[3]: r"triangular $4\times4$, $J_2/J_1=0.175$ (held out)",
}
colors = {"u1": "#35618d", "su2": "#d55e00"}

fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.4), sharex=True, sharey=True)
for panel, (ax, case_id) in enumerate(zip(axes.flat, case_order)):
    case = case_by_id(prl_pareto, case_id)
    for backbone, marker in (("u1", "s"), ("su2", "o")):
        medians, counts = [], []
        for width in (2, 3, 4, 5, 6):
            arm = f"{backbone}_D{width}_translation"
            trials = [row for row in case["trials"] if row["arm_id"] == arm]
            errors = np.asarray([row["metrics"]["energy_error_per_site"] for row in trials])
            count = int(trials[0]["parameter_count"])
            counts.append(count)
            medians.append(np.median(errors))
            ax.loglog(np.full(errors.size, count), positive(errors), marker,
                      ms=3.2, linestyle="none", color=colors[backbone], alpha=.24)
        ax.loglog(counts, positive(medians), marker=marker, ms=6, lw=1.8,
                  color=colors[backbone],
                  label="fixed $S_z$" if backbone == "u1" else "exact SU(2)")
    stats = prl_analysis["cases"][case_id]["parameter_pareto"]
    for boundary in stats["common_budget_interval"]:
        ax.axvline(boundary, color="0.45", lw=.7, linestyle=":")
    factor = stats["geometric_error_advantage_factor"]
    lower = stats["independent_cluster_bootstrap"]["average_log10_advantage"]["one_sided_95_lower"]
    ax.set_title(titles[case_id], fontsize=10)
    ax.text(.04, .06, rf"integrated gain $={factor:.2f}\times$" + "\n" +
            rf"95% lower $={lower:.2f}$ dex", transform=ax.transAxes,
            fontsize=8, bbox={"facecolor": "white", "alpha": .78, "edgecolor": "none"})
    if panel >= 2:
        ax.set_xlabel("raw trainable parameters")
    if panel % 2 == 0:
        ax.set_ylabel(r"final energy error/site $\Delta E/N$")
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
fig.suptitle("Translation-sector parameter--accuracy Pareto")
fig.tight_layout(rect=(0, .06, 1, .96))
save_figure(fig, "fig02_spin_parameter_accuracy_pareto")
plt.show()
"""
        ),
        markdown(
            r"""
## Figure 3 — physical origin of the residual variational error

At triangular \(4\times4\), \(J_2/J_1=0.075\), the physical global route is
the lowest-energy tested arm, but the stronger conclusion is diagnostic: most
remaining error tracks weight outside \(\Gamma\)-\(A_1\).  After sector
selection, differences reappear first on larger reduced-density-matrix
supports.  Since the retained sector has dimension only 14, this is a
mechanism control rather than a large-system expressivity result.
"""
        ),
        code(
            r"""
p1 = physics["P1_P2_P5"]
trials = [row for row in p1["trials"] if row["training_projection"] == "translation"]
arms = (
    "single_tree_d4", "route_vit_coefficients_d4", "route_vit_conditioned_d4",
    "route_vit_scrambled_tokens_d4", "route_local_conditioned_d4",
    "single_tree_matched_width",
)
labels = ("D4", "coeff.", "physical\nglobal", "scrambled\nglobal", "physical\nlocal", "D6")
colors = dict(zip(arms, ("#4c78a8", "#9d755d", "#e45756", "#72b7b2", "#f2cf5b", "#b279a2")))

fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.8))
for trial in trials:
    by_arm = {row["arm_id"]: row for row in trial["arms"]}
    axes[0, 0].semilogy(range(6), [by_arm[a]["metrics"]["energy_error_per_site"] for a in arms],
                        "o-", lw=1.25, ms=4.5, label=f"seed {trial['seed']}")
    axes[0, 1].semilogy(range(6), [by_arm[a]["structure_factor_error"]["rms_error"] for a in arms],
                        "o-", lw=1.25, ms=4.5)
for ax, title in zip(axes[0], ("(a) variational energy", "(b) magnetic correlations")):
    ax.set_xticks(range(6), labels)
    ax.set_title(title)
axes[0, 0].set_ylabel(r"$(E-E_0)/N$")
axes[0, 1].set_ylabel(r"RMS error of $S(\mathbf{q})$")
axes[0, 0].legend(frameon=False, fontsize=8)

missing, errors = [], []
for trial_index, trial in enumerate(trials):
    for row in trial["arms"]:
        x = 1.0 - row["gamma_a1_decomposition"]["weight"]
        y = row["metrics"]["energy_error_per_site"]
        missing.append(x); errors.append(y)
        axes[1, 0].scatter(x, y, color=colors[row["arm_id"]], s=31,
                           edgecolor="white", linewidth=.35,
                           label=labels[arms.index(row["arm_id"])].replace("\n", " ")
                           if trial_index == 0 else None)
correlation = float(np.corrcoef(missing, errors)[0, 1])
axes[1, 0].set(xscale="log", yscale="log", xlabel=r"missing $\Gamma$-$A_1$ weight",
               ylabel=r"$(E-E_0)/N$", title=f"(c) sector leakage ($r={correlation:.3f}$)")
axes[1, 0].legend(frameon=False, fontsize=6.7, ncol=2)

support_keys = ("bond_2", "triangle_3", "rhombus_4", "block_2x3_6", "half_torus_8")
support_sizes = np.asarray((2, 3, 4, 6, 8))
for arm in (arms[0], arms[2], arms[3], arms[5]):
    values = np.asarray([[
        next(row for row in trial["arms"] if row["arm_id"] == arm)
        ["gamma_a1_decomposition"]["local_rdm_error"][key]["trace_distance_mean"]
        for key in support_keys] for trial in trials])
    axes[1, 1].semilogy(support_sizes, np.median(values, axis=0), "o-", lw=1.35,
                        ms=4.2, color=colors[arm], label=labels[arms.index(arm)].replace("\n", " "))
    axes[1, 1].fill_between(support_sizes, values.min(0), values.max(0),
                            color=colors[arm], alpha=.09)
axes[1, 1].set(xlabel="support size", ylabel=r"trace distance after $\Gamma$-$A_1$ selection",
               title="(d) sector-internal error grows with scale")
axes[1, 1].legend(frameon=False, fontsize=7)
fig.suptitle(r"Triangular $4\times4$, $J_2/J_1=0.075$: symmetry organization")
fig.tight_layout()
save_figure(fig, "fig03_route_vit_physical_error_decomposition")
plt.show()
"""
        ),
        markdown(
            r"""
## Figure 4 — symmetry-resolved finite-cluster mechanism

The full-\(D_6\) 12-site HNF cluster contains K/M/X stars and permits exact
response decomposition in small singlet/triplet multiplicity spaces.  It
shows a soft, nearly single-mode \(\Gamma\)-\(E_2\) bond response while the
magnetic response transfers from K toward M.  The cell is a thin torus, so
this motivates a larger-size hypothesis; it is not evidence of spontaneous
nematic order or a phase boundary.
"""
        ),
        code(
            r"""
cases = physics["P3_P4"]["cases"]
j2 = np.asarray([row["j2_over_j1"] for row in cases])
fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.6))
series = {
    r"$\Gamma$-$E_2$ bond": [row["Gamma_E2_bond_nematic"]["lowest_gap"] for row in cases],
    r"$X^+$ bond": [row["X_plus_bond_branch"]["lowest_gap"] for row in cases],
    "K triplet": [row["spin_response"]["K"]["lowest_gap"] for row in cases],
    "M triplet": [row["spin_response"]["M"]["lowest_gap"] for row in cases],
}
for (label, values), marker in zip(series.items(), ("o", "s", "^", "v")):
    axes[0, 0].plot(j2, values, marker=marker, lw=1.45, ms=4.6, label=label)
axes[0, 0].axvspan(.10, .125, color="0.9", zorder=0)
axes[0, 0].set(ylabel=r"gap $\Delta/J_1$", title="(a) symmetry-resolved soft modes")
axes[0, 0].legend(frameon=False, fontsize=7.5)

for label, key, color, marker in (
    (r"bond-nematic $\Gamma$-$E_2$", "Gamma_E2_bond_nematic", "#e45756", "o"),
    (r"uniform chirality $\Gamma$-$A_2$", "Gamma_A2_uniform_chirality_response", "#4c78a8", "s"),
):
    axes[0, 1].semilogy(j2, [row[key]["static_response"] for row in cases],
                        marker=marker, lw=1.45, ms=4.6, color=color, label=label)
axes[0, 1].set(ylabel="static response", title="(b) nematic response grows fastest")
axes[0, 1].legend(frameon=False, fontsize=7.5)

axes[1, 0].plot(j2, [row["Gamma_E2_bond_nematic"]["lowest_doublet_fraction"] for row in cases],
                 "o-", lw=1.45, ms=4.6, label=r"$\Gamma$-$E_2$ doublet")
axes[1, 0].plot(j2, [row["X_plus_bond_branch"]["lowest_pole_fraction"] for row in cases],
                 "s-", lw=1.45, ms=4.6, label=r"$X^+$ pole")
axes[1, 0].set(ylim=(.88, .985), ylabel="source spectral weight",
               title="(c) nearly single-mode bond response")
axes[1, 0].legend(frameon=False, fontsize=7.5)

for momentum, color, marker in (("K", "#4c78a8", "o"), ("M", "#f28e2b", "s")):
    axes[1, 1].plot(j2, [row["spin_response"][momentum]["static_response_isotropic_per_site"]
                         for row in cases], marker=marker, lw=1.45, ms=4.6,
                         color=color, label=momentum)
axes[1, 1].set(ylabel="isotropic magnetic response/site",
               title=r"(d) competing magnetic response")
axes[1, 1].legend(frameon=False)
for ax in axes.flat:
    ax.set_xlabel(r"$J_2/J_1$")
fig.suptitle("Triangular N=12: symmetry-resolved response")
fig.tight_layout()
save_figure(fig, "fig04_triangular_symmetry_resolved_soft_modes")
plt.show()
"""
        ),
        markdown(
            r"""
## Figure 5 — the local variational win also repairs QSL-relevant observables

On the common triangular (4\times4) support at (J_2/J_1=0.125), model
selection uses variational energy only; the exact state is untouched until
this audit.  The exact-SU(2), Γ-(A_1) Operator-ViT beats the optimized
pair-Jastrow plus two-Lanczos control in energy and variance, while also
reducing the preregistered symmetry-observable error.  The (E_2) bond source
is much softer than uniform (A_2) chirality.  This corroborates a local
singlet-resonance mechanism but does not identify a QSL on (4\times4).
"""
        ),
        code(
            r"""
diagnostic = qsl_symmetry["diagnostics"]
exact = diagnostic["exact"]
arms = (
    "bare_pi_flux",
    "optimized_pi_flux_jastrow_two_lanczos",
    "pi_flux_fixed_sz_gamma_a1_spatial_vit",
    "pi_flux_exact_su2_a1_operator_vit",
)
labels = ("bare\npi-flux", "Jastrow\n+ 2LS", "fixed-$S^z$\nViT", "exact-SU(2)\nOperator-ViT")
colors = ("#9d755d", "#4c78a8", "#f28e2b", "#e45756")
rows = [diagnostic["variational"][arm] for arm in arms]
x = np.arange(len(arms))

fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.5))

energy_gap = [row["energy_per_site"] - exact["energy_per_site"] for row in rows]
axes[0, 0].semilogy(x, positive(energy_gap), "o-", color="0.25", lw=1.4)
for index, color in enumerate(colors):
    axes[0, 0].scatter(index, positive([energy_gap[index]])[0], color=color, s=48, zorder=3)
axes[0, 0].set(ylabel=r"$(E-E_0)/N$", title="(a) variational energy error")

variance = [row["variance_per_site"] for row in rows]
axes[0, 1].semilogy(x, variance, "o-", color="0.25", lw=1.4)
for index, color in enumerate(colors):
    axes[0, 1].scatter(index, variance[index], color=color, s=48, zorder=3)
axes[0, 1].set(ylabel=r"$\mathrm{Var}(H)/N$", title="(b) physical variance")

observable_error = [row["median_symmetry_observable_relative_error"] for row in rows]
axes[1, 0].semilogy(x, observable_error, "o-", color="0.25", lw=1.4)
for index, color in enumerate(colors):
    axes[1, 0].scatter(index, observable_error[index], color=color, s=48, zorder=3)
axes[1, 0].set(ylabel="median relative error", title=r"(c) $E_2$/chirality/dimer/K-M panel")

all_rows = rows + [exact]
all_x = np.arange(len(all_rows))
axes[1, 1].semilogy(all_x, [row["e2_state_referenced_sma_gap"] for row in all_rows],
                    "o-", color="#e45756", lw=1.4, label=r"$E_2$ bond")
axes[1, 1].semilogy(all_x, [row["chirality_state_referenced_sma_gap"] for row in all_rows],
                    "s-", color="#4c78a8", lw=1.4, label=r"$A_2$ chirality")
axes[1, 1].set(ylabel="state-referenced SMA scale", title="(d) singlet-channel hierarchy")
axes[1, 1].legend(frameon=False)

for ax in axes.flat[:3]:
    ax.set_xticks(x, labels)
axes[1, 1].set_xticks(all_x, (*labels, "exact"))
for ax in axes.flat:
    ax.tick_params(axis="x", labelsize=8)
fig.suptitle(r"Triangular $4\times4$, $J_2/J_1=0.125$: energy and physics corroboration")
fig.tight_layout()
save_figure(fig, "fig05_qsl_symmetry_corroboration")
plt.show()

print("Local corroboration gates:", diagnostic["gates"])
print("Paired comparisons:", diagnostic["comparisons"])
"""
        ),
        markdown(
            r"""
## Figure 6 — changing the state manifold removes the amplitude bottleneck

The frozen two-Lanczos and Operator-ViT states already have the exact
physical-basis sign structure and no nodes on triangular (4\times4).  Their
remaining error is therefore an amplitude error, not missing nodal physics.
Four complete rhombus Schur cells span all 1430 singlets, and a Spatial-ViT
learns configuration-dependent reduced coefficients before an exact
(\Gamma)-(A_1) projection.  An additive carrier anchor fixes the optimization
basin without restricting the final nodes.

At the same 160 updates used by the frozen Operator-ViT, all three fresh
anchored seeds improve its energy; the median total-energy error falls by
roughly three orders of magnitude.  At 1000 updates all anchored seeds reach
the exact finite-cluster state within double precision.  This is a decisive
small-system variational-capacity result, but the exact-support implementation
is not yet a scalable (6\times6) algorithm.
"""
        ),
        code(
            r"""
suite = state_manifold["suite"]
exact_energy = suite["exact_ground"]["energy_total"]
source_operator = qsl_symmetry["selected_arm_metadata"]["pi_flux_exact_su2_a1_operator_vit"]
operator_error = source_operator["energy_total"] - exact_energy

fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
colors = {
    "standalone_complete_multiplicity_vit": "#4c78a8",
    "operator_anchored_complete_multiplicity_vit": "#e45756",
}
labels = {
    "standalone_complete_multiplicity_vit": "standalone Schur-ViT",
    "operator_anchored_complete_multiplicity_vit": "carrier-anchored Schur-ViT",
}

operator_history = source_operator["training"]["history"]
axes[0].semilogy(
    [row["step"] for row in operator_history],
    positive(np.minimum.accumulate(
        [row["loss_before_update"] - exact_energy for row in operator_history]
    )),
    "o-", ms=3.2, lw=1.2, color="#9d755d", label="Operator-ViT",
)
for arm in colors:
    histories = suite["runs"][arm]
    steps = np.asarray([row["step"] for row in histories[0]["optimizer"]["history"]])
    errors = np.asarray([
        [checkpoint["loss_before_update"] - exact_energy
         for checkpoint in run["optimizer"]["history"]]
        for run in histories
    ])
    for row in errors:
        axes[0].semilogy(steps, positive(np.minimum.accumulate(row)),
                         color=colors[arm], alpha=.22, lw=.8)
    records = np.minimum.accumulate(errors, axis=1)
    axes[0].semilogy(steps, positive(np.median(records, axis=0)), "o-", ms=3.2,
                     lw=1.4, color=colors[arm], label=labels[arm])
axes[0].set(xlabel="updates", ylabel=r"total $E-E_0$",
            title="(a) best-so-far energy")
axes[0].legend(frameon=False, fontsize=7)

audit = suite["frozen_state_bottleneck_audit"]
audit_names = (
    "optimized_pi_flux_jastrow_two_lanczos",
    "pi_flux_exact_su2_a1_operator_vit",
)
audit_labels = ("Jastrow + 2LS", "Operator-ViT")
for index, (name, label) in enumerate(zip(audit_names, audit_labels)):
    row = audit[name]
    values = (
        row["energy_error_total"],
        row["exact_phase_candidate_magnitude_energy_error_total"],
        row["candidate_phase_exact_magnitude_energy_error_total"],
    )
    axes[1].semilogy((0, 1, 2), positive(values), "o-", lw=1.35,
                     label=label, color=("#4c78a8", "#e45756")[index])
axes[1].set_xticks((0, 1, 2), ("actual", "exact phase\ntrial magnitude", "trial phase\nexact magnitude"))
axes[1].set(ylabel=r"total $E-E_0$", title="(b) error is in amplitudes")
axes[1].legend(frameon=False, fontsize=7)

arms = (
    "pi_flux_exact_su2_a1_operator_vit",
    "standalone_complete_multiplicity_vit",
    "operator_anchored_complete_multiplicity_vit",
)
final_labels = ("Operator-ViT", "standalone", "anchored")
axes[2].scatter(
    [audit[arms[0]]["energy_error_total"]],
    [audit[arms[0]]["variance_total"]],
    marker="D", s=55, color="#9d755d", label=final_labels[0],
)
for arm, label in zip(arms[1:], final_labels[1:]):
    rows = suite["runs"][arm]
    axes[2].scatter(
        positive([row["metrics"]["energy_error_total"] for row in rows]),
        positive([row["metrics"]["variance_total"] for row in rows]),
        s=43, color=colors[arm], label=label,
    )
axes[2].set(xscale="log", yscale="log", xlabel=r"total $E-E_0$",
            ylabel=r"$\mathrm{Var}(H)$", title="(c) three fresh seeds")
axes[2].legend(frameon=False, fontsize=7)
fig.suptitle(r"Triangular $4\times4$, $J_2/J_1=0.125$: full SU(2) multiplicity manifold")
fig.tight_layout()
save_figure(fig, "fig06_complete_multiplicity_state_manifold")
plt.show()

anchored = suite["summary"]["operator_anchored_complete_multiplicity_vit"]
print("Promotion gates:", suite["promotion_gates"])
print("160-step median improvement factor:",
      operator_error / np.median(anchored["all_seed_matched_step_energy_errors_total"]))
"""
        ),
        markdown(
            r"""
## Current claim boundary

The strongest established result is now a finite-cluster **state-manifold
win** from exact SU(2) and lattice symmetry.  The shallow Operator-ViT first
improves energy, variance, fidelity, and the preregistered observable panel;
the complete configuration-dependent multiplicity Spatial-ViT then removes
its remaining amplitude error robustly on fresh seeds.  At 160 updates its
median total-energy error is (9.94\times10^{-8}), versus
(1.15\times10^{-4}) for Operator-ViT, while the 1000-update anchored runs
reach the exact finite-cluster state within double precision.

This result uses 1744 rather than 528 parameters and about 4.1 rather than
0.26 CPU seconds through 160 updates, so it is a variational-accuracy win, not
yet a raw-time win.  Its present implementation materializes the full
fixed-(M) support; a scalable sampled autoregressive realization and an
independent larger-size energy win remain mandatory before a PRL scaling or
QSL claim.

The attempted \(E_2\otimes E_2\to A_1\) ground-state carrier is intentionally
excluded from the paper figures: a same-work isotropic \(A_1\) shell-pair
control has lower energy in all six tested states.  The \(E_2\) direction is
complementary after \(A_1\), but has not earned promotion to a variational-win
claim.  Its full audit remains in the exploration notebook.
"""
        ),
    ]

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
    print(build())
