#!/usr/bin/env python3
"""Single command-line interface for active ClebschTree calculations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _workflow(domain: str):
    if domain == "spin":
        from projects.clebsch_tree_nqs.workflows.spin import WORKFLOW

        return WORKFLOW
    if domain == "hubbard":
        from projects.clebsch_tree_nqs.workflows.hubbard import WORKFLOW

        return WORKFLOW
    raise ValueError(f"unknown domain: {domain!r}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run reproducible physical-input ClebschTree campaigns."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    exact = commands.add_parser("exact", help="run an exact-support benchmark")
    exact.add_argument("--domain", required=True, choices=("spin", "hubbard"))
    exact.add_argument("--config", required=True, type=Path)
    exact.add_argument("--output", type=Path)
    exact.add_argument("--overwrite", action="store_true")
    sampled = commands.add_parser(
        "sampled-preflight",
        help="validate spin sampled estimators against exact Born references",
    )
    sampled.add_argument("--config", required=True, type=Path)
    sampled.add_argument("--output", type=Path)
    sampled.add_argument("--overwrite", action="store_true")
    singlet_force = commands.add_parser(
        "singlet-force-efficiency",
        help="compare full-H and diagonal-3Hz exact-singlet force estimators",
    )
    singlet_force.add_argument("--config", required=True, type=Path)
    singlet_force.add_argument("--output", type=Path)
    singlet_force.add_argument("--overwrite", action="store_true")
    spatial_cg = commands.add_parser(
        "spatial-cg-exact",
        help="run the paired exact-support support-aware Spatial-CG pilot",
    )
    spatial_cg.add_argument("--config", required=True, type=Path)
    spatial_cg.add_argument("--output", type=Path)
    spatial_cg.add_argument("--overwrite", action="store_true")
    spatial_cg_capacity = commands.add_parser(
        "spatial-cg-capacity",
        help="diagnose physical and scrambled route spans on a frozen backbone",
    )
    spatial_cg_capacity.add_argument("--config", required=True, type=Path)
    spatial_cg_capacity.add_argument("--pilot", required=True, type=Path)
    spatial_cg_capacity.add_argument("--output", type=Path)
    spatial_cg_capacity.add_argument("--overwrite", action="store_true")
    spatial_vit = commands.add_parser(
        "spatial-vit-exact",
        help="run the fixed-Sz factored Spatial-ViT local frontier",
    )
    spatial_vit.add_argument("--config", required=True, type=Path)
    spatial_vit.add_argument("--output", type=Path)
    spatial_vit.add_argument("--overwrite", action="store_true")
    rvb_covering = commands.add_parser(
        "spin-rvb-covering-frontier",
        help="run the exact-small invariant RVB-covering Transformer gate",
    )
    rvb_covering.add_argument("--config", required=True, type=Path)
    rvb_covering.add_argument("--output", type=Path)
    rvb_covering.add_argument("--overwrite", action="store_true")
    rvb_matrix_free = commands.add_parser(
        "spin-rvb-covering-matrix-free-gate",
        help="run the N36 RVB loop-estimator, control, and phase-transfer gate",
    )
    rvb_matrix_free.add_argument("--config", required=True, type=Path)
    rvb_matrix_free.add_argument("--output", type=Path)
    rvb_matrix_free.add_argument("--overwrite", action="store_true")
    flux_loop = commands.add_parser(
        "spin-flux-loop-interference-confirmation",
        help="confirm cross-flux and dimer-resonance gains against exact-sector scrambles",
    )
    flux_loop.add_argument("--config", required=True, type=Path)
    flux_loop.add_argument("--output", type=Path)
    flux_loop.add_argument("--overwrite", action="store_true")
    flux_loop_n36 = commands.add_parser(
        "spin-n36-flux-loop-preflight",
        help="run the bounded 6x6 paired-carrier support and dimer-action gate",
    )
    flux_loop_n36.add_argument("--config", required=True, type=Path)
    flux_loop_n36.add_argument("--output", type=Path)
    flux_loop_n36.add_argument("--overwrite", action="store_true")
    relative_flux_n36 = commands.add_parser(
        "spin-n36-relative-flux-crossfit",
        help="cross-fit stable relative-flux modes and their dimer response",
    )
    relative_flux_n36.add_argument("--config", required=True, type=Path)
    relative_flux_n36.add_argument("--output", type=Path)
    relative_flux_n36.add_argument("--overwrite", action="store_true")
    neural_backflow = commands.add_parser(
        "spin-neural-backflow-pfaffian",
        help="run the exact-N16 symmetry-projected neural-backflow Pfaffian gate",
    )
    neural_backflow.add_argument("--config", required=True, type=Path)
    neural_backflow.add_argument("--output", type=Path)
    neural_backflow.add_argument("--overwrite", action="store_true")
    residual_backflow = commands.add_parser(
        "spin-residual-neural-backflow",
        help="run the carrier-preserving exact-N16 neural-Pfaffian contingency",
    )
    residual_backflow.add_argument("--config", required=True, type=Path)
    residual_backflow.add_argument("--output", type=Path)
    residual_backflow.add_argument("--overwrite", action="store_true")
    resonating_pfaffian = commands.add_parser(
        "spin-resonating-neural-pfaffian",
        help="run the coherent exact-N16 multi-Pfaffian rank-matched gate",
    )
    resonating_pfaffian.add_argument("--config", required=True, type=Path)
    resonating_pfaffian.add_argument("--output", type=Path)
    resonating_pfaffian.add_argument("--overwrite", action="store_true")
    loop_rvb = commands.add_parser(
        "spin-long-range-loop-rvb",
        help="run the exact-N16 phase-aware long-range loop-RVB gate",
    )
    loop_rvb.add_argument("--config", required=True, type=Path)
    loop_rvb.add_argument("--output", type=Path)
    loop_rvb.add_argument("--overwrite", action="store_true")
    su2_route_vit = commands.add_parser(
        "su2-route-vit-exact",
        help="run the exact-SU2 route-token Transformer variational pilot",
    )
    su2_route_vit.add_argument("--config", required=True, type=Path)
    su2_route_vit.add_argument("--output", type=Path)
    su2_route_vit.add_argument("--overwrite", action="store_true")
    su2_route_vit_causal = commands.add_parser(
        "su2-route-vit-causal-scan",
        help="run matched locality controls and the exact-small coupling scan",
    )
    su2_route_vit_causal.add_argument("--config", required=True, type=Path)
    su2_route_vit_causal.add_argument("--output", type=Path)
    su2_route_vit_causal.add_argument("--overwrite", action="store_true")
    su2_route_vit_physical = commands.add_parser(
        "su2-route-vit-physical-routes",
        help="compare the triangular three-orientation dimer-complete route bank",
    )
    su2_route_vit_physical.add_argument("--config", required=True, type=Path)
    su2_route_vit_physical.add_argument("--output", type=Path)
    su2_route_vit_physical.add_argument("--overwrite", action="store_true")
    physics_interpretation = commands.add_parser(
        "spin-physics-interpretation",
        help=(
            "run the unified P1-P5 Route-ViT observables, RDM, spectroscopy, "
            "response, and physical error-channel audit"
        ),
    )
    physics_interpretation.add_argument("--config", required=True, type=Path)
    physics_interpretation.add_argument("--output", type=Path)
    physics_interpretation.add_argument("--overwrite", action="store_true")
    e2_resonance = commands.add_parser(
        "spin-e2-resonance-audit",
        help="audit E2-pair carriers against dimension-matched Krylov controls",
    )
    e2_resonance.add_argument("--config", required=True, type=Path)
    e2_resonance.add_argument("--output", type=Path)
    e2_resonance.add_argument("--overwrite", action="store_true")
    carrier_four_arm = commands.add_parser(
        "spin-symmetry-carrier-four-arm",
        help="compare baseline/A1/A1+E2/generic carriers at matched Ritz work",
    )
    carrier_four_arm.add_argument("--config", required=True, type=Path)
    carrier_four_arm.add_argument("--output", type=Path)
    carrier_four_arm.add_argument("--overwrite", action="store_true")
    adaptive_pair = commands.add_parser(
        "spin-adaptive-pair-carrier",
        help="test nonlinear projected-pair carrier deformations at exact support",
    )
    adaptive_pair.add_argument("--config", required=True, type=Path)
    adaptive_pair.add_argument("--output", type=Path)
    adaptive_pair.add_argument("--overwrite", action="store_true")
    su2_route_vit_n36 = commands.add_parser(
        "su2-route-vit-n36-smoke",
        help="run the triangular-6x6 projected dimer sampling/force smoke",
    )
    su2_route_vit_n36.add_argument("--config", required=True, type=Path)
    su2_route_vit_n36.add_argument("--output", type=Path)
    su2_route_vit_n36.add_argument("--overwrite", action="store_true")
    su2_route_vit_n36_audit = commands.add_parser(
        "su2-route-vit-n36-audit",
        help="audit N36 proposal mixing, full-H energy, one step, and resume",
    )
    su2_route_vit_n36_audit.add_argument("--config", required=True, type=Path)
    su2_route_vit_n36_audit.add_argument("--output", type=Path)
    su2_route_vit_n36_audit.add_argument("--overwrite", action="store_true")
    su2_route_vit_n36_basin = commands.add_parser(
        "su2-route-vit-n36-basin-gate",
        help="run the four-stratum D6/dimer-independence N36 mixing gate",
    )
    su2_route_vit_n36_basin.add_argument("--config", required=True, type=Path)
    su2_route_vit_n36_basin.add_argument("--output", type=Path)
    su2_route_vit_n36_basin.add_argument("--overwrite", action="store_true")
    carrier_vit_n36 = commands.add_parser(
        "spin-n36-carrier-vit-matched-work",
        help="compare SU2 A1/E2 carriers and common-base ViT controls at N36",
    )
    carrier_vit_n36.add_argument("--config", required=True, type=Path)
    carrier_vit_n36.add_argument("--output", type=Path)
    carrier_vit_n36.add_argument("--overwrite", action="store_true")
    multiplicity_spatial_n36 = commands.add_parser(
        "spin-n36-multiplicity-spatial-attention",
        help="run repeated paired audits of SU2-scalar multiplicity attention",
    )
    multiplicity_spatial_n36.add_argument("--config", required=True, type=Path)
    multiplicity_spatial_n36.add_argument("--output", type=Path)
    multiplicity_spatial_n36.add_argument("--overwrite", action="store_true")
    pi_flux_n36 = commands.add_parser(
        "spin-n36-pi-flux-benchmark",
        help="measure the projected pi/0-flux energy with independent chains",
    )
    pi_flux_n36.add_argument("--config", required=True, type=Path)
    pi_flux_n36.add_argument("--output", type=Path)
    pi_flux_n36.add_argument("--overwrite", action="store_true")
    operator_tangent_n36 = commands.add_parser(
        "spin-n36-operator-tangent-transfer",
        help="test physical scalar-operator response transfer from N16 to N36",
    )
    operator_tangent_n36.add_argument("--config", required=True, type=Path)
    operator_tangent_n36.add_argument("--output", type=Path)
    operator_tangent_n36.add_argument("--overwrite", action="store_true")
    preregistered_channel_n36 = commands.add_parser(
        "spin-n36-preregistered-channel-state",
        help="audit N16-selected resonance states with new N36 mixture samples",
    )
    preregistered_channel_n36.add_argument("--config", required=True, type=Path)
    preregistered_channel_n36.add_argument("--output", type=Path)
    preregistered_channel_n36.add_argument("--overwrite", action="store_true")
    operator_capacity_n64 = commands.add_parser(
        "spin-n64-operator-capacity",
        help="run the triangular 8x8 one-layer operator capacity/runtime bridge",
    )
    operator_capacity_n64.add_argument("--config", required=True, type=Path)
    operator_capacity_n64.add_argument("--output", type=Path)
    operator_capacity_n64.add_argument("--overwrite", action="store_true")
    square_operator_capacity_n36 = commands.add_parser(
        "spin-square-n36-operator-capacity",
        help="run the square 6x6 scalar-operator capacity/runtime control",
    )
    square_operator_capacity_n36.add_argument("--config", required=True, type=Path)
    square_operator_capacity_n36.add_argument("--output", type=Path)
    square_operator_capacity_n36.add_argument("--overwrite", action="store_true")
    physical_three_arm_n36 = commands.add_parser(
        "spin-n36-physical-three-arm",
        help="compare pure ViT, pi-flux correlator, and distilled exact-SU2 attention",
    )
    physical_three_arm_n36.add_argument("--config", required=True, type=Path)
    physical_three_arm_n36.add_argument("--output", type=Path)
    physical_three_arm_n36.add_argument("--overwrite", action="store_true")
    su2_scalar_dressing_n36 = commands.add_parser(
        "spin-n36-su2-scalar-dressing",
        help=(
            "compare two-Lanczos, fixed-Sz ViT, and a configuration-dependent "
            "exact-SU2 scalar carrier dressing on matched N36 chains"
        ),
    )
    su2_scalar_dressing_n36.add_argument("--config", required=True, type=Path)
    su2_scalar_dressing_n36.add_argument("--output", type=Path)
    su2_scalar_dressing_n36.add_argument("--overwrite", action="store_true")
    g1_operator_vit = commands.add_parser(
        "spin-g1-operator-vit",
        help="run the exact-small or N36 H+resonance Operator-ViT G1 gate",
    )
    g1_operator_vit.add_argument("--config", required=True, type=Path)
    g1_operator_vit.add_argument("--output", type=Path)
    g1_operator_vit.add_argument("--overwrite", action="store_true")
    g1_operator_vit.add_argument(
        "--materialize-initialization", action="store_true"
    )
    g1_operator_vit.add_argument(
        "--maximum-updates-this-invocation", type=int
    )
    g1_regularized = commands.add_parser(
        "spin-g1-regularized-operator",
        help="run separated development/audit stages for the regularized scalar space",
    )
    g1_regularized.add_argument("--config", required=True, type=Path)
    g1_regularized.add_argument("--output", type=Path)
    g1_regularized.add_argument("--overwrite", action="store_true")
    coherent_su2_carrier_n36 = commands.add_parser(
        "spin-n36-su2-coherent-carrier",
        help="test an explicit pi-flux plus exact-SU2 multiplicity channel",
    )
    coherent_su2_carrier_n36.add_argument("--config", required=True, type=Path)
    coherent_su2_carrier_n36.add_argument("--output", type=Path)
    coherent_su2_carrier_n36.add_argument("--overwrite", action="store_true")
    scalar_operator_frontier = commands.add_parser(
        "spin-scalar-operator-frontier",
        help="compare SU2/A1 scalar carrier responses with matched Krylov",
    )
    scalar_operator_frontier.add_argument("--config", required=True, type=Path)
    scalar_operator_frontier.add_argument("--output", type=Path)
    scalar_operator_frontier.add_argument("--overwrite", action="store_true")
    qsl_three_arm = commands.add_parser(
        "spin-qsl-three-arm-frontier",
        help="compare optimized pi-flux/Jastrow/Lanczos and two ViT arms",
    )
    qsl_three_arm.add_argument("--config", required=True, type=Path)
    qsl_three_arm.add_argument("--output", type=Path)
    qsl_three_arm.add_argument("--overwrite", action="store_true")
    qsl_symmetry = commands.add_parser(
        "spin-qsl-symmetry-diagnostics",
        help="audit E2, chirality, K/M, and dimer physics of the QSL frontier",
    )
    qsl_symmetry.add_argument("--config", required=True, type=Path)
    qsl_symmetry.add_argument("--output", type=Path)
    qsl_symmetry.add_argument("--overwrite", action="store_true")
    schur_config_vit = commands.add_parser(
        "spin-schur-config-vit",
        help="run the configuration-dependent exact-SU2 Schur Spatial-ViT gate",
    )
    schur_config_vit.add_argument("--config", required=True, type=Path)
    schur_config_vit.add_argument("--output", type=Path)
    schur_config_vit.add_argument("--overwrite", action="store_true")
    four_cell_schur = commands.add_parser(
        "spin-four-cell-schur-frontier",
        help="run the complete N=12/N=16 local-Schur Spatial-ViT physics gate",
    )
    four_cell_schur.add_argument("--config", required=True, type=Path)
    four_cell_schur.add_argument("--output", type=Path)
    four_cell_schur.add_argument("--overwrite", action="store_true")
    state_manifold = commands.add_parser(
        "spin-triangular-n16-state-manifold",
        help="train the complete triangular-4x4 SU2/A1 multiplicity-space ViT",
    )
    state_manifold.add_argument("--config", required=True, type=Path)
    state_manifold.add_argument("--output", type=Path)
    state_manifold.add_argument("--overwrite", action="store_true")
    many_cell_matrix_free = commands.add_parser(
        "spin-many-cell-matrix-free",
        help="replay N16 and audit the fail-closed N36 many-cell Schur gate",
    )
    many_cell_matrix_free.add_argument("--config", required=True, type=Path)
    many_cell_matrix_free.add_argument("--output", type=Path)
    many_cell_matrix_free.add_argument("--overwrite", action="store_true")
    many_cell_simplex = commands.add_parser(
        "spin-n36-many-cell-simplex-gate",
        help="run the local F-move simplex amplitude/H-ratio/checkpoint gate",
    )
    many_cell_simplex.add_argument("--config", required=True, type=Path)
    many_cell_simplex.add_argument("--output", type=Path)
    many_cell_simplex.add_argument("--overwrite", action="store_true")
    additive_joint_simplex = commands.add_parser(
        "spin-n36-additive-joint-simplex-gate",
        help="train a carrier-nested joint simplex residual on triangular 6x6",
    )
    additive_joint_simplex.add_argument("--config", required=True, type=Path)
    additive_joint_simplex.add_argument("--output", type=Path)
    additive_joint_simplex.add_argument("--overwrite", action="store_true")
    physical_fmove_carrier = commands.add_parser(
        "spin-n16-physical-fmove-carrier-gate",
        help=(
            "apply an SU2-scalar simplex/F-move operator directly to the "
            "projected-pi-flux/Krylov physical amplitudes"
        ),
    )
    physical_fmove_carrier.add_argument("--config", required=True, type=Path)
    physical_fmove_carrier.add_argument("--output", type=Path)
    physical_fmove_carrier.add_argument("--overwrite", action="store_true")
    channel_attention = commands.add_parser(
        "spin-channel-attention",
        help="compare local spin-channel transitions and staged scalar attention",
    )
    channel_attention.add_argument("--config", required=True, type=Path)
    channel_attention.add_argument("--output", type=Path)
    channel_attention.add_argument("--overwrite", action="store_true")
    channel_mechanism = commands.add_parser(
        "spin-channel-mechanism",
        help="separate local-q transfer and pure multiplicity at non-saturated rank",
    )
    channel_mechanism.add_argument("--config", required=True, type=Path)
    channel_mechanism.add_argument("--output", type=Path)
    channel_mechanism.add_argument("--overwrite", action="store_true")
    exchange_environment = commands.add_parser(
        "spin-exchange-environment",
        help="test affine environment-dependent complete exchanges at low rank",
    )
    exchange_environment.add_argument("--config", required=True, type=Path)
    exchange_environment.add_argument("--output", type=Path)
    exchange_environment.add_argument("--overwrite", action="store_true")
    environment_complement = commands.add_parser("spin-environment-complement", help="test environment beyond the complete three-exchange space")
    environment_complement.add_argument("--config", required=True, type=Path)
    environment_complement.add_argument("--output", type=Path)
    environment_complement.add_argument("--overwrite", action="store_true")
    strong_innovation = commands.add_parser("spin-strong-carrier-innovation", help="test the frozen environment operator after strong-carrier reoptimization and H-step deflation")
    strong_innovation.add_argument("--config", required=True, type=Path)
    strong_innovation.add_argument("--output", type=Path)
    environment_preflight = commands.add_parser("spin-environment-preflight", help="bounded N36 environment-action and first-derivative probe")
    environment_preflight.add_argument("--config", required=True, type=Path)
    environment_preflight.add_argument("--output", type=Path)
    environment_preflight.add_argument("--overwrite", action="store_true")
    finite_environment = commands.add_parser("spin-finite-environment-dressing", help="finite strict-A1 environment dressing with independent paired samples")
    finite_environment.add_argument("--config", required=True, type=Path)
    finite_environment.add_argument("--output", type=Path)
    finite_environment.add_argument("--overwrite", action="store_true")
    finite_audit = commands.add_parser("spin-finite-environment-audit", help="enlarged frozen-state audit with independent baseline-tail pools")
    finite_audit.add_argument("--config", required=True, type=Path)
    finite_audit.add_argument("--output", type=Path)
    finite_audit.add_argument("--stream-index", type=int, help="run one stream; omit to finalize all completed streams")
    stratified_audit = commands.add_parser("spin-stratified-environment-audit", help="frozen-state two-phase tail audit with explicit inclusion weights")
    stratified_audit.add_argument("--config", required=True, type=Path)
    stratified_audit.add_argument("--output", type=Path)
    stratified_audit.add_argument("--pool-index", type=int, help="run one independent pool; omit to finalize")
    sequential_cell_gate = commands.add_parser(
        "spin-sequential-cell-gate",
        help="audit the cell-sequential SU2 Spatial-ViT representation gate",
    )
    sequential_cell_gate.add_argument("--config", required=True, type=Path)
    sequential_cell_gate.add_argument("--output", type=Path)
    sequential_cell_gate.add_argument("--overwrite", action="store_true")
    schur_scaling = commands.add_parser(
        "spin-schur-spatial-scaling-preflight",
        help="profile N16/N36/N64 exact-SU2 spatial-Schur scaling costs",
    )
    schur_scaling.add_argument("--config", required=True, type=Path)
    schur_scaling.add_argument("--output", type=Path)
    schur_scaling.add_argument("--overwrite", action="store_true")
    scale_first_three_arm = commands.add_parser(
        "spin-scale-first-three-arm-preflight",
        help="admit the matched N36 SU2/fixed-Sz/generic device-H campaign",
    )
    scale_first_three_arm.add_argument("--config", required=True, type=Path)
    scale_first_three_arm.add_argument("--output", type=Path)
    scale_first_three_arm.add_argument("--overwrite", action="store_true")
    accelerator_slice = commands.add_parser(
        "spin-scale-first-accelerator-slice",
        help="time one N36 system/arm/batch in a fresh accelerator process",
    )
    accelerator_slice.add_argument("--config", required=True, type=Path)
    accelerator_slice.add_argument("--case-id", required=True)
    accelerator_slice.add_argument("--arm-id", required=True)
    accelerator_slice.add_argument("--batch-size", required=True, type=int)
    accelerator_slice.add_argument("--amplitude-tile-size", type=int)
    accelerator_slice.add_argument("--device-microbatch-size", type=int)
    accelerator_slice.add_argument("--output", required=True, type=Path)
    accelerator_slice.add_argument("--overwrite", action="store_true")
    accelerator_merge = commands.add_parser(
        "spin-scale-first-accelerator-merge",
        help="merge and audit all isolated N36 accelerator timing slices",
    )
    accelerator_merge.add_argument("--config", required=True, type=Path)
    accelerator_merge.add_argument("--input-root", required=True, type=Path)
    accelerator_merge.add_argument("--output", required=True, type=Path)
    accelerator_merge.add_argument("--overwrite", action="store_true")
    training_calibration = commands.add_parser(
        "spin-scale-first-training-calibration",
        help="run compile plus warm matched N36 full-H/force/SR steps",
    )
    training_calibration.add_argument("--config", required=True, type=Path)
    training_calibration.add_argument("--case-id", required=True)
    training_calibration.add_argument("--arm-id", required=True)
    training_calibration.add_argument("--output", required=True, type=Path)
    training_calibration.add_argument("--overwrite", action="store_true")
    initialization_audit = commands.add_parser(
        "spin-scale-first-initialization-audit",
        help="audit a common nonstationary initialization before N36 production",
    )
    initialization_audit.add_argument("--config", required=True, type=Path)
    initialization_audit.add_argument("--output", type=Path)
    initialization_audit.add_argument("--overwrite", action="store_true")
    initialization_pilot = commands.add_parser(
        "spin-scale-first-initialization-pilot",
        help="run checkpointed two-step N36 training with the corrected initialization",
    )
    initialization_pilot.add_argument("--config", required=True, type=Path)
    initialization_pilot.add_argument("--output", type=Path)
    initialization_pilot.add_argument("--case-id")
    initialization_pilot.add_argument("--arm-id")
    initialization_pilot.add_argument("--overwrite", action="store_true")
    convergence_merge = commands.add_parser(
        "spin-scale-first-convergence-merge",
        help="merge six one-seed N36 trajectories and apply the descent gate",
    )
    convergence_merge.add_argument("--config", required=True, type=Path)
    convergence_merge.add_argument("--input-root", required=True, type=Path)
    convergence_merge.add_argument("--output", required=True, type=Path)
    convergence_merge.add_argument("--overwrite", action="store_true")
    checkpoint_audit = commands.add_parser(
        "spin-scale-first-checkpoint-audit",
        help="evaluate sealed N36 trajectory checkpoints on fresh sample streams",
    )
    checkpoint_audit.add_argument("--config", required=True, type=Path)
    checkpoint_audit.add_argument("--checkpoint-root", required=True, type=Path)
    checkpoint_audit.add_argument("--case-id", required=True)
    checkpoint_audit.add_argument("--arm-id", required=True)
    checkpoint_audit.add_argument("--output", required=True, type=Path)
    checkpoint_audit.add_argument("--overwrite", action="store_true")
    checkpoint_audit_merge = commands.add_parser(
        "spin-scale-first-checkpoint-audit-merge",
        help="merge six held-out N36 checkpoint-audit slices",
    )
    checkpoint_audit_merge.add_argument("--config", required=True, type=Path)
    checkpoint_audit_merge.add_argument("--input-root", required=True, type=Path)
    checkpoint_audit_merge.add_argument("--output", required=True, type=Path)
    checkpoint_audit_merge.add_argument("--overwrite", action="store_true")
    plaquette_carrier = commands.add_parser(
        "spin-scale-first-plaquette-carrier-gate",
        help="screen a one-angle Gamma-A1 plaquette-RVB carrier on cached N16 support",
    )
    plaquette_carrier.add_argument("--config", required=True, type=Path)
    plaquette_carrier.add_argument("--output", type=Path)
    plaquette_carrier.add_argument("--overwrite", action="store_true")
    coherent_frame = commands.add_parser(
        "spin-coherent-plaquette-frame-gate",
        help="compare coherent translated/rotated plaquette carriers with geometry scrambles",
    )
    coherent_frame.add_argument("--config", required=True, type=Path)
    coherent_frame.add_argument("--output", type=Path)
    coherent_frame.add_argument("--overwrite", action="store_true")
    coherent_frame_n36 = commands.add_parser(
        "spin-n36-coherent-frame-support",
        help="measure coherent-frame inter-cell virtual-spin support on 6x6",
    )
    coherent_frame_n36.add_argument("--config", required=True, type=Path)
    coherent_frame_n36.add_argument("--output", type=Path)
    coherent_frame_n36.add_argument("--overwrite", action="store_true")
    coherent_frame_canary = commands.add_parser(
        "spin-n36-coherent-frame-canary",
        help="run the finite-FS/KL exact-SU2 coherent-frame energy canary",
    )
    coherent_frame_canary.add_argument("--config", required=True, type=Path)
    coherent_frame_canary.add_argument("--output", type=Path)
    coherent_frame_canary.add_argument("--overwrite", action="store_true")
    multibody_frame = commands.add_parser(
        "spin-n36-multibody-frame-resonance",
        help="cross-fit the full exact-SU2 local-singlet-count frame bank",
    )
    multibody_frame.add_argument("--config", required=True, type=Path)
    multibody_frame.add_argument("--output", type=Path)
    multibody_frame.add_argument("--overwrite", action="store_true")
    imaginary_time_compression = commands.add_parser(
        "spin-n16-imaginary-time-compression",
        help="compress one exact finite imaginary-time step into the balanced Schur ViT",
    )
    imaginary_time_compression.add_argument("--config", required=True, type=Path)
    imaginary_time_compression.add_argument("--output", type=Path)
    imaginary_time_compression.add_argument("--overwrite", action="store_true")
    imaginary_time_tangent = commands.add_parser(
        "spin-n16-imaginary-time-tangent",
        help="project the exact imaginary-time response into the balanced Schur ViT tangent",
    )
    imaginary_time_tangent.add_argument("--config", required=True, type=Path)
    imaginary_time_tangent.add_argument("--output", type=Path)
    imaginary_time_tangent.add_argument("--overwrite", action="store_true")
    hamiltonian_filtration = commands.add_parser(
        "spin-n16-hamiltonian-filtration",
        help="locate the exact energy residual across bond classes and Schur decisions",
    )
    hamiltonian_filtration.add_argument("--config", required=True, type=Path)
    hamiltonian_filtration.add_argument("--output", type=Path)
    hamiltonian_filtration.add_argument("--overwrite", action="store_true")
    bond_causal_tangent = commands.add_parser(
        "spin-n16-bond-causal-tangent",
        help="compare a physical bond-causal Schur-ViT tangent with bond scrambles",
    )
    bond_causal_tangent.add_argument("--config", required=True, type=Path)
    bond_causal_tangent.add_argument("--output", type=Path)
    bond_causal_tangent.add_argument("--overwrite", action="store_true")
    bond_causal_multistep = commands.add_parser(
        "spin-n16-bond-causal-multistep",
        help="run paired exact multistep bond-causal heads and untouched sampler audits",
    )
    bond_causal_multistep.add_argument("--config", required=True, type=Path)
    bond_causal_multistep.add_argument("--output", type=Path)
    bond_causal_multistep.add_argument("--overwrite", action="store_true")
    causal_operator_oracle = commands.add_parser(
        "spin-n16-causal-operator-oracle",
        help="test exact causal Hamiltonian-residual messages against scrambles and two-Lanczos",
    )
    causal_operator_oracle.add_argument("--config", required=True, type=Path)
    causal_operator_oracle.add_argument("--output", type=Path)
    causal_operator_oracle.add_argument("--overwrite", action="store_true")
    krylov_causal_oracle = commands.add_parser(
        "spin-n16-krylov-causal-oracle",
        help="test causal operator dressing beyond a two-Lanczos carrier",
    )
    krylov_causal_oracle.add_argument("--config", required=True, type=Path)
    krylov_causal_oracle.add_argument("--output", type=Path)
    krylov_causal_oracle.add_argument("--overwrite", action="store_true")
    causal_krylov_hybrid = commands.add_parser(
        "spin-n16-causal-krylov-hybrid",
        help="compare a causal extra state with a rank-matched next Krylov column",
    )
    causal_krylov_hybrid.add_argument("--config", required=True, type=Path)
    causal_krylov_hybrid.add_argument("--output", type=Path)
    causal_krylov_hybrid.add_argument("--overwrite", action="store_true")
    carrier_distillation = commands.add_parser(
        "spin-scale-first-carrier-distillation",
        help="encode one exact N16 plaquette-RVB carrier in all three backbones",
    )
    carrier_distillation.add_argument("--config", required=True, type=Path)
    carrier_distillation.add_argument("--output", type=Path)
    carrier_distillation.add_argument("--overwrite", action="store_true")
    carrier_transfer = commands.add_parser(
        "spin-scale-first-carrier-transfer-preflight",
        help="load N16 carrier artifacts into N36 templates and run local full-H canaries",
    )
    carrier_transfer.add_argument("--config", required=True, type=Path)
    carrier_transfer.add_argument("--output", type=Path)
    carrier_transfer.add_argument("--overwrite", action="store_true")
    n36_carrier_distillation = commands.add_parser(
        "spin-scale-first-n36-carrier-distillation",
        help="initialize N36 arms directly from analytic plaquette-carrier samples",
    )
    n36_carrier_distillation.add_argument("--config", required=True, type=Path)
    n36_carrier_distillation.add_argument("--output", type=Path)
    n36_carrier_distillation.add_argument("--overwrite", action="store_true")
    carrier_single_seed = commands.add_parser(
        "spin-scale-first-carrier-single-seed",
        help="run one checkpointed N36 VMC arm from the admitted carrier",
    )
    carrier_single_seed.add_argument("--config", required=True, type=Path)
    carrier_single_seed.add_argument("--case-id", required=True)
    carrier_single_seed.add_argument("--arm-id", required=True)
    carrier_single_seed.add_argument("--output", required=True, type=Path)
    carrier_single_seed.add_argument("--maximum-steps-this-call", type=int)
    carrier_single_seed.add_argument("--overwrite", action="store_true")
    carrier_checkpoint_audit = commands.add_parser(
        "spin-scale-first-carrier-checkpoint-audit",
        help="audit carrier-initialized N36 checkpoints on untouched sample streams",
    )
    carrier_checkpoint_audit.add_argument("--config", required=True, type=Path)
    carrier_checkpoint_audit.add_argument("--checkpoint-root", required=True, type=Path)
    carrier_checkpoint_audit.add_argument("--case-id", required=True)
    carrier_checkpoint_audit.add_argument("--arm-id", required=True)
    carrier_checkpoint_audit.add_argument("--output", required=True, type=Path)
    carrier_checkpoint_audit.add_argument("--overwrite", action="store_true")
    carrier_checkpoint_merge = commands.add_parser(
        "spin-scale-first-carrier-checkpoint-audit-merge",
        help="merge carrier checkpoint audits and apply the production-release gate",
    )
    carrier_checkpoint_merge.add_argument("--config", required=True, type=Path)
    carrier_checkpoint_merge.add_argument("--input-root", required=True, type=Path)
    carrier_checkpoint_merge.add_argument("--output", required=True, type=Path)
    carrier_checkpoint_merge.add_argument("--overwrite", action="store_true")
    carrier_convergence_merge = commands.add_parser(
        "spin-scale-first-carrier-convergence-merge",
        help="merge six carrier-initialized one-seed trajectories before audit",
    )
    carrier_convergence_merge.add_argument("--config", required=True, type=Path)
    carrier_convergence_merge.add_argument("--input-root", required=True, type=Path)
    carrier_convergence_merge.add_argument("--output", required=True, type=Path)
    carrier_convergence_merge.add_argument("--overwrite", action="store_true")
    carrier_schur_audit = commands.add_parser(
        "spin-scale-first-carrier-schur-channel-audit",
        help="measure local and inter-cell spin-channel support in sealed SU2 checkpoints",
    )
    carrier_schur_audit.add_argument("--config", required=True, type=Path)
    carrier_schur_audit.add_argument("--checkpoint-root", required=True, type=Path)
    carrier_schur_audit.add_argument("--output", type=Path)
    carrier_schur_audit.add_argument("--overwrite", action="store_true")
    carrier_cell_circuit = commands.add_parser(
        "spin-carrier-cell-scalar-circuit",
        help="falsify a carrier-preserving local-Schur scalar circuit",
    )
    carrier_cell_circuit.add_argument("--config", required=True, type=Path)
    carrier_cell_circuit.add_argument("--output", type=Path)
    carrier_cell_circuit.add_argument("--overwrite", action="store_true")
    entanglement_channel = commands.add_parser(
        "spin-entanglement-channel-audit",
        help="resolve sealed N16 errors into SU2 boundary-spin channels",
    )
    entanglement_channel.add_argument("--config", required=True, type=Path)
    entanglement_channel.add_argument("--output", type=Path)
    entanglement_channel.add_argument("--overwrite", action="store_true")
    entanglement_filter = commands.add_parser(
        "spin-entanglement-filter-gate",
        help="run the SU2 Schmidt-transport and matched filter gates",
    )
    entanglement_filter.add_argument("--config", required=True, type=Path)
    entanglement_filter.add_argument("--output", type=Path)
    entanglement_filter.add_argument("--overwrite", action="store_true")
    entanglement_bridge = commands.add_parser(
        "spin-entanglement-filter-scalable-bridge",
        help="compress the N16 SU2 entanglement filter into a transferable NQS",
    )
    entanglement_bridge.add_argument("--config", required=True, type=Path)
    entanglement_bridge.add_argument("--output", type=Path)
    entanglement_bridge.add_argument("--overwrite", action="store_true")
    projected_multiplicity = commands.add_parser(
        "spin-projected-carrier-multiplicity-dressing",
        help="train the exact-S0/Gamma-A1 carrier plus multiplicity dressing gate",
    )
    projected_multiplicity.add_argument("--config", required=True, type=Path)
    projected_multiplicity.add_argument("--output", type=Path)
    projected_multiplicity.add_argument("--overwrite", action="store_true")
    krylov_full_vit = commands.add_parser(
        "spin-n16-krylov-anchored-full-vit",
        help="compare the full exact-SU2/fixed-Sz ViTs with degree-four Krylov",
    )
    krylov_full_vit.add_argument("--config", required=True, type=Path)
    krylov_full_vit.add_argument("--output", type=Path)
    krylov_full_vit.add_argument("--overwrite", action="store_true")
    full_vit_sr = commands.add_parser(
        "spin-n16-full-vit-sr-reachability",
        help="audit full-manifold tangent rank and exact-state SR reachability",
    )
    full_vit_sr.add_argument("--config", required=True, type=Path)
    full_vit_sr.add_argument("--output", type=Path)
    full_vit_sr.add_argument("--overwrite", action="store_true")
    full_vit_sr_continuation = commands.add_parser(
        "spin-n16-full-vit-sr-continuation",
        help="run the one bounded continuation of the sealed N16 trust-region state",
    )
    full_vit_sr_continuation.add_argument("--config", required=True, type=Path)
    full_vit_sr_continuation.add_argument("--output", type=Path)
    full_vit_sr_continuation.add_argument("--overwrite", action="store_true")
    spectral_tomography = commands.add_parser(
        "spin-n16-spectral-error-tomography",
        help="test matched-rank neural/Krylov complementarity in the sealed N16 sector",
    )
    spectral_tomography.add_argument("--config", required=True, type=Path)
    spectral_tomography.add_argument("--output", type=Path)
    spectral_tomography.add_argument("--overwrite", action="store_true")
    n36_projected_multiplicity = commands.add_parser(
        "spin-n36-projected-multiplicity-preflight",
        help="run the matched-work 6x6 projected-multiplicity sampled gate",
    )
    n36_projected_multiplicity.add_argument("--config", required=True, type=Path)
    n36_projected_multiplicity.add_argument("--output", type=Path)
    n36_projected_multiplicity.add_argument("--overwrite", action="store_true")
    schur_sector_frontier = commands.add_parser(
        "spin-schur-sector-matched-frontier",
        help="run strict N12 sector heads and N12/N16 matched-work controls",
    )
    schur_sector_frontier.add_argument("--config", required=True, type=Path)
    schur_sector_frontier.add_argument("--output", type=Path)
    schur_sector_frontier.add_argument("--overwrite", action="store_true")
    additive_carrier = commands.add_parser(
        "spin-four-cell-additive-carrier",
        help="compare physical and scrambled additive carriers on the Schur ViT",
    )
    additive_carrier.add_argument("--config", required=True, type=Path)
    additive_carrier.add_argument("--output", type=Path)
    additive_carrier.add_argument("--overwrite", action="store_true")
    additive_confirmation = commands.add_parser(
        "spin-four-cell-additive-carrier-confirmation",
        help="confirm the additive-carrier basin gain on untouched seeds",
    )
    additive_confirmation.add_argument("--config", required=True, type=Path)
    additive_confirmation.add_argument("--output", type=Path)
    additive_confirmation.add_argument("--overwrite", action="store_true")
    paired_z2 = commands.add_parser(
        "spin-four-cell-paired-z2-frontier",
        help="run the development or untouched paired-Z2 exact-small gate",
    )
    paired_z2.add_argument("--config", required=True, type=Path)
    paired_z2.add_argument("--output", type=Path)
    paired_z2.add_argument("--overwrite", action="store_true")
    scalar_correlator = commands.add_parser(
        "spin-four-cell-scalar-correlator-frontier",
        help="run the development or untouched SU2-scalar-correlator gate",
    )
    scalar_correlator.add_argument("--config", required=True, type=Path)
    scalar_correlator.add_argument("--output", type=Path)
    scalar_correlator.add_argument("--overwrite", action="store_true")
    operator_scalar = commands.add_parser(
        "spin-four-cell-operator-scalar-frontier",
        help="run the development or untouched SU2-scalar-operator gate",
    )
    operator_scalar.add_argument("--config", required=True, type=Path)
    operator_scalar.add_argument("--output", type=Path)
    operator_scalar.add_argument("--overwrite", action="store_true")
    staged_operator = commands.add_parser(
        "spin-four-cell-staged-operator-scalar",
        help="run the shared-warmup staged SU2-scalar-operator gate",
    )
    staged_operator.add_argument("--config", required=True, type=Path)
    staged_operator.add_argument("--output", type=Path)
    staged_operator.add_argument("--overwrite", action="store_true")
    overlapping_recoupling = commands.add_parser(
        "spin-four-cell-overlapping-recoupling",
        help="run the exact-A1 overlapping local-Schur recoupling gate",
    )
    overlapping_recoupling.add_argument("--config", required=True, type=Path)
    overlapping_recoupling.add_argument("--output", type=Path)
    overlapping_recoupling.add_argument("--overwrite", action="store_true")
    recoupling_ritz = commands.add_parser(
        "spin-four-cell-recoupling-ritz",
        help="run the shared-backbone exact Ritz recoupling calibration",
    )
    recoupling_ritz.add_argument("--config", required=True, type=Path)
    recoupling_ritz.add_argument("--output", type=Path)
    recoupling_ritz.add_argument("--overwrite", action="store_true")
    recoupling_vp = commands.add_parser(
        "spin-four-cell-recoupling-variable-projection",
        help="run trainable exact-small recoupling variable projection",
    )
    recoupling_vp.add_argument("--config", required=True, type=Path)
    recoupling_vp.add_argument("--output", type=Path)
    recoupling_vp.add_argument("--overwrite", action="store_true")
    recoupling_vp_audit = commands.add_parser(
        "spin-four-cell-recoupling-variable-projection-audit",
        help="run untouched seeds for the frozen recoupling variable projection",
    )
    recoupling_vp_audit.add_argument("--config", required=True, type=Path)
    recoupling_vp_audit.add_argument("--output", type=Path)
    recoupling_vp_audit.add_argument("--overwrite", action="store_true")
    symmetry_response = commands.add_parser(
        "spin-symmetry-response-operator",
        help="test preregistered E2/B1/B2/A2 response-return loops",
    )
    symmetry_response.add_argument("--config", required=True, type=Path)
    symmetry_response.add_argument("--output", type=Path)
    symmetry_response.add_argument("--overwrite", action="store_true")
    b2_response_audit = commands.add_parser(
        "spin-square-b2-response-audit",
        help="audit the frozen square B2 response-return hypothesis",
    )
    b2_response_audit.add_argument("--config", required=True, type=Path)
    b2_response_audit.add_argument("--output", type=Path)
    b2_response_audit.add_argument("--overwrite", action="store_true")
    conditioned_response = commands.add_parser(
        "spin-conditioned-response-gate",
        help="test a configuration-dependent SU2-scalar response gate",
    )
    conditioned_response.add_argument("--config", required=True, type=Path)
    conditioned_response.add_argument("--output", type=Path)
    conditioned_response.add_argument("--overwrite", action="store_true")
    labelled_loop = commands.add_parser(
        "spin-symmetry-labelled-loop-screen",
        help="screen E2/A2 virtual loop-resonance returns on frozen SU2 states",
    )
    labelled_loop.add_argument("--config", required=True, type=Path)
    labelled_loop.add_argument("--output", type=Path)
    labelled_loop.add_argument("--overwrite", action="store_true")
    loop_coupling = commands.add_parser(
        "spin-symmetry-loop-coupling-scan",
        help="test phase-selective A2 virtual-loop response on fresh SU2 states",
    )
    loop_coupling.add_argument("--config", required=True, type=Path)
    loop_coupling.add_argument("--output", type=Path)
    loop_coupling.add_argument("--overwrite", action="store_true")
    spatial_vit_sr = commands.add_parser(
        "spatial-vit-sr-canary",
        help="run the matrix-free exact-state SR optimizer-rescue canary",
    )
    spatial_vit_sr.add_argument("--config", required=True, type=Path)
    spatial_vit_sr.add_argument("--output", type=Path)
    spatial_vit_sr.add_argument("--overwrite", action="store_true")
    strict_schur = commands.add_parser(
        "strict-schur-spatial-gate",
        help="seal the N=4,6,8 strict local-Schur/support-ledger exact gate",
    )
    strict_schur.add_argument("--config", required=True, type=Path)
    strict_schur.add_argument("--output", type=Path)
    strict_schur.add_argument("--overwrite", action="store_true")
    strict_schur_n16 = commands.add_parser(
        "strict-schur-spatial-n16-smoke",
        help="seal the square-4x4 strict Schur/CG forward-JVP-Hpsi smoke",
    )
    strict_schur_n16.add_argument("--config", required=True, type=Path)
    strict_schur_n16.add_argument("--output", type=Path)
    strict_schur_n16.add_argument("--overwrite", action="store_true")
    triangle_patch_n36 = commands.add_parser(
        "triangle-patch-n36-runtime-canary",
        help="profile the unprojected square/triangular 6x6 strict-CG kernels",
    )
    triangle_patch_n36.add_argument("--config", required=True, type=Path)
    triangle_patch_n36.add_argument("--output", type=Path)
    triangle_patch_n36.add_argument("--overwrite", action="store_true")
    triangle_patch_energy = commands.add_parser(
        "triangle-patch-n36-energy-smoke",
        help="run the projected ten-update 6x6 strict-CG energy smoke",
    )
    triangle_patch_energy.add_argument("--config", required=True, type=Path)
    triangle_patch_energy.add_argument("--output", type=Path)
    triangle_patch_energy.add_argument("--overwrite", action="store_true")
    triangle_patch_init = commands.add_parser(
        "triangle-patch-n36-physical-init-readout",
        help="seal the zero-update projected physical-cluster initialization readout",
    )
    triangle_patch_init.add_argument("--config", required=True, type=Path)
    triangle_patch_init.add_argument("--output", type=Path)
    triangle_patch_init.add_argument("--overwrite", action="store_true")
    triangle_patch_sr = commands.add_parser(
        "triangle-patch-n36-sr-pilot",
        help="run the resumable two-update projected matrix-free-SR canary",
    )
    triangle_patch_sr.add_argument("--config", required=True, type=Path)
    triangle_patch_sr.add_argument("--output", type=Path)
    triangle_patch_sr.add_argument("--overwrite", action="store_true")
    triangle_patch_resonance = commands.add_parser(
        "triangle-patch-n36-low-spin-resonance",
        help="run the nested frozen-core q=0/q=2 causal profile",
    )
    triangle_patch_resonance.add_argument("--config", required=True, type=Path)
    triangle_patch_resonance.add_argument("--output", type=Path)
    triangle_patch_resonance.add_argument("--overwrite", action="store_true")
    triangle_patch_resonance.add_argument(
        "--case-id",
        choices=("square_6x6_j2_0p5", "triangular_6x6_j2_0p125"),
    )
    triangle_patch_resonance.add_argument(
        "--smoke",
        action="store_true",
        help="reduce only Monte Carlo work and seal the execution as a smoke",
    )
    triangle_patch_resonance.add_argument(
        "--local-pilot",
        action="store_true",
        help="run and seal the intermediate 64-train/32-audit-sample laptop tier",
    )
    observable = commands.add_parser(
        "observable-validation",
        help="seal the exact-small full-spin structure-factor validation",
    )
    observable.add_argument("--config", required=True, type=Path)
    observable.add_argument("--output", type=Path)
    observable.add_argument("--overwrite", action="store_true")
    runtime = commands.add_parser(
        "sampled-runtime",
        help="run an on-demand physical-spin MCMC/runtime readiness smoke",
    )
    runtime.add_argument("--config", required=True, type=Path)
    runtime.add_argument("--output", type=Path)
    runtime.add_argument("--overwrite", action="store_true")
    observable_probe = commands.add_parser(
        "sampled-observable-probe",
        help="probe full-spin observables from sealed 6x6 runtime checkpoints",
    )
    observable_probe.add_argument("--config", required=True, type=Path)
    observable_probe.add_argument("--output", type=Path)
    observable_probe.add_argument("--overwrite", action="store_true")
    sampled_training = commands.add_parser(
        "sampled-training",
        help="run a resumable projected physical-spin sampled-Adam preflight",
    )
    sampled_training.add_argument("--config", required=True, type=Path)
    sampled_training.add_argument("--output", type=Path)
    sampled_training.add_argument("--overwrite", action="store_true")
    optimizer_scan_v2 = commands.add_parser(
        "sampled-optimizer-scan-v2",
        help="run one resumable center-only Adam learning-rate candidate",
    )
    optimizer_scan_v2.add_argument("--config", required=True, type=Path)
    optimizer_scan_v2.add_argument("--output", type=Path)
    optimizer_scan_v2.add_argument("--overwrite", action="store_true")
    optimizer_select_v2 = commands.add_parser(
        "sampled-optimizer-select-v2",
        help="audit three sealed v2 LR candidates and select one route per geometry",
    )
    optimizer_select_v2.add_argument("--config", required=True, type=Path)
    optimizer_select_v2.add_argument("--output", type=Path)
    optimizer_select_v2.add_argument("--overwrite", action="store_true")
    optimizer_extension_v1 = commands.add_parser(
        "sampled-optimizer-confirmatory-extension-v1",
        help=(
            "preflight or run the parent-bound center-only Adam 50-to-100 "
            "confirmatory extension"
        ),
    )
    optimizer_extension_v1.add_argument("--config", required=True, type=Path)
    optimizer_extension_v1.add_argument("--output", type=Path)
    optimizer_extension_v1.add_argument("--overwrite", action="store_true")
    optimizer_extension_v1.add_argument(
        "--preflight-only",
        action="store_true",
        help="audit selector/parents and model identities without training",
    )
    optimizer_select_metric_correction = commands.add_parser(
        "sampled-optimizer-select-v2-metric-correction",
        help=(
            "audit the immutable literal selector and apply the registered "
            "conditional-acceptance sensitivity"
        ),
    )
    optimizer_select_metric_correction.add_argument(
        "--config", required=True, type=Path
    )
    optimizer_select_metric_correction.add_argument("--output", type=Path)
    optimizer_select_metric_correction.add_argument(
        "--overwrite", action="store_true"
    )
    training_analysis = commands.add_parser(
        "sampled-training-post-analysis",
        help="replay and audit sealed sampled-training fixed-parameter blocks",
    )
    training_analysis.add_argument("--config", required=True, type=Path)
    training_analysis.add_argument("--output", type=Path)
    training_analysis.add_argument("--overwrite", action="store_true")
    analyze = commands.add_parser(
        "analyze-prl",
        help="validate and summarize the sealed local PRL Pareto campaigns",
    )
    analyze.add_argument("--base", required=True, type=Path)
    analyze.add_argument("--long", required=True, type=Path)
    analyze.add_argument("--su2-long", type=Path)
    analyze.add_argument("--output", required=True, type=Path)
    analyze.add_argument("--bootstrap-resamples", type=int, default=100_000)
    analyze.add_argument("--bootstrap-seed", type=int, default=910_240)
    analyze.add_argument("--time-grid-size", type=int, default=31)
    commands.add_parser("notebook", help="rebuild the compact paper-facing notebook")
    commands.add_parser(
        "notebook-exploration",
        help="rebuild the comprehensive diagnostic/exploration notebook",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "notebook":
        from projects.clebsch_tree_nqs.build_notebook import build

        print(build())
        return 0
    if args.command == "notebook-exploration":
        from projects.clebsch_tree_nqs.build_exploration_notebook import build

        print(build())
        return 0

    if args.command == "sampled-preflight":
        from projects.clebsch_tree_nqs.workflows.spin_sampled_preflight import (
            run_spin_sampled_preflight,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_sampled_preflight(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "singlet-force-efficiency":
        from projects.clebsch_tree_nqs.workflows.spin_singlet_force_efficiency import (
            run_spin_singlet_force_efficiency,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_singlet_force_efficiency(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spatial-cg-exact":
        from projects.clebsch_tree_nqs.workflows.spin_spatial_cg import (
            run_spin_spatial_cg,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_spatial_cg(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spatial-cg-capacity":
        from projects.clebsch_tree_nqs.workflows.spin_spatial_cg_capacity import (
            run_spin_spatial_cg_capacity,
        )

        output = args.output or PROJECT / "results" / "spin_spatial_cg_capacity_v1"
        run_spin_spatial_cg_capacity(
            args.config,
            args.pilot,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spatial-vit-exact":
        from projects.clebsch_tree_nqs.workflows.spin_spatial_vit import (
            run_spin_spatial_vit,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_spatial_vit(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-rvb-covering-frontier":
        from projects.clebsch_tree_nqs.workflows.spin_rvb_covering_frontier import (
            run_spin_rvb_covering_frontier,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_rvb_covering_frontier(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-rvb-covering-matrix-free-gate":
        from projects.clebsch_tree_nqs.workflows.spin_rvb_covering_matrix_free_gate import (
            run_spin_rvb_covering_matrix_free_gate,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_rvb_covering_matrix_free_gate(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-flux-loop-interference-confirmation":
        from projects.clebsch_tree_nqs.workflows.spin_flux_loop_interference_confirmation import (
            run_spin_flux_loop_interference_confirmation,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_flux_loop_interference_confirmation(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-n36-flux-loop-preflight":
        from projects.clebsch_tree_nqs.workflows.spin_n36_flux_loop_preflight import (
            run_spin_n36_flux_loop_preflight,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_n36_flux_loop_preflight(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0
    if args.command == "spin-n36-relative-flux-crossfit":
        from projects.clebsch_tree_nqs.workflows.spin_n36_relative_flux_crossfit import (
            run_spin_n36_relative_flux_crossfit,
        )

        output = args.output or (
            PROJECT / "results/spin_n36_relative_flux_crossfit_v1"
        )
        run_spin_n36_relative_flux_crossfit(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0
    if args.command == "spin-neural-backflow-pfaffian":
        from projects.clebsch_tree_nqs.workflows.spin_neural_backflow_pfaffian import (
            run_spin_neural_backflow_pfaffian,
        )

        output = args.output or (
            PROJECT / "results/spin_neural_backflow_pfaffian_v1"
        )
        run_spin_neural_backflow_pfaffian(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0
    if args.command == "spin-residual-neural-backflow":
        from projects.clebsch_tree_nqs.workflows.spin_residual_neural_backflow import (
            run_spin_residual_neural_backflow,
        )

        output = args.output or (
            PROJECT / "results/spin_residual_neural_backflow_v1"
        )
        run_spin_residual_neural_backflow(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0
    if args.command == "spin-resonating-neural-pfaffian":
        from projects.clebsch_tree_nqs.workflows.spin_resonating_neural_pfaffian import (
            run_spin_resonating_neural_pfaffian,
        )

        output = args.output or (
            PROJECT / "results/spin_resonating_neural_pfaffian_v1"
        )
        run_spin_resonating_neural_pfaffian(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0
    if args.command == "spin-long-range-loop-rvb":
        from projects.clebsch_tree_nqs.workflows.spin_long_range_loop_rvb import (
            run_spin_long_range_loop_rvb,
        )

        output = args.output or (
            PROJECT / "results/spin_long_range_loop_rvb_v1"
        )
        run_spin_long_range_loop_rvb(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spatial-vit-sr-canary":
        from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_sr import (
            run_spin_spatial_vit_sr,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_spatial_vit_sr(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "su2-route-vit-exact":
        from projects.clebsch_tree_nqs.workflows.spin_su2_route_vit import (
            run_spin_su2_route_vit,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_su2_route_vit(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "su2-route-vit-n36-smoke":
        from projects.clebsch_tree_nqs.workflows.spin_su2_route_vit_n36_smoke import (
            run_spin_su2_route_vit_n36_smoke,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_su2_route_vit_n36_smoke(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "su2-route-vit-n36-audit":
        from projects.clebsch_tree_nqs.workflows.spin_su2_route_vit_n36_audit import (
            run_spin_su2_route_vit_n36_audit,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_su2_route_vit_n36_audit(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "su2-route-vit-n36-basin-gate":
        from projects.clebsch_tree_nqs.workflows.spin_su2_route_vit_n36_basin_gate import (
            run_spin_su2_route_vit_n36_basin_gate,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_su2_route_vit_n36_basin_gate(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-n36-carrier-vit-matched-work":
        from projects.clebsch_tree_nqs.workflows.spin_n36_carrier_vit_matched_work import (
            run_spin_n36_carrier_vit_matched_work,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_n36_carrier_vit_matched_work(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-n36-multiplicity-spatial-attention":
        from projects.clebsch_tree_nqs.workflows.spin_n36_multiplicity_spatial_attention import (
            run_spin_n36_multiplicity_spatial_attention,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_n36_multiplicity_spatial_attention(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-n36-pi-flux-benchmark":
        from projects.clebsch_tree_nqs.workflows.spin_n36_pi_flux_benchmark import (
            run_spin_n36_pi_flux_benchmark,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_n36_pi_flux_benchmark(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-n36-operator-tangent-transfer":
        from projects.clebsch_tree_nqs.workflows.spin_n36_operator_tangent_transfer import (
            run_spin_n36_operator_tangent_transfer,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_n36_operator_tangent_transfer(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-n36-preregistered-channel-state":
        from projects.clebsch_tree_nqs.workflows.spin_n36_preregistered_channel_state import (
            run_spin_n36_preregistered_channel_state,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_n36_preregistered_channel_state(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-n64-operator-capacity":
        from projects.clebsch_tree_nqs.workflows.spin_n64_operator_capacity import (
            run_spin_n64_operator_capacity,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_n64_operator_capacity(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-square-n36-operator-capacity":
        from projects.clebsch_tree_nqs.workflows.spin_square_n36_operator_capacity import (
            run_spin_square_n36_operator_capacity,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_square_n36_operator_capacity(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-n36-physical-three-arm":
        from projects.clebsch_tree_nqs.workflows.spin_n36_physical_three_arm import (
            run_spin_n36_physical_three_arm,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_n36_physical_three_arm(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-n36-su2-scalar-dressing":
        from projects.clebsch_tree_nqs.workflows.spin_n36_su2_scalar_dressing import (
            run_spin_n36_su2_scalar_dressing,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_n36_su2_scalar_dressing(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-n36-su2-coherent-carrier":
        from projects.clebsch_tree_nqs.workflows.spin_n36_su2_coherent_carrier import (
            run_spin_n36_su2_coherent_carrier,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_n36_su2_coherent_carrier(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-scalar-operator-frontier":
        from projects.clebsch_tree_nqs.workflows.spin_scalar_operator_frontier import (
            run_spin_scalar_operator_frontier,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_scalar_operator_frontier(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-qsl-three-arm-frontier":
        from projects.clebsch_tree_nqs.workflows.spin_qsl_three_arm_frontier import (
            run_spin_qsl_three_arm_frontier,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_qsl_three_arm_frontier(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-qsl-symmetry-diagnostics":
        from projects.clebsch_tree_nqs.workflows.spin_qsl_symmetry_diagnostics import (
            run_spin_qsl_symmetry_diagnostics,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_qsl_symmetry_diagnostics(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-schur-config-vit":
        from projects.clebsch_tree_nqs.workflows.spin_schur_config_vit import (
            run_spin_schur_config_vit,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_schur_config_vit(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-four-cell-schur-frontier":
        from projects.clebsch_tree_nqs.workflows.spin_four_cell_schur_frontier import (
            run_spin_four_cell_schur_frontier,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_four_cell_schur_frontier(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-triangular-n16-state-manifold":
        from projects.clebsch_tree_nqs.workflows.spin_triangular_n16_state_manifold import (
            run_spin_triangular_n16_state_manifold,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_triangular_n16_state_manifold(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-many-cell-matrix-free":
        from projects.clebsch_tree_nqs.workflows.spin_many_cell_matrix_free import (
            run_spin_many_cell_matrix_free,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_many_cell_matrix_free(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-n36-many-cell-simplex-gate":
        from projects.clebsch_tree_nqs.workflows.spin_n36_many_cell_simplex_gate import (
            run_spin_n36_many_cell_simplex_gate,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_n36_many_cell_simplex_gate(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-n36-additive-joint-simplex-gate":
        from projects.clebsch_tree_nqs.workflows.spin_n36_additive_joint_simplex_gate import (
            run_spin_n36_additive_joint_simplex_gate,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_n36_additive_joint_simplex_gate(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-n16-physical-fmove-carrier-gate":
        from projects.clebsch_tree_nqs.workflows.spin_n16_physical_fmove_carrier_gate import (
            run_spin_n16_physical_fmove_carrier_gate,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_n16_physical_fmove_carrier_gate(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-channel-attention":
        from projects.clebsch_tree_nqs.workflows.spin_channel_attention import run_spin_channel_attention

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_channel_attention(args.config, output, overwrite=args.overwrite)
        return 0

    if args.command == "spin-channel-mechanism":
        from projects.clebsch_tree_nqs.workflows.spin_channel_mechanism import run_spin_channel_mechanism

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_channel_mechanism(args.config, output, overwrite=args.overwrite)
        return 0

    if args.command == "spin-stratified-environment-audit":
        from projects.clebsch_tree_nqs.workflows.spin_stratified_environment_audit import run_spin_stratified_environment_audit

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_stratified_environment_audit(args.config, output, pool_index=args.pool_index)
        return 0

    if args.command == "spin-finite-environment-audit":
        from projects.clebsch_tree_nqs.workflows.spin_finite_environment_audit import run_spin_finite_environment_audit

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_finite_environment_audit(args.config, output, stream_index=args.stream_index)
        return 0

    if args.command == "spin-finite-environment-dressing":
        from projects.clebsch_tree_nqs.workflows.spin_finite_environment_dressing import run_spin_finite_environment_dressing

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_finite_environment_dressing(args.config, output, overwrite=args.overwrite)
        return 0

    if args.command == "spin-environment-preflight":
        from projects.clebsch_tree_nqs.workflows.spin_environment_preflight import run_spin_environment_preflight

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_environment_preflight(args.config, output, overwrite=args.overwrite)
        return 0

    if args.command == "spin-strong-carrier-innovation":
        from projects.clebsch_tree_nqs.workflows.spin_strong_carrier_innovation import run_spin_strong_carrier_innovation

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_strong_carrier_innovation(args.config, output)
        return 0

    if args.command == "spin-environment-complement":
        from projects.clebsch_tree_nqs.workflows.spin_environment_complement import run_spin_environment_complement

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_environment_complement(args.config, output, overwrite=args.overwrite)
        return 0

    if args.command == "spin-exchange-environment":
        from projects.clebsch_tree_nqs.workflows.spin_exchange_environment import run_spin_exchange_environment

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_exchange_environment(args.config, output, overwrite=args.overwrite)
        return 0

    if args.command == "spin-sequential-cell-gate":
        from projects.clebsch_tree_nqs.workflows.spin_sequential_cell_gate import (
            run_spin_sequential_cell_gate,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_sequential_cell_gate(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-schur-spatial-scaling-preflight":
        from projects.clebsch_tree_nqs.workflows.spin_schur_spatial_scaling_preflight import (
            run_spin_schur_spatial_scaling_preflight,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_schur_spatial_scaling_preflight(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-scale-first-three-arm-preflight":
        from projects.clebsch_tree_nqs.workflows.spin_scale_first_three_arm_preflight import (
            run_spin_scale_first_three_arm_preflight,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_scale_first_three_arm_preflight(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-scale-first-accelerator-slice":
        from projects.clebsch_tree_nqs.workflows.spin_scale_first_accelerator_slices import (
            run_spin_scale_first_accelerator_slice,
        )

        run_spin_scale_first_accelerator_slice(
            args.config,
            args.output,
            case_id=args.case_id,
            arm_id=args.arm_id,
            batch_size=args.batch_size,
            amplitude_tile_size=args.amplitude_tile_size,
            device_microbatch_size=args.device_microbatch_size,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-scale-first-accelerator-merge":
        from projects.clebsch_tree_nqs.workflows.spin_scale_first_accelerator_slices import (
            run_spin_scale_first_accelerator_merge,
        )

        run_spin_scale_first_accelerator_merge(
            args.config,
            args.input_root,
            args.output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-scale-first-training-calibration":
        from projects.clebsch_tree_nqs.workflows.spin_scale_first_training_calibration import (
            run_spin_scale_first_training_calibration,
        )

        run_spin_scale_first_training_calibration(
            args.config,
            args.output,
            case_id=args.case_id,
            arm_id=args.arm_id,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-scale-first-initialization-audit":
        from projects.clebsch_tree_nqs.workflows.spin_scale_first_initialization_audit import (
            run_spin_scale_first_initialization_audit,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_scale_first_initialization_audit(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-scale-first-initialization-pilot":
        from projects.clebsch_tree_nqs.workflows.spin_scale_first_initialization_pilot import (
            run_spin_scale_first_initialization_pilot,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_pilot_v1"
        run_spin_scale_first_initialization_pilot(
            args.config,
            output,
            case_id=args.case_id,
            arm_id=args.arm_id,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-scale-first-convergence-merge":
        from projects.clebsch_tree_nqs.workflows.spin_scale_first_convergence_merge import (
            run_spin_scale_first_convergence_merge,
        )

        run_spin_scale_first_convergence_merge(
            args.config,
            args.input_root,
            args.output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-scale-first-checkpoint-audit":
        from projects.clebsch_tree_nqs.workflows.spin_scale_first_checkpoint_audit import (
            run_spin_scale_first_checkpoint_audit,
        )

        run_spin_scale_first_checkpoint_audit(
            args.config,
            args.checkpoint_root,
            args.output,
            case_id=args.case_id,
            arm_id=args.arm_id,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-scale-first-checkpoint-audit-merge":
        from projects.clebsch_tree_nqs.workflows.spin_scale_first_checkpoint_audit import (
            run_spin_scale_first_checkpoint_audit_merge,
        )

        run_spin_scale_first_checkpoint_audit_merge(
            args.config,
            args.input_root,
            args.output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-scale-first-plaquette-carrier-gate":
        from projects.clebsch_tree_nqs.workflows.spin_scale_first_plaquette_carrier_gate import (
            run_spin_scale_first_plaquette_carrier_gate,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_scale_first_plaquette_carrier_gate(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-coherent-plaquette-frame-gate":
        from projects.clebsch_tree_nqs.workflows.spin_coherent_plaquette_frame_gate import (
            run_spin_coherent_plaquette_frame_gate,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_coherent_plaquette_frame_gate(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-n36-coherent-frame-support":
        from projects.clebsch_tree_nqs.workflows.spin_n36_coherent_frame_support import (
            run_spin_n36_coherent_frame_support,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_n36_coherent_frame_support(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-n36-coherent-frame-canary":
        from projects.clebsch_tree_nqs.workflows.spin_n36_coherent_frame_canary import (
            run_spin_n36_coherent_frame_canary,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_n36_coherent_frame_canary(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-n36-multibody-frame-resonance":
        from projects.clebsch_tree_nqs.workflows.spin_n36_multibody_frame_resonance import (
            run_spin_n36_multibody_frame_resonance,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_n36_multibody_frame_resonance(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-n16-imaginary-time-compression":
        from projects.clebsch_tree_nqs.workflows.spin_n16_imaginary_time_compression import (
            run_spin_n16_imaginary_time_compression,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_n16_imaginary_time_compression(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-n16-imaginary-time-tangent":
        from projects.clebsch_tree_nqs.workflows.spin_n16_imaginary_time_tangent import (
            run_spin_n16_imaginary_time_tangent,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_n16_imaginary_time_tangent(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-n16-hamiltonian-filtration":
        from projects.clebsch_tree_nqs.workflows.spin_n16_hamiltonian_filtration import (
            run_spin_n16_hamiltonian_filtration,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_n16_hamiltonian_filtration(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-n16-bond-causal-tangent":
        from projects.clebsch_tree_nqs.workflows.spin_n16_bond_causal_tangent import (
            run_spin_n16_bond_causal_tangent,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_n16_bond_causal_tangent(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-n16-bond-causal-multistep":
        from projects.clebsch_tree_nqs.workflows.spin_n16_bond_causal_multistep import (
            run_spin_n16_bond_causal_multistep,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_n16_bond_causal_multistep(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-n16-causal-operator-oracle":
        from projects.clebsch_tree_nqs.workflows.spin_n16_causal_operator_oracle import (
            run_spin_n16_causal_operator_oracle,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_n16_causal_operator_oracle(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-n16-krylov-causal-oracle":
        from projects.clebsch_tree_nqs.workflows.spin_n16_krylov_causal_oracle import (
            run_spin_n16_krylov_causal_oracle,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_n16_krylov_causal_oracle(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-n16-causal-krylov-hybrid":
        from projects.clebsch_tree_nqs.workflows.spin_n16_causal_krylov_hybrid import (
            run_spin_n16_causal_krylov_hybrid,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_n16_causal_krylov_hybrid(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-scale-first-carrier-distillation":
        from projects.clebsch_tree_nqs.workflows.spin_scale_first_carrier_distillation import (
            run_spin_scale_first_carrier_distillation,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_scale_first_carrier_distillation(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-scale-first-carrier-transfer-preflight":
        from projects.clebsch_tree_nqs.workflows.spin_scale_first_carrier_transfer_preflight import (
            run_spin_scale_first_carrier_transfer_preflight,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_scale_first_carrier_transfer_preflight(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-scale-first-n36-carrier-distillation":
        from projects.clebsch_tree_nqs.workflows.spin_scale_first_n36_carrier_distillation import (
            run_spin_scale_first_n36_carrier_distillation,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_scale_first_n36_carrier_distillation(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-scale-first-carrier-single-seed":
        from projects.clebsch_tree_nqs.workflows.spin_scale_first_carrier_single_seed import (
            run_spin_scale_first_carrier_single_seed,
        )

        run_spin_scale_first_carrier_single_seed(
            args.config,
            args.output,
            case_id=args.case_id,
            arm_id=args.arm_id,
            maximum_steps_this_call=args.maximum_steps_this_call,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-scale-first-carrier-checkpoint-audit":
        from projects.clebsch_tree_nqs.workflows.spin_scale_first_carrier_checkpoint_audit import (
            run_spin_scale_first_carrier_checkpoint_audit,
        )

        run_spin_scale_first_carrier_checkpoint_audit(
            args.config,
            args.checkpoint_root,
            args.output,
            case_id=args.case_id,
            arm_id=args.arm_id,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-scale-first-carrier-checkpoint-audit-merge":
        from projects.clebsch_tree_nqs.workflows.spin_scale_first_carrier_checkpoint_audit import (
            run_spin_scale_first_carrier_checkpoint_audit_merge,
        )

        run_spin_scale_first_carrier_checkpoint_audit_merge(
            args.config,
            args.input_root,
            args.output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-scale-first-carrier-convergence-merge":
        from projects.clebsch_tree_nqs.workflows.spin_scale_first_carrier_convergence_merge import (
            run_spin_scale_first_carrier_convergence_merge,
        )

        run_spin_scale_first_carrier_convergence_merge(
            args.config,
            args.input_root,
            args.output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-scale-first-carrier-schur-channel-audit":
        from projects.clebsch_tree_nqs.workflows.spin_scale_first_carrier_schur_channel_audit import (
            run_spin_scale_first_carrier_schur_channel_audit,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_scale_first_carrier_schur_channel_audit(
            args.config,
            args.checkpoint_root,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-carrier-cell-scalar-circuit":
        from projects.clebsch_tree_nqs.workflows.spin_carrier_cell_scalar_circuit import (
            run_spin_carrier_cell_scalar_circuit,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_carrier_cell_scalar_circuit(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-entanglement-channel-audit":
        from projects.clebsch_tree_nqs.workflows.spin_entanglement_channel_audit import (
            run_spin_entanglement_channel_audit,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_entanglement_channel_audit(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-entanglement-filter-gate":
        from projects.clebsch_tree_nqs.workflows.spin_entanglement_filter_gate import (
            run_spin_entanglement_filter_gate,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_entanglement_filter_gate(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-entanglement-filter-scalable-bridge":
        from projects.clebsch_tree_nqs.workflows.spin_entanglement_filter_scalable_bridge import (
            run_spin_entanglement_filter_scalable_bridge,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_entanglement_filter_scalable_bridge(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-projected-carrier-multiplicity-dressing":
        from projects.clebsch_tree_nqs.workflows.spin_projected_carrier_multiplicity_dressing import (
            run_spin_projected_carrier_multiplicity_dressing,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_projected_carrier_multiplicity_dressing(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-n16-krylov-anchored-full-vit":
        from projects.clebsch_tree_nqs.workflows.spin_n16_krylov_anchored_full_vit import (
            run_spin_n16_krylov_anchored_full_vit,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_n16_krylov_anchored_full_vit(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-n16-full-vit-sr-reachability":
        from projects.clebsch_tree_nqs.workflows.spin_n16_full_vit_sr_reachability import (
            run_spin_n16_full_vit_sr_reachability,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_n16_full_vit_sr_reachability(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-n16-full-vit-sr-continuation":
        from projects.clebsch_tree_nqs.workflows.spin_n16_full_vit_sr_continuation import (
            run_spin_n16_full_vit_sr_continuation,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_n16_full_vit_sr_continuation(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-n16-spectral-error-tomography":
        from projects.clebsch_tree_nqs.workflows.spin_n16_spectral_error_tomography import (
            run_spin_n16_spectral_error_tomography,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_n16_spectral_error_tomography(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-n36-projected-multiplicity-preflight":
        from projects.clebsch_tree_nqs.workflows.spin_n36_projected_multiplicity_preflight import (
            run_spin_n36_projected_multiplicity_preflight,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_n36_projected_multiplicity_preflight(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-schur-sector-matched-frontier":
        from projects.clebsch_tree_nqs.workflows.spin_schur_sector_matched_frontier import (
            run_spin_schur_sector_matched_frontier,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_schur_sector_matched_frontier(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-four-cell-additive-carrier":
        from projects.clebsch_tree_nqs.workflows.spin_four_cell_additive_carrier import (
            run_spin_four_cell_additive_carrier,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_four_cell_additive_carrier(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-four-cell-additive-carrier-confirmation":
        from projects.clebsch_tree_nqs.workflows.spin_four_cell_additive_carrier_confirmation import (
            run_spin_four_cell_additive_carrier_confirmation,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_four_cell_additive_carrier_confirmation(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-four-cell-paired-z2-frontier":
        from projects.clebsch_tree_nqs.workflows.spin_four_cell_paired_z2_frontier import (
            run_spin_four_cell_paired_z2_frontier,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_four_cell_paired_z2_frontier(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-four-cell-scalar-correlator-frontier":
        from projects.clebsch_tree_nqs.workflows.spin_four_cell_scalar_correlator_frontier import (
            run_spin_four_cell_scalar_correlator_frontier,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_four_cell_scalar_correlator_frontier(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-four-cell-operator-scalar-frontier":
        from projects.clebsch_tree_nqs.workflows.spin_four_cell_operator_scalar_frontier import (
            run_spin_four_cell_operator_scalar_frontier,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_four_cell_operator_scalar_frontier(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-four-cell-staged-operator-scalar":
        from projects.clebsch_tree_nqs.workflows.spin_four_cell_staged_operator_scalar import (
            run_spin_four_cell_staged_operator_scalar,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_four_cell_staged_operator_scalar(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-four-cell-overlapping-recoupling":
        from projects.clebsch_tree_nqs.workflows.spin_four_cell_overlapping_recoupling import (
            run_spin_four_cell_overlapping_recoupling,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_four_cell_overlapping_recoupling(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-four-cell-recoupling-ritz":
        from projects.clebsch_tree_nqs.workflows.spin_four_cell_recoupling_ritz import (
            run_spin_four_cell_recoupling_ritz,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_four_cell_recoupling_ritz(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-four-cell-recoupling-variable-projection":
        from projects.clebsch_tree_nqs.workflows.spin_four_cell_recoupling_variable_projection import (
            run_spin_four_cell_recoupling_variable_projection,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_four_cell_recoupling_variable_projection(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-four-cell-recoupling-variable-projection-audit":
        from projects.clebsch_tree_nqs.workflows.spin_four_cell_recoupling_variable_projection_audit import (
            run_spin_four_cell_recoupling_variable_projection_audit,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_four_cell_recoupling_variable_projection_audit(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-symmetry-response-operator":
        from projects.clebsch_tree_nqs.workflows.spin_symmetry_response_operator import (
            run_spin_symmetry_response_operator,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_symmetry_response_operator(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-square-b2-response-audit":
        from projects.clebsch_tree_nqs.workflows.spin_square_b2_response_audit import (
            run_spin_square_b2_response_audit,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_square_b2_response_audit(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-conditioned-response-gate":
        from projects.clebsch_tree_nqs.workflows.spin_conditioned_response_gate import (
            run_spin_conditioned_response_gate,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_conditioned_response_gate(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-symmetry-labelled-loop-screen":
        from projects.clebsch_tree_nqs.workflows.spin_symmetry_labelled_loop_screen import (
            run_spin_symmetry_labelled_loop_screen,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_symmetry_labelled_loop_screen(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-symmetry-loop-coupling-scan":
        from projects.clebsch_tree_nqs.workflows.spin_symmetry_loop_coupling_scan import (
            run_spin_symmetry_loop_coupling_scan,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_symmetry_loop_coupling_scan(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "su2-route-vit-causal-scan":
        from projects.clebsch_tree_nqs.workflows.spin_su2_route_vit_causal_scan import (
            run_spin_su2_route_vit_causal_scan,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_su2_route_vit_causal_scan(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "su2-route-vit-physical-routes":
        from projects.clebsch_tree_nqs.workflows.spin_su2_route_vit_physical_routes import (
            run_spin_su2_route_vit_physical_routes,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_su2_route_vit_physical_routes(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-physics-interpretation":
        from projects.clebsch_tree_nqs.workflows.spin_physics_interpretation import (
            run_spin_physics_interpretation,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_physics_interpretation(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-e2-resonance-audit":
        from projects.clebsch_tree_nqs.workflows.spin_e2_resonance_audit import (
            run_spin_e2_resonance_audit,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_e2_resonance_audit(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-symmetry-carrier-four-arm":
        from projects.clebsch_tree_nqs.workflows.spin_symmetry_carrier_four_arm import (
            run_spin_symmetry_carrier_four_arm,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_symmetry_carrier_four_arm(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-adaptive-pair-carrier":
        from projects.clebsch_tree_nqs.workflows.spin_adaptive_pair_carrier import (
            run_spin_adaptive_projected_pair_carrier,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_adaptive_projected_pair_carrier(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "strict-schur-spatial-gate":
        from projects.clebsch_tree_nqs.workflows.spin_strict_schur_spatial import (
            run_strict_schur_spatial_gate,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_strict_schur_spatial_gate(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "strict-schur-spatial-n16-smoke":
        from projects.clebsch_tree_nqs.workflows.spin_strict_schur_spatial_n16 import (
            run_strict_schur_spatial_n16_smoke,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_strict_schur_spatial_n16_smoke(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "triangle-patch-n36-runtime-canary":
        from projects.clebsch_tree_nqs.workflows.spin_triangle_patch_n36_runtime import (
            run_triangle_patch_n36_runtime_canary,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_triangle_patch_n36_runtime_canary(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "triangle-patch-n36-energy-smoke":
        from projects.clebsch_tree_nqs.workflows.spin_triangle_patch_n36_energy_smoke import (
            run_triangle_patch_n36_energy_smoke,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_triangle_patch_n36_energy_smoke(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "triangle-patch-n36-physical-init-readout":
        from projects.clebsch_tree_nqs.workflows.spin_triangle_patch_n36_physical_init_readout import (
            run_triangle_patch_n36_physical_init_readout,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_triangle_patch_n36_physical_init_readout(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "triangle-patch-n36-sr-pilot":
        from projects.clebsch_tree_nqs.workflows.spin_triangle_patch_n36_sr_pilot import (
            run_triangle_patch_n36_sr_pilot,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_triangle_patch_n36_sr_pilot(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "triangle-patch-n36-low-spin-resonance":
        from projects.clebsch_tree_nqs.workflows.spin_triangle_patch_n36_low_spin_resonance import (
            run_triangle_patch_n36_low_spin_resonance,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_triangle_patch_n36_low_spin_resonance(
            args.config,
            output,
            overwrite=args.overwrite,
            case_id=args.case_id,
            smoke=args.smoke,
            local_pilot=args.local_pilot,
        )
        return 0

    if args.command == "observable-validation":
        from projects.clebsch_tree_nqs.workflows.spin_observable_validation import (
            run_spin_observable_validation,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_observable_validation(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "sampled-runtime":
        from projects.clebsch_tree_nqs.workflows.spin_sampled_runtime import (
            run_spin_sampled_runtime,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v2"
        run_spin_sampled_runtime(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "spin-g1-operator-vit":
        from projects.clebsch_tree_nqs.workflows.spin_g1_operator_vit import (
            materialize_g1_operator_vit_initialization,
            run_spin_g1_operator_vit,
        )

        if args.materialize_initialization:
            manifest = materialize_g1_operator_vit_initialization(
                args.config, overwrite=args.overwrite
            )
            print(manifest["pytree_sha256"])
            return 0
        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_g1_operator_vit(
            args.config,
            output,
            overwrite=args.overwrite,
            maximum_updates_this_invocation=args.maximum_updates_this_invocation,
        )
        return 0

    if args.command == "spin-g1-regularized-operator":
        from projects.clebsch_tree_nqs.workflows.spin_g1_regularized_operator import (
            run_spin_g1_regularized_operator,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_g1_regularized_operator(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "sampled-observable-probe":
        from projects.clebsch_tree_nqs.workflows.spin_sampled_observable_probe import (
            run_spin_sampled_observable_probe,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_sampled_observable_probe(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "sampled-training":
        from projects.clebsch_tree_nqs.workflows.spin_sampled_training import (
            run_spin_sampled_training,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_sampled_training(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "sampled-optimizer-scan-v2":
        from projects.clebsch_tree_nqs.workflows.spin_sampled_training_v2 import (
            run_spin_sampled_training_v2,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v2"
        run_spin_sampled_training_v2(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "sampled-optimizer-select-v2":
        from projects.clebsch_tree_nqs.workflows.spin_sampled_training_v2_selector import (
            run_spin_sampled_training_v2_selector,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_sampled_training_v2_selector(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "sampled-optimizer-confirmatory-extension-v1":
        from projects.clebsch_tree_nqs.workflows.spin_sampled_training_v2_extension import (
            preflight_spin_sampled_training_v2_extension,
            run_spin_sampled_training_v2_extension,
        )
        from su2_nqs.frontier.artifacts import atomic_json

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        if args.preflight_only:
            output.mkdir(parents=True, exist_ok=True)
            destination = output / "preflight.json"
            if destination.exists() and not args.overwrite:
                raise FileExistsError(
                    f"preflight already exists; pass --overwrite: {destination}"
                )
            atomic_json(
                destination,
                preflight_spin_sampled_training_v2_extension(args.config),
            )
        else:
            run_spin_sampled_training_v2_extension(
                args.config,
                output,
                overwrite=args.overwrite,
            )
        return 0

    if args.command == "sampled-optimizer-select-v2-metric-correction":
        from projects.clebsch_tree_nqs.workflows.spin_sampled_training_v2_metric_correction_selector import (
            run_spin_sampled_training_v2_metric_correction_selector,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v2"
        run_spin_sampled_training_v2_metric_correction_selector(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "sampled-training-post-analysis":
        from projects.clebsch_tree_nqs.workflows.spin_sampled_training_post_analysis import (
            run_spin_sampled_training_post_analysis,
        )

        output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
        run_spin_sampled_training_post_analysis(
            args.config,
            output,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "analyze-prl":
        from projects.clebsch_tree_nqs.analysis import analyze_prl_results
        from su2_nqs.frontier.artifacts import atomic_json

        summary = analyze_prl_results(
            args.base,
            args.long,
            args.su2_long,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
            time_grid_size=args.time_grid_size,
        )
        atomic_json(args.output, summary)
        print(args.output)
        return 0

    from projects.clebsch_tree_nqs.workflows.common import run_exact_campaign

    output = args.output or PROJECT / "results" / f"{args.config.stem}_v1"
    run_exact_campaign(
        args.config,
        output,
        workflow=_workflow(args.domain),
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
