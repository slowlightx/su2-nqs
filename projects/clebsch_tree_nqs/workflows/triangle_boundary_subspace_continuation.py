"""One fixed same-q Procrustes/polar continuation; no training or A1 tuning."""
from __future__ import annotations

import argparse
from pathlib import Path
import signal
from time import perf_counter

import numpy as np

from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.frontier.retained_boundary_response import interpolate_fixed_q_frame, response_capture
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from .triangle_boundary_response_enrichment import frozen_responses, project_candidate
from .triangle_hierarchy_initialization import native_bank_from_state
from .triangle_sector_transfer import load_source


def run(output: Path, old_state: Path, proposed_frame: Path) -> dict:
    """Freeze t=1/16 before candidate evaluation; export one J2=.125 state.

    The proposed frame is the already frozen adaptive-density candidate at
    node8/q2, in exactly the old tau8 child basis. Only this subspace changes.
    No Hamiltonian eigenproblem is solved. The projected-block H is queried
    only to evaluate this fixed candidate, never to choose its step.
    """
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("choose a fresh folder; previous evidence is immutable")
    helper = Path(__file__).parents[3]/"su2_nqs/frontier/retained_boundary_response.py"
    registered = dict(task="fixed_same_width_Procrustes_polar_continuation_v1",
        j2=.125, changed_node=8, changed_doubled_q=2, retained_width=8,
        step_numerator=1, step_denominator=16, step_selection="fixed BEFORE candidate A1 evaluation; no step optimization",
        old_state=str(old_state.resolve()), old_state_sha256=file_sha256(old_state),
        proposed_frame=str(proposed_frame.resolve()), proposed_frame_sha256=file_sha256(proposed_frame),
        workflow_sha256=file_sha256(Path(__file__)), helper_sha256=file_sha256(helper),
        wall_seconds=90, no_eigensolve=True, no_training=True, no_remote_jobs=True,
        alignment="unitary Procrustes in SAME physical child-space rows; polar retraction",
        ancestor_handling="normalized projection of old raw state; overlap and QR gauges absorbed",
        full_A1_quality="unknown; parent separately reuses fixed common-Q rows, development only",
        serial_mitigation_after_raw_fidelity_failed_to_protect_normalized_A1_overlap=True)
    atomic_json(output/"registration.json", registered)
    previous = signal.getsignal(signal.SIGALRM)
    def timeout(*_):
        raise TimeoutError("fixed90s one-candidate continuation cap")
    signal.signal(signal.SIGALRM, timeout); signal.setitimer(signal.ITIMER_REAL, 90)
    started = perf_counter()
    try:
        source, provenance = load_source(.125)
        state = load_checkpoint(old_state)["params"]
        proposed = np.load(proposed_frame)["q2_frame"]
        old = np.asarray(state["maps"][8][2])
        frame, geometry = interpolate_fixed_q_frame(2, old, proposed, step=1/16)
        data, weighting = frozen_responses(state, source)
        candidate, quality = project_candidate(state, frame, .125, weighting["down_q2"])
        native, embedding = native_bank_from_state(source, candidate)
        readouts = {label: {part: response_capture(q, columns) for part, columns in data.items()}
                    for label, q in (("old", old), ("proposed_frozen", proposed), ("fixed_step", frame))}
        old_energy = float(state["blocks"][-1].h_by_q[0][0, 0].real)/144
        metadata = dict(model=dict(length=12, multiplicity_cap=20, maximum_q=10),
            j2=.125, arm="fixed_Procrustes_polar_t1_16", embedding=embedding,
            optimizer_reset_required=True, no_training_updates=True,
            effective_folded_bank_not_rank8_binary=True, registration=registered, **quality)
        save_checkpoint(output/"state.pkl", candidate, optimizer_state=None, metadata=metadata)
        save_checkpoint(output/"initialized_bank.pkl", native, optimizer_state=None, metadata=metadata)
        np.savez_compressed(output/"thin_frame.npz", q2_frame=frame,
                            old_overlap=frame.conj().T @ old)
        result = dict(registration=registered, interpolation=geometry, quality=quality,
            response=readouts, source=provenance, parent_raw_energy_per_site=old_energy,
            raw_energy_change_per_site=quality["raw_energy_per_site"]-old_energy,
            native_embedding=embedding, full_A1_fidelity_known=False, full_A1_energy_known=False,
            no_outcome_dependent_tuning=True, seconds=perf_counter()-started,
            checkpoints_sha256={name: file_sha256(output/name) for name in ("state.pkl", "initialized_bank.pkl", "thin_frame.npz")})
        atomic_json(output/"results.json", result)
        print(dict(seconds=result["seconds"], raw_fidelity=quality["normalized_RAW_fidelity_to_parent"],
                   raw_energy=quality["raw_energy_per_site"], raw_delta=result["raw_energy_change_per_site"]), flush=True)
        return result
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-state", type=Path, required=True)
    parser.add_argument("--proposed-frame", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.old_state, args.proposed_frame)
