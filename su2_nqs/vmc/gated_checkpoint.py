"""Checkpoint pre-update gates without inventing a completed training step.

A zero-step endpoint has EMPTY training history. Put its control decision in
metadata; the generic checkpoint loader correctly insists history length equals
completed_steps. This helper does not alter any numeric state or old artifact.
"""
from __future__ import annotations

from pathlib import Path

from .resumable_checkpoint import save_resumable_checkpoint


def save_gated_checkpoint(path: str | Path, params: object, *, optimizer_state: object,
                          run_identity: dict, decision: dict,
                          recovery_source_sha256: str | None = None) -> None:
    """Save one bounded gate attempt, counting only actual candidate construction.

    A denied proposal is not a completed optimizer step. If a raw candidate
    was constructed, record one attempted step whether committed or rejected.
    Every decision is retained in metadata, including a denied zero-step gate.
    next_sample_index=0: this wrapper is for cache-only bounded attempts.
    """
    attempted=decision.get('raw_candidate_constructed')
    if not isinstance(attempted,bool) or not isinstance(decision.get('accepted'),bool):
        raise ValueError('explicit boolean candidate/acceptance decisions required')
    if decision['accepted'] and not attempted:
        raise ValueError('cannot accept an unconstructed candidate')
    metadata=dict(pre_update_decision=decision)
    if recovery_source_sha256 is not None:
        if (not isinstance(recovery_source_sha256,str) or len(recovery_source_sha256)!=64
                or any(x not in '0123456789abcdef' for x in recovery_source_sha256)):
            raise ValueError('recovery source must have a SHA256 digest')
        metadata['recovery_source_sha256']=recovery_source_sha256
    save_resumable_checkpoint(path,params,optimizer_state=optimizer_state,
        completed_steps=int(attempted),next_sample_index=0,run_identity=run_identity,
        history=[decision] if attempted else [],metadata=metadata)


__all__=['save_gated_checkpoint']
