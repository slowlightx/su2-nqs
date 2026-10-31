"""Fresh-process worker for the three-root resume trajectory test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

from su2_nqs.models.compact_tree_gru import CompactTreeGRUConfig, init_compact_tree_gru
from su2_nqs.su2.fusion_tree import FusionTree
from su2_nqs.vmc.resumable_tree_adam import pytree_sha256
from su2_nqs.vmc.resumable_tree_multistate_adam import (
    run_resumable_three_root_tree_adam,
)


jax.config.update("jax_enable_x64", True)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--maximum", type=int, default=None)
    parser.add_argument("--samples-per-root", type=int, default=4)
    parser.add_argument(
        "--overlap-diagonal-policy",
        choices=("legacy", "exact_normalized"),
        default="legacy",
    )
    return parser.parse_args()


def _deterministic_history(history) -> list[dict[str, object]]:
    timing = {
        "sampling_seconds",
        "connectivity_seconds",
        "action_seconds",
        "force_seconds",
    }
    return [
        {key: value for key, value in row.items() if key not in timing}
        for row in history
    ]


def main() -> None:
    args = _arguments()
    config = CompactTreeGRUConfig(
        FusionTree.balanced(6), 0, embedding_dim=2, hidden_dim=4
    )
    params = tuple(
        init_compact_tree_gru(key, config)
        for key in jax.random.split(jax.random.PRNGKey(92701), 3)
    )
    bonds = tuple((site, (site + 1) % 6, 1.0) for site in range(6)) + (
        (0, 3, 0.17),
        (1, 4, -0.11),
    )
    result = run_resumable_three_root_tree_adam(
        params,
        config=config,
        bonds=bonds,
        total_steps=2,
        samples_per_root=args.samples_per_root,
        learning_rate=2.0e-3,
        sample_key=jax.random.PRNGKey(92702),
        run_identity={
            "test": "fresh_process_three_root_resume",
            "parameter_master_key": 92701,
            "parameter_split": "jax.random.split_3",
        },
        checkpoint_path=args.checkpoint,
        resume=args.resume,
        checkpoint_every=1,
        max_steps_this_invocation=args.maximum,
        registered_widths=(4, 8, 16, 32, 64),
        sampling_microbatch_size=3,
        action_microbatch_size=3,
        maximum_condition_number=1.0e8,
        # The exact-normalized choice is retained only as a rejection probe:
        # force-capable trajectories fail closed on that Ritz-only policy.
        exact_normalized_overlap_diagonal=(
            args.overlap_diagonal_policy == "exact_normalized"
        ),
        devices=(jax.local_devices()[0],),
    )
    print(
        json.dumps(
            {
                "params_sha256": pytree_sha256(result.params),
                "root_params_sha256": [
                    pytree_sha256(root) for root in result.params
                ],
                "optimizer_states_sha256": pytree_sha256(
                    result.optimizer_states
                ),
                "root_optimizer_state_sha256": [
                    pytree_sha256(state) for state in result.optimizer_states
                ],
                "completed_steps": result.completed_steps,
                "next_root_sample_indices": list(
                    result.next_root_sample_indices
                ),
                "next_sample_index": result.next_sample_index,
                "history": _deterministic_history(result.history),
                "resumed": result.resumed,
                "stopped_early": result.stopped_early,
                "recovered_checkpoint_failures": list(
                    result.recovered_checkpoint_failures
                ),
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
