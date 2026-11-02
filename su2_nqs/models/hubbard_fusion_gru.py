"""Normalized autoregressive model in fixed-charge Hubbard fusion space."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from ..su2.hubbard_fusion import count_hubbard_fusion_states
from ._core import embed, gru_step, heads, init_gru_core


# empty, single/down-fusion, single/up-fusion, doublon
ACTION_OCCUPATIONS = jnp.asarray([0, 1, 1, 2], dtype=jnp.int32)
ACTION_DELTA_Q = jnp.asarray([0, -1, 1, 0], dtype=jnp.int32)
BOS_TOKEN = 4


@lru_cache(maxsize=None)
def hubbard_reachability_table(
    n_sites: int, n_electrons: int, target_q: int
) -> np.ndarray:
    """Exact suffix reachability indexed by ``remaining, charge_left, q``."""

    if count_hubbard_fusion_states(n_sites, n_electrons, target_q) == 0:
        raise ValueError("requested Hubbard charge/spin sector is empty")
    table = np.zeros(
        (n_sites + 1, 2 * n_sites + 1, n_sites + 1), dtype=bool
    )
    table[0, 0, target_q] = True
    occupations = np.asarray(ACTION_OCCUPATIONS)
    deltas = np.asarray(ACTION_DELTA_Q)
    for remaining in range(1, n_sites + 1):
        for charge_left in range(2 * remaining + 1):
            for current_q in range(n_sites + 1):
                for occupation, delta_q in zip(occupations, deltas):
                    next_q = current_q + int(delta_q)
                    if occupation == 1:
                        locally_valid = next_q >= 0
                    else:
                        locally_valid = next_q == current_q
                    next_charge_left = charge_left - int(occupation)
                    if (
                        locally_valid
                        and 0 <= next_q <= n_sites
                        and 0 <= next_charge_left <= 2 * (remaining - 1)
                        and table[
                            remaining - 1, next_charge_left, next_q
                        ]
                    ):
                        table[remaining, charge_left, current_q] = True
                        break
    if not table[n_sites, n_electrons, 0]:
        raise RuntimeError("reachability table disagrees with exact enumeration")
    return table


@dataclass(frozen=True)
class HubbardFusionGRUConfig:
    n_sites: int
    n_electrons: int
    target_q: int = 0
    embedding_dim: int = 8
    hidden_dim: int = 32

    def __post_init__(self) -> None:
        if count_hubbard_fusion_states(
            self.n_sites, self.n_electrons, self.target_q
        ) == 0:
            raise ValueError("requested Hubbard charge/spin sector is empty")
        if self.embedding_dim < 1 or self.hidden_dim < 1:
            raise ValueError("embedding_dim and hidden_dim must be positive")


def init_hubbard_fusion_gru(
    key: jax.Array, config: HubbardFusionGRUConfig
) -> dict[str, object]:
    """Initialize a four-action charge/spin fusion GRU."""

    return init_gru_core(
        key,
        embedding_sizes=(
            5,
            config.n_sites + 1,
            2 * config.n_sites + 1,
            config.n_sites,
            config.n_sites + 1,
        ),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        output_dim=4,
    )


def transfer_hubbard_fusion_gru_parameters(
    source_params: dict[str, object],
    source_config: HubbardFusionGRUConfig,
    target_config: HubbardFusionGRUConfig,
    key: jax.Array,
) -> dict[str, object]:
    """Transfer a learned recurrent core between Hubbard system sizes.

    Action, recurrent, probability, and phase parameters are size independent
    and are copied exactly. Rows with the same integer label are copied for
    cumulative-spin, charge, site, and remaining-site embeddings; only rows
    absent from the source retain their seeded target initialization.
    """

    if (
        source_config.embedding_dim != target_config.embedding_dim
        or source_config.hidden_dim != target_config.hidden_dim
    ):
        raise ValueError("source and target model widths must agree")
    target = init_hubbard_fusion_gru(key, target_config)
    transferred = {
        "h0": jnp.array(source_params["h0"]),
        "gru": jax.tree_util.tree_map(jnp.array, source_params["gru"]),
        "probability": jax.tree_util.tree_map(
            jnp.array, source_params["probability"]
        ),
        "phase": jax.tree_util.tree_map(jnp.array, source_params["phase"]),
    }
    embeddings = []
    for source_embedding, target_embedding in zip(
        source_params["embeddings"], target["embeddings"], strict=True
    ):
        shared_rows = min(source_embedding.shape[0], target_embedding.shape[0])
        embeddings.append(
            target_embedding.at[:shared_rows].set(source_embedding[:shared_rows])
        )
    transferred["embeddings"] = tuple(embeddings)
    return transferred


def _action_mask(
    current_q: jax.Array,
    charge: jax.Array,
    remaining_after: int,
    config: HubbardFusionGRUConfig,
    table: jax.Array,
) -> jax.Array:
    next_q = current_q[..., None] + ACTION_DELTA_Q
    next_charge = charge[..., None] + ACTION_OCCUPATIONS
    charge_left = config.n_electrons - next_charge
    q_valid = (next_q >= 0) & (next_q <= config.n_sites)
    charge_valid = (charge_left >= 0) & (charge_left <= 2 * remaining_after)
    safe_q = jnp.clip(next_q, 0, config.n_sites)
    safe_charge = jnp.clip(charge_left, 0, 2 * config.n_sites)
    reachable = table[remaining_after, safe_charge, safe_q]
    return q_valid & charge_valid & reachable


def hubbard_log_psi(
    params: dict[str, object],
    labels: jax.Array,
    config: HubbardFusionGRUConfig,
) -> jax.Array:
    """Return ``0.5*log(p)+i*phase`` for interleaved charge/spin labels."""

    labels = jnp.asarray(labels, dtype=jnp.int32)
    if labels.shape != (2 * config.n_sites,):
        raise ValueError(
            f"expected label shape ({2 * config.n_sites},), got {labels.shape}"
        )
    table = jnp.asarray(
        hubbard_reachability_table(
            config.n_sites, config.n_electrons, config.target_q
        )
    )
    hidden = params["h0"]
    previous = jnp.asarray(BOS_TOKEN, dtype=jnp.int32)
    current_q = jnp.asarray(0, dtype=jnp.int32)
    charge = jnp.asarray(0, dtype=jnp.int32)
    log_probability = jnp.asarray(0.0, dtype=jnp.float64)
    phase = jnp.asarray(0.0, dtype=jnp.float64)
    valid_state = jnp.asarray(True)
    for site in range(config.n_sites):
        remaining_after = config.n_sites - site - 1
        safe_q = jnp.clip(current_q, 0, config.n_sites)
        safe_charge = jnp.clip(charge, 0, 2 * config.n_sites)
        inputs = embed(
            params, (previous, safe_q, safe_charge, site, remaining_after)
        )
        hidden = gru_step(params["gru"], hidden, inputs)
        logits, phases = heads(params, hidden)
        mask = _action_mask(
            current_q, charge, remaining_after, config, table
        )
        desired_occupation = labels[2 * site]
        desired_q = labels[2 * site + 1]
        matches = (
            (ACTION_OCCUPATIONS == desired_occupation)
            & (current_q + ACTION_DELTA_Q == desired_q)
            & mask
        )
        action = jnp.argmax(matches).astype(jnp.int32)
        valid_state = valid_state & jnp.any(matches)
        masked_logits = jnp.where(mask, logits, -1.0e30)
        log_probability += jax.nn.log_softmax(masked_logits)[action]
        phase += phases[action]
        charge = charge + desired_occupation
        current_q = desired_q
        previous = action
    valid_state = (
        valid_state
        & (charge == config.n_electrons)
        & (current_q == config.target_q)
    )
    value = 0.5 * log_probability + 1j * phase
    return jnp.where(valid_state, value, -jnp.inf + 0j)


def sample_hubbard_fusion(
    params: dict[str, object],
    key: jax.Array,
    n_samples: int,
    config: HubbardFusionGRUConfig,
) -> jax.Array:
    """Draw exact independent samples in the requested charge/spin sector."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    table = jnp.asarray(
        hubbard_reachability_table(
            config.n_sites, config.n_electrons, config.target_q
        )
    )
    hidden = jnp.broadcast_to(params["h0"], (n_samples, config.hidden_dim))
    previous = jnp.full((n_samples,), BOS_TOKEN, dtype=jnp.int32)
    current_q = jnp.zeros((n_samples,), dtype=jnp.int32)
    charge = jnp.zeros((n_samples,), dtype=jnp.int32)
    labels = []
    for site in range(config.n_sites):
        key, subkey = jax.random.split(key)
        remaining_after = config.n_sites - site - 1
        inputs = embed(
            params,
            (
                previous,
                current_q,
                charge,
                jnp.full((n_samples,), site, dtype=jnp.int32),
                jnp.full(
                    (n_samples,), remaining_after, dtype=jnp.int32
                ),
            ),
        )
        hidden = gru_step(params["gru"], hidden, inputs)
        logits, _ = heads(params, hidden)
        mask = _action_mask(
            current_q, charge, remaining_after, config, table
        )
        logits = jnp.where(mask, logits, -1.0e30)
        action = jax.random.categorical(subkey, logits, axis=-1).astype(
            jnp.int32
        )
        occupation = ACTION_OCCUPATIONS[action]
        current_q = current_q + ACTION_DELTA_Q[action]
        charge = charge + occupation
        labels.extend((occupation, current_q))
        previous = action
    return jnp.stack(labels, axis=1)


def hubbard_probabilities(
    params: dict[str, object],
    states: jax.Array,
    config: HubbardFusionGRUConfig,
) -> jax.Array:
    values = jax.vmap(
        lambda state: hubbard_log_psi(params, state, config)
    )(states)
    return jnp.exp(2.0 * jnp.real(values))


__all__ = [
    "ACTION_DELTA_Q",
    "ACTION_OCCUPATIONS",
    "HubbardFusionGRUConfig",
    "hubbard_log_psi",
    "hubbard_probabilities",
    "hubbard_reachability_table",
    "init_hubbard_fusion_gru",
    "sample_hubbard_fusion",
    "transfer_hubbard_fusion_gru_parameters",
]
