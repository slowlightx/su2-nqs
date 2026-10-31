"""Small global proposals in fixed doubled M=0, independent of the PEPS.

Bit zero is doubled m=-1, bit one is doubled m=+1. Exact reachability
is masked before every conditional softmax. These are physical-bit proposal
laws, not SU(2) wavefunctions. The target PEPS is never changed.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from peps_vmc.proposals import CoveringBridgeProposal

Kind = Literal["attention", "gru"]


def _layer_norm(x: Any) -> Any:
    return (x-jnp.mean(x, axis=-1, keepdims=True))/jnp.sqrt(
        jnp.var(x, axis=-1, keepdims=True)+1e-6)


@dataclass(frozen=True)
class FixedMagnetizationAR:
    """Causal attention or GRU over every earlier bit, with exact doubled M=0."""

    n_sites: int
    kind: Kind
    width: int = 16
    hidden: int = 19

    def __post_init__(self) -> None:
        if (isinstance(self.n_sites, bool) or not isinstance(self.n_sites, int)
                or self.n_sites < 2 or self.n_sites % 2):
            raise ValueError("positive even site count required for doubled M=0")
        if self.kind not in ("attention", "gru") or self.width < 2 or self.hidden < 2:
            raise ValueError("valid architecture and positive widths required")

    def initialize(self, seed: int) -> dict[str, Any]:
        """Return a deterministic small real network; no Hamiltonian data."""
        rng = np.random.default_rng(seed)
        def normal(shape: tuple[int, ...], scale: float) -> Any:
            return jnp.asarray(rng.normal(size=shape)*scale, dtype=jnp.float64)
        d, h, n = self.width, self.hidden, self.n_sites
        p = dict(embedding=normal((3, d), .1), position=normal((n, d), .1))
        if self.kind == "attention":
            p.update(qkv=normal((d, 3*d), d**-.5), out=normal((d, d), d**-.5),
                     ff1=normal((d, 2*d), d**-.5), b1=jnp.zeros(2*d),
                     ff2=normal((2*d, d), (2*d)**-.5), b2=jnp.zeros(d),
                     head=normal((d, 2), .02), bias=jnp.zeros(2))
        else:
            p.update(gates=normal((d+h, 2*h), (d+h)**-.5), bg=jnp.zeros(2*h),
                     candidate=normal((d+h, h), (d+h)**-.5), bc=jnp.zeros(h),
                     head=normal((h, 2), .02), bias=jnp.zeros(2))
        return p

    def logits(self, p: dict[str, Any], rows: Any) -> Any:
        """Return conditionals using only bits strictly before each position."""
        rows = jnp.asarray(rows, jnp.int32)
        shifted = jnp.concatenate((jnp.full((len(rows), 1), 2, jnp.int32), rows[:, :-1]), axis=1)
        x = p["embedding"][shifted]+p["position"][None]
        if self.kind == "attention":
            q, k, v = jnp.split(_layer_norm(x)@p["qkv"], 3, axis=-1)
            scores = jnp.einsum("bid,bjd->bij", q, k)/jnp.sqrt(self.width)
            causal = jnp.arange(self.n_sites)[None, :] <= jnp.arange(self.n_sites)[:, None]
            weights = jax.nn.softmax(jnp.where(causal, scores, -jnp.inf), axis=-1)
            x = x+jnp.einsum("bij,bjd->bid", weights, v)@p["out"]
            x = x+jax.nn.gelu(_layer_norm(x)@p["ff1"]+p["b1"])@p["ff2"]+p["b2"]
            raw = _layer_norm(x)@p["head"]+p["bias"]
        else:
            def step(h: Any, xt: Any) -> tuple[Any, Any]:
                r, z = jnp.split(jax.nn.sigmoid(jnp.concatenate((xt, h), -1)@p["gates"]+p["bg"]), 2, -1)
                c = jnp.tanh(jnp.concatenate((xt, r*h), -1)@p["candidate"]+p["bc"])
                h = (1-z)*h+z*c
                return h, h@p["head"]+p["bias"]
            _, output = jax.lax.scan(step, jnp.zeros((len(rows), self.hidden)), jnp.swapaxes(x, 0, 1))
            raw = jnp.swapaxes(output, 0, 1)
        used = jnp.cumsum(rows, axis=1)-rows
        remaining = self.n_sites-jnp.arange(self.n_sites)-1
        next_used = used[:, :, None]+jnp.arange(2)
        valid = ((next_used <= self.n_sites//2)
                 & (next_used+remaining[None, :, None] >= self.n_sites//2))
        return jnp.where(valid, raw, -jnp.inf)

    def log_probability(self, p: dict[str, Any], rows: Any) -> Any:
        """Exact normalized autoregressive log law, zero outside doubled M=0."""
        rows = jnp.asarray(rows, jnp.int32)
        terms = jax.nn.log_softmax(self.logits(p, rows), axis=-1)
        values = jnp.take_along_axis(terms, rows[:, :, None], axis=-1)[:, :, 0]
        valid = jnp.all((rows == 0) | (rows == 1), axis=1) & (jnp.sum(rows, axis=1) == self.n_sites//2)
        return jnp.where(valid, values.sum(axis=1), -jnp.inf)

    def symmetric_log_probability(self, p: dict[str, Any], rows: Any) -> Any:
        """Normalized global-spin-flip average; singlet probabilities share it."""
        return jnp.logaddexp(self.log_probability(p, rows), self.log_probability(p, 1-rows))-jnp.log(2.)

    def sample(self, p: dict[str, Any], key: Any, count: int) -> Any:
        """Construct legal rows with every reachability mask applied in advance."""
        def step(i: Any, state: tuple[Any, Any]) -> tuple[Any, Any]:
            rows, current_key = state
            current_key, draw_key = jax.random.split(current_key)
            bit = jax.random.categorical(draw_key, self.logits(p, rows)[:, i], axis=-1).astype(jnp.int32)
            return rows.at[:, i].set(bit), current_key
        rows, key = jax.lax.fori_loop(0, self.n_sites, step,
                                     (jnp.zeros((count, self.n_sites), jnp.int32), key))
        flip = jax.random.bernoulli(key, shape=(count, 1))
        return jnp.where(flip, 1-rows, rows)


@dataclass(frozen=True)
class DefensiveLearnedProposal:
    """Exact mixture retaining the original full-support H-exchange bridge."""

    model: FixedMagnetizationAR
    parameters: dict[str, Any]
    bridge: CoveringBridgeProposal
    learned_fraction: float = .75

    def __post_init__(self) -> None:
        if self.bridge.n_sites != self.model.n_sites or not 0 < self.learned_fraction < 1:
            raise ValueError("matching sites and a strict defensive fraction required")

    def density(self, rows: np.ndarray) -> np.ndarray:
        """Full-support exact law; its uniform floor also protects H² at nodes."""
        rows = self.bridge._rows(rows)
        learned = np.exp(np.asarray(self.model.symmetric_log_probability(self.parameters, rows)))
        return (1-self.learned_fraction)*self.bridge.density(rows)+self.learned_fraction*learned

    def draw(self, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """IID draws from the frozen known mixture, never accept/reject by PEPS."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("positive sample count required")
        selected = rng.random(count) < self.learned_fraction
        rows, _ = self.bridge.draw(count, rng)
        if np.any(selected):
            key = jax.random.PRNGKey(int(rng.integers(2**31)))
            fn = jax.jit(lambda p, key: self.model.sample(p, key, int(selected.sum())))
            rows[selected] = np.asarray(fn(self.parameters, key))
        return rows, self.density(rows)

    @property
    def uniform_floor(self) -> float:
        """Analytic probability floor across every fixed doubled M=0 row."""
        return (1-self.learned_fraction)*self.bridge.epsilon/comb(self.model.n_sites, self.model.n_sites//2)
