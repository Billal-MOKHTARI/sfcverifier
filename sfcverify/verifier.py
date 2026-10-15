"""Transition verification with the fired-transition matrix Phi.

For two consecutive activation vectors S1 = S(t-1) and S2 = S(t):

    l   = S1 o (1 - S2)                 steps that left (deactivated)
    n   = S2 o (1 - S1)                 steps that entered (activated)
    Phi = diag(l) . X . diag(n)  =  diag(l) . E(t) . diag(n),   E(t) = X . diag(S2)

Phi keeps exactly the edges that fired between t-1 and t. The transition is valid iff

    R1  no deadlock      1^T S2 >= 1
    R2  justification    (1^T Phi)_j != 0                      for every entered j
    R3  divergence       (Phi 1)_i = G_i^k for exactly one k    for every left i
                         (G_i^k = sum of the delta_ij^k of divergence group k of step i)
    R4  convergence      (1^T Phi)_j = sum_{p in C} X[p, j]     for one convergence group C of j

Because every entry of X is a distinct symbol, comparing the symbolic sums is the same as
comparing the *sets* of fired edges. The implementation below therefore encodes rows and
columns of Phi as integer bitmasks; ``sfcverify.symbolic`` contains the literal SymPy
version of the formulas and the test-suite checks that both agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np

from .model import SFCModel

RULES = {
    "R1": "No deadlock: at least one step is active",
    "R2": "Justification: every entered step is reached by a fired edge",
    "R3": "Divergence: every left step fires exactly one whole divergence group",
    "R4": "Convergence: every entered step is reached by exactly one convergence group",
}


@dataclass(frozen=True)
class VerifierConfig:
    """Which rules are enforced; used for the ablation study.

    divergence = "full"  R3 as defined above
                 "weak"  only require that a left step fired at least one edge
                 "off"   no check on the rows of Phi
    convergence = "full" R4 as defined above, "off" disables it (R2 still applies)
    """

    name: str = "full"
    divergence: Literal["full", "weak", "off"] = "full"
    convergence: Literal["full", "off"] = "full"


PRESETS: dict[str, VerifierConfig] = {
    "full": VerifierConfig("full", "full", "full"),
    "no_divergence": VerifierConfig("no_divergence", "weak", "full"),
    "no_convergence": VerifierConfig("no_convergence", "full", "off"),
    "adjacency": VerifierConfig("adjacency", "weak", "off"),
}


@dataclass(frozen=True)
class Violation:
    rule: str
    step: int | None
    message: str


@dataclass
class TransitionCheck:
    """Full result of one verification, used by the GUI."""

    valid: bool
    left: list[int]
    entered: list[int]
    fired: list[tuple[int, int, int]]                 # non-zero entries of Phi: (i, j, k)
    violations: list[Violation] = field(default_factory=list)
    divergence_match: dict[int, int | None] = field(default_factory=dict)          # i -> k
    convergence_match: dict[int, list[int] | None] = field(default_factory=dict)   # j -> group


def to_mask(state: Iterable[bool] | np.ndarray) -> int:
    """Encode an activation vector as an integer bitmask (bit i = step i)."""
    arr = np.asarray(state, dtype=bool)
    return int.from_bytes(np.packbits(arr, bitorder="little").tobytes(), "little")


def bits(mask: int) -> Iterable[int]:
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def from_mask(mask: int, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=bool)
    for i in bits(mask):
        out[i] = True
    return out


class TransitionVerifier:
    """Checks consecutive activation vectors against the model (rules R1-R4)."""

    def __init__(self, model: SFCModel, config: VerifierConfig | str = "full") -> None:
        self.model = model
        self.config = PRESETS[config] if isinstance(config, str) else config
        n = model.n
        self._succ = [sum(1 << j for j in model.successors[i]) for i in range(n)]
        self._pred = [sum(1 << i for i in model.predecessors[j]) for j in range(n)]
        self._div = [{sum(1 << j for j in g): k for k, g in model.divergence_groups[i].items()}
                     for i in range(n)]
        self._conv = [{sum(1 << p for p in g): sorted(g) for g in model.convergence_groups(j)}
                      for j in range(n)]

    # ------------------------------------------------------------------ fast path
    def is_valid(self, prev: int, curr: int) -> bool:
        """Verify one transition given as bitmasks. Allocation-free hot path."""
        if curr == 0:
            return False                                   # R1
        if prev == curr:
            return True
        left, entered = prev & ~curr, curr & ~prev
        conv_full = self.config.convergence == "full"
        m = entered
        while m:
            low = m & -m
            j = low.bit_length() - 1
            m ^= low
            col = self._pred[j] & left                     # column j of Phi
            if not col:
                return False                               # R2
            if conv_full and col not in self._conv[j]:
                return False                               # R4
        div = self.config.divergence
        if div == "off":
            return True
        m = left
        while m:
            low = m & -m
            i = low.bit_length() - 1
            m ^= low
            row = self._succ[i] & entered                  # row i of Phi
            if div == "full":
                if row not in self._div[i]:
                    return False                           # R3
            elif not row:
                return False                               # R3 (weak)
        return True

    def alarms(self, masks: list[int]) -> np.ndarray:
        """alarm[t] is True when the transition S(t-1) -> S(t) is invalid (alarm[0]: R1 only)."""
        out = np.zeros(len(masks), dtype=bool)
        if not masks:
            return out
        out[0] = masks[0] == 0
        for t in range(1, len(masks)):
            out[t] = not self.is_valid(masks[t - 1], masks[t])
        return out

    # ------------------------------------------------------------------ detailed result
    def check(self, prev: Iterable[bool] | int, curr: Iterable[bool] | int) -> TransitionCheck:
        """Verify one transition and report Phi, matched groups and every violation."""
        p = prev if isinstance(prev, int) else to_mask(prev)
        c = curr if isinstance(curr, int) else to_mask(curr)
        left, entered = p & ~c, c & ~p
        labels = self.model.labels
        result = TransitionCheck(valid=True, left=list(bits(left)), entered=list(bits(entered)),
                                 fired=[])
        for i in bits(left):
            for j in bits(self._succ[i] & entered):
                result.fired.append((i, j, int(labels[i, j])))

        if c == 0:
            result.violations.append(Violation("R1", None, "deadlock: no step is active"))
        for j in bits(entered):
            col = self._pred[j] & left
            if not col:
                result.violations.append(Violation(
                    "R2", j, f"step {j} entered without any fired edge (no predecessor left)"))
                result.convergence_match[j] = None
                continue
            group = self._conv[j].get(col)
            result.convergence_match[j] = group
            if group is None and self.config.convergence == "full":
                result.violations.append(Violation(
                    "R4", j, f"step {j} entered from {sorted(bits(col))}, which is not one of its "
                             f"convergence groups {[sorted(g) for g in self.model.convergence_groups(j)]}"))
        for i in bits(left):
            row = self._succ[i] & entered
            k = self._div[i].get(row)
            result.divergence_match[i] = k
            if self.config.divergence == "full" and k is None:
                groups = {k: sorted(g) for k, g in self.model.divergence_groups[i].items()}
                fired = sorted(bits(row))
                reason = "fired nothing" if not row else f"fired to {fired}"
                result.violations.append(Violation(
                    "R3", i, f"step {i} left and {reason}; expected exactly one group of {groups}"))
            elif self.config.divergence == "weak" and not row:
                result.violations.append(Violation("R3", i, f"step {i} left and fired nothing"))
        result.valid = not result.violations
        return result
