"""Attack / fault injection on observed activation traces.

Attacks are applied to what the monitor observes (the activation vector S(t) read from the
PLC coils), on top of a clean simulated trace. Each injection produces an *episode*:

    scans a .. a+L-1      manipulated scans (the attack window)
    scans a .. horizon    evaluation horizon: an alarm in this range counts as a detection
                          (it includes the scans right after the attack, where a discontinuity
                          may reveal it)

Two families are distinguished:
  * structure-violating: the observed execution is impossible for the SFC structure;
  * structure-preserving (stealthy): the observed execution is a legal SFC execution that
    is nevertheless wrong (wrong timing, wrong branch, replayed data). These mark the
    theoretical limit of any purely structural monitor and are expected to be missed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from .model import SFCModel
from .simulator import Trace
from .verifier import bits

MAX_LENGTH = 8      # maximal attack window, in scans
MARGIN = 5          # scans evaluated after the attack window
WARMUP = 30         # scans skipped at the beginning of a trace


@dataclass(frozen=True)
class AttackSpec:
    name: str
    family: str          # "violating" | "stealthy"
    title: str
    description: str


@dataclass
class Episode:
    attack: str
    start: int           # first manipulated scan a
    end: int             # first scan after the manipulation
    horizon: int         # last evaluated scan (inclusive)
    segment: list[int]   # attacked masks for scans a-1 .. horizon (segment[0] is clean)
    clean: list[int]     # clean masks for the same scans


ATTACKS: dict[str, AttackSpec] = {s.name: s for s in [
    AttackSpec("deadlock", "violating", "Deadlock",
               "All coils forced to 0: no step is active."),
    AttackSpec("coil_manipulation", "violating", "Coil manipulation",
               "A random subset of coils receives random binary values at every scan."),
    AttackSpec("spurious_activation", "violating", "Spurious activation",
               "An inactive step is forced active."),
    AttackSpec("step_suppression", "violating", "Step suppression",
               "An active step is forced inactive (lost token)."),
    AttackSpec("illegal_jump", "violating", "Illegal jump",
               "The token of an active step is moved to a step that is not one of its successors."),
    AttackSpec("or_violation", "violating", "OR exclusivity violation",
               "When an OR divergence fires, a second alternative is activated as well."),
    AttackSpec("partial_and", "violating", "Partial AND divergence",
               "When an AND divergence fires, one of its branches is not activated."),
    AttackSpec("premature_join", "violating", "Premature AND convergence",
               "A convergence step is activated before all its predecessor branches finished."),
    AttackSpec("delay", "stealthy", "Delay (freeze)",
               "The observed configuration is frozen, then the execution continues late."),
    AttackSpec("wrong_branch", "stealthy", "Wrong OR branch",
               "At an OR divergence the observation follows the other, legal alternative."),
    AttackSpec("replay", "stealthy", "Replay",
               "A recorded legal segment starting from the same configuration is replayed."),
]}


class Injector:
    """Draws attack episodes on one clean trace."""

    def __init__(self, model: SFCModel, trace: Trace, seed: int = 0,
                 max_length: int = MAX_LENGTH, margin: int = MARGIN) -> None:
        self.model = model
        self.trace = trace
        self.masks = trace.masks
        self.rng = random.Random(seed)
        self.max_length = max_length
        self.margin = margin
        self.n = model.n
        self._succ = [sum(1 << j for j in model.successors[i]) for i in range(self.n)]
        self._groups = [[sum(1 << j for j in g) for g in model.divergence_groups[i].values()]
                        for i in range(self.n)]
        self._by_prev: dict[int, list[int]] | None = None
        self._dispatch: dict[str, Callable[[], Episode | None]] = {
            name: getattr(self, f"_{name}") for name in ATTACKS}

    # ------------------------------------------------------------------ public
    def inject(self, attack: str, tries: int = 60) -> Episode | None:
        """One effective episode of ``attack``, or None when the attack is not applicable."""
        for _ in range(tries):
            ep = self._dispatch[attack]()
            if ep is not None and ep.segment != ep.clean:
                return ep
        return None

    # ------------------------------------------------------------------ helpers
    def _window(self) -> tuple[int, int]:
        length = self.rng.randint(1, self.max_length)
        a = self.rng.randint(WARMUP, len(self.masks) - 2 * self.max_length - self.margin - 2)
        return a, length

    def _episode(self, attack: str, a: int, end: int, horizon: int,
                 transform: Callable[[int, int], int]) -> Episode:
        clean = self.masks[a - 1: horizon + 1]
        seg = [clean[0]] + [transform(t, self.masks[t]) for t in range(a, horizon + 1)]
        return Episode(attack, a, end, horizon, seg, list(clean))

    def _simple(self, attack: str, f: Callable[[int, int], int]) -> Episode:
        a, length = self._window()
        end = a + length
        return self._episode(attack, a, end, end + self.margin - 1,
                             lambda t, m: f(t, m) if t < end else m)

    def _pick(self, mask: int) -> int | None:
        options = list(bits(mask))
        return self.rng.choice(options) if options else None

    def _firing_scans(self, predicate: Callable[[int, int], bool]) -> list[tuple[int, int, int]]:
        lo, hi = WARMUP, len(self.masks) - self.max_length - self.margin - 2
        return [(t, pre, post) for t in range(lo, hi)
                for pre, post in self.trace.firings[t] if predicate(pre, post)]

    def _same_configuration(self, a: int) -> list[int]:
        if self._by_prev is None:
            self._by_prev = {}
            for t in range(1, len(self.masks) - self.max_length - self.margin - 2):
                self._by_prev.setdefault(self.masks[t - 1], []).append(t)
        return [c for c in self._by_prev.get(self.masks[a - 1], []) if c != a]

    # ------------------------------------------------------------------ violating
    def _deadlock(self) -> Episode:
        return self._simple("deadlock", lambda t, m: 0)

    def _coil_manipulation(self) -> Episode:
        k = self.rng.randint(1, max(2, self.n // 4))
        coils = self.rng.sample(range(self.n), k)
        values = {}

        def f(t: int, m: int) -> int:
            for c in coils:
                bit = values.setdefault((t, c), self.rng.random() < 0.5)
                m = m | (1 << c) if bit else m & ~(1 << c)
            return m
        return self._simple("coil_manipulation", f)

    def _spurious_activation(self) -> Episode | None:
        a, length = self._window()
        x = self._pick(((1 << self.n) - 1) & ~(self.masks[a - 1] | self.masks[a]))
        if x is None:
            return None
        end = a + length
        return self._episode("spurious_activation", a, end, end + self.margin - 1,
                             lambda t, m: m | (1 << x) if t < end else m)

    def _step_suppression(self) -> Episode | None:
        a, length = self._window()
        x = self._pick(self.masks[a - 1] & self.masks[a])
        if x is None:
            return None
        end = a + length
        return self._episode("step_suppression", a, end, end + self.margin - 1,
                             lambda t, m: m & ~(1 << x) if t < end else m)

    def _illegal_jump(self) -> Episode | None:
        a, length = self._window()
        x = self._pick(self.masks[a - 1] & self.masks[a])
        if x is None:
            return None
        forbidden = self._succ[x] | (1 << x) | self.masks[a - 1] | self.masks[a]
        y = self._pick(((1 << self.n) - 1) & ~forbidden)
        if y is None:
            return None
        end = a + length
        return self._episode("illegal_jump", a, end, end + self.margin - 1,
                             lambda t, m: (m & ~(1 << x)) | (1 << y) if t < end else m)

    def _or_violation(self) -> Episode | None:
        cands = self._firing_scans(
            lambda pre, post: any(len(self._groups[i]) > 1 for i in bits(pre)))
        if not cands:
            return None
        a, pre, post = self.rng.choice(cands)
        i = self.rng.choice([i for i in bits(pre) if len(self._groups[i]) > 1])
        others = [g for g in self._groups[i] if not g & post]
        if not others:
            return None
        extra = self.rng.choice(others)
        end = a + self.rng.randint(1, self.max_length)
        return self._episode("or_violation", a, end, end + self.margin - 1,
                             lambda t, m: m | extra if t < end else m)

    def _partial_and(self) -> Episode | None:
        cands = self._firing_scans(lambda pre, post: post & (post - 1) != 0)
        if not cands:
            return None
        a, _, post = self.rng.choice(cands)
        b = self._pick(post)
        end = a + self.rng.randint(1, self.max_length)
        return self._episode("partial_and", a, end, end + self.margin - 1,
                             lambda t, m: m & ~(1 << b) if t < end else m)

    def _premature_join(self) -> Episode | None:
        cands = self._firing_scans(lambda pre, post: pre & (pre - 1) != 0)
        self.rng.shuffle(cands)
        for tau, pre, post in cands[:20]:
            run = 0                                   # scans before tau where all of pre is active
            while run < self.max_length and tau - run - 2 >= WARMUP and \
                    self.masks[tau - run - 2] & pre == pre:
                run += 1
            if run < 1:
                continue
            length = self.rng.randint(1, run)
            a = tau - length
            members = list(bits(pre))
            early = self.rng.sample(members, self.rng.randint(1, len(members) - 1))
            w = sum(1 << s for s in early)
            return self._episode("premature_join", a, tau, tau + self.margin - 1,
                                 lambda t, m: (m & ~w) | post if t < tau else m)
        return None

    # ------------------------------------------------------------------ stealthy
    def _delay(self) -> Episode | None:
        a, length = self._window()
        end = a + length
        frozen = self.masks[a - 1]
        return self._episode("delay", a, end, end + self.margin - 1,
                             lambda t, m: frozen if t < end else self.masks[t - length])

    def _wrong_branch(self) -> Episode | None:
        cands = self._firing_scans(
            lambda pre, post: any(len(self._groups[i]) > 1 for i in bits(pre)))
        self.rng.shuffle(cands)
        for a, _, _ in cands[:20]:
            others = [c for c in self._same_configuration(a) if self.masks[c] != self.masks[a]]
            if others:
                c = self.rng.choice(others)
                end = a + self.rng.randint(1, self.max_length)
                return self._episode("wrong_branch", a, end, end + self.margin - 1,
                                     lambda t, m: self.masks[c + t - a])
        return None

    def _replay(self) -> Episode | None:
        a, length = self._window()
        others = self._same_configuration(a)
        if not others:
            return None
        c = self.rng.choice(others)
        end = a + length
        return self._episode("replay", a, end, end + self.margin - 1,
                             lambda t, m: self.masks[c + t - a] if t < end else m)
