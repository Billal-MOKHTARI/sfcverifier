"""Timed token-game simulation of an SFC (normal behaviour).

The simulator only uses the transition matrix (successors and divergence labels). It never
reads the convergence groups and never calls the verifier, so a false alarm on a simulated
trace is a genuine disagreement between the semantics below and the verification rules.

Semantics (one sample = one scan of period ``dt`` seconds):
  * an active step runs for max(1, round(duration * U(1-jitter, 1+jitter) / dt)) scans and then
    becomes *ready*; a ready step stays active until its outgoing transition fires;
  * when a step becomes ready it selects one divergence group (uniformly among its k values);
    the edges of the other groups become *dead* (unless the choice is a loop-back edge);
  * a step j can be entered when every forward edge into j is either dead or comes from a
    ready step that selected j, with at least one such step (dead-path elimination: a branch
    that was not chosen never blocks a convergence);
  * all steps whose selected successors can all be entered fire together in the same scan
    (AND divergences start all their branches at once, AND convergences wait for every branch);
  * a back edge (loop) restarts the part of the graph downstream of its target.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .model import SFCModel


@dataclass
class Trace:
    """A sampled execution: masks[t] is the bitmask of active steps at scan t."""

    masks: list[int]
    dt: float
    # firings[t]: transitions fired between t-1 and t, as (pre_mask, post_mask) components
    firings: list[list[tuple[int, int]]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.masks)


class Simulator:
    def __init__(self, model: SFCModel, seed: int = 0, dt: float = 1.0, jitter: float = 0.2) -> None:
        self.model = model
        self.rng = random.Random(seed)
        self.dt = dt
        self.jitter = jitter
        n = model.n
        self.out = [list(model.successors[i]) for i in range(n)]
        self.label = {(i, j): k for i, j, k in model.edges}
        self.back = self._back_edges()
        self.fwd_in: list[list[int]] = [[] for _ in range(n)]
        for i, j, _ in model.edges:
            if (i, j) not in self.back:
                self.fwd_in[j].append(i)
        self.dead: set[tuple[int, int]] = set()
        self.done: set[int] = set()
        self.remaining: dict[int, int] = {}          # active step -> scans left (0 = ready)
        self.selected: dict[int, list[int]] = {}     # ready step -> successors it fires to
        for s in model.initial:
            self._enter(s)

    # ------------------------------------------------------------------ structure
    def _back_edges(self) -> set[tuple[int, int]]:
        """Edges closing a cycle in a DFS from the initial steps (iterative)."""
        back: set[tuple[int, int]] = set()
        state: dict[int, int] = {}
        for root in self.model.initial:
            if root in state:
                continue
            state[root] = 1
            stack = [(root, iter(self.out[root]))]
            while stack:
                u, it = stack[-1]
                v = next(it, None)
                if v is None:
                    state[u] = 2
                    stack.pop()
                elif state.get(v) == 1:
                    back.add((u, v))
                elif v not in state:
                    state[v] = 1
                    stack.append((v, iter(self.out[v])))
        return back

    def _downstream(self, start: int) -> set[int]:
        seen, stack = {start}, [start]
        while stack:
            u = stack.pop()
            for v in self.out[u]:
                if (u, v) not in self.back and v not in seen:
                    seen.add(v)
                    stack.append(v)
        return seen

    # ------------------------------------------------------------------ token game
    def _dwell(self, step: int) -> int:
        d = self.model.durations[step] * self.rng.uniform(1 - self.jitter, 1 + self.jitter)
        return max(1, round(d / self.dt))

    def _enter(self, step: int) -> None:
        self.done.add(step)
        self.remaining[step] = self._dwell(step)

    def _restart(self, region: set[int]) -> None:
        """Forget the history of a loop body before it is executed again."""
        for v in region:
            self.done.discard(v)
            for p in self.fwd_in[v]:
                if p in region:
                    self.dead.discard((p, v))

    def _kill(self, i: int, j: int) -> None:
        if (i, j) in self.back:
            return
        self.dead.add((i, j))
        if j not in self.done and all((p, j) in self.dead for p in self.fwd_in[j]):
            self.done.add(j)                       # every way into j is dead: j is skipped
            for w in self.out[j]:
                self._kill(j, w)

    def _select(self, step: int) -> list[int]:
        groups: dict[int, list[int]] = {}
        for j in self.out[step]:
            groups.setdefault(self.label[(step, j)], []).append(j)
        k = self.rng.choice(sorted(groups))
        chosen = groups[k]
        if not any((step, j) in self.back for j in chosen):
            for j in self.out[step]:
                if self.label[(step, j)] != k:
                    self._kill(step, j)
        return chosen

    def _can_enter(self, i: int, j: int, firing: set[int]) -> bool:
        if (i, j) in self.back:
            return True
        token = False
        for p in self.fwd_in[j]:
            if (p, j) in self.dead:
                continue
            if p in firing and j in self.selected[p]:
                token = True
            else:
                return False                       # still waiting for this predecessor
        return token

    def _fire(self) -> list[tuple[int, int]]:
        firing = set(self.selected)
        changed = True
        while changed:                             # keep the steps whose whole selection can enter
            changed = False
            for i in list(firing):
                if not all(self._can_enter(i, j, firing) for j in self.selected[i]):
                    firing.discard(i)
                    changed = True
        pairs = [(i, j) for i in firing for j in self.selected[i]]
        for i in firing:
            del self.selected[i]
            del self.remaining[i]
        for i, j in pairs:
            if (i, j) in self.back:
                self._restart(self._downstream(j))
        for _, j in pairs:
            if j not in self.remaining:
                self._enter(j)
        return _components(pairs)

    def step(self) -> list[tuple[int, int]]:
        for s in list(self.remaining):
            if self.remaining[s] > 0:
                self.remaining[s] -= 1
                if self.remaining[s] == 0:
                    self.selected[s] = self._select(s)
        return self._fire()

    @property
    def mask(self) -> int:
        m = 0
        for s in self.remaining:
            m |= 1 << s
        return m

    def run(self, scans: int) -> Trace:
        masks, firings = [self.mask], [[]]
        for _ in range(scans - 1):
            firings.append(self.step())
            masks.append(self.mask)
        return Trace(masks=masks, dt=self.dt, firings=firings)


def _components(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Split fired edges into transitions (connected components of the bipartite graph)."""
    parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(x: tuple[str, int]) -> tuple[str, int]:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        parent[find(("s", i))] = find(("t", j))
    comps: dict[tuple[str, int], list[int]] = {}
    for i, j in pairs:
        root = find(("s", i))
        c = comps.setdefault(root, [0, 0])
        c[0] |= 1 << i
        c[1] |= 1 << j
    return [(pre, post) for pre, post in comps.values()]


def simulate(model: SFCModel, scans: int, seed: int = 0, dt: float = 1.0, jitter: float = 0.2) -> Trace:
    return Simulator(model, seed=seed, dt=dt, jitter=jitter).run(scans)


def stalled(trace: Trace, window: int) -> bool:
    """True if the configuration never changes during the last ``window`` scans."""
    return len(trace.masks) > window and len(set(trace.masks[-window:])) == 1
