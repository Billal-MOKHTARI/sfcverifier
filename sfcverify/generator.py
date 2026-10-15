"""Random generation of block-structured SFC benchmark graphs.

Graphs are produced from a small grammar, nested recursively:

    Block := Step
           | Seq(B1, ..., Bm)              sequence
           | Par(B1, ..., Bm)              AND: all branches start together and are joined
           | Alt(B1, ..., Bm [, skip])     OR: one branch (or an empty "skip" branch) is chosen
           | Loop(B)                       body repeated while the tail step chooses "again"

Connecting two blocks creates one transition from the *tails* of the first block to the
*heads* of the second. When both sides contain several steps (Par followed by Par) this is a
combined AND-convergence/AND-divergence transition, i.e. a complete bipartite set of edges.
Every graph starts in an INIT step and the last step loops back to it (cyclic process).

The convergence groups written to the model follow directly from the construction: each
transition contributes its source set as one group of each of its target steps.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
import numpy as np

from .model import SFCModel


@dataclass(frozen=True)
class Profile:
    """Shape of the generated graphs."""

    name: str
    min_steps: int
    max_steps: int
    max_depth: int
    weights: dict[str, float]          # relative weight of seq / par / alt / loop
    max_branches: int = 4
    skip_probability: float = 0.25


PROFILES: dict[str, Profile] = {
    "small": Profile("small", 10, 25, 3, {"seq": 0.40, "par": 0.25, "alt": 0.25, "loop": 0.10}, 3),
    "medium": Profile("medium", 26, 60, 4, {"seq": 0.35, "par": 0.28, "alt": 0.25, "loop": 0.12}),
    "large": Profile("large", 61, 120, 5, {"seq": 0.35, "par": 0.28, "alt": 0.25, "loop": 0.12}),
    "complex": Profile("complex", 40, 110, 8, {"seq": 0.15, "par": 0.40, "alt": 0.30, "loop": 0.15}, 5),
}

# number of benchmark graphs per profile (100 in total)
BENCHMARK_MIX = {"small": 25, "medium": 30, "large": 25, "complex": 20}

Node = tuple   # ("step",) | ("seq", [..]) | ("par", [..]) | ("alt", [..], skip) | ("loop", child)


# ---------------------------------------------------------------------- structure
def _split(total: int, parts: int, rng: random.Random) -> list[int]:
    """Random composition of ``total`` into ``parts`` positive integers."""
    cuts = sorted(rng.sample(range(1, total), parts - 1))
    return [b - a for a, b in zip([0, *cuts], [*cuts, total])]


def _random_block(budget: int, depth: int, parent: str, p: Profile, rng: random.Random) -> Node:
    if budget <= 1:
        return ("step",)
    if depth >= p.max_depth:
        return ("seq", [("step",)] * budget)
    options = {k: w for k, w in p.weights.items() if k != parent or k == "loop"}
    if budget < 4:
        options.pop("alt", None)       # split + merge + two alternatives
    if budget < 3:
        options.pop("loop", None)      # head + tail + body
    kind = rng.choices(list(options), weights=list(options.values()))[0]
    if kind in ("seq", "par"):
        m = rng.randint(2, min(p.max_branches if kind == "par" else 5, budget))
        return (kind, [_random_block(b, depth + 1, kind, p, rng) for b in _split(budget, m, rng)])
    if kind == "alt":
        m = rng.randint(2, min(3, budget - 2))
        skip = rng.random() < p.skip_probability
        return ("alt", [_random_block(b, depth + 1, kind, p, rng)
                        for b in _split(budget - 2, m, rng)], skip)
    return ("loop", _random_block(budget - 2, depth + 1, kind, p, rng))


def describe(node: Node) -> str:
    """Compact textual form of the block structure, e.g. Seq(s,Par(s,s),Loop(s))."""
    kind = node[0]
    if kind == "step":
        return "s"
    if kind == "loop":
        return f"Loop({describe(node[1])})"
    inner = ",".join(describe(c) for c in node[1])
    if kind == "alt" and node[2]:
        inner += ",skip"
    return f"{kind.capitalize()}({inner})"


# ---------------------------------------------------------------------- construction
class _Builder:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.durations: list[float] = []
        self.edges: dict[tuple[int, int], int] = {}
        self.conv: dict[int, list[frozenset[int]]] = {}
        self.next_k: dict[int, int] = {}

    def new_step(self, duration: float | None = None) -> int:
        self.durations.append(self._duration() if duration is None else duration)
        return len(self.durations) - 1

    def _duration(self) -> float:
        r = self.rng.random()
        if r < 0.08:
            return 0.0                                        # instantaneous step
        if r < 0.40:
            return float(self.rng.randint(1, 3))
        if r < 0.85:
            return float(self.rng.randint(3, 8))
        return float(self.rng.randint(8, 15))

    def connect(self, tails: frozenset[int], heads: frozenset[int]) -> None:
        """One transition tails -> heads (one new divergence group per source step)."""
        for i in tails:
            k = self.next_k.get(i, 1)
            self.next_k[i] = k + 1
            for j in heads:
                assert (i, j) not in self.edges, "duplicate edge"
                self.edges[(i, j)] = k
        for j in heads:
            self.conv.setdefault(j, []).append(frozenset(tails))

    def build(self, node: Node) -> tuple[frozenset[int], frozenset[int]]:
        kind = node[0]
        if kind == "step":
            s = frozenset({self.new_step()})
            return s, s
        if kind == "seq":
            parts = [self.build(c) for c in node[1]]
            for (_, tails), (heads, _) in zip(parts, parts[1:]):
                self.connect(tails, heads)
            return parts[0][0], parts[-1][1]
        if kind == "par":
            parts = [self.build(c) for c in node[1]]
            return (frozenset().union(*(h for h, _ in parts)),
                    frozenset().union(*(t for _, t in parts)))
        if kind == "alt":
            split, merge = self.new_step(), None
            parts = [self.build(c) for c in node[1]]
            merge = self.new_step()
            for heads, tails in parts:
                self.connect(frozenset({split}), heads)
                self.connect(tails, frozenset({merge}))
            if node[2]:
                self.connect(frozenset({split}), frozenset({merge}))
            return frozenset({split}), frozenset({merge})
        # loop
        head = self.new_step()
        body_heads, body_tails = self.build(node[1])
        tail = self.new_step()
        self.connect(frozenset({head}), body_heads)
        self.connect(body_tails, frozenset({tail}))
        self.connect(frozenset({tail}), frozenset({head}))       # loop-back edge
        return frozenset({head}), frozenset({tail})


def generate(seed: int, profile: Profile, name: str | None = None) -> SFCModel:
    """Generate one cyclic block-structured SFC with a size inside the profile range."""
    rng = random.Random(seed)
    target = rng.randint(profile.min_steps, profile.max_steps)
    body = _random_block(target - 2, 0, "", profile, rng)
    b = _Builder(rng)
    init = b.new_step(0.0)
    heads, tails = b.build(body)
    final = b.new_step(1.0)
    b.connect(frozenset({init}), heads)
    b.connect(tails, frozenset({final}))
    b.connect(frozenset({final}), frozenset({init}))

    n = len(b.durations)
    labels = np.zeros((n, n), dtype=np.int16)
    for (i, j), k in b.edges.items():
        labels[i, j] = k
    preds = labels.astype(bool).sum(axis=0)
    convergence = {j: tuple(dict.fromkeys(groups)) for j, groups in b.conv.items() if preds[j] >= 2}
    names = ["INIT"] + [f"S{i}" for i in range(1, n - 1)] + ["END"]
    return SFCModel(
        name=name or f"{profile.name}_{seed}",
        labels=labels,
        convergence=convergence,
        durations=np.array(b.durations),
        initial=(init,),
        step_names=tuple(names),
        description=f"Randomly generated {profile.name} block-structured SFC (seed {seed}).",
        metadata={"tier": profile.name, "seed": seed, "structure": describe(body)},
    )


def benchmark(base_seed: int = 2026) -> list[SFCModel]:
    """The 100-graph benchmark: deterministic given ``base_seed``."""
    models, idx = [], 0
    for tier, count in BENCHMARK_MIX.items():
        for _ in range(count):
            models.append(generate(base_seed + idx, PROFILES[tier], name=f"G{idx + 1:03d}_{tier}"))
            idx += 1
    return models
