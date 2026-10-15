"""Simulate the SFC graph of control_logic.py and write the activation vectors to CSV.

Semantics (strict SFC, token game on the edges (i, j, k) of the matrix M):
  * a state runs for max(1, ceil(durations[i] / dt)) scan cycles, then becomes "ready";
    a ready state STAYS ACTIVE until its outgoing transition fires;
  * when a state becomes ready it selects its successors:
      - single successor        -> that edge
      - all k equal (AND)       -> every outgoing edge
      - different k (OR)        -> one k group chosen at random, the other edges become "dead"
  * a transition fires when every selected successor can be entered, all at once
    (AND divergence enters all its successors in the same scan);
  * a state with several incoming edges is a join: it is entered once every incoming edge is
    either dead or comes from a ready predecessor that selected it, with at least one such
    predecessor. Dead edges are propagated (dead-path elimination), so an OR branch that was
    not taken never blocks the join;
  * back edges (the loop edges 25->18 and 25->0) restart everything downstream of their target.

Usage:
    python generate_data.py --rows 5000 --out data.csv
"""
import argparse
import contextlib
import csv
import io
import math
import random
import re
from collections import defaultdict
from datetime import datetime, timedelta

SYMBOL_RE = re.compile(r"[A-Za-z]+_(\d+)_(\d+)\^(\d+)")


def load_model():
    with contextlib.redirect_stdout(io.StringIO()):  # control_logic prints on import
        import control_logic
    edges = sorted(
        tuple(int(g) for g in SYMBOL_RE.fullmatch(str(s)).groups())
        for s in control_logic.M.free_symbols if SYMBOL_RE.fullmatch(str(s))
    )
    return edges, dict(control_logic.durations), control_logic.M.rows


class SFCSimulator:
    def __init__(self, edges, durations, n_states, dt, rng, root=0):
        self.dt, self.rng, self.root, self.n = dt, rng, root, n_states
        self.durations = durations
        self.k = {(i, j): k for i, j, k in edges}
        self.out = defaultdict(list)
        for i, j, _ in edges:
            self.out[i].append(j)
        self.back = self._find_back_edges()
        self.fwd_in = defaultdict(list)
        for i, j, _ in edges:
            if (i, j) not in self.back:
                self.fwd_in[j].append(i)
        self.dead = set()      # forward edges eliminated in the current epoch
        self.done = set()      # states entered (or dead) in the current epoch
        self.active = {}       # state -> remaining scan cycles (0 = ready, waiting to fire)
        self.selected = {}     # ready state -> successors it fires to
        self._reset(set(range(n_states)))
        self._enter(root)

    def _find_back_edges(self):
        back, state = set(), {}
        def dfs(u):
            state[u] = 1
            for v in self.out[u]:
                if state.get(v) == 1:
                    back.add((u, v))
                elif v not in state:
                    dfs(v)
            state[u] = 2
        dfs(self.root)
        return back

    def _descendants(self, start):
        seen, stack = {start}, [start]
        while stack:
            u = stack.pop()
            for v in self.out[u]:
                if (u, v) not in self.back and v not in seen:
                    seen.add(v)
                    stack.append(v)
        return seen

    def _reset(self, states):
        for v in states:
            self.done.discard(v)
            for p in self.fwd_in[v]:
                self.dead.discard((p, v))

    def _enter(self, v):
        self.done.add(v)
        self.active[v] = max(1, math.ceil(self.durations[v] / self.dt))

    def _kill(self, i, j):
        if (i, j) in self.back:
            return
        self.dead.add((i, j))
        if j not in self.done and all((p, j) in self.dead for p in self.fwd_in[j]):
            self.done.add(j)               # every incoming edge dead -> j is dead too
            for w in self.out[j]:
                self._kill(j, w)

    def _select(self, v):
        groups = defaultdict(list)
        for w in self.out[v]:
            groups[self.k[(v, w)]].append(w)
        if not groups:
            return []
        chosen = self.rng.choice(sorted(groups)) if len(groups) > 1 else next(iter(groups))
        for w in self.out[v]:
            if self.k[(v, w)] != chosen:
                self._kill(v, w)
        return groups[chosen]

    def _can_enter(self, i, j, firing):
        if (i, j) in self.back:
            return True
        token = False
        for p in self.fwd_in[j]:
            if (p, j) in self.dead:
                continue
            if p in firing and j in self.selected[p]:
                token = True
            else:
                return False               # still waiting for this predecessor
        return token

    def _fire(self):
        firing = set(self.selected)
        changed = True
        while changed:                     # keep only ready states whose whole selection can enter
            changed = False
            for i in list(firing):
                if not all(self._can_enter(i, j, firing) for j in self.selected[i]):
                    firing.discard(i)
                    changed = True
        targets = []
        for i in firing:
            targets += [(i, j) for j in self.selected.pop(i)]
            del self.active[i]
        for i, j in targets:
            if (i, j) in self.back:        # loop: restart everything downstream of j
                self._reset(self._descendants(j))
        for _, j in targets:
            self._enter(j)

    def vector(self):
        return [v in self.active for v in range(self.n)]

    def step(self):
        for v in list(self.active):
            if self.active[v] > 0:
                self.active[v] -= 1
                if self.active[v] == 0:
                    self.selected[v] = self._select(v)
        self._fire()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data.csv")
    ap.add_argument("--rows", type=int, default=5000)
    ap.add_argument("--dt", type=float, default=1.0, help="scan period in seconds")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--start", default="2024-01-01 00:00:00")
    args = ap.parse_args()

    edges, durations, n = load_model()
    sim = SFCSimulator(edges, durations, n, args.dt, random.Random(args.seed))
    t0 = datetime.fromisoformat(args.start)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "timestamp"] + [str(i) for i in range(n)])
        for r in range(args.rows):
            ts = t0 + timedelta(seconds=r * args.dt)
            w.writerow([r, ts.isoformat(sep=" ")] + [str(b) for b in sim.vector()])
            sim.step()
    print(f"Wrote {args.rows} rows to {args.out}")


if __name__ == "__main__":
    main()
