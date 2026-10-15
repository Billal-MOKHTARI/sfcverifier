"""SFC model: transition matrix X with synchronization labels and convergence groups.

The control logic is a directed graph over steps 0..n-1, encoded as an integer matrix

    labels[i, j] = k > 0   <=>   X[i, j] = delta_ij^(k)   (edge i -> j in divergence group k)
    labels[i, j] = 0       <=>   no edge

Edges leaving the same step with the same k form one *divergence group*: they are
activated together (AND). Different k values are mutually exclusive alternatives (OR).

Steps with several predecessors additionally carry *convergence groups*: the sets of
predecessors that enter the step together. One group containing every predecessor is an
AND-convergence, one singleton group per predecessor is an OR-convergence. Steps with a
single predecessor, or steps not listed, default to one singleton group per predecessor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

Group = frozenset[int]


class ModelError(ValueError):
    """Raised when a model file is structurally invalid."""


@dataclass(frozen=True, eq=False)
class SFCModel:
    name: str
    labels: np.ndarray
    convergence: dict[int, tuple[Group, ...]]
    durations: np.ndarray
    initial: tuple[int, ...] = (0,)
    step_names: tuple[str, ...] = ()
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels, dtype=np.int16)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "durations", np.asarray(self.durations, dtype=float))
        if not self.step_names:
            object.__setattr__(self, "step_names", tuple(f"S{i}" for i in range(self.n)))
        self.validate()

    # ------------------------------------------------------------------ structure
    @property
    def n(self) -> int:
        return int(self.labels.shape[0])

    @cached_property
    def adjacency(self) -> np.ndarray:
        return self.labels > 0

    @cached_property
    def edges(self) -> tuple[tuple[int, int, int], ...]:
        """All edges as (i, j, k)."""
        rows, cols = np.nonzero(self.labels)
        return tuple((int(i), int(j), int(self.labels[i, j])) for i, j in zip(rows, cols))

    @cached_property
    def successors(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(int(j) for j in np.flatnonzero(row)) for row in self.adjacency)

    @cached_property
    def predecessors(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(int(i) for i in np.flatnonzero(col)) for col in self.adjacency.T)

    @cached_property
    def divergence_groups(self) -> tuple[dict[int, Group], ...]:
        """For every step i: {k: set of successors in divergence group k}."""
        out: list[dict[int, Group]] = []
        for i in range(self.n):
            groups: dict[int, set[int]] = {}
            for j in self.successors[i]:
                groups.setdefault(int(self.labels[i, j]), set()).add(j)
            out.append({k: frozenset(v) for k, v in sorted(groups.items())})
        return tuple(out)

    def convergence_groups(self, j: int) -> tuple[Group, ...]:
        """Convergence groups of step j (explicit, or one singleton per predecessor)."""
        if j in self.convergence:
            return self.convergence[j]
        return tuple(frozenset({p}) for p in self.predecessors[j])

    def divergence_kind(self, i: int) -> str:
        groups = self.divergence_groups[i]
        if not groups:
            return "none"
        if len(groups) == 1:
            return "AND" if len(next(iter(groups.values()))) > 1 else "simple"
        return "OR" if all(len(g) == 1 for g in groups.values()) else "AND/OR"

    def convergence_kind(self, j: int) -> str:
        groups = self.convergence_groups(j)
        if len(self.predecessors[j]) <= 1:
            return "simple" if self.predecessors[j] else "none"
        if len(groups) == 1:
            return "AND"
        return "OR" if all(len(g) == 1 for g in groups) else "AND/OR"

    def symbol(self, i: int, j: int) -> str:
        """Plain-text name of the matrix entry X[i, j]."""
        k = int(self.labels[i, j])
        return f"delta_{i}_{j}^{k}" if k else "0"

    # ------------------------------------------------------------------ checks
    def validate(self) -> None:
        n = self.n
        if self.labels.ndim != 2 or self.labels.shape != (n, n):
            raise ModelError(f"{self.name}: matrix must be square, got {self.labels.shape}")
        if (self.labels < 0).any():
            raise ModelError(f"{self.name}: labels must be >= 0")
        if np.diag(self.labels).any():
            raise ModelError(f"{self.name}: self-loops are not allowed")
        if self.durations.shape != (n,) or (self.durations < 0).any():
            raise ModelError(f"{self.name}: need {n} non-negative durations")
        if len(self.step_names) != n:
            raise ModelError(f"{self.name}: need {n} step names")
        if not self.initial or any(not 0 <= s < n for s in self.initial):
            raise ModelError(f"{self.name}: invalid initial steps {self.initial}")
        for j, groups in self.convergence.items():
            preds = set(self.predecessors[j])
            covered: set[int] = set()
            for g in groups:
                if not g or not g <= preds:
                    raise ModelError(f"{self.name}: convergence group {sorted(g)} of step {j} "
                                     f"is not a subset of its predecessors {sorted(preds)}")
                covered |= g
            if covered != preds:
                raise ModelError(f"{self.name}: convergence groups of step {j} do not cover "
                                 f"predecessors {sorted(preds - covered)}")
        for i in range(n):
            if not self.successors[i]:
                raise ModelError(f"{self.name}: step {i} has no successor (dead end)")

    # ------------------------------------------------------------------ statistics
    def stats(self) -> dict[str, Any]:
        div = [self.divergence_kind(i) for i in range(self.n)]
        conv = [self.convergence_kind(j) for j in range(self.n)]
        return {
            "name": self.name,
            "steps": self.n,
            "edges": len(self.edges),
            "and_divergences": div.count("AND"),
            "or_divergences": div.count("OR") + div.count("AND/OR"),
            "and_convergences": conv.count("AND"),
            "or_convergences": conv.count("OR") + conv.count("AND/OR"),
            "max_out_degree": int(self.adjacency.sum(axis=1).max()),
            "max_in_degree": int(self.adjacency.sum(axis=0).max()),
            "tier": self.metadata.get("tier", "case study"),
        }

    # ------------------------------------------------------------------ (de)serialisation
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "n_steps": self.n,
            "initial": list(self.initial),
            "steps": [{"id": i, "name": self.step_names[i], "duration": float(self.durations[i])}
                      for i in range(self.n)],
            "matrix": self.labels.tolist(),
            "convergence": {str(j): [sorted(g) for g in groups]
                            for j, groups in sorted(self.convergence.items())},
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SFCModel":
        steps = data["steps"]
        return cls(
            name=data["name"],
            labels=np.array(data["matrix"], dtype=np.int16),
            convergence={int(j): tuple(frozenset(g) for g in groups)
                         for j, groups in data.get("convergence", {}).items()},
            durations=np.array([s["duration"] for s in steps], dtype=float),
            initial=tuple(data.get("initial", [0])),
            step_names=tuple(s["name"] for s in steps),
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SFCModel":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str | Path) -> None:
        """Write the model as JSON, one matrix row / step per line for readability."""
        data = self.to_dict()
        rows = ",\n    ".join(json.dumps(r, separators=(",", ":")) for r in data["matrix"])
        steps = ",\n    ".join(json.dumps(s) for s in data["steps"])
        text = json.dumps({**data, "steps": "@STEPS@", "matrix": "@MATRIX@"}, indent=2)
        text = (text.replace('"@STEPS@"', "[\n    " + steps + "\n  ]")
                    .replace('"@MATRIX@"', "[\n    " + rows + "\n  ]"))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text + "\n", encoding="utf-8")
