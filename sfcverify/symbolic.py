"""Literal SymPy implementation of the verification rules (reference / documentation).

This module follows the matrix formulas word for word and is used by the test-suite to
check that the fast bitmask implementation in ``verifier.py`` computes the same verdicts.
It is far too slow for monitoring and is not used at runtime.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from .model import SFCModel


def transition_matrix(model: SFCModel) -> sp.Matrix:
    """X with X[i, j] = delta_ij^(k)."""
    n = model.n
    return sp.Matrix(n, n, lambda i, j: sp.Symbol(f"delta_{i}_{j}^{model.labels[i, j]}")
                     if model.labels[i, j] else 0)


def verify_symbolic(model: SFCModel, X: sp.Matrix, s1: np.ndarray, s2: np.ndarray) -> bool:
    n = model.n
    s1 = np.asarray(s1, dtype=int)
    s2 = np.asarray(s2, dtype=int)
    left = s1 * (1 - s2)
    entered = s2 * (1 - s1)

    E = X * sp.diag(*[int(v) for v in s2])                       # constraint matrix E(t)
    Phi = sp.diag(*[int(v) for v in left]) * E * sp.diag(*[int(v) for v in entered])
    row_sums = Phi * sp.ones(n, 1)                               # Phi 1
    col_sums = sp.ones(1, n) * Phi                               # 1^T Phi

    if s2.sum() == 0:                                            # R1
        return False
    for j in np.flatnonzero(entered):
        if col_sums[j] == 0:                                     # R2
            return False
        polys = [sum(X[p, j] for p in g) for g in model.convergence_groups(int(j))]
        if col_sums[j] not in polys:                             # R4
            return False
    for i in np.flatnonzero(left):
        groups = [sum(X[i, j] for j in g) for g in model.divergence_groups[int(i)].values()]
        if sum(row_sums[i] == g for g in groups) != 1:           # R3
            return False
    return True
