"""The bitmask verifier must agree with the literal SymPy matrix formulation."""

import random

import pytest

from sfcverify import catalog
from sfcverify.simulator import simulate
from sfcverify.symbolic import transition_matrix, verify_symbolic
from sfcverify.verifier import TransitionVerifier, from_mask


@pytest.mark.parametrize("name", ["CS_reference", "G002_small", "G030_medium", "G085_complex"])
def test_bitmask_matches_symbolic(name):
    model = catalog.load_graph(name)
    X = transition_matrix(model)
    verifier = TransitionVerifier(model)
    trace = simulate(model, 400, seed=7)
    rng = random.Random(1)
    pairs = [(p, c) for p, c in zip(trace.masks, trace.masks[1:]) if p != c][:40]
    for p, c in list(pairs):                       # corrupted variants
        for _ in range(2):
            pairs.append((p, c ^ (1 << rng.randrange(model.n))))
    for p, c in pairs:
        expected = verify_symbolic(model, X, from_mask(p, model.n), from_mask(c, model.n))
        assert verifier.is_valid(p, c) == expected, (name, bin(p), bin(c))
