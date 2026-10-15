import pytest

from sfcverify import catalog
from sfcverify.verifier import TransitionVerifier, to_mask


def mask(*steps, n=26):
    return to_mask([i in steps for i in range(n)])


@pytest.fixture(scope="module")
def ref():
    return TransitionVerifier(catalog.load_graph("CS_reference"))


def rules(v, prev, curr):
    return {x.rule for x in v.check(prev, curr).violations}


def test_and_divergence_with_persistent_step(ref):
    # 6 leaves, 7 and 10 enter, 1 keeps running: the case the E1^T + E2 operator rejected
    assert ref.is_valid(mask(1, 6), mask(1, 7, 10))
    c = ref.check(mask(1, 6), mask(1, 7, 10))
    assert c.left == [6] and c.entered == [7, 10]
    assert sorted(c.fired) == [(6, 7, 1), (6, 10, 1)]


def test_stuttering_is_valid(ref):
    assert ref.is_valid(mask(19, 20, 21), mask(19, 20, 21))


def test_and_convergence(ref):
    assert ref.is_valid(mask(8, 11), mask(9, 12))
    assert rules(ref, mask(8, 21), mask(9, 21)) == {"R4"}          # 11 still missing


def test_or_exclusivity(ref):
    assert rules(ref, mask(10), mask(11, 15)) == {"R3"}


def test_partial_and_divergence(ref):
    assert rules(ref, mask(6), mask(7)) == {"R3"}


def test_jump_deadlock_and_lost_token(ref):
    assert "R2" in rules(ref, mask(1), mask(3))
    assert "R1" in rules(ref, mask(1, 6), mask())
    assert rules(ref, mask(19, 20, 21), mask(19, 20)) == {"R3"}


def test_ablations():
    model = catalog.load_graph("CS_reference")
    adjacency = TransitionVerifier(model, "adjacency")
    no_div = TransitionVerifier(model, "no_divergence")
    no_conv = TransitionVerifier(model, "no_convergence")
    assert adjacency.is_valid(mask(10), mask(11, 15))      # OR violation needs R3
    assert no_div.is_valid(mask(6), mask(7))               # partial AND needs R3
    assert no_conv.is_valid(mask(8, 21), mask(9, 21))      # premature join needs R4
    assert not TransitionVerifier(model).is_valid(mask(8, 21), mask(9, 21))
