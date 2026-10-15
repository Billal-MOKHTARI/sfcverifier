import pytest

from sfcverify import catalog
from sfcverify.attacks import ATTACKS, Injector
from sfcverify.simulator import simulate, stalled
from sfcverify.verifier import TransitionVerifier


@pytest.mark.parametrize("name", catalog.graph_names())
def test_no_false_alarm_on_normal_operation(name):
    model = catalog.load_graph(name)
    trace = simulate(model, 3_000, seed=11)
    assert not stalled(trace, 300), "simulation deadlocked"
    assert not TransitionVerifier(model).alarms(trace.masks).any()


@pytest.mark.parametrize("name", ["CS_reference", "G040_medium", "G095_complex"])
def test_attack_families(name):
    model = catalog.load_graph(name)
    trace = simulate(model, 3_000, seed=5)
    verifier = TransitionVerifier(model)
    for attack, spec in ATTACKS.items():
        ep = Injector(model, trace, seed=3).inject(attack)
        if ep is None:
            continue
        alarms = verifier.alarms(ep.segment)[1:]
        if spec.family == "violating":
            assert alarms.any(), attack
        elif attack in ("delay", "wrong_branch"):
            assert not alarms.any(), attack          # legal executions: undetectable by design
