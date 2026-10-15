import numpy as np
import pytest

from sfcverify import catalog
from sfcverify.model import ModelError, SFCModel


def test_all_graphs_load():
    names = catalog.graph_names()
    assert names[0] == "CS_reference"
    assert len([n for n in names if n.startswith("G")]) == 100
    for name in names:
        model = catalog.load_graph(name)
        assert model.n == model.labels.shape[0]


def test_round_trip(tmp_path):
    model = catalog.load_graph("G090_complex")
    model.save(tmp_path / "g.json")
    again = SFCModel.load(tmp_path / "g.json")
    assert np.array_equal(model.labels, again.labels)
    assert model.convergence == again.convergence


def test_reference_structure():
    m = catalog.load_graph("CS_reference")
    assert m.divergence_kind(0) == "AND"          # 0 -> {1, 6}
    assert m.divergence_kind(10) == "OR"          # 10 -> 11 (k=1) | 15 (k=2)
    assert m.convergence_kind(25) == "AND"        # {22, 23, 24} -> 25
    assert set(m.convergence_groups(3)) == {frozenset({2, 9}), frozenset({2})}


def test_invalid_convergence_group_rejected():
    labels = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    with pytest.raises(ModelError):
        SFCModel("bad", labels, {2: (frozenset({0}),)}, np.ones(3))
