"""Access to the graph files in ``data/graphs``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .model import SFCModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRAPH_DIR = PROJECT_ROOT / "data" / "graphs"
RESULTS_DIR = PROJECT_ROOT / "results"


def graph_paths(directory: Path = GRAPH_DIR) -> list[Path]:
    """Graph files, the reference case study(ies) first, then the benchmark in order."""
    paths = sorted(directory.glob("*.json"))
    return sorted(paths, key=lambda p: (not p.stem.startswith("CS_"), p.stem))


@lru_cache(maxsize=256)
def load_graph(name: str, directory: Path = GRAPH_DIR) -> SFCModel:
    return SFCModel.load(directory / f"{name}.json")


def graph_names(directory: Path = GRAPH_DIR) -> list[str]:
    return [p.stem for p in graph_paths(directory)]
