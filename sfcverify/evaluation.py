"""Experimental protocol.

For every graph (the reference case study and the 100 benchmark graphs):

  1. Simulate a clean *test* trace and an independent clean *training* trace
     (different seeds, timing jitter so that parallel branches interleave differently).
  2. False alarms: run every detector on the clean test trace.
  3. Detection: inject ``episodes`` episodes of every attack class into the test trace and
     record whether (and how fast) each detector raises an alarm within the episode horizon.
     "Onset" detection means the alarm is raised on the very first manipulated scan.
  4. Rule attribution: for the full verifier, record which rules fired at the first alarm.
  5. Cost: time the verifier on all state-changing transitions of the test trace.

Detectors: the full verifier (R1-R4), three ablations (no divergence semantics, no
convergence semantics, adjacency only) and learned whitelists trained on 1k-100k scans of
normal operation (see ``baselines``).
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from .attacks import ATTACKS, Injector
from .baselines import StateWhitelist, TransitionWhitelist
from .catalog import GRAPH_DIR, RESULTS_DIR, graph_paths
from .model import SFCModel
from .simulator import simulate
from .verifier import PRESETS, TransitionVerifier

Detector = Callable[[int, int], bool]


@dataclass(frozen=True)
class EvalConfig:
    profile: str = "full"
    episodes: int = 40                 # episodes per attack class and graph
    test_scans: int = 50_000
    train_scans: tuple[int, ...] = (1_000, 5_000, 20_000, 100_000)
    dt: float = 1.0
    jitter: float = 0.2
    workers: int = field(default_factory=lambda: max(1, min(16, (os.cpu_count() or 2) - 2)))

    @classmethod
    def from_profile(cls, profile: str, **overrides) -> "EvalConfig":
        presets = {
            "full": {},
            "quick": {"episodes": 8, "test_scans": 3_000, "train_scans": (500, 2_000, 8_000)},
        }
        return cls(profile=profile, **{**presets[profile], **overrides})


DETECTOR_LABELS = {
    "phi_full": "Phi verifier (R1-R4)",
    "phi_no_divergence": "Ablation: no divergence semantics",
    "phi_no_convergence": "Ablation: no convergence semantics",
    "phi_adjacency": "Ablation: adjacency only",
}


def detector_labels(cfg: EvalConfig) -> dict[str, str]:
    labels = dict(DETECTOR_LABELS)
    for n in cfg.train_scans:
        labels[f"wl_transition_{n}"] = f"Transition whitelist ({n:,} scans)"
    labels[f"wl_state_{cfg.train_scans[-1]}"] = f"State whitelist ({cfg.train_scans[-1]:,} scans)"
    return labels


# ---------------------------------------------------------------------- one graph
def evaluate_graph(path: str, index: int, cfg: EvalConfig) -> dict:
    model = SFCModel.load(path)
    test = simulate(model, cfg.test_scans, seed=10_000 + index, dt=cfg.dt, jitter=cfg.jitter)
    train = simulate(model, max(cfg.train_scans), seed=20_000 + index, dt=cfg.dt, jitter=cfg.jitter)

    verifiers = {f"phi_{k}": TransitionVerifier(model, vc) for k, vc in PRESETS.items()}
    detectors: dict[str, Detector] = {k: v.is_valid for k, v in verifiers.items()}
    for n in cfg.train_scans:
        detectors[f"wl_transition_{n}"] = TransitionWhitelist(train.masks[:n]).is_valid
    detectors[f"wl_state_{cfg.train_scans[-1]}"] = StateWhitelist(train.masks).is_valid

    masks = test.masks
    changes = [(p, c) for p, c in zip(masks, masks[1:]) if p != c]
    false_alarms = {name: sum(not det(p, c) for p, c in zip(masks, masks[1:]))
                    for name, det in detectors.items()}

    full = verifiers["phi_full"]
    sample = changes[:5_000]
    best = math.inf
    for _ in range(5):                       # minimum of repeated runs: robust to CPU contention
        start = time.perf_counter()
        for p, c in sample:
            full.is_valid(p, c)
        best = min(best, time.perf_counter() - start)
    timing_us = best / max(1, len(sample)) * 1e6

    episodes: dict[str, dict] = {}
    for a_idx, attack in enumerate(ATTACKS):
        inj = Injector(model, test, seed=30_000 + 100 * index + a_idx)
        rec = {"count": 0, "detected": {d: 0 for d in detectors},
               "latency": {d: [] for d in detectors}, "rules": {}}
        for _ in range(cfg.episodes):
            ep = inj.inject(attack)
            if ep is None:
                break
            rec["count"] += 1
            seg = ep.segment
            for name, det in detectors.items():
                for t in range(1, len(seg)):
                    if not det(seg[t - 1], seg[t]):
                        rec["detected"][name] += 1
                        rec["latency"][name].append((t - 1) * cfg.dt)
                        if name == "phi_full":
                            for rule in {v.rule for v in full.check(seg[t - 1], seg[t]).violations}:
                                rec["rules"][rule] = rec["rules"].get(rule, 0) + 1
                        break
        episodes[attack] = rec

    active = [bin(m).count("1") for m in masks]
    stats = model.stats() | {"max_concurrency": max(active),
                             "mean_concurrency": round(statistics.fmean(active), 3),
                             "structure": model.metadata.get("structure", "")}
    return {"graph": model.name, "index": index, "stats": stats, "test_scans": len(masks),
            "test_changes": len(changes), "false_alarms": false_alarms,
            "timing_us": timing_us, "episodes": episodes}


# ---------------------------------------------------------------------- statistics
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "q1": None, "q3": None, "mean": None}
    v = sorted(values)
    q = statistics.quantiles(v, n=4) if len(v) > 1 else [v[0]] * 3
    return {"median": statistics.median(v), "q1": q[0], "q3": q[2], "mean": statistics.fmean(v)}


def aggregate(results: list[dict], cfg: EvalConfig) -> dict:
    bench = [r for r in results if not r["graph"].startswith("CS_")]
    cases = [r for r in results if r["graph"].startswith("CS_")]
    detectors = list(results[0]["false_alarms"])

    def detection(rows: list[dict]) -> dict:
        out: dict[str, dict] = {}
        for d in detectors:
            out[d] = {}
            for attack, spec in ATTACKS.items():
                n = sum(r["episodes"][attack]["count"] for r in rows)
                k = sum(r["episodes"][attack]["detected"][d] for r in rows)
                lat = [x for r in rows for x in r["episodes"][attack]["latency"][d]]
                onset = sum(1 for x in lat if x == 0)
                lo, hi = wilson(k, n)
                out[d][attack] = {"episodes": n, "detected": k, "rate": k / n if n else None,
                                  "ci": [lo, hi], "onset_rate": onset / n if n else None,
                                  "latency": _quantiles(lat),
                                  "graphs": sum(r["episodes"][attack]["count"] > 0 for r in rows)}
            for fam in ("violating", "stealthy"):
                n = sum(out[d][a]["episodes"] for a, s in ATTACKS.items() if s.family == fam)
                k = sum(out[d][a]["detected"] for a, s in ATTACKS.items() if s.family == fam)
                k0 = sum(out[d][a]["onset_rate"] * out[d][a]["episodes"]
                         for a, s in ATTACKS.items() if s.family == fam and out[d][a]["episodes"])
                out[d][f"family:{fam}"] = {"episodes": n, "detected": k,
                                           "rate": k / n if n else None, "ci": list(wilson(k, n)),
                                           "onset_rate": k0 / n if n else None}
        return out

    def false_alarms(rows: list[dict]) -> dict:
        out = {}
        for d in detectors:
            scans = sum(r["test_scans"] - 1 for r in rows)
            k = sum(r["false_alarms"][d] for r in rows)
            upper = 1 - 0.05 ** (1 / scans) if k == 0 else wilson(k, scans)[1]
            out[d] = {"transitions": scans, "alarms": k, "rate": k / scans, "upper95": upper,
                      "per_hour": k / (scans * cfg.dt / 3600),
                      "graphs_with_alarms": sum(r["false_alarms"][d] > 0 for r in rows)}
        return out

    rules = {}
    for attack in ATTACKS:
        counts: dict[str, int] = {}
        for r in bench:
            for rule, c in r["episodes"][attack]["rules"].items():
                counts[rule] = counts.get(rule, 0) + c
        det = sum(r["episodes"][attack]["detected"]["phi_full"] for r in bench)
        rules[attack] = {rule: counts.get(rule, 0) / det if det else None
                         for rule in ("R1", "R2", "R3", "R4")}

    tiers: dict[str, list[dict]] = {}
    for r in bench:
        tiers.setdefault(r["stats"]["tier"], []).append(r)
    tier_stats = {}
    for tier, rows in tiers.items():
        def rng(key: str) -> dict:
            v = [r["stats"][key] for r in rows]
            return {"min": min(v), "median": statistics.median(v), "max": max(v)}
        viol = detection(rows)["phi_full"]["family:violating"]
        tier_stats[tier] = {"graphs": len(rows), "steps": rng("steps"), "edges": rng("edges"),
                            "and_divergences": rng("and_divergences"),
                            "or_divergences": rng("or_divergences"),
                            "and_convergences": rng("and_convergences"),
                            "or_convergences": rng("or_convergences"),
                            "max_concurrency": rng("max_concurrency"),
                            "violating_detection": viol["rate"]}

    return {
        "meta": {"generated": datetime.now().isoformat(timespec="seconds"),
                 "config": asdict(cfg), "graphs": len(bench), "case_studies": len(cases),
                 "episodes": sum(r["episodes"][a]["count"] for r in results for a in ATTACKS),
                 "transitions": sum(r["test_scans"] - 1 for r in results),
                 "detectors": detector_labels(cfg),
                 "attacks": {a: asdict(s) for a, s in ATTACKS.items()}},
        "benchmark": {"tiers": tier_stats, "detection": detection(bench),
                      "false_alarms": false_alarms(bench), "rules": rules},
        "case_studies": {r["graph"]: {"detection": detection([r]), "false_alarms": false_alarms([r]),
                                      "stats": r["stats"]} for r in cases},
        "graphs": [{"graph": r["graph"], **r["stats"], "test_changes": r["test_changes"],
                    "timing_us": r["timing_us"],
                    "false_alarms_full": r["false_alarms"]["phi_full"],
                    "false_alarms_wl": r["false_alarms"][f"wl_transition_{cfg.train_scans[-1]}"],
                    "violating_detected": sum(r["episodes"][a]["detected"]["phi_full"]
                                              for a, s in ATTACKS.items() if s.family == "violating"),
                    "violating_episodes": sum(r["episodes"][a]["count"]
                                              for a, s in ATTACKS.items() if s.family == "violating"),
                    "stealthy_detected": sum(r["episodes"][a]["detected"]["phi_full"]
                                             for a, s in ATTACKS.items() if s.family == "stealthy"),
                    "stealthy_episodes": sum(r["episodes"][a]["count"]
                                             for a, s in ATTACKS.items() if s.family == "stealthy")}
                   for r in results],
    }


# ---------------------------------------------------------------------- driver
def run(cfg: EvalConfig, graph_dir: Path = GRAPH_DIR, out_dir: Path = RESULTS_DIR,
        progress: Callable[[int, int, str], None] | None = None) -> dict:
    paths = graph_paths(graph_dir)
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
        futures = {pool.submit(evaluate_graph, str(p), i, cfg): p for i, p in enumerate(paths)}
        for done, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if progress:
                progress(done, len(paths), futures[fut].stem)
    results.sort(key=lambda r: r["index"])
    summary = aggregate(results, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw_results.json").write_text(json.dumps(results), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary
