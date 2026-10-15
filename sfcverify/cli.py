"""Command line interface: ``python -m sfcverify <command>``."""

from __future__ import annotations

import argparse
import sys

from . import catalog
from .evaluation import EvalConfig, run
from .generator import benchmark
from .simulator import simulate
from .traces import read_csv, write_csv
from .verifier import TransitionVerifier


def _graphs(args: argparse.Namespace) -> int:
    if args.regenerate:
        for g in benchmark(args.seed):
            g.save(catalog.GRAPH_DIR / f"{g.name}.json")
        print(f"wrote 100 benchmark graphs to {catalog.GRAPH_DIR}")
    for name in catalog.graph_names():
        s = catalog.load_graph(name).stats()
        print(f"{name:20s} {s['tier']:10s} steps={s['steps']:4d} edges={s['edges']:4d} "
              f"AND={s['and_divergences']:3d} OR={s['or_divergences']:3d}")
    return 0


def _simulate(args: argparse.Namespace) -> int:
    model = catalog.load_graph(args.graph)
    trace = simulate(model, args.scans, seed=args.seed, dt=args.dt, jitter=args.jitter)
    write_csv(args.out, trace.masks, model.n, dt=args.dt)
    print(f"wrote {len(trace)} scans to {args.out}")
    return 0


def _verify(args: argparse.Namespace) -> int:
    model = catalog.load_graph(args.graph)
    masks, stamps = read_csv(args.csv)
    verifier = TransitionVerifier(model)
    alarms = 0
    for t in range(len(masks)):
        prev = masks[t - 1] if t else masks[0]
        check = verifier.check(prev, masks[t])
        if not check.valid:
            alarms += 1
            for v in check.violations:
                print(f"{stamps[t]}  {v.rule}  {v.message}")
    print(f"{alarms} invalid transitions out of {len(masks) - 1}")
    return 1 if alarms else 0


def _evaluate(args: argparse.Namespace) -> int:
    from .report import build

    overrides = {"workers": args.workers} if args.workers else {}
    cfg = EvalConfig.from_profile(args.profile, **overrides)

    def progress(done: int, total: int, name: str) -> None:
        print(f"\r[{done:3d}/{total}] {name:24s}", end="", flush=True)

    summary = run(cfg, progress=progress)
    print()
    build(summary)
    print(f"results written to {catalog.RESULTS_DIR}")
    return 0


def _report(_: argparse.Namespace) -> int:
    from .report import build

    build()
    print(f"report rebuilt in {catalog.RESULTS_DIR}")
    return 0


def _gui(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(catalog.PROJECT_ROOT))
    from gui.server import serve

    serve(port=args.port, open_browser=not args.no_browser)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sfcverify", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("graphs", help="list the graphs (optionally regenerate the benchmark)")
    p.add_argument("--regenerate", action="store_true")
    p.add_argument("--seed", type=int, default=2026)
    p.set_defaults(func=_graphs)

    p = sub.add_parser("simulate", help="simulate a normal trace and write it as CSV")
    p.add_argument("graph")
    p.add_argument("--scans", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--jitter", type=float, default=0.2)
    p.add_argument("--out", default="trace.csv")
    p.set_defaults(func=_simulate)

    p = sub.add_parser("verify", help="verify a CSV trace (id, timestamp, 0..n-1)")
    p.add_argument("graph")
    p.add_argument("csv")
    p.set_defaults(func=_verify)

    p = sub.add_parser("evaluate", help="run the evaluation on all graphs and build the report")
    p.add_argument("--profile", choices=["quick", "full"], default="full")
    p.add_argument("--workers", type=int, default=0)
    p.set_defaults(func=_evaluate)

    p = sub.add_parser("report", help="rebuild figures and tables from results/summary.json")
    p.set_defaults(func=_report)

    p = sub.add_parser("gui", help="start the web interface")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(func=_gui)

    args = parser.parse_args(argv)
    return args.func(args)
