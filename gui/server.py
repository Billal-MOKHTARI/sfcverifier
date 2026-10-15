"""HTTP backend of the GUI (standard library only).

Static files:  /            -> gui/static/index.html
               /static/...  -> gui/static
               /results/... -> results (figures, LaTeX tables, report)
JSON API:      GET  /api/graphs                    list of graphs
               GET  /api/graphs/<name>             model + layout levels
               POST /api/scenario                  simulate (+ optional attack) and verify
               POST /api/trace                     verify an imported CSV trace
               POST /api/check                     detailed verification of one transition
               GET  /api/evaluation                summary.json of the last evaluation
               POST /api/evaluation/run            start an evaluation in the background
               GET  /api/evaluation/status         progress of the running evaluation
"""

from __future__ import annotations

import json
import mimetypes
import os
import sys
import threading
import traceback
import webbrowser
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfcverify import catalog  # noqa: E402
from sfcverify.attacks import ATTACKS, Injector  # noqa: E402
from sfcverify.simulator import Simulator, simulate  # noqa: E402
from sfcverify.traces import parse_rows  # noqa: E402
from sfcverify.verifier import RULES, TransitionVerifier, bits, to_mask  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
MAX_SCANS = 20_000


# ---------------------------------------------------------------------- helpers
@lru_cache(maxsize=128)
def _verifier(name: str) -> TransitionVerifier:
    return TransitionVerifier(catalog.load_graph(name))


def _levels(name: str) -> list[int]:
    """Longest-path layer of every step over the forward (non-loop) edges, for the layout."""
    model = catalog.load_graph(name)
    back = Simulator(model).back
    level = [0] * model.n
    indeg = [0] * model.n
    for i, j, _ in model.edges:
        if (i, j) not in back:
            indeg[j] += 1
    queue = [i for i in range(model.n) if indeg[i] == 0]
    while queue:
        i = queue.pop()
        for j in model.successors[i]:
            if (i, j) in back:
                continue
            level[j] = max(level[j], level[i] + 1)
            indeg[j] -= 1
            if indeg[j] == 0:
                queue.append(j)
    return level


def graph_payload(name: str) -> dict:
    model = catalog.load_graph(name)
    back = Simulator(model).back
    return {
        "name": model.name,
        "description": model.description,
        "metadata": model.metadata,
        "stats": model.stats(),
        "n": model.n,
        "initial": list(model.initial),
        "steps": [{"id": i, "name": model.step_names[i], "duration": float(model.durations[i]),
                   "divergence": model.divergence_kind(i), "convergence": model.convergence_kind(i),
                   "divergence_groups": {str(k): sorted(g)
                                         for k, g in model.divergence_groups[i].items()},
                   "convergence_groups": [sorted(g) for g in model.convergence_groups(i)]}
                  for i in range(model.n)],
        "edges": [{"from": i, "to": j, "k": k, "back": (i, j) in back} for i, j, k in model.edges],
        "levels": _levels(name),
    }


def verify_masks(name: str, masks: list[int]) -> dict:
    verifier = _verifier(name)
    states, alarms, fired, violations = [], [], [], {}
    for t, m in enumerate(masks):
        states.append(list(bits(m)))
        prev = masks[t - 1] if t else m
        if t and prev == m and m:
            alarms.append(False)
            fired.append([])
            continue
        check = verifier.check(prev, m)
        alarms.append(not check.valid)
        fired.append([[i, j] for i, j, _ in check.fired])
        if not check.valid:
            violations[t] = [{"rule": v.rule, "step": v.step, "message": v.message}
                             for v in check.violations]
    return {"states": states, "alarms": alarms, "fired": fired, "violations": violations}


def scenario(body: dict) -> dict:
    name = body["graph"]
    model = catalog.load_graph(name)
    scans = max(50, min(int(body.get("scans", 600)), MAX_SCANS))
    trace = simulate(model, scans, seed=int(body.get("seed", 0)), jitter=float(body.get("jitter", 0.2)))
    masks = list(trace.masks)
    attack = None
    if body.get("attack"):
        if scans < 80:
            raise ValueError("an attack needs at least 80 scans")
        ep = Injector(model, trace, seed=int(body.get("attack_seed", 1))).inject(body["attack"])
        if ep is None:
            raise ValueError(f"attack '{body['attack']}' is not applicable to {name} "
                             "(the graph has no matching structure in this trace)")
        masks[ep.start - 1: ep.horizon + 1] = ep.segment
        spec = ATTACKS[ep.attack]
        attack = {"name": spec.name, "title": spec.title, "family": spec.family,
                  "description": spec.description, "start": ep.start, "end": ep.end,
                  "horizon": ep.horizon}
    return {**verify_masks(name, masks), "attack": attack, "dt": trace.dt}


def check(body: dict) -> dict:
    name = body["graph"]
    model = catalog.load_graph(name)
    prev = to_mask([i in set(body["prev"]) for i in range(model.n)])
    curr = to_mask([i in set(body["curr"]) for i in range(model.n)])
    c = _verifier(name).check(prev, curr)
    return {"valid": c.valid, "left": c.left, "entered": c.entered,
            "fired": [list(f) for f in c.fired],
            "violations": [{"rule": v.rule, "step": v.step, "message": v.message}
                           for v in c.violations],
            "divergence_match": {str(k): v for k, v in c.divergence_match.items()},
            "convergence_match": {str(k): v for k, v in c.convergence_match.items()}}


# ---------------------------------------------------------------------- evaluation job
class EvaluationJob:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.state = {"running": False, "done": 0, "total": 0, "current": "", "error": None,
                      "profile": None}

    def start(self, profile: str) -> dict:
        with self.lock:
            if self.state["running"]:
                return self.status()
            self.state.update(running=True, done=0, total=len(catalog.graph_paths()), current="",
                              error=None, profile=profile)
        threading.Thread(target=self._run, args=(profile,), daemon=True).start()
        return self.status()

    def _run(self, profile: str) -> None:
        from sfcverify.evaluation import EvalConfig, run
        from sfcverify.report import build

        def progress(done: int, total: int, current: str) -> None:
            with self.lock:
                self.state.update(done=done, total=total, current=current)
        try:
            summary = run(EvalConfig.from_profile(profile), progress=progress)
            with self.lock:
                self.state["current"] = "building figures"
            build(summary)
        except Exception as exc:  # reported to the GUI
            traceback.print_exc()
            with self.lock:
                self.state["error"] = str(exc)
        finally:
            with self.lock:
                self.state["running"] = False

    def status(self) -> dict:
        with self.lock:
            return dict(self.state)


JOB = EvaluationJob()


# ---------------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "sfcverify-gui"

    def log_message(self, *args) -> None:  # quiet
        pass

    def _json(self, payload, status: int = 200) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path: Path) -> None:
        if not path.is_file():
            return self._json({"error": "not found"}, 404)
        data = path.read_bytes()
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix in (".tex", ".md"):
            ctype = "text/plain; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    @staticmethod
    def _safe(base: Path, rel: str) -> Path:
        path = (base / rel).resolve()
        if base.resolve() not in path.parents and path != base.resolve():
            raise PermissionError(rel)
        return path

    def do_GET(self) -> None:
        url = urlparse(self.path)
        path = unquote(url.path)
        try:
            if path in ("/", "/index.html"):
                return self._file(STATIC / "index.html")
            if path.startswith("/static/"):
                return self._file(self._safe(STATIC, path[len("/static/"):]))
            if path.startswith("/results/"):
                return self._file(self._safe(catalog.RESULTS_DIR, path[len("/results/"):]))
            if path == "/api/graphs":
                return self._json([{**catalog.load_graph(n).stats(),
                                    "description": catalog.load_graph(n).description}
                                   for n in catalog.graph_names()])
            if path.startswith("/api/graphs/"):
                return self._json(graph_payload(path.rsplit("/", 1)[1]))
            if path == "/api/meta":
                return self._json({"rules": RULES,
                                   "attacks": {k: vars(v) for k, v in ATTACKS.items()}})
            if path == "/api/evaluation":
                f = catalog.RESULTS_DIR / "summary.json"
                return self._json(json.loads(f.read_text()) if f.exists() else {"available": False})
            if path == "/api/evaluation/status":
                return self._json(JOB.status())
            self._json({"error": "not found"}, 404)
        except FileNotFoundError:
            self._json({"error": "not found"}, 404)
        except PermissionError:
            self._json({"error": "forbidden"}, 403)

    def do_POST(self) -> None:
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            path = urlparse(self.path).path
            if path == "/api/scenario":
                return self._json(scenario(body))
            if path == "/api/trace":
                masks, _ = parse_rows([r.split(",") for r in body["csv"].strip().splitlines()])
                return self._json({**verify_masks(body["graph"], masks), "attack": None, "dt": 1.0})
            if path == "/api/check":
                return self._json(check(body))
            if path == "/api/evaluation/run":
                return self._json(JOB.start(body.get("profile", "full")))
            self._json({"error": "not found"}, 404)
        except (KeyError, ValueError, IndexError) as exc:
            self._json({"error": str(exc)}, 400)


def serve(port: int | None = None, open_browser: bool | None = None) -> None:
    """Serve the GUI.

    Without an explicit port, PORT and HOST from the environment win, so the
    server also runs behind a platform that assigns its own port; it then binds
    every interface and skips opening a browser.
    """
    env_port = os.environ.get("PORT")
    managed = port is None and bool(env_port)
    port = port if port is not None else int(env_port or 8765)
    host = os.environ.get("HOST") or ("0.0.0.0" if managed else "127.0.0.1")
    if open_browser is None:
        open_browser = not managed
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"SFC verifier GUI on {url}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        webbrowser.open(f"http://127.0.0.1:{port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    serve()
