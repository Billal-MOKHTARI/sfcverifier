"""Paper-ready figures (SVG + PDF), LaTeX tables and a Markdown report from summary.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .attacks import ATTACKS  # noqa: E402
from .catalog import RESULTS_DIR  # noqa: E402

# categorical slots in fixed order (validated palette: blue, orange, aqua, yellow)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TIER_ORDER = ["small", "medium", "large", "complex"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK_2,
    "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
})


def _save(fig: plt.Figure, out: Path, name: str) -> None:
    fig.tight_layout()
    for ext in ("svg", "pdf"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def _last_wl(summary: dict) -> tuple[str, str]:
    n = summary["meta"]["config"]["train_scans"][-1]
    return f"wl_transition_{n}", f"wl_state_{n}"


# ---------------------------------------------------------------------- figures
def fig_detection(summary: dict, out: Path) -> None:
    det = summary["benchmark"]["detection"]
    labels = summary["meta"]["detectors"]
    wl_t, wl_s = _last_wl(summary)
    attacks = list(ATTACKS)
    titles = [ATTACKS[a].title for a in attacks]
    n_violating = sum(s.family == "violating" for s in ATTACKS.values())
    panels = [
        ("(a) Detected within the attack horizon", "rate", ["phi_full", wl_t, wl_s]),
        ("(b) Detected on the first manipulated scan", "onset_rate",
         ["phi_full", "phi_no_divergence", "phi_no_convergence", "phi_adjacency"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    for ax, (title, key, dets) in zip(axes, panels):
        h = 0.84 / len(dets)
        for s, d in enumerate(dets):
            ys = [i + (s - (len(dets) - 1) / 2) * h for i in range(len(attacks))]
            vals = [100 * (det[d][a][key] or 0) for a in attacks]
            ax.barh(ys, vals, height=h - 0.03, color=SERIES[s], label=labels[d])
        ax.set_yticks(range(len(attacks)), titles)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Episodes detected (%)")
        ax.set_title(title, loc="left", fontsize=9.5, color=INK, pad=34)
        ax.axhline(n_violating - 0.5, color=INK_2, lw=0.8, ls=(0, (3, 3)))
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=7.5,
                  handlelength=1.2, columnspacing=1.0)
    axes[1].text(2, n_violating + 1, "structure-preserving (stealthy) attacks:\nno detector fires on the first scan",
                 fontsize=7.5, color=INK_2, va="center")
    axes[1].text(2, n_violating - 0.5, "structure-violating \u2193", fontsize=7.5, color=INK_2,
                 va="center", bbox={"fc": "white", "ec": "none", "pad": 1.5})
    _save(fig, out, "fig_detection")


def fig_false_alarms(summary: dict, out: Path) -> None:
    fa = summary["benchmark"]["false_alarms"]
    sizes = summary["meta"]["config"]["train_scans"]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    wl = [fa[f"wl_transition_{n}"]["per_hour"] for n in sizes]
    ax.plot(sizes, wl, marker="o", color=SERIES[1], lw=2, ms=6, label="Transition whitelist (learned)")
    ax.plot(sizes, [fa["phi_full"]["per_hour"]] * len(sizes), marker="o", color=SERIES[0], lw=2,
            ms=6, label="Phi verifier (training-free)")
    ax.set_xscale("log")
    ax.set_xlabel("Training length (scans of normal operation)")
    ax.set_ylabel("False alarms per hour")
    ax.set_ylim(bottom=-max(wl) * 0.04)
    for x, y in zip(sizes, wl):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7.5, color=INK_2)
    ax.annotate(f"{fa['phi_full']['alarms']} alarms", (sizes[-1], 0), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=7.5, color=INK_2)
    ax.legend(fontsize=8)
    _save(fig, out, "fig_false_alarms")


def fig_scalability(summary: dict, out: Path) -> None:
    rows = [g for g in summary["graphs"] if not g["graph"].startswith("CS_")]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
    for c, tier in enumerate(TIER_ORDER):
        sub = [g for g in rows if g["tier"] == tier]
        axes[0].scatter([g["steps"] for g in sub], [g["timing_us"] for g in sub], s=22,
                        color=SERIES[c], edgecolor="white", linewidth=0.8, label=tier)
        axes[1].scatter([g["steps"] for g in sub], [g["max_concurrency"] for g in sub], s=22,
                        color=SERIES[c], edgecolor="white", linewidth=0.8, label=tier)
    axes[0].set_xlabel("Steps |N|")
    axes[0].set_ylabel("Verification time per transition (µs)")
    axes[0].set_ylim(bottom=0)
    axes[0].set_title("(a) Runtime cost", loc="left", fontsize=9.5)
    axes[1].set_xlabel("Steps |N|")
    axes[1].set_ylabel("Max. simultaneously active steps")
    axes[1].set_title("(b) Concurrency of the benchmark", loc="left", fontsize=9.5)
    axes[1].legend(title="tier", fontsize=8, title_fontsize=8)
    _save(fig, out, "fig_scalability")


def fig_rules(summary: dict, out: Path) -> None:
    rules = summary["benchmark"]["rules"]
    attacks = [a for a, s in ATTACKS.items() if s.family == "violating"]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    h = 0.2
    for s, rule in enumerate(["R1", "R2", "R3", "R4"]):
        ys = [i + (s - 1.5) * h for i in range(len(attacks))]
        ax.barh(ys, [100 * (rules[a][rule] or 0) for a in attacks], height=h - 0.03,
                color=SERIES[s], label=rule)
    ax.set_yticks(range(len(attacks)), [ATTACKS[a].title for a in attacks])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Detected episodes in which the rule fires at the first alarm (%)")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.38), fontsize=8)
    _save(fig, out, "fig_rules")


# ---------------------------------------------------------------------- tables
def _pct(x: float | None, digits: int = 1) -> str:
    return "--" if x is None else f"{100 * x:.{digits}f}"


def tab_benchmark(summary: dict) -> str:
    tiers = summary["benchmark"]["tiers"]

    def r(v: dict) -> str:
        return f"{v['min']}--{v['max']} ({v['median']:g})"
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Synthetic SFC benchmark (min--max, median in parentheses).}",
             r"\label{tab:benchmark}", r"\small", r"\begin{tabular}{lrlllll}", r"\toprule",
             r"Tier & Graphs & Steps & Edges & AND div. & OR div. & Max. active \\", r"\midrule"]
    for t in TIER_ORDER:
        if t in tiers:
            v = tiers[t]
            lines.append(f"{t} & {v['graphs']} & {r(v['steps'])} & {r(v['edges'])} & "
                         f"{r(v['and_divergences'])} & {r(v['or_divergences'])} & "
                         f"{r(v['max_concurrency'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def tab_detection(summary: dict) -> str:
    det = summary["benchmark"]["detection"]
    wl_t, wl_s = _last_wl(summary)
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Detection on the 100-graph benchmark. DR: episodes detected within the "
             r"horizon (95\% Wilson interval); onset: detected on the first manipulated scan; "
             r"latency in scans (median).}",
             r"\label{tab:detection}", r"\small", r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
             r" & & & \multicolumn{3}{c}{$\Phi$ verifier} & \multicolumn{3}{c}{Onset, ablations} "
             r"& WL \\", r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
             r"Attack & Family & Ep. & DR (\%) & Onset & Lat. & no div. & no conv. & adj. & DR \\",
             r"\midrule"]
    for a, s in ATTACKS.items():
        f = det["phi_full"][a]
        lat = f["latency"]["median"]
        lines.append(
            f"{s.title} & {s.family} & {f['episodes']} & {_pct(f['rate'])} "
            f"[{_pct(f['ci'][0])}, {_pct(f['ci'][1])}] & {_pct(f['onset_rate'])} & "
            f"{'--' if lat is None else f'{lat:g}'} & {_pct(det['phi_no_divergence'][a]['onset_rate'])} & "
            f"{_pct(det['phi_no_convergence'][a]['onset_rate'])} & "
            f"{_pct(det['phi_adjacency'][a]['onset_rate'])} & {_pct(det[wl_t][a]['rate'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def tab_false_alarms(summary: dict) -> str:
    fa = summary["benchmark"]["false_alarms"]
    labels = summary["meta"]["detectors"]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{False alarms on clean traces of the 100 benchmark graphs.}",
             r"\label{tab:false-alarms}", r"\small", r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Detector & Transitions & Alarms & FPR (95\% upper) & Alarms/h \\", r"\midrule"]
    for d, v in fa.items():
        lines.append(f"{labels[d]} & {v['transitions']:,} & {v['alarms']:,} & "
                     f"{v['rate']:.2e} ({v['upper95']:.1e}) & {v['per_hour']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------- report
def markdown(summary: dict) -> str:
    m, b = summary["meta"], summary["benchmark"]
    det, fa = b["detection"], b["false_alarms"]
    wl_t, _ = _last_wl(summary)
    full_v, full_s = det["phi_full"]["family:violating"], det["phi_full"]["family:stealthy"]
    cfg = m["config"]
    out = [
        "# Evaluation report", "",
        f"Generated {m['generated']} - profile `{cfg['profile']}`.", "",
        "## Protocol", "",
        f"* {m['graphs']} synthetic block-structured SFCs (tiers small / medium / large / complex) "
        f"and {m['case_studies']} reference case study.",
        f"* Clean test trace of {cfg['test_scans']:,} scans per graph (scan = {cfg['dt']} s, "
        f"duration jitter ±{int(100 * cfg['jitter'])} %), independent training trace for the baselines.",
        f"* {cfg['episodes']} episodes per attack class and graph, attack windows of 1-8 scans, "
        f"5 scans of horizon after each window; {m['episodes']:,} episodes in total.",
        "", "## Key results", "",
        f"* False alarms of the Phi verifier: **{fa['phi_full']['alarms']}** on "
        f"{fa['phi_full']['transitions']:,} clean transitions "
        f"(FPR 95 % upper bound {fa['phi_full']['upper95']:.1e}), without any training.",
        f"* Structure-violating attacks: **{_pct(full_v['rate'], 2)} %** detected "
        f"(95 % CI {_pct(full_v['ci'][0], 2)}-{_pct(full_v['ci'][1], 2)} %), "
        f"{_pct(full_v['onset_rate'], 1)} % on the first manipulated scan.",
        f"* Structure-preserving (stealthy) attacks: {_pct(full_s['rate'], 1)} % detected - these "
        "executions are legal for the SFC structure, which is the expected limit of the method "
        "(replays are only caught at the splice discontinuity).",
        f"* Learned transition whitelist ({cfg['train_scans'][-1]:,} training scans): "
        f"{fa[wl_t]['per_hour']:.2f} false alarms per hour.",
        "", "## Detection per attack (Phi verifier)", "",
        "| Attack | Family | Episodes | Detected | Onset | Median latency (scans) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for a, s in ATTACKS.items():
        f = det["phi_full"][a]
        lat = f["latency"]["median"]
        out.append(f"| {s.title} | {s.family} | {f['episodes']} | {_pct(f['rate'])} % | "
                   f"{_pct(f['onset_rate'])} % | {'-' if lat is None else f'{lat:g}'} |")
    out += ["", "Figures: `figures/fig_detection`, `fig_false_alarms`, `fig_rules`, "
            "`fig_scalability` (SVG and PDF). LaTeX tables: `tables/`."]
    return "\n".join(out) + "\n"


def build(summary: dict | None = None, out_dir: Path = RESULTS_DIR) -> None:
    if summary is None:
        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    figs, tabs = out_dir / "figures", out_dir / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)
    fig_detection(summary, figs)
    fig_false_alarms(summary, figs)
    fig_scalability(summary, figs)
    fig_rules(summary, figs)
    (tabs / "tab_benchmark.tex").write_text(tab_benchmark(summary) + "\n", encoding="utf-8")
    (tabs / "tab_detection.tex").write_text(tab_detection(summary) + "\n", encoding="utf-8")
    (tabs / "tab_false_alarms.tex").write_text(tab_false_alarms(summary) + "\n", encoding="utf-8")
    (out_dir / "REPORT.md").write_text(markdown(summary), encoding="utf-8")
