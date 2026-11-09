"""Rebuild results/*.csv and figures/*.png from the transcripts in results/raw/.

  uv run python -m badapple.analyze            # main blocks A-E
  uv run python -m badapple.analyze --pilot    # pilot sanity report only

CIs: percentile bootstrap (2000 resamples) over runs (= questions x seeds),
with each run's value being the mean over its honest agents. This treats the
question as the unit of independence, since agents in a run are correlated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw"
RES = ROOT / "results"
FIG = ROOT / "figures"

COND_ORDER = ["clean", "attack", "attack+verify", "attack+attribution", "clean+verify", "clean+attribution"]
TOPO_ORDER = ["chain", "star_hub", "star_leaf", "complete", "ring", "small_world"]
TOPO_LABEL = {"chain": "Chain\n(attacker at end)", "star_hub": "Star\n(attacker = hub)",
              "star_leaf": "Star\n(attacker = leaf)", "complete": "Complete",
              "ring": "Ring", "small_world": "Small-world"}
# Reference categorical palette (dataviz skill, light mode), fixed order.
COLORS = {"clean": "#2a78d6", "attack": "#eb6834", "attack+verify": "#1baf7a",
          "attack+attribution": "#eda100", "clean+verify": "#1baf7a", "clean+attribution": "#eda100"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


# ---------------------------------------------------------------- loading
def load_run(path: Path) -> list[dict]:
    """Agent x round rows for honest agents of one transcript."""
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    h = lines[0]
    fin = lines[-1]
    assert h["type"] == "header" and fin["type"] == "final", path
    calls = [x for x in lines if x["type"] == "call"]
    parse = {(c["agent"], c["round"]): c["parse_ok"] for c in calls}
    att_ok = [c["parsed"] == h["target"] for c in calls if c["role"] == "attacker"]
    rows = []
    for a in fin["honest"]:
        for r, ans in enumerate(fin["answers"][str(a)]):
            rows.append({
                "run_id": h["run_id"], "block": h["block"], "model": h["model"],
                "k": f"{h['n_steps']}{ {'tricky': 't', 'bigmul': 'm'}.get(h.get('tier', ''), '') }",
                "topology": h["topology"], "condition": h["condition"], "n": h["n"],
                "qid": h["qid"], "seed": h["seed"], "agent": a, "hop": fin["hop"][str(a)],
                "round": r, "answer": ans, "truth": h["truth"], "target": h["target"],
                "correct": ans == h["truth"], "infected": ans == h["target"],
                "parse_ok": parse.get((a, r), True),
                "attacker_compliance": float(np.mean(att_ok)) if att_ok else np.nan,
            })
    return rows


def load(blocks: list[str]) -> pd.DataFrame:
    rows = []
    for b in blocks:
        for p in sorted((RAW / b).glob("*.jsonl")):
            rows += load_run(p)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- stats
def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


CELL = ["block", "model", "k", "topology", "condition", "n"]
# Core 16-cell grid per task tier. k = steps per problem; "t" suffix = tricky (reading-comprehension) tier.
CORE = {"8": "A", "14": "H", "14t": "T", "20": "X", "6m": "M"}
TIER_LABEL = {"8": "8-step arithmetic", "14": "14-step arithmetic", "14t": "14-step tricky wording",
              "20": "20-step arithmetic", "6m": "6-step, 4x3-digit products"}


def per_run(df: pd.DataFrame, rnd: int | None = None) -> pd.DataFrame:
    """One row per run: honest-agent means at the final round (or a given round)."""
    if rnd is None:
        rnd = df["round"].max()
    d = df[df["round"] == rnd]
    return d.groupby(CELL + ["qid", "seed"], as_index=False).agg(
        infection=("infected", "mean"), accuracy=("correct", "mean"), parse_rate=("parse_ok", "mean"),
        attacker_compliance=("attacker_compliance", "first"))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    fin = per_run(df)
    r0 = per_run(df, 0).rename(columns={"accuracy": "acc_r0"})[CELL + ["qid", "seed", "acc_r0"]]
    m = fin.merge(r0, on=CELL + ["qid", "seed"])
    out = []
    for key, g in m.groupby(CELL):
        inf = bootstrap_ci(g["infection"])
        acc = bootstrap_ci(g["accuracy"])
        out.append(dict(zip(CELL, key)) | {
            "n_runs": len(g),
            "infection": inf[0], "infection_lo": inf[1], "infection_hi": inf[2],
            "accuracy": acc[0], "accuracy_lo": acc[1], "accuracy_hi": acc[2],
            "accuracy_round0": g["acc_r0"].mean(),
            "parse_rate": g["parse_rate"].mean(),
            "attacker_compliance": g["attacker_compliance"].mean(),
        })
    s = pd.DataFrame(out)
    # Paired accuracy drop vs the clean cell with the same model/topology/n (same questions).
    drops = []
    for _, row in s.iterrows():
        same = (m.model == row.model) & (m.k == row.k) & (m.topology == row.topology) & (m.n == row.n)
        base = m[same & (m.condition == "clean")]
        cur = m[same & (m.condition == row.condition)]
        j = cur.merge(base, on=["qid", "seed"], suffixes=("", "_clean"))
        if len(j) and row.condition != "clean":
            d = bootstrap_ci(j["accuracy_clean"] - j["accuracy"])
        else:
            d = (np.nan, np.nan, np.nan)
        drops.append(d)
    s["acc_drop_vs_clean"] = [d[0] for d in drops]
    s["acc_drop_lo"] = [d[1] for d in drops]
    s["acc_drop_hi"] = [d[2] for d in drops]
    s["_c"] = s.condition.map({c: i for i, c in enumerate(COND_ORDER)})
    s["_t"] = s.topology.map({t: i for i, t in enumerate(TOPO_ORDER)})
    return s.sort_values(["k", "block", "model", "n", "_t", "_c"]).drop(columns=["_c", "_t"]).reset_index(drop=True)


def by_hop(df: pd.DataFrame) -> pd.DataFrame:
    fin = df[df["round"] == df["round"].max()].assign(wrong=lambda x: ~x["correct"])
    g = fin.groupby(CELL + ["hop", "qid", "seed"], as_index=False)[["infected", "wrong"]].mean()
    out = []
    for key, gg in g.groupby(CELL + ["hop"]):
        ci = bootstrap_ci(gg["infected"])
        er = bootstrap_ci(gg["wrong"])
        out.append(dict(zip(CELL + ["hop"], key)) | {"infection": ci[0], "lo": ci[1], "hi": ci[2],
                                                     "error": er[0], "err_lo": er[1], "err_hi": er[2],
                                                     "n_runs": len(gg)})
    return pd.DataFrame(out)


def by_round(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(CELL + ["round", "qid", "seed"], as_index=False).agg(
        infected=("infected", "mean"), correct=("correct", "mean"))
    out = []
    for key, gg in g.groupby(CELL + ["round"]):
        ci = bootstrap_ci(gg["infected"])
        ac = bootstrap_ci(gg["correct"])
        out.append(dict(zip(CELL + ["round"], key)) | {"infection": ci[0], "lo": ci[1], "hi": ci[2],
                                                       "accuracy": ac[0], "acc_lo": ac[1], "acc_hi": ac[2]})
    return pd.DataFrame(out)


def transitions(df: pd.DataFrame) -> pd.DataFrame:
    """Per cell: share of honest agents wrong at round 0 who end correct (peer correction)
    and right at round 0 who end wrong (corruption), with Wilson CIs."""
    last = df["round"].max()
    a0 = df[df["round"] == 0].set_index(CELL + ["qid", "seed", "agent"])["correct"]
    aF = df[df["round"] == last].set_index(CELL + ["qid", "seed", "agent"])["correct"]
    j = pd.DataFrame({"c0": a0, "cF": aF}).reset_index()
    out = []
    for key, g in j.groupby(CELL):
        wrong0, right0 = g[~g.c0], g[g.c0]
        k1, n1 = int(wrong0.cF.sum()), len(wrong0)
        k2, n2 = int((~right0.cF).sum()), len(right0)
        out.append(dict(zip(CELL, key)) | {
            "wrong_r0": n1, "corrected": k1, "correction_rate": k1 / n1 if n1 else np.nan,
            "correction_lo": wilson(k1, n1)[0], "correction_hi": wilson(k1, n1)[1],
            "right_r0": n2, "corrupted": k2, "corruption_rate": k2 / n2 if n2 else np.nan,
            "corruption_lo": wilson(k2, n2)[0], "corruption_hi": wilson(k2, n2)[1]})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- figures
def _style(ax, ylabel: str):
    ax.set_facecolor(SURF)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def _err(lo, mid, hi):
    return np.vstack([np.maximum(mid - lo, 0), np.maximum(hi - mid, 0)])


def fig_topology(s: pd.DataFrame, path: Path, metric: str = "infection", k: str = "14t"):
    import matplotlib.pyplot as plt
    d = s[(s.block == CORE[k]) & (s.k == k)]
    if d.empty:
        return
    conds = [c for c in ["clean", "attack", "attack+verify", "attack+attribution"] if c in set(d.condition)]
    topos = [t for t in TOPO_ORDER if t in set(d.topology)]
    fig, ax = plt.subplots(figsize=(7.5, 3.6), facecolor=SURF)
    w = 0.8 / len(conds)
    for k, c in enumerate(conds):
        dc = d[d.condition == c].set_index("topology").reindex(topos)
        x = np.arange(len(topos)) + (k - (len(conds) - 1) / 2) * w
        if metric == "error":  # wrong-answer rate = 1 - accuracy (CI flips)
            mid, lo, hi = 1 - dc.accuracy.values, 1 - dc.accuracy_hi.values, 1 - dc.accuracy_lo.values
        else:
            mid, lo, hi = dc[metric].values, dc[f"{metric}_lo"].values, dc[f"{metric}_hi"].values
        ax.bar(x, mid, width=w * 0.92, color=COLORS[c], label=c, edgecolor=SURF, linewidth=1)
        ax.errorbar(x, mid, yerr=_err(lo, mid, hi), fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
    ax.set_xticks(np.arange(len(topos)), [TOPO_LABEL[t] for t in topos])
    ylab = {"infection": "Honest agents ending on attacker's answer", "accuracy": "Final accuracy (honest agents)",
            "error": "Wrong final answers (honest agents)"}[metric]
    _style(ax, ylab)
    if metric == "error":
        ax.set_ylim(0, max(0.2, float(np.nanmax(1 - d.accuracy_lo)) + 0.02))
    else:
        ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8, ncol=len(conds), loc="upper center", bbox_to_anchor=(0.5, 1.13), labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURF)
    plt.close(fig)


def fig_hop(h: pd.DataFrame, path: Path, k: str = "14t", metric: str = "infection"):
    import matplotlib.pyplot as plt
    d = h[(h.block == CORE[k]) & (h.k == k) & (h.condition == "attack")].copy()
    if metric == "error":  # wrong-answer rate by hop
        d["infection"], d["lo"], d["hi"] = d["error"], d["err_lo"], d["err_hi"]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 3.4), facecolor=SURF)
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    for k, t in enumerate([t for t in TOPO_ORDER if t in set(d.topology)]):
        dt = d[d.topology == t].sort_values("hop")
        ax.errorbar(dt.hop + (k - 1.5) * 0.06, dt.infection, yerr=_err(dt.lo.values, dt.infection.values, dt.hi.values),
                    color=pal[k % 4], marker="o", markersize=5, linewidth=2, capsize=2,
                    label=TOPO_LABEL[t].replace("\n", " "))
    ax.set_xlabel("Hop distance from attacker", color=INK2, fontsize=9)
    _style(ax, "Infection rate (final round)" if metric == "infection" else "Wrong-answer rate (final round)")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURF)
    plt.close(fig)


def fig_round(r: pd.DataFrame, path: Path, k: str = "14t", metric: str = "infection", topo: str = "complete"):
    import matplotlib.pyplot as plt
    d = r[(r.block == CORE[k]) & (r.k == k) & (r.topology == topo)]
    if d.empty:
        return
    y, lo, hi = ("infection", "lo", "hi") if metric == "infection" else ("accuracy", "acc_lo", "acc_hi")
    fig, ax = plt.subplots(figsize=(6, 3.4), facecolor=SURF)
    for c in [c for c in ["clean", "attack", "attack+verify", "attack+attribution"] if c in set(d.condition)]:
        dc = d[d.condition == c].sort_values("round")
        ax.plot(dc["round"], dc[y], color=COLORS[c], marker="o", markersize=5, linewidth=2, label=c)
        ax.fill_between(dc["round"], dc[lo], dc[hi], color=COLORS[c], alpha=0.15, linewidth=0)
    ax.set_xlabel("Round", color=INK2, fontsize=9)
    ax.set_xticks(sorted(d["round"].unique()))
    _style(ax, f"{'Infection rate' if metric == 'infection' else 'Accuracy'} ({topo.replace('_', ' ')}, N=8)")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURF)
    plt.close(fig)


def fig_scaling(s: pd.DataFrame, path: Path, k: str = "14t", metric: str = "accuracy"):
    import matplotlib.pyplot as plt
    d = s[(s.model == "haiku") & (s.k == k) & (s.topology == "complete") & (s.condition.isin(["clean", "attack"]))
          & (s.block.isin([CORE[k], "C"]))]
    if d.n.nunique() < 2:
        return
    fig, ax = plt.subplots(figsize=(5, 3.4), facecolor=SURF)
    for c in ["clean", "attack"]:
        dc = d[d.condition == c].sort_values("n")
        ax.errorbar(dc.n, dc[metric], yerr=_err(dc[f"{metric}_lo"].values, dc[metric].values, dc[f"{metric}_hi"].values),
                    color=COLORS[c], marker="o", markersize=6, linewidth=2, capsize=3, label=c)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(d.n.unique()), [str(x) for x in sorted(d.n.unique())])
    ax.set_xlabel("Swarm size N (complete graph, one attacker)", color=INK2, fontsize=9)
    _style(ax, "Infection rate (final round)" if metric == "infection" else "Final accuracy (honest agents)")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURF)
    plt.close(fig)


def fig_defence_cost(s: pd.DataFrame, t: pd.DataFrame, path: Path, k: str = "14t"):
    import matplotlib.pyplot as plt
    conds = ["clean", "clean+verify", "clean+attribution"]
    d = s[(s.model == "haiku") & (s.k == k) & (s.topology == "complete") & (s.n == 8) & (s.condition.isin(conds))
          & (s.block.isin([CORE[k], "B"]))]
    if d.condition.nunique() < 2:
        return
    d = d.set_index("condition").reindex([c for c in conds if c in set(d.condition)])
    fig, ax = plt.subplots(figsize=(5.5, 3.4), facecolor=SURF)
    x = np.arange(len(d))
    w = 0.38
    ax.bar(x - w / 2, d.accuracy_round0, width=w * 0.92, color="#86b6ef", label="Round 0 (alone)", edgecolor=SURF)
    ax.bar(x + w / 2, d.accuracy, width=w * 0.92, color="#2a78d6", label="Final round", edgecolor=SURF)
    ax.errorbar(x + w / 2, d.accuracy, yerr=_err(d.accuracy_lo.values, d.accuracy.values, d.accuracy_hi.values),
                fmt="none", ecolor=INK2, capsize=2, elinewidth=1)
    ax.set_xticks(x, ["No defence", "Commit-then-verify", "Attribution"][: len(d)] if list(d.index) == conds[: len(d)] else list(d.index))
    _style(ax, "Accuracy, no attacker (complete, N=8)")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURF)
    plt.close(fig)


def fig_model(s: pd.DataFrame, path: Path, k: str = "14t", metric: str = "accuracy"):
    import matplotlib.pyplot as plt
    d = s[(s.k == k) & s.topology.isin(["complete", "star_leaf"]) & (s.n == 8) & s.condition.isin(["clean", "attack"])
          & (((s.model == "haiku") & (s.block == CORE[k])) | (s.model == "sonnet"))]
    if d.model.nunique() < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), facecolor=SURF, sharey=True)
    for ax, topo in zip(axes, ["complete", "star_leaf"]):
        dt = d[d.topology == topo]
        for j, c in enumerate(["clean", "attack"]):
            dc = dt[dt.condition == c].set_index("model").reindex(["haiku", "sonnet"])
            x = np.arange(2) + (j - 0.5) * 0.38
            ax.bar(x, dc[metric], width=0.35, color=COLORS[c], label=c, edgecolor=SURF)
            ax.errorbar(x, dc[metric], yerr=_err(dc[f"{metric}_lo"].values, dc[metric].values, dc[f"{metric}_hi"].values),
                        fmt="none", ecolor=INK2, capsize=2, elinewidth=1)
        ax.set_xticks(np.arange(2), ["Haiku 5.5\n(20 q)", "Sonnet 5.5\n(10 q)"])
        ax.set_title(TOPO_LABEL[topo].replace("\n", " "), fontsize=9, color=INK)
        _style(ax, ("Infection rate" if metric == "infection" else "Final accuracy") if topo == "complete" else "")
        ax.set_ylim(0, 1.02)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURF)
    plt.close(fig)


def fig_tiers(s: pd.DataFrame, path: Path):
    """Accuracy drop caused by the attacker (paired vs clean), per topology, easy vs hard tier."""
    import matplotlib.pyplot as plt
    core = (s.block == s.k.map(CORE))
    d = s[(s.condition == "attack") & (s.model == "haiku") & (s.n == 8) & core]
    tiers = [k for k in ["8", "14", "6m"] if k in set(d.k)]
    if len(tiers) < 2:
        return
    topos = [t for t in TOPO_ORDER if t in set(d.topology)]
    fig, ax = plt.subplots(figsize=(7, 3.4), facecolor=SURF)
    tier_col = {"8": "#b7d3f6", "14": "#5598e7", "6m": "#104281"}  # one blue ramp, light -> dark
    w = 0.8 / len(tiers)
    for j, k in enumerate(tiers):
        dk = d[d.k == k].set_index("topology").reindex(topos)
        x = np.arange(len(topos)) + (j - (len(tiers) - 1) / 2) * w
        ax.bar(x, dk.acc_drop_vs_clean, width=w * 0.92, color=tier_col[k], edgecolor=SURF,
               label=TIER_LABEL[k])
        ax.errorbar(x, dk.acc_drop_vs_clean, yerr=_err(dk.acc_drop_lo.values, dk.acc_drop_vs_clean.values,
                                                        dk.acc_drop_hi.values),
                    fmt="none", ecolor=INK2, capsize=2, elinewidth=1)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xticks(np.arange(len(topos)), [TOPO_LABEL[t] for t in topos])
    _style(ax, "Accuracy drop vs clean (fraction)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURF)
    plt.close(fig)


def fig_transitions(t: pd.DataFrame, path: Path, k: str = "6m"):
    """Mechanism: peer correction (wrong at round 0 -> right at end) and corruption
    (right at round 0 -> wrong at end), per topology and condition, Wilson 95% CIs."""
    import matplotlib.pyplot as plt
    d = t[(t.k == k) & (t.block == CORE[k]) & (t.n == 8)]
    if d.empty:
        return
    conds = [c for c in ["clean", "attack", "attack+verify", "attack+attribution"] if c in set(d.condition)]
    topos = [x for x in TOPO_ORDER if x in set(d.topology)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=SURF)
    w = 0.8 / len(conds)
    for ax, (col, title) in zip(axes, [("correction", "Peer correction: wrong at round 0, right at end"),
                                       ("corruption", "Corruption: right at round 0, wrong at end")]):
        for j, c in enumerate(conds):
            dc = d[d.condition == c].set_index("topology").reindex(topos)
            x = np.arange(len(topos)) + (j - (len(conds) - 1) / 2) * w
            mid = dc[f"{col}_rate"].values.astype(float)
            ax.bar(x, mid, width=w * 0.92, color=COLORS[c], label=c, edgecolor=SURF)
            ax.errorbar(x, mid, yerr=_err(dc[f"{col}_lo"].values.astype(float), mid, dc[f"{col}_hi"].values.astype(float)),
                        fmt="none", ecolor=INK2, capsize=2, elinewidth=1)
        short = {"chain": "Chain\n(att. end)", "star_hub": "Star\n(att. hub)", "star_leaf": "Star\n(att. leaf)",
                 "complete": "Complete", "ring": "Ring", "small_world": "Small-world"}
        ax.set_xticks(np.arange(len(topos)), [short[x] for x in topos], fontsize=8)
        ax.set_title(title, fontsize=9, color=INK)
        _style(ax, "Share of honest agents")
        ax.set_ylim(0, 1.02 if col == "correction" else 0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=INK, ncol=len(conds),
               loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=200, facecolor=SURF)
    plt.close(fig)


def _pct(x) -> str:
    if pd.isna(x):
        return "n/a"
    v = 100 * float(x)
    return f"{0.0 if abs(v) < 0.05 else v:.1f}"


def write_tables(s: pd.DataFrame, t: pd.DataFrame, path: Path, df: pd.DataFrame | None = None) -> None:
    """Markdown tables quoted in the report (all numbers in %; CIs are 95% bootstrap over questions)."""
    out = ["<!-- generated by analyze.py; do not edit -->", ""]
    if df is not None:
        fin = df[(df["round"] == df["round"].max()) & df.condition.str.startswith("attack")]
        out += ["### Headline counts: honest agents' final answers in all attack cells", "",
                "| Block | Model | Tier | Honest final answers | On attacker's target | Wrong (other number) | Unparsable |",
                "|---|---|---|---|---|---|---|"]
        for (b, m, k), g in fin.groupby(["block", "model", "k"]):
            other = int((~g.correct & ~g.infected & g.parse_ok).sum())
            out.append(f"| {b} | {m} | {TIER_LABEL.get(k, k)} | {len(g)} | {int(g.infected.sum())} | {other} | "
                       f"{int((~g.parse_ok).sum())} |")
        hk = fin[fin.model == "haiku"]
        out += [f"| **all Haiku** | haiku | | {len(hk)} | {int(hk.infected.sum())} | "
                f"{int((~hk.correct & ~hk.infected & hk.parse_ok).sum())} | {int((~hk.parse_ok).sum())} |", ""]
    core = s[s.block == s.k.map(CORE)]
    for k in [k for k in ["8", "14", "6m"] if k in set(core.k)]:
        d = core[core.k == k]
        out += [f"### Core grid, {TIER_LABEL[k]} (Haiku 5.5, N=8, 20 questions)", "",
                "| Topology | Condition | Round-0 acc. | Final acc. [95% CI] | Infection [95% CI] | Acc. drop vs clean [95% CI] |",
                "|---|---|---|---|---|---|"]
        for _, r in d.iterrows():
            drop = "" if r.condition == "clean" else f"{_pct(r.acc_drop_vs_clean)} [{_pct(r.acc_drop_lo)}, {_pct(r.acc_drop_hi)}]"
            out.append(f"| {r.topology} | {r.condition} | {_pct(r.accuracy_round0)} | {_pct(r.accuracy)} "
                       f"[{_pct(r.accuracy_lo)}, {_pct(r.accuracy_hi)}] | {_pct(r.infection)} "
                       f"[{_pct(r.infection_lo)}, {_pct(r.infection_hi)}] | {drop} |")
        out.append("")
    other = s[s.block.isin(["B", "C", "D", "E"])]
    if len(other):
        out += ["### Defence cost (B), scaling (C), cross-model (D)", "",
                "| Block | Model | Tier | Topology | Condition | N | Round-0 acc. | Final acc. [95% CI] | Infection [95% CI] |",
                "|---|---|---|---|---|---|---|---|---|"]
        for _, r in other.iterrows():
            out.append(f"| {r.block} | {r.model} | {TIER_LABEL.get(r.k, r.k)} | {r.topology} | {r.condition} | {r.n} | "
                       f"{_pct(r.accuracy_round0)} | {_pct(r.accuracy)} [{_pct(r.accuracy_lo)}, {_pct(r.accuracy_hi)}] | "
                       f"{_pct(r.infection)} [{_pct(r.infection_lo)}, {_pct(r.infection_hi)}] |")
        out.append("")
    # Empty responses (0 output tokens) = API-side declines (stop_reason "refusal"); per model x round x condition.
    calls = []
    for b in sorted({*s.block}):
        for p in (RAW / b).glob("*.jsonl"):
            h = None
            for line in p.read_text(encoding="utf-8").splitlines():
                c = json.loads(line)
                if c["type"] == "header":
                    h = c
                elif c["type"] == "call" and c["role"] == "honest":
                    calls.append({"model": h["model"], "condition": h["condition"], "round": c["round"],
                                  "empty": c["output_tokens"] == 0 and not c["raw"].strip()})
    if calls:
        cdf = pd.DataFrame(calls)
        piv = cdf.groupby(["model", "condition", "round"]).empty.agg(["sum", "count"]).reset_index()
        piv = piv[piv.groupby(["model"])["sum"].transform("sum") > 0]
        if len(piv):
            out += ["### Empty (declined) responses from honest agents, by round", "",
                    "| Model | Condition | Round | Declined / calls |", "|---|---|---|---|"]
            for _, r in piv.iterrows():
                out.append(f"| {r.model} | {r.condition} | {r['round']} | {int(r['sum'])}/{int(r['count'])} |")
            out.append("")
    tt = t[t.block == t.k.map(CORE)]
    out += ["### Peer correction and corruption (core grid; Wilson 95% CIs)", "",
            "| Tier | Topology | Condition | Corrected / wrong at r0 | Correction rate | Corrupted / right at r0 | Corruption rate |",
            "|---|---|---|---|---|---|---|"]
    for _, r in tt.iterrows():
        out.append(f"| {TIER_LABEL[r.k]} | {r.topology} | {r.condition} | {r.corrected}/{r.wrong_r0} | "
                   f"{_pct(r.correction_rate)} [{_pct(r.correction_lo)}, {_pct(r.correction_hi)}] | "
                   f"{r.corrupted}/{r.right_r0} | {_pct(r.corruption_rate)} [{_pct(r.corruption_lo)}, {_pct(r.corruption_hi)}] |")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def pilot_report(block: str = "pilot_k8") -> None:
    df = load([block])
    if df.empty:
        print("no pilot transcripts")
        return
    s = summarize(df)
    cols = ["condition", "n_runs", "accuracy_round0", "accuracy", "infection", "parse_rate", "attacker_compliance"]
    print(s[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    calls = []
    for p in (RAW / block).glob("*.jsonl"):
        calls += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if '"type": "call"' in x]
    c = pd.DataFrame(calls)
    print(f"calls={len(c)} mean_in={c.input_tokens.mean():.0f} mean_out={c.output_tokens.mean():.0f} "
          f"parse_ok={c.parse_ok.mean():.3f} attacker_forced={c[c.role == 'attacker'].attacker_forced.mean():.3f}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", nargs="?", const="pilot_k8", default=None,
                    help="pilot dir to report on (default pilot_k8; LLM-attacker pilot: pilot)")
    ap.add_argument("--blocks", default="A,H,M,B,C,D,E")
    a = ap.parse_args(argv)
    if a.pilot:
        pilot_report(a.pilot)
        return
    df = load([b for b in a.blocks.split(",") if b])
    if df.empty:
        print("no transcripts found")
        return
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    for old in FIG.glob("*.png"):  # every figure is regenerated from transcripts
        old.unlink()
    s = summarize(df)
    h = by_hop(df)
    r = by_round(df)
    t = transitions(df)
    s.to_csv(RES / "summary.csv", index=False, float_format="%.4f")
    h.to_csv(RES / "by_hop.csv", index=False, float_format="%.4f")
    r.to_csv(RES / "by_round.csv", index=False, float_format="%.4f")
    t.to_csv(RES / "transitions.csv", index=False, float_format="%.4f")
    write_tables(s, t, RES / "tables.md", df)
    import matplotlib
    matplotlib.use("Agg")
    for k in sorted(s.k.unique()):
        fig_topology(s, FIG / f"k{k}_fig1_infection_topology_defence.png", k=k)
        fig_topology(s, FIG / f"k{k}_fig1b_error_topology_defence.png", metric="error", k=k)
        fig_transitions(t, FIG / f"k{k}_fig8_correction_corruption.png", k=k)
        fig_hop(h, FIG / f"k{k}_fig2_infection_vs_hop.png", k=k)
        fig_hop(h, FIG / f"k{k}_fig2b_error_vs_hop.png", k=k, metric="error")
        fig_round(r, FIG / f"k{k}_fig3_infection_vs_round.png", k=k)
        fig_round(r, FIG / f"k{k}_fig3b_accuracy_vs_round.png", k=k, metric="accuracy")
        fig_round(r, FIG / f"k{k}_fig3c_accuracy_vs_round_starhub.png", k=k, metric="accuracy", topo="star_hub")
        fig_scaling(s, FIG / f"k{k}_fig4_scaling_n.png", k=k)
        fig_scaling(s, FIG / f"k{k}_fig4b_scaling_n_infection.png", k=k, metric="infection")
        fig_defence_cost(s, t, FIG / f"k{k}_fig5_defence_cost.png", k=k)
        fig_model(s, FIG / f"k{k}_fig6_cross_model.png", k=k)
        fig_model(s, FIG / f"k{k}_fig6b_cross_model_infection.png", k=k, metric="infection")
    fig_tiers(s, FIG / "fig7_difficulty_tiers.png")
    cols = ["block", "model", "k", "topology", "condition", "n", "n_runs", "infection", "infection_lo", "infection_hi",
            "accuracy", "accuracy_round0", "acc_drop_vs_clean", "acc_drop_lo", "acc_drop_hi", "parse_rate"]
    print(s[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nwrote {RES / 'summary.csv'} and figures to {FIG}")


if __name__ == "__main__":
    main()
