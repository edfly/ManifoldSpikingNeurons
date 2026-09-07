"""Paired statistics for v38 supplementary experiments (P2-1 5-seed, P2-7 sphere).

All methods share the same (seed, fold) ordering (seed-major, fold-minor), so
accs[i] of any JSON aligns with accs[i] of every other JSON on the same dataset.

Outputs:
  spikergnn_results/v38_stats.json  - machine-readable
  spikergnn_results/v38_stats.md    - readable report
"""

# ── path resolution (adapted for the post/ layout) ─────────────────
# Results live in ../results relative to this file; the manuscript lives
# two directories up (the project root). Override with env vars if needed.
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)                     # .../post
RESULTS_ROOT = _os.environ.get(
    "RESULTS_ROOT", _os.path.join(_ROOT, "results"))
POST_ROOT = _os.environ.get("POST_ROOT", _ROOT)
FIGURES_ROOT = _os.environ.get(
    "FIGURES_ROOT", _os.path.join(_ROOT, "figures"))
# project root is TWO levels above post/ (post/ lives in programfiles/)
PAPER_TEX = _os.environ.get(
    "PAPER_TEX", _os.path.join(_os.path.dirname(_os.path.dirname(_ROOT)),
                               "ManifoldValuedSpikingNeurons_Neurocomputing_v38.tex"))
# ───────────────────────────────────────────────────────────────────
import json, os
import numpy as np
from scipy import stats

R5 = _os.path.join(RESULTS_ROOT, "v38_5seeds")
R2 = _os.path.join(RESULTS_ROOT, "v38_s2")
OUT_MD = _os.path.join(RESULTS_ROOT, "v38_stats.md")
OUT_JSON = _os.path.join(RESULTS_ROOT, "v38_stats.json")

DATASETS = ["MUTAG", "NCI1", "PROTEINS", "DD"]
METHODS5 = ["OM", "GIN", "GCN", "SAGE", "NS", "MSG", "SGC"]
SEEDS5 = [42, 123, 456, 789, 2024]
SEEDS3 = [42, 123, 456]
NF = 3


def load(fp):
    with open(fp) as f:
        return json.load(f)


def paired(om, base, n=None):
    """Return aligned paired samples (diffs = om - base)."""
    n = n if n is not None else min(len(om), len(base))
    a, b = np.array(om[:n], float), np.array(base[:n], float)
    return a - b, a, b


def bh_correct(pvals):
    """Benjamini-Hochberg FDR correction. Returns corrected p-values."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    corr = ranked * m / (np.arange(1, m + 1))
    corr = np.minimum.accumulate(corr[::-1])[::-1]
    corr = np.minimum(corr, 1.0)
    out = np.empty_like(corr)
    out[order] = corr
    return out


def fmt(v):
    return f"{v:.4f}"


def run():
    rows = []          # one entry per comparison
    table5 = []        # main 15-sample table rows
    table_s2 = []      # 9-sample sphere table rows

    # ---- P2-1: 15-sample paired tests, OM vs every baseline ----
    for ds in DATASETS:
        om = load(os.path.join(R5, f"{ds}_OM.json"))["accs"]
        for m in METHODS5:
            if m == "OM":
                continue
            b = load(os.path.join(R5, f"{ds}_{m}.json"))["accs"]
            diffs, a, b2 = paired(om, b, 15)
            n = len(diffs)
            sw = stats.shapiro(diffs)                       # normality of diffs
            tt = stats.ttest_rel(a, b2)                     # paired t
            wx = stats.wilcoxon(a, b2)                      # signed-rank
            cohend = float(np.mean(diffs) / (np.std(diffs, ddof=1) + 1e-12))
            rows.append({
                "scope": "P2-1", "dataset": ds, "method": m, "n": n,
                "om_mean": float(a.mean()), "om_std": float(a.std(ddof=1)),
                "base_mean": float(b2.mean()), "base_std": float(b2.std(ddof=1)),
                "diff_mean": float(diffs.mean()),
                "shapiro_w": float(sw.statistic), "shapiro_p": float(sw.pvalue),
                "ttest_p": float(tt.pvalue), "wilcoxon_p": float(wx.pvalue),
                "cohen_d": cohend,
            })
            table5.append([ds, m, n, a.mean(), b2.mean(), diffs.mean(),
                           tt.pvalue, wx.pvalue, sw.pvalue, cohend])

    # BH across P2-1 comparisons (use t if normal else wilcoxon as primary)
    for r in rows:
        r["primary_p"] = r["ttest_p"] if r["shapiro_p"] > 0.05 else r["wilcoxon_p"]
    pv = [r["primary_p"] for r in rows]
    corr = bh_correct(pv)
    for r, c in zip(rows, corr):
        r["bh_q"] = float(c)
        r["sig"] = bool(c < 0.05)

    # ---- P2-7: 9-sample sphere comparison (S2 vs OM / vs GIN), aligned first 9 ----
    rows_s2 = []
    for ds in DATASETS:
        s2 = load(os.path.join(R2, f"{ds}_S2.json"))["accs"]
        om = load(os.path.join(R5, f"{ds}_OM.json"))["accs"]
        gin = load(os.path.join(R5, f"{ds}_GIN.json"))["accs"]
        for name, ref in [("OM", om), ("GIN", gin)]:
            diffs, a, b2 = paired(s2, ref, 9)
            n = len(diffs)
            sw = stats.shapiro(diffs)
            tt = stats.ttest_rel(a, b2)
            wx = stats.wilcoxon(a, b2)
            cohend = float(np.mean(diffs) / (np.std(diffs, ddof=1) + 1e-12))
            rows_s2.append({
                "scope": "P2-7", "dataset": ds, "vs": name, "n": n,
                "s2_mean": float(a.mean()), "s2_std": float(a.std(ddof=1)),
                "ref_mean": float(b2.mean()), "ref_std": float(b2.std(ddof=1)),
                "diff_mean": float(diffs.mean()),
                "shapiro_p": float(sw.pvalue),
                "ttest_p": float(tt.pvalue), "wilcoxon_p": float(wx.pvalue),
                "cohen_d": cohend,
            })
            table_s2.append([ds, name, n, a.mean(), b2.mean(), diffs.mean(),
                             tt.pvalue, wx.pvalue, sw.pvalue, cohend])
    for r in rows_s2:
        r["primary_p"] = r["ttest_p"] if r["shapiro_p"] > 0.05 else r["wilcoxon_p"]
    pv = [r["primary_p"] for r in rows_s2]
    corr = bh_correct(pv)
    for r, c in zip(rows_s2, corr):
        r["bh_q"] = float(c)
        r["sig"] = bool(c < 0.05)

    # ---- report ----
    L = []
    L.append("# v38 补充实验配对统计报告")
    L.append("")
    L.append(f"生成时间：本地时间。P2-1 主表 n=15（5 seeds × 3 folds，"
             f"SEEDS={SEEDS5}）；P2-7 对照 n=9（3 seeds × 3 folds，"
             f"SEEDS={SEEDS3}，与 OM/GIN 前 9 样本对齐）。")
    L.append("")
    L.append("检验约定：差值 Shapiro-Wilk 正态（p>0.05 接受正态）→ 正态用配对 t，"
             "否则 Wilcoxon signed-rank，作为 primary_p；BH (Benjamini-Hochberg) "
             "FDR 校正（q<0.05 显著）。效应量 Cohen's d。")
    L.append("")
    L.append("## P2-1：OM vs 各基线（15 配对样本）")
    L.append("")
    L.append("| 数据集 | 基线 | n | OM mean | Base mean | Δ(OM-B) | t-test p | WSR p | SW p | Cohen's d | q(BH) | 显著 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for ds, m, n, am, bm, dm, tp, wp, sp, cd in table5:
        r = next(x for x in rows if x["dataset"] == ds and x["method"] == m)
        L.append(f"| {ds} | {m} | {n} | {am:.1f} | {bm:.1f} | {dm:+.1f} | "
                 f"{fmt(tp)} | {fmt(wp)} | {fmt(sp)} | {cd:+.2f} | {fmt(r['bh_q'])} | "
                 f"{'✅' if r['sig'] else '—'} |")
    L.append("")
    L.append("## P2-7：S2 球面三分量 vs OM / GIN（9 配对样本）")
    L.append("")
    L.append("| 数据集 | 对照 | n | S2 mean | Ref mean | Δ(S2-Ref) | t-test p | WSR p | SW p | Cohen's d | q(BH) | 显著 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for ds, m, n, am, bm, dm, tp, wp, sp, cd in table_s2:
        r = next(x for x in rows_s2 if x["dataset"] == ds and x["vs"] == m)
        L.append(f"| {ds} | {m} | {n} | {am:.1f} | {bm:.1f} | {dm:+.1f} | "
                 f"{fmt(tp)} | {fmt(wp)} | {fmt(sp)} | {cd:+.2f} | {fmt(r['bh_q'])} | "
                 f"{'✅' if r['sig'] else '—'} |")
    L.append("")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    json.dump({"P2-1": rows, "P2-7": rows_s2},
              open(OUT_JSON, "w"), indent=2, ensure_ascii=False)

    print("\n".join(L))

    # console summary of significant findings
    print("\n=== 显著结果摘要 (q<0.05) ===")
    for r in rows + rows_s2:
        tag = "P2-1" if r["scope"] == "P2-1" else "P2-7"
        if r["sig"]:
            print(f"  {tag} {r['dataset']} {r.get('method', r.get('vs'))}: "
                  f"Δ={r['diff_mean']:+.1f} q={r['bh_q']:.4f} d={r['cohen_d']:+.2f}")


if __name__ == "__main__":
    run()
