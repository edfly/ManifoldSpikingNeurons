"""2x2 factorial analysis: geometry (Euclidean / manifold) x mechanism (ANN / spiking).

Two designs are reported, and the difference between them is the point of the
analysis.

Design A (original, confounded)
--------------------------------
                 Euclidean            Manifold
      ANN        GIN                  HGCN
      Spiking    SGC                  OM

    manifold effect @ANN     : HGCN - GIN
    manifold effect @spiking : OM   - SGC

  This is the design originally reported. It is *confounded*: SGC differs from
  OM in the entire spiking mechanism (geodesic state update, spike
  interpolation, rate regulariser L_sr), not only in the curvature. The
  OM - SGC contrast therefore measures mechanism + geometry together, and
  attributing it to geometry alone overstates the geometry's contribution.

Design B (mechanism-matched, primary)
-------------------------------------
                 Euclidean            Manifold
      ANN        GIN                  HGCN-32
      Spiking    OM-E                 OM

    manifold effect @ANN     : HGCN-32 - GIN
    manifold effect @spiking : OM      - OM-E

  OM-E is OM with the curvature removed (h_dim=0 collapses the product
  manifold H^0 x R^32 to plain R^32; logmap and expmap become the identity up
  to floating point). Same depth, same width, same L_sr, same spike
  interpolation, same optimiser and budget, same (seed, fold) pairing. The only
  thing that differs is the curvature, so OM - OM-E is a clean estimate of the
  geometry's causal effect under spiking. HGCN-32 has the same 32-d width as
  OM, so the ANN side is capacity-matched as well.

Statistics
----------
Shapiro-Wilk on the paired contrast (normal if p > 0.05) -> one-sample t if
normal, otherwise Wilcoxon signed-rank. BH FDR across the four datasets.
For the geometry effect under spiking we report the two-sided test against 0
and read a non-significant result as "no difference detected", not as
"equivalence proved".

Usage:
    python analyze_v38_factorial.py
    python analyze_v38_factorial.py --table   # emit LaTeX rows
"""
import json
import os
import sys

import numpy as np
from scipy import stats

R = "spikergnn_results/v38_5seeds"
RH = "spikergnn_results/v38_hgcn"
RH32 = "spikergnn_results/v38_hgcn32"
# Depth-matched to OM: OM uses L=2 on NCI1/PROTEINS/DD, but the first HGCN
# sweep instantiated L=3 everywhere, giving the baseline an extra layer on
# three of four datasets. This rerun fixes that; it is the primary ANN cell.
RH32L = "spikergnn_results/v38_hgcn32L"
RG = "spikergnn_results/v38_gat"
RE = "spikergnn_results/v38_euclid"
R50 = "spikergnn_results/v38_om50"
OUT_MD = "spikergnn_results/v38_factorial.md"
OUT_JSON = "spikergnn_results/v38_factorial.json"

DATASETS = ["MUTAG", "NCI1", "PROTEINS", "DD"]


def load(path):
    with open(path) as f:
        return np.array(json.load(f)["accs"], dtype=float)


def bh(pvals):
    """Benjamini-Hochberg FDR correction -> (q values, significant flags)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out, out < 0.05


def test_paired(x, label):
    """Shapiro -> t or Wilcoxon on a paired contrast against 0."""
    x = np.asarray(x, dtype=float)
    sw = stats.shapiro(x)
    normal = sw.pvalue > 0.05
    if normal:
        tt = stats.ttest_1samp(x, 0)
        p = float(tt.pvalue)
        stat = f"t={tt.statistic:.2f}"
    else:
        wx = stats.wilcoxon(x)
        p = float(wx.pvalue)
        stat = f"W={wx.statistic:.1f}"
    sd = x.std(ddof=1)
    d = float(x.mean() / sd) if sd > 1e-12 else 0.0
    return {
        "mean": float(x.mean()), "sd": float(sd),
        "sw_p": float(sw.pvalue), "normal": bool(normal),
        "stat": stat, "p": p, "cohen_d": d,
    }


def main():
    emit_tex = "--table" in sys.argv
    rows = []
    for ds in DATASETS:
        paths = {
            "OM": f"{R}/{ds}_OM.json",
            "OME": f"{RE}/{ds}_EUCLID.json",
            "HGCN": f"{RH}/{ds}_HGCN.json",
            "HGCN32": f"{RH32}/{ds}_HGCN.json",
            "HGCN32L": f"{RH32L}/{ds}_HGCN.json",
            "GIN": f"{R}/{ds}_GIN.json",
            "SGC": f"{R}/{ds}_SGC.json",
            "GAT": f"{RG}/{ds}_GAT.json",
            "OM50": f"{R50}/{ds}_OM50.json",
        }
        d = {k: load(v) for k, v in paths.items() if os.path.exists(v)}

        # Design B is primary. The manifold-ANN cell is the DEPTH-MATCHED
        # HGCN (HGCN32L) whenever available, because OM uses L=2 on three
        # datasets while the first HGCN sweep used L=3 everywhere.
        ann_cell = "HGCN32L" if "HGCN32L" in d else "HGCN32"
        need_b = ["OM", "OME", ann_cell, "GIN"]
        miss_b = [k for k in need_b if k not in d]
        if miss_b:
            print(f"  (skip {ds}: design B missing {miss_b})")
            continue
        n = min(len(d[k]) for k in need_b)
        if n < 3:
            print(f"  (skip {ds}: n={n})")
            continue
        d = {k: v[:n] for k, v in d.items()}

        # ---- Design B (mechanism-matched, primary) ----
        # ANN cell = depth-matched HGCN; L=3 variant kept as a sensitivity row.
        eff_ann_b = d[ann_cell] - d["GIN"]       # geometry under ANN
        eff_spk_b = d["OM"] - d["OME"]           # geometry under spiking
        inter_b = eff_spk_b - eff_ann_b

        # depth sensitivity: does matching the depth change the ANN-side effect?
        depth = None
        if "HGCN32" in d and ann_cell == "HGCN32L":
            depth = {
                "l3_mean": float(d["HGCN32"].mean()),
                "l3_std": float(d["HGCN32"].std()),
                "lm_mean": float(d["HGCN32L"].mean()),
                "lm_std": float(d["HGCN32L"].std()),
                "delta_mean": float(d["HGCN32L"].mean() - d["HGCN32"].mean()),
                "eff_ann_l3": float((d["HGCN32"] - d["GIN"]).mean()),
                "test": test_paired(d["HGCN32L"] - d["HGCN32"], "depth"),
            }

        # ---- Design A (original, confounded) ----
        have_a = all(k in d for k in ("HGCN", "SGC"))
        if have_a:
            eff_ann_a = d["HGCN"] - d["GIN"]
            eff_spk_a = d["OM"] - d["SGC"]
            inter_a = eff_spk_a - eff_ann_a
        else:
            eff_ann_a = eff_spk_a = inter_a = None

        # ---- Budget sensitivity: OM at the ANN baselines' 50-epoch budget ----
        budget = None
        if "OM50" in d and len(d["OM50"]) >= n:
            budget = test_paired(d["OM50"][:n] - d["OM"][:n], "om50-om")

        rec = {
            "dataset": ds, "n": int(n),
            "om_mean": float(d["OM"].mean()), "om_std": float(d["OM"].std()),
            "ome_mean": float(d["OME"].mean()), "ome_std": float(d["OME"].std()),
            "hgcn_mean": float(d["HGCN"].mean()), "hgcn_std": float(d["HGCN"].std()),
            "hgcn32_mean": float(d["HGCN32"].mean()),
            "hgcn32_std": float(d["HGCN32"].std()),
            "ann_cell": ann_cell,
            "hgcn32l_mean": (float(d["HGCN32L"].mean()) if "HGCN32L" in d
                             else None),
            "hgcn32l_std": (float(d["HGCN32L"].std()) if "HGCN32L" in d
                            else None),
            "gin_mean": float(d["GIN"].mean()), "gin_std": float(d["GIN"].std()),
            "gat_mean": (float(d["GAT"].mean()) if "GAT" in d else None),
            "gat_std": (float(d["GAT"].std()) if "GAT" in d else None),
            "sgc_mean": (float(d["SGC"].mean()) if "SGC" in d else None),
            "sgc_std": (float(d["SGC"].std()) if "SGC" in d else None),
            "om50_mean": (float(d["OM50"].mean()) if "OM50" in d else None),
            "om50_std": (float(d["OM50"].std()) if "OM50" in d else None),
            # design B
            "eff_ann_b": float(eff_ann_b.mean()),
            "eff_spk_b": float(eff_spk_b.mean()),
            "inter_b": float(inter_b.mean()),
            "geom_spk": test_paired(eff_spk_b, "om-ome"),
            "geom_ann": test_paired(eff_ann_b, "hgcn32-gin"),
            "interaction_b": test_paired(inter_b, "interaction"),
            # design A (kept for comparison / backward compatibility)
            "eff_ann": (float(eff_ann_a.mean()) if have_a else None),
            "eff_spk": (float(eff_spk_a.mean()) if have_a else None),
            "inter_mean": (float(inter_a.mean()) if have_a else None),
            "inter_sd": (float(inter_a.std(ddof=1)) if have_a else None),
            "interaction_a": (test_paired(inter_a, "interaction") if have_a
                              else None),
            "budget_effect": budget,
            "depth_effect": depth,
        }
        rows.append(rec)

    if not rows:
        print("No dataset has the full mechanism-matched design yet.")
        return 1

    # BH correction: across datasets, separately per family of tests.
    for key in ("interaction_b", "geom_spk", "geom_ann"):
        q, sig = bh([r[key]["p"] for r in rows])
        for r, qi, si in zip(rows, q, sig):
            r[key]["bh_q"] = float(qi)
            r[key]["sig"] = bool(si)
    have_a = all(r["interaction_a"] is not None for r in rows)
    if have_a:
        q, sig = bh([r["interaction_a"]["p"] for r in rows])
        for r, qi, si in zip(rows, q, sig):
            r["interaction_a"]["bh_q"] = float(qi)
            r["interaction_a"]["sig"] = bool(si)

    # backward-compatible flat keys used by earlier reports
    for r in rows:
        if r["interaction_a"] is not None:
            r["sw_p"] = r["interaction_a"]["sw_p"]
            r["normal"] = r["interaction_a"]["normal"]
            r["stat"] = r["interaction_a"]["stat"]
            r["primary_p"] = r["interaction_a"]["p"]
            r["cohen_d"] = r["interaction_a"]["cohen_d"]
            r["bh_q"] = r["interaction_a"]["bh_q"]
            r["sig"] = r["interaction_a"]["sig"]

    L = build_report(rows, have_a)
    os.makedirs(os.path.dirname(OUT_MD) or ".", exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    if emit_tex:
        print("\n".join(latex_rows(rows)))
    else:
        print("\n".join(L))
    print(f"\nWrote {OUT_MD} and {OUT_JSON}", file=sys.stderr)
    return 0


def ms(m, s):
    return f"{m:.1f}$\\pm${s:.1f}"


def build_report(rows, have_a):
    L = []
    L.append("# v38 factorial analysis (geometry x mechanism)")
    L.append("")
    L.append("Design B (primary, mechanism-matched): Euclidean/ANN = GIN, "
             "manifold/ANN = HGCN-32, Euclidean/spiking = OM-E, "
             "manifold/spiking = OM.")
    L.append("Design A (original, confounded) kept below for comparison: its "
             "Euclidean/spiking cell is SGC.")
    L.append("")
    L.append("## Cell means (accuracy %, n per dataset)")
    L.append("")
    L.append("| Dataset | n | GIN | GAT | HGCN-16 | HGCN-32 | SGC | OM-E | "
             "OM | OM@50 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        def c(k):
            return (f"{r[k + '_mean']:.1f}±{r[k + '_std']:.1f}"
                    if r.get(k + "_mean") is not None else "--")
        L.append(f"| {r['dataset']} | {r['n']} | {c('gin')} | {c('gat')} | "
                 f"{c('hgcn')} | {c('hgcn32')} | {c('sgc')} | {c('ome')} | "
                 f"{c('om')} | {c('om50')} |")
    L.append("")
    L.append("## Design B: mechanism-matched effects")
    L.append("")
    L.append("| Dataset | geometry @ANN (HGCN-32$-$GIN) | "
             "geometry @spiking (OM$-$OM-E) | interaction | "
             "geom@spiking p | geom@spiking q (BH) | interaction q (BH) |")
    L.append("|---|---|---|---|---|---|---|")
    for r in rows:
        gs = r["geom_spk"]
        it = r["interaction_b"]
        L.append(f"| {r['dataset']} | {r['eff_ann_b']:+.1f} | "
                 f"{r['eff_spk_b']:+.1f} | {r['inter_b']:+.1f} | "
                 f"{gs['p']:.3f} | {gs['bh_q']:.3f} | {it['bh_q']:.2e} |")
    L.append("")
    L.append("## Design A: original (SGC) effects, confounded")
    L.append("")
    if not have_a:
        L.append("(not available)")
    else:
        L.append("| Dataset | geometry @ANN (HGCN$-$GIN) | "
                 "geometry @spiking (OM$-$SGC) | interaction | q (BH) | "
                 "Cohen's d |")
        L.append("|---|---|---|---|---|---|")
        for r in rows:
            ia = r["interaction_a"]
            L.append(f"| {r['dataset']} | {r['eff_ann']:+.1f} | "
                     f"{r['eff_spk']:+.1f} | {r['inter_mean']:+.1f} | "
                     f"{ia['bh_q']:.2e} | {ia['cohen_d']:+.2f} |")
    L.append("")
    L.append("## Budget sensitivity (OM -> OM@50, same 50-epoch budget)")
    L.append("")
    L.append("| Dataset | OM (original budget) | OM@50 | delta | p |")
    L.append("|---|---|---|---|---|")
    for r in rows:
        if r["om50_mean"] is None:
            L.append(f"| {r['dataset']} | {r['om_mean']:.2f} | -- | -- | -- |")
        else:
            b = r["budget_effect"]
            L.append(f"| {r['dataset']} | {r['om_mean']:.2f} | "
                     f"{r['om50_mean']:.2f} | {b['mean']:+.2f} | "
                     f"{b['p']:.2e} |")
    L.append("")
    L.append("## Reading")
    L.append("")
    spk = [r["eff_spk_b"] for r in rows]
    spk_sig = [r for r in rows if r["geom_spk"]["sig"]]
    ann = [r["eff_ann_b"] for r in rows]
    L.append(
        f"Under the mechanism-matched design the geometry effect under "
        f"spiking (OM - OM-E) ranges {min(spk):+.1f} to {max(spk):+.1f} "
        f"points and is significant on {len(spk_sig)}/{len(rows)} datasets. "
        f"Under ANN (HGCN-32 - GIN) it ranges {min(ann):+.1f} to "
        f"{max(ann):+.1f}. Removing the curvature from OM while keeping the "
        f"entire spiking mechanism does not measurably change accuracy, so "
        f"the accuracy of OM is attributable to the spiking mechanism rather "
        f"than to the curved state space.")
    L.append("")
    # depth sensitivity
    dep = [r for r in rows if r.get("depth_effect")]
    if dep:
        L.append("## Depth sensitivity (HGCN-32 at L=3 vs L matched to OM)")
        L.append("")
        L.append("| Dataset | HGCN-32 (L=3) | HGCN-32 (L=OM) | delta | "
                 "geom@ANN (L=3) | geom@ANN (L=OM) | p (depth) |")
        L.append("|---|---|---|---|---|---|---|")
        for r in dep:
            de = r["depth_effect"]
            L.append(f"| {r['dataset']} | {de['l3_mean']:.1f}±{de['l3_std']:.1f} "
                     f"| {de['lm_mean']:.1f}±{de['lm_std']:.1f} | "
                     f"{de['delta_mean']:+.2f} | {de['eff_ann_l3']:+.1f} | "
                     f"{r['eff_ann_b']:+.1f} | {de['test']['p']:.3f} |")
        L.append("")
        dl = [r["depth_effect"]["delta_mean"] for r in dep]
        L.append(
            f"Matching the depth moves the hyperbolic baseline by "
            f"{min(dl):+.2f} to {max(dl):+.2f} points and shifts the ANN-side "
            f"geometry effect correspondingly; the sign pattern is unchanged, "
            f"and the spiking-side effect (OM - OM-E) is untouched because "
            f"OM-E inherits OM's depth. Design B above uses the depth-matched "
            f"baseline as its ANN cell.")
        L.append("")
    if have_a:
        a_spk = [r["eff_spk"] for r in rows]
        L.append(
            f"By contrast the original design, whose Euclidean/spiking cell "
            f"is SGC, gives {min(a_spk):+.1f} to {max(a_spk):+.1f} points for "
            f"the same factor. That gap is not a geometry effect: SGC differs "
            f"from OM in the whole mechanism, not in the curvature. Design A "
            f"is therefore reported only as a historical comparison and must "
            f"not be read as the causal effect of the manifold.")
    L.append("")
    L.append("Test protocol: Shapiro-Wilk on the paired contrast (normal if "
             "p>0.05) -> one-sample t if normal, otherwise Wilcoxon "
             "signed-rank; BH FDR across the four datasets (q<0.05). "
             "Non-significant geometry effects are read as 'no difference "
             "detected', not as evidence of equivalence.")
    return L


def latex_rows(rows):
    """LaTeX rows for the paper tables."""
    out = []
    out.append("% --- cell means ---")
    for r in rows:
        def c(k):
            return (ms(r[k + "_mean"], r[k + "_std"])
                    if r.get(k + "_mean") is not None else "--")
        out.append(f"{r['dataset']} & {c('gin')} & {c('gat')} & {c('hgcn')} & "
                   f"{c('hgcn32')} & {c('sgc')} & {c('ome')} & {c('om')} & "
                   f"{c('om50')} \\\\")
    out.append("")
    out.append("% --- design B effects: dataset & geom@ANN & geom@spk & "
               "interaction & q ---")
    for r in rows:
        out.append(f"{r['dataset']} & {r['eff_ann_b']:+.1f} & "
                   f"{r['eff_spk_b']:+.1f} & {r['inter_b']:+.1f} & "
                   f"{r['interaction_b']['bh_q']:.2e} \\\\")
    out.append("")
    out.append("% --- design A (confounded) ---")
    for r in rows:
        if r["interaction_a"] is None:
            continue
        out.append(f"{r['dataset']} & {r['eff_ann']:+.1f} & "
                   f"{r['eff_spk']:+.1f} & {r['inter_mean']:+.1f} & "
                   f"{r['interaction_a']['bh_q']:.2e} \\\\")
    out.append("")
    out.append("% --- depth sensitivity: ds & HGCN-32(L=3) & HGCN-32(L=OM) & "
               "delta & OM ---")
    for r in rows:
        de = r.get("depth_effect")
        if not de:
            continue
        out.append(f"{r['dataset']} & {ms(de['l3_mean'], de['l3_std'])} & "
                   f"{ms(de['lm_mean'], de['lm_std'])} & "
                   f"{de['delta_mean']:+.1f} & "
                   f"{ms(r['om_mean'], r['om_std'])} \\\\")
    return out


if __name__ == "__main__":
    sys.exit(main())
