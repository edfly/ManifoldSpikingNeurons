"""Verify every v38 paper number against its source JSON (reproducibility audit)."""

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
# The blind manuscript ships at the root of THIS package, so verify that
# copy rather than the non-blind one in the project root.
PAPER_TEX = _os.environ.get(
    "PAPER_TEX", _os.path.join(_ROOT,
                               "ManifoldValuedSpikingNeurons_Neurocomputing_v38.tex"))
# Likewise verify the supplementary that ships in THIS package.
SUPTEX = _os.environ.get(
    "SUPTEX", _os.path.join(_ROOT, "supplementary_material.tex"))
# ───────────────────────────────────────────────────────────────────
import json, os, re
import numpy as np

R5 = _os.path.join(RESULTS_ROOT, "v38_5seeds")
R2 = _os.path.join(RESULTS_ROOT, "v38_s2")
ROG = _os.path.join(RESULTS_ROOT, "v38_ogb")
RGAT = _os.path.join(RESULTS_ROOT, "v38_gat")
RH16 = _os.path.join(RESULTS_ROOT, "v38_hgcn")
RH32 = _os.path.join(RESULTS_ROOT, "v38_hgcn32")
RH32L = _os.path.join(RESULTS_ROOT, "v38_hgcn32L")
REUC = _os.path.join(RESULTS_ROOT, "v38_euclid")
R50 = _os.path.join(RESULTS_ROOT, "v38_om50")
RFAC = _os.path.join(RESULTS_ROOT, "v38_factorial.json")
TEX = PAPER_TEX

fails, oks = [], []


def load(fp):
    with open(fp) as f:
        return json.load(f)


def mstd(accs):
    return np.mean(accs), np.std(accs, ddof=0)


text = open(TEX, encoding="utf-8").read()

# ---------- 1. main table ----------
print("=" * 66)
print("1. MAIN TABLE (tab:main) vs v38_5seeds")
print("=" * 66)
main_block = re.search(r"\\label\{tab:main\}.*?\\bottomrule", text, re.S).group(0)
for ds in ["MUTAG", "NCI1", "PROTEINS", "DD"]:
    row = re.search(rf"^{ds} & (.+?) \\\\", main_block, re.M | re.S)
    if not row:
        fails.append(f"main table: row {ds} not found")
        continue
    cells = re.findall(r"(\\textbf\{)?([\d.]+)\\?\$(?:\\pm)\$?([\d.]+)", row.group(1))
    cells = [c for c in re.findall(r"([\d.]+)\$\\pm\$([\d.]+)", row.group(1))]
    methods = ["OM", "NS", "MSG", "SGC", "GIN", "GCN", "SAGE"]
    if len(cells) != 7:
        fails.append(f"main {ds}: parsed {len(cells)} cells, expected 7")
        continue
    for m, (tm, ts) in zip(methods, cells):
        accs = load(f"{R5}/{ds}_{m}.json")["accs"]
        mu, sd = mstd(accs)
        ok_mu = abs(mu - float(tm)) < 0.06
        ok_sd = abs(sd - float(ts)) < 0.06
        tag = "OK " if (ok_mu and ok_sd) else "FAIL"
        if ok_mu and ok_sd:
            oks.append(f"main {ds}/{m}: {tm}±{ts}")
        else:
            fails.append(f"main {ds}/{m}: paper {tm}±{ts} vs json {mu:.1f}±{sd:.1f}")
        print(f"  [{tag}] {ds:8s} {m:4s} paper={tm}±{ts}  json={mu:.1f}±{sd:.1f}")

# ---------- 2. sphere table ----------
print()
print("=" * 66)
print("2. SPHERE TABLE (tab:sphere) vs v38_s2")
print("=" * 66)
sph = re.search(r"\\label\{tab:sphere\}.*?\\bottomrule", text, re.S).group(0)
for ds in ["MUTAG", "NCI1", "PROTEINS", "DD"]:
    row = re.search(rf"^{ds} & (.+?) \\\\", sph, re.M).group(1)
    vals = re.findall(r"([\d.]+)\$\\pm\$([\d.]+)", row)
    om9 = load(f"{R5}/{ds}_OM.json")["accs"][:9]
    s2 = load(f"{R2}/{ds}_S2.json")["accs"]
    mu_om, sd_om = mstd(om9)
    mu_s2, sd_s2 = mstd(s2)
    ok1 = abs(mu_om - float(vals[0][0])) < 0.06 and abs(sd_om - float(vals[0][1])) < 0.06
    ok2 = abs(mu_s2 - float(vals[1][0])) < 0.06 and abs(sd_s2 - float(vals[1][1])) < 0.06
    diff = mu_s2 - mu_om
    print(f"  [{'OK ' if ok1 and ok2 else 'FAIL'}] {ds:8s} OM9={mu_om:.1f}±{sd_om:.1f} "
          f"S2={mu_s2:.1f}±{sd_s2:.1f}  Δ={diff:+.1f} (paper {row.split('&')[2].strip()})")
    (oks if ok1 and ok2 else fails).append(f"sphere {ds}: json Δ={diff:+.1f}")

# ---------- 3. OGB table ----------
print()
print("=" * 66)
print("3. OGB TABLE (tab:ogb) vs v38_ogb")
print("=" * 66)
for key, label in [("molhiv_OM", "OM"), ("molhiv_GIN", "GIN")]:
    d = load(f"{ROG}/{key}.json")
    mu, sd = np.mean(d["rocaucs"]), np.std(d["rocaucs"], ddof=0)
    in_tex = f"{mu:.4f}" in text
    print(f"  [{'OK ' if in_tex else 'FAIL'}] {label:4s} rocauc={mu:.4f}±{sd:.4f} "
          f"present_in_tex={in_tex}")
    (oks if in_tex else fails).append(f"ogb {label} {mu:.4f}")

# ---------- 4. energy table sr ----------
print()
print("=" * 66)
print("4. ENERGY TABLE (tab:energy) firing rates")
print("=" * 66)
en = re.search(r"\\label\{tab:energy\}.*?\\bottomrule", text, re.S).group(0)
for ds in ["MUTAG", "NCI1", "PROTEINS", "DD"]:
    row = re.search(rf"^{ds} & ([\d.]+) &", en, re.M).group(1)
    sr = float(np.mean(load(f"{R5}/{ds}_OM.json")["srs"]))
    ok = abs(sr - float(row)) < 0.0016
    print(f"  [{'OK ' if ok else 'FAIL'}] {ds:8s} paper r̄={row}  json r̄={sr:.3f}")
    (oks if ok else fails).append(f"energy {ds} sr {sr:.3f}")

# ---------- 5. per-fold table ----------
print()
print("=" * 66)
print("5. PER-FOLD TABLE (tab:perfold) row-by-row")
print("=" * 66)
pf = re.search(r"\\label\{tab:perfold\}.*?\\bottomrule", text, re.S).group(0)
mname = {"OM": "OM", "No-Spikes": "NS", "MSG": "MSG", "SGC": "SGC"}
cur_ds = None
for line in pf.split("\n"):
    m = re.match(r"^\s*(?:([A-Z0-9]+)\s+)?&\s+(OM|No-Spikes|MSG|SGC)\s+&\s+(.+?)\s*\\\\", line)
    if m:
        if m.group(1):
            cur_ds = m.group(1)
        meth = mname[m.group(2)]
        cells = [c.strip() for c in m.group(3).split("&")]
        vals = []
        for c in cells:
            vals += [float(x) for x in c.split(",")]
        accs = load(f"{R5}/{cur_ds}_{meth}.json")["accs"]
        match = len(vals) == len(accs) and all(
            abs(a - b) < 0.06 for a, b in zip(vals, accs))
        tag = "OK " if match else "FAIL"
        print(f"  [{tag}] {cur_ds:8s} {meth:3s} n={len(vals)} "
              f"maxdiff={max(abs(a-b) for a,b in zip(vals,accs)):.2f}")
        (oks if match else fails).append(f"perfold {cur_ds}/{meth}")

# ---------- 6. stats table vs v38_stats.json ----------
print()
print("=" * 66)
print("6. STATS TABLES vs analyze_v38_stats.py output")
print("=" * 66)
st = load(_os.path.join(RESULTS_ROOT, "v38_stats.json"))
for r in st["P2-1"]:
    if r["method"] == "MSG":
        in_tex = f"{r['diff_mean']:+.1f}".replace("+", "+") in text or \
                 f"+{r['diff_mean']:.1f}" in text
        print(f"  OM-MSG {r['dataset']:8s} Δ={r['diff_mean']:+.1f} "
              f"q={r['bh_q']:.2e}  in_tex={in_tex}")
for r in st["P2-1"]:
    if r["method"] == "NS":
        print(f"  OM-NS  {r['dataset']:8s} Δ={r['diff_mean']:+.1f} q={r['bh_q']:.3f}")

# ---------- 7. factorial cells (design B) ----------
print()
print("=" * 66)
print("7. FACTORIAL CELLS (tab:factorial) design B")
print("=" * 66)
fac = re.search(r"\\label\{tab:factorial\}.*?\\bottomrule", text, re.S).group(0)
# columns: GIN, HGCN-32, OM-E, OM
FAC_SRC = [
    ("GIN", lambda ds: f"{R5}/{ds}_GIN.json"),
    # Design B uses the DEPTH-MATCHED HGCN: OM runs L=2 on NCI1/PROTEINS/DD,
    # the first sweep instantiated L=3 everywhere.
    ("HGCN-32", lambda ds: f"{RH32L}/{ds}_HGCN.json"),
    ("OM-E", lambda ds: f"{REUC}/{ds}_EUCLID.json"),
    ("OM", lambda ds: f"{R5}/{ds}_OM.json"),
]
for ds in ["MUTAG", "NCI1", "PROTEINS", "DD"]:
    row = re.search(rf"^{ds} & (.+?) \\\\", fac, re.M)
    if not row:
        fails.append(f"factorial: row {ds} not found")
        continue
    cells = re.findall(r"([\d.]+)\$\\pm\$([\d.]+)", row.group(1))
    if len(cells) != 4:
        fails.append(f"factorial {ds}: parsed {len(cells)} cells, expected 4")
        continue
    for (label, fn), (tm, ts) in zip(FAC_SRC, cells):
        accs = load(fn(ds))["accs"]
        mu, sd = mstd(accs)
        ok = abs(mu - float(tm)) < 0.06 and abs(sd - float(ts)) < 0.06
        print(f"  [{'OK ' if ok else 'FAIL'}] {ds:8s} {label:8s} "
              f"paper={tm}±{ts}  json={mu:.1f}±{sd:.1f}")
        (oks if ok else fails).append(f"factorial {ds}/{label}")

# ---------- 8. factorial reference columns ----------
print()
print("=" * 66)
print("8. FACTORIAL REFERENCE (tab:factorial_ref)")
print("=" * 66)
fref = re.search(r"\\label\{tab:factorial_ref\}.*?\\bottomrule", text, re.S).group(0)
# columns: GAT, HGCN-16, SGC, OM@50  (OM@50 absent for MUTAG)
REF_SRC = [
    ("GAT", lambda ds: f"{RGAT}/{ds}_GAT.json"),
    ("HGCN-16", lambda ds: f"{RH16}/{ds}_HGCN.json"),
    ("SGC", lambda ds: f"{R5}/{ds}_SGC.json"),
    ("OM@50", lambda ds: f"{R50}/{ds}_OM50.json"),
]
for ds in ["MUTAG", "NCI1", "PROTEINS", "DD"]:
    row = re.search(rf"^{ds} & (.+?) \\\\", fref, re.M)
    if not row:
        fails.append(f"factorial_ref: row {ds} not found")
        continue
    cells = re.findall(r"([\d.]+)\$\\pm\$([\d.]+)", row.group(1))
    expect = 3 if ds == "MUTAG" else 4
    if len(cells) != expect:
        fails.append(f"factorial_ref {ds}: parsed {len(cells)}, expected {expect}")
        continue
    for (label, fn), (tm, ts) in zip(REF_SRC, cells):
        accs = load(fn(ds))["accs"]
        mu, sd = mstd(accs)
        ok = abs(mu - float(tm)) < 0.06 and abs(sd - float(ts)) < 0.06
        print(f"  [{'OK ' if ok else 'FAIL'}] {ds:8s} {label:8s} "
              f"paper={tm}±{ts}  json={mu:.1f}±{sd:.1f}")
        (oks if ok else fails).append(f"factorial_ref {ds}/{label}")

# ---------- 9. mechanism-matched contrasts (design B) ----------
print()
print("=" * 66)
print("9. DESIGN-B CONTRASTS (tab:interaction) vs v38_factorial.json")
print("=" * 66)
frows = {r["dataset"]: r for r in load(RFAC)}
inter = re.search(r"\\label\{tab:interaction\}.*?\\bottomrule", text, re.S).group(0)


def _strip(tok):
    """Remove LaTeX emphasis and spacing wrappers from a table cell."""
    tok = tok.strip()
    m = re.match(r"\\(?:textbf|textit|emph)\{(.+?)\}$", tok)
    if m:
        tok = m.group(1).strip()
    return tok.replace("$", "").replace("{", "").replace("}", "").strip()


def signed(tok):
    """Parse '$-$2.2', '+1.0', '0.57', '\\textbf{+10.2}' into a float."""
    tok = _strip(tok)
    neg = tok.startswith("$-$") or tok.startswith("-") or tok.startswith("−")
    tok = tok.replace("$-$", "").replace("−", "").replace("+", "").replace("-", "")
    v = float(tok)
    return -v if neg else v


def sci(tok):
    """Parse either a plain decimal ('0.41') or '$7.5\\\\times10^{-12}$'.

    The exponent sign must be captured inside the group; a leading '-?'
    outside it would swallow the minus and turn 10^{-12} into 10^{12}.
    """
    tok_raw = re.sub(r"^\\(?:textbf|textit|emph)\{(.+?)\}$", r"\1", tok.strip())
    tok_raw = tok_raw.replace("$", "").strip()
    m = re.search(r"([\d.]+)\s*\\times\s*10\^\{?(-?\d+)\}?", tok_raw)
    if m:
        return float(m.group(1)) * 10.0 ** float(m.group(2))
    return float(_strip(tok))


for ds in ["MUTAG", "NCI1", "PROTEINS", "DD"]:
    row = re.search(rf"^{ds} & (.+?) \\\\", inter, re.M)
    if not row:
        fails.append(f"tab:interaction: row {ds} not found")
        continue
    toks = [t.strip() for t in row.group(1).split("&")]
    if len(toks) != 5:
        fails.append(f"tab:interaction {ds}: {len(toks)} cells, expected 5")
        continue
    r = frows[ds]
    # columns: geom@ANN, geom@spk, q(geom spk), interaction, q(inter)
    checks = [
        ("geom@ANN", signed(toks[0]), r["eff_ann_b"], 0.06),
        ("geom@spk", signed(toks[1]), r["eff_spk_b"], 0.06),
        ("q_geomspk", float(toks[2]), r["geom_spk"]["bh_q"], 0.006),
        ("inter", signed(toks[3]), r["inter_b"], 0.06),
    ]
    for label, paper, truth, tol in checks:
        ok = abs(paper - truth) < tol
        print(f"  [{'OK ' if ok else 'FAIL'}] {ds:8s} {label:10s} "
              f"paper={paper:+.2f}  json={truth:+.2f}")
        (oks if ok else fails).append(f"interaction {ds}/{label}")
    # q(inter) is typeset in scientific notation; compare on a relative tolerance
    q_paper = sci(toks[4])
    q_json = r["interaction_b"]["bh_q"]
    ok_q = abs(q_paper - q_json) <= 0.02 * max(abs(q_json), 1e-12)
    print(f"  [{'OK ' if ok_q else 'FAIL'}] {ds:8s} {'q_inter':10s} "
          f"paper={q_paper:.2e}  json={q_json:.2e}")
    (oks if ok_q else fails).append(f"interaction {ds}/q_inter")

# ---------- 10. confounded design A contrasts ----------
print()
print("=" * 66)
print("10. DESIGN-A CONTRASTS (tab:interaction_a) [confounded, retained]")
print("=" * 66)
ia = re.search(r"\\label\{tab:interaction_a\}.*?\\bottomrule", text, re.S).group(0)
for ds in ["MUTAG", "NCI1", "PROTEINS", "DD"]:
    row = re.search(rf"^{ds} & (.+?) \\\\", ia, re.M)
    if not row:
        fails.append(f"tab:interaction_a: row {ds} not found")
        continue
    toks = [t.strip() for t in row.group(1).split("&")]
    if len(toks) != 3:
        fails.append(f"tab:interaction_a {ds}: {len(toks)} cells, expected 3")
        continue
    r = frows[ds]
    for label, tok, truth in [("geom@ANN", toks[0], r["eff_ann"]),
                              ("geom@spk", toks[1], r["eff_spk"]),
                              ("inter", toks[2], r["inter_mean"])]:
        ok = abs(signed(tok) - truth) < 0.06
        print(f"  [{'OK ' if ok else 'FAIL'}] {ds:8s} {label:9s} "
              f"paper={signed(tok):+.1f}  json={truth:+.1f}")
        (oks if ok else fails).append(f"interaction_a {ds}/{label}")

# ---------- 11. budget sensitivity ----------
print()
print("=" * 66)
print("11. BUDGET SENSITIVITY (tab:budget) vs v38_om50")
print("=" * 66)
bud = re.search(r"\\label\{tab:budget\}.*?\\bottomrule", text, re.S).group(0)
for ds in ["MUTAG", "NCI1", "PROTEINS", "DD"]:
    row = re.search(rf"^{ds} & (.+?) \\\\", bud, re.M)
    if not row:
        fails.append(f"tab:budget: row {ds} not found")
        continue
    toks = [t.strip() for t in row.group(1).split("&")]
    if ds == "MUTAG":
        # no OM@50 row: OM already used 50 epochs
        om = load(f"{R5}/{ds}_OM.json")["accs"]
        ok = abs(float(toks[0]) - np.mean(om)) < 0.06
        print(f"  [{'OK ' if ok else 'FAIL'}] {ds:8s} OM(protocol)="
              f"{toks[0]}  json={np.mean(om):.1f}  (OM@50: n/a)")
        (oks if ok else fails).append(f"budget {ds}/OM")
        continue
    om = load(f"{R5}/{ds}_OM.json")["accs"]
    o50 = load(f"{R50}/{ds}_OM50.json")["accs"]
    delta = np.mean(o50) - np.mean(om)
    checks = [
        ("OM", float(toks[0]), np.mean(om), 0.06),
        ("OM@50", float(toks[1]), np.mean(o50), 0.06),
        ("delta", signed(toks[2]), delta, 0.06),
    ]
    for label, paper, truth, tol in checks:
        ok = abs(paper - truth) < tol
        print(f"  [{'OK ' if ok else 'FAIL'}] {ds:8s} {label:6s} "
              f"paper={paper:+.2f}  json={truth:+.2f}")
        (oks if ok else fails).append(f"budget {ds}/{label}")

# ---------- 12. depth sensitivity of the hyperbolic baseline ----------
print()
print("=" * 66)
print("12. DEPTH SENSITIVITY (tab:factorial_depth)")
print("=" * 66)
dep = re.search(r"\\label\{tab:factorial_depth\}.*?\\bottomrule", text, re.S)
if not dep:
    fails.append("tab:factorial_depth not found in the manuscript")
else:
    dep = dep.group(0)
    for ds in ["MUTAG", "NCI1", "PROTEINS", "DD"]:
        row = re.search(rf"^{ds} & (.+?) \\\\", dep, re.M)
        if not row:
            fails.append(f"tab:factorial_depth: row {ds} not found")
            continue
        toks = [t.strip() for t in row.group(1).split("&")]
        if len(toks) != 4:
            fails.append(f"tab:factorial_depth {ds}: {len(toks)} cells, "
                         f"expected 4")
            continue
        l3 = mstd(load(f"{RH32}/{ds}_HGCN.json")["accs"])
        lm = mstd(load(f"{RH32L}/{ds}_HGCN.json")["accs"])
        om = mstd(load(f"{R5}/{ds}_OM.json")["accs"])
        # column 3 is the delta between the two HGCN variants
        checks = [
            ("HGCN L=3", toks[0], l3),
            ("HGCN L=OM", toks[1], lm),
            ("OM", toks[3], om),
        ]
        for label, tok, (mu, sd) in checks:
            cells = re.findall(r"([\d.]+)\$\\pm\$([\d.]+)", tok)
            if not cells:
                fails.append(f"factorial_depth {ds}/{label}: unparsable")
                continue
            ok = (abs(mu - float(cells[0][0])) < 0.06
                  and abs(sd - float(cells[0][1])) < 0.06)
            print(f"  [{'OK ' if ok else 'FAIL'}] {ds:8s} {label:10s} "
                  f"paper={cells[0][0]}±{cells[0][1]}  json={mu:.1f}±{sd:.1f}")
            (oks if ok else fails).append(f"factorial_depth {ds}/{label}")
        delta_paper = signed(toks[2])
        delta_json = lm[0] - l3[0]
        ok = abs(delta_paper - delta_json) < 0.06
        print(f"  [{'OK ' if ok else 'FAIL'}] {ds:8s} {'delta':10s} "
              f"paper={delta_paper:+.2f}  json={delta_json:+.2f}")
        (oks if ok else fails).append(f"factorial_depth {ds}/delta")

# ---------- 13. supplementary: per-fold table for the Section 5.7 controls ----
print()
print("=" * 66)
print("13. SUPPLEMENTARY per-fold table (Section C controls)")
print("=" * 66)
if not os.path.exists(SUPTEX):
    fails.append(f"supplementary not found at {SUPTEX}")
    print(f"  [FAIL] supplementary not found: {SUPTEX}")
else:
    sup = open(SUPTEX, encoding="utf-8").read()
    block = re.search(r"\\subsection\*\{C\..*?\\bottomrule", sup, re.S)
    if not block:
        fails.append("supplementary section C table not found")
        print("  [FAIL] section C table not found")
    else:
        SUP_SRC = {
            "OM-E": lambda ds: f"{REUC}/{ds}_EUCLID.json",
            "HGCN-16": lambda ds: f"{RH16}/{ds}_HGCN.json",
            "HGCN-32": lambda ds: f"{RH32}/{ds}_HGCN.json",
            "HGCN-32L": lambda ds: f"{RH32L}/{ds}_HGCN.json",
            "GAT": lambda ds: f"{RGAT}/{ds}_GAT.json",
            "OM@50": lambda ds: f"{R50}/{ds}_OM50.json",
        }
        cur_ds = None
        n_rows = 0
        for line in block.group(0).split("\n"):
            # dataset column is empty on continuation rows, as in tab:perfold
            m = re.match(r"^\s*(?:([A-Z0-9]+)\s+)?&\s+(\S+)\s+&\s+(.+?)\s*\\\\",
                         line)
            if not m:
                continue
            if m.group(1):
                cur_ds = m.group(1)
            name, rest = m.group(2), m.group(3)
            if name not in SUP_SRC or cur_ds is None:
                continue
            vals = []
            for cell in rest.split("&"):
                vals += [float(x) for x in cell.split(",")]
            accs = load(SUP_SRC[name](cur_ds))["accs"]
            match = (len(vals) == len(accs)
                     and all(abs(a - b) < 0.06 for a, b in zip(vals, accs)))
            n_rows += 1
            md = max(abs(a - b) for a, b in zip(vals, accs)) if len(vals) == len(accs) else float("nan")
            print(f"  [{'OK ' if match else 'FAIL'}] {cur_ds:8s} {name:9s} "
                  f"n={len(vals)} maxdiff={md:.2f}")
            (oks if match else fails).append(f"supplementary {cur_ds}/{name}")
        if n_rows < 22:
            fails.append(f"supplementary section C: only {n_rows} rows parsed, "
                         f"expected 22")
            print(f"  [FAIL] only {n_rows}/22 rows parsed")

# ---------- 14. front-matter limits (journal submission rules) ----------
print()
print("=" * 66)
print("14. FRONT MATTER (abstract / keywords / highlights limits)")
print("=" * 66)


def _rendered(s):
    """Approximate typeset text: strip markup, keep visible characters."""
    s = re.sub(r"\\%", "%", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = s.replace("$", "").replace("{", "").replace("}", "")
    s = s.replace("~", " ").replace("--", "-")
    s = s.replace("^", "").replace("*", "")
    return re.sub(r"\s+", " ", s).strip()


def _words(tex):
    """Word count of the typeset text (inline maths counts as one word)."""
    s = re.sub(r"\$[^$]*\$", " W ", tex)
    s = re.sub(r"\\(?:cite|ref|label)\{[^}]*\}", "", s)
    s = re.sub(r"\\(?:textbf|textit|emph|textsc)\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    s = s.replace("~", " ").replace("--", " ")
    return len([w for w in s.split() if any(c.isalnum() for c in w)])


def _check(label, ok, detail):
    print(f"  [{'OK ' if ok else 'FAIL'}] {label:34s} {detail}")
    (oks if ok else fails).append(f"front matter: {label}")


m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.S)
if not m:
    _check("abstract present", False, "no abstract environment")
else:
    n = _words(m.group(1))
    _check("abstract <= 250 words", n <= 250, f"{n} words")

m = re.search(r"\\begin\{keywords\}(.*?)\\end\{keywords\}", text, re.S)
if not m:
    _check("keywords present", False, "no keywords environment")
else:
    ks = [x.strip() for x in m.group(1).split("\\sep")]
    _check("keywords count 5-7", 5 <= len(ks) <= 7, f"{len(ks)} keywords")

m = re.search(r"\\begin\{highlights\}(.*?)\\end\{highlights\}", text, re.S)
if not m:
    _check("highlights present", False, "no highlights environment")
else:
    items = [re.sub(r"\\item\s*", "", x).strip()
             for x in m.group(1).split("\\item") if x.strip()]
    _check("highlights count 3-5", 3 <= len(items) <= 5,
           f"{len(items)} bullets")
    over = [(i, len(_rendered(s))) for i, s in enumerate(items, 1)
            if len(_rendered(s)) > 85]
    _check("each highlight <= 85 chars", not over,
           f"longest {max(len(_rendered(s)) for s in items)} chars"
           + (f", over: {over}" if over else ""))

print()
print("=" * 66)
print(f"RESULT: {len(oks)} OK, {len(fails)} FAILURES")
print("=" * 66)
for f in fails:
    print(f"  [!!] {f}")
