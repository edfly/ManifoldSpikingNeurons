"""v38 data audit: cross-validate JSON integrity & reproducibility."""

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
import json, glob, os
import numpy as np

R5 = _os.path.join(RESULTS_ROOT, "v38_5seeds")
R2 = _os.path.join(RESULTS_ROOT, "v38_s2")
R37 = _os.path.join(RESULTS_ROOT, "v37")

issues = []
oks = []


def ok(msg):
    oks.append(msg)


def issue(msg):
    issues.append(msg)


def load(fp):
    with open(fp) as f:
        return json.load(f)


print("=" * 70)
print("AUDIT 1: OM append consistency (v38_5seeds first 9 == v37 original)")
print("=" * 70)
for ds in ["MUTAG", "NCI1", "PROTEINS", "DD"]:
    v37 = load(os.path.join(R37, f"{ds}_OM.json"))
    v38 = load(os.path.join(R5, f"{ds}_OM.json"))
    a37, a38 = v37["accs"], v38["accs"]
    same_acc = a37 == a38[:9] or np.allclose(a37, a38[:9])
    same_sr = np.allclose(v37["srs"], v38["srs"][:9])
    same_ut = all(np.allclose(x, y) for x, y in zip(v37["uths"], v38["uths"][:9]))
    tag = "OK" if (same_acc and same_sr and same_ut) else "MISMATCH"
    if tag == "OK":
        ok(f"  {ds}_OM: v38 first 9 == v37 (acc/sr/uth)  n={len(a38)}")
    else:
        issue(f"  {ds}_OM: MISMATCH acc={same_acc} sr={same_sr} uth={same_ut}")
    # config consistency
    c37 = v37.get("config", {}); c38 = v38.get("config", {})
    for k in c37:
        if c37[k] != c38.get(k):
            issue(f"  {ds}_OM config diff: {k} {c37[k]} vs {c38.get(k)}")
    print(f"  {ds}: v37 n={len(a37)} accs={[round(x,1) for x in a37]} | "
          f"v38 n={len(a38)} {tag}")

print()
print("=" * 70)
print("AUDIT 2: sample alignment (every JSON same (seed, fold) order)")
print("=" * 70)
SEEDS5 = [42, 123, 456, 789, 2024]
for ds in ["MUTAG", "NCI1", "PROTEINS", "DD"]:
    files = sorted(glob.glob(os.path.join(R5, f"{ds}_*.json")))
    lens = {os.path.basename(f): len(load(f)["accs"]) for f in files}
    bad = {k: v for k, v in lens.items() if v != 15}
    if bad:
        issue(f"  {ds}: non-15 runs {bad}")
    else:
        ok(f"  {ds}: all {len(files)} methods have 15 runs")
    print(f"  {ds}: {lens}")

print()
print("=" * 70)
print("AUDIT 3: P2-7 S2 alignment with OM first 9 (seed-major fold-minor)")
print("=" * 70)
for ds in ["MUTAG", "NCI1", "PROTEINS", "DD"]:
    s2 = load(os.path.join(R2, f"{ds}_S2.json"))
    om = load(os.path.join(R5, f"{ds}_OM.json"))
    # S2 uses seeds 42/123/456 -> accs[0:3]=s42, [3:6]=s123, [6:9]=s456
    # OM first 9 must be identical ordering (v37 SEEDS=[42,123,456])
    n2 = len(s2["accs"])
    if n2 != 9:
        issue(f"  {ds}_S2: n={n2} != 9")
    cfg2 = s2.get("config", {})
    cfg_om = om.get("config", {})
    # compare only shared keys (s2 adds s_dim)
    shared = {k: v for k, v in cfg2.items()
              if k in cfg_om and k != "s_dim"}
    diff = {k: (v, cfg_om[k]) for k, v in shared.items() if v != cfg_om[k]}
    if diff:
        issue(f"  {ds}_S2 vs OM config diff: {diff}")
    else:
        ok(f"  {ds}_S2: n=9, config matches OM ({shared})")
    print(f"  {ds}_S2: n={n2} cfg_shared={shared} diff={diff or 'none'}")
    print(f"    S2 accs   = {[round(x,1) for x in s2['accs']]}")
    print(f"    OM first9 = {[round(x,1) for x in om['accs'][:9]]}")

print()
print("=" * 70)
print("AUDIT 4: degenerate policy records")
print("=" * 70)
for f in sorted(glob.glob(os.path.join(R5, "*.json"))):
    d = load(f)
    deg = d.get("degenerate", [])
    if deg:
        # verify reported accs[i] == rerun_acc when idx in degenerate
        ok_list = []
        for e in deg:
            idx = e["idx"]
            ok_list.append(np.isclose(d["accs"][idx], e["rerun_acc"]))
        allok = all(ok_list)
        deg_str = ", ".join(
            "idx%d s%d->s%d %.1f->%.1f" % (e["idx"], e["seed"],
                                           e["rerun_seed"], e["acc"],
                                           e["rerun_acc"]) for e in deg)
        print(f"  {os.path.basename(f)}: {len(deg)} degenerate "
              f"[{deg_str}] rerun-recorded={'OK' if allok else 'MISMATCH'}")
        if not allok:
            issue(f"  {os.path.basename(f)}: degenerate acc vs rerun_acc mismatch")
    else:
        print(f"  {os.path.basename(f)}: no degenerate runs")

print()
print("=" * 70)
print("AUDIT 5: SGC diagnostics recorded")
print("=" * 70)
for ds in ["MUTAG", "NCI1", "PROTEINS", "DD"]:
    d = load(os.path.join(R5, f"{ds}_SGC.json"))
    ex = d.get("extras", [])
    has_pm = all("peak_membrane" in e for e in ex) if ex else False
    nz = sum(1 for e in ex if e.get("spike_count", 0) == 0)
    print(f"  {ds}_SGC: extras n={len(ex)} has_peak_membrane={has_pm} "
          f"zero-spike runs={nz}/15")

print()
print("=" * 70)
print("AUDIT 6: log vs JSON spot check (last MUTAG_MSG & DD_SGC entries)")
print("=" * 70)
# MUTAG_MSG s2024 f3 expected from log: 66.1 (idx 14)
d = load(os.path.join(R5, "MUTAG_MSG.json"))
print(f"  MUTAG_MSG accs[14]={d['accs'][14]:.1f} (log: 66.1)")
d = load(os.path.join(R5, "DD_SGC.json"))
print(f"  DD_SGC accs[14]={d['accs'][14]:.1f} (log: 58.7)")

print()
print("=" * 70)
print(f"SUMMARY: {len(oks)} OK, {len(issues)} ISSUES")
print("=" * 70)
for m in oks:
    print(f"  [OK] {m}")
for m in issues:
    print(f"  [!!] {m}")
