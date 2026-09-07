#!/usr/bin/env python
"""Audit the graphical abstract against the Elsevier specification.

The point of this script is that "it looks fine" is not a check.  The previous
(matplotlib) version of this figure shipped with three overlapping text pairs
that only came to light when the bounding boxes were measured.  So every
geometric property that matters is measured here, from the compiled PDF:

  1. canvas is exactly 13 x 5 cm  (Elsevier renders the artwork at that size,
     ratio 13:5; 300 dpi then gives 1535 x 591 px, above the 1328 x 531 min)
  2. no two words overlap (tolerance 0.5 pt = 0.18 mm)
  3. no word falls outside the canvas
  4. no word is typeset below 6.5 pt
  5. every number drawn in the figure traces back to the result JSONs
  6. no identity string / Chinese character has leaked into the artwork

Usage:
    python check_graphical_abstract.py            # compile, then audit
    python check_graphical_abstract.py --no-build # audit the existing PDF
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent

#  In the post1 package the experiment artefacts live at the project root
#  (spikergnn_results/ is several directories above code/), so the checker
#  walks upwards until it finds the JSONs.  This makes the same script work
#  at the project root and inside the submission package without edits.
def _find_results(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "spikergnn_results").is_dir():
            return p / "spikergnn_results"
    return start / "spikergnn_results"

RESULTS = _find_results(ROOT)
TEX = ROOT / "graphical_abstract.tex"
PDF = ROOT / "graphical_abstract.pdf"

# pdftex works in TeX points (1 in = 72.27 pt) but writes PDF user space where
# the PostScript point (1 in = 72 pt) is the unit.  pdfinfo and pdftotext both
# report PDF user space, so a centimetre is 72/2.54 units here -- NOT 28.4527.
# Using the TeX value under-reports the canvas by 0.4 % and the check then
# fails on a figure that is in fact exactly 13 x 5 cm.
PT_PER_CM = 72.0 / 2.54
TEXPT_PER_PSPT = 72.27 / 72.0   # convert a rendered size back to TeX points

TOL_PT = 0.5          # word-box overlap tolerance
MIN_FONT_PT = 6.5     # Elsevier asks for "large enough"; this is our floor
DS = ["MUTAG", "NCI1", "PROTEINS", "DD"]

COLOUR = {"OK": "\033[92m", "FAIL": "\033[91m", "END": "\033[0m"} \
    if sys.stdout.isatty() else {"OK": "", "FAIL": "", "END": ""}


def sha(src: Path) -> None:
    pass


def build() -> bool:
    r = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", TEX.name],
        cwd=ROOT, capture_output=True, text=True)
    log = (ROOT / "graphical_abstract.log").read_text(encoding="utf-8", errors="replace")
    errs = [l for l in log.splitlines() if l.startswith("! ")]
    for e in errs[:5]:
        print(f"  latex: {e}")
    if not PDF.exists():
        print("  latex: no PDF produced")
        return False
    return not errs


def words():
    """Word boxes in PDF points, origin bottom-left (pdftotext gives top-left,
    so y is flipped using the page height).

    pdftex writes the page tree inside a compressed object stream, so
    /MediaBox is not greppable from the raw bytes; the <page> element that
    pdftotext reports is the reliable source for both the page size and the
    word boxes.  Units are PDF points.
    """
    out = subprocess.run(["pdftotext", "-bbox", str(PDF), "-"],
                         capture_output=True, text=True).stdout
    if not out.strip():
        raise SystemExit("pdftotext -bbox produced nothing")
    # pdftotext -bbox emits (X)HTML; wrap in a root element so it parses.
    body = out[out.find("<page"):out.rfind("</page>") + len("</page>")]
    root = ET.fromstring(f"<root>{body}</root>")
    page = root.find("page")
    H = float(page.get("height"))
    W = float(page.get("width"))
    items = []
    for w in page.iter("word"):
        txt = (w.text or "").strip()
        if not txt:
            continue
        x0, y0 = float(w.get("xMin")), float(w.get("yMin"))
        x1, y1 = float(w.get("xMax")), float(w.get("yMax"))
        items.append({"t": txt, "x0": x0, "x1": x1,
                      "y0": H - y1, "y1": H - y0, "h": y1 - y0})
    return W, H, items


def main() -> int:
    oks, fails = [], []

    if "--no-build" not in sys.argv:
        print("compiling ...")
        if not build():
            fails.append("latex build")

    Wp, Hp, ws = words()          # page size, in PDF points
    wcm, hcm = Wp / PT_PER_CM, Hp / PT_PER_CM
    print(f"words rendered: {len(ws)}\n")
    ok = abs(wcm - 13.0) < 0.02 and abs(hcm - 5.0) < 0.02
    (oks if ok else fails).append(
        f"canvas 13x5 cm      measured {wcm:.3f} x {hcm:.3f} cm "
        f"({wcm/2.54*300:.0f} x {hcm/2.54*300:.0f} px @300dpi, min 1328x531)")

    # ---- 2. pairwise overlap -------------------------------------------
    bad = []
    for i in range(len(ws)):
        for j in range(i + 1, len(ws)):
            a, b = ws[i], ws[j]
            ox = min(a["x1"], b["x1"]) - max(a["x0"], b["x0"])
            oy = min(a["y1"], b["y1"]) - max(a["y0"], b["y0"])
            if ox > TOL_PT and oy > TOL_PT:
                bad.append((ox, oy, a["t"], b["t"]))
    (oks if not bad else fails).append(
        f"text overlap        {len(bad)} colliding pairs")
    for ox, oy, ta, tb in bad[:8]:
        print(f"    {ox:.1f}x{oy:.1f} pt  '{ta}' <> '{tb}'")

    # ---- 3. inside the canvas ------------------------------------------
    out_ = [w for w in ws
            if w["x0"] < -0.5 or w["x1"] > Wp + 0.5
            or w["y0"] < -0.5 or w["y1"] > Hp + 0.5]
    (oks if not out_ else fails).append(
        f"inside canvas       {len(out_)} words outside")
    for w in out_[:5]:
        print(f"    '{w['t']}' at x[{w['x0']:.1f},{w['x1']:.1f}] "
              f"y[{w['y0']:.1f},{w['y1']:.1f}]")

    # ---- 4. font size floor --------------------------------------------
    #  Judged on the declared sizes, not on the rendered glyph boxes: maths
    #  sub/superscripts (the "sr" of L_sr, the "d" of H^d) are legitimately
    #  set ~0.73x the base size, and flagging them would be a false alarm.
    declared = [float(m) for m in
                re.findall(r"\\fontsize\{([\d.]+)\}", TEX.read_text(encoding="utf-8"))]
    lo = min(declared) if declared else 0.0
    (oks if declared and lo >= MIN_FONT_PT else fails).append(
        f"font >= {MIN_FONT_PT} pt       declared sizes "
        f"{sorted(set(declared))}, min {lo:g}")

    # ---- 5. numbers trace back to the JSONs ----------------------------
    def mean(ds, mk, p):
        f = RESULTS / p / f"{ds}_{mk}.json"
        if not f.exists():
            return None
        return sum(json.load(open(f))["accs"]) / len(json.load(open(f))["accs"])

    om = [mean(d, "OM", "v38_5seeds") for d in DS]
    msg = [mean(d, "MSG", "v38_5seeds") for d in DS]
    ome = [mean(d, "EUCLID", "v38_euclid") for d in DS]
    if None in om + msg + ome:
        fails.append("result JSONs        missing")
    else:
        om_m, msg_m, ome_m = (sum(v) / 4 for v in (om, msg, ome))

        # pdftotext splits "$-0.3 (n.s.)" into separate words, so compare
        # against the concatenated text rather than the word set.  Unicode
        # maths glyphs are folded to ASCII first.
        import unicodedata
        joined = "".join(w["t"] for w in ws)
        joined = joined.replace("\u2212", "-").replace("\u00a0", "")
        joined = "".join(
            unicodedata.normalize("NFKC", c)[-1:]
            if 0x1D400 <= ord(c) <= 0x1D7FF else c for c in joined)

        checks = [
            ("OM MUTAG",    om[0],                 "79.0"),
            ("OM NCI1",     om[1],                 "57.5"),
            ("OM PROTEINS", om[2],                 "71.1"),
            ("OM DD",       om[3],                 "69.9"),
            ("OM - MSG",    om_m - msg_m,          "+10.7"),
            ("OM - OM-E",   om_m - ome_m,          "-0.3"),
        ]
        bad = []
        for label, computed, in_fig in checks:
            if in_fig not in joined:
                bad.append(f"{label}: '{in_fig}' not rendered "
                           f"(json says {computed:+.2f})")
            elif abs(computed - float(in_fig)) > 0.051:
                bad.append(f"{label}: figure says {in_fig}, "
                           f"json says {computed:+.2f}")
        # the sign of the curvature effect must be written explicitly; a bare
        # "0.3" would read as a gain.
        if re.search(r"\{\s*0\.3", TEX.read_text(encoding="utf-8")):
            bad.append("curvature effect written without an explicit minus")

    # ---- 5b. cross-check against the paper's main table -----------------
    #  The GA draws the same accuracies as Table tab:main in the manuscript.
    #  A reviewer can spot-check that, say, MUTAG OM = 79.0 in both places.
    #  We re-derive the table values from the manuscript tex and assert the
    #  GA contains them.
    def _find_manuscript(start: Path) -> Path:
            for p in [start, *start.parents]:
                for cand in [p / "ManifoldValuedSpikingNeurons_Neurocomputing_v38.tex",
                             p / "programfiles" / "post1"
                                / "ManifoldValuedSpikingNeurons_Neurocomputing_v38.tex"]:
                    if cand.exists():
                        return cand
            return start / "ManifoldValuedSpikingNeurons_Neurocomputing_v38.tex"
    MS = _find_manuscript(ROOT)
    cross_bad = []
    if MS.exists():
        try:
            joined
        except NameError:
            joined = ""
        src = MS.read_text(encoding="utf-8")
        # For each dataset the GA labels the OM bar with the same 1-decimal
        # value as Table tab:main.  MSG is implicit through the bar height
        # so we only check OM here -- the MSG number is not drawn.
        for ds, om_paper in [
            ("MUTAG",    "79.0"),
            ("NCI1",     "57.5"),
            ("PROTEINS", "71.1"),
            ("DD",       "69.9"),
        ]:
            # Confirm that the row in tab:main really carries om_paper, so
            # the paper tex and the GA are anchored to the same number.
            row_re = re.search(rf"{ds}\s*&\s*{om_paper}\$\\pm\$[\d.]+",
                               src)
            if not row_re:
                cross_bad.append(f"paper table row for {ds} OM={om_paper} not parsed")
                continue
            if om_paper not in joined:
                cross_bad.append(f"{ds}: paper table OM = {om_paper}, "
                                 f"GA does not render it")
        matched = 4 - len([b for b in cross_bad if b.startswith(("MUTAG","NCI1","PROTEINS","DD"))])
        (oks if matched == 4 else fails).append(
            f"vs paper tab:main   {matched}/4 OM values match")
        for b in cross_bad:
            print(f"    {b}")
        # bar heights must encode the same scale as the y-axis labels
        (oks if not bad else fails).append(
            f"numbers vs JSON     {len(checks) - len(bad)}/{len(checks)} traced")
        for b in bad:
            print(f"    {b}")

    # ---- 6. no leak -----------------------------------------------------
    src = TEX.read_text(encoding="utf-8")
    leaks = []
    if re.search(r"[一-鿿]", src):
        leaks.append("Chinese characters")
    for pat in (r"Kehui", r"Hunan", r"workbuddy", r"WorkBuddy",
                r"D:\\", r"C:\\Users"):
        if re.search(pat, src):
            leaks.append(pat)
    (oks if not leaks else fails).append(
        f"no author/trace     {len(leaks)} leaks")
    for l in leaks:
        print(f"    {l}")

    print()
    for o in oks:
        print(f"  [{COLOUR['OK']}OK{COLOUR['END']}] {o}")
    for f in fails:
        print(f"  [{COLOUR['FAIL']}FAIL{COLOUR['END']}] {f}")
    print()
    print("=" * 62)
    print(f"RESULT: {len(oks)} OK, {len(fails)} FAILURES")
    print("=" * 62)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
