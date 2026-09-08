#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_is_submission.py -- machine checks for the *Information Sciences*
version of the manuscript (MVSN_InformationSciences.tex), derived from the
Neurocomputing blind manuscript (MVSN.tex).

Why this exists
---------------
The whole point of the rewrite was to change *framing* without touching
*content*.  Prose edits are exactly the kind of change where a number can be
"tidied" by accident and no one notices, because the PDF still compiles and
still looks right.  So every claim below is checked against an artifact:
the abstract word count against the rendered PDF, the number set against the
source manuscript and the result JSONs, and the citation keys against the
original file.

Usage
-----
    python verify_is_submission.py                 # checks
    python verify_is_submission.py --recompile     # pdflatex x2 first

Exit code 0 = all checks passed.
"""

import io
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STEM = "MVSN_InformationSciences"
_NEW_NAME = STEM + ".tex"
_BASE_NAME = "MVSN.tex"
TRACE_FILE_NAME = ".trace_patterns"


def _candidates(names):
    """Locate inputs without hard-coding post/ vs post1/.

    The same script is shipped in both packages, so it cannot assume where
    the manuscript, its anonymous base version and the result JSONs live.
    Nearest ancestor wins; the sibling ``post``/``post1`` directories of each
    ancestor are considered too, because the base manuscript only exists in
    post1 while this script also ships in post/code.
    """
    seen, out = set(), []
    anc, d = [], os.path.abspath(HERE)
    for _ in range(4):
        anc.append(d)
        d = os.path.dirname(d)
    for a in anc:
        parent = os.path.dirname(a)
        for base in (a, os.path.join(parent, "post1"), os.path.join(parent, "post")):
            for n in names:
                p = os.path.join(base, n)
                if p not in seen:
                    seen.add(p)
                    out.append(p)
    return out


def _first(names, what):
    for p in _candidates(names):
        if os.path.exists(p):
            return p
    raise SystemExit("cannot locate %s; looked in:\n  %s"
                     % (what, "\n  ".join(_candidates(names))))


NEW = _first([_NEW_NAME], "the Information Sciences manuscript")
BASE = _first([_BASE_NAME], "the Neurocomputing base manuscript (MVSN.tex)")
RESULTS = _first(["results"], "the results directory")
WORKDIR = os.path.dirname(NEW)

ABSTRACT_LIMIT = 200          # Information Sciences guide for authors
HIGHLIGHT_LIMIT = 85          # Elsevier highlight length (rendered characters)

OK, FAIL, SKIP = [], [], []


def check(tag, cond, detail=""):
    (OK if cond else FAIL).append(tag)
    print("%-4s %-46s %s" % ("ok" if cond else "FAIL", tag, detail))


def skip(tag, why):
    SKIP.append(tag)
    print("%-4s %-46s %s" % ("skip", tag, why))


def trace_patterns():
    """Development-environment names that must not survive into a copy.

    Read from a sidecar file next to the manuscript (or one level up) rather
    than hard-coded: the external package must not contain the tool names it
    is supposed to be checking for.
    """
    for d in (WORKDIR, os.path.dirname(WORKDIR), HERE, os.path.dirname(HERE)):
        f = os.path.join(d, TRACE_FILE_NAME)
        if os.path.exists(f):
            pats = [ln.strip() for ln in read(f).split("\n")]
            pats = [p for p in pats if p and not p.startswith("#")]
            return pats, f
    return None, None


def read(path, errors="strict"):
    with io.open(path, encoding="utf-8", errors=errors) as f:
        return f.read()


def strip_comments(text):
    out = []
    for line in text.split("\n"):
        line = re.sub(r"(?<!\\)%.*$", "", line)
        out.append(line)
    return "\n".join(out)


def slice_between(text, start, end):
    i = text.index(start)
    j = text.index(end, i)
    return text[i:j]


NUM = re.compile(r"\d+(?:[.,]\d+)?")


def numbers(text):
    return NUM.findall(text)


def main():
    recompile = "--recompile" in sys.argv
    new_raw = read(NEW)
    base_raw = read(BASE)
    new = strip_comments(new_raw)
    base = strip_comments(base_raw)

    # ---------------------------------------------------------- [0] compile
    log = os.path.join(WORKDIR, STEM + ".log")
    if recompile or not os.path.exists(log):
        for _ in range(2):
            subprocess.call(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                 os.path.basename(NEW)],
                cwd=WORKDIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(log):
        logtxt = read(log, errors="replace")
        errs = len(re.findall(r"^!", logtxt, re.M))
        und = len(re.findall(r"LaTeX Warning: (?:Reference|Citation)", logtxt))
        check("[0a] compiles without errors", errs == 0, "%d errors" % errs)
        check("[0b] no undefined refs/citations", und == 0, "%d warnings" % und)
        over = [float(m) for m in
                re.findall(r"Overfull \\hbox \((\d+\.\d+)pt too wide\)", logtxt)]
        big = [x for x in over if x > 5.0]
        check("[0c] no NEW large overfull hbox",
              len(big) <= 1,
              "%d overfull >5pt (1 is the cas-dc \\maketitle template artifact)"
              % len(big))
    else:
        check("[0] log present", False, "run with --recompile")

    # ------------------------------------------------- [1] abstract length
    pdf = os.path.join(WORKDIR, STEM + ".pdf")
    words = None
    if os.path.exists(pdf):
        try:
            txt = subprocess.run(["pdftotext", "-f", "1", "-l", "2", pdf, "-"],
                                 capture_output=True, text=True).stdout
            i = txt.index("Graph-structured records")
            j = txt.index("spiking mechanism supplies the accuracy.") + 39
            ab = re.sub(r"\s+", " ", txt[i:j].replace("\n", " "))
            words = len(ab.split())
        except Exception as exc:            # pragma: no cover
            check("[1] abstract word count", False, "pdftotext failed: %s" % exc)
    check("[1] abstract <= %d words" % ABSTRACT_LIMIT,
          words is not None and words <= ABSTRACT_LIMIT,
          "%s words" % words)

    # ------------------------------------------------------- [2] highlights
    hl = re.search(r"\\begin\{highlights\}(.*?)\\end\{highlights\}", new, re.S)
    items = [x.strip() for x in re.findall(r"\\item (.+)", hl.group(1))]
    check("[2a] highlights present (3-5)", 3 <= len(items) <= 5,
          "%d items" % len(items))
    plain = []
    for it in items:
        t = it.replace("\\emph", "").replace("\\textbf", "")
        t = re.sub(r"\$([^$]*)\$", r"\1", t)
        t = re.sub(r"\\[a-zA-Z]+", " ", t)
        t = re.sub(r"[{}\\~^_]", "", t).strip()
        plain.append(t)
    over = [(t, len(t)) for t in plain if len(t) > HIGHLIGHT_LIMIT]
    check("[2b] every highlight <= %d chars" % HIGHLIGHT_LIMIT, not over,
          "; ".join("%d" % n for _, n in over) or "max %d" % max(
              len(t) for t in plain))

    # -------------------------------------------------------- [3] keywords
    kw = re.search(r"\\begin\{keywords\}(.*?)\\end\{keywords\}", new, re.S)
    nkw = len([k for k in kw.group(1).split("\\sep") if k.strip()])
    check("[3] keywords <= 6", nkw <= 6, "%d keywords" % nkw)

    # ----------------------------------------- [4] no anonymity / tool trace
    # [4a] journal-related residue -- hard-coded, applies to every copy.
    for pat in ("Anonymous", "withheld", "anonymized for review"):
        check("[4a] no '%s' residue" % pat, pat not in new)

    # [4b] development-environment residue.  The patterns are read from a
    # sidecar file that is deliberately NOT shipped in the external package:
    # hard-coding the tool names would put them into the very file that is
    # supposed to be free of them.  When the file is absent the check reports
    # SKIP explicitly -- never a silent pass.
    pats, pat_src = trace_patterns()
    if pats is None:
        skip("[4b] development-environment traces",
             "no %s file in this package" % TRACE_FILE_NAME)
    else:
        for pat in pats:
            check("[4b] no '%s' trace" % pat, pat not in new,
                  "from %s" % os.path.basename(pat_src))

    # --------------------------------------- [5] structure vs base unchanged
    sec_new = re.findall(r"\\(?:sub)?section\*?\{(.+?)\}", new)
    sec_base = re.findall(r"\\(?:sub)?section\*?\{(.+?)\}", base)
    check("[5a] section titles identical", sec_new == sec_base,
          "%d sections" % len(sec_new))
    for env in ("table", "figure", "equation", "algorithm"):
        cn = len(re.findall(r"\\begin\{%s\*?\}" % env, new))
        cb = len(re.findall(r"\\begin\{%s\*?\}" % env, base))
        check("[5b] %s env count unchanged" % env, cn == cb, "%d vs %d" % (cn, cb))
    lab_new = set(re.findall(r"\\label\{(.+?)\}", new))
    lab_base = set(re.findall(r"\\label\{(.+?)\}", base))
    check("[5c] labels unchanged", lab_new == lab_base, "%d labels" % len(lab_new))

    # --------------------------------------------- [6] citation keys intact
    cit_new = re.findall(r"\\cite[a-z]*\{(.+?)\}", new)
    cit_base = re.findall(r"\\cite[a-z]*\{(.+?)\}", base)
    keys_new = sorted(k for c in cit_new for k in c.split(","))
    keys_base = sorted(k for c in cit_base for k in c.split(","))
    check("[6a] citation key multiset unchanged", keys_new == keys_base,
          "%d keys" % len(keys_new))
    bib_new = sorted(re.findall(r"\\bibitem(?:\[[^\]]*\])?\{(.+?)\}", new))
    bib_base = sorted(re.findall(r"\\bibitem(?:\[[^\]]*\])?\{(.+?)\}", base))
    check("[6b] bibliography entries unchanged", bib_new == bib_base,
          "%d entries" % len(bib_new))

    # ------------------------------------ [7] numbers in experiments intact
    # [7] covers the scientific content only: Experiments through Conclusion.
    # The administrative block after it (Data Availability, Reproducibility
    # Statement, declarations) carries *meta* numbers -- the check counts of
    # the verification suites -- which legitimately change when a suite grows.
    # Those counts are guarded by [7b] instead, which is a stronger check than
    # incidental string equality.
    new_exp = slice_between(new, r"\section{Experiments}",
                            r"\section*{Data Availability}")
    base_exp = slice_between(base, r"\section{Experiments}",
                             r"\section*{Data Availability}")
    check("[7] every result number unchanged",
          numbers(new_exp) == numbers(base_exp),
          "%d numeric tokens" % len(numbers(new_exp)))

    # [7b] the Reproducibility Statement must state the real suite sizes.
    SUITES = [("audit_v38_data.py", "12"),
              ("verify_v38_numbers.py", "170"),
              ("verify_hyperparams.py", "18"),
              ("verify_algorithm_mapping.py", "13"),
              ("verify_math.py", "38"),
              ("check_graphical_abstract.py", "7")]
    expected = "/".join(n for _f, n in SUITES)
    m = re.search(r"All six suites currently pass\s*\((\d+(?:/\d+){5})\)", new)
    check("[7b] reproducibility statement counts", bool(m) and m.group(1) == expected,
          (m.group(1) if m else "not found") + " (expected %s)" % expected)
    code_dir = os.path.join(WORKDIR, "code")
    missing = [f for f, _n in SUITES
               if not os.path.exists(os.path.join(code_dir, f))]
    check("[7b] every suite script is present", not missing,
          "missing: " + ", ".join(missing) if missing else "6 files")

    # ------------------------- [8] no fabricated number in rewritten prose
    # Any number in the reframed regions must already exist in the source
    # manuscript or in one of the result JSONs.  This is the guard that a
    # "harmless" wording edit cannot smuggle in a new figure.
    idx = set()
    for root, _dirs, files in os.walk(RESULTS):
        for fn in files:
            if not fn.endswith(".json"):
                continue
            try:
                data = json.load(io.open(os.path.join(root, fn), encoding="utf-8"))
            except Exception:
                continue
            idx.update(NUM.findall(json.dumps(data)))
    known = set(numbers(base)) | set(idx)
    prose = slice_between(new, r"\section{Introduction}",
                          r"\section*{Declaration of Generative AI")
    unknown = sorted({n for n in numbers(prose) if n not in known})
    check("[8] no number in reframed prose is unsourced", not unknown,
          ", ".join(unknown[:10]) if unknown else "all traceable")

    # ------------------------------------------------------------- report
    total = "%d OK, %d FAILED" % (len(OK), len(FAIL))
    if SKIP:
        total = "%d OK, %d SKIPPED, %d FAILED" % (len(OK), len(SKIP), len(FAIL))
    print("\n" + total)
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
