"""Verify Algorithm 1 (paper) against the implementation, line by line.

Motivation: the first three checks cover NUMBERS (tables <-> JSON) and
PARAMETERS (declared <-> runtime), but neither covers MECHANISM DESCRIPTION
(pseudocode <-> code logic). Five paper/code mechanism mismatches were found
by hand, the last and most serious one (claiming a Riemannian log/exp
aggregation that the code does not implement) only when the pseudocode was
compared against the source line by line. This script automates that
comparison: each Algorithm step declares an "implementation fingerprint"
(a regex that must be present in a given source file). If the code changes
in a way that breaks a fingerprint, the check fails and the pseudocode must
be updated.

This cannot be fully automatic (regexes must be written once per step), but
it converts an invisible manual audit into a repeatable one.

Usage:
    python verify_algorithm_mapping.py          # check
    python verify_algorithm_mapping.py --table  # print LaTeX rows for Appendix C
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
import os
import re
import sys

TEX = PAPER_TEX

NEURON = "spikergnn/neurons.py"
MODEL = "experiment_v37.py"

# (algorithm step as printed, source file, fingerprint regex, note)
# Fingerprints are deliberately tolerant of formatting but specific enough to
# break on a substantive mechanism change.
CHECKS = [
    ("p(0): project X onto M (expmap0)",
     MODEL, r"expmap0\s*\(", "hyperbolic component seeded at the origin"),
    ("u_mag <- 0 (membrane init)",
     NEURON, r"u_mag\s*=\s*torch\.zeros|_u_mag\s*=\s*torch\.zeros",
     "membrane state initialised per graph"),
    ("I(t): GIN aggregation on ambient coords",
     MODEL, r"GINConv\s*\(", "Euclidean GIN; NOT a log/exp operator"),
    ("u_mag <- alpha*u_mag + ||W I||/sqrt(d)",
     NEURON, r"u_mag\s*=\s*self\.alpha\s*\*\s*u_mag\s*\+\s*w_input_mag",
     "leaky integration of normalised synaptic magnitude"),
    ("input magnitude cap",
     NEURON, r"input_mag_cap", "absolute cap (not scaled by u_th)"),
    ("s(t): threshold crossing (ATan surrogate)",
     NEURON, r"surrogate\s*\(\s*u_mag\s*-\s*effective_threshold\s*\)",
     "hard threshold forward / ATan gradient backward"),
    ("membrane cap 1.5*u_th",
     NEURON, r"torch\.clamp\(\s*u_mag,\s*max\s*=\s*effective_threshold",
     "runaway-excitation guard"),
    ("p(t): geodesic update exp_{p}(-beta log_p(pbar) + gamma I)",
     NEURON, r"logmap\(\s*p_bar", "leak toward the running reference point "
     "(note: the call is line-wrapped, so the pattern must allow whitespace "
     "after the open paren)"),
    ("hard reset u_mag <- u_mag (1-s)",
     NEURON, r"u_mag\s*\*\s*\(\s*1\s*-\s*s_out\s*\)",
     "reset-to-zero on spike (soft delta_r reset removed)"),
    ("geodesic spike interpolation via sigma(alpha_s)",
     NEURON, r"spike_alpha", "learnable interpolation fraction"),
    ("L_sr = lambda (rbar - r*)^2",
     MODEL, r"LAMBDA_SR\s*\*\s*\(\s*sr\s*-\s*R_TARGET\s*\)\s*\*\*\s*2",
     "differentiable rate via the surrogate"),
    ("readout: mean-pool + linear classifier",
     MODEL, r"global_mean_pool", "plain mean pooling (no spike gating)"),
]

# Aggregation must NOT be Riemannian: this guards the scope note.
NEGATIVE_CHECKS = [
    ("aggregation contains no log/exp map (Euclidean by design)",
     MODEL, r"cv\s*\(\s*xs\[:, t, :\],\s*ei\s*\)",
     r"logmap|expmap",
     "Algorithm 1 line 3 is Euclidean on ambient coordinates"),
]


def strip_comments(src):
    """Drop comments AND docstrings so fingerprints match only live code.

    Two real instances motivated this:
      * ``u_mag -= delta_r`` no longer exists as code but survives in a ``#``
        comment ("Previous soft reset ... failed") and was matched by a naive
        regex;
      * the file-header docstrings added for the sync reminder contain words
        like ``logmap`` / ``reset_membrane``, which would equally satisfy a
        fingerprint if docstrings were not stripped.
    A fingerprint must only ever match executable code.
    """
    # Only the MODULE-level docstring is stripped (anchored at file start,
    # count=1). Stripping every """-block globally is unsafe: if any docstring
    # contains an unbalanced quote the pairing shifts and live code gets eaten
    # (verified: global stripping removed the real `logmap(p_bar_...)` call).
    src = re.sub(r"^\s*\"\"\"[\s\S]*?\"\"\"", "", src, count=1)
    out = []
    for line in src.split("\n"):
        if "#" in line:
            line = line.split("#", 1)[0]
        out.append(line)
    return "\n".join(out)


def main():
    for f in (TEX, NEURON, MODEL):
        if not os.path.exists(f):
            print(f"MISSING: {f}")
            return 1

    src = {NEURON: strip_comments(open(NEURON, encoding="utf-8").read()),
           MODEL: strip_comments(open(MODEL, encoding="utf-8").read())}

    print("=" * 74)
    print("ALGORITHM 1  <->  IMPLEMENTATION  MAPPING CHECK")
    print("=" * 74)
    print(f"{'algorithm step':52s} {'status':>9s}  source")
    print("-" * 74)
    fails = []
    for label, fpath, pattern, note in CHECKS:
        hit = re.search(pattern, src[fpath])
        ok = hit is not None
        print(f"{label[:52]:52s} {'OK' if ok else '**FAIL**':>9s}  {fpath}")
        if not ok:
            fails.append((label, fpath, pattern, note))

    print("-" * 74)
    # negative check: the aggregation line must exist and must not use log/exp
    for label, fpath, pos, neg, note in NEGATIVE_CHECKS:
        hit = re.search(pos, src[fpath])
        if not hit:
            print(f"{label[:52]:52s} {'**FAIL**':>9s}  {fpath} (line not found)")
            fails.append((label, fpath, pos, note))
            continue
        # inspect a window around the aggregation call
        seg = src[fpath][hit.start():hit.start() + 300]
        bad = re.search(neg, seg)
        ok = bad is None
        print(f"{label[:52]:52s} {'OK' if ok else '**FAIL**':>9s}  {fpath}")
        if not ok:
            fails.append((label, fpath, neg, note))

    print("-" * 74)
    if fails:
        print(f"\n{len(fails)} MISMATCH(ES) between Algorithm 1 and the code:")
        for label, fpath, pattern, note in fails:
            print(f"  - {label}\n      file: {fpath}\n      "
                  f"expected fingerprint: {pattern}\n      note: {note}")
        print("\nAction: update Algorithm 1 / the scope note in the paper, or "
              "change the code and rerun the affected experiments.")
        return 1

    print(f"\nALL {len(CHECKS) + len(NEGATIVE_CHECKS)} ALGORITHM STEPS MATCH "
          f"THE IMPLEMENTATION.")

    if "--table" in sys.argv:
        print("\n% ---- LaTeX rows for Appendix C ----")
        for label, fpath, pattern, note in CHECKS:
            print(f"{label} & \\texttt{{{os.path.basename(fpath)}}} & {note} \\\\")
    return 0


if __name__ == "__main__":
    sys.exit(main())
