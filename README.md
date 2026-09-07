# Manifold-Valued Spiking Neurons — Code and Verification Suite

Reference implementation and machine-verification suite for the manuscript
*“Manifold-Valued Spiking Neurons for Graph Representation Learning:
On-Manifold Dynamics, Spike-Rate Regularization, and the Limits of Curvature”*.

The model introduces a spiking neuron whose membrane state evolves **on** a
geodesically complete Riemannian manifold: the leaky integrate-and-fire update
is written entirely in exponential and logarithmic maps, so the state never
leaves the manifold and no tangent-space projection is introduced. A learnable
threshold with a quadratic spike-rate regulariser
`L_sr = lambda_sr * (r_bar - r*)^2` fixes the operating point of the code, and
spike-gated geodesic interpolation keeps the update manifold-consistent.

## Repository layout

```
code/
  spikergnn/            core package: neuron, manifold ops, message passing
  msg_baseline/         tangent-space MSG baseline used for comparison
  experiment_v38_*.py   experiment drivers, one per sweep (TU / OGB / controls)
  verify_*.py           machine checks, see "Verification" below
  check_all.sh          runs the full verification suite
  make_figures_v38.py   figure generation
  graphical_abstract.*  Elsevier graphical abstract (source + PDF)
  _legacy_v30/          earlier v30 implementation, kept for provenance
results/
  v38_*/                raw per-run result JSONs referenced by the checks
  v38_stats.json        paired significance tests
  v38_factorial.json    factorial design analysis
```

## Environment

Tested on Python 3.13 with PyTorch and PyTorch Geometric. Core dependencies:

```
torch  torch-geometric  geoopt  numpy  scipy
```

Datasets (MUTAG, NCI1, PROTEINS, DD) are downloaded on demand through
PyTorch Geometric and are **not** included. The OGB experiments use
`ogbg-molhiv`, fetched through the OGB package.

## Verification

The point of this repository is that every number in the manuscript can be
re-derived from the artefact on disk. `check_all.sh` runs five groups of
checks:

| # | Script | What it guards against |
|---|---|---|
| 1 | `audit_v38_data.py` | internal inconsistency in the result files |
| 2 | `verify_v38_numbers.py` | a tabulated number that is not in its source JSON |
| 3 | `verify_hyperparams.py` | a hyperparameter in the paper that differs from the value read at runtime |
| 4 | `verify_algorithm_mapping.py` | pseudocode describing a mechanism the code does not implement |
| 5 | `check_graphical_abstract.py` | a graphical abstract that violates the Elsevier specification |

```bash
cd code
bash check_all.sh                 # all five groups
bash check_all.sh --compile       # additionally recompile the manuscript
```

If several interpreters are present, pin one with `PY=/path/to/python`
(the script fails loudly rather than silently using one without `numpy`).

**Manuscript dependency.** Groups 2–5 compare the code against the
manuscript, so they need the LaTeX source. It is not bundled here while the
paper is under review. Point the scripts at your local copy:

```bash
export PAPER_TEX=/path/to/ManifoldValuedSpikingNeurons_Neurocomputing_v38.tex
export SUPTEX=/path/to/supplementary_material.tex
cd code && bash check_all.sh
```

Once the manuscript is published, dropping those two files at the repository
root makes the suite run out of the box.

A few console messages in the scripts are in Chinese; they are progress
output only and do not affect behaviour.

## Reproducing the main results

```bash
cd code
python experiment_v38_5seeds.py     # TU benchmarks, 5 seeds x 3-fold
python experiment_v38_ogb.py        # ogbg-molhiv
python experiment_v38_hgcn.py       # hyperbolic control
python experiment_v38_gat.py        # GAT control
python experiment_v38_euclid.py     # Euclidean (OM-E) control
python experiment_v38_om50.py       # budget-matched re-run at 50 epochs
```

## Note on the negative result

The repository deliberately ships the control that produced the paper's
negative result. `OM-E` — the same model with the hyperbolic component set to
zero, degenerating exactly to `R^32` — matches the full model within 1.3
accuracy points on all four benchmarks. The accuracy comes from the spiking
mechanism, not from the curved state space; the curved space supplies the
formulation and the consistency guarantee.

## Licence

No licence file is included, so the default copyright applies: the code may be
read and cited, but reuse requires permission. Add a `LICENSE` file if you
intend to permit reuse.
