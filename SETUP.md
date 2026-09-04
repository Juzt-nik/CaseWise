# Setup -- CaseWise
## Files you need (final versions only)

```
disputes_10k.csv              (Phrase 1 dataset)
generate_disputes.py          (Phrase 1 generator -- only needed if regenerating data)
train_model.py                (trains + calibrates + saves model_artifacts/)
train_and_compare.py          (Phrase 2 -- baseline vs SMOTE vs SynthEdge)
evaluate.py                   (Phrase 3)
heuristic_baseline.py         (Phrase 3/6 -- ML vs heuristic)
pipeline.py                   (Phrase 4)
grounding_check.py            (imported by pipeline.py -- must sit next to it)
robustness_check.py           (multi-seed check on the Phrase 2 finding)
streamlit_app.py              (Phrase 6 demo -- imports pipeline.py directly)
README.md
SETUP.md
failure_case.md
cost_asymmetry_disclosure.md
.gitignore

model_artifacts/
├── win_probability_model.json
├── probability_calibrator.joblib
├── feature_columns.json
└── training_meta.json
```

All `.py` files and `disputes_10k.csv` go in one project root directory.
`model_artifacts/` is a subfolder of that same root.

**Note on naming:** `train_and_compare.py` is the Phrase 2 comparison script
(baseline vs. SMOTE vs. SynthEdge, three-way split reconciled with
production). If you have an earlier copy of a file with this same name
from before the SynthEdge API was confirmed against the real package, it
referenced a `DensityAwareAugmenter` import that doesn't exist and will
not run -- make sure the version you have matches this repo's current one,
not an early draft.

## Install

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn joblib matplotlib streamlit ctgan
pip install synthedge   # or, better: install YOUR patched fork -- see note below
```

**Important:** for fully reproducible SynthEdge runs, install from your own
GitHub repo (the one with the `synthesizer.py` seeding fix), not the
original PyPI package:

```bash
pip uninstall synthedge
pip install git+https://github.com/Juzt-nik/SynthEdge.git
```

Without this, `SynthEdge.fill()`'s CTGAN step is not fully deterministic --
harmless, but re-running `train_model.py` or `train_and_compare.py` may add
a slightly different number of synthetic rows each run.

## Two ways to run this

**Option A -- use the model as already trained (recommended).**
The `model_artifacts/` files above are already trained and calibrated.
Every number in `README.md` comes from this exact artifact set. Drop them
into `model_artifacts/` and skip straight to running `pipeline.py`,
`evaluate.py`, `heuristic_baseline.py`, or `streamlit_app.py` -- no
training needed.

**Option B -- retrain from scratch.**
```bash
python train_model.py
```
This regenerates `model_artifacts/`. With the patched SynthEdge installed,
results should closely match Option A's numbers but will not be
byte-identical. If you retrain, re-check `training_meta.json`'s Brier
scores and `rows_added_by_synthedge` before quoting numbers -- use what you
actually got, not what's quoted in `README.md`.

## Run order

```bash
# Only if regenerating data from scratch (otherwise use disputes_10k.csv directly)
python generate_disputes.py --n 10000 --seed 42 --out disputes_10k.csv

# Only if retraining (Option B) -- otherwise use the provided model_artifacts/
python train_model.py

# These all read model_artifacts/ + disputes_10k.csv -- any order
python train_and_compare.py
python evaluate.py
python heuristic_baseline.py
python pipeline.py

# Multi-seed robustness check on the Phrase 2 finding (takes longer -- trains
# baseline/SMOTE/SynthEdge across 5 different splits)
python robustness_check.py

# Interactive dashboard -- run from the project root
streamlit run streamlit_app.py
```

## Known non-blocking caveats

- `evaluate.py`'s test split excludes the calibration slice `train_model.py`
  carves out, so `train_and_compare.py`'s SynthEdge row-count and
  `training_meta.json`'s may differ by a small amount even on the same
  data -- both are correct, they're just measuring slightly different
  training sets by design (see the note in `train_and_compare.py`).
- The `synthedge compare` CLI command has an unpatched test-set-leakage bug
  (separate from the CTGAN determinism fix) -- none of the scripts above
  use it, so it doesn't affect these results, but don't use that CLI
  command directly for anything you plan to report.
- `robustness_check.py` trains the full baseline/SMOTE/SynthEdge comparison
  five times (once per seed), so it takes noticeably longer than any other
  script here -- this is expected, not a hang.
