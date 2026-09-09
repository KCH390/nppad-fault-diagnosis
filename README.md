# nppad-fault-diagnosis

Multi-class accident diagnosis for pressurized water reactors, built in Rust
against **NPPAD** (Nuclear Power Plant Accident Data) — an open,
PCTRAN-simulated dataset of PWR accident and normal-operation transients.

Same approach as my other industrial ML projects (SECOM, Tennessee Eastman,
C-MAPSS): work from real data, document the quirks instead of hiding them,
and pair the ML pipeline with a from-scratch physics module.

## Background

NPPAD comes out of Tsinghua's INET group, built with PCTRAN (a widely used
PWR/BWR simulator):

> Qi, B., Xiao, X., Liang, J. et al. *An open time-series simulated dataset
> covering various accidents for nuclear power plants.* Sci Data 9, 766
> (2022). https://doi.org/10.1038/s41597-022-01879-1

Dataset: https://github.com/thu-inet/NuclearPowerPlantAccidentData (MIT).

It covers 18 operating conditions on a 3-loop PWR — normal operation plus 17
accident types (loss of coolant, steam line breaks, steam generator tube
ruptures, rod withdrawal/insertion, loss of AC power, ATWS, turbine trip,
etc.) — each a 97-column time series (`TIME` + 96 sensors: temperatures,
pressures, flows, reactivity, radiation monitors) sampled every 10 simulated
seconds.

## Getting the data

The raw dataset is ~600MB, so it's not in this repo. Grab just the
`Operation_csv_data` folder (the operating-parameter series — not using the
`Dose_csv_data` radionuclide data for now) with a sparse checkout:

```powershell
mkdir data\raw
cd data\raw
git init
git remote add origin https://github.com/thu-inet/NuclearPowerPlantAccidentData.git
git config core.sparseCheckout true
"Operation_csv_data/*" | Out-File -Encoding ascii .git\info\sparse-checkout
git fetch --depth 1 origin main
git checkout main
```

You should end up with `data/raw/Operation_csv_data/<ACCIDENT_TYPE>/<severity>.csv`.

## Toolchain note

This sandbox runs rustc 1.75. `linfa` pulls in `noisy_float` transitively
(via `sprs`), and the latest `noisy_float` release needs a newer `const fn`
feature than 1.75 has. `Cargo.lock` pins it to `noisy_float 0.2.0`, the last
version that builds here. If `cargo update` ever bumps it back, re-pin with:

```
cargo update -p noisy_float --precise 0.2.0
```

## Workspace structure

```
Cargo.toml          workspace manifest (core + charts + models members)
core/                dependency-light: data loading, EDA, features, splitting, CLIs
  Cargo.toml
  src/lib.rs         declares data/eda/features/split as this crate's public library
  src/data.rs        parses NPPAD CSVs into a Vec<Sample>, normalized to one canonical sensor schema
  src/eda.rs         class-balance and run-duration stats over a Vec<Sample>
  src/features.rs    rolling-window feature engineering (mean/std/slope per sensor, run-aware)
  src/split.rs       run-level train/test split + per-class subsampling
  src/main.rs        binary "nppad-fault-diagnosis" (default): Phase 1 EDA summary
  src/bin/
    extract_features.rs   binary "extract_features": Phase 2 windowed-feature extraction
charts/              isolated: everything depending on `plotters` (or reading model output) lives here, not in core
  Cargo.toml
  src/lib.rs         chart-drawing functions (bar chart, histogram, multi-line overlay, confusion matrix heatmap)
  src/main.rs        binary "eda-charts" (default): the three Phase 1 charts
  src/bin/
    feature_charts.rs      binary "feature_charts": the two Phase 2 charts
    baseline_charts.rs     binary "baseline_charts": Phase 3's confusion matrix heatmaps
models/              isolated: everything depending on `linfa`/`smartcore` lives here, not in core
  Cargo.toml
  src/main.rs        binary "train-baseline": Phase 3 — trains + evaluates both baseline models
  src/metrics.rs     hand-rolled accuracy, macro F1, confusion matrix
data/
  raw/               gitignored — fetched locally, see "Getting the data"
  features/          gitignored — Phase 2's windowed-feature CSV (~450MB, regenerable)
  results/           checked in — eda_summary.json, feature_summary.json, baseline_results.json, charts/*.svg
```

## How to run

From the workspace root, after fetching the data:

```
cargo run                                     # core, default: Phase 1 EDA summary
cargo run --bin extract_features              # core: Phase 2 windowed-feature extraction
cargo run --bin extract_features -- 10 5      # same, window_size=10 rows, stride=5 rows
cargo run -p models                           # models: Phase 3, trains + evaluates both baselines
cargo run -p models -- 5 3 0.2 2000           # same, with explicit window_size/stride/test_fraction/max_per_class
cargo run -p charts                           # charts, default: the three Phase 1 charts
cargo run -p charts --bin feature_charts      # charts: the two Phase 2 charts
cargo run -p charts --bin baseline_charts     # charts: Phase 3's confusion matrix heatmaps (run models first)
```

Everything writes into `data/results/` (and `extract_features` also writes
the large intermediate `data/features/windows.csv`). For anything beyond a
quick check, build `models` with `cargo build --release -p models` first —
training on the full dataset in debug mode is noticeably slower.

## Phase 1 findings

- **Class imbalance is baked into the dataset, not a sampling artifact.**
  "Severity" accident types (LOCA, SGATR, RW, etc.) have ~100 severity
  levels each. "Other" types (Normal, ATWS, LACP, LOF, SP, TT) have exactly
  one file each. That's roughly 100:1 between the two groups before any
  modeling even starts — flagging it now so it's not a surprise once
  Phase 3's baseline shows a suspiciously high accuracy.
- **Runs aren't a fixed length.** Across 1,217 samples, total duration
  ranges from 100s to 7,200s (`duration_histogram.svg`). A fixed-length
  window needs a deliberate choice here, not an assumption.
- **"Normal" isn't flat 100% power.** I expected it to be a steady
  full-power baseline. It's not — `Normal/1.csv`'s `PWR` ramps down from
  100% to ~40% over the first ~600 seconds and holds there
  (`power_response_by_accident.svg`). Looks like a planned load-following
  maneuver, not an accident, just not the constant baseline I assumed.
  Worth remembering for feature engineering — don't treat "Normal" as
  time-invariant.
- **ATWS is the one case where power stays high.** Same chart: every other
  featured accident type shows a sharp power drop from a successful scram,
  but ATWS (Anticipated Transient Without Scram) sits near 100% for the
  whole run — which is exactly what the accident is. Good sanity check
  that the sensor mapping is right.

## Phase 2 findings

- **25 of 101 `SLBIC` files don't match the standard schema.** Most NPPAD
  CSVs have 97 columns (`TIME` + 96 sensors); these 25 have 100 — three
  extra trailing columns (`WPCS`, `WPMU`, `WPFW`) that show up nowhere
  else in the dataset. Found this the hard way: `extract_features` threw a
  CSV write error the first time it ran against the full dataset ("found
  record with 301 fields, but the previous record has 292 fields"). Fixed
  by having `data.rs` scan every file's header up front, pick the dataset's
  most common 96-column schema as canonical, and normalize every file to
  it — logging a note whenever it has to drop the extra columns. Every
  `Sample` now has a consistent shape no matter which file it came from.
- **Windowing makes the imbalance worse, not just carried over.** Compare
  `class_balance.svg` (Phase 1) with `window_class_balance.svg` (Phase 2):
  the one-sample accident types (ATWS, LACP, SP) end up with only
  ~150-200 windows, while types like MD and LOCA — already ~100x more raw
  samples, and longer-running on top of that — produce tens of thousands
  of windows each. Phase 3's baseline has a worse imbalance problem than
  the raw sample counts alone would suggest.
- **No accident type came out with zero windows** at the current defaults
  (window_size=5 rows / 50s, stride=3 rows / 30s), even though the
  shortest run in the dataset is only 11 rows (RI, ~100s). Confirmed in
  `data/results/feature_summary.json` — `accident_types_with_zero_windows`
  is empty. A bigger window size could still hit that edge case, which is
  why the CLI warns explicitly if it ever happens.

## Phase 3 findings

- **Random Forest clearly beats Gaussian Naive Bayes as a baseline** —
  macro F1 of 0.91 vs. 0.46 on the same run-level split (`window_size=5`
  rows, `stride=3` rows, `test_fraction=0.2`, capped at 2,000
  training / 1,000 test windows per class). Not a surprising result on its
  own, but worth confirming rather than assuming, and the two models'
  confusion matrices (`confusion_matrix_random_forest.svg`,
  `confusion_matrix_gaussian_nb.svg`) tell different stories about *how*
  each one fails, not just by how much.
- **LOCA and LOCAC are genuinely hard to tell apart, even for the better
  model.** They're both loss-of-coolant accidents differing only in break
  location (hot leg vs. cold leg) — physically similar enough that Random
  Forest's F1 drops to 0.65 and 0.55 on exactly those two classes while
  scoring 0.9+ on almost everything else. That's visible directly in the
  heatmap as the one clearly off-diagonal block. This is a real signal
  about where mean/std/slope window features run out of discriminating
  power, not a bug — a useful thing to know before deciding whether Phase 4
  needs richer features for just this pair, or whether Phase 5's physics
  module happens to be the more natural place to draw that distinction.
- **Naive Bayes over-predicts a few specific classes rather than failing
  evenly.** Its confusion matrix shows heavy false-positive columns at SP
  and SLBOC — lots of other classes' windows get called one of those two —
  rather than errors spread uniformly across the grid. Consistent with
  Gaussian NB's core assumption (each feature is independently normally
  distributed per class) breaking down harder for some classes' sensor
  distributions than others.
- **Evaluation is only as honest as the split underneath it, and this
  dataset's single-run accident types are a real limit on that.** ATWS,
  LACP, LOF, Normal, SP, and TT each have exactly one run, so there's no
  way to hold out a genuinely unseen run for them (see `core::split`'s
  module docs for the full reasoning) — they're trained on but excluded
  from the macro F1 and confusion matrix here. Both models' JSON output
  (`data/results/baseline_results.json`) still lists them with
  `support: 0` rather than silently dropping them, so it's explicit
  exactly which classes the headline number does and doesn't cover.

## Phase roadmap

1. **Data & workspace setup** — loading, EDA, workspace scaffolding
2. **Feature engineering** — rolling-window features per sensor, run-aware
3. **Baseline classification** *(current)* — `linfa` (Gaussian Naive Bayes) + `smartcore` (Random Forest), macro F1 + confusion matrix over raw accuracy
4. **From-scratch classifier** — extend CART/boosting code from C-MAPSS to multi-class
5. **Physics module** — hand-rolled point reactor kinetics (delayed-neutron ODEs)
6. **Sequence modeling (stretch)** — sequence extraction + LSTM via `candle`
7. **Reach: out-of-distribution detection** — flag scenarios outside the known accident types
8. **Reach: streaming/online classification** — replay NPPAD as a simulated live feed, classify incrementally
9. **Reach: thermal feedback coupling** — extend point kinetics with temperature-reactivity feedback
10. **Reach: interpretability** — permutation importance / SHAP-style attribution over sensors
