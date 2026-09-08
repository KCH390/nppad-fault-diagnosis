# nppad-fault-diagnosis

Multi-class accident diagnosis for pressurized water reactors, built in Rust
against the **NPPAD** (Nuclear Power Plant Accident Data) dataset — an open,
PCTRAN-simulated dataset of PWR accident and normal-operation transients.

This project follows the same phased, dataset-first structure as my other
industrial ML portfolio projects (SECOM fault detection, Tennessee Eastman
Process fault diagnosis, C-MAPSS turbofan RUL prediction): real data, honest
documentation of quirks and disagreements, and a from-scratch physics module
alongside the ML pipeline.

## Background

NPPAD was built by Qi, Xiao, and Liang (Tsinghua University, INET) using
PCTRAN, a widely used PWR/BWR simulator, and released publicly:

> Qi, B., Xiao, X., Liang, J. et al. *An open time-series simulated dataset
> covering various accidents for nuclear power plants.* Sci Data 9, 766
> (2022). https://doi.org/10.1038/s41597-022-01879-1

Dataset source: https://github.com/thu-inet/NuclearPowerPlantAccidentData
(MIT licensed).

It covers 18 operating conditions on a 3-loop PWR: normal operation plus 17
accident types (loss of coolant, steam line breaks, steam generator tube
ruptures, rod withdrawal/insertion, loss of AC power, ATWS, turbine trip,
and more), each recorded as a 97-column time series (`TIME` + 96 named
sensors: temperatures, pressures, flows, reactivity components, radiation
monitors, etc.), sampled every 10 simulated seconds.

## Getting the data

The raw dataset is ~600MB, so it's not checked into this repo. Fetch just
the `Operation_csv_data` folder (the operating-parameter time series; this
project doesn't currently use the companion `Dose_csv_data` radionuclide
data) with a sparse checkout:

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

## Workspace structure

```
Cargo.toml          workspace manifest (core + charts members)
core/                dependency-light: data loading, EDA, features, CLIs
  Cargo.toml
  src/lib.rs         declares data/eda/features as this crate's public library
  src/data.rs        parses NPPAD CSVs into a Vec<Sample>, normalized to one canonical sensor schema
  src/eda.rs         class-balance and run-duration statistics over a Vec<Sample>
  src/features.rs    rolling-window feature engineering (mean/std/slope per sensor, run-aware)
  src/main.rs        binary "nppad-fault-diagnosis" (default): Phase 1 EDA summary
  src/bin/
    extract_features.rs   binary "extract_features": Phase 2 windowed-feature extraction
charts/              isolated: everything that depends on `plotters` lives here, not in core
  Cargo.toml
  src/lib.rs         chart-drawing functions (bar chart, histogram, multi-line overlay)
  src/main.rs        binary "eda-charts" (default): the three Phase 1 charts
  src/bin/
    feature_charts.rs      binary "feature_charts": the two Phase 2 charts
data/
  raw/               gitignored — fetched locally per "Getting the data" above
  features/          gitignored — Phase 2's windowed-feature CSV (~450MB, regenerable)
  results/           checked in — eda_summary.json, feature_summary.json, charts/*.svg
```

## How to run

From the workspace root, after fetching the data:

```
cargo run                                     # core, default binary: Phase 1 EDA summary
cargo run --bin extract_features              # core: Phase 2 windowed-feature extraction
cargo run --bin extract_features -- 10 5      # same, with window_size=10 rows, stride=5 rows
cargo run -p charts                           # charts, default binary: the three Phase 1 charts
cargo run -p charts --bin feature_charts      # charts: the two Phase 2 charts
```

All of these write into `data/results/` (and `extract_features` additionally
writes the large intermediate `data/features/windows.csv`).

## Phase 1 findings

Real, documented quirks discovered while building the loader and EDA, not
smoothed over:

- **Severe class imbalance by design, not by accident (pun intended).**
  "Severity"-class accident types (LOCA, SGATR, RW, etc. — see NPPAD's own
  Table 1) have ~100 simulated severity levels each. "Other"-class types
  (Normal, ATWS, LACP, LOF, SP, TT) have exactly **one** file each. Any
  classifier trained on this dataset as-is will see roughly 100:1 imbalance
  between these two groups — worth being explicit about before Phase 3's
  baseline model, the same way SECOM's class imbalance was called out
  early rather than discovered via a suspiciously high accuracy number.
- **Run length is not fixed.** Across 1,217 loaded samples, total run
  duration ranges from 100s to 7,200s (see `duration_histogram.svg`). Any
  fixed-length feature window (as used in C-MAPSS) needs truncation or
  padding here, and that choice should be made deliberately in Phase 2,
  not assumed away.
- **"Normal" doesn't mean flat 100% power.** I initially expected the
  `Normal` accident type to represent steady full-power operation used as
  a diagnostic baseline. It doesn't — `Normal/1.csv`'s `PWR` (core thermal
  power, %) ramps from 100% down to ~40% over the first ~600 seconds and
  holds there (see `power_response_by_accident.svg`). This looks like a
  planned load-following maneuver, not an accident — a legitimate "normal
  operation" scenario, just not the constant-baseline one I assumed going
  in. Worth remembering during feature engineering: don't build features
  that assume "Normal" is time-invariant.
- **ATWS keeps power high while everything else scrams.** In the same
  chart, ATWS (Anticipated Transient Without Scram) is the one accident
  type where `PWR` stays near 100% for the full run — the reactor fails to
  automatically shut down, which is exactly the accident's definition.
  Every other featured accident type shows a sharp power drop from a
  successful scram. This is a genuinely useful diagnostic signal, and a
  good sanity check that the loader and sensor mapping are correct.

## Phase 2 findings

- **25 of 101 `SLBIC` files have a different schema than everything else.**
  Most NPPAD CSVs have 97 columns (`TIME` + 96 sensors). 25 `SLBIC` files
  have **100** — three extra trailing columns (`WPCS`, `WPMU`, `WPFW`) that
  appear nowhere else in the dataset. This first surfaced as a hard CSV
  write error ("found record with 301 fields, but the previous record has
  292 fields") the first time `extract_features` ran — a real bug caught by
  actually running the pipeline against the full dataset, not a
  hypothetical edge case. `data.rs` now determines the dataset's most
  common ("canonical") 96-sensor schema up front and normalizes every file
  to it, logging a note wherever it has to drop extra columns. Every
  `Sample` is guaranteed to have the same `sensor_names` and feature-vector
  length as a result.
- **Windowing does not just carry over Phase 1's class imbalance — it
  reshapes it, and in the same direction.** Compare
  `class_balance.svg` (Phase 1, raw sample counts) with
  `window_class_balance.svg` (Phase 2, window counts): the "Other"-class
  accident types (ATWS, LACP, SP — one raw sample each) end up with only
  ~150-200 windows, while "Severity"-class types like MD and LOCA — already
  ~100x more raw samples — produce tens of thousands of windows each,
  because they also tend to run longer. The imbalance Phase 3's baseline
  model has to deal with is worse after windowing than the raw sample
  counts alone would suggest.
- **No accident type produced zero windows at the chosen defaults**
  (`window_size=5` rows / 50s, `stride=3` rows / 30s) — worth checking
  explicitly given the shortest run in the whole dataset is 11 rows (RI,
  ~100s); see `accident_types_with_zero_windows` in
  `data/results/feature_summary.json`, which is empty. A larger
  `window_size` could still produce zero windows for that run; the CLI
  warns explicitly if that ever happens for any accident type.

## Phase roadmap

1. **Data & workspace setup** — loading, EDA, workspace scaffolding
2. **Feature engineering** *(this phase)* — rolling-window features per sensor, run-aware
3. **Baseline classification** — `linfa`/`smartcore`, macro F1 + confusion matrix over raw accuracy
4. **From-scratch classifier** — extend CART/boosting code from C-MAPSS to multi-class
5. **Physics module** — hand-rolled point reactor kinetics (delayed-neutron ODEs)
6. **Sequence modeling (stretch)** — sequence extraction + LSTM via `candle`
7. **Reach: out-of-distribution detection** — flag scenarios outside the known accident types
8. **Reach: streaming/online classification** — replay NPPAD as a simulated live feed, classify incrementally
9. **Reach: thermal feedback coupling** — extend point kinetics with temperature-reactivity feedback
10. **Reach: interpretability** — permutation importance / SHAP-style attribution over sensors
