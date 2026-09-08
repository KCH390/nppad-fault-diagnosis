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
core/                dependency-light: data loading, EDA, CLI (default cargo run/build target)
  Cargo.toml
  src/lib.rs         declares the data and eda modules as this crate's public library
  src/data.rs        parses the NPPAD CSVs into a Vec<Sample>
  src/eda.rs         class-balance and run-duration statistics over a Vec<Sample>
  src/main.rs        CLI: load data, print + save the Phase 1 EDA summary
charts/              isolated: everything that depends on `plotters` lives here, not in core
  Cargo.toml
  src/lib.rs         the actual chart-drawing functions (bar chart, histogram, multi-line overlay)
  src/main.rs        loads data via core, generates the three Phase 1 charts
data/
  raw/               gitignored — fetched locally per "Getting the data" above
  results/           checked in — eda_summary.json + charts/*.svg
```

## How to run

From the workspace root, after fetching the data:

```
cargo run                # runs core: prints + saves the Phase 1 EDA summary
cargo run -p charts      # runs charts: generates the three Phase 1 SVG charts
```

Both write into `data/results/`.

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

## Phase roadmap

1. **Data & workspace setup** *(this phase)* — loading, EDA, workspace scaffolding
2. **Feature engineering** — rolling-window features per sensor, run-aware
3. **Baseline classification** — `linfa`/`smartcore`, macro F1 + confusion matrix over raw accuracy
4. **From-scratch classifier** — extend CART/boosting code from C-MAPSS to multi-class
5. **Physics module** — hand-rolled point reactor kinetics (delayed-neutron ODEs)
6. **Sequence modeling (stretch)** — sequence extraction + LSTM via `candle`
7. **Reach: out-of-distribution detection** — flag scenarios outside the known accident types
8. **Reach: streaming/online classification** — replay NPPAD as a simulated live feed, classify incrementally
9. **Reach: thermal feedback coupling** — extend point kinetics with temperature-reactivity feedback
10. **Reach: interpretability** — permutation importance / SHAP-style attribution over sensors
