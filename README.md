# nppad-fault-diagnosis

Multi-class accident diagnosis for pressurized water reactors, built in Rust
against NPPAD (Nuclear Power Plant Accident Data), an open, PCTRAN-simulated
dataset of PWR accident and normal-operation transients.

Same approach I've used on my other industrial ML projects (SECOM,
Tennessee Eastman, C-MAPSS). Work from real data, document the quirks
instead of hiding them, and pair the ML pipeline with a from-scratch
physics module.

## Background

NPPAD comes out of Tsinghua's INET group, built with PCTRAN, a widely used
PWR/BWR simulator.

> Qi, B., Xiao, X., Liang, J. et al. *An open time-series simulated dataset
> covering various accidents for nuclear power plants.* Sci Data 9, 766
> (2022). https://doi.org/10.1038/s41597-022-01879-1

Dataset source is https://github.com/thu-inet/NuclearPowerPlantAccidentData (MIT licensed).

It covers 18 operating conditions on a 3-loop PWR, normal operation plus 17
accident types (loss of coolant, steam line breaks, steam generator tube
ruptures, rod withdrawal/insertion, loss of AC power, ATWS, turbine trip,
and so on). Each is a 97-column time series, `TIME` plus 96 sensors
(temperatures, pressures, flows, reactivity, radiation monitors), sampled
every 10 simulated seconds.

## Getting the data

The raw dataset is around 600MB, so it's not in this repo. Grab just the
`Operation_csv_data` folder (the operating-parameter series, not using the
`Dose_csv_data` radionuclide data for now) with a sparse checkout.

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
feature than 1.75 has. `Cargo.lock` pins it to `noisy_float 0.2.0`, the
last version that builds here. If `cargo update` ever bumps it back, re-pin
it.

```
cargo update -p noisy_float --precise 0.2.0
```

`sequence`'s `candle-core` dependency has a separate, unrelated version
mismatch: several of its own dependencies pull in `rand 0.9`-compatible
`half` builds while candle-core's source code itself uses the `rand 0.8`
API directly. If `cargo build -p sequence` fails inside `candle-core`
with `SampleUniform`/`SampleBorrow` trait errors, re-pin those three
crates back onto the rand-0.8-compatible set (see the comment in
`sequence/Cargo.toml` for the full explanation):

```
cargo update -p half --precise 2.4.1
cargo update -p rand_distr --precise 0.4.3
cargo update -p rand --precise 0.8.5
```

## Workspace structure

```
Cargo.toml          workspace manifest (core + charts + models + physics + sequence members)
core/                dependency-light: data loading, EDA, features, splitting, CLIs
  Cargo.toml
  src/lib.rs         declares data/eda/features/sequences/split as this crate's public library
  src/data.rs        parses NPPAD CSVs into a Vec<Sample>, normalized to one canonical sensor schema
  src/eda.rs         class-balance and run-duration stats over a Vec<Sample>
  src/features.rs    rolling-window feature engineering (mean/std/slope per sensor, run-aware)
  src/sequences.rs   fixed-length, normalized sequence construction per run, for Phase 6's LSTM
  src/split.rs       run-level train/test split plus per-class subsampling
  src/main.rs        binary "nppad-fault-diagnosis" (default), Phase 1 EDA summary
  src/bin/
    extract_features.rs   binary "extract_features", Phase 2 windowed-feature extraction
charts/              isolated: everything depending on plotters (or reading model output) lives here, not in core
  Cargo.toml
  src/lib.rs         chart-drawing functions (bar chart, histogram, multi-line overlay, confusion matrix heatmap)
  src/main.rs        binary "eda-charts" (default), the three Phase 1 charts
  src/bin/
    feature_charts.rs      binary "feature_charts", the two Phase 2 charts
    baseline_charts.rs     binary "baseline_charts", Phase 3's confusion matrix heatmaps
    scratch_charts.rs      binary "scratch_charts", Phase 4's confusion matrix heatmaps
    kinetics_charts.rs     binary "kinetics_charts", Phase 5's simulated vs real overlay charts
    lstm_charts.rs         binary "lstm_charts", Phase 6's confusion matrix heatmap
models/              isolated: everything depending on linfa/smartcore, plus the from-scratch models, lives here, not in core
  Cargo.toml
  src/lib.rs         declares metrics/tree/cart_classifier/boosting as this crate's public library
  src/main.rs        binary "train-baseline" (default), Phase 3, trains and evaluates the library-backed baselines
  src/metrics.rs     hand-rolled accuracy, macro F1, confusion matrix (shared by both phases' binaries)
  src/tree.rs        from-scratch CART regression tree (variance/SSE-reduction splits), zero dependencies
  src/cart_classifier.rs   from-scratch CART classification tree (Gini-impurity splits), zero dependencies
  src/boosting.rs    from-scratch multi-class gradient boosting, built on tree.rs's regression trees
  src/bin/
    train_scratch.rs       binary "train-scratch", Phase 4, trains and evaluates the from-scratch models
physics/             point reactor kinetics, zero-dependency ODE model, only needs core for the comparison traces
  Cargo.toml
  src/lib.rs         declares kinetics as this crate's public library
  src/kinetics.rs    six-group delayed neutron point kinetics model, hand-rolled RK4 solver
  src/bin/
    simulate_kinetics.rs   binary "simulate-kinetics" (default), Phase 5, simulates reactivity ramps and pulls real RW/RI traces for comparison
sequence/            isolated: everything depending on candle lives here, not in core
  Cargo.toml
  src/lib.rs         declares lstm/metrics as this crate's public library
  src/lstm.rs         LSTM classifier built on candle-nn's LSTM layer plus a final linear layer
  src/metrics.rs      hand-rolled accuracy, macro F1, confusion matrix (duplicated from models::metrics to keep candle isolated)
  src/bin/
    train_lstm.rs          binary "train-lstm" (default), Phase 6, builds sequences, trains and evaluates the LSTM
data/
  raw/               gitignored, fetched locally, see "Getting the data"
  features/          gitignored, Phase 2's windowed-feature CSV (~450MB, regenerable)
  results/           checked in: eda_summary.json, feature_summary.json, baseline_results.json, scratch_results.json, kinetics_results.json, lstm_results.json, charts/*.svg
```

## How to run

From the workspace root, after fetching the data.

```
cargo run                                     # core, default, Phase 1 EDA summary
cargo run --bin extract_features              # core, Phase 2 windowed-feature extraction
cargo run --bin extract_features -- 10 5      # same, window_size=10 rows, stride=5 rows
cargo run -p models                           # models, default, Phase 3, trains and evaluates both library-backed baselines
cargo run -p models -- 5 3 0.2 2000           # same, with explicit window_size/stride/test_fraction/max_per_class
cargo run -p models --bin train-scratch       # models, Phase 4, trains and evaluates both from-scratch models
cargo run -p physics                          # physics, default, Phase 5, simulates reactivity transients
cargo run -p sequence                         # sequence, default, Phase 6, builds sequences, trains and evaluates the LSTM
cargo run -p charts                           # charts, default, the three Phase 1 charts
cargo run -p charts --bin feature_charts      # charts, the two Phase 2 charts
cargo run -p charts --bin baseline_charts     # charts, Phase 3's confusion matrix heatmaps (run models first)
cargo run -p charts --bin scratch_charts      # charts, Phase 4's confusion matrix heatmaps (run train-scratch first)
cargo run -p charts --bin kinetics_charts     # charts, Phase 5's comparison charts (run physics first)
cargo run -p charts --bin lstm_charts         # charts, Phase 6's confusion matrix heatmap (run sequence first)
```

Everything writes into `data/results/` (and `extract_features` also writes
the large intermediate `data/features/windows.csv`). For anything beyond a
quick check, build `models` with `cargo build --release -p models` first.
Training on the full dataset in debug mode is noticeably slower, and Phase
4's gradient boosting in particular is only practical in release mode.
Phase 5's `physics` crate runs fine in debug mode either way, since it's a
much smaller amount of computation than training a model. Phase 6's
`sequence` crate should also be built in release mode for training, same
reasoning as Phases 3-4.

## Phase 1 findings

- **Class imbalance is baked into the dataset, not a sampling artifact.**
  "Severity" accident types (LOCA, SGATR, RW, etc.) have around 100
  severity levels each. "Other" types (Normal, ATWS, LACP, LOF, SP, TT)
  have exactly one file each. That's roughly 100 to 1 between the two
  groups before any modeling even starts, so I'm flagging it now instead
  of being surprised later by Phase 3's baseline showing a suspiciously
  high accuracy.
- **Runs aren't a fixed length.** Across 1,217 samples, total duration
  ranges from 100s to 7,200s (`duration_histogram.svg`). A fixed-length
  window needs a deliberate choice here, not an assumption.
- **"Normal" isn't flat 100% power.** I expected it to be a steady
  full-power baseline. It's not. `Normal/1.csv`'s `PWR` ramps down from
  100% to about 40% over the first 600 seconds or so and holds there
  (`power_response_by_accident.svg`). Looks like a planned load-following
  maneuver, not an accident, just not the constant baseline I assumed.
  Worth remembering for feature engineering, since "Normal" shouldn't be
  treated as time-invariant.
- **ATWS is the one case where power stays high.** Same chart. Every other
  featured accident type shows a sharp power drop from a successful scram,
  but ATWS (Anticipated Transient Without Scram) sits near 100% for the
  whole run, which is exactly what the accident is. Good sanity check that
  the sensor mapping is right.

## Phase 2 findings

- **25 of 101 `SLBIC` files don't match the standard schema.** Most NPPAD
  CSVs have 97 columns (`TIME` plus 96 sensors). These 25 have 100, three
  extra trailing columns (`WPCS`, `WPMU`, `WPFW`) that show up nowhere else
  in the dataset. I found this the hard way. `extract_features` threw a
  CSV write error the first time it ran against the full dataset ("found
  record with 301 fields, but the previous record has 292 fields"). Fixed
  it by having `data.rs` scan every file's header up front, pick the
  dataset's most common 96-column schema as canonical, and normalize every
  file to it, logging a note whenever it has to drop the extra columns.
  Every `Sample` now has a consistent shape no matter which file it came
  from.
- **Windowing makes the imbalance worse, not just carried over.** Compare
  `class_balance.svg` (Phase 1) with `window_class_balance.svg` (Phase 2).
  The one-sample accident types (ATWS, LACP, SP) end up with only around
  150 to 200 windows, while types like MD and LOCA, already about 100x
  more raw samples and longer-running on top of that, produce tens of
  thousands of windows each. Phase 3's baseline has a worse imbalance
  problem than the raw sample counts alone would suggest.
- **No accident type came out with zero windows** at the current defaults
  (window_size=5 rows / 50s, stride=3 rows / 30s), even though the
  shortest run in the dataset is only 11 rows (RI, about 100s). Confirmed
  in `data/results/feature_summary.json`, where
  `accident_types_with_zero_windows` is empty. A bigger window size could
  still hit that edge case, which is why the CLI warns explicitly if it
  ever happens.

## Phase 3 findings

- **Random Forest clearly beats Gaussian Naive Bayes as a baseline.**
  Macro F1 of 0.91 vs 0.46 on the same run-level split (`window_size=5`
  rows, `stride=3` rows, `test_fraction=0.2`, capped at 2,000 training /
  1,000 test windows per class). Not a surprising result on its own, but
  worth confirming rather than assuming. The two models' confusion
  matrices (`confusion_matrix_random_forest.svg`,
  `confusion_matrix_gaussian_nb.svg`) tell different stories about how
  each one fails, not just by how much.
- **LOCA and LOCAC are genuinely hard to tell apart, even for the better
  model.** They're both loss-of-coolant accidents differing only in break
  location (hot leg vs cold leg), physically similar enough that Random
  Forest's F1 drops to 0.65 and 0.55 on exactly those two classes while
  scoring 0.9+ on almost everything else. That's visible directly in the
  heatmap as the one clearly off-diagonal block. It's a real signal about
  where mean/std/slope window features run out of discriminating power,
  not a bug. Useful to know before deciding whether Phase 4 needs richer
  features for just this pair, or whether Phase 5's physics module ends up
  the more natural place to draw that distinction.
- **Naive Bayes over-predicts a few specific classes rather than failing
  evenly.** Its confusion matrix shows heavy false-positive columns at SP
  and SLBOC, lots of other classes' windows getting called one of those
  two, rather than errors spread uniformly across the grid. Consistent
  with Gaussian NB's core assumption (each feature is independently
  normally distributed per class) breaking down harder for some classes'
  sensor distributions than others.
- **Evaluation is only as honest as the split underneath it, and this
  dataset's single-run accident types are a real limit on that.** ATWS,
  LACP, LOF, Normal, SP, and TT each have exactly one run, so there's no
  way to hold out a genuinely unseen run for them (see `core::split`'s
  module docs for the full reasoning). They're trained on but excluded
  from the macro F1 and confusion matrix here. Both models' JSON output
  (`data/results/baseline_results.json`) still lists them with
  `support: 0` rather than silently dropping them, so it's explicit
  exactly which classes the headline number does and doesn't cover.

## Phase 4 findings

- **A real O(n squared) performance bug, found by actually trying to train
  the boosting model.** The first version of `best_split` (in both
  `tree.rs` and `cart_classifier.rs`) tried every candidate threshold by
  calling `.partition(...)` over the full row set from scratch each time.
  That's an O(n) rescan per threshold, times up to n thresholds, times
  every feature. On this project's roughly 300-feature windows, even a
  training set of barely 1,000 rows didn't finish building 324 boosting
  trees inside a several-minute timeout. Fixed it by sorting each
  feature's values once and sweeping left to right with running sums
  instead of recomputing variance/Gini from scratch at every threshold.
  Same fix in spirit as the C-MAPSS project's from-scratch gradient
  boosting hitting a comparable issue on its larger dataset. After the
  fix, the full pipeline (7,039 train / 3,000 test windows, 324 trees)
  runs in under two minutes in release mode.
- **From-scratch gradient boosting nearly matches smartcore's Random
  Forest.** Macro F1 0.892 vs 0.911 on directly comparable splits (same
  `core::split` run-level split, same features, same evaluation code,
  only the model implementation differs). That's a meaningful result on
  its own. It's evidence the from-scratch implementation is actually
  doing the right thing, not just compiling and running without crashing.
- **A single from-scratch CART tree is a much weaker baseline than
  boosting.** Macro F1 0.673 vs 0.892, with LR and RI both landing at zero
  recall (`confusion_matrix_cart_scratch.svg` shows their rows entirely
  blank). At `max_depth=8` a single tree has to carve up an 18-class,
  roughly 300-feature space with at most a few hundred leaves, clearly not
  enough capacity for some classes to ever get a leaf of their own, which
  boosting's additive ensemble of many shallow trees doesn't run into.
- **The LOCA/LOCAC confusion from Phase 3 shows up again here, in both
  from-scratch models.** Same physically similar accident pair (loss of
  coolant, hot leg vs cold leg), same off-diagonal block in the heatmap,
  regardless of whether the model is smartcore's Random Forest or
  from-scratch gradient boosting. Seeing the same failure mode across four
  independently-implemented models (two library, two from-scratch) is
  good evidence this is a genuine limit of the mean/std/slope window
  features on this specific pair, not an artifact of any one model's
  implementation.

## Phase 5 findings

- **A held positive reactivity, without feedback, just keeps growing.**
  With no thermal feedback in this model (see `kinetics.rs`'s module
  docs), any sustained positive reactivity eventually produces unbounded
  exponential power growth. Early on I tried a fairly large ramp (+0.003,
  about 46% of the total delayed neutron fraction), and by 60 seconds the
  simulated power had grown to about 780 times its starting value. That's
  physically correct behavior for a model with nothing to cancel the
  applied reactivity out, but it also made the real NPPAD comparison trace
  invisible on the same chart, just a flat line near the bottom. I ended
  up dropping the magnitude to +/-0.0006 for the charts so both curves
  land in a readable range together. The unbounded growth itself is worth
  keeping in mind though, since it's exactly the gap reach goal 9
  (thermal feedback coupling) is meant to close.
- **The real RW trace stays essentially flat for the first 90-100 seconds,
  then rises and turns back over.** `kinetics_rw_comparison.svg` shows
  this clearly against the simulated curve, which climbs steadily from
  the moment the ramp starts. NPPAD's real RW accident either has a slower
  effective reactivity insertion than my illustrative ramp, active plant
  control response, or both. Either way, this model has no way to capture
  that shape yet, since it has no feedback and no control logic, just an
  externally prescribed reactivity history.
- **The real RI trace has a much sharper drop than the simulated one, and
  it happens later.** `kinetics_ri_comparison.svg` shows power holding
  near 1.0 until about t=100s, then dropping fast, down to about 0.06 by
  t=120s. My simulated ramp declines earlier and more gradually, reaching
  about 0.63 by t=60s. This looks like the real accident involves a sharp
  event (a scram or a fast rod insertion) rather than a slow, steady rod
  movement, which is a genuinely different reactivity shape than a linear
  ramp. This comparison was never meant to be a fit (see the note field in
  `kinetics_results.json`), and this is a good concrete example of why
  that caveat matters.
- **Choosing which real run to compare against turned out to matter.** My
  first attempt picked the lowest-severity-id run for each accident type,
  which for RI happened to be only 11 rows (about 100 seconds total, see
  the Phase 1 README notes on run-length variability), barely enough data
  to see anything. Switched to picking the longest-duration run per type
  instead, which is what both comparison charts actually use now.

## Phase 6 notes

- **A fixed sequence length is a new requirement Phases 2-4 didn't have.**
  Windowing (Phase 2) sidestepped run-length variability by working on
  short, fixed-size slices of each run. An LSTM classifying a whole run at
  once needs every training example to be the same length so they can be
  batched into one tensor. `core::sequences` truncates longer runs to the
  first `seq_len` rows (keeping the accident's onset, which is where most
  of the distinguishing signal lives, rather than a long near-steady-state
  tail) and pads shorter runs by repeating the last observed row rather
  than zero-padding, since zero-padding would look to the model like a
  sudden, physically nonsensical jump to every sensor reading exactly 0.
- **Normalization matters here in a way it didn't for the tree models.**
  Phases 3-4's trees split on one feature at a time, so it doesn't matter
  if `PWR` is a percentage and `TAVG` is a temperature in the hundreds.
  An LSTM's gates mix every input feature together in the same linear
  layer, so raw NPPAD sensor units sitting on wildly different scales
  would let some features dominate purely because of their units, not
  their actual signal. `core::sequences::NormalizationStats` z-score
  normalizes every sensor, fit on the training split only, same "no
  leakage from test data" discipline as the run-level split itself.
- **`sequence::metrics` duplicates `models::metrics` rather than importing
  it.** Depending on the `models` crate as a library would pull `linfa`
  and `smartcore` into `sequence`'s dependency tree along with it, since
  Cargo resolves a whole crate's dependencies regardless of which modules
  you actually use. Same "duplication over cross-crate dependency"
  tradeoff already used in `charts`' `baseline_charts.rs` and
  `scratch_charts.rs`, just applied to a larger, more important piece of
  shared logic this time. Worth knowing this exists in two places if the
  evaluation logic ever needs to change.

## Phase roadmap

1. **Data & workspace setup**, loading, EDA, workspace scaffolding
2. **Feature engineering**, rolling-window features per sensor, run-aware
3. **Baseline classification**, `linfa` (Gaussian Naive Bayes) plus `smartcore` (Random Forest), macro F1 and confusion matrix over raw accuracy
4. **From-scratch classifier**, extend CART/boosting code from C-MAPSS to multi-class
5. **Physics module** (current), hand-rolled point reactor kinetics (delayed-neutron ODEs)
6. **Sequence modeling (stretch)**, sequence extraction plus LSTM via `candle`
7. **Reach: out-of-distribution detection**, flag scenarios outside the known accident types
8. **Reach: streaming/online classification**, replay NPPAD as a simulated live feed, classify incrementally
9. **Reach: thermal feedback coupling**, extend point kinetics with temperature-reactivity feedback
10. **Reach: interpretability**, permutation importance / SHAP-style attribution over sensors
