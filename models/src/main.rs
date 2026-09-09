//! Phase 3 CLI entry point: baseline multi-class accident classification.
//!
//! WHAT THIS FILE DOES, IN PLAIN LANGUAGE:
//! 1. Loads the dataset and extracts rolling-window features, reusing
//!    exactly the same `core::data` and `core::features` code Phase 2
//!    used — there's still only one implementation of "how to load NPPAD"
//!    and "how to window it" in this whole project.
//! 2. Splits at the RUN level (see `core::split`) so no window's near-twin
//!    ends up on the other side of the train/test boundary.
//! 3. Subsamples the (very lopsided — see Phase 2 README notes) window
//!    counts per class down to something tractable to train on.
//! 4. Trains two independent baselines — a Gaussian Naive Bayes (via
//!    `linfa-bayes`) and a Random Forest (via `smartcore`) — on the exact
//!    same train/test split, so their results are directly comparable.
//! 5. Writes `data/results/baseline_results.json` (metrics + confusion
//!    matrices for both models) and prints a summary table.
//!
//! HOW TO RUN THIS:
//! `cargo run -p models`
//! Optional arguments override the defaults:
//! `cargo run -p models -- <window_size_rows> <stride_rows> <test_fraction> <max_per_class>`

use core::data;
use core::features::{self, WindowFeatures};
use core::split::{self, Split};
use linfa::prelude::*;
use linfa_bayes::GaussianNbParams;
use ndarray::{Array1, Array2};
use smartcore::ensemble::random_forest_classifier::{
    RandomForestClassifier, RandomForestClassifierParameters,
};
use smartcore::linalg::basic::matrix::DenseMatrix;
use std::collections::HashMap;

use models::metrics;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let window_size: usize = args.first().map(|s| s.parse()).transpose()?.unwrap_or(5);
    let stride: usize = args.get(1).map(|s| s.parse()).transpose()?.unwrap_or(3);
    let test_fraction: f64 = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(0.2);
    let max_per_class: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(2000);

    println!("Loading NPPAD dataset...");
    let samples = data::load_dataset("data/raw/Operation_csv_data")?;
    println!("Loaded {} samples.", samples.len());

    // --- Run-level split, BEFORE any windowing happens -------------------
    // Splitting the RUNS first (not the windows) is what guarantees no
    // window from a test run's neighborhood ever leaks into training — see
    // core::split's module docs for the full reasoning.
    let run_split = split::split_by_run(&samples, test_fraction);
    println!(
        "Run-level split: {} accident types have only one run and are train-only: {:?}",
        run_split.single_run_types.len(),
        run_split.single_run_types
    );

    // The full, sorted label set — all 18 accident types, single-run or
    // not. The model needs to know about single-run classes too (it's
    // still trained on them), even though they'll never appear in
    // `y_test`.
    let mut labels: Vec<String> = samples.iter().map(|s| s.accident_type.clone()).collect();
    labels.sort();
    labels.dedup();
    let label_index: HashMap<&str, usize> =
        labels.iter().enumerate().map(|(i, l)| (l.as_str(), i)).collect();

    println!(
        "Extracting rolling-window features (window_size={window_size} rows, stride={stride} rows)..."
    );
    let all_windows = features::extract_dataset_features(&samples, window_size, stride);
    println!("Extracted {} windows total.", all_windows.len());

    // --- Route every window to train or test, per core::split's plan -----
    let mut train_windows = Vec::new();
    let mut test_windows = Vec::new();
    for window in all_windows {
        let goes_to_train = split::is_single_run_type(&run_split, &window.accident_type)
            || run_split.assignment_for(&window.accident_type, &window.severity_id) == Some(Split::Train);

        if goes_to_train {
            train_windows.push(window);
        } else {
            test_windows.push(window);
        }
    }
    println!(
        "Routed windows by run: {} train, {} test (before subsampling).",
        train_windows.len(),
        test_windows.len()
    );

    // --- Subsample for tractability --------------------------------------
    // Even after the run split, the most common accident types produce
    // tens of thousands of highly-overlapping windows from a handful of
    // runs (see Phase 2's window_class_balance.svg). Training on all of
    // them buys very little — those windows are mostly redundant — at a
    // real cost in training time. Subsampling evenly across each run's
    // duration (see core::split::subsample_by_class) keeps the training
    // set's time-coverage intact while capping its size.
    let train_windows =
        split::subsample_by_class(train_windows, |w: &WindowFeatures| w.accident_type.as_str(), max_per_class);
    let test_windows = split::subsample_by_class(
        test_windows,
        |w: &WindowFeatures| w.accident_type.as_str(),
        max_per_class / 2, // test sets don't need to be as large as train
    );
    println!(
        "After subsampling (max {max_per_class}/class train, {}/class test): {} train, {} test.",
        max_per_class / 2,
        train_windows.len(),
        test_windows.len()
    );

    // --- Build feature matrices and label vectors -------------------------
    let (x_train, y_train) = to_arrays(&train_windows, &label_index);
    let (x_test, y_test) = to_arrays(&test_windows, &label_index);

    // === Model 1: Gaussian Naive Bayes (linfa) ============================
    println!("\nTraining Gaussian Naive Bayes (linfa)...");
    let train_dataset = Dataset::new(x_train.clone(), y_train.clone());
    let nb_model = GaussianNbParams::new().fit(&train_dataset)?;
    let nb_preds: Array1<usize> = nb_model.predict(&x_test);
    let nb_eval = metrics::evaluate(y_test.as_slice().unwrap(), nb_preds.as_slice().unwrap(), &labels);
    print_eval("Gaussian Naive Bayes", &nb_eval);

    // === Model 2: Random Forest (smartcore) ===============================
    println!("\nTraining Random Forest (smartcore, 50 trees)...");
    let x_train_vec: Vec<Vec<f64>> = x_train.rows().into_iter().map(|r| r.to_vec()).collect();
    let x_train_dm = DenseMatrix::from_2d_vec(&x_train_vec);
    let y_train_i32: Vec<i32> = y_train.iter().map(|&v| v as i32).collect();

    let rf_params = RandomForestClassifierParameters::default().with_n_trees(50);
    let rf_model = RandomForestClassifier::fit(&x_train_dm, &y_train_i32, rf_params)?;

    let x_test_vec: Vec<Vec<f64>> = x_test.rows().into_iter().map(|r| r.to_vec()).collect();
    let x_test_dm = DenseMatrix::from_2d_vec(&x_test_vec);
    let rf_preds_i32 = rf_model.predict(&x_test_dm)?;
    let rf_preds: Vec<usize> = rf_preds_i32.iter().map(|&v| v as usize).collect();
    let rf_eval = metrics::evaluate(y_test.as_slice().unwrap(), &rf_preds, &labels);
    print_eval("Random Forest", &rf_eval);

    // --- Write results -----------------------------------------------------
    #[derive(serde::Serialize)]
    struct BaselineResults {
        window_size_rows: usize,
        stride_rows: usize,
        test_fraction: f64,
        max_per_class_train: usize,
        single_run_types_train_only: Vec<String>,
        train_windows: usize,
        test_windows: usize,
        gaussian_naive_bayes: metrics::EvalResult,
        random_forest: metrics::EvalResult,
    }

    let results = BaselineResults {
        window_size_rows: window_size,
        stride_rows: stride,
        test_fraction,
        max_per_class_train: max_per_class,
        single_run_types_train_only: run_split.single_run_types.clone(),
        train_windows: train_windows.len(),
        test_windows: test_windows.len(),
        gaussian_naive_bayes: nb_eval,
        random_forest: rf_eval,
    };

    std::fs::create_dir_all("data/results")?;
    std::fs::write(
        "data/results/baseline_results.json",
        serde_json::to_string_pretty(&results)?,
    )?;
    println!("\nWrote data/results/baseline_results.json");

    Ok(())
}

/// Turn a `&[WindowFeatures]` into the `(features, labels)` ndarray pair
/// both `linfa` and (via an intermediate `Vec<Vec<f64>>`) `smartcore` want.
fn to_arrays(windows: &[WindowFeatures], label_index: &HashMap<&str, usize>) -> (Array2<f64>, Array1<usize>) {
    let n_rows = windows.len();
    let n_cols = windows.first().map(|w| w.features.len()).unwrap_or(0);

    // `Array2::from_shape_vec` wants one flat Vec<f64> in row-major order —
    // `.flat_map(...)` over each window's `features` (already a flat
    // per-window Vec) does exactly that, without us hand-writing nested
    // loops with manual index arithmetic.
    let flat: Vec<f64> = windows.iter().flat_map(|w| w.features.iter().copied()).collect();
    let x = Array2::from_shape_vec((n_rows, n_cols), flat).expect("row/col shape must match flattened data");

    let y = Array1::from_vec(
        windows
            .iter()
            .map(|w| label_index[w.accident_type.as_str()])
            .collect(),
    );

    (x, y)
}

fn print_eval(model_name: &str, eval: &metrics::EvalResult) {
    println!("{model_name}: accuracy={:.4}, macro_f1={:.4}", eval.accuracy, eval.macro_f1);
    println!("  {:<8} {:>10} {:>10} {:>10} {:>10}", "Class", "Precision", "Recall", "F1", "Support");
    for c in &eval.per_class {
        if c.support > 0 {
            println!(
                "  {:<8} {:>10.3} {:>10.3} {:>10.3} {:>10}",
                c.label, c.precision, c.recall, c.f1, c.support
            );
        }
    }
}
