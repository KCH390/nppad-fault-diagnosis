//! Phase 4 CLI entry point: from-scratch multi-class classification.
//!
//! WHAT THIS FILE DOES, IN PLAIN LANGUAGE:
//! Structurally almost identical to Phase 3's `main.rs` — load data, run
//! the exact same `core::split` run-level split, extract the exact same
//! rolling-window features — but trains two ZERO-DEPENDENCY models instead
//! of library-backed ones: a single Gini-impurity CART tree
//! (`cart_classifier`) and multi-class gradient boosting built from
//! scratch on top of `tree.rs`'s regression trees (`boosting`). Same
//! `metrics::evaluate` at the end, so Phase 3 and Phase 4's numbers are
//! directly comparable — same split, same features, same evaluation code,
//! only the model implementation differs.
//!
//! HOW TO RUN THIS:
//! `cargo run -p models --bin train-scratch`
//! (`train-baseline`, Phase 3's binary, stays the crate's `default-run`,
//! so plain `cargo run -p models` still runs Phase 3 — this one needs an
//! explicit `--bin`.)
//!
//! Optional arguments override the defaults:
//! `cargo run -p models --bin train-scratch -- <window_size_rows> <stride_rows> <test_fraction> <max_per_class>`

use core::data;
use core::features::{self, WindowFeatures};
use core::split::{self, Split};
use models::boosting::{BoostParams, GradientBoostedClassifier};
use models::cart_classifier::{self, ClassificationTree, TreeParams as ClassifierTreeParams};
use models::metrics;
use models::tree::TreeParams as RegressionTreeParams;
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let window_size: usize = args.first().map(|s| s.parse()).transpose()?.unwrap_or(5);
    let stride: usize = args.get(1).map(|s| s.parse()).transpose()?.unwrap_or(3);
    let test_fraction: f64 = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(0.2);
    // A smaller default cap than Phase 3's 2000/class: hand-rolled trees
    // that scan every feature at every node (no histogram binning, no
    // parallelism — see the README's Phase 4 notes) are meaningfully
    // slower than smartcore's optimized implementation, and gradient
    // boosting trains 18 classes' worth of trees PER ROUND on top of that.
    // 500/class keeps a full run tractable without changing what's being
    // measured — it's still the same run-level split, just a smaller slice
    // of it.
    let max_per_class: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(500);

    println!("Loading NPPAD dataset...");
    let samples = data::load_dataset("data/raw/Operation_csv_data")?;
    println!("Loaded {} samples.", samples.len());

    // Identical split logic to Phase 3 — see models/src/main.rs and
    // core::split for the full reasoning. Reusing it here (rather than
    // rederiving anything) is what makes Phase 3 vs. Phase 4 a fair
    // comparison instead of two experiments on different data.
    let run_split = split::split_by_run(&samples, test_fraction);
    println!(
        "Run-level split: {} accident types have only one run and are train-only: {:?}",
        run_split.single_run_types.len(),
        run_split.single_run_types
    );

    let mut labels: Vec<String> = samples.iter().map(|s| s.accident_type.clone()).collect();
    labels.sort();
    labels.dedup();
    let n_classes = labels.len();
    let label_index: HashMap<&str, usize> = cart_classifier::build_label_index(&labels);

    println!(
        "Extracting rolling-window features (window_size={window_size} rows, stride={stride} rows)..."
    );
    let all_windows = features::extract_dataset_features(&samples, window_size, stride);
    println!("Extracted {} windows total.", all_windows.len());

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

    let train_windows =
        split::subsample_by_class(train_windows, |w: &WindowFeatures| w.accident_type.as_str(), max_per_class);
    let test_windows = split::subsample_by_class(
        test_windows,
        |w: &WindowFeatures| w.accident_type.as_str(),
        max_per_class / 2,
    );
    println!(
        "After subsampling (max {max_per_class}/class train, {}/class test): {} train, {} test.",
        max_per_class / 2,
        train_windows.len(),
        test_windows.len()
    );

    let x_train: Vec<Vec<f64>> = train_windows.iter().map(|w| w.features.clone()).collect();
    let y_train: Vec<usize> = train_windows.iter().map(|w| label_index[w.accident_type.as_str()]).collect();
    let x_test: Vec<Vec<f64>> = test_windows.iter().map(|w| w.features.clone()).collect();
    let y_test: Vec<usize> = test_windows.iter().map(|w| label_index[w.accident_type.as_str()]).collect();

    // === Model 1: single CART classification tree (Gini impurity) =========
    println!("\nTraining from-scratch CART classification tree...");
    let tree_params = ClassifierTreeParams { max_depth: 8, min_samples_leaf: 5 };
    let cart_model = ClassificationTree::fit(&x_train, &y_train, n_classes, tree_params);
    let cart_preds = cart_model.predict(&x_test);
    let cart_eval = metrics::evaluate(&y_test, &cart_preds, &labels);
    print_eval("From-Scratch CART Tree", &cart_eval);

    // === Model 2: from-scratch multi-class gradient boosting ===============
    // Shallower trees (max_depth=3) and a modest round count are standard
    // gradient boosting practice — each individual tree is meant to be a
    // "weak learner" that only nudges the ensemble a little; the boosting
    // process itself is what builds up a strong model over many rounds,
    // not any single deep tree.
    println!("\nTraining from-scratch gradient boosting (18 rounds x 18 classes = 324 trees)...");
    let boost_params = BoostParams {
        n_rounds: 18,
        learning_rate: 0.3,
        tree_params: RegressionTreeParams { max_depth: 3, min_samples_leaf: 10 },
    };
    let boosting_model = GradientBoostedClassifier::fit(&x_train, &y_train, n_classes, boost_params);
    let boosting_preds = boosting_model.predict(&x_test);
    let boosting_eval = metrics::evaluate(&y_test, &boosting_preds, &labels);
    print_eval("From-Scratch Gradient Boosting", &boosting_eval);

    // --- Write results -------------------------------------------------
    #[derive(serde::Serialize)]
    struct ScratchResults {
        window_size_rows: usize,
        stride_rows: usize,
        test_fraction: f64,
        max_per_class_train: usize,
        single_run_types_train_only: Vec<String>,
        train_windows: usize,
        test_windows: usize,
        cart_tree: metrics::EvalResult,
        gradient_boosting: metrics::EvalResult,
    }

    let results = ScratchResults {
        window_size_rows: window_size,
        stride_rows: stride,
        test_fraction,
        max_per_class_train: max_per_class,
        single_run_types_train_only: run_split.single_run_types.clone(),
        train_windows: train_windows.len(),
        test_windows: test_windows.len(),
        cart_tree: cart_eval,
        gradient_boosting: boosting_eval,
    };

    std::fs::create_dir_all("data/results")?;
    std::fs::write(
        "data/results/scratch_results.json",
        serde_json::to_string_pretty(&results)?,
    )?;
    println!("\nWrote data/results/scratch_results.json");

    Ok(())
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
