//! Phase 7 CLI entry point: out-of-distribution (OOD) accident detection.
//!
//! WHAT THIS FILE DOES, IN PLAIN LANGUAGE:
//! Every classifier in this project so far has been trained and tested on
//! the same 18 known accident types. Real plants don't get that guarantee.
//! An operator will eventually see something the model was never trained
//! on, and the useful behavior there isn't "confidently call it the
//! closest known accident type," it's "recognize this doesn't look like
//! anything I know and say so."
//!
//! To actually test that without real novel-accident data (which doesn't
//! exist, NPPAD only has these 18 types), this binary holds out one or
//! more accident types ENTIRELY: their runs are never used for training,
//! not even for the run-level test split Phases 3-4 use, they're kept
//! completely separate as a stand-in for "a real novel scenario." It then
//! trains `models::boosting`'s gradient boosting classifier on the
//! remaining (known) types only, and checks whether the model's own
//! confidence (its softmax max probability, the standard "maximum
//! softmax probability" baseline OOD detector, Hendrycks & Gimpel 2017)
//! is measurably lower on the held-out type's windows than on ordinary
//! held-out TEST windows from types it actually knows about.
//!
//! HOW TO RUN THIS:
//! `cargo build --release -p models` then `./target/release/detect-ood`
//! Optional arguments override the defaults:
//! `detect-ood <window_size_rows> <stride_rows> <test_fraction> <max_per_class> <ood_types_comma_separated>`

use core::data;
use core::features::{self, WindowFeatures};
use core::split::{self, Split};
use models::boosting::{BoostParams, GradientBoostedClassifier};
use models::metrics;
use models::tree::TreeParams as RegressionTreeParams;
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let window_size: usize = args.first().map(|s| s.parse()).transpose()?.unwrap_or(5);
    let stride: usize = args.get(1).map(|s| s.parse()).transpose()?.unwrap_or(3);
    let test_fraction: f64 = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(0.2);
    let max_per_class: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(500);
    // FLB and SGBTR: both "Severity"-class types with plenty of runs (so
    // the held-out set is large enough to draw a real conclusion from),
    // and neither has already been singled out in an earlier phase's
    // findings (unlike LOCA/LOCAC or RW/RI), so this is a reasonably
    // "generic" choice of what a novel accident type might look like
    // rather than one picked to make the result look better or worse.
    let ood_types_arg = args.get(4).cloned().unwrap_or_else(|| "FLB,SGBTR".to_string());
    let ood_types: Vec<String> = ood_types_arg.split(',').map(|s| s.trim().to_string()).collect();

    println!("Loading NPPAD dataset...");
    let all_samples = data::load_dataset("data/raw/Operation_csv_data")?;

    // Split into "known" (everything the model is allowed to see) and
    // "ood" (held out entirely, never touched during training or even
    // during the known-class run-level split).
    let (known_samples, ood_samples): (Vec<_>, Vec<_>) =
        all_samples.into_iter().partition(|s| !ood_types.contains(&s.accident_type));
    println!(
        "Known samples: {} (excluding {:?}). OOD holdout samples: {}.",
        known_samples.len(),
        ood_types,
        ood_samples.len()
    );

    let mut known_labels: Vec<String> = known_samples.iter().map(|s| s.accident_type.clone()).collect();
    known_labels.sort();
    known_labels.dedup();
    let n_known_classes = known_labels.len();
    let label_index: HashMap<&str, usize> =
        known_labels.iter().enumerate().map(|(i, l)| (l.as_str(), i)).collect();

    // Run-level split computed ONLY over the known-class samples, the OOD
    // types were already removed above, so they can't leak into this
    // split's bookkeeping at all.
    let run_split = split::split_by_run(&known_samples, test_fraction);
    println!(
        "Among known types, {} have only one run and are train-only: {:?}",
        run_split.single_run_types.len(),
        run_split.single_run_types
    );

    println!("Extracting windows (window_size={window_size} rows, stride={stride} rows)...");
    let known_windows = features::extract_dataset_features(&known_samples, window_size, stride);
    let ood_windows = features::extract_dataset_features(&ood_samples, window_size, stride);

    let mut train_windows = Vec::new();
    let mut known_test_windows = Vec::new();
    for window in known_windows {
        let goes_to_train = split::is_single_run_type(&run_split, &window.accident_type)
            || run_split.assignment_for(&window.accident_type, &window.severity_id) == Some(Split::Train);
        if goes_to_train {
            train_windows.push(window);
        } else {
            known_test_windows.push(window);
        }
    }

    let train_windows = split::subsample_by_class(
        train_windows,
        |w: &WindowFeatures| w.accident_type.as_str(),
        max_per_class,
    );
    let known_test_windows = split::subsample_by_class(
        known_test_windows,
        |w: &WindowFeatures| w.accident_type.as_str(),
        (max_per_class / 2).max(1),
    );
    // The OOD windows were never part of any training decision, so there's
    // no "train vs. test" distinction to preserve here, just subsample for
    // evaluation tractability, the same way the other groups are.
    let ood_windows = split::subsample_by_class(
        ood_windows,
        |w: &WindowFeatures| w.accident_type.as_str(),
        (max_per_class / 2).max(1),
    );
    println!(
        "After subsampling: {} train, {} known-test, {} OOD windows.",
        train_windows.len(),
        known_test_windows.len(),
        ood_windows.len()
    );

    let x_train: Vec<Vec<f64>> = train_windows.iter().map(|w| w.features.clone()).collect();
    let y_train: Vec<usize> = train_windows.iter().map(|w| label_index[w.accident_type.as_str()]).collect();

    // Same hyperparameters as Phase 4's gradient boosting, only the label
    // set is smaller here (known classes only), so this is a fair "same
    // model, different question" comparison, not a retuned one.
    let boost_params = BoostParams {
        n_rounds: 18,
        learning_rate: 0.3,
        tree_params: RegressionTreeParams { max_depth: 3, min_samples_leaf: 10 },
    };
    println!("\nTraining gradient boosting on {n_known_classes} known classes...");
    let model = GradientBoostedClassifier::fit(&x_train, &y_train, n_known_classes, boost_params);

    // --- Sanity check: ordinary classification quality on known-test ----
    // (i.e. "did excluding the OOD types break anything about the model's
    // ability to do its original job on the classes it still knows?")
    let x_known_test: Vec<Vec<f64>> = known_test_windows.iter().map(|w| w.features.clone()).collect();
    let y_known_test: Vec<usize> =
        known_test_windows.iter().map(|w| label_index[w.accident_type.as_str()]).collect();
    let known_preds = model.predict(&x_known_test);
    let classification_eval = metrics::evaluate(&y_known_test, &known_preds, &known_labels);
    println!(
        "Known-class test accuracy={:.4}, macro_f1={:.4}",
        classification_eval.accuracy, classification_eval.macro_f1
    );

    // --- The actual OOD question: max softmax probability ----------------
    let known_test_probs = model.predict_proba(&x_known_test);
    let mut known_test_max_prob: Vec<f64> = known_test_probs.iter().map(|p| max_of(p)).collect();

    let x_ood: Vec<Vec<f64>> = ood_windows.iter().map(|w| w.features.clone()).collect();
    let ood_probs = model.predict_proba(&x_ood);
    let mut ood_max_prob: Vec<f64> = ood_probs.iter().map(|p| max_of(p)).collect();

    // Calibrate a detection threshold from the KNOWN-test distribution
    // only (never from the OOD windows, using the OOD data to pick the
    // threshold would be leakage of exactly the kind this whole project
    // has tried to avoid elsewhere). The 5th percentile of known max-prob
    // values means: "we accept flagging about 5% of genuinely known
    // windows as unusual, in exchange for having a threshold at all."
    known_test_max_prob.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let threshold_idx = ((known_test_max_prob.len() as f64) * 0.05).floor() as usize;
    let threshold = known_test_max_prob[threshold_idx.min(known_test_max_prob.len() - 1)];

    let false_alarm_rate =
        known_test_max_prob.iter().filter(|&&p| p < threshold).count() as f64 / known_test_max_prob.len() as f64;
    ood_max_prob.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ood_detection_rate =
        ood_max_prob.iter().filter(|&&p| p < threshold).count() as f64 / ood_max_prob.len() as f64;

    println!("\n--- Out-of-distribution detection (max softmax probability) ---");
    println!("Threshold (5th percentile of known-test max-prob): {threshold:.4}");
    println!("Mean known-test max-prob: {:.4}", mean(&known_test_max_prob));
    println!("Mean OOD max-prob:        {:.4}", mean(&ood_max_prob));
    println!("False alarm rate on known-test windows: {:.4}", false_alarm_rate);
    println!("Detection rate on OOD ({ood_types:?}) windows: {:.4}", ood_detection_rate);

    // A single percentile choice for the threshold is somewhat arbitrary
    // (a different false-alarm tolerance would give a different detection
    // rate), so also report a threshold-INDEPENDENT summary: the
    // probability that a randomly chosen OOD window's max-prob is lower
    // than a randomly chosen known-test window's max-prob. This is
    // exactly AUROC for this detection task, computed directly from its
    // definition (treating "OOD" as the positive class we're trying to
    // detect, using max-prob as the detector's score, where LOWER means
    // "more likely OOD"), rather than through the more standard
    // ranking-based formula, since a direct pairwise count is easy to
    // follow and, for the number of pairs here, plenty fast.
    let mut concordant_pairs = 0.0;
    for &ood_score in &ood_max_prob {
        for &known_score in &known_test_max_prob {
            if ood_score < known_score {
                concordant_pairs += 1.0;
            } else if ood_score == known_score {
                concordant_pairs += 0.5; // ties count as half-credit, standard AUROC convention
            }
        }
    }
    let auroc = concordant_pairs / (ood_max_prob.len() as f64 * known_test_max_prob.len() as f64);
    println!("AUROC (threshold-independent): {auroc:.4}");

    #[derive(serde::Serialize)]
    struct OodResults {
        window_size_rows: usize,
        stride_rows: usize,
        test_fraction: f64,
        max_per_class_train: usize,
        ood_types: Vec<String>,
        known_labels: Vec<String>,
        classification_eval: metrics::EvalResult,
        threshold: f64,
        false_alarm_rate: f64,
        ood_detection_rate: f64,
        auroc: f64,
        known_test_max_prob_sorted: Vec<f64>,
        ood_max_prob_sorted: Vec<f64>,
    }

    let results = OodResults {
        window_size_rows: window_size,
        stride_rows: stride,
        test_fraction,
        max_per_class_train: max_per_class,
        ood_types: ood_types.clone(),
        known_labels: known_labels.clone(),
        classification_eval,
        threshold,
        false_alarm_rate,
        ood_detection_rate,
        auroc,
        known_test_max_prob_sorted: known_test_max_prob,
        ood_max_prob_sorted: ood_max_prob,
    };

    std::fs::create_dir_all("data/results")?;
    std::fs::write("data/results/ood_results.json", serde_json::to_string_pretty(&results)?)?;
    println!("\nWrote data/results/ood_results.json");

    Ok(())
}

fn max_of(values: &[f64]) -> f64 {
    values.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}
