//! Generates the Phase 3 baseline-comparison charts as SVG files under
//! `data/results/charts/`.
//!
//! WHAT THIS FILE DOES, IN PLAIN LANGUAGE:
//! Reads `data/results/baseline_results.json` (written by `models`) and
//! draws one confusion-matrix heatmap per model. This binary does NOT
//! retrain anything or touch `linfa`/`smartcore` — it only ever reads the
//! small JSON summary those models already produced, which is why `charts`
//! can stay free of the heavy ML dependency tree (see the Cargo.toml note).
//!
//! HOW TO RUN THIS:
//! `cargo run -p models` first (to produce baseline_results.json), then
//! `cargo run -p charts --bin baseline_charts`

use serde::Deserialize;

/// Mirrors the shape of `models::metrics::EvalResult`'s JSON output.
/// We don't import that type directly — doing so would mean depending on
/// the `models` crate, which would pull `linfa`/`smartcore` into `charts`.
/// Duplicating just the handful of fields this chart actually needs is a
/// small amount of repetition in exchange for keeping that isolation
/// intact.
#[derive(Deserialize)]
struct EvalResult {
    accuracy: f64,
    macro_f1: f64,
    labels: Vec<String>,
    confusion_matrix: Vec<Vec<usize>>,
}

#[derive(Deserialize)]
struct BaselineResults {
    gaussian_naive_bayes: EvalResult,
    random_forest: EvalResult,
}

fn main() -> anyhow::Result<()> {
    let json = std::fs::read_to_string("data/results/baseline_results.json")
        .map_err(|e| anyhow::anyhow!("couldn't read data/results/baseline_results.json (run `cargo run -p models` first): {e}"))?;
    let results: BaselineResults = serde_json::from_str(&json)?;

    std::fs::create_dir_all("data/results/charts")?;

    for (name, eval, filename) in [
        ("Gaussian Naive Bayes", &results.gaussian_naive_bayes, "confusion_matrix_gaussian_nb.svg"),
        ("Random Forest", &results.random_forest, "confusion_matrix_random_forest.svg"),
    ] {
        let path = format!("data/results/charts/{filename}");
        charts::confusion_matrix_heatmap(
            &format!("{name}: accuracy={:.3}, macro F1={:.3}", eval.accuracy, eval.macro_f1),
            &eval.labels,
            &eval.confusion_matrix,
            &path,
        )?;
        println!("Wrote {path}");
    }

    Ok(())
}
