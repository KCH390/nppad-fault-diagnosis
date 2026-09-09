//! Generates the Phase 4 charts as SVG files under `data/results/charts/`.
//!
//! WHAT THIS FILE DOES, IN PLAIN LANGUAGE:
//! Same structure as `baseline_charts.rs` (Phase 3) — reads
//! `data/results/scratch_results.json` (written by `models`'s
//! `train-scratch` binary) and draws one confusion-matrix heatmap per
//! from-scratch model. No retraining, no `models` crate dependency — just
//! reading the small JSON summary those models already produced.
//!
//! HOW TO RUN THIS:
//! `cargo run -p models --bin train-scratch` first (to produce
//! scratch_results.json), then
//! `cargo run -p charts --bin scratch_charts`

use serde::Deserialize;

/// Same duplication-over-dependency tradeoff as baseline_charts.rs's
/// `EvalResult`/`BaselineResults` — see that file's comment for why.
#[derive(Deserialize)]
struct EvalResult {
    accuracy: f64,
    macro_f1: f64,
    labels: Vec<String>,
    confusion_matrix: Vec<Vec<usize>>,
}

#[derive(Deserialize)]
struct ScratchResults {
    cart_tree: EvalResult,
    gradient_boosting: EvalResult,
}

fn main() -> anyhow::Result<()> {
    let json = std::fs::read_to_string("data/results/scratch_results.json").map_err(|e| {
        anyhow::anyhow!(
            "couldn't read data/results/scratch_results.json (run `cargo run -p models --bin train-scratch` first): {e}"
        )
    })?;
    let results: ScratchResults = serde_json::from_str(&json)?;

    std::fs::create_dir_all("data/results/charts")?;

    for (name, eval, filename) in [
        ("From-Scratch CART Tree", &results.cart_tree, "confusion_matrix_cart_scratch.svg"),
        ("From-Scratch Gradient Boosting", &results.gradient_boosting, "confusion_matrix_boosting_scratch.svg"),
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
