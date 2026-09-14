//! Generates the Phase 6 chart as an SVG file under `data/results/charts/`.
//!
//! WHAT THIS FILE DOES, IN PLAIN LANGUAGE:
//! Same pattern as `baseline_charts.rs` and `scratch_charts.rs`: reads
//! `data/results/lstm_results.json` (written by `sequence`'s
//! `train-lstm` binary) and draws its confusion matrix as a heatmap. No
//! dependency on the `sequence` crate itself, so this binary builds and
//! runs independently of whatever `candle` toolchain requirements
//! `sequence` has.
//!
//! HOW TO RUN THIS:
//! `cargo run -p sequence` first (to produce lstm_results.json), then
//! `cargo run -p charts --bin lstm_charts`

use serde::Deserialize;

#[derive(Deserialize)]
struct EvalResult {
    accuracy: f64,
    macro_f1: f64,
    labels: Vec<String>,
    confusion_matrix: Vec<Vec<usize>>,
}

#[derive(Deserialize)]
struct LstmResults {
    eval: EvalResult,
}

fn main() -> anyhow::Result<()> {
    let json = std::fs::read_to_string("data/results/lstm_results.json").map_err(|e| {
        anyhow::anyhow!("couldn't read data/results/lstm_results.json (run `cargo run -p sequence` first): {e}")
    })?;
    let results: LstmResults = serde_json::from_str(&json)?;

    std::fs::create_dir_all("data/results/charts")?;

    let path = "data/results/charts/confusion_matrix_lstm.svg";
    charts::confusion_matrix_heatmap(
        &format!("LSTM: accuracy={:.3}, macro F1={:.3}", results.eval.accuracy, results.eval.macro_f1),
        &results.eval.labels,
        &results.eval.confusion_matrix,
        path,
    )?;
    println!("Wrote {path}");

    Ok(())
}
