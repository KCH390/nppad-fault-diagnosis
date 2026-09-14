//! Generates the Phase 7 chart as an SVG file under `data/results/charts/`.
//!
//! WHAT THIS FILE DOES, IN PLAIN LANGUAGE:
//! Reads `data/results/ood_results.json` (written by `models`'s
//! `detect-ood` binary) and plots the empirical distribution of
//! max-softmax-probability for known-test windows against held-out OOD
//! windows, as two sorted-value curves (an empirical CDF, effectively) on
//! one chart via the existing `multi_line_chart`. No new charting code
//! needed, since "sorted values against their rank" is exactly what that
//! function already draws for any two labeled series.
//!
//! HOW TO RUN THIS:
//! `cargo run -p models --bin detect-ood` first (to produce
//! ood_results.json), then `cargo run -p charts --bin ood_charts`

use charts::Series;
use serde::Deserialize;

#[derive(Deserialize)]
struct OodResults {
    ood_types: Vec<String>,
    auroc: f64,
    threshold: f64,
    known_test_max_prob_sorted: Vec<f64>,
    ood_max_prob_sorted: Vec<f64>,
}

fn main() -> anyhow::Result<()> {
    let json = std::fs::read_to_string("data/results/ood_results.json").map_err(|e| {
        anyhow::anyhow!("couldn't read data/results/ood_results.json (run `cargo run -p models --bin detect-ood` first): {e}")
    })?;
    let results: OodResults = serde_json::from_str(&json)?;

    std::fs::create_dir_all("data/results/charts")?;

    // Convert each sorted value list into (rank_fraction, value) points,
    // an empirical CDF, effectively "as you increase the max-prob
    // threshold from 0 to 1, what fraction of windows in this group have
    // AT MOST this max-prob." A left-shifted curve (like the OOD one
    // should be) means most of that group's windows have low confidence.
    let known_points: Vec<(f64, f64)> = results
        .known_test_max_prob_sorted
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as f64 / results.known_test_max_prob_sorted.len() as f64, v))
        .collect();
    let ood_points: Vec<(f64, f64)> = results
        .ood_max_prob_sorted
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as f64 / results.ood_max_prob_sorted.len() as f64, v))
        .collect();

    let series = vec![
        Series { label: "Known-test windows".to_string(), points: known_points },
        Series { label: format!("OOD windows ({})", results.ood_types.join(", ")), points: ood_points },
    ];

    let path = "data/results/charts/ood_max_prob_distribution.svg";
    charts::multi_line_chart(
        &format!(
            "Phase 7: Max Softmax Probability, Known vs. OOD (AUROC={:.3}, threshold={:.3})",
            results.auroc, results.threshold
        ),
        &series,
        "Rank fraction within group",
        "Max softmax probability",
        path,
    )?;
    println!("Wrote {path}");

    Ok(())
}
