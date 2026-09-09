//! Generates the Phase 5 charts as SVG files under `data/results/charts/`.
//!
//! WHAT THIS FILE DOES, IN PLAIN LANGUAGE:
//! Reads `data/results/kinetics_results.json` (written by `physics`'s
//! `simulate-kinetics` binary) and draws two overlay charts: the simulated
//! positive reactivity ramp against a real RW (rod withdrawal) trace, and
//! the simulated negative ramp against a real RI (rod insertion) trace.
//! Same "read the small JSON summary, don't touch the crate that produced
//! it" pattern as `baseline_charts.rs` and `scratch_charts.rs` — this
//! binary needs no `physics` dependency at all.
//!
//! HOW TO RUN THIS:
//! `cargo run -p physics` first (to produce kinetics_results.json), then
//! `cargo run -p charts --bin kinetics_charts`

use charts::Series;
use serde::Deserialize;

#[derive(Deserialize)]
struct KineticsResults {
    positive_ramp_reactivity: f64,
    negative_ramp_reactivity: f64,
    simulated_positive_ramp: Vec<(f64, f64)>,
    simulated_negative_ramp: Vec<(f64, f64)>,
    real_rw_relative_power: Vec<(f64, f64)>,
    real_ri_relative_power: Vec<(f64, f64)>,
}

fn main() -> anyhow::Result<()> {
    let json = std::fs::read_to_string("data/results/kinetics_results.json").map_err(|e| {
        anyhow::anyhow!(
            "couldn't read data/results/kinetics_results.json (run `cargo run -p physics` first): {e}"
        )
    })?;
    let results: KineticsResults = serde_json::from_str(&json)?;

    std::fs::create_dir_all("data/results/charts")?;

    charts::multi_line_chart(
        "Phase 5: Positive Reactivity Ramp vs. Real RW (Rod Withdrawal)",
        &[
            Series {
                label: format!("Simulated point kinetics ({:+.4} rho ramp)", results.positive_ramp_reactivity),
                points: results.simulated_positive_ramp,
            },
            Series { label: "Real NPPAD RW (relative power)".to_string(), points: results.real_rw_relative_power },
        ],
        "Time (s)",
        "Relative power (n / n0)",
        "data/results/charts/kinetics_rw_comparison.svg",
    )?;
    println!("Wrote data/results/charts/kinetics_rw_comparison.svg");

    charts::multi_line_chart(
        "Phase 5: Negative Reactivity Ramp vs. Real RI (Rod Insertion)",
        &[
            Series {
                label: format!("Simulated point kinetics ({:+.4} rho ramp)", results.negative_ramp_reactivity),
                points: results.simulated_negative_ramp,
            },
            Series { label: "Real NPPAD RI (relative power)".to_string(), points: results.real_ri_relative_power },
        ],
        "Time (s)",
        "Relative power (n / n0)",
        "data/results/charts/kinetics_ri_comparison.svg",
    )?;
    println!("Wrote data/results/charts/kinetics_ri_comparison.svg");

    Ok(())
}
