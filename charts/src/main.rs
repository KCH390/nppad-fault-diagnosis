//! Generates the three Phase 1 EDA charts as SVG files under
//! `data/results/charts/`.
//!
//! WHAT THIS FILE DOES, IN PLAIN LANGUAGE:
//! 1. Loads the dataset (reusing `core::data::load_dataset` — the exact
//!    same loader the main `core` binary uses, so there's only one place
//!    that logic can go wrong).
//! 2. Builds three charts from it: class balance, run-duration histogram,
//!    and a reactor-power comparison across a handful of accident types.
//! 3. Writes each as an SVG into `data/results/charts/`.
//!
//! HOW TO RUN THIS:
//! From the workspace root: `cargo run -p charts`
//! (The `-p charts` is required because `charts` is deliberately NOT in the
//! workspace's `default-members` — see the root Cargo.toml comment — so
//! plain `cargo run` will build/run `core` instead.)

fn main() -> anyhow::Result<()> {
    let data_root = "data/raw/Operation_csv_data";
    println!("Loading dataset for charting...");
    let samples = core::data::load_dataset(data_root)?;
    println!("Loaded {} samples.", samples.len());

    std::fs::create_dir_all("data/results/charts")?;

    // --- Chart 1: class balance -------------------------------------
    // Same grouping approach as core::eda::summarize (sorted, deduped
    // accident type names), but we only need the counts here, not the
    // full duration statistics.
    let mut accident_types: Vec<&str> = samples.iter().map(|s| s.accident_type.as_str()).collect();
    accident_types.sort();
    accident_types.dedup();

    let class_counts: Vec<(String, f64)> = accident_types
        .iter()
        .map(|&t| {
            let count = samples.iter().filter(|s| s.accident_type == t).count();
            (t.to_string(), count as f64)
        })
        .collect();

    charts::bar_chart(
        "NPPAD: Samples per Accident Type",
        &class_counts,
        "Sample count",
        "data/results/charts/class_balance.svg",
    )?;
    println!("Wrote data/results/charts/class_balance.svg");

    // --- Chart 2: run-duration histogram -----------------------------
    // Every sample's total duration, across the WHOLE dataset (not split
    // by accident type) — this is the chart that visually makes the point
    // that these runs are not a fixed length, which matters for any model
    // downstream that expects fixed-size input.
    let durations: Vec<f64> = samples.iter().filter_map(|s| s.duration_seconds()).collect();

    charts::histogram(
        "NPPAD: Run Duration Distribution",
        &durations,
        500.0, // 500-second bins
        "Duration (s)",
        "data/results/charts/duration_histogram.svg",
    )?;
    println!("Wrote data/results/charts/duration_histogram.svg");

    // --- Chart 3: reactor power response by accident type ------------
    // A curated subset rather than all 18 accident types — with this many
    // distinct colors on one chart, more lines would hurt legibility more
    // than they'd add insight. This selection deliberately spans both
    // "Severity" (LOCA, SGATR) and "Other" (Normal, ATWS, TT, LOF)
    // accident classes from NPPAD's own Table 1, so the comparison shows
    // genuinely different plant-response shapes, not just several flavors
    // of the same failure mode.
    let featured_types = ["Normal", "LOCA", "SGATR", "ATWS", "TT", "LOF"];
    let sensor_name = "PWR"; // Power, Core thermal (%) — see NPPAD Table 2.

    let mut series = Vec::new();
    for &accident_type in &featured_types {
        // For "Severity"-class accidents (LOCA, SGATR) there are ~100
        // files; we just need ONE representative run per type for this
        // comparison chart, so we take whichever sample we find first
        // rather than trying to pick a "canonical" severity level — a
        // choice worth revisiting once we're past EDA and into deliberate
        // feature engineering.
        let Some(sample) = samples.iter().find(|s| s.accident_type == accident_type) else {
            // `let ... else` is a Rust pattern for "either successfully
            // destructure this value, or run this else-block (which must
            // diverge, e.g. via `continue`/`return`/`panic!`)." Here, if a
            // featured accident type is somehow missing from the loaded
            // dataset, we skip it and keep going rather than crashing the
            // whole chart-generation run over one missing folder.
            eprintln!("warning: no samples found for {accident_type}, skipping");
            continue;
        };

        let Some(sensor_idx) = sample.sensor_names.iter().position(|n| n == sensor_name) else {
            eprintln!("warning: sensor {sensor_name} not found, skipping {accident_type}");
            continue;
        };

        let points: Vec<(f64, f64)> = sample
            .time_seconds
            .iter()
            .zip(sample.values.iter())
            // `.zip(...)` pairs up two iterators element-by-element —
            // here, each timestamp with its corresponding row of sensor
            // values — stopping as soon as either iterator runs out. Since
            // `time_seconds.len() == values.len()` by construction in
            // `data.rs`, both run out together.
            .map(|(&t, row)| (t, row[sensor_idx]))
            .collect();

        series.push(charts::Series {
            label: format!("{accident_type} (severity {})", sample.severity_id),
            points,
        });
    }

    charts::multi_line_chart(
        "NPPAD: Reactor Power Response by Accident Type",
        &series,
        "Time (s)",
        "Power, core thermal (%)",
        "data/results/charts/power_response_by_accident.svg",
    )?;
    println!("Wrote data/results/charts/power_response_by_accident.svg");

    Ok(())
}
