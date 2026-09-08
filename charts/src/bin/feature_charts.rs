//! Generates the Phase 2 feature-engineering charts as SVG files under
//! `data/results/charts/`.
//!
//! WHAT THIS FILE DOES, IN PLAIN LANGUAGE:
//! Two charts, both meant to answer "what did windowing actually do to the
//! data?" rather than just "here is a chart":
//!   1. Window counts per accident type — the post-windowing version of
//!      Phase 1's class-balance chart. Windowing doesn't just carry over
//!      the raw sample imbalance, it can reshape it, since longer runs
//!      produce more windows than shorter ones even within the same
//!      accident type.
//!   2. A single sensor's raw signal vs. its rolling mean, for one run —
//!      a visual explanation of what "rolling window features" actually
//!      compute, rather than asking you to just trust the arithmetic in
//!      `core::features`.
//!
//! HOW TO RUN THIS:
//! `cargo run -p charts --bin feature_charts`
//! (The `charts` crate now has two binaries too — `eda-charts` from Phase 1,
//! and this one. `-p charts` picks the crate; `--bin feature_charts` picks
//! which of its binaries to run, the same reason `core` needed `--bin
//! extract_features` to run anything other than its own default.)

use core::features;

fn main() -> anyhow::Result<()> {
    let data_root = "data/raw/Operation_csv_data";
    println!("Loading dataset for Phase 2 charts...");
    let samples = core::data::load_dataset(data_root)?;
    println!("Loaded {} samples.", samples.len());

    std::fs::create_dir_all("data/results/charts")?;

    // Match Phase 2's default window parameters (see
    // core/src/bin/extract_features.rs) so this chart reflects the same
    // windowing that actually produced data/features/windows.csv.
    let window_size = 5;
    let stride = 3;

    // --- Chart 1: window counts per accident type ---------------------
    let mut accident_types: Vec<&str> = samples.iter().map(|s| s.accident_type.as_str()).collect();
    accident_types.sort();
    accident_types.dedup();

    let mut window_counts: Vec<(String, f64)> = Vec::new();
    for &accident_type in &accident_types {
        let group: Vec<&core::data::Sample> =
            samples.iter().filter(|s| s.accident_type == accident_type).collect();
        let total_windows: usize = group
            .iter()
            .map(|s| features::extract_rolling_features(s, window_size, stride).len())
            .sum();
        window_counts.push((accident_type.to_string(), total_windows as f64));
    }

    charts::bar_chart(
        "NPPAD Phase 2: Windows per Accident Type (window=5 rows, stride=3 rows)",
        &window_counts,
        "Window count",
        "data/results/charts/window_class_balance.svg",
    )?;
    println!("Wrote data/results/charts/window_class_balance.svg");

    // --- Chart 2: raw signal vs. rolling mean, one representative run --
    // Same accident type and sensor as Phase 1's power-response chart
    // (LOCA / PWR) so the two charts are easy to relate to each other.
    let accident_type = "LOCA";
    let sensor_name = "PWR";

    let Some(sample) = samples.iter().find(|s| s.accident_type == accident_type) else {
        anyhow::bail!("no samples found for {accident_type}");
    };
    let Some(sensor_idx) = sample.sensor_names.iter().position(|n| n == sensor_name) else {
        anyhow::bail!("sensor {sensor_name} not found");
    };

    // The raw signal: every (time, value) point straight from the sample.
    let raw_points: Vec<(f64, f64)> = sample
        .time_seconds
        .iter()
        .zip(sample.values.iter())
        .map(|(&t, row)| (t, row[sensor_idx]))
        .collect();

    // The rolling mean: reuse the exact same `extract_rolling_features`
    // function the real Phase 2 pipeline uses (not a simplified
    // reimplementation just for this chart), plotted at each window's
    // midpoint. `PWR` is the first sensor alphabetically after `P` in the
    // canonical schema; rather than hardcode its position in the flattened
    // feature vector, we look it up the same way `feature_names` builds
    // it, so this chart can't silently drift out of sync if the feature
    // layout ever changes.
    let feature_names = features::feature_names(&sample.sensor_names);
    let mean_col_name = format!("{sensor_name}_mean");
    let mean_feature_idx = feature_names
        .iter()
        .position(|n| n == &mean_col_name)
        .expect("PWR_mean must exist in the feature name list");

    let windows = features::extract_rolling_features(sample, window_size, stride);
    let rolling_mean_points: Vec<(f64, f64)> = windows
        .iter()
        .map(|w| {
            let midpoint = (w.window_start_seconds + w.window_end_seconds) / 2.0;
            (midpoint, w.features[mean_feature_idx])
        })
        .collect();

    let series = vec![
        charts::Series {
            label: format!("{sensor_name} (raw)"),
            points: raw_points,
        },
        charts::Series {
            label: format!("{sensor_name} (rolling mean, window={window_size} rows)"),
            points: rolling_mean_points,
        },
    ];

    charts::multi_line_chart(
        &format!("NPPAD Phase 2: Raw vs. Rolling Mean ({accident_type}, {sensor_name})"),
        &series,
        "Time (s)",
        "Power, core thermal (%)",
        "data/results/charts/rolling_mean_illustration.svg",
    )?;
    println!("Wrote data/results/charts/rolling_mean_illustration.svg");

    Ok(())
}
