//! Phase 2 CLI entry point: rolling-window feature extraction.
//!
//! WHAT THIS FILE DOES, IN PLAIN LANGUAGE:
//! Loads every sample the same way Phase 1's `main.rs` does, then slides a
//! window across each run computing mean/std/slope per sensor (see
//! `features.rs` for the actual math), and writes the result two ways:
//!   - `data/features/windows.csv` — every window as one row, with its
//!     inherited accident-type label. This is the large, regenerable
//!     intermediate that Phase 3's classifier will train on. It's
//!     deterministic from the raw data + this code, so it's gitignored
//!     rather than checked in (same reasoning as the raw dataset itself).
//!   - `data/results/feature_summary.json` — small, checked-in summary:
//!     how many windows came from each accident type, and the window
//!     parameters used to produce them. This is the artifact that's
//!     actually worth version-controlling, since it's what you'd look at
//!     later to remember exactly how Phase 2 windowed the data.
//!
//! HOW TO RUN THIS:
//! `cargo run --bin extract_features`
//! (This crate now has two binaries — `nppad-fault-diagnosis`, the Phase 1
//! EDA tool, and this one. Because there are two, plain `cargo run` alone
//! is now ambiguous for THIS crate specifically, but `default-run` in
//! core/Cargo.toml still makes plain `cargo run` at the workspace root
//! resolve to `nppad-fault-diagnosis`. Use `--bin extract_features` to
//! explicitly pick this one instead.)
//!
//! Optional arguments override the window parameters:
//! `cargo run --bin extract_features -- <window_size_rows> <stride_rows>`

use core::data;
use core::eda; // reused just for the sorted, deduped accident-type name list
use core::features;
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    // `std::env::args()` yields the program name as element 0, then each
    // space-separated argument after `--` on the command line. `.skip(1)`
    // drops the program name, since we only care about the actual
    // arguments a caller passed in.
    let args: Vec<String> = std::env::args().skip(1).collect();

    // Defaults are chosen so that even the shortest run in this dataset
    // (11 rows / ~100 seconds, in the RI accident type — see the Phase 1
    // README notes) still produces multiple windows rather than zero.
    let window_size: usize = args.first().map(|s| s.parse()).transpose()?.unwrap_or(5);
    let stride: usize = args.get(1).map(|s| s.parse()).transpose()?.unwrap_or(3);
    // `.map(|s| s.parse())` turns `Option<&String>` into `Option<Result<usize, _>>`;
    // `.transpose()` flips that into `Result<Option<usize>, _>` so the `?`
    // can propagate a genuine parse error (e.g. someone passing "abc"),
    // while `.unwrap_or(5)` supplies the default only in the "argument
    // wasn't given at all" case (a real `None`), not the "argument was
    // given but garbled" case.

    println!("Loading NPPAD dataset...");
    let data_root = "data/raw/Operation_csv_data";
    let samples = data::load_dataset(data_root)?;
    println!(
        "Loaded {} samples. Extracting rolling-window features (window_size={window_size} rows, stride={stride} rows)...",
        samples.len()
    );

    let windows = features::extract_dataset_features(&samples, window_size, stride);
    println!("Extracted {} windows total.\n", windows.len());

    // Per-accident-type window counts, and — importantly — which samples
    // (if any) were too short to produce even one window with these
    // parameters. Reusing `eda`'s sorted/deduped accident-type list here
    // keeps this summary in the same order as Phase 1's, so the two are
    // easy to compare side by side.
    let summary = eda::summarize(&samples);
    let mut windows_per_type: HashMap<&str, usize> = HashMap::new();
    for w in &windows {
        *windows_per_type.entry(w.accident_type.as_str()).or_insert(0) += 1;
    }

    println!("{:<8} {:>12} {:>10}", "Type", "RawSamples", "Windows");
    let mut zero_window_types = Vec::new();
    for t in &summary.accident_types {
        let count = windows_per_type.get(t.accident_type.as_str()).copied().unwrap_or(0);
        println!("{:<8} {:>12} {:>10}", t.accident_type, t.sample_count, count);
        if count == 0 {
            zero_window_types.push(t.accident_type.clone());
        }
    }
    if !zero_window_types.is_empty() {
        println!(
            "\nWarning: these accident types produced ZERO windows at window_size={window_size}: {zero_window_types:?}"
        );
    }

    // --- Write data/features/windows.csv (large, gitignored) -----------
    std::fs::create_dir_all("data/features")?;
    let sensor_names = &samples[0].sensor_names; // identical across every sample; see data.rs
    let feature_cols = features::feature_names(sensor_names);

    let mut writer = csv::Writer::from_path("data/features/windows.csv")?;
    let mut header = vec![
        "accident_type".to_string(),
        "severity_id".to_string(),
        "window_start_seconds".to_string(),
        "window_end_seconds".to_string(),
    ];
    header.extend(feature_cols);
    writer.write_record(&header)?;

    for w in &windows {
        let mut record = vec![
            w.accident_type.clone(),
            w.severity_id.clone(),
            w.window_start_seconds.to_string(),
            w.window_end_seconds.to_string(),
        ];
        record.extend(w.features.iter().map(|f| f.to_string()));
        writer.write_record(&record)?;
    }
    writer.flush()?;
    println!("\nWrote data/features/windows.csv ({} rows)", windows.len());

    // --- Write data/results/feature_summary.json (small, checked in) ----
    #[derive(serde::Serialize)]
    struct FeatureSummary {
        window_size_rows: usize,
        stride_rows: usize,
        total_windows: usize,
        windows_per_accident_type: Vec<(String, usize)>,
        accident_types_with_zero_windows: Vec<String>,
    }

    let mut windows_per_accident_type: Vec<(String, usize)> = summary
        .accident_types
        .iter()
        .map(|t| {
            (
                t.accident_type.clone(),
                windows_per_type.get(t.accident_type.as_str()).copied().unwrap_or(0),
            )
        })
        .collect();
    windows_per_accident_type.sort();

    let feature_summary = FeatureSummary {
        window_size_rows: window_size,
        stride_rows: stride,
        total_windows: windows.len(),
        windows_per_accident_type,
        accident_types_with_zero_windows: zero_window_types,
    };

    std::fs::create_dir_all("data/results")?;
    std::fs::write(
        "data/results/feature_summary.json",
        serde_json::to_string_pretty(&feature_summary)?,
    )?;
    println!("Wrote data/results/feature_summary.json");

    Ok(())
}
