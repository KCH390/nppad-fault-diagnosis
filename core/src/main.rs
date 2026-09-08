//! Phase 1 CLI entry point.
//!
//! WHAT THIS FILE DOES, IN PLAIN LANGUAGE:
//! Runs end-to-end: load every CSV under `data/raw/Operation_csv_data/`,
//! compute the Phase 1 EDA summary (class balance + run-duration stats),
//! print a human-readable table to the terminal, and write the same data as
//! JSON to `data/results/eda_summary.json` so it's a checked-in, versioned
//! artifact rather than something that only ever exists in your terminal
//! history.
//!
//! HOW TO RUN THIS:
//! From the workspace root: `cargo run`
//! (Because `core` is the workspace's only `default-members` entry and its
//! `default-run` points at this binary, plain `cargo run` — no `-p` flag
//! needed — builds and runs exactly this file.)

// `core::data` / `core::eda` here refers to THIS crate's own library target
// (see lib.rs) — `main.rs` and `lib.rs` are two halves of the same package,
// so main.rs can use the library part by its crate name, "core".
use core::data;
use core::eda;

// `fn main() -> anyhow::Result<()>` lets `main` itself use `?` to bail out
// on error. If `main` returns `Err(e)`, Rust prints `e`'s Debug output and
// exits with a nonzero status code — this is the standard "let errors
// propagate all the way to the top and let the runtime report them"
// pattern for small CLI tools, instead of writing a top-level try/except.
fn main() -> anyhow::Result<()> {
    let data_root = "data/raw/Operation_csv_data";

    println!("Loading NPPAD dataset from {data_root}...");
    let samples = data::load_dataset(data_root)?;
    println!("Loaded {} samples.\n", samples.len());

    let summary = eda::summarize(&samples);

    // Human-readable table, widest field first so columns roughly line up
    // for the accident-type names we actually have (max 6 characters).
    println!(
        "{:<8} {:>7} {:>12} {:>12} {:>12}",
        "Type", "Count", "MinDur(s)", "MeanDur(s)", "MaxDur(s)"
    );
    for t in &summary.accident_types {
        println!(
            "{:<8} {:>7} {:>12.0} {:>12.0} {:>12.0}",
            t.accident_type,
            t.sample_count,
            t.min_duration_seconds,
            t.mean_duration_seconds,
            t.max_duration_seconds
        );
    }
    println!("\nTotal samples: {}", summary.total_samples);

    // Make sure the output directory exists before we write into it.
    // `create_dir_all` is the Rust equivalent of Python's
    // `os.makedirs(path, exist_ok=True)` — it creates every missing
    // component of the path and does NOT error if the directory already
    // exists (unlike plain `create_dir`, which would).
    std::fs::create_dir_all("data/results")?;

    let json = serde_json::to_string_pretty(&summary)?;
    std::fs::write("data/results/eda_summary.json", json)?;
    println!("\nWrote data/results/eda_summary.json");

    Ok(())
}
