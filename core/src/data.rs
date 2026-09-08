//! Loads the NPPAD (Nuclear Power Plant Accident Data) CSVs into memory.
//!
//! WHAT THIS FILE DOES, IN PLAIN LANGUAGE:
//! The raw dataset (from https://github.com/thu-inet/NuclearPowerPlantAccidentData)
//! is a folder tree: one subfolder per accident type (LOCA, SGATR, Normal,
//! ...), and inside each, one CSV per "severity" instance of that accident
//! (e.g. `Operation_csv_data/LOCA/37.csv` is a 37%-break-size loss-of-coolant
//! run). Every CSV has the same 97 columns: `TIME` plus 96 named sensor
//! readings (temperatures, pressures, flows, radiation monitors, etc.),
//! sampled every 10 simulated seconds. Runs are NOT all the same length —
//! that's a real, documented quirk of this dataset (see the Phase 1 EDA
//! notes in the README), not a bug in this loader.
//!
//! This file turns that folder tree into a `Vec<Sample>` — one `Sample` per
//! CSV file — that the rest of the project (EDA, feature engineering,
//! models, charts) can all share.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// One simulated accident run: everything from a single CSV file.
///
/// A quick Rust note on why this is a `struct` with owned fields (`String`,
/// `Vec<f64>`) rather than borrowed references (`&str`, `&[f64]`): a
/// `Sample` needs to outlive the short-lived act of reading the file, and
/// get stored in a `Vec<Sample>` that lives for the whole program. Borrowed
/// references would need a "lifetime" tying them back to the original file
/// buffer, which would infect every function signature that touches
/// `Sample` with lifetime annotations. Owning the data avoids that
/// complexity at the (small, for this dataset size) cost of a few extra
/// heap allocations. This is a very common beginner fork in the road in
/// Rust: "should I borrow or own?" — when in doubt, own, and only reach for
/// borrowing once profiling says allocation is actually a bottleneck.
#[derive(Debug, Clone)]
pub struct Sample {
    /// Folder name this came from, e.g. "LOCA", "Normal", "SGATR".
    pub accident_type: String,
    /// The CSV's filename (without `.csv`), e.g. "1", "37". For
    /// "Severity"-class accidents (see Table 1 in the NPPAD README) this
    /// number is a percentage break size or similar; for "Other"-class
    /// accidents (Normal, ATWS, LACP, LOF, SP, TT) there's only ever one
    /// file, conventionally "1".
    pub severity_id: String,
    /// The `TIME` column, in seconds, one entry per row.
    pub time_seconds: Vec<f64>,
    /// The 96 sensor column names, in the order they appear in the CSV
    /// header (e.g. "P", "TAVG", "THA", ...). Every `Sample` has the same
    /// `sensor_names`, since every NPPAD CSV shares the same schema — we
    /// still store it per-sample rather than as a global constant so that
    /// downstream code doesn't have to reach into a different module just
    /// to know which column is which.
    pub sensor_names: Vec<String>,
    /// The sensor readings themselves, as a row-major matrix:
    /// `values[row_index][sensor_index]`. `values.len() ==
    /// time_seconds.len()`, and each inner `Vec<f64>` has length
    /// `sensor_names.len()`.
    pub values: Vec<Vec<f64>>,
}

impl Sample {
    /// Number of timesteps (rows) in this run.
    ///
    /// `&self` here borrows the sample immutably — we're only reading
    /// `self.time_seconds`, not changing it, so we don't need `&mut self`.
    /// Rust's borrow checker enforces this at compile time: if you tried to
    /// mutate `self` inside a method that only takes `&self`, it simply
    /// wouldn't compile. That's a guarantee Python has no equivalent of.
    pub fn n_rows(&self) -> usize {
        self.time_seconds.len()
    }

    /// Total simulated duration of this run, in seconds.
    ///
    /// `Option<f64>` (rather than plain `f64`) is Rust's way of encoding
    /// "this value might not exist" directly in the type system — there's
    /// no `null`/`None`-by-default in Rust. Here, a run with zero rows has
    /// no last timestamp, so we return `None` instead of an arbitrary
    /// sentinel like `-1.0` or `0.0` that callers might mistake for a real
    /// value.
    pub fn duration_seconds(&self) -> Option<f64> {
        // `.last()` on a slice returns `Option<&f64>`; `.copied()` turns
        // that `Option<&f64>` into `Option<f64>` since f64 is cheap to copy
        // (it implements the `Copy` trait). This little chain is idiomatic
        // Rust for "give me the last element as an owned value, or None
        // if the collection is empty."
        self.time_seconds.last().copied()
    }
}

/// Look at the dataset root directory and return the accident type names
/// (subfolder names), sorted alphabetically for reproducible output.
///
/// `impl AsRef<Path>` is a common Rust pattern for "accept anything that can
/// be viewed as a filesystem path" — a `&str`, a `String`, a `PathBuf`, or a
/// `&Path` will all work as an argument here, because they all implement
/// the `AsRef<Path>` trait. It's Rust's version of Python's duck typing,
/// except the compiler checks it ahead of time instead of failing at
/// runtime.
pub fn discover_accident_types(root: impl AsRef<Path>) -> Result<Vec<String>> {
    let root = root.as_ref();
    let mut names = Vec::new();

    // `std::fs::read_dir` returns an iterator of `Result<DirEntry, io::Error>`.
    // The `?` after `.with_context(...)` propagates an error out of this
    // function immediately if the directory can't be read (e.g. wrong
    // path) — equivalent in spirit to letting a Python exception bubble up,
    // but explicit in the function signature (`-> Result<...>`) rather than
    // implicit.
    let entries = std::fs::read_dir(root)
        .with_context(|| format!("reading dataset root directory {root:?}"))?;

    for entry in entries {
        let entry = entry?; // Each iteration can itself fail (e.g. permissions).
        let path = entry.path();
        if path.is_dir() {
            // `file_name()` returns `Option<&OsStr>` (paths aren't
            // guaranteed to be valid UTF-8 on every OS). `.to_string_lossy()`
            // converts it to a `String`, substituting the Unicode
            // replacement character for anything that isn't valid UTF-8.
            // For our purposes (ASCII folder names like "LOCA") this is
            // always a clean, lossless conversion — we use `_lossy` instead
            // of a stricter method because it never panics, which matters
            // more than the (nonexistent, here) precision we'd be giving up.
            if let Some(name) = path.file_name() {
                names.push(name.to_string_lossy().into_owned());
            }
        }
    }

    names.sort();
    Ok(names)
}

/// Load every CSV under `root/<accident_type>/` into a `Vec<Sample>`.
pub fn load_accident_type(root: impl AsRef<Path>, accident_type: &str) -> Result<Vec<Sample>> {
    let dir = root.as_ref().join(accident_type);
    let mut samples = Vec::new();

    let entries = std::fs::read_dir(&dir)
        .with_context(|| format!("reading accident directory {dir:?}"))?;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();

        // Only look at `.csv` files — the dataset's Operation_csv_data
        // folders contain nothing else, but being explicit here means this
        // loader won't silently break if that ever changes upstream.
        if path.extension().and_then(|e| e.to_str()) == Some("csv") {
            // `file_stem()` gives the filename without its extension, e.g.
            // "37" from "37.csv". We use this as the severity id.
            let severity_id = path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "unknown".to_string());

            let sample = load_sample_csv(&path, accident_type, &severity_id)
                .with_context(|| format!("loading sample {path:?}"))?;
            samples.push(sample);
        }
    }

    Ok(samples)
}

/// Load every accident type under `root` into one flat `Vec<Sample>`.
pub fn load_dataset(root: impl AsRef<Path>) -> Result<Vec<Sample>> {
    let root = root.as_ref();
    let accident_types = discover_accident_types(root)?;
    let mut all_samples = Vec::new();

    for accident_type in &accident_types {
        let mut samples = load_accident_type(root, accident_type)?;
        // `append` moves every element out of `samples` and into
        // `all_samples`, leaving `samples` empty — cheaper than `.clone()`
        // since we don't need the original `samples` Vec afterward.
        all_samples.append(&mut samples);
    }

    Ok(all_samples)
}

/// Parse a single NPPAD CSV file into a `Sample`.
fn load_sample_csv(path: &PathBuf, accident_type: &str, severity_id: &str) -> Result<Sample> {
    // `csv::Reader` handles the fiddly parts of CSV parsing (quoted fields,
    // different line endings, etc.) for us — this is the "crates for
    // plumbing" half of the dependency philosophy. `from_path` opens the
    // file and reads it, returning a `Result` we propagate with `?`.
    let mut reader = csv::Reader::from_path(path)?;

    // `.headers()?` gives us the first row ("TIME,P,TAVG,...") as a
    // `csv::StringRecord`. We clone it into an owned `Vec<String>` because
    // the `reader` object (and the borrow of its headers) won't outlive
    // this function, but we need `sensor_names` to live on inside the
    // `Sample` we return.
    let headers = reader.headers()?.clone();
    let sensor_names: Vec<String> = headers.iter().skip(1).map(|s| s.to_string()).collect();
    // `.skip(1)` drops the "TIME" column from `sensor_names`, since we track
    // time separately in `Sample::time_seconds` rather than mixing it into
    // the numeric sensor matrix.

    let mut time_seconds = Vec::new();
    let mut values = Vec::new();

    // `reader.records()` iterates over the remaining rows. Each `record` is
    // a `Result<StringRecord, csv::Error>` — a row could fail to parse
    // (e.g. malformed data), so we still need the `?` inside the loop.
    for record in reader.records() {
        let record = record?;

        // `record.get(0)` returns `Option<&str>` — `Some(...)` if that
        // column index exists, `None` if the row is unexpectedly short.
        // `.context(...)` turns a `None` into a proper `anyhow` error with
        // a helpful message, so a malformed row fails loudly with context
        // instead of quietly panicking or producing a garbage 0.0.
        let time_str = record.get(0).context("row missing TIME column")?;
        let time_val: f64 = time_str
            .parse()
            .with_context(|| format!("parsing TIME value {time_str:?}"))?;
        time_seconds.push(time_val);

        // Parse every remaining column (index 1..) into f64. This is a
        // fairly dense iterator chain, so here's what each link does:
        //   record.iter()        -> iterate over every field in the row as &str
        //   .skip(1)             -> drop the TIME field, already handled above
        //   .map(|s| s.parse())  -> attempt String -> f64 conversion per field
        //   .collect::<Result<Vec<f64>, _>>() -> gather results into one Vec,
        //                          but if ANY field failed to parse, this whole
        //                          expression short-circuits to that single Err.
        // That last step is a very idiomatic Rust trick: collecting an
        // iterator of `Result`s into a `Result` of a collection, so a
        // single bad field cleanly aborts the whole row instead of us
        // needing to check each one by hand.
        let row: Vec<f64> = record
            .iter()
            .skip(1)
            .map(|s| s.parse::<f64>())
            .collect::<std::result::Result<Vec<f64>, _>>()
            .with_context(|| format!("parsing sensor row at TIME={time_val}"))?;
        values.push(row);
    }

    Ok(Sample {
        accident_type: accident_type.to_string(),
        severity_id: severity_id.to_string(),
        time_seconds,
        sensor_names,
        values,
    })
}
