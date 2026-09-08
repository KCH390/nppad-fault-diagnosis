//! Loads the NPPAD (Nuclear Power Plant Accident Data) CSVs into memory.
//!
//! WHAT THIS FILE DOES, IN PLAIN LANGUAGE:
//! The raw dataset (from https://github.com/thu-inet/NuclearPowerPlantAccidentData)
//! is a folder tree: one subfolder per accident type (LOCA, SGATR, Normal,
//! ...), and inside each, one CSV per "severity" instance of that accident
//! (e.g. `Operation_csv_data/LOCA/37.csv` is a 37%-break-size loss-of-coolant
//! run). Almost every CSV has the same 97 columns: `TIME` plus 96 named
//! sensor readings (temperatures, pressures, flows, radiation monitors,
//! etc.), sampled every 10 simulated seconds. Runs are NOT all the same
//! length — that's a real, documented quirk of this dataset (see the Phase 1
//! EDA notes in the README), not a bug in this loader.
//!
//! "ALMOST" every CSV, because of a second real quirk found while building
//! Phase 2: 25 of the 101 `SLBIC` files have **100** columns instead of 97
//! — three extra sensors (`WPCS`, `WPMU`, `WPFW`) appended after the
//! standard set. Every other accident type, and the other 76 SLBIC files,
//! use the standard 97. Rather than let that surface as a downstream
//! "mismatched row length" error in feature extraction (which is exactly
//! how it WAS first found), this loader determines the dataset's most
//! common ("canonical") column schema up front and normalizes every file
//! to it — dropping the rare extra columns so every `Sample` has a
//! consistent, comparable shape. See `determine_canonical_sensor_schema`
//! below for how.
//!
//! This file turns that folder tree into a `Vec<Sample>` — one `Sample` per
//! CSV file — that the rest of the project (EDA, feature engineering,
//! models, charts) can all share.

use anyhow::{Context, Result};
use std::collections::HashMap;
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
    /// The canonical 96 sensor column names, in the dataset's dominant
    /// column order (e.g. "P", "TAVG", "THA", ...). EVERY `Sample` has
    /// exactly this same `sensor_names` list — even the 25 SLBIC files that
    /// natively have 3 extra columns get normalized down to it during
    /// loading, so nothing downstream ever has to special-case a variable
    /// number of sensors.
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

/// Scan every CSV under `root` and return the most common set of sensor
/// column names (i.e. every column except `TIME`) — the "canonical schema"
/// every `Sample` gets normalized to.
///
/// This only reads each file's HEADER row, not its data — `csv::Reader`
/// only actually reads as much of the file as you ask it to, and
/// `.headers()` asks for just the first line. Scanning 1,217 headers this
/// way is fast; it's a cheap up-front pass to protect the much larger main
/// loading pass from ever hitting a surprise mismatched schema again.
fn determine_canonical_sensor_schema(
    root: &Path,
    accident_types: &[String],
) -> Result<Vec<String>> {
    // `HashMap<Vec<String>, usize>` — keying by the ENTIRE header (as a
    // Vec of column names), counting how many files had exactly that
    // header. Rust lets you use a `Vec<String>` as a HashMap key as long as
    // it implements `Eq` and `Hash`, which it does automatically here
    // because `String` implements both and `Vec<T>` implements them
    // whenever `T` does. Two files with the SAME columns in the SAME order
    // hash and compare equal, even though they're different `Vec`
    // instances in memory — value equality, not identity, which is what we
    // want here.
    let mut header_counts: HashMap<Vec<String>, usize> = HashMap::new();

    for accident_type in accident_types {
        let dir = root.join(accident_type);
        let entries = std::fs::read_dir(&dir)
            .with_context(|| format!("reading accident directory {dir:?}"))?;

        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("csv") {
                let mut reader = csv::Reader::from_path(&path)?;
                let headers = reader.headers()?.clone();
                let sensor_names: Vec<String> =
                    headers.iter().skip(1).map(|s| s.to_string()).collect();
                // `.entry(...).or_insert(0)` is the standard Rust idiom for
                // "get this key's value, inserting a default first if it's
                // not there yet" — equivalent to Python's
                // `dict.setdefault(key, 0)` followed by an increment.
                *header_counts.entry(sensor_names).or_insert(0) += 1;
            }
        }
    }

    // `.into_iter().max_by_key(...)` consumes the HashMap and finds the
    // entry with the largest count — the most common header wins as our
    // canonical schema. `.map(...)` then discards the count, keeping just
    // the winning column-name list.
    header_counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(sensor_names, _)| sensor_names)
        .context("no CSV files found while determining canonical sensor schema")
}

/// Load every CSV under `root/<accident_type>/` into a `Vec<Sample>`,
/// normalizing every file's columns to `canonical_sensors`.
pub fn load_accident_type(
    root: impl AsRef<Path>,
    accident_type: &str,
    canonical_sensors: &[String],
) -> Result<Vec<Sample>> {
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

            let sample = load_sample_csv(&path, accident_type, &severity_id, canonical_sensors)
                .with_context(|| format!("loading sample {path:?}"))?;
            samples.push(sample);
        }
    }

    Ok(samples)
}

/// Load every accident type under `root` into one flat `Vec<Sample>`, all
/// normalized to the dataset's dominant sensor schema.
pub fn load_dataset(root: impl AsRef<Path>) -> Result<Vec<Sample>> {
    let root = root.as_ref();
    let accident_types = discover_accident_types(root)?;
    let canonical_sensors = determine_canonical_sensor_schema(root, &accident_types)?;
    let mut all_samples = Vec::new();

    for accident_type in &accident_types {
        let mut samples = load_accident_type(root, accident_type, &canonical_sensors)?;
        // `append` moves every element out of `samples` and into
        // `all_samples`, leaving `samples` empty — cheaper than `.clone()`
        // since we don't need the original `samples` Vec afterward.
        all_samples.append(&mut samples);
    }

    Ok(all_samples)
}

/// Parse a single NPPAD CSV file into a `Sample`, projecting its columns
/// onto `canonical_sensors` (dropping any extras, erroring if any expected
/// canonical column is unexpectedly missing).
fn load_sample_csv(
    path: &PathBuf,
    accident_type: &str,
    severity_id: &str,
    canonical_sensors: &[String],
) -> Result<Sample> {
    // `csv::Reader` handles the fiddly parts of CSV parsing (quoted fields,
    // different line endings, etc.) for us — this is the "crates for
    // plumbing" half of the dependency philosophy. `from_path` opens the
    // file and reads it, returning a `Result` we propagate with `?`.
    let mut reader = csv::Reader::from_path(path)?;

    // `.headers()?` gives us the first row ("TIME,P,TAVG,...") as a
    // `csv::StringRecord`. We clone it into an owned `Vec<String>` because
    // the `reader` object (and the borrow of its headers) won't outlive
    // this function.
    let headers = reader.headers()?.clone();
    let file_sensor_names: Vec<String> = headers.iter().skip(1).map(|s| s.to_string()).collect();

    // For every CANONICAL sensor name, find where it lives in THIS file's
    // header. Almost always this is a no-op identity mapping (the file's
    // columns already match canonical order exactly); for the 25 SLBIC
    // files with 3 extra trailing columns, this naturally skips those
    // extras, because we're only ever looking up canonical names — nothing
    // asks "what's in column 97?", so an unexpected column 98/99/100 just
    // never gets read.
    let mut column_indices = Vec::with_capacity(canonical_sensors.len());
    for canonical_name in canonical_sensors {
        let idx = file_sensor_names
            .iter()
            .position(|n| n == canonical_name)
            .with_context(|| {
                format!(
                    "file {path:?} is missing expected sensor column {canonical_name:?} \
                     (it has {} sensor columns; canonical schema has {})",
                    file_sensor_names.len(),
                    canonical_sensors.len()
                )
            })?;
        column_indices.push(idx);
    }

    if file_sensor_names.len() > canonical_sensors.len() {
        // A visible note, not a silent drop — this is exactly the kind of
        // thing that should be easy to notice while reading loader output,
        // even though it's expected and handled correctly.
        eprintln!(
            "note: {path:?} has {} extra sensor column(s) beyond the canonical {}; \
             extras are dropped so every Sample has a consistent schema (see README)",
            file_sensor_names.len() - canonical_sensors.len(),
            canonical_sensors.len()
        );
    }

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

        // Walk `column_indices` (one entry per CANONICAL sensor, holding
        // that sensor's position in THIS file) rather than every field in
        // the row — this is what actually performs the projection down to
        // the canonical schema. `record.get(idx + 1)` (the `+1` skips back
        // over the TIME column we already handled above) fetches exactly
        // the field we want, wherever it happens to sit in this
        // particular file's row.
        let row: Vec<f64> = column_indices
            .iter()
            .map(|&idx| {
                let field = record
                    .get(idx + 1)
                    .with_context(|| format!("row missing expected column at index {idx}"))?;
                field
                    .parse::<f64>()
                    .with_context(|| format!("parsing value {field:?} at TIME={time_val}"))
            })
            .collect::<Result<Vec<f64>>>()?;
        values.push(row);
    }

    Ok(Sample {
        accident_type: accident_type.to_string(),
        severity_id: severity_id.to_string(),
        time_seconds,
        sensor_names: canonical_sensors.to_vec(),
        values,
    })
}

