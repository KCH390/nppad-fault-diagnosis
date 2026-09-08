//! Exploratory data analysis over a loaded `Vec<Sample>`.
//!
//! WHAT THIS FILE DOES, IN PLAIN LANGUAGE:
//! Two questions drive Phase 1's EDA, because they're both real quirks of
//! this dataset that affect every later phase:
//!   1. Class balance — how many runs does each accident type have? (Spoiler:
//!      wildly uneven — "Other"-type accidents like Normal/ATWS/TT have
//!      exactly one file each, while "Severity"-type accidents like LOCA
//!      have ~100.)
//!   2. Run duration — how long (in simulated seconds) does each run last?
//!      Runs are NOT a fixed length, which matters a lot for feature
//!      engineering and any model that expects fixed-size input.
//! Both get written to a small JSON summary in `data/results/` so they're
//! checked into git as documented, reproducible findings — the same
//! "honest documentation" pattern as your other projects — rather than
//! being numbers that only ever existed in a terminal scrollback.

use crate::data::Sample;
use serde::Serialize;

/// How many runs exist for one accident type, plus duration statistics
/// across those runs.
///
/// `#[derive(Serialize)]` is a "derive macro": we're asking the compiler to
/// auto-generate the (fairly mechanical, tedious-to-write-by-hand) code
/// that converts this struct into JSON, via the `serde` crate. This is the
/// same "crates for plumbing, hand-rolled for differentiating logic" split
/// as `data.rs` — turning a struct into JSON text is plumbing; the
/// statistics computed below are the actual analysis.
#[derive(Debug, Serialize)]
pub struct AccidentTypeSummary {
    pub accident_type: String,
    pub sample_count: usize,
    pub min_duration_seconds: f64,
    pub mean_duration_seconds: f64,
    pub max_duration_seconds: f64,
}

/// The full Phase 1 EDA summary: one entry per accident type, plus the
/// overall totals used to talk about class imbalance.
#[derive(Debug, Serialize)]
pub struct EdaSummary {
    pub total_samples: usize,
    pub accident_types: Vec<AccidentTypeSummary>,
}

/// Compute the Phase 1 EDA summary from a loaded dataset.
///
/// `samples: &[Sample]` (a "slice") rather than `&Vec<Sample>` is the
/// idiomatic Rust choice for "give me read-only access to a sequence of
/// values" — a slice works whether the caller has a `Vec`, an array, or
/// part of a larger `Vec`, whereas `&Vec<Sample>` only works for an actual
/// `Vec`. You'll see this "prefer `&[T]` over `&Vec<T>` in function
/// signatures" pattern throughout idiomatic Rust code.
pub fn summarize(samples: &[Sample]) -> EdaSummary {
    // Group samples by accident type. We can't use a HashMap<String,
    // Vec<&Sample>> and then just iterate — HashMap iteration order isn't
    // guaranteed, and we want reproducible (sorted, deterministic) output
    // every time this runs, matching your reproducibility-first practice.
    // So: collect distinct accident type names first, sorted, then filter
    // the samples for each one in turn.
    let mut accident_type_names: Vec<&str> =
        samples.iter().map(|s| s.accident_type.as_str()).collect();
    accident_type_names.sort();
    accident_type_names.dedup(); // removes consecutive duplicates; sorting first makes this remove ALL duplicates.

    let mut accident_types = Vec::new();

    for name in accident_type_names {
        // `.filter(...)` keeps only samples matching this accident type;
        // `.collect()` gathers the matching `&Sample` references into a
        // `Vec<&Sample>`. We only need to look at them, not own them, so
        // references are enough here — no cloning of the (potentially
        // large) sensor matrices.
        let group: Vec<&Sample> = samples.iter().filter(|s| s.accident_type == name).collect();

        let durations: Vec<f64> = group
            .iter()
            .filter_map(|s| s.duration_seconds())
            // `filter_map` is `map` + `filter` fused: the closure returns
            // `Option<f64>`, and any `None` (an empty run with no last
            // timestamp) is silently dropped rather than crashing the
            // whole summary. This matters because we'd rather have an
            // honest summary that skips a malformed run than a panic that
            // kills the whole EDA pass over one bad file.
            .collect();

        let (min_duration, mean_duration, max_duration) = duration_stats(&durations);

        accident_types.push(AccidentTypeSummary {
            accident_type: name.to_string(),
            sample_count: group.len(),
            min_duration_seconds: min_duration,
            mean_duration_seconds: mean_duration,
            max_duration_seconds: max_duration,
        });
    }

    EdaSummary {
        total_samples: samples.len(),
        accident_types,
    }
}

/// Hand-rolled min/mean/max over a slice of durations. This is genuinely
/// simple enough that pulling in a stats crate for it would be adding a
/// dependency for three lines of arithmetic — exactly the kind of thing
/// your dependency philosophy says to hand-roll.
fn duration_stats(durations: &[f64]) -> (f64, f64, f64) {
    if durations.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    // `fold` walks the slice once, carrying an accumulator forward — here,
    // a running `(min, max)` pair — updating it per element. It's the Rust
    // equivalent of Python's `functools.reduce`, and it's how you write a
    // "manual" reduction without a library function for this exact shape
    // of problem (there's no built-in `Iterator::min_max` in stable Rust,
    // because `f64` doesn't implement `Ord` — NaN breaks total ordering, so
    // you have to make a judgment call about how to compare, which `fold`
    // lets us do explicitly below with `f64::min` / `f64::max`).
    let (min, max) = durations
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &d| {
            (min.min(d), max.max(d))
        });

    let mean = durations.iter().sum::<f64>() / durations.len() as f64;
    // `durations.len()` is a `usize`; f64 arithmetic needs an `f64`, so `as
    // f64` performs an explicit numeric cast. Rust never converts numeric
    // types implicitly (unlike Python, C, or JS) — you always have to say
    // `as` and mean it, which avoids a whole category of silent
    // precision-loss bugs.

    (min, mean, max)
}
