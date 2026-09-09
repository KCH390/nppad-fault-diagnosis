//! Run-level train/test splitting.
//!
//! WHAT THIS FILE DOES, IN PLAIN LANGUAGE:
//! Phase 2's windows overlap: with window_size=5 rows and stride=3 rows,
//! consecutive windows from the same run share 2 of their 5 rows. If we
//! split at the WINDOW level — putting some windows from a run in train and
//! others from the very same run in test — the classifier would effectively
//! get to see near-duplicates of its test data during training. That
//! inflates every metric and hides how well the model actually generalizes
//! to a run it's never seen. The fix is to split at the RUN level instead:
//! every window from a given run goes entirely to train, or entirely to
//! test, never both.
//!
//! THE COMPLICATION: 6 of the 18 accident types (ATWS, LACP, LOF, Normal,
//! SP, TT) have exactly ONE run each (see the Phase 1 README notes). You
//! cannot hold out a test run for a class that only has one run without
//! leaving zero training data for that class. Rather than fudge this
//! (e.g. quietly falling back to a window-level split just for those
//! classes, reintroducing the leakage this whole file exists to avoid),
//! this implementation makes an explicit, documented choice: single-run
//! accident types are used for training only, and excluded from held-out
//! test evaluation. `models` reports macro F1 and the confusion matrix
//! only over the accident types that actually have a genuine held-out
//! test run. That's a real limitation of this dataset, not a limitation
//! of this code — better to say so plainly than to report a number that
//! looks precise but isn't measuring what it claims to.

use crate::data::Sample;
use std::collections::{HashMap, HashSet};

/// Which side of the split a given run landed on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Split {
    Train,
    Test,
}

/// The result of splitting a dataset's runs: which (accident_type,
/// severity_id) pairs go to train vs. test, and which accident types were
/// single-run and therefore excluded from test entirely.
pub struct RunSplit {
    assignments: HashMap<(String, String), Split>,
    pub single_run_types: Vec<String>,
}

impl RunSplit {
    /// Which split a given run belongs to, if it's part of one. Returns
    /// `None` for single-run accident types, which this module doesn't
    /// assign a Test half to at all (see the module-level docs above) —
    /// callers that want ALL windows from single-run types routed to
    /// training should check `single_run_types` themselves, which is
    /// exactly what `core::features` windows filtering does not need to
    /// worry about, since Phase 3's `models` crate handles that routing.
    pub fn assignment_for(&self, accident_type: &str, severity_id: &str) -> Option<Split> {
        self.assignments
            .get(&(accident_type.to_string(), severity_id.to_string()))
            .copied()
    }
}

/// Split every accident type with 2+ runs into train/test at the RUN
/// level, targeting roughly `test_fraction` of runs per type going to
/// test. Accident types with exactly 1 run are recorded in
/// `single_run_types` and get no Test assignment at all.
///
/// The split is deterministic (no RNG): runs within an accident type are
/// sorted by severity id, then every `round(1 / test_fraction)`-th one is
/// assigned to test. Determinism matters here for the same reason seed
/// fixing matters elsewhere in your projects — a report that says "12.3%
/// macro F1" should mean the same thing if someone re-runs this six months
/// from now.
pub fn split_by_run(samples: &[Sample], test_fraction: f64) -> RunSplit {
    // Group samples by accident type first — `HashMap<&str, Vec<&Sample>>`
    // rather than `Vec<Sample>` clones, since we only need to look at
    // `accident_type`/`severity_id` here, not the (potentially large)
    // sensor matrices each Sample carries.
    let mut by_type: HashMap<&str, Vec<&Sample>> = HashMap::new();
    for sample in samples {
        by_type.entry(sample.accident_type.as_str()).or_default().push(sample);
    }

    let mut assignments = HashMap::new();
    let mut single_run_types = Vec::new();

    // Sort accident type names for deterministic iteration order — a
    // HashMap's own iteration order isn't guaranteed, and even though it
    // doesn't affect the SPLIT RESULT here, keeping iteration order
    // reproducible is a cheap habit that avoids subtle "why did the output
    // order change between runs" confusion later.
    let mut type_names: Vec<&str> = by_type.keys().copied().collect();
    type_names.sort();

    for type_name in type_names {
        let mut runs = by_type[type_name].clone();
        if runs.len() < 2 {
            single_run_types.push(type_name.to_string());
            continue;
        }

        // Sort runs by severity id numerically where possible (NPPAD's
        // severity ids are percentages/counts like "1", "37", or the
        // negative "-1" seen in the RI accident type), falling back to
        // plain string ordering for anything that doesn't parse — this
        // keeps the split deterministic either way.
        runs.sort_by(|a, b| {
            let a_num: Option<f64> = a.severity_id.parse().ok();
            let b_num: Option<f64> = b.severity_id.parse().ok();
            match (a_num, b_num) {
                (Some(x), Some(y)) => x.partial_cmp(&y).unwrap(),
                _ => a.severity_id.cmp(&b.severity_id),
            }
        });

        // Every Nth run (by this sorted order) becomes a test run. Using
        // integer division here means `step` rounds DOWN — e.g.
        // test_fraction=0.2 gives step=5, so every 5th run (20%) is held
        // out, which is the intended behavior.
        let step = (1.0 / test_fraction).round().max(2.0) as usize;

        for (idx, run) in runs.iter().enumerate() {
            let split = if (idx + 1) % step == 0 { Split::Test } else { Split::Train };
            assignments.insert((run.accident_type.clone(), run.severity_id.clone()), split);
        }

        // Guard against a degenerate case: a small accident type (say, 3
        // runs) where the step size above happens to assign EVERY run to
        // Train and none to Test. If that happened, force the last run
        // (in sorted order) to Test so every multi-run type gets at least
        // one held-out run.
        let has_test = runs
            .iter()
            .any(|r| assignments.get(&(r.accident_type.clone(), r.severity_id.clone())) == Some(&Split::Test));
        if !has_test {
            if let Some(last) = runs.last() {
                assignments.insert((last.accident_type.clone(), last.severity_id.clone()), Split::Test);
            }
        }
    }

    RunSplit { assignments, single_run_types }
}

/// Convenience check used by `models` when deciding which windows to
/// evaluate: `true` if `accident_type` is one of the single-run types this
/// split had to exclude from test.
pub fn is_single_run_type(run_split: &RunSplit, accident_type: &str) -> bool {
    // `HashSet` isn't used for `single_run_types` on `RunSplit` itself
    // since the list is tiny (at most a handful of accident types) and
    // built once — a linear `.contains` scan over a `Vec` this small costs
    // nothing measurable, and avoids carrying an extra `HashSet` field
    // just for a handful of lookups.
    run_split.single_run_types.iter().any(|t| t == accident_type)
}

/// Deterministic, evenly-spaced subsampling of a labeled collection down to
/// at most `max_per_class` items per label. Used to keep Phase 3's training
/// set tractable given how lopsided window counts are across accident
/// types (see the Phase 2 README notes — some types have 100x more windows
/// than others purely from run length, not real class frequency).
///
/// `label_of: impl Fn(&T) -> &str` is a "closure parameter": instead of
/// requiring every `T` this function is used with to implement some
/// `HasLabel` trait, the caller just hands over a function that knows how
/// to pull a label out of whatever `T` actually is. This is a common Rust
/// pattern for keeping a generic helper decoupled from the specific types
/// it's used with.
pub fn subsample_by_class<T>(
    items: Vec<T>,
    label_of: impl Fn(&T) -> &str,
    max_per_class: usize,
) -> Vec<T> {
    let mut by_label: HashMap<String, Vec<T>> = HashMap::new();
    for item in items {
        by_label.entry(label_of(&item).to_string()).or_default().push(item);
    }

    let mut result = Vec::new();
    for (_, mut group) in by_label {
        if group.len() <= max_per_class {
            result.append(&mut group);
            continue;
        }

        // Evenly-spaced index selection (rather than "just take the first
        // max_per_class") so the kept windows still span the whole run's
        // duration instead of clustering at the start.
        let n = group.len();
        let mut kept = Vec::with_capacity(max_per_class);
        for i in 0..max_per_class {
            let idx = i * (n - 1) / (max_per_class - 1).max(1);
            kept.push(idx);
        }
        kept.dedup(); // guards against rounding producing the same index twice

        // Walk the group once, pulling out only the indices we decided to
        // keep. We build a `HashSet` of the (usually few thousand) kept
        // indices so this membership check is O(1) instead of O(n) per
        // item, which matters here since `group` can be tens of thousands
        // of windows long.
        let kept_set: HashSet<usize> = kept.into_iter().collect();
        for (idx, item) in group.into_iter().enumerate() {
            if kept_set.contains(&idx) {
                result.push(item);
            }
        }
    }

    result
}
