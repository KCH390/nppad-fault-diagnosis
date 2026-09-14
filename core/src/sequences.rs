//! Fixed-length sequence construction, for sequence models (Phase 6's
//! LSTM) as an alternative to Phase 2's rolling-window feature vectors.
//!
//! WHAT THIS FILE DOES, IN PLAIN LANGUAGE:
//! Phases 2-4 collapsed each run into many short, overlapping windows of
//! hand-computed statistics (mean/std/slope). An LSTM works differently:
//! it wants each run as a single, whole SEQUENCE of raw per-timestep
//! sensor readings, and it learns its own notion of "what matters over
//! time" internally rather than being handed pre-computed summary
//! statistics. That means this file solves a different problem than
//! `features.rs` does, even though both start from the same `Sample` data.
//!
//! TWO REAL PROBLEMS THIS FILE HAS TO SOLVE THAT WINDOWING DIDN'T:
//! 1. **Fixed length.** An LSTM (at least the straightforward way this
//!    project uses it, one label per whole sequence) wants every training
//!    example to be the same length so they can be batched into one
//!    tensor. Runs range from 11 to over 700 rows (see the Phase 1 README
//!    notes), so every run needs to be truncated or padded to one shared
//!    `seq_len`.
//! 2. **Feature scale.** Raw sensor units vary enormously (a percentage
//!    like `PWR`, a temperature in the hundreds, a flow rate, a pressure)
//!    living side by side in the same input vector. Neural network
//!    training is notoriously sensitive to this. Unlike the tree-based
//!    models in Phases 3-4 (trees split on ONE feature at a time, so
//!    relative scale across features doesn't matter to them), an LSTM's
//!    gates mix all input features together in the same linear layer, so
//!    features on wildly different scales can dominate purely due to
//!    their units, not their actual signal. This file z-score normalizes
//!    every sensor, matching standard practice for neural sequence models.

use crate::data::Sample;

/// One run, converted to LSTM input shape: `sequence[timestep][sensor]`,
/// always exactly `seq_len` timesteps long regardless of the run's actual
/// duration.
#[derive(Debug, Clone)]
pub struct SequenceExample {
    pub accident_type: String,
    pub severity_id: String,
    pub sequence: Vec<Vec<f64>>,
}

/// Per-sensor mean and standard deviation, used to z-score normalize every
/// sequence. Computed ONCE from the training set only (see
/// `fit_normalization` below) and then applied identically to both train
/// and test sequences, the same "fit on training data only" leakage
/// discipline used elsewhere in this project (e.g. imputation in your
/// other projects, or this project's run-level train/test split itself).
/// Fitting normalization stats on the test set too, even innocently,
/// would leak information about the test distribution into training.
#[derive(Debug, Clone)]
pub struct NormalizationStats {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
}

impl NormalizationStats {
    /// Compute per-sensor mean/std across every timestep of every sequence
    /// in `examples`. Call this ONLY on the training split.
    pub fn fit(examples: &[SequenceExample]) -> Self {
        let n_sensors = examples[0].sequence[0].len();
        let mut sums = vec![0.0; n_sensors];
        let mut count = 0.0;

        for example in examples {
            for row in &example.sequence {
                for (i, &v) in row.iter().enumerate() {
                    sums[i] += v;
                }
                count += 1.0;
            }
        }
        let mean: Vec<f64> = sums.iter().map(|&s| s / count).collect();

        let mut sq_diff_sums = vec![0.0; n_sensors];
        for example in examples {
            for row in &example.sequence {
                for (i, &v) in row.iter().enumerate() {
                    sq_diff_sums[i] += (v - mean[i]).powi(2);
                }
            }
        }
        let std: Vec<f64> = sq_diff_sums
            .iter()
            .map(|&s| {
                let variance = s / count;
                // A small epsilon floor avoids dividing by exactly zero in
                // `apply` below for any sensor that happens to be
                // perfectly constant across the whole training set (e.g.
                // a sensor that never moves during "Normal" runs) — a
                // real edge case worth guarding against explicitly rather
                // than producing NaN/inf silently.
                variance.sqrt().max(1e-8)
            })
            .collect();

        NormalizationStats { mean, std }
    }

    /// Apply z-score normalization ( (x - mean) / std ) to every value in
    /// `examples`, in place.
    pub fn apply(&self, examples: &mut [SequenceExample]) {
        for example in examples.iter_mut() {
            for row in example.sequence.iter_mut() {
                for (i, v) in row.iter_mut().enumerate() {
                    *v = (*v - self.mean[i]) / self.std[i];
                }
            }
        }
    }
}

/// Build one fixed-length `SequenceExample` from a `Sample`.
///
/// - If the run has MORE than `seq_len` rows, keep only the first
///   `seq_len` (truncate). This is a real information loss for the
///   longest runs, but keeping the accident's ONSET and early evolution
///   (rather than, say, the middle or a random slice) is what actually
///   matters for diagnosis — most of the signal distinguishing accident
///   types shows up early, not in a long, often near-steady-state tail
///   (see the Phase 1 README's notes on run-duration variability).
/// - If the run has FEWER than `seq_len` rows, pad by repeating the LAST
///   observed row until the sequence reaches `seq_len`. This is
///   "hold-last-value" padding: since these are continuous physical
///   sensor readings, repeating the final state is a far more physically
///   reasonable choice than padding with zeros, which would look to the
///   model like a sudden, nonsensical jump to "every sensor reads exactly
///   0" rather than "the plant settled and nothing more happened."
pub fn build_sequence(sample: &Sample, seq_len: usize) -> SequenceExample {
    let mut sequence = Vec::with_capacity(seq_len);

    for i in 0..seq_len {
        if i < sample.values.len() {
            sequence.push(sample.values[i].clone());
        } else {
            // `sample.values.len() - 1` is safe here because every Sample
            // has at least one row by construction in `data.rs` (a CSV
            // with only a header and no data rows would already have
            // failed to load earlier in the pipeline).
            let last_row = sample.values[sample.values.len() - 1].clone();
            sequence.push(last_row);
        }
    }

    SequenceExample {
        accident_type: sample.accident_type.clone(),
        severity_id: sample.severity_id.clone(),
        sequence,
    }
}

/// Build one `SequenceExample` per sample in `samples`.
pub fn build_sequences(samples: &[Sample], seq_len: usize) -> Vec<SequenceExample> {
    samples.iter().map(|s| build_sequence(s, seq_len)).collect()
}
