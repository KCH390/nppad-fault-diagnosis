//! Rolling-window feature engineering over loaded `Sample`s.
//!
//! WHAT THIS FILE DOES, IN PLAIN LANGUAGE:
//! A raw `Sample` is a full accident run — hundreds of timesteps, one row
//! per 10 simulated seconds. That's not directly usable as a training
//! example for a classifier: we need fixed-size feature vectors, not
//! variable-length runs. The standard move (used in the C-MAPSS project
//! too) is a **sliding window**: slide a short window of consecutive rows
//! along the run, and for each window, compute a handful of summary
//! statistics per sensor — here, mean, standard deviation, and linear
//! trend (slope) — collapsing that window's rows into one fixed-length
//! feature vector. Each window becomes one training example, inheriting
//! its parent run's accident-type label.
//!
//! "RUN-AWARE" MEANS: windows never slide across the boundary between two
//! different `Sample`s. Since every `Sample` here already represents one
//! complete, independent accident run, this falls out naturally — we just
//! process each `Sample` separately and never concatenate their time
//! series together. (In C-MAPSS this took more care, since multiple
//! engines' data lived in one combined table; here the dataset's own
//! folder-per-run structure does the separating for us.)

use crate::data::Sample;

/// One sliding window's worth of features, plus the metadata needed to
/// trace it back to its parent run and label a training example with it.
#[derive(Debug, Clone)]
pub struct WindowFeatures {
    pub accident_type: String,
    pub severity_id: String,
    pub window_start_seconds: f64,
    pub window_end_seconds: f64,
    /// Flattened features, in the same order as `feature_names()` would
    /// produce for this sample's `sensor_names`: for each sensor in turn,
    /// its `[mean, std, slope]` triple. Keeping this as a plain `Vec<f64>`
    /// (rather than, say, a `HashMap<String, f64>`) keeps memory use and
    /// CSV-writing simple — a hundred thousand windows each holding a
    /// small HashMap would be far more wasteful than each holding one flat
    /// `Vec` and relying on a shared, separately-computed name list.
    pub features: Vec<f64>,
}

/// Column names matching the layout `WindowFeatures::features` uses, given
/// a sample's `sensor_names`. Kept as a free function (not stored on every
/// `WindowFeatures`) since it's identical for every window from every
/// sample in this dataset — there's no reason to repeat it thousands of
/// times in memory when computing it once and reusing it is free.
pub fn feature_names(sensor_names: &[String]) -> Vec<String> {
    let mut names = Vec::with_capacity(sensor_names.len() * 3);
    for sensor in sensor_names {
        names.push(format!("{sensor}_mean"));
        names.push(format!("{sensor}_std"));
        names.push(format!("{sensor}_slope"));
    }
    names
}

/// Slide a window of `window_size` rows across `sample`, moving `stride`
/// rows at a time, and compute one `WindowFeatures` per window position.
///
/// A window that would run past the end of the sample's rows is simply not
/// produced — we don't pad short runs with fabricated data. That means a
/// very short run (this dataset's shortest is 11 rows, ~100 seconds of RI
/// data) can legitimately produce very few windows, or with an
/// aggressively large `window_size`, none at all. That's a real property
/// of the data worth knowing about, not something to silently paper over.
pub fn extract_rolling_features(
    sample: &Sample,
    window_size: usize,
    stride: usize,
) -> Vec<WindowFeatures> {
    let n_rows = sample.n_rows();
    let n_sensors = sample.sensor_names.len();
    let mut windows = Vec::new();

    if window_size < 2 || n_rows < window_size {
        // Slope needs at least 2 points to be meaningful, and obviously a
        // window can't be bigger than the run it's drawn from. Returning
        // an empty Vec here (rather than panicking) lets the caller just
        // treat "this run was too short" as "zero windows from this run,"
        // which is the honest outcome, not an error condition.
        return windows;
    }

    // `(0..).step_by(stride)` is an infinite iterator of start indices
    // (0, stride, 2*stride, ...); `.take_while(...)` cuts it off as soon as
    // a window starting there would run past the end of the data. This is
    // a common Rust idiom for "generate candidate positions, then stop as
    // soon as one doesn't fit" without having to hand-compute how many
    // windows there will be up front.
    let start_indices = (0..).step_by(stride).take_while(|&start| start + window_size <= n_rows);

    for start in start_indices {
        let end = start + window_size; // exclusive
        let window_times = &sample.time_seconds[start..end];

        let mut features = Vec::with_capacity(n_sensors * 3);
        for sensor_idx in 0..n_sensors {
            // Pull out just this sensor's values for this window. Each row
            // in `sample.values` is one timestep across ALL sensors, so we
            // walk the window's rows and pick out one column —
            // `sample.values[row][sensor_idx]` — for each.
            let window_values: Vec<f64> = sample.values[start..end]
                .iter()
                .map(|row| row[sensor_idx])
                .collect();

            let (mean, std) = mean_and_std(&window_values);
            let slope = linear_slope(window_times, &window_values);

            features.push(mean);
            features.push(std);
            features.push(slope);
        }

        windows.push(WindowFeatures {
            accident_type: sample.accident_type.clone(),
            severity_id: sample.severity_id.clone(),
            window_start_seconds: window_times[0],
            window_end_seconds: window_times[window_times.len() - 1],
            features,
        });
    }

    windows
}

/// Run `extract_rolling_features` over every sample in `samples` and
/// collect the results into one flat `Vec<WindowFeatures>`.
///
/// Note the type of `samples` here: `&[Sample]`, a slice, not `&Vec<Sample>`
/// — same reasoning as `eda::summarize` (see eda.rs): a slice accepts
/// anything sequence-shaped, not just a `Vec` specifically.
pub fn extract_dataset_features(
    samples: &[Sample],
    window_size: usize,
    stride: usize,
) -> Vec<WindowFeatures> {
    // `flat_map` calls the closure once per sample (producing a
    // `Vec<WindowFeatures>` each time) and flattens all those Vecs into one
    // combined iterator, which `.collect()` then gathers into a single
    // `Vec<WindowFeatures>`. This avoids the manual "make an empty Vec,
    // loop, `.extend(...)` each sample's windows into it" pattern — flat_map
    // is that pattern, expressed as one expression instead of a loop body.
    samples
        .iter()
        .flat_map(|sample| extract_rolling_features(sample, window_size, stride))
        .collect()
}

/// Hand-rolled population mean and standard deviation over a slice of
/// values. Simple enough (and central enough to what this file is doing)
/// that it belongs here rather than as a dependency, per the "hand-roll
/// differentiating logic" half of the dependency philosophy.
fn mean_and_std(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;

    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    // `.powi(2)` squares using integer exponentiation (faster and more
    // precise than `.powf(2.0)` for a fixed small integer power like this).

    (mean, variance.sqrt())
}

/// Hand-rolled least-squares slope of `values` against `times` — the same
/// "line of best fit" you'd get from `numpy.polyfit(times, values, 1)[0]`
/// in Python, computed directly from the closed-form formula instead of
/// pulling in a whole linear algebra crate for one coefficient:
///
///   slope = Σ (tᵢ - t̄)(vᵢ - v̄) / Σ (tᵢ - t̄)²
///
/// where t̄ and v̄ are the means of `times` and `values`.
fn linear_slope(times: &[f64], values: &[f64]) -> f64 {
    let n = times.len() as f64;
    let t_mean = times.iter().sum::<f64>() / n;
    let v_mean = values.iter().sum::<f64>() / n;

    let mut numerator = 0.0;
    let mut denominator = 0.0;
    // `.zip(...)` pairs up `times` and `values` element-by-element — see
    // charts/src/main.rs for a longer explanation of `.zip`. Here it lets
    // us walk both slices in lockstep in a single loop rather than
    // indexing into both separately.
    for (&t, &v) in times.iter().zip(values.iter()) {
        numerator += (t - t_mean) * (v - v_mean);
        denominator += (t - t_mean).powi(2);
    }

    if denominator == 0.0 {
        // Every timestamp in the window is identical — degenerate, but
        // guard against it explicitly rather than dividing by zero and
        // producing a silent NaN that would poison every downstream
        // computation touching this feature.
        0.0
    } else {
        numerator / denominator
    }
}
