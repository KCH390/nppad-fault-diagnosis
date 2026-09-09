//! Charting functions for the NPPAD EDA. Kept in a separate crate from
//! `core` so that `plotters` (and everything it pulls in) never shows up in
//! `cargo tree -p core` — only people who actually want to regenerate
//! charts pay for that dependency, via `cargo run -p charts`.
//!
//! WHAT THIS FILE DOES, IN PLAIN LANGUAGE:
//! Three chart-drawing functions, each taking already-computed data (never
//! raw `Sample`s — that keeps this file focused purely on drawing, with all
//! the "what to plot" decisions made by the caller in `main.rs`):
//!   1. `bar_chart` — generic labeled-bars chart. Used for both the class
//!      balance chart (bars = accident types) and could be reused for
//!      anything else shaped like "a handful of labeled counts."
//!   2. `histogram` — buckets a list of numbers into fixed-width bins and
//!      draws them as a bar chart. Used for the run-duration distribution.
//!   3. `multi_line_chart` — overlays several (label, series) time-series
//!      lines on one set of axes with a legend. Used for the
//!      sensor-response-by-accident-type comparison.

use anyhow::Result;
use plotters::prelude::*;
use plotters::style::Palette;

/// Draw a bar chart of (label, value) pairs to an SVG file.
///
/// `labels_values: &[(String, f64)]` — a slice of tuples. Tuples in Rust are
/// fixed-size, heterogeneous groupings (here, always a `String` paired with
/// an `f64`); unlike a `struct`, the fields are accessed positionally
/// (`.0`, `.1`) rather than by name, which is fine for a short-lived
/// "just passing two related values together" case like this one.
pub fn bar_chart(
    title: &str,
    labels_values: &[(String, f64)],
    y_label: &str,
    output_path: &str,
) -> Result<()> {
    let root = SVGBackend::new(output_path, (900, 500)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_value = labels_values
        .iter()
        .map(|(_, v)| *v)
        // `fold` again, this time just tracking a running maximum — see
        // eda.rs's `duration_stats` for the longer explanation of why we
        // can't just call `.max()` on an iterator of `f64` directly (f64
        // doesn't implement `Ord` because of NaN).
        .fold(0.0_f64, f64::max)
        * 1.1; // 10% headroom above the tallest bar so it doesn't touch the top edge.

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(60)
        .y_label_area_size(60)
        // `0..labels_values.len()` gives each bar an integer "slot" on the
        // x-axis; we'll turn those integers back into the real labels
        // (accident type names) via `.x_label_formatter` below, since
        // plotters' Cartesian axes want a numeric range, not arbitrary
        // strings, as the underlying coordinate space.
        .build_cartesian_2d(0..labels_values.len(), 0.0..max_value)?;

    chart
        .configure_mesh()
        .disable_x_mesh() // gridlines behind categorical bars just add visual noise
        .x_label_formatter(&|idx| {
            labels_values
                .get(*idx)
                .map(|(label, _)| label.clone())
                .unwrap_or_default()
        })
        .y_desc(y_label)
        .draw()?;

    // `draw_series` takes an iterator of drawable elements. `.enumerate()`
    // pairs each `(label, value)` with its index (0, 1, 2, ...) so we know
    // which x-slot to draw each bar in. Each bar is a `Rectangle` spanning
    // from just left of its slot to just right of it, and from 0 up to its
    // value.
    chart.draw_series(labels_values.iter().enumerate().map(|(idx, (_, value))| {
        let color = Palette99::pick(idx).filled();
        Rectangle::new([(idx, 0.0), (idx + 1, *value)], color)
    }))?;

    root.present()?;
    Ok(())
}

/// Bucket `values` into fixed-width bins and draw the result as a bar
/// chart, i.e. a histogram.
pub fn histogram(
    title: &str,
    values: &[f64],
    bin_width: f64,
    x_label: &str,
    output_path: &str,
) -> Result<()> {
    if values.is_empty() {
        // An empty histogram isn't an error condition worth propagating —
        // there's just nothing to draw. Returning `Ok(())` here rather than
        // an `Err` reflects that this is a legitimate (if uninteresting)
        // input, not a bug.
        return Ok(());
    }

    let max_value = values.iter().cloned().fold(0.0_f64, f64::max);
    let n_bins = (max_value / bin_width).ceil() as usize + 1;

    // Build the bin counts by hand: start with a Vec of zeros, one per bin,
    // then walk every value and increment the bin it falls into. This is
    // the hand-rolled equivalent of e.g. numpy.histogram's binning step —
    // simple enough that pulling in a stats crate for it isn't warranted.
    let mut bin_counts = vec![0usize; n_bins];
    for &v in values {
        let bin_idx = (v / bin_width).floor() as usize;
        // `.min(n_bins - 1)` guards against the maximum value landing
        // exactly on a bin boundary and computing an index one past the
        // end of `bin_counts` due to floating-point rounding.
        bin_counts[bin_idx.min(n_bins - 1)] += 1;
    }

    let labels_values: Vec<(String, f64)> = bin_counts
        .iter()
        .enumerate()
        .map(|(i, &count)| {
            let bin_start = i as f64 * bin_width;
            (format!("{bin_start:.0}"), count as f64)
        })
        .collect();

    bar_chart(
        &format!("{title} (bin start, {x_label})"),
        &labels_values,
        "Count",
        output_path,
    )?;
    // Reusing `bar_chart` here (instead of writing separate drawing code)
    // is the same instinct as reusing `core`'s data-loading code from the
    // `charts` binary: don't duplicate logic that's already correct
    // elsewhere in the workspace. `bar_chart`'s x-axis labels are the bin
    // start values (e.g. "0", "500", "1000"...); folding `x_label` into the
    // title itself is a simple way to still say what unit those numbers
    // are in without changing `bar_chart`'s signature just for this caller.
    Ok(())
}

/// One named series to plot in `multi_line_chart`.
pub struct Series {
    pub label: String,
    /// (x, y) points, e.g. (time_seconds, sensor_value).
    pub points: Vec<(f64, f64)>,
}

/// Overlay several labeled time-series lines on one chart, with a legend.
pub fn multi_line_chart(
    title: &str,
    series: &[Series],
    x_label: &str,
    y_label: &str,
    output_path: &str,
) -> Result<()> {
    let root = SVGBackend::new(output_path, (1000, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Find the overall x and y ranges across ALL series, so every line fits
    // on one shared set of axes. `flat_map` turns "a Vec of Vecs of points"
    // into "one flat iterator of points" — we don't care which series a
    // point came from for the purpose of finding the min/max bounds.
    let all_points = series.iter().flat_map(|s| s.points.iter());
    let (mut x_min, mut x_max) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut y_min, mut y_max) = (f64::INFINITY, f64::NEG_INFINITY);
    for &(x, y) in all_points {
        x_min = x_min.min(x);
        x_max = x_max.max(x);
        y_min = y_min.min(y);
        y_max = y_max.max(y);
    }
    // A little padding on the y-axis so lines near the top/bottom of their
    // range don't visually clip against the plot border.
    let y_pad = (y_max - y_min).max(1.0) * 0.05;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d((x_min..x_max).step(1.0), (y_min - y_pad)..(y_max + y_pad))?;

    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .draw()?;

    for (idx, s) in series.iter().enumerate() {
        let color = Palette99::pick(idx);
        // `.draw_series(...)` returns a handle we can call `.label(...)`
        // and `.legend(...)` on to register this series in the chart's
        // legend — without this, the lines would be drawn but there'd be
        // no key explaining which color is which accident type.
        chart
            .draw_series(LineSeries::new(s.points.iter().copied(), &color))?
            .label(&s.label)
            // The legend callback can be invoked more than once (once per
            // legend entry drawn), so it needs `Fn`, not `FnOnce` — and `Fn`
            // closures can't consume (move out of) a captured variable on
            // each call, only borrow or copy it. `PaletteColor` turns out to
            // implement neither `Copy` nor `Clone` (a slightly awkward
            // corner of the plotters API), so instead of capturing `color`
            // itself, we capture the plain `usize` index — which IS
            // `Copy` — and re-derive the same color from it fresh inside
            // the closure body on every call.
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], Palette99::pick(idx))
            });
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

/// Draw a confusion matrix as a heatmap: rows are true labels, columns are
/// predicted labels, and each cell's shading shows what fraction of that
/// TRUE class's test windows landed in that predicted column (i.e. each row
/// sums to 1.0, or 0.0 for a class with no test support at all). Rows
/// normalized this way — rather than raw counts — keep the coloring
/// meaningful even when classes have very different support sizes.
pub fn confusion_matrix_heatmap(
    title: &str,
    labels: &[String],
    matrix: &[Vec<usize>],
    output_path: &str,
) -> Result<()> {
    let n = labels.len();
    let root = SVGBackend::new(output_path, (900, 900)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 22))
        .margin(20)
        .x_label_area_size(80)
        .y_label_area_size(80)
        .build_cartesian_2d(0..n, 0..n)?;

    chart
        .configure_mesh()
        .x_desc("Predicted")
        .y_desc("Actual (reads bottom-to-top)")
        // Force one tick per class — plotters otherwise thins labels out
        // when it thinks they'd overlap, which for an 18-class confusion
        // matrix means half the class names silently disappear. This is
        // exactly the kind of default that looks fine on a quick glance
        // and only turns out to be wrong when you actually check the chart
        // against the data it's supposed to represent.
        .x_labels(n)
        .y_labels(n)
        .x_label_formatter(&|idx| labels.get(*idx).cloned().unwrap_or_default())
        // NOTE: no inversion here — row `idx` is labeled `labels[idx]`
        // directly, matching exactly how the x-axis (and Phase 1/2's bar
        // charts) already do their labeling, which plotters centers
        // correctly by default. An earlier version of this function tried
        // to flip the y-axis so row 0 would read top-to-bottom, the more
        // conventional way to draw a confusion matrix — but plotters'
        // automatic label-centering logic doesn't invert cleanly along
        // with a custom coordinate flip, and the result was labels
        // shifted by half a cell from the data they were supposed to
        // label. Reading bottom-to-top is a little less conventional, but
        // a chart whose labels definitely line up with its data beats one
        // that looks more standard but is subtly wrong.
        .y_label_formatter(&|idx| labels.get(*idx).cloned().unwrap_or_default())
        .disable_mesh()
        .draw()?;

    for (true_idx, row) in matrix.iter().enumerate() {
        let support: usize = row.iter().sum();
        for (pred_idx, &count) in row.iter().enumerate() {
            let fraction = if support == 0 { 0.0 } else { count as f64 / support as f64 };
            let color = heat_color(fraction);
            chart.draw_series(std::iter::once(Rectangle::new(
                [(pred_idx, true_idx), (pred_idx + 1, true_idx + 1)],
                color.filled(),
            )))?;
        }
    }

    root.present()?;
    Ok(())
}

/// Map a fraction in `[0, 1]` to a color on a plain white-to-blue scale.
/// Hand-rolled rather than pulled from a colormap crate — it's a three-line
/// linear interpolation, well within "hand-roll the simple stuff" territory.
fn heat_color(fraction: f64) -> RGBColor {
    let f = fraction.clamp(0.0, 1.0);
    // At f=0: white (255,255,255). At f=1: a solid blue (30,60,150).
    // Every channel is linearly interpolated between those two endpoints.
    let r = 255.0 + (30.0 - 255.0) * f;
    let g = 255.0 + (60.0 - 255.0) * f;
    let b = 255.0 + (150.0 - 255.0) * f;
    RGBColor(r.round() as u8, g.round() as u8, b.round() as u8)
}
