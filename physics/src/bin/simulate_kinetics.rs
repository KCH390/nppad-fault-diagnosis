//! Phase 5 CLI entry point: point reactor kinetics simulation.
//!
//! WHAT THIS FILE DOES, IN PLAIN LANGUAGE:
//! Runs two illustrative transients with the point kinetics model from
//! `physics::kinetics`: a positive reactivity ramp (the same DIRECTION of
//! change as an uncontrolled rod withdrawal, NPPAD's "RW" accident type)
//! and a negative reactivity ramp (the same direction as rod insertion,
//! "RI"). For context, it also loads a representative real RW and RI run
//! from the actual dataset and normalizes each to relative power (dividing
//! by the run's own starting value, so it's on the same 1.0-at-t=0 scale
//! the simulation uses). All four series get written to
//! `data/results/kinetics_results.json` for charting.
//!
//! THIS IS DELIBERATELY NOT A FIT. The reactivity ramp rate and magnitude
//! here are chosen to be illustrative of the right qualitative direction
//! and rough timescale, not calibrated against the real traces — this
//! model has no thermal feedback (see kinetics.rs's module docs), so it
//! CAN'T track a real accident's full trajectory once feedback and plant
//! control start to matter, typically within the first tens of seconds.
//! The point of the comparison chart is to make that gap visible, not to
//! paper over it.
//!
//! HOW TO RUN THIS:
//! `cargo run -p physics`

use core::data;
use physics::kinetics::{self, ReactivityFn};

fn main() -> anyhow::Result<()> {
    // --- Simulated transients -------------------------------------------
    // Both ramps start at t=10s, matching NPPAD's own convention of
    // initiating accidents 10 seconds into a run (noted in the Phase 1
    // README observations about run structure), ramp linearly for 15
    // seconds, then hold at their final reactivity.
    //
    // MAGNITUDE CHOICE: with no thermal feedback in this model (see
    // kinetics.rs's module docs), ANY sustained positive reactivity
    // eventually produces unbounded exponential growth — there's nothing
    // to cancel it out. A magnitude large enough to be dramatic (say,
    // +0.003, about 46% of beta) reaches many hundreds of times the
    // starting power within a minute, which is a genuinely useful thing to
    // SEE once (it's exactly why reach goal 9's thermal feedback coupling
    // exists), but it makes the real NPPAD comparison trace an invisible
    // flat line on the same linear y-axis. +0.0006 (about 9% of beta) is
    // chosen empirically here so the simulated and real curves land in a
    // comparable visual range over the 60-second window, NOT because it
    // was fit to match the real trace's magnitude.
    let ramp_magnitude = 0.0006;
    let positive_ramp: ReactivityFn = Box::new(move |t: f64| ramp_reactivity(t, 10.0, 25.0, ramp_magnitude));
    let negative_ramp: ReactivityFn =
        Box::new(move |t: f64| ramp_reactivity(t, 10.0, 25.0, -ramp_magnitude));

    let t_end = 60.0;
    let dt = 1e-3;
    println!("Simulating positive reactivity ramp (rod-withdrawal-like)...");
    let positive_sim = kinetics::simulate(positive_ramp, t_end, dt);
    println!("Simulating negative reactivity ramp (rod-insertion-like)...");
    let negative_sim = kinetics::simulate(negative_ramp, t_end, dt);

    // Downsample for the JSON/chart output: a millisecond-resolution
    // 60-second run is 60,000 points, far more than a chart needs and far
    // more than is pleasant to render as SVG line segments. Keeping one
    // point per 0.1 simulated seconds (600 points) preserves the shape of
    // the curve while keeping the output small.
    let positive_sim = downsample(&positive_sim, 100);
    let negative_sim = downsample(&negative_sim, 100);

    // --- Real NPPAD comparison traces -------------------------------------
    println!("\nLoading NPPAD dataset for comparison traces...");
    let samples = data::load_dataset("data/raw/Operation_csv_data")?;
    let rw_trace = representative_relative_power(&samples, "RW")?;
    let ri_trace = representative_relative_power(&samples, "RI")?;

    // --- Write results -----------------------------------------------------
    #[derive(serde::Serialize)]
    struct KineticsResults {
        note: String,
        positive_ramp_reactivity: f64,
        negative_ramp_reactivity: f64,
        ramp_start_seconds: f64,
        ramp_end_seconds: f64,
        simulated_positive_ramp: Vec<(f64, f64)>,
        simulated_negative_ramp: Vec<(f64, f64)>,
        real_rw_relative_power: Vec<(f64, f64)>,
        real_ri_relative_power: Vec<(f64, f64)>,
    }

    let results = KineticsResults {
        note: "Simulated curves use point kinetics with no thermal feedback \
               and are NOT fit to the real traces; see README Phase 5 notes."
            .to_string(),
        positive_ramp_reactivity: ramp_magnitude,
        negative_ramp_reactivity: -ramp_magnitude,
        ramp_start_seconds: 10.0,
        ramp_end_seconds: 25.0,
        simulated_positive_ramp: positive_sim,
        simulated_negative_ramp: negative_sim,
        real_rw_relative_power: rw_trace,
        real_ri_relative_power: ri_trace,
    };

    std::fs::create_dir_all("data/results")?;
    std::fs::write("data/results/kinetics_results.json", serde_json::to_string_pretty(&results)?)?;
    println!("\nWrote data/results/kinetics_results.json");

    Ok(())
}

/// A reactivity history that's zero before `start`, ramps linearly from 0
/// to `magnitude` between `start` and `end`, and holds at `magnitude`
/// after that.
fn ramp_reactivity(t: f64, start: f64, end: f64, magnitude: f64) -> f64 {
    if t < start {
        0.0
    } else if t < end {
        magnitude * (t - start) / (end - start)
    } else {
        magnitude
    }
}

/// Downsample a `(time, value)` series by keeping every `stride`-th point.
/// Simple and deterministic; a true signal-preserving decimation filter
/// would be overkill for a series this smooth (point kinetics output has
/// no high-frequency noise to worry about aliasing).
fn downsample(series: &[(f64, f64)], stride: usize) -> Vec<(f64, f64)> {
    series.iter().step_by(stride).copied().collect()
}

/// Find one representative sample of `accident_type` and return its `PWR`
/// (core thermal power, %) trace normalized to "relative to this run's own
/// starting value", the same 1.0-at-t=0 scale the simulation uses, since
/// NPPAD reports power as a raw percentage while the simulation tracks
/// relative change from wherever it started.
///
/// "Representative" here means the LONGEST-duration run of this accident
/// type, not (say) the lowest severity id. This matters concretely for RI
/// (rod insertion): its shortest runs are only ~100 seconds (11 rows, see
/// the Phase 1 README notes), which would barely overlap the 60-second
/// window this phase's simulated transients cover. Picking the longest run
/// available gives the comparison chart something meaningful to show
/// across that whole window instead of running out of real data after a
/// few points.
fn representative_relative_power(
    samples: &[data::Sample],
    accident_type: &str,
) -> anyhow::Result<Vec<(f64, f64)>> {
    let mut candidates: Vec<&data::Sample> =
        samples.iter().filter(|s| s.accident_type == accident_type).collect();
    candidates.sort_by(|a, b| {
        let a_dur = a.duration_seconds().unwrap_or(0.0);
        let b_dur = b.duration_seconds().unwrap_or(0.0);
        // Descending by duration: longest run first.
        b_dur.partial_cmp(&a_dur).unwrap()
    });

    let sample = candidates
        .first()
        .ok_or_else(|| anyhow::anyhow!("no samples found for accident type {accident_type}"))?;

    let sensor_idx = sample
        .sensor_names
        .iter()
        .position(|n| n == "PWR")
        .ok_or_else(|| anyhow::anyhow!("PWR sensor not found in {accident_type} sample"))?;

    let initial_power = sample.values[0][sensor_idx];
    let trace: Vec<(f64, f64)> = sample
        .time_seconds
        .iter()
        .zip(sample.values.iter())
        .map(|(&t, row)| (t, row[sensor_idx] / initial_power))
        // Truncate to the first two minutes. The full run can run to
        // thousands of seconds (see the Phase 1 README notes on run-length
        // variability), but the simulated transients here only cover 60
        // seconds, so plotting the full real run alongside them would
        // squeeze the actually-comparable region into an unreadable sliver
        // of the chart.
        .take_while(|&(t, _)| t <= 120.0)
        .collect();

    Ok(trace)
}
