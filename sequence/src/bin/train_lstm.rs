//! Phase 6 CLI entry point: LSTM sequence classification.
//!
//! WHAT THIS FILE DOES, IN PLAIN LANGUAGE:
//! Same overall shape as Phases 3-4's training binaries (load data, split
//! by run via `core::split`, subsample for tractability, train, evaluate,
//! write JSON), but built around whole per-run SEQUENCES
//! (`core::sequences`) instead of Phase 2's windowed feature vectors, and
//! trained with `sequence::lstm::LstmClassifier` instead of a tree-based
//! model.
//!
//! HOW TO RUN THIS:
//! `cargo run -p sequence`
//! Optional arguments override the defaults:
//! `cargo run -p sequence -- <seq_len> <test_fraction> <max_per_class> <hidden_dim> <n_epochs>`

use candle_core::{DType, Device, Tensor};
use candle_nn::{loss::cross_entropy, AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use core::data;
use core::sequences::{self, NormalizationStats, SequenceExample};
use core::split::{self, Split};
use sequence::lstm::LstmClassifier;
use sequence::metrics;
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // 30 rows (300s) balances keeping most of a run's onset behavior
    // against how much "hold-last-value" padding the shortest runs need
    // (RI's shortest is 11 rows — see core::sequences' module docs and the
    // Phase 1 README notes on run-length variability).
    let seq_len: usize = args.first().map(|s| s.parse()).transpose()?.unwrap_or(30);
    let test_fraction: f64 = args.get(1).map(|s| s.parse()).transpose()?.unwrap_or(0.2);
    // A smaller cap than Phases 3-4's tree models: every training example
    // here is a full (seq_len x 96) sequence rather than one flat feature
    // vector, and the LSTM has to run its recurrent step seq_len times per
    // example, per epoch, so the practical per-example cost is much higher.
    let max_per_class: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(300);
    let hidden_dim: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(32);
    let n_epochs: usize = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(30);

    println!("Loading NPPAD dataset...");
    let samples = data::load_dataset("data/raw/Operation_csv_data")?;
    println!("Loaded {} samples.", samples.len());

    let run_split = split::split_by_run(&samples, test_fraction);
    println!(
        "Run-level split: {} accident types have only one run and are train-only: {:?}",
        run_split.single_run_types.len(),
        run_split.single_run_types
    );

    let mut labels: Vec<String> = samples.iter().map(|s| s.accident_type.clone()).collect();
    labels.sort();
    labels.dedup();
    let n_classes = labels.len();
    let label_index: HashMap<&str, usize> =
        labels.iter().enumerate().map(|(i, l)| (l.as_str(), i)).collect();

    println!("Building fixed-length sequences (seq_len={seq_len} rows)...");
    let all_sequences = sequences::build_sequences(&samples, seq_len);

    let mut train_examples = Vec::new();
    let mut test_examples = Vec::new();
    for (sample, example) in samples.iter().zip(all_sequences.into_iter()) {
        let goes_to_train = split::is_single_run_type(&run_split, &sample.accident_type)
            || run_split.assignment_for(&sample.accident_type, &sample.severity_id) == Some(Split::Train);
        if goes_to_train {
            train_examples.push(example);
        } else {
            test_examples.push(example);
        }
    }

    // Unlike Phases 3-4 (subsampling overlapping WINDOWS from the same
    // handful of runs), each sequence example here already corresponds to
    // one whole, distinct run — so `max_per_class` is capping the number
    // of RUNS per class, not overlapping slices of them. Some accident
    // types simply won't have `max_per_class` runs to give at all (recall
    // most "Severity" types have ~100 runs and "Other" types have exactly
    // 1); `subsample_by_class` already handles "fewer available than the
    // cap" by keeping everything in that case.
    let train_examples = split::subsample_by_class(
        train_examples,
        |e: &SequenceExample| e.accident_type.as_str(),
        max_per_class,
    );
    let test_examples = split::subsample_by_class(
        test_examples,
        |e: &SequenceExample| e.accident_type.as_str(),
        (max_per_class / 2).max(1),
    );
    println!(
        "After subsampling: {} train sequences, {} test sequences.",
        train_examples.len(),
        test_examples.len()
    );

    // Fit normalization on TRAIN ONLY, then apply to both splits — see
    // core::sequences' module docs for why fitting on test data too would
    // be a leakage bug, not just an unnecessary step.
    let norm_stats = NormalizationStats::fit(&train_examples);
    let mut train_examples = train_examples;
    let mut test_examples = test_examples;
    norm_stats.apply(&mut train_examples);
    norm_stats.apply(&mut test_examples);

    let device = Device::Cpu;
    let input_dim = train_examples[0].sequence[0].len();

    let x_train = sequences_to_tensor(&train_examples, &device)?;
    let y_train = labels_to_tensor(&train_examples, &label_index, &device)?;
    let x_test = sequences_to_tensor(&test_examples, &device)?;
    let y_test_indices: Vec<usize> =
        test_examples.iter().map(|e| label_index[e.accident_type.as_str()]).collect();

    // `VarMap` tracks every trainable parameter the model creates (via the
    // `VarBuilder` handed to `LstmClassifier::new`) so the optimizer knows
    // what to update. This is candle's equivalent of a PyTorch module's
    // `.parameters()` — the mechanism that connects "the model" to "the
    // thing the optimizer adjusts during `backward_step`."
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = LstmClassifier::new(input_dim, hidden_dim, n_classes, vb)?;

    let adamw_params = ParamsAdamW { lr: 1e-3, ..Default::default() };
    let mut optimizer = AdamW::new(varmap.all_vars(), adamw_params)?;

    println!("\nTraining LSTM ({n_epochs} epochs, hidden_dim={hidden_dim})...");
    for epoch in 0..n_epochs {
        let logits = model.forward(&x_train)?;
        let loss = cross_entropy(&logits, &y_train)?;
        optimizer.backward_step(&loss)?;

        if epoch % 5 == 0 || epoch == n_epochs - 1 {
            let loss_value = loss.to_scalar::<f32>()?;
            println!("  epoch {epoch}: training loss = {loss_value:.4}");
        }
    }

    println!("\nEvaluating on held-out test sequences...");
    let test_logits = model.forward(&x_test)?;
    // `argmax(1)` picks the highest-scoring class along dimension 1 (the
    // n_classes dimension of a (batch, n_classes) logits tensor) — the
    // same "which class did the model think was most likely" operation as
    // `argmax` in models::cart_classifier, just via candle's tensor API
    // instead of a plain Rust slice.
    let predictions = test_logits.argmax(1)?.to_vec1::<u32>()?;
    let y_pred: Vec<usize> = predictions.iter().map(|&p| p as usize).collect();

    let eval = metrics::evaluate(&y_test_indices, &y_pred, &labels);
    println!("LSTM: accuracy={:.4}, macro_f1={:.4}", eval.accuracy, eval.macro_f1);
    for c in &eval.per_class {
        if c.support > 0 {
            println!(
                "  {:<8} precision={:.3} recall={:.3} f1={:.3} support={}",
                c.label, c.precision, c.recall, c.f1, c.support
            );
        }
    }

    #[derive(serde::Serialize)]
    struct LstmResults {
        seq_len: usize,
        test_fraction: f64,
        max_per_class_train: usize,
        hidden_dim: usize,
        n_epochs: usize,
        train_sequences: usize,
        test_sequences: usize,
        single_run_types_train_only: Vec<String>,
        eval: metrics::EvalResult,
    }

    let results = LstmResults {
        seq_len,
        test_fraction,
        max_per_class_train: max_per_class,
        hidden_dim,
        n_epochs,
        train_sequences: train_examples.len(),
        test_sequences: test_examples.len(),
        single_run_types_train_only: run_split.single_run_types.clone(),
        eval,
    };

    std::fs::create_dir_all("data/results")?;
    std::fs::write("data/results/lstm_results.json", serde_json::to_string_pretty(&results)?)?;
    println!("\nWrote data/results/lstm_results.json");

    Ok(())
}

/// Flatten a batch of `SequenceExample`s into one `(batch, seq_len,
/// input_dim)` tensor, batch-first layout. Confirmed against a real run
/// on Kerry's machine to be what this project's installed `candle-nn`
/// version's `RNN::seq()` actually expects (it reads dimension 0 as
/// batch, not time). An earlier version of this function assumed
/// `(seq_len, batch, input_dim)` instead, which silently fed the model a
/// batch size equal to `seq_len` and vice versa, surfacing as a
/// `cross_entropy` batch-size mismatch between the logits and the target
/// labels rather than a compile error.
///
/// The nested loop order here (`b`, then `t`, then `f`) has to match
/// exactly how `Tensor::from_vec` interprets a flat buffer for a given
/// shape: candle (like `ndarray` and numpy) treats the LAST dimension as
/// fastest-varying ("row-major"/"C order"), so for shape
/// `(batch, seq_len, input_dim)` the flat buffer must vary `input_dim`
/// fastest, then `seq_len`, then `batch` slowest, exactly the iteration
/// order below.
fn sequences_to_tensor(examples: &[SequenceExample], device: &Device) -> candle_core::Result<Tensor> {
    let batch = examples.len();
    let seq_len = examples[0].sequence.len();
    let input_dim = examples[0].sequence[0].len();

    let mut flat = Vec::with_capacity(batch * seq_len * input_dim);
    for example in examples {
        for t in 0..seq_len {
            for &v in &example.sequence[t] {
                flat.push(v as f32);
            }
        }
    }

    Tensor::from_vec(flat, (batch, seq_len, input_dim), device)
}

/// Build the `(batch,)` u32 label tensor `cross_entropy` expects as its
/// target argument.
fn labels_to_tensor(
    examples: &[SequenceExample],
    label_index: &HashMap<&str, usize>,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let labels: Vec<u32> =
        examples.iter().map(|e| label_index[e.accident_type.as_str()] as u32).collect();
    Tensor::from_vec(labels, (examples.len(),), device)
}
