//! Classification metrics: confusion matrix, per-class precision/recall/F1,
//! macro F1, and accuracy.
//!
//! WHAT THIS FILE DOES, IN PLAIN LANGUAGE:
//! Functionally identical to `models::metrics` (same formulas, same
//! output shape), copied here rather than imported. Depending on
//! `models` as a library would pull `linfa` and `smartcore` into this
//! crate's dependency tree along with it, since Cargo resolves a whole
//! crate's dependencies regardless of which of its modules you actually
//! use, defeating the entire point of keeping `sequence` isolated to just
//! `candle`. This is the same "duplication over cross-crate dependency"
//! tradeoff already used in `charts/src/bin/baseline_charts.rs` and
//! `scratch_charts.rs` for reading JSON structs, just applied to a
//! slightly larger and more important piece of shared logic. If this
//! logic needs to change in the future, it needs to change in both
//! places — a real cost worth naming, not hiding.

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct ClassMetrics {
    pub label: String,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub support: usize,
}

#[derive(Debug, Serialize)]
pub struct EvalResult {
    pub accuracy: f64,
    pub macro_f1: f64,
    pub labels: Vec<String>,
    pub confusion_matrix: Vec<Vec<usize>>,
    pub per_class: Vec<ClassMetrics>,
}

pub fn evaluate(y_true: &[usize], y_pred: &[usize], labels: &[String]) -> EvalResult {
    let n_labels = labels.len();
    let mut confusion = vec![vec![0usize; n_labels]; n_labels];
    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        confusion[t][p] += 1;
    }

    let mut per_class = Vec::with_capacity(n_labels);
    for k in 0..n_labels {
        let true_positives = confusion[k][k];
        let support: usize = confusion[k].iter().sum();
        let predicted_as_k: usize = (0..n_labels).map(|i| confusion[i][k]).sum();

        let precision = if predicted_as_k == 0 {
            0.0
        } else {
            true_positives as f64 / predicted_as_k as f64
        };
        let recall = if support == 0 { 0.0 } else { true_positives as f64 / support as f64 };
        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };

        per_class.push(ClassMetrics { label: labels[k].clone(), precision, recall, f1, support });
    }

    let evaluated_classes: Vec<&ClassMetrics> = per_class.iter().filter(|c| c.support > 0).collect();
    let macro_f1 = if evaluated_classes.is_empty() {
        0.0
    } else {
        evaluated_classes.iter().map(|c| c.f1).sum::<f64>() / evaluated_classes.len() as f64
    };

    let correct = y_true.iter().zip(y_pred.iter()).filter(|(t, p)| t == p).count();
    let accuracy = correct as f64 / y_true.len() as f64;

    EvalResult {
        accuracy,
        macro_f1,
        labels: labels.to_vec(),
        confusion_matrix: confusion,
        per_class,
    }
}
