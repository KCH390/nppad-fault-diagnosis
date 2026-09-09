//! Classification metrics: confusion matrix, per-class precision/recall/F1,
//! macro F1, and accuracy — all hand-rolled.
//!
//! WHAT THIS FILE DOES, IN PLAIN LANGUAGE:
//! Given a model's predictions on the test set, this turns
//! `(true_label, predicted_label)` pairs into the numbers that actually
//! answer "is this model any good": a confusion matrix (what got confused
//! with what), and per-class + macro-averaged precision/recall/F1. This is
//! genuinely simple enough that a metrics crate would be overkill — see the
//! dependency philosophy note in `models/Cargo.toml` — and hand-rolling it
//! means every number here is fully auditable against the formula that
//! produced it.
//!
//! WHY ACCURACY ALONE ISN'T ENOUGH HERE: given the class imbalance
//! documented in the Phase 1/2 README notes, a model that always predicts
//! the majority class could score a deceptively high raw accuracy while
//! being useless at recognizing rare accident types. Macro F1 — averaging
//! each class's F1 with equal weight, regardless of how many test windows
//! that class has — is far more honest about that failure mode, which is
//! exactly why the Phase 3 roadmap called for "macro F1 + confusion matrix
//! over raw accuracy" from the start.

use serde::Serialize;

/// Precision/recall/F1/support for one class.
#[derive(Debug, Clone, Serialize)]
pub struct ClassMetrics {
    pub label: String,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    /// Number of TEST windows whose true label is this class. A class with
    /// `support == 0` has no held-out test data at all (this happens for
    /// this dataset's single-run accident types — see `core::split`) and
    /// is excluded from the `macro_f1` average, though it's still listed
    /// here with whatever precision/recall its (zero) support implies.
    pub support: usize,
}

/// Full evaluation result for one model on one test set.
#[derive(Debug, Serialize)]
pub struct EvalResult {
    pub accuracy: f64,
    /// Macro-averaged F1, over only the classes that have `support > 0`
    /// (see `ClassMetrics::support` above and the module docs).
    pub macro_f1: f64,
    pub labels: Vec<String>,
    /// `confusion_matrix[true_idx][predicted_idx]`, indexed the same way as
    /// `labels`.
    pub confusion_matrix: Vec<Vec<usize>>,
    pub per_class: Vec<ClassMetrics>,
}

/// Compute a full `EvalResult` from parallel `y_true`/`y_pred` label-index
/// arrays (indices into `labels`).
pub fn evaluate(y_true: &[usize], y_pred: &[usize], labels: &[String]) -> EvalResult {
    let n_labels = labels.len();

    // `confusion[true][predicted] += 1` for every test window. A
    // `Vec<Vec<usize>>` (rather than a flat `Vec<usize>` with manual index
    // math) keeps the `[true][predicted]` indexing readable at the cost of
    // one extra level of heap indirection — completely fine at this scale
    // (n_labels is 18, so the whole matrix is 18*18 = 324 entries).
    let mut confusion = vec![vec![0usize; n_labels]; n_labels];
    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        confusion[t][p] += 1;
    }

    let mut per_class = Vec::with_capacity(n_labels);
    for k in 0..n_labels {
        let true_positives = confusion[k][k];

        // Support: how many test windows actually belong to class k — the
        // sum along ROW k (every column is "predicted as," row k fixes
        // "actually is k").
        let support: usize = confusion[k].iter().sum();

        // How many windows (of ANY true class) the model predicted as
        // class k — the sum down COLUMN k.
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

        per_class.push(ClassMetrics {
            label: labels[k].clone(),
            precision,
            recall,
            f1,
            support,
        });
    }

    // Macro F1: unweighted mean of F1 across classes that actually have
    // test support. `.filter(...)` here is what implements the "exclude
    // single-run accident types from the headline metric" decision
    // documented in core::split — those classes always have support == 0
    // in the test set, by construction, since core::split never assigns
    // them a Test half.
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
