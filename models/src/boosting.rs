//! From-scratch multi-class gradient boosting, built on top of `tree.rs`'s
//! regression trees.
//!
//! WHAT THIS FILE DOES, IN PLAIN LANGUAGE:
//! A single decision tree (`cart_classifier.rs`) is a weak model — it can
//! only draw a handful of straight-line boundaries between classes.
//! Gradient boosting builds a whole SEQUENCE of small trees, where each new
//! tree is trained to correct the mistakes of everything trained before
//! it, and the final prediction adds all of them together. This is the
//! same core idea as the C-MAPSS project's from-scratch gradient boosting,
//! extended here from a single continuous target (RUL) to K simultaneous
//! class scores.
//!
//! HOW MULTI-CLASS BOOSTING WORKS, CONCRETELY: instead of one running
//! score per row, we keep K running scores per row — one per class,
//! `F_0(x), F_1(x), ..., F_{K-1}(x)`. At each round:
//!   1. Turn the K scores into class PROBABILITIES via softmax.
//!   2. For each class k, compute a "pseudo-residual": how far off the
//!      current probability is from the true 0/1 label for class k. This
//!      is exactly the negative gradient of cross-entropy loss with
//!      respect to `F_k` — the direction each score needs to move to make
//!      the model's predictions less wrong.
//!   3. Fit ONE regression tree per class to that class's residuals (so K
//!      trees get added per round, not just one).
//!   4. Nudge each `F_k` toward its tree's prediction, scaled by a small
//!      learning rate (so no single round can overcorrect).
//! Repeat for `n_rounds`, then predict a row's class as whichever `F_k`
//! ends up largest.
//!
//! This is the real multi-class formulation used by libraries like
//! scikit-learn's `GradientBoostingClassifier` — not a simplified
//! one-vs-rest approximation — reimplemented here from the underlying math
//! rather than an existing crate.

use crate::tree::{RegressionTree, TreeParams};

pub struct BoostParams {
    pub n_rounds: usize,
    pub learning_rate: f64,
    pub tree_params: TreeParams,
}

pub struct GradientBoostedClassifier {
    n_classes: usize,
    learning_rate: f64,
    /// `trees[round][class]` — one regression tree per class, per round.
    /// A `Vec<Vec<RegressionTree>>` rather than one flat list, since both
    /// dimensions matter: `trees.len() == n_rounds`, and
    /// `trees[r].len() == n_classes` for every round `r`.
    trees: Vec<Vec<RegressionTree>>,
}

impl GradientBoostedClassifier {
    pub fn fit(x: &[Vec<f64>], y: &[usize], n_classes: usize, params: BoostParams) -> Self {
        let n = x.len();

        // Running per-row, per-class scores, all starting at 0.0 — i.e.
        // the model starts out predicting a uniform distribution over
        // every class (softmax of all-zeros is uniform), which is exactly
        // the right "no information yet" starting point.
        let mut scores: Vec<Vec<f64>> = vec![vec![0.0; n_classes]; n];
        let mut trees: Vec<Vec<RegressionTree>> = Vec::with_capacity(params.n_rounds);

        for _round in 0..params.n_rounds {
            let probs: Vec<Vec<f64>> = scores.iter().map(|row| softmax(row)).collect();

            let mut round_trees = Vec::with_capacity(n_classes);
            for k in 0..n_classes {
                // Pseudo-residual for class k, row i: (actual 0/1 label) -
                // (current predicted probability). This is the negative
                // gradient of cross-entropy loss w.r.t. F_k — "how much,
                // and in which direction, would nudging this row's score
                // for class k reduce the loss right now."
                let residuals: Vec<f64> = (0..n)
                    .map(|i| {
                        let true_label = if y[i] == k { 1.0 } else { 0.0 };
                        true_label - probs[i][k]
                    })
                    .collect();

                let tree = RegressionTree::fit(x, &residuals, params.tree_params);

                // Nudge every row's class-k score toward this tree's
                // prediction for that row, scaled by the learning rate.
                // Small steps (a low learning rate) mean no single round's
                // tree can overcorrect based on residuals that were
                // themselves computed from a still-rough model — the same
                // reason gradient descent uses a small step size rather
                // than jumping straight to wherever the gradient points.
                for i in 0..n {
                    scores[i][k] += params.learning_rate * tree.predict_one(&x[i]);
                }

                round_trees.push(tree);
            }
            trees.push(round_trees);
        }

        GradientBoostedClassifier { n_classes, learning_rate: params.learning_rate, trees }
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<usize> {
        x.iter().map(|row| self.predict_one(row)).collect()
    }

    fn predict_one(&self, row: &[f64]) -> usize {
        let mut scores = vec![0.0; self.n_classes];
        for round_trees in &self.trees {
            for (k, tree) in round_trees.iter().enumerate() {
                scores[k] += self.learning_rate * tree.predict_one(row);
            }
        }
        argmax(&scores)
    }
}

/// Softmax: turn a row of raw scores into probabilities that sum to 1.
///
/// Subtracting the row's max value before exponentiating (`s - max`,
/// rather than exponentiating `s` directly) is the standard "softmax
/// stability trick": `f64::exp` overflows to infinity for even moderately
/// large inputs, but shifting every value down by the max first guarantees
/// the largest exponent computed is `exp(0) = 1`, with everything else
/// smaller — same final probabilities (the shift cancels out in the
/// division), computed without ever risking an overflow.
fn softmax(scores: &[f64]) -> Vec<f64> {
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|&s| (s - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

fn argmax(scores: &[f64]) -> usize {
    let mut best_idx = 0;
    let mut best_val = scores[0];
    for (idx, &v) in scores.iter().enumerate().skip(1) {
        if v > best_val {
            best_idx = idx;
            best_val = v;
        }
    }
    best_idx
}
