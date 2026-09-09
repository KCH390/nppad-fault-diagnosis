//! A from-scratch multi-class CART classification tree.
//!
//! WHAT THIS FILE DOES, IN PLAIN LANGUAGE:
//! Structurally almost identical to `tree.rs`'s regression tree — same
//! recursive binary-split idea — but with two differences suited to
//! classification instead of regression:
//!   1. Splits are chosen by **Gini impurity** reduction, not variance
//!      reduction. Gini impurity measures how "mixed" the class labels are
//!      in a group of rows (0 = perfectly pure, one class only; higher =
//!      more mixed across classes).
//!   2. A leaf stores a **class count distribution** (how many training
//!      rows of each class ended up there), not a single number — so
//!      prediction is "which class was most common among training rows
//!      that reached this leaf," not a continuous average.
//!
//! This is deliberately a SEPARATE, self-contained tree implementation
//! rather than trying to force `tree.rs`'s `RegressionTree` to also handle
//! classification (e.g. by one-hot-encoding classes into 0/1 targets and
//! wrapping it). The splitting criteria are genuinely different math
//! (Gini vs. variance), and keeping them as two small, focused files is
//! easier to read and verify than one file trying to do both jobs behind
//! a shared abstraction.

use std::collections::HashMap;

#[derive(Debug, Clone)]
enum Node {
    /// `class_counts[k]` = how many training rows of class `k` reached
    /// this leaf. Kept as counts (not just the majority class) so a
    /// distribution is available if a future phase wants probabilities,
    /// not just a single best guess.
    Leaf { class_counts: Vec<usize> },
    Split { feature_idx: usize, threshold: f64, left: Box<Node>, right: Box<Node> },
}

#[derive(Debug, Clone)]
pub struct ClassificationTree {
    root: Node,
}

#[derive(Debug, Clone, Copy)]
pub struct TreeParams {
    pub max_depth: usize,
    pub min_samples_leaf: usize,
}

impl ClassificationTree {
    /// Fit a classification tree. `y` holds class INDICES (0..n_classes),
    /// the same encoding `core::split`/`models::main` already use
    /// elsewhere in this project — not one-hot vectors or string labels.
    pub fn fit(x: &[Vec<f64>], y: &[usize], n_classes: usize, params: TreeParams) -> Self {
        let indices: Vec<usize> = (0..x.len()).collect();
        let root = build_node(x, y, &indices, n_classes, 0, params);
        ClassificationTree { root }
    }

    /// Predict the single most likely class for each row (the class with
    /// the most training rows at the leaf that row lands in).
    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<usize> {
        x.iter().map(|row| predict_one(&self.root, row)).collect()
    }
}

fn predict_one(node: &Node, row: &[f64]) -> usize {
    match node {
        Node::Leaf { class_counts } => argmax(class_counts),
        Node::Split { feature_idx, threshold, left, right } => {
            if row[*feature_idx] <= *threshold {
                predict_one(left, row)
            } else {
                predict_one(right, row)
            }
        }
    }
}

/// Index of the largest value in `counts`. Ties break toward the lower
/// index — an arbitrary but deterministic choice, which matters more than
/// which specific rule is used (a non-deterministic tie-break would make
/// two runs on identical data disagree for no real reason).
fn argmax(counts: &[usize]) -> usize {
    let mut best_idx = 0;
    let mut best_count = counts[0];
    for (idx, &count) in counts.iter().enumerate().skip(1) {
        if count > best_count {
            best_idx = idx;
            best_count = count;
        }
    }
    best_idx
}

fn class_counts(y: &[usize], indices: &[usize], n_classes: usize) -> Vec<usize> {
    let mut counts = vec![0usize; n_classes];
    for &i in indices {
        counts[y[i]] += 1;
    }
    counts
}

/// Gini impurity of a set of class counts: `1 - sum(p_k^2)` over class
/// proportions `p_k`. 0.0 means every row in this group is the same
/// class (perfectly pure); it approaches 1.0 as the group gets more evenly
/// mixed across many classes.
fn gini(counts: &[usize]) -> f64 {
    let total: usize = counts.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let total_f = total as f64;
    let sum_sq_proportions: f64 = counts
        .iter()
        .map(|&c| {
            let p = c as f64 / total_f;
            p * p
        })
        .sum();
    1.0 - sum_sq_proportions
}

fn build_node(
    x: &[Vec<f64>],
    y: &[usize],
    indices: &[usize],
    n_classes: usize,
    depth: usize,
    params: TreeParams,
) -> Node {
    let counts = class_counts(y, indices, n_classes);

    if depth >= params.max_depth || indices.len() < 2 * params.min_samples_leaf {
        return Node::Leaf { class_counts: counts };
    }

    match best_split(x, y, indices, n_classes, params) {
        Some((feature_idx, threshold, left_indices, right_indices)) => Node::Split {
            feature_idx,
            threshold,
            left: Box::new(build_node(x, y, &left_indices, n_classes, depth + 1, params)),
            right: Box::new(build_node(x, y, &right_indices, n_classes, depth + 1, params)),
        },
        None => Node::Leaf { class_counts: counts },
    }
}

/// Search every (feature, threshold) candidate and return the one that
/// most reduces Gini impurity, split into its two resulting index groups.
/// Returns `None` if no valid split was found.
///
/// Uses the same sorted-sweep strategy as `tree.rs`'s `best_split` (see the
/// performance note there for the full story): sort each feature's values
/// once, then sweep left-to-right maintaining running per-class counts,
/// rather than re-partitioning the full index list from scratch for every
/// candidate threshold. The win here is smaller than in the regression
/// tree (Gini needs O(n_classes) work per step either way, since it has to
/// touch every class's count), but the O(n) rescan this replaces still
/// mattered enough to fix at the same time rather than leave a second copy
/// of the same slow pattern in the codebase.
fn best_split(
    x: &[Vec<f64>],
    y: &[usize],
    indices: &[usize],
    n_classes: usize,
    params: TreeParams,
) -> Option<(usize, f64, Vec<usize>, Vec<usize>)> {
    let n_features = x[0].len();
    let n = indices.len();
    let total_counts = class_counts(y, indices, n_classes);
    let parent_gini = gini(&total_counts);

    // `best` tracks `(gini_reduction, feature_idx, threshold)` — index
    // groups are built once at the end, not per candidate (see tree.rs's
    // best_split for the fuller explanation of why that matters).
    let mut best: Option<(f64, usize, f64)> = None;

    for feature_idx in 0..n_features {
        // Sort this feature's (value, class) pairs together once.
        let mut pairs: Vec<(f64, usize)> = indices.iter().map(|&i| (x[i][feature_idx], y[i])).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Running per-class counts for everything seen so far ("left" of
        // the current sweep position). `right_counts` at each step is just
        // `total_counts[k] - left_counts[k]` — no separate rescan needed.
        let mut left_counts = vec![0usize; n_classes];
        let mut left_total = 0usize;

        for i in 0..n {
            let (value, class) = pairs[i];
            left_counts[class] += 1;
            left_total += 1;

            let is_last = i == n - 1;
            let next_value_differs = !is_last && pairs[i + 1].0 != value;
            if is_last || !next_value_differs {
                continue;
            }

            let right_total = n - left_total;
            if left_total < params.min_samples_leaf || right_total < params.min_samples_leaf {
                continue;
            }

            // `right_counts[k] = total_counts[k] - left_counts[k]` for
            // every class — built fresh each candidate step since it's
            // only O(n_classes) work (18 here), not O(n).
            let right_counts: Vec<usize> =
                (0..n_classes).map(|k| total_counts[k] - left_counts[k]).collect();

            let weighted_child_gini = (left_total as f64 / n as f64) * gini(&left_counts)
                + (right_total as f64 / n as f64) * gini(&right_counts);
            let reduction = parent_gini - weighted_child_gini;

            let is_better = match &best {
                None => reduction > 1e-12,
                Some((best_reduction, ..)) => reduction > *best_reduction,
            };
            if is_better {
                best = Some((reduction, feature_idx, value));
            }
        }
    }

    best.map(|(_, feature_idx, threshold)| {
        let (left, right): (Vec<usize>, Vec<usize>) =
            indices.iter().partition(|&&i| x[i][feature_idx] <= threshold);
        (feature_idx, threshold, left, right)
    })
}

/// Build a `label name -> index` lookup, the same encoding used everywhere
/// else in this project (`core::split`, `models::main`). Provided here so
/// `train_scratch.rs` doesn't have to re-derive this from scratch itself.
pub fn build_label_index(labels: &[String]) -> HashMap<&str, usize> {
    labels.iter().enumerate().map(|(i, l)| (l.as_str(), i)).collect()
}
