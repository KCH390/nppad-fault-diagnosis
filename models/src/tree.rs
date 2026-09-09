//! A from-scratch CART regression tree — zero dependencies, same spirit as
//! the hand-rolled CART/boosting code in the C-MAPSS project.
//!
//! WHAT THIS FILE DOES, IN PLAIN LANGUAGE:
//! Given a matrix of features and a real-valued target per row, this
//! builds a binary decision tree: at each node, pick the (feature,
//! threshold) split that best separates the rows into two groups with
//! more similar target values than the parent node had — "more similar"
//! measured by variance. Recurse on each half until a stopping rule (max
//! depth, or too few rows to split further) is hit, then the leaf's
//! prediction is just the mean target value of whatever rows landed there.
//!
//! This is a REGRESSION tree — it predicts a continuous number, not a
//! class. It's used two ways in Phase 4: directly, as one ingredient of
//! `cart_classifier.rs`'s tree-of-class-counts (see that file for how
//! classification differs), and as the actual base learner inside
//! `boosting.rs`'s gradient boosting, where "the target" is a
//! per-round pseudo-residual rather than a class label.

/// A node in the tree: either a leaf with a predicted value, or an
/// internal split node.
///
/// `Box<Node>` here is necessary, not just stylistic: `Node` contains
/// `Node` (a `Split` holds two child `Node`s), and Rust needs to know the
/// exact size of every type at compile time. A type that directly contains
/// itself would be infinitely large, which doesn't compile — `Box` breaks
/// that cycle by storing the child on the heap and keeping only a
/// fixed-size pointer to it inline. This is the standard way to write
/// recursive data structures in Rust; Python doesn't need the equivalent
/// because every Python object is already heap-allocated and referred to
/// by pointer under the hood.
#[derive(Debug, Clone)]
pub enum Node {
    Leaf {
        value: f64,
    },
    Split {
        feature_idx: usize,
        threshold: f64,
        left: Box<Node>,
        right: Box<Node>,
    },
}

/// A trained regression tree, plus the hyperparameters used to build it
/// (kept around mainly for the record — `predict` only needs `root`).
#[derive(Debug, Clone)]
pub struct RegressionTree {
    root: Node,
}

/// Hyperparameters controlling how deep/eagerly a tree grows. Both
/// stopping rules exist for the same reason: an unconstrained tree will
/// happily grow one leaf per training row, which fits the training data
/// perfectly and generalizes terribly — the classic overfitting failure
/// mode for decision trees.
#[derive(Debug, Clone, Copy)]
pub struct TreeParams {
    pub max_depth: usize,
    pub min_samples_leaf: usize,
}

impl RegressionTree {
    /// Fit a regression tree to `x` (row-major feature matrix, `x[row]` is
    /// one sample's features) and `y` (one target value per row).
    ///
    /// `indices: Vec<usize>` — rather than slicing `x`/`y` directly at
    /// every recursive call — is a common trick for tree-building code:
    /// instead of copying rows into new, smaller Vecs at every split (which
    /// gets expensive fast, since the same underlying data would get
    /// copied over and over as the tree gets deeper), each node just holds
    /// the *indices* of the rows that belong to it, and always reads from
    /// the one original `x`/`y` the whole build shares.
    pub fn fit(x: &[Vec<f64>], y: &[f64], params: TreeParams) -> Self {
        let indices: Vec<usize> = (0..x.len()).collect();
        let root = build_node(x, y, &indices, 0, params);
        RegressionTree { root }
    }

    /// Predict a single row's target value by walking the tree from the
    /// root: at each `Split`, go left if this row's `feature_idx` value is
    /// `<= threshold`, right otherwise, until a `Leaf` is reached.
    pub fn predict_one(&self, row: &[f64]) -> f64 {
        predict_node(&self.root, row)
    }

    /// Predict every row in `x`. A thin convenience wrapper — `.iter().map(...)`
    /// over `predict_one` — kept as its own method so callers don't all
    /// have to write that same one-liner themselves.
    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter().map(|row| self.predict_one(row)).collect()
    }
}

fn predict_node(node: &Node, row: &[f64]) -> f64 {
    match node {
        Node::Leaf { value } => *value,
        Node::Split { feature_idx, threshold, left, right } => {
            if row[*feature_idx] <= *threshold {
                predict_node(left, row)
            } else {
                predict_node(right, row)
            }
        }
    }
}

/// Recursively build one node (and, if it splits, its whole subtree) over
/// the rows named by `indices`.
fn build_node(x: &[Vec<f64>], y: &[f64], indices: &[usize], depth: usize, params: TreeParams) -> Node {
    let leaf_value = mean(indices.iter().map(|&i| y[i]));

    // Stopping rules: give up and return a leaf if we've hit the depth
    // limit, or if splitting further would leave a child with fewer than
    // `min_samples_leaf` rows (checked inside `best_split` too, but this
    // outer check short-circuits the whole search when there obviously
    // aren't enough rows to split at all).
    if depth >= params.max_depth || indices.len() < 2 * params.min_samples_leaf {
        return Node::Leaf { value: leaf_value };
    }

    match best_split(x, y, indices, params) {
        Some((feature_idx, threshold, left_indices, right_indices)) => Node::Split {
            feature_idx,
            threshold,
            left: Box::new(build_node(x, y, &left_indices, depth + 1, params)),
            right: Box::new(build_node(x, y, &right_indices, depth + 1, params)),
        },
        // No split improved on just predicting the mean — every candidate
        // either failed the min_samples_leaf rule or didn't reduce
        // variance at all (e.g. every row has an identical target value
        // already). Either way, a leaf is the right answer here.
        None => Node::Leaf { value: leaf_value },
    }
}

/// Search every (feature, threshold) candidate and return the one that
/// most reduces the variance of `y` within `indices`, split into its two
/// resulting index groups. Returns `None` if no valid split was found.
///
/// PERFORMANCE NOTE: an earlier version of this function tried every
/// threshold by calling `.partition(...)` over `indices` from scratch each
/// time — an O(n) full rescan per candidate threshold, times up to n
/// candidate thresholds, times every feature: O(n² · n_features) per node.
/// On this project's ~300-feature windows that made even a shallow tree
/// take minutes. The fix below sorts each feature's values ONCE, then
/// sweeps through that sorted order left-to-right, updating running sums
/// incrementally instead of recomputing variance from scratch at every
/// threshold — O(n log n · n_features) per node instead. Same idea as the
/// C-MAPSS project's from-scratch gradient boosting hitting (and fixing) a
/// comparable performance issue on its larger dataset.
/// Search every (feature, threshold) candidate and return the one that
/// most reduces the variance of `y` within `indices`, split into its two
/// resulting index groups. Returns `None` if no valid split was found.
///
/// PERFORMANCE NOTE: an earlier version of this function tried every
/// threshold by calling `.partition(...)` over `indices` from scratch each
/// time — an O(n) full rescan per candidate threshold, times up to n
/// candidate thresholds, times every feature: O(n² · n_features) per node.
/// On this project's ~300-feature windows that made even a shallow tree
/// take minutes to train, discovered when Phase 4's gradient boosting (324
/// trees, one call to this function per node per tree) didn't finish
/// inside a several-minute timeout on a training set of barely 1,000 rows.
/// The fix below sorts each feature's values ONCE, then sweeps through
/// that sorted order left-to-right, updating running sums incrementally
/// instead of recomputing sum-of-squared-error from scratch at every
/// threshold — O(n log n · n_features) per node instead. Same category of
/// issue as the C-MAPSS project's from-scratch gradient boosting hitting
/// (and fixing) a comparable performance problem on its larger dataset.
fn best_split(
    x: &[Vec<f64>],
    y: &[f64],
    indices: &[usize],
    params: TreeParams,
) -> Option<(usize, f64, Vec<usize>, Vec<usize>)> {
    let n_features = x[0].len();
    let n = indices.len();
    let total_sum: f64 = indices.iter().map(|&i| y[i]).sum();
    let total_sq_sum: f64 = indices.iter().map(|&i| y[i] * y[i]).sum();
    let parent_sse = total_sq_sum - (total_sum * total_sum) / n as f64;

    // `best` tracks `(sse_reduction, feature_idx, threshold)` for whichever
    // candidate split has been best so far. We defer actually building the
    // left/right index Vecs until after the search picks a winner — no
    // point allocating them for every candidate we're going to discard.
    let mut best: Option<(f64, usize, f64)> = None;

    for feature_idx in 0..n_features {
        // Sort THIS feature's (value, target) pairs together, once, rather
        // than sorting `indices` — we still need to know each row's `y`
        // value as we sweep, and pairing them up front avoids re-indexing
        // into `y` via `indices[i]` inside the hot loop below.
        let mut pairs: Vec<(f64, f64)> = indices.iter().map(|&i| (x[i][feature_idx], y[i])).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Sweep left-to-right, maintaining the running sum and sum-of-
        // squares of everything seen so far ("left" of the current
        // position). At each candidate split point (between two distinct
        // values), the right side's stats are just the precomputed totals
        // minus the left side's running stats — O(1) per step instead of
        // re-scanning the right side from scratch.
        let mut left_sum = 0.0;
        let mut left_sq_sum = 0.0;
        let mut left_count = 0usize;

        for i in 0..n {
            let (value, target) = pairs[i];
            left_sum += target;
            left_sq_sum += target * target;
            left_count += 1;

            // Only a valid split point if the NEXT value differs (splitting
            // between two equal values would put identical feature values
            // on both sides of the boundary depending on tie-breaking,
            // which isn't a meaningful split) and both sides meet the
            // minimum leaf size.
            let is_last = i == n - 1;
            let next_value_differs = !is_last && pairs[i + 1].0 != value;
            if is_last || !next_value_differs {
                continue;
            }

            let right_count = n - left_count;
            if left_count < params.min_samples_leaf || right_count < params.min_samples_leaf {
                continue;
            }

            let left_sse = left_sq_sum - (left_sum * left_sum) / left_count as f64;
            let right_sum = total_sum - left_sum;
            let right_sq_sum = total_sq_sum - left_sq_sum;
            let right_sse = right_sq_sum - (right_sum * right_sum) / right_count as f64;

            let sse_reduction = parent_sse - (left_sse + right_sse);

            let is_better = match &best {
                None => sse_reduction > 1e-12, // tiny epsilon guards against floating-point noise counting as "improvement"
                Some((best_reduction, ..)) => sse_reduction > *best_reduction,
            };
            if is_better {
                let threshold = value; // split rule is "<= value" goes left
                best = Some((sse_reduction, feature_idx, threshold));
            }
        }
    }

    // Now that we know the winning (feature, threshold), actually build
    // the index groups — exactly one `.partition()` call total, not one
    // per candidate threshold like the old version.
    best.map(|(_, feature_idx, threshold)| {
        let (left, right): (Vec<usize>, Vec<usize>) =
            indices.iter().partition(|&&i| x[i][feature_idx] <= threshold);
        (feature_idx, threshold, left, right)
    })
}

/// Population mean of an iterator of `f64`. Generic over `impl Iterator<Item
/// = f64>` (rather than taking a `&[f64]`) so callers can pass in a `.map(...)`
/// chain directly, like `mean(indices.iter().map(|&i| y[i]))` above, without
/// needing to `.collect()` an intermediate Vec first. Still used by
/// `build_node` for leaf values and the mean-based stopping-rule check —
/// `best_split` itself no longer needs a separate `variance` helper since
/// its incremental sweep tracks sum-of-squared-error directly (see the
/// performance note on `best_split` above).
fn mean(values: impl Iterator<Item = f64> + Clone) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for v in values {
        sum += v;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}
