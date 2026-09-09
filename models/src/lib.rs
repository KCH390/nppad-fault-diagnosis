//! `models` is the library half of this crate, same pattern as `core`
//! (see core/src/lib.rs). This crate now has TWO binaries — `train-baseline`
//! (Phase 3, the library-backed models) and `train-scratch` (Phase 4, the
//! from-scratch ones) — and both need `metrics::evaluate`, so it has to
//! live somewhere both binaries can `use`, not private inside either
//! `main.rs`.
//!
//! `pub mod metrics;`, `pub mod tree;`, `pub mod cart_classifier;`, and
//! `pub mod boosting;` compile each of those files as a submodule and mark
//! them public, so `src/main.rs` and `src/bin/train_scratch.rs` can both
//! write `use models::metrics;` (or `models::tree`, etc.) and get the exact
//! same code — no copy-pasting the evaluation logic between binaries.

pub mod boosting;
pub mod cart_classifier;
pub mod metrics;
pub mod tree;
