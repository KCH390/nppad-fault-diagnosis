//! `sequence` is the library half of this crate, same pattern as `core`,
//! `models`, and `physics`. `pub mod lstm;` and `pub mod metrics;` compile
//! `src/lstm.rs` and `src/metrics.rs` as public submodules so
//! `src/bin/train_lstm.rs` can `use sequence::lstm` and
//! `use sequence::metrics`.

pub mod lstm;
pub mod metrics;
