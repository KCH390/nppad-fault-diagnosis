//! `core` is the library half of this crate. Rust packages can have both a
//! library target (`lib.rs`, what other crates/binaries can depend on) and
//! one or more binary targets (`main.rs`-style, things that produce a
//! runnable executable). We use that split deliberately here: `main.rs` is a
//! thin CLI wrapper, and all the real logic lives in modules declared below
//! so that the `charts` crate (in Phase 1's charting binary) can reuse the
//! exact same data-loading and EDA code instead of duplicating it.
//!
//! `pub mod data;`, `pub mod eda;`, `pub mod features;`, `pub mod
//! sequences;`, and `pub mod split;` below do two things at once:
//! 1. Tell Rust to compile `src/data.rs`, `src/eda.rs`, `src/features.rs`,
//!    `src/sequences.rs`, and `src/split.rs` as submodules of this crate
//!    (the filename becomes the module name).
//! 2. Mark them `pub` (public) so code outside this crate — like the
//!    `charts`, `models`, `physics`, and `sequence` crates, or this
//!    crate's own `src/bin/` binaries — can call `core::data::...`,
//!    `core::eda::...`, `core::features::...`, `core::sequences::...`,
//!    and `core::split::...`.
//!
//! If we left off `pub`, the modules would still compile, but only code
//! *inside* the `core` crate could see them. This is Rust's privacy system:
//! everything is private by default, and you opt into visibility
//! explicitly. It's the same instinct as Python's underscore-prefix
//! convention, except the compiler actually enforces it.

pub mod data;
pub mod eda;
pub mod features;
pub mod sequences;
pub mod split;

