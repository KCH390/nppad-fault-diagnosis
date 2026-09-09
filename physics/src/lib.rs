//! `physics` is the library half of this crate, same pattern as `core` and
//! `models`. `pub mod kinetics;` compiles `src/kinetics.rs` as a submodule
//! and marks it public, so `src/bin/simulate_kinetics.rs` (and, later,
//! reach goal 9's thermal feedback extension) can `use physics::kinetics`.

pub mod kinetics;
