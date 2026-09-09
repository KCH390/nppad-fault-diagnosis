//! Point reactor kinetics: the standard six-delayed-neutron-group model of
//! how reactor power responds to a reactivity change, integrated with a
//! hand-rolled 4th-order Runge-Kutta solver.
//!
//! WHAT THIS FILE DOES, IN PLAIN LANGUAGE:
//! "Point kinetics" treats the whole reactor core as a single point, no
//! spatial detail, and asks one question: given a reactivity insertion
//! (a control rod moving, a change in coolant conditions, whatever
//! initiates a transient), how does neutron population (and therefore
//! reactor power) change over time? Most of that response isn't
//! instantaneous, because a small fraction of neutrons ("delayed
//! neutrons") come from the radioactive decay of fission fragments a
//! fraction of a second to tens of seconds AFTER the fission event, rather
//! than immediately. The six-group model tracks those delayed neutrons
//! explicitly: six populations of "precursor" fragments, each with its own
//! fraction of the total delayed neutrons and its own decay rate, that
//! collectively give reactor power its characteristic response speed. This
//! is exactly why reactors are controllable at all. If every neutron were
//! prompt (produced instantly at fission), the reactor's response time
//! would be measured in microseconds, far too fast for a mechanical
//! control system to react to.
//!
//! WHAT THIS MODEL DOES NOT DO: no thermal feedback. A real PWR's
//! reactivity depends on fuel and coolant temperature (as the core heats
//! up, negative feedback coefficients push reactivity back down, which is
//! part of why real reactors don't just run away during most transients).
//! This model takes reactivity as an external, prescribed input
//! (`ReactivityFn`) and doesn't compute it FROM the simulated power or
//! temperature at all. That coupling is reach goal 9's job, not this
//! phase's. Comparing this model's output against a real NPPAD accident
//! trace (see `simulate_kinetics.rs`) is explicitly a qualitative
//! illustration of that gap, not a validated fit.

/// The six-group delayed neutron data used here (fraction `beta_i` and
/// decay constant `lambda_i`, per second) are the standard thermal-fission
/// values for U-235, as tabulated in reactor physics references (e.g.
/// Duderstadt & Hamilton, *Nuclear Reactor Analysis*). NPPAD's PWR is a
/// U-235-fueled 3-loop plant, so these are the physically appropriate
/// values to use here, not an arbitrary placeholder set.
pub const DELAYED_NEUTRON_FRACTIONS: [f64; 6] =
    [0.000247, 0.0013845, 0.001222, 0.0026455, 0.0008320, 0.0001690];
pub const DELAYED_NEUTRON_DECAY_CONSTANTS: [f64; 6] = [0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01];

/// Prompt neutron generation time, in seconds. ~2e-5s (20 microseconds) is
/// a typical value for a thermal PWR. This is what makes the "prompt"
/// part of the response so fast compared to the delayed-neutron terms
/// above (which act on timescales of tenths of a second to tens of
/// seconds).
pub const PROMPT_GENERATION_TIME: f64 = 2.0e-5;

/// A function of time returning reactivity (in absolute delta-k/k, not
/// pcm or dollars) at that instant. Using a boxed closure (`Box<dyn Fn(f64)
/// -> f64>`) rather than a fixed enum of "step" or "ramp" variants keeps
/// the simulation function itself agnostic to what SHAPE of transient it's
/// being asked to run. `simulate_kinetics.rs` builds whichever closure it
/// needs (step, ramp, or anything else) and hands it in.
///
/// `dyn Fn(f64) -> f64` is a "trait object". Rather than the compiler
/// generating a specialized copy of every function that takes a
/// `ReactivityFn` for each concrete closure type (which is what happens
/// with `impl Fn(f64) -> f64` parameters elsewhere in this project), a
/// `Box<dyn Fn>` stores a pointer to the closure's code and data together
/// on the heap, letting `simulate` below accept ANY closure with the right
/// signature at runtime, decided by the caller. This is the right tool
/// here specifically because the closure gets passed through and called
/// repeatedly across the whole simulation, not just used once inline.
pub type ReactivityFn = Box<dyn Fn(f64) -> f64>;

/// The full ODE state at one instant: relative neutron population
/// (`n`, where `n = 1.0` means "at the power level we started the
/// simulation at") plus each delayed-neutron precursor group's relative
/// concentration.
#[derive(Debug, Clone, Copy)]
pub struct KineticsState {
    pub n: f64,
    pub precursors: [f64; 6],
}

impl KineticsState {
    /// The equilibrium (steady-state) starting condition for a reactor
    /// sitting at constant relative power `n0` with zero net reactivity.
    /// At equilibrium, each precursor group's production
    /// (`beta_i / Lambda * n`) exactly balances its own decay
    /// (`lambda_i * C_i`), which rearranges to
    /// `C_i = (beta_i / (Lambda * lambda_i)) * n`. Starting a simulation
    /// from anything OTHER than this equilibrium would show an artificial
    /// transient in the first few seconds that has nothing to do with the
    /// reactivity insertion being studied, so getting this right matters.
    pub fn equilibrium(n0: f64) -> Self {
        let mut precursors = [0.0; 6];
        for i in 0..6 {
            precursors[i] = (DELAYED_NEUTRON_FRACTIONS[i]
                / (PROMPT_GENERATION_TIME * DELAYED_NEUTRON_DECAY_CONSTANTS[i]))
                * n0;
        }
        KineticsState { n: n0, precursors }
    }
}

/// The point kinetics equations themselves: given the current state and
/// reactivity, what's the instantaneous rate of change of every quantity?
///
///   dn/dt      = ((rho - beta) / Lambda) * n + sum_i(lambda_i * C_i)
///   dC_i/dt    = (beta_i / Lambda) * n - lambda_i * C_i
///
/// `beta` (with no subscript) is the TOTAL delayed neutron fraction, the
/// sum of all six `beta_i`. The first term of `dn/dt` says prompt neutrons
/// alone would make power grow (or shrink) at a rate set by how far
/// reactivity `rho` is from the delayed fraction `beta`. This is why
/// keeping `rho` under `beta` matters enormously in real operations
/// (a reactor goes "prompt critical" the moment `rho` exceeds `beta`, at
/// which point it would no longer need delayed neutrons to sustain the
/// chain reaction at all, and the whole point of this six-group model,
/// that power changes on a controllable delayed-neutron timescale, breaks
/// down). The second term says precursor decay keeps steadily feeding
/// neutrons back in regardless of what the prompt term is doing right now.
/// That's the delayed, stabilizing contribution.
fn derivatives(state: &KineticsState, rho: f64) -> KineticsState {
    let beta: f64 = DELAYED_NEUTRON_FRACTIONS.iter().sum();

    let mut precursor_source: f64 = 0.0; // sum_i(lambda_i * C_i)
    for i in 0..6 {
        precursor_source += DELAYED_NEUTRON_DECAY_CONSTANTS[i] * state.precursors[i];
    }

    let dn_dt = ((rho - beta) / PROMPT_GENERATION_TIME) * state.n + precursor_source;

    let mut d_precursors = [0.0; 6];
    for i in 0..6 {
        d_precursors[i] = (DELAYED_NEUTRON_FRACTIONS[i] / PROMPT_GENERATION_TIME) * state.n
            - DELAYED_NEUTRON_DECAY_CONSTANTS[i] * state.precursors[i];
    }

    KineticsState { n: dn_dt, precursors: d_precursors }
}

/// One 4th-order Runge-Kutta ("RK4") integration step, advancing `state`
/// forward by `dt` seconds under reactivity `rho` (held constant across
/// this one step, the caller is responsible for calling this repeatedly
/// with small enough `dt` that reactivity looks approximately constant
/// within each step, which `simulate` below does automatically).
///
/// RK4 is a workhorse method for exactly this kind of "stiff-ish" ODE
/// system (the prompt term's timescale, microseconds, is many orders of
/// magnitude faster than the delayed terms', seconds). It's dramatically
/// more accurate per step than the simplest method (Euler's, which just
/// walks forward using the derivative at the CURRENT point) because it
/// samples the derivative at four points across the step (the start, two
/// estimates at the midpoint, and the end) and combines them with weights
/// chosen so the error per step shrinks much faster as `dt` gets smaller.
fn rk4_step(state: &KineticsState, rho: f64, dt: f64) -> KineticsState {
    let k1 = derivatives(state, rho);
    let k2 = derivatives(&add_scaled(state, &k1, dt / 2.0), rho);
    let k3 = derivatives(&add_scaled(state, &k2, dt / 2.0), rho);
    let k4 = derivatives(&add_scaled(state, &k3, dt), rho);

    // The classic RK4 combination: weight the four slope estimates
    // 1:2:2:1 (giving the two midpoint estimates double weight, since
    // they're the most representative of the step's average behavior),
    // then advance by that weighted-average slope times the step size.
    let mut result = *state;
    result.n += (dt / 6.0) * (k1.n + 2.0 * k2.n + 2.0 * k3.n + k4.n);
    for i in 0..6 {
        result.precursors[i] += (dt / 6.0)
            * (k1.precursors[i] + 2.0 * k2.precursors[i] + 2.0 * k3.precursors[i] + k4.precursors[i]);
    }
    result
}

/// Helper for RK4's intermediate evaluations: `state + derivative * scale`,
/// applied to every field. A small arithmetic helper kept separate mainly
/// so `rk4_step` itself reads as "the four k-estimates, then the weighted
/// combination" without this scaling arithmetic cluttering that shape.
fn add_scaled(state: &KineticsState, derivative: &KineticsState, scale: f64) -> KineticsState {
    let mut result = *state;
    result.n += derivative.n * scale;
    for i in 0..6 {
        result.precursors[i] += derivative.precursors[i] * scale;
    }
    result
}

/// Run a full simulation from `t = 0` to `t = t_end`, in fixed steps of
/// `dt` seconds, under the reactivity history described by `reactivity`.
/// Returns `(time, relative_power)` pairs, one per step.
///
/// `dt` needs to be small relative to the FASTEST timescale in the system
/// to keep RK4 numerically stable. Here, that's the prompt-term response,
/// not the (much slower) delayed-neutron decay constants. `1e-3` seconds
/// (1 millisecond) is comfortably small enough for the reactivity
/// insertions this project simulates; a much larger insertion approaching
/// prompt criticality would need an even smaller step.
pub fn simulate(reactivity: ReactivityFn, t_end: f64, dt: f64) -> Vec<(f64, f64)> {
    let mut state = KineticsState::equilibrium(1.0);
    let mut t = 0.0;
    let mut history = Vec::new();
    history.push((t, state.n));

    while t < t_end {
        let rho = reactivity(t);
        state = rk4_step(&state, rho, dt);
        t += dt;
        history.push((t, state.n));
    }

    history
}
