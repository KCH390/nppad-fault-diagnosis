//! An LSTM-based multi-class accident classifier, built on `candle-nn`'s
//! LSTM layer plus a final linear (fully-connected) layer.
//!
//! WHAT THIS FILE DOES, IN PLAIN LANGUAGE:
//! Takes a whole run's sequence of sensor readings (from
//! `core::sequences`) and predicts one accident-type label for the whole
//! sequence, "many-to-one" in RNN terminology. The LSTM reads the sequence
//! one timestep at a time, updating an internal "hidden state" (its
//! working memory of everything relevant it's seen so far) and a separate
//! "cell state" (a more protected long-term memory, gated so the network
//! can choose what to keep, forget, or write at every step, which is
//! exactly what makes an LSTM able to learn from PATTERNS OVER TIME
//! instead of being told which summary statistics to compute, unlike
//! Phases 2-4's hand-engineered mean/std/slope window features). After
//! reading the whole sequence, only the FINAL hidden state gets passed to
//! a linear layer that maps it down to one score per accident type
//! ("logits"), which softmax + cross-entropy loss then trains against the
//! true label during training, or `argmax` picks the winner from during
//! prediction.
//!
//! WHY AN LSTM RATHER THAN A PLAIN RNN: a plain recurrent network's
//! hidden state gets multiplied by the same weight matrix at every
//! timestep, so gradients flowing backward through a long sequence during
//! training either shrink toward zero or blow up exponentially (the
//! "vanishing/exploding gradient" problem), making it very hard to learn
//! dependencies more than a few timesteps back. An LSTM's separate cell
//! state has a more direct, mostly-additive path across timesteps (gated
//! by "forget" and "input" gates rather than repeatedly matrix-multiplied
//! like a plain RNN's hidden state), which is specifically what lets it
//! learn longer-range temporal patterns without that same gradient decay.
//! For NPPAD sequences (tens to low hundreds of timesteps after Phase 2's
//! kind of windowing, or `core::sequences`' fixed `seq_len` here), that
//! matters more than it would for a handful of timesteps.

use candle_core::{Result, Tensor};
use candle_nn::rnn::{LSTMConfig, RNN};
use candle_nn::{linear, Linear, Module, VarBuilder};

pub struct LstmClassifier {
    lstm: candle_nn::rnn::LSTM,
    output: Linear,
}

impl LstmClassifier {
    /// Build a new (untrained) classifier. `input_dim` is the number of
    /// sensors per timestep (96, the canonical NPPAD schema from
    /// `core::data`), `hidden_dim` is the LSTM's internal memory size (a
    /// hyperparameter, bigger means more capacity to learn but slower and
    /// more prone to overfitting on a dataset this size), and `n_classes`
    /// is 18, the number of accident types.
    ///
    /// `VarBuilder` is candle's mechanism for creating and naming a
    /// model's trainable parameters (weights and biases) so they can be
    /// tracked by a `VarMap` for optimization and later saved/loaded by
    /// name. `vb.pp("lstm")` ("push prefix") namespaces every parameter
    /// the LSTM layer creates under an "lstm." prefix, so this model's
    /// full parameter list stays organized (e.g. "lstm.weight_ih_l0",
    /// "output.weight") even though `LstmClassifier` itself has no
    /// parameters of its own outside its two sub-layers.
    pub fn new(input_dim: usize, hidden_dim: usize, n_classes: usize, vb: VarBuilder) -> Result<Self> {
        let lstm_config = LSTMConfig::default();
        let lstm = candle_nn::rnn::lstm(input_dim, hidden_dim, lstm_config, vb.pp("lstm"))?;
        let output = linear(hidden_dim, n_classes, vb.pp("output"))?;
        Ok(Self { lstm, output })
    }

    /// Run the model forward on a batch of sequences, returning raw class
    /// scores ("logits", pre-softmax) of shape `(batch, n_classes)`.
    ///
    /// `xs` is expected in `(batch, seq_len, input_dim)` layout
    /// (batch-first). Confirmed against a real run: this project's
    /// installed `candle-nn` version's `RNN::seq()` reads dimension 0 as
    /// batch, not time, so batch-first is the layout that actually
    /// matches what `.seq()` does here.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // `.seq(xs)` runs the LSTM across every timestep in one call,
        // returning one `LSTMState` (hidden + cell state) PER timestep —
        // the RNN trait's default-provided convenience over manually
        // looping `.step()` yourself once per timestep.
        let states = self.lstm.seq(xs)?;

        // We only want the LAST timestep's hidden state — everything the
        // network decided was worth remembering by the time it finished
        // reading the whole sequence. `states` is a plain `Vec`, so
        // `.last()` is the standard Rust way to grab that final element
        // (returning `Option<&LSTMState>`, hence the `.expect(...)` since
        // a non-empty input sequence guarantees at least one state).
        let last_state = states.last().expect("sequence must have at least one timestep");
        let hidden = last_state.h();

        self.output.forward(hidden)
    }
}
