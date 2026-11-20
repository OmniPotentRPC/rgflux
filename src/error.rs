//! Typed errors shared across rare-event kinetics estimators and dynamics engines.

use thiserror::Error;

/// Error returned by kinetics estimators, statistical diagnostics, and dynamics engines.
#[derive(Clone, Debug, PartialEq, Error)]
#[non_exhaustive]
pub enum FluxError {
    /// An input slice, trajectory collection, or coordinate vector was empty.
    #[error("empty input: {0}")]
    EmptyInput(&'static str),

    /// No reactive-flux trajectory had a strictly positive initial normal velocity.
    #[error("reactive-flux ensemble contains no positive-flux trajectories")]
    NoPositiveFlux,

    /// The candidate plateau window failed the stationarity diagnostic.
    #[error(
        "non-stationary plateau: |z| = {z_score:.4} exceeds limit {max_z_score:.4} (drift slope {drift_slope:.4e})"
    )]
    NonStationaryPlateau {
        /// Observed two-window stationarity z-score on the plateau window.
        z_score: f64,
        /// Configured maximum allowed stationarity z-score.
        max_z_score: f64,
        /// Ordinary-least-squares per-step linear drift slope across the window.
        drift_slope: f64,
    },

    /// Array or trajectory dimensions did not agree.
    #[error("shape mismatch in {context}: expected {expected}, got {got}")]
    Shape {
        /// Expected length.
        expected: usize,
        /// Observed length.
        got: usize,
        /// Context naming the mismatched quantity.
        context: &'static str,
    },

    /// A floating-point input or intermediate quantity was NaN or infinite.
    #[error("non-finite value in {0}")]
    NonFinite(&'static str),

    /// A configuration field or physical parameter violated its domain invariant.
    #[error("invalid configuration: {0}")]
    InvalidConfig(&'static str),

    /// A numerically probed Cartesian translation could not be classified.
    #[error(
        "PAFI reactant translation probe is ambiguous on the {axis} axis by {ambiguity:.4e}: response {response:.4e}, finite-difference uncertainty {uncertainty:.4e}, roundoff floor {roundoff_floor:.4e}; set PafiConfig::translation_invariance explicitly"
    )]
    AmbiguousTranslationMode {
        /// Cartesian axis (`"x"`, `"y"`, or `"z"`).
        axis: &'static str,
        /// Norm of the estimated mass-weighted Hessian response.
        response: f64,
        /// Estimated truncation uncertainty of that response.
        uncertainty: f64,
        /// Estimated floating-point roundoff floor.
        roundoff_floor: f64,
        /// Width by which the response uncertainty interval overlaps the floor.
        ambiguity: f64,
    },

    /// A holonomic hyperplane constraint residual exceeded its numerical tolerance.
    #[error(
        "holonomic hyperplane constraint residual {residual:.4e} exceeds tolerance {tolerance:.4e}"
    )]
    ConstraintViolation {
        /// Observed relative residual `|n · x - c| / (1 + |c| + ||x||_2)`.
        residual: f64,
        /// Configured or machine-derived relative tolerance.
        tolerance: f64,
    },

    /// Dividing-surface velocity rejection sampling exhausted its redraw budget.
    #[error("dividing-surface velocity rejection exceeded {max_attempts} attempts")]
    SurfaceRejection {
        /// Maximum redraw attempts configured for the dividing surface.
        max_attempts: usize,
    },

    /// Evaluating a potential energy surface or force callback failed.
    #[error("surface evaluation failed: {0}")]
    Surface(String),

    /// A feature-gated external C ABI dynamics engine stub has no linked runtime session.
    #[error("dynamics backend unavailable: {0}")]
    BackendUnavailable(&'static str),

    /// Error returned by the optional lammpc C ABI backend.
    #[error("lammpc backend error: {0}")]
    Lammpc(String),

    /// A public entry point that is not yet implemented was called.
    #[error("not yet implemented: {0}")]
    Unimplemented(&'static str),
}

impl From<rgsaddle::SaddleError> for FluxError {
    fn from(err: rgsaddle::SaddleError) -> Self {
        Self::Surface(err.to_string())
    }
}
