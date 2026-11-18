//! Projected Average Force Integration (PAFI) along minimum-energy paths.
//!
//! Integrates the curvature-corrected hyperplane mean force
//! `<(grad V · n)(1 - psi)>` across planes orthogonal to a converged band or
//! spline reaction path to obtain finite-temperature free-energy barriers
//! without a predefined collective variable.
//!
//! Note: implementation moves from rgsaddle.
//!
//! ## References
//!
//! - Goswami, R., Gunde, M., and Jónsson, H. (2026). *Enhanced climbing image
//!   nudged elastic band method with Hessian eigenmode alignment*. Frontiers
//!   in Chemistry, 14, 1807063. <https://doi.org/10.3389/fchem.2026.1807063>
//! - Goswami, R. (2026). *Reproducible orchestration of best practices for
//!   reaction path optimization with the nudged elastic band*. MethodsX, 16,
//!   103899. <https://doi.org/10.1016/j.mex.2026.103899>
//! - Swinburne, T. D., and Marinica, M.-C. (2018). *Unsupervised Calculation
//!   of Free Energy Barriers in Large Crystalline Systems*. Physical Review
//!   Letters, 120, 135503. <https://doi.org/10.1103/PhysRevLett.120.135503>

use ndarray::ArrayView2;

use crate::engine::DynamicsEngine;
use crate::error::FluxError;
use crate::stats::StatsConfig;

/// Configuration for Projected Average Force Integration (PAFI).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct PafiConfig {
    /// Number of hyperplanes discretized along the reaction path (`>= 2`).
    pub planes: usize,
    /// Equilibration steps discarded on each hyperplane.
    pub burn_in_steps: usize,
    /// Production steps recorded per hyperplane per round.
    pub sample_steps: usize,
    /// Integration timestep `dt > 0`.
    pub time_step: f64,
    /// Optional target standard error on the integrated barrier `Delta F`.
    pub target_error: Option<f64>,
    /// Maximum adaptive resampling rounds when `target_error` is set.
    pub max_rounds: usize,
    /// Tolerance on the zero-temperature path-force integral vs `Delta E_0`.
    pub zero_temperature_tol: f64,
    /// Deterministic PRNG seed.
    pub seed: u64,
    /// Batch-means uncertainty settings for per-plane forces.
    pub stats: StatsConfig,
}

impl Default for PafiConfig {
    fn default() -> Self {
        Self {
            planes: 9,
            burn_in_steps: 500,
            sample_steps: 4_000,
            time_step: 0.002,
            target_error: None,
            max_rounds: 4,
            zero_temperature_tol: 0.01,
            seed: 1,
            stats: StatsConfig::default(),
        }
    }
}

with_setters!(PafiConfig {
    with_planes => planes: val usize,
    with_burn_in_steps => burn_in_steps: val usize,
    with_sample_steps => sample_steps: val usize,
    with_time_step => time_step: val f64,
    with_target_error => target_error: opt f64,
    with_max_rounds => max_rounds: val usize,
    with_zero_temperature_tol => zero_temperature_tol: val f64,
    with_seed => seed: val u64,
    with_stats => stats: val StatsConfig,
});

/// Integrated free-energy profile along a discretized reaction path.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct FreeEnergyProfile {
    /// Normalized arc-length coordinates `s_k` along the path.
    pub arc_lengths: Vec<f64>,
    /// Integrated free energy `F(s_k) - F(s_0)` at each plane.
    pub free_energy: Vec<f64>,
    /// Propagated block-averaged standard error at each plane.
    pub standard_error: Vec<f64>,
    /// Maximum free-energy barrier `Delta F` along the profile.
    pub barrier: f64,
    /// Standard error of [`Self::barrier`].
    pub barrier_error: f64,
}

/// Integrate the PAFI free-energy profile along `path_images` (implementation moves from rgsaddle).
pub fn pafi_profile<E: DynamicsEngine + ?Sized>(
    _engine: &mut E,
    _path_images: ArrayView2<'_, f64>,
    _config: &PafiConfig,
) -> Result<FreeEnergyProfile, FluxError> {
    Err(FluxError::Unimplemented(
        "pafi: implementation moves from rgsaddle",
    ))
}
