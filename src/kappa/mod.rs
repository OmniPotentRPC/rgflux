//! Reactive-flux transmission coefficients (`kappa`) across dividing surfaces.
//!
//! Transition state theory overcounts the reactive rate whenever trajectories
//! recross the dividing surface before thermalizing in a basin. This module
//! houses the complete [`bennett_chandler`] estimator (both engine-free over
//! recorded side histories and generic over [`crate::engine::DynamicsEngine`]),
//! along with Kramers' analytical barrier-recrossing factor.
//!
//! Note: steepest-descent quench basin classifiers and van Erp first-crossing
//! effective positive flux estimators are not yet implemented.
//!
//! ## Example
//!
//! ```rust
//! use rgflux::kappa::kramers_transmission_coefficient;
//!
//! // At gamma = 2 * omega_b (alpha = 1), kappa_K = sqrt(2) - 1.
//! let kappa = kramers_transmission_coefficient(4.0, 2.0)?;
//! assert!((kappa - (2.0_f64.sqrt() - 1.0)).abs() <= 1e-15);
//! # Ok::<(), rgflux::FluxError>(())
//! ```
//!
//! ## References
//!
//! - Goswami, R. (2025). *Efficient Exploration of Chemical Kinetics*.
//!   arXiv:2510.21368. <https://doi.org/10.48550/arXiv.2510.21368>
//! - Goswami, R., and Jónsson, H. (2026). *Adaptive Pruning for Increased
//!   Robustness and Reduced Computational Overhead in Gaussian Process
//!   Accelerated Saddle Point Searches*. ChemPhysChem, 27, e202500730.
//!   <https://doi.org/10.1002/cphc.202500730>
//! - Bennett, C. H. (1977). *Molecular Dynamics and Transition State Theory:
//!   The Simulation of Infrequent Events*. In *Algorithms for Chemical
//!   Computations*, ACS Symposium Series, 46, 63-97.
//!   <https://doi.org/10.1021/bk-1977-0046.ch004>
//! - Chandler, D. (1978). *Statistical mechanics of isomerization dynamics in
//!   liquids and the transition state approximation*. The Journal of Chemical
//!   Physics, 68, 2959-2970. <https://doi.org/10.1063/1.436049>
//! - Kramers, H. A. (1940). *Brownian motion in a field of force and the
//!   diffusion model of chemical reactions*. Physica, 7, 284-304.
//!   <https://doi.org/10.1016/S0031-8914(40)90098-2>
//! - Anderson, J. B. (1973). *Statistical theories of chemical reactions.
//!   Distributions in the transition region*. The Journal of Chemical Physics,
//!   58, 4684-4692. <https://doi.org/10.1063/1.1679032>
//! - van Erp, T. S., Moroni, D., and Bolhuis, P. G. (2003). *A novel path
//!   sampling method for the calculation of rate constants*. The Journal of
//!   Chemical Physics, 118, 7762-7774. <https://doi.org/10.1063/1.1562614>

pub mod bennett_chandler;

pub use bennett_chandler::{
    BennettChandlerConfig, BennettChandlerEstimate, ReactiveTrajectory, ShootingConfig, Side,
    estimate_bennett_chandler, shoot_bennett_chandler, shoot_reactive_trajectories,
};

use crate::error::FluxError;

/// Default consecutive basin-commitment checkpoints in [`KappaConfig`].
pub const DEFAULT_COMMIT_CHECKPOINTS: usize = 4;

/// Configuration for basin-commitment stopping and surface-quench / first-crossing
/// transmission workflows.
///
/// [`shoot_reactive_trajectories`] and [`shoot_bennett_chandler`] read
/// [`Self::commit_checkpoints`] through [`ShootingConfig::kappa`] (or
/// [`ShootingConfig::with_commit_checkpoints`]). Steepest-descent endpoint
/// quenching is not yet implemented, so [`Self::quench_force_tol`] and
/// [`Self::quench_max_steps`] are not read yet.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct KappaConfig {
    /// Consecutive basin-commitment checkpoints on a committed side (`s < 0` or `s > 0`)
    /// required to stop a half-trajectory and copy that side into every later
    /// checkpoint of `kappa(t)` (`<= steps`; `0` runs every half to `steps`).
    ///
    /// The window must outlast the time a trajectory lingers at the barrier top:
    /// a shorter window commits trajectories that recross later and raises `kappa`.
    ///
    /// Default: [`DEFAULT_COMMIT_CHECKPOINTS`] (`4`).
    pub commit_checkpoints: usize,
    /// Steepest-descent force tolerance when assigning end states by quenching
    /// (not yet implemented).
    ///
    /// Default: `0.001`.
    pub quench_force_tol: f64,
    /// Maximum quench steps per trajectory endpoint (not yet implemented).
    ///
    /// Default: `2_000`.
    pub quench_max_steps: usize,
}

impl Default for KappaConfig {
    fn default() -> Self {
        Self {
            commit_checkpoints: DEFAULT_COMMIT_CHECKPOINTS,
            quench_force_tol: 0.001,
            quench_max_steps: 2_000,
        }
    }
}

with_setters!(KappaConfig {
    with_commit_checkpoints => commit_checkpoints: val usize,
    with_quench_force_tol => quench_force_tol: val f64,
    with_quench_max_steps => quench_max_steps: val usize,
});

/// Kramers spatial-diffusion transmission coefficient across a parabolic barrier
/// of unstable angular frequency `barrier_frequency` (`omega_b > 0`) under
/// Langevin friction `gamma >= 0` (Kramers 1940):
///
/// `kappa_K = sqrt(1 + alpha^2) - alpha = 1 / (sqrt(1 + alpha^2) + alpha)`
/// where `alpha = gamma / (2 * omega_b)`.
pub fn kramers_transmission_coefficient(
    gamma: f64,
    barrier_frequency: f64,
) -> Result<f64, FluxError> {
    if !gamma.is_finite() || !barrier_frequency.is_finite() {
        return Err(FluxError::NonFinite("kramers_transmission_coefficient"));
    }
    if gamma < 0.0 {
        return Err(FluxError::InvalidConfig(
            "Langevin friction gamma must be non-negative",
        ));
    }
    if barrier_frequency <= 0.0 {
        return Err(FluxError::InvalidConfig(
            "barrier_frequency omega_b must be strictly positive",
        ));
    }
    let alpha = gamma / (2.0 * barrier_frequency);
    Ok(1.0 / (1.0_f64.hypot(alpha) + alpha))
}
