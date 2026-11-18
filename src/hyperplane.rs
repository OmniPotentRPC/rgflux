//! Constrained hyperplane sampling, mean force, torque, and control variates.
//!
//! Samples configurations restricted to a hyperplane `n · x - c = 0` through
//! [`crate::engine::DynamicsEngine`] to estimate the hyperplane free-energy
//! derivatives `dA/dc` and `dA/dn` along with in-plane virial and zero-variance
//! harmonic control variates.
//!
//! Note: implementation moves from rgsaddle.
//!
//! ## References
//!
//! - Sprik, M., and Ciccotti, G. (1998). *Free energy from constrained
//!   molecular dynamics*. The Journal of Chemical Physics, 109, 7737–7744.
//!   <https://doi.org/10.1063/1.477419>
//! - Jóhannesson, G. H., and Jónsson, H. (2001). *Optimization of hyperplanar
//!   transition states*. The Journal of Chemical Physics, 115, 9644–9656.
//!   <https://doi.org/10.1063/1.1415499>
//! - Leimkuhler, B., and Matthews, C. (2012). *Rational Construction of
//!   Stochastic Numerical Methods for Molecular Sampling*. Applied Mathematics
//!   Research eXpress. <https://doi.org/10.1093/amrx/abs010>
//! - Leimkuhler, B., and Matthews, C. (2016). *Efficient molecular dynamics
//!   using geodesic integration and solvent–solute splitting*. Proceedings of
//!   the Royal Society A, 472, 20160138. <https://doi.org/10.1098/rspa.2016.0138>
//! - Assaraf, R., and Caffarel, M. (1999). *Zero-Variance Principle for Monte
//!   Carlo Algorithms*. Physical Review Letters, 83, 4682–4685.
//!   <https://doi.org/10.1103/PhysRevLett.83.4682>

pub use crate::engine::HyperplaneConstraint;

use ndarray::Array2;

use crate::engine::DynamicsEngine;
use crate::error::FluxError;
use crate::stats::StatsConfig;

/// Configuration for sampling free-energy gradients and configurations on a hyperplane.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct HyperplaneConfig {
    /// Equilibration steps discarded before recording observables.
    pub burn_in_steps: usize,
    /// Production steps integrated on the hyperplane.
    pub sample_steps: usize,
    /// Steps between retained dividing-surface configurations (`>= 1`).
    pub retention_stride: usize,
    /// Integration timestep `dt > 0`.
    pub time_step: f64,
    /// Deterministic PRNG seed.
    pub seed: u64,
    /// Batch-means and stationarity controls for the sampled mean force.
    pub stats: StatsConfig,
}

impl Default for HyperplaneConfig {
    fn default() -> Self {
        Self {
            burn_in_steps: 500,
            sample_steps: 4_000,
            retention_stride: 10,
            time_step: 0.005,
            seed: 1,
            stats: StatsConfig::default(),
        }
    }
}

with_setters!(HyperplaneConfig {
    with_burn_in_steps => burn_in_steps: val usize,
    with_sample_steps => sample_steps: val usize,
    with_retention_stride => retention_stride: val usize,
    with_time_step => time_step: val f64,
    with_seed => seed: val u64,
    with_stats => stats: val StatsConfig,
});

/// Summary of a constrained hyperplane sampling trajectory.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct HyperplaneSample {
    /// Sampled hyperplane definition.
    pub plane: HyperplaneConstraint,
    /// Mean holonomic constraint force `<lambda> = dA/dc`.
    pub mean_force: f64,
    /// Block-averaged standard error of [`Self::mean_force`].
    pub standard_error: f64,
    /// Retained configurations on the hyperplane (`(n_retained, dof)`).
    pub configurations: Array2<f64>,
}

/// Sample a hyperplane through `engine` (implementation moves from rgsaddle).
pub fn sample_hyperplane<E: DynamicsEngine + ?Sized>(
    _engine: &mut E,
    _plane: &HyperplaneConstraint,
    _config: &HyperplaneConfig,
) -> Result<HyperplaneSample, FluxError> {
    Err(FluxError::Unimplemented(
        "hyperplane: implementation moves from rgsaddle",
    ))
}
