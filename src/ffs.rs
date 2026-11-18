//! Forward Flux Sampling (FFS) across ordered reaction-coordinate interfaces.
//!
//! Computes rare-event transition rates `k_AB = Phi_A0 * prod_i P(lambda_{i+1} | lambda_i)`
//! in equilibrium or non-equilibrium stochastic dynamics without requiring
//! detailed balance or knowledge of the stationary phase-space density.
//!
//! ## References
//!
//! - Allen, R. J., Frenkel, D., and ten Wolde, P. R. (2006). *Simulating rare
//!   events in equilibrium or nonequilibrium stochastic systems*. The Journal
//!   of Chemical Physics, 124, 024102. <https://doi.org/10.1063/1.2140273>
//! - Allen, R. J., Frenkel, D., and ten Wolde, P. R. (2006). *Forward flux
//!   sampling-type schemes for simulating rare events: Efficiency analysis*.
//!   The Journal of Chemical Physics, 124, 194111.
//!   <https://doi.org/10.1063/1.2198827>
//! - Allen, R. J., Warren, P. B., and ten Wolde, P. R. (2005). *Sampling Rare
//!   Switching Events in Biochemical Networks*. Physical Review Letters, 94,
//!   018104. <https://doi.org/10.1103/PhysRevLett.94.018104>
//! - Allen, R. J., Valeriani, C., and ten Wolde, P. R. (2009). *Forward flux
//!   sampling for rare event simulations*. Journal of Physics: Condensed
//!   Matter, 21, 463102. <https://doi.org/10.1088/0953-8984/21/46/463102>

use ndarray::ArrayView1;

use crate::engine::DynamicsEngine;
use crate::error::FluxError;
use crate::stats::StatsConfig;

/// Configuration for direct Forward Flux Sampling across ordered interfaces.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct FfsConfig {
    /// Strictly monotone interface levels `lambda_0 < lambda_1 < ... < lambda_n`.
    pub interfaces: Vec<f64>,
    /// Target crossing configurations collected at each interface.
    pub points_per_interface: usize,
    /// Trial trajectories launched per interface to estimate `P(lambda_{i+1} | lambda_i)`.
    pub trials_per_interface: usize,
    /// Maximum integration steps per trial trajectory before aborting.
    pub max_trial_steps: usize,
    /// Integration timestep `dt > 0`.
    pub time_step: f64,
    /// Deterministic PRNG seed.
    pub seed: u64,
    /// Batch-means controls for the initial escape flux `Phi_{A,0}`.
    pub stats: StatsConfig,
}

impl Default for FfsConfig {
    fn default() -> Self {
        Self {
            interfaces: vec![-0.5, 0.0, 0.5],
            points_per_interface: 128,
            trials_per_interface: 256,
            max_trial_steps: 10_000,
            time_step: 0.005,
            seed: 1,
            stats: StatsConfig::default(),
        }
    }
}

with_setters!(FfsConfig {
    with_interfaces => interfaces: val Vec<f64>,
    with_points_per_interface => points_per_interface: val usize,
    with_trials_per_interface => trials_per_interface: val usize,
    with_max_trial_steps => max_trial_steps: val usize,
    with_time_step => time_step: val f64,
    with_seed => seed: val u64,
    with_stats => stats: val StatsConfig,
});

/// Forward Flux Sampling rate and per-interface conditional advancement probabilities.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct FfsEstimate {
    /// Initial positive crossing flux `Phi_{A,0}` out of basin `A` across `lambda_0`.
    pub initial_flux: f64,
    /// Conditional probabilities `P(lambda_{i+1} | lambda_i)` across consecutive interfaces.
    pub interface_probabilities: Vec<f64>,
    /// Composite forward transition rate `k_{AB}`.
    pub rate: f64,
    /// Propagated standard error on [`Self::rate`].
    pub standard_error: f64,
}

/// Run Forward Flux Sampling on `engine` using order parameter `order_param`.
pub fn forward_flux_rate<E, F>(
    _engine: &mut E,
    _order_param: F,
    _config: &FfsConfig,
) -> Result<FfsEstimate, FluxError>
where
    E: DynamicsEngine + ?Sized,
    F: Fn(ArrayView1<'_, f64>) -> f64,
{
    Err(FluxError::Unimplemented(
        "ffs: forward flux sampler awaits wiring",
    ))
}
