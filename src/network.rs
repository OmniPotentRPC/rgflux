//! Rate networks, symmetry-resolved defect graphs, and kinetic Monte Carlo rate tables.
//!
//! Organizes elementary transition rates `k_{i -> j}(T)` into master-equation
//! graphs, first-passage / absorbing-boundary reductions, and BKL / Gillespie
//! kinetic Monte Carlo (kMC) event tables.
//!
//! Note: implementation moves from rgsaddle.
//!
//! ## Example
//!
//! ```rust
//! use rgflux::FluxError;
//! use rgflux::network::{NetworkConfig, RateEdge, RateTableConfig, build_rate_table};
//!
//! let edge = RateEdge::new(0, 1, 2.5e9)?;
//! assert_eq!(edge.from, 0);
//! assert_eq!(edge.to, 1);
//! assert_eq!(edge.rate, 2.5e9);
//!
//! let net_cfg = NetworkConfig::default().with_temperature(500.0).with_min_rate(1e-6);
//! let kmc_cfg = RateTableConfig::default().with_max_steps(10_000).with_seed(42);
//! assert_eq!(kmc_cfg.max_steps, 10_000);
//! assert!(matches!(
//!     build_rate_table(2, &[edge], &net_cfg),
//!     Err(FluxError::Unimplemented(_))
//! ));
//! # Ok::<(), rgflux::FluxError>(())
//! ```
//!
//! ## References
//!
//! - Goswami, R. (2025). *Efficient Exploration of Chemical Kinetics*.
//!   arXiv:2510.21368. <https://doi.org/10.48550/arXiv.2510.21368>
//! - Goswami, R. (2025). *Managing concurrency for high throughput adaptive
//!   kinetic monte carlo*. ACS SciMeetings.
//!   <https://doi.org/10.1021/scimeetings.5c11344>
//! - Goswami, R., et al. (2026). *TheochemUI/eOn: v2.9.0.1*. Zenodo.
//!   <https://doi.org/10.5281/zenodo.18529496>
//! - Bortz, A. B., Kalos, M. H., and Lebowitz, J. L. (1975). *A new algorithm
//!   for Monte Carlo simulation of Ising spin systems*. Journal of
//!   Computational Physics, 17, 10-18.
//!   <https://doi.org/10.1016/0021-9991(75)90060-1>
//! - Gillespie, D. T. (1976). *A general method for numerically simulating the
//!   stochastic time evolution of coupled chemical reactions*. Journal of
//!   Computational Physics, 22, 403-434.
//!   <https://doi.org/10.1016/0021-9991(76)90041-3>
//! - Henkelman, G., and Jónsson, H. (2001). *Long time scale kinetic Monte
//!   Carlo simulations without lattice approximation and predefined event
//!   table*. The Journal of Chemical Physics, 115, 9657-9666.
//!   <https://doi.org/10.1063/1.1415500>
//! - Vineyard, G. H. (1957). *Frequency factors and isotope effects in solid
//!   state rate processes*. Journal of Physics and Chemistry of Solids, 3,
//!   121-127. <https://doi.org/10.1016/0022-3697(57)90059-8>

use crate::error::FluxError;

/// Configuration for assembling a master-equation rate network.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct NetworkConfig {
    /// Absolute temperature `T > 0`.
    ///
    /// Default: `300.0`.
    pub temperature: f64,
    /// Boltzmann constant `k_B > 0` in the barrier's energy-per-temperature unit.
    ///
    /// Default: `1.0`.
    pub boltzmann: f64,
    /// Enforce detailed balance across reverse edges when state free energies are supplied.
    ///
    /// Default: `true`.
    pub enforce_detailed_balance: bool,
    /// Optional minimum rate floor below which negligible channels are pruned.
    ///
    /// Default: `None`.
    pub min_rate: Option<f64>,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            temperature: 300.0,
            boltzmann: 1.0,
            enforce_detailed_balance: true,
            min_rate: None,
        }
    }
}

with_setters!(NetworkConfig {
    with_temperature => temperature: val f64,
    with_boltzmann => boltzmann: val f64,
    with_enforce_detailed_balance => enforce_detailed_balance: val bool,
    with_min_rate => min_rate: opt f64,
});

/// Configuration for kinetic Monte Carlo (kMC) residence-time rate tables.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct RateTableConfig {
    /// Maximum kMC steps per trajectory (`>= 1`).
    ///
    /// Default: `100_000`.
    pub max_steps: usize,
    /// Deterministic PRNG seed for event selection and residence-time draws.
    ///
    /// Default: `1`.
    pub seed: u64,
}

impl Default for RateTableConfig {
    fn default() -> Self {
        Self {
            max_steps: 100_000,
            seed: 1,
        }
    }
}

with_setters!(RateTableConfig {
    with_max_steps => max_steps: val usize,
    with_seed => seed: val u64,
});

/// Directed elementary transition edge in a kinetic rate network.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct RateEdge {
    /// Source basin index.
    pub from: usize,
    /// Target basin index.
    pub to: usize,
    /// First-order transition rate constant `k_{from -> to} >= 0`.
    pub rate: f64,
}

impl RateEdge {
    /// Construct a validated elementary transition edge.
    pub fn new(from: usize, to: usize, rate: f64) -> Result<Self, FluxError> {
        if !rate.is_finite() {
            return Err(FluxError::NonFinite("RateEdge::rate"));
        }
        if rate < 0.0 {
            return Err(FluxError::InvalidConfig(
                "RateEdge::rate must be non-negative",
            ));
        }
        Ok(Self { from, to, rate })
    }
}

/// Build a cumulative kMC rate table from `edges` (implementation moves from rgsaddle).
pub fn build_rate_table(
    _n_states: usize,
    _edges: &[RateEdge],
    _config: &NetworkConfig,
) -> Result<Vec<f64>, FluxError> {
    Err(FluxError::Unimplemented(
        "network: implementation moves from rgsaddle",
    ))
}
