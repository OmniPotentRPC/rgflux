//! Optimal Hyperplanar Transition State Theory (OH-TST).
//!
//! Variationally maximizes the free energy of a dividing hyperplane on the
//! product manifold `S(U) x R` via [`rgmin`]'s Riemannian trust region,
//! combines reversible work along plane paths by quadrature, BAR, or MBAR, and
//! pairs the optimal plane with a recrossing transmission coefficient.
//!
//! Note: implementation moves from rgsaddle.
//!
//! ## References
//!
//! - Goswami, R., and Jónsson, H. (2026). *Adaptive Pruning for Increased
//!   Robustness and Reduced Computational Overhead in Gaussian Process
//!   Accelerated Saddle Point Searches*. ChemPhysChem, 27, e202500730.
//!   <https://doi.org/10.1002/cphc.202500730>
//! - Goswami, R., Gunde, M., and Jónsson, H. (2026). *Enhanced climbing image
//!   nudged elastic band method with Hessian eigenmode alignment*. Frontiers
//!   in Chemistry, 14, 1807063. <https://doi.org/10.3389/fchem.2026.1807063>
//! - Jóhannesson, G. H., and Jónsson, H. (2001). *Optimization of hyperplanar
//!   transition states*. The Journal of Chemical Physics, 115, 9644–9656.
//!   <https://doi.org/10.1063/1.1415499>
//! - Bligaard, T., and Jónsson, H. (2005). *Optimization of hyperplanar
//!   transition states: Application to 2D test problems*. Computer Physics
//!   Communications, 169, 284–288. <https://doi.org/10.1016/j.cpc.2005.03.064>
//! - Bennett, C. H. (1976). *Efficient estimation of free energy differences
//!   from Monte Carlo data*. Journal of Computational Physics, 22, 245–268.
//!   <https://doi.org/10.1016/0021-9991(76)90078-4>
//! - Shirts, M. R., and Chodera, J. D. (2008). *Statistically optimal analysis
//!   of samples from multiple equilibrium states*. The Journal of Chemical
//!   Physics, 129, 124105. <https://doi.org/10.1063/1.2978177>

use crate::engine::{DynamicsEngine, HyperplaneConstraint};
use crate::error::FluxError;
use crate::stats::StatsConfig;

/// Configuration for variational hyperplane optimization and OH-TST rate evaluation.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct OhtstConfig {
    /// Maximum Riemannian trust-region outer iterations.
    pub max_outer_steps: usize,
    /// Consecutive accepted planes within gradient noise required to stop.
    pub patience: usize,
    /// Trust-region cap on normal translation per step.
    pub max_translation: f64,
    /// Trust-region cap on normal rotation angle (radians) per step.
    pub max_rotation: f64,
    /// Minimum two-state BAR overlap before emitting a low-overlap warning.
    pub min_overlap: f64,
    /// Combine reversible work across all plane nodes with MBAR when `true`.
    pub use_mbar: bool,
    /// Deterministic PRNG seed.
    pub seed: u64,
    /// Batch-means uncertainty settings.
    pub stats: StatsConfig,
}

impl Default for OhtstConfig {
    fn default() -> Self {
        Self {
            max_outer_steps: 32,
            patience: 3,
            max_translation: 0.25,
            max_rotation: 0.25,
            min_overlap: 0.03,
            use_mbar: true,
            seed: 1,
            stats: StatsConfig::default(),
        }
    }
}

with_setters!(OhtstConfig {
    with_max_outer_steps => max_outer_steps: val usize,
    with_patience => patience: val usize,
    with_max_translation => max_translation: val f64,
    with_max_rotation => max_rotation: val f64,
    with_min_overlap => min_overlap: val f64,
    with_use_mbar => use_mbar: val bool,
    with_seed => seed: val u64,
    with_stats => stats: val StatsConfig,
});

/// Variational OH-TST rate estimate and optimal dividing hyperplane.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct OhtstEstimate {
    /// Optimized dividing hyperplane.
    pub plane: HyperplaneConstraint,
    /// Variational hyperplanar transition-state rate constant `k_OHTST`.
    pub rate: f64,
    /// Propagated standard error of [`Self::rate`].
    pub standard_error: f64,
}

/// Optimize a dividing hyperplane from `seed_plane` (implementation moves from rgsaddle).
pub fn optimize_hyperplane<E: DynamicsEngine + ?Sized>(
    _engine: &mut E,
    _seed_plane: &HyperplaneConstraint,
    _config: &OhtstConfig,
) -> Result<OhtstEstimate, FluxError> {
    Err(FluxError::Unimplemented(
        "ohtst: implementation moves from rgsaddle",
    ))
}
