//! Projected Average Force Integration (PAFI) along minimum-energy paths.
//!
//! [`pafi_profile`] turns a converged reaction path, such as a nudged elastic
//! band, into a finite-temperature free-energy profile `F(s)`, a barrier
//! `Delta F`, and a normalized rate, without a predefined collective variable.
//! It samples the constrained dynamics on hyperplanes orthogonal to a spline
//! through the path images and integrates the mean constraint force along the
//! path (Swinburne and Marinica, 2018).
//!
//! ## Theory
//!
//! Let `X(s)` be the path, `M = diag(m_i)` the mass matrix of the engine, and
//! `s` the arc length in the mass-weighted metric, `ds^2 = dX^T M dX`. The
//! plane at `s` has the unit normal `n(s) = M X'(s) / |M X'(s)|` and the
//! equation `sigma(x, s) = n(s) . (x - X(s)) = 0`, so the planes are
//! orthogonal to the path in the metric `M`. With `lambda` the Lagrange
//! multiplier of the holonomic constraint on that plane, the free-energy
//! gradient along the path is the constrained mean force with a correction for
//! the tilt of neighbouring planes:
//!
//! `dF/ds = < lambda (n . X' - (x - X) . n') >_{plane at s}`
//!
//! where `n' = dn/ds` is the curvature of the normal. On a straight path
//! `n' = 0` and this is the Sprik and Ciccotti (1998) mean constraint force,
//! which [`crate::hyperplane::sample_hyperplane`] samples for a single plane.
//! The term `(x - X) . n'` is the Jacobian of the plane foliation. Omitting it
//! on a curved path gives a wrong profile. A Cartesian translation that
//! [`PafiConfig::translation_invariance`] declares, or that the reactant
//! Hessian-image test accepts, is removed from `x - X` before that dot. A
//! tether that the same test rejects stays in the Jacobian.
//!
//! The plane means `g(s) = dF/ds` are integrated with a cubic Hermite rule.
//! With two planes the derivative correction cancels and the rule is the
//! trapezoid rule, exact for linear `g`. With three planes the derivative
//! window is quadratic, so the rule is exact for quadratic `g`. With four or
//! more planes the window has at least four nodes, the rule is exact for
//! cubic `g`, and the global error is `O(h^4)` for smoother `g`. The dividing
//! plane `s*` is the global maximum of that interpolant, not the highest
//! sampled plane: on each segment the derivative is a cubic, whose roots in
//! the segment are bracketed between its critical points and refined by
//! bisection, and the barrier is the largest value at those roots and the
//! sampled endpoints. The endpoint slopes are formed from derivative weights
//! already scaled by the segment width, so a very wide or very narrow plane
//! spacing neither underflows nor overflows them, and a slope or interpolant
//! value that is still not finite returns [`FluxError::NonFinite`]. The
//! barrier is `Delta F = F(s*) - F(s_0)`.
//!
//! The reactant partition is the doubled trapezoid of the Boltzmann weight up
//! to `s_div`, the right endpoint `s_{k+1}` of the segment `[s_k, s_{k+1}]`
//! when `s*` lies in `(s_k, s_{k+1}]` (or `s_k` when `s* = s_k`),
//!
//! `Z_R = 2 * \int_0^{s_div} exp(-F(s) / k_B T) ds`,
//! `k = sqrt(k_B T / (2 pi)) exp(-Delta F / k_B T) / Z_R`.
//!
//! The integral is formed in log space.
//! [`FreeEnergyProfile::reactant_log_partition`] and
//! [`FreeEnergyProfile::normalized_log_rate`] are authoritative for every
//! finite value, while [`FreeEnergyProfile::reactant_partition`] and
//! [`FreeEnergyProfile::normalized_rate`] are `None` when the exponential of
//! that finite logarithm overflows or underflows. A non-finite per-plane
//! exponent `-F(s_k) / k_B T` returns [`FluxError::NonFinite`]. The Hermite
//! derivative weights are products of node gaps and are formed in log space
//! for the same reason. The projected frequency `omega_0` of the
//! mass-weighted reactant Hessian is reported separately in
//! [`FreeEnergyProfile::reactant_frequency`] and is not this normalization.
//!
//! What `k` is. It is the transition-state rate of the one-dimensional
//! reduction along the path, `sqrt(k_B T / 2 pi) exp(-Delta F / k_B T)` over
//! the reactant partition function, with the dividing plane at the maximum
//! of the profile and `Z_R` the doubled trapezoid integral above. It assumes
//! a canonical reactant population, no recrossing (transmission coefficient
//! `1`), and a reactant partition function that the sampled planes between
//! the first image and the dividing plane resolve. `omega_0` plays no part in
//! `k`; for a reactant well that is harmonic along the path normal over the
//! thermal range `k` approaches the Vineyard rate `(omega_0 / 2 pi)
//! exp(-Delta F / k_B T)`.
//!
//! What `k` is not. It is not a dynamical rate: recrossing, which lowers the
//! rate, is not estimated here. It is not the Gaussian-reactant rate either.
//! On an anharmonic reactant well the integrated `Z_R` differs from the
//! Gaussian `sqrt(2 pi k_B T) / omega_0`, and `k` follows the integral.
//! `Delta F` is the Hermite integral of the sampled plane means and
//! approaches `F(s*) - F(s_0)` of the path foliation as the sampling and the
//! plane spacing are refined.
//!
//! ## What the code does
//!
//! 1. The images become a cubic spline in the chord-length parameter
//!    `sqrt(sum_j m_j dx_j^2)` between neighbours. The end second derivatives
//!    come from linear extrapolation of the second derivative (the not-a-knot
//!    condition), so the first and last planes keep the curvature of the path.
//!    Three images give one parabola and two images a straight line.
//! 2. `planes` planes sit at equally spaced values of that parameter, from
//!    the first to the last image. Each carries its anchor `X(s_k)`, the
//!    normal, and the curvature `n'`.
//! 3. A zero-temperature check integrates `dV/ds` over the anchors with the
//!    trapezoid rule. It divides the absolute disagreement with the end-to-end
//!    energy change `Delta E_0` by `max(E_unit, |Delta E_0|)`, where `E_unit`
//!    is one energy unit of the engine, and compares that dimensionless value
//!    with the effective tolerance from [`PafiConfig::zero_temperature_tol`].
//!    A mismatch returns [`FluxError::InvalidConfig`] before any sampling.
//! 4. The first image is checked as a reactant minimum before any sampling.
//!    Its forces must be finite. At `dof <= 128`, the mass-weighted Hessian
//!    must also be positive definite once eigenvalues on the zero-mode floor
//!    are set aside; above that limit dense certification is skipped and
//!    [`FreeEnergyProfile::reactant_hessian_certified`] is `false`. The
//!    projected curvature must remain positive at every size. Hessian products
//!    are central differences whose step is a fixed fraction of the
//!    displacement, so a translation of every coordinate leaves the Hessian
//!    unchanged. `omega_0` is then solved by conjugate gradients on those
//!    products.
//! 5. On each plane the engine is constrained, placed at the anchor, given
//!    Maxwell-Boltzmann velocities, run for `burn_in_steps`, and then run for
//!    `sample_steps` steps that each record
//!    `lambda (n . X' - (x - X) . n')`. Batch means over the samples give the
//!    mean force and its standard error on each plane.
//! 6. The profile, barrier and rate follow from the plane means. The
//!    standard errors of the planes are independent and propagate in log space
//!    through the Hermite weights, then combine by `hypot` with the absolute
//!    difference between the Hermite integral and the trapezoid rule on the
//!    same nodes (or at `s*` for the barrier), and propagate in log space
//!    through both the barrier and `Z_R` for the rate by the delta method.
//! 7. With `target_error` set, steps 5 and 6 repeat for up to `max_rounds`
//!    rounds, each adding `sample_steps` fresh samples to every plane, until
//!    the standard error of the barrier is at most `target_error`. Without
//!    it there is one round. The returned profile uses all samples so far.
//!    Missing the target after `max_rounds` rounds is not an error: compare
//!    [`FreeEnergyProfile::barrier_error`] with the target.
//!
//! The engine is restored on return, including on errors. Positions,
//! velocities, the periodic cell, the thermostat and the constraint go back to
//! their old values. The stochastic stream does not: each plane reseeds it
//! with `seed + 4099 * plane + 1000003 * round` (wrapping), so the stream is
//! left where the last plane put it.
//!
//! ## Assumptions
//!
//! - The images form a smooth path of at least two images, distinct in the
//!   mass-weighted metric, with one coordinate per degree of freedom of the
//!   engine.
//! - Thermal excursions in a plane stay inside the tube where neighbouring
//!   planes do not intersect, so `n . X' - (x - X) . n'`, the speed at which
//!   the planes sweep through space, stays positive.
//! - Each plane equilibrates within `burn_in_steps`. The samples start at the
//!   anchor, so a plane whose thermal centre lies far from its anchor needs a
//!   long burn-in.
//! - The first image is the reactant minimum. The call checks that its forces
//!   are finite and, at `dof <= 128`, that the mass-weighted Hessian is positive
//!   definite there after excluding zero modes. Above that limit the dense
//!   check is skipped. At every size, a non-positive curvature along the path
//!   normal makes the projected frequency undefined and the call fails. A
//!   profile whose maximum is the first sample has an empty reactant interval,
//!   so `Z_R` is undefined and the call fails.
//!
//! ## Units
//!
//! - Positions, path images and masses use the units of the engine. Arc
//!   lengths `s` are mass-weighted, `sqrt(mass) * length`, and equal lengths
//!   for unit masses.
//! - Energies, `free_energy`, `standard_error`, `barrier`, `barrier_error`,
//!   `target_error` and the energy part of the tolerance are in engine energy
//!   units. `time_step` is in engine time units.
//! - [`FreeEnergyProfile::normalized_rate`] and
//!   [`FreeEnergyProfile::reactant_frequency`] have units of inverse engine
//!   time. [`FreeEnergyProfile::reactant_partition`] and
//!   [`FreeEnergyProfile::dividing_arc`] have the units of `s`.
//!   [`FreeEnergyProfile::quadrature_residual`] is in engine energy units.
//! - `k_B T` comes from [`crate::engine::ThermostatConfig::thermal_energy`] of
//!   the engine and must be positive.
//!
//! ## Engine wiring
//!
//! [`pafi_profile`] accepts any `E: DynamicsEngine + ?Sized`: the native
//! engine over a [`rgsaddle::PointSurface`], an [`crate::EindirSurface`] or an
//! `rgpot` surface, a session engine, or `LammpcEngine`. [`pafi_profile_dyn`]
//! takes `&mut dyn DynamicsEngine`. The engine has to report
//! [`crate::StepReport::constraint_force`] on constrained steps, and provide
//! [`crate::DynamicsEngine::energy_and_forces`] for the zero-temperature
//! check and for the reactant Hessian.
//!
//! ## Example
//!
//! A straight path through the widening double well
//! `V(x, y) = (1 - x^2)^2 + 0.5 (1.5 + 0.5 x) y^2` with `k_B T = 0.2`. The
//! exact barrier from `x = -1` to the saddle at `x = 0` is
//! `1 + 0.5 k_B T ln(1.5) = 1.04`. This run is far too short for that
//! accuracy, so the example only checks the shape of the result.
//!
//! ```rust
//! use ndarray::{Array1, Array2, ArrayView1};
//! use rgflux::engine::ThermostatConfig;
//! use rgflux::engine::native::{NativeConfig, NativeEngine};
//! use rgflux::pafi::{PafiConfig, pafi_profile};
//! use rgflux::stats::StatsConfig;
//!
//! struct WideningWell;
//! impl rgsaddle::PointSurface for WideningWell {
//!     fn eval(
//!         &self,
//!         r: ArrayView1<'_, f64>,
//!     ) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
//!         let (x, y) = (r[0], r[1]);
//!         let well = 1.0 - x * x;
//!         let k = 1.5 + 0.5 * x;
//!         Ok((
//!             well * well + 0.5 * k * y * y,
//!             Array1::from_vec(vec![-4.0 * x * well + 0.25 * y * y, k * y]),
//!         ))
//!     }
//! }
//!
//! let config = PafiConfig::default()
//!     .with_burn_in_steps(200)
//!     .with_sample_steps(400)
//!     .with_stats(StatsConfig::default().with_min_batches(4));
//! let mut engine = NativeEngine::new(
//!     WideningWell,
//!     NativeConfig::default()
//!         .with_masses(Array1::from_vec(vec![1.0, 1.0]))
//!         .with_thermostat(
//!             ThermostatConfig::default()
//!                 .with_gamma(1.5)
//!                 .with_temperature(0.2),
//!         ),
//! )?;
//!
//! let mut path = Array2::<f64>::zeros((config.planes, 2));
//! for k in 0..config.planes {
//!     path[[k, 0]] = -1.0 + (k as f64) / ((config.planes - 1) as f64);
//! }
//! let profile = pafi_profile(&mut engine, path.view(), &config)?;
//! assert_eq!(profile.free_energy.len(), config.planes);
//! assert_eq!(profile.free_energy[0], 0.0);
//! assert!(profile.barrier.is_finite() && profile.barrier_error > 0.0);
//! assert!(profile.normalized_rate.is_some_and(|rate| rate > 0.0));
//! # Ok::<(), rgflux::FluxError>(())
//! ```
//!
//! ## References
//!
//! - Goswami, R. (2025). *Efficient Exploration of Chemical Kinetics*.
//!   arXiv:2510.21368. <https://doi.org/10.48550/arXiv.2510.21368>
//! - Goswami, R., Gunde, M., and Jónsson, H. (2026). *Enhanced climbing image
//!   nudged elastic band method with Hessian eigenmode alignment*. Frontiers
//!   in Chemistry, 14, 1807063. <https://doi.org/10.3389/fchem.2026.1807063>
//! - Goswami, R. (2026). *Reproducible orchestration of best practices for
//!   reaction path optimization with the nudged elastic band*. MethodsX, 16,
//!   103899. <https://doi.org/10.1016/j.mex.2026.103899>
//! - Swinburne, T. D., and Marinica, M.-C. (2018). *Unsupervised Calculation
//!   of Free Energy Barriers in Large Crystalline Systems*. Physical Review
//!   Letters, 120, 135503. <https://doi.org/10.1103/PhysRevLett.120.135503>
//! - Sprik, M., and Ciccotti, G. (1998). *Free energy from constrained
//!   molecular dynamics*. The Journal of Chemical Physics, 109, 7737 to 7744.
//!   <https://doi.org/10.1063/1.477419>

use std::f64::consts::{LN_2, PI, TAU};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rgmin::vecops::{dot, nrm2};

use crate::engine::{
    DynamicsEngine, EngineSnapshot, HyperplaneConstraint, UNIT_NORMAL_TOL, validate_finite_slice,
    validate_masses,
};
use crate::error::FluxError;
use crate::stats::{StatsConfig, batch_means};

/// Number of planes in the default [`PafiConfig`].
const DEFAULT_PLANES: usize = 9;

/// Coefficient `C` in the relative trapezoid error `C / (planes - 1)^2` of the
/// zero-temperature path-force integral along a quarter-turn circular arc in
/// mass-weighted coordinates through a harmonic well.
///
/// The leading Euler-Maclaurin term of the trapezoid rule on `dV/ds` gives
/// `pi^2 / 12` for that arc, independent of the radius and of the two force
/// constants. Measured on the curved analytic test path the product
/// `error * (planes - 1)^2` stays between 0.82 and 0.86 for 3 to 129 planes
/// and for 9 to 129 path images.
const QUARTER_ARC_TRAPEZOID_COEFFICIENT: f64 = PI * PI / 12.0;

/// Margin by which the default zero-temperature tolerance exceeds the
/// quarter-arc trapezoid error at [`DEFAULT_PLANES`].
///
/// The leading relative trapezoid error for `N` planes is
/// `pi^2 / (12 (N - 1)^2)`, or about `1.29%` at `N = 9`, which exceeded the
/// previous `0.01` (`1%`) default tolerance on `develop`. Adding one leading
/// term sets the safety margin to `1.0` and gives the `9`-plane default of
/// `2 * (pi^2 / 12) / 8^2`, about `2.57%`.
const ZERO_TEMPERATURE_SAFETY_MARGIN: f64 = 1.0;

const fn default_zero_temperature_tol(planes: usize) -> f64 {
    let intervals = (planes - 1) as f64;
    let leading_term = QUARTER_ARC_TRAPEZOID_COEFFICIENT / (intervals * intervals);
    leading_term * (1.0 + ZERO_TEMPERATURE_SAFETY_MARGIN)
}

fn effective_zero_temperature_tol(config: &PafiConfig) -> f64 {
    config
        .zero_temperature_tol
        .unwrap_or_else(|| default_zero_temperature_tol(config.planes))
}

/// Configuration for Projected Average Force Integration (PAFI).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct PafiConfig {
    /// Number of hyperplanes placed at equal spacing along the spline path
    /// (`>= 2`).
    ///
    /// Default: `9`.
    pub planes: usize,
    /// Equilibration steps discarded on each hyperplane before sampling
    /// (`>= 1`). Every plane and every round starts from the anchor again.
    ///
    /// Default: `500`.
    pub burn_in_steps: usize,
    /// Production steps added per hyperplane and adaptive round (`>= 2`).
    ///
    /// Default: `4_000`.
    pub sample_steps: usize,
    /// Integration timestep `dt > 0` in engine time units.
    ///
    /// Default: `0.002`.
    pub time_step: f64,
    /// Optional target standard error on the integrated barrier `Delta F`
    /// (finite and `>= 0`), in engine energy units. `None` runs one round.
    ///
    /// Default: `None`.
    pub target_error: Option<f64>,
    /// Maximum sampling rounds (`>= 1`). Rounds after the first only run when
    /// `target_error` is set and not yet reached.
    ///
    /// Default: `4`.
    pub max_rounds: usize,
    /// Dimensionless tolerance for the zero-temperature path-force check.
    ///
    /// Let `I_T` be the trapezoid integral of `dV/ds` over the plane anchors
    /// and `Delta E_0` the endpoint energy change, both in the engine's energy
    /// units. The check compares
    /// `|I_T - Delta E_0| / max(E_unit, |Delta E_0|)` with the effective
    /// tolerance, where `E_unit` is one energy unit of the engine. Thus the
    /// check is relative when `|Delta E_0| >= E_unit` and otherwise bounds the
    /// absolute discrepancy in engine energy units.
    ///
    /// `None`, the default, sets the effective tolerance to
    /// `2 * (pi^2 / 12) / (planes - 1)^2`. At the default 9 planes that value
    /// is about `0.0257`, and changing `planes` scales it with the trapezoid
    /// error. `Some(value)` uses `value` at every plane count, including when
    /// `value` equals the 9-plane tolerance.
    ///
    /// The scaling follows the trapezoid error on a quarter-turn circular arc
    /// through a harmonic well. Paths that turn through more than a quarter
    /// circle can need a larger tolerance. Larger values weaken the guard
    /// against inconsistent forces and tangents. An explicit value must be
    /// finite and `>= 0`.
    ///
    /// Default: `None`.
    pub zero_temperature_tol: Option<f64>,
    /// Base of the deterministic per-plane seeds `seed + 4099 * plane + 1000003 * round`
    /// (wrapping).
    ///
    /// Default: `1`.
    pub seed: u64,
    /// Batch-means uncertainty settings for the per-plane force series.
    ///
    /// Default: [`StatsConfig::default`].
    pub stats: StatsConfig,
    /// Optional declaration of rigid-translation invariance on the Cartesian
    /// `[x, y, z]` axes.
    ///
    /// `Some([..])` bypasses numerical symmetry detection: `true` projects
    /// that mass-weighted translation and `false` retains it. `None` probes
    /// each axis independently with two central-difference step sizes. The
    /// probe compares a Richardson truncation estimate with a roundoff floor;
    /// an uncertainty interval that straddles the floor returns
    /// [`FluxError::AmbiguousTranslationMode`] rather than silently choosing.
    ///
    /// Engines that evaluate forces internally in `f32` can quantize both
    /// probes to the same value, especially over a large background force.
    /// Such engines should declare known symmetries explicitly because an
    /// unresolved sub-`f32` force change cannot be inferred from returned
    /// `f64` values.
    ///
    /// Numerical probing cannot resolve a physical tether whose Hessian
    /// response is below the floating-point floor. With a large background
    /// gradient this floor contains a term of order `eps * |gradient| / h`;
    /// it also contains the propagated probe-position rounding bound. A
    /// smaller tether can therefore be classified as invariant and projected.
    /// Set the corresponding axis to `false` when such a weak tether is known
    /// to exist. The ambiguous band is only the interval where the
    /// Richardson truncation uncertainty straddles the computed roundoff
    /// floor; its total width is twice that uncertainty.
    ///
    /// Default: `None`.
    pub translation_invariance: Option<[bool; 3]>,
}

impl Default for PafiConfig {
    fn default() -> Self {
        Self {
            planes: DEFAULT_PLANES,
            burn_in_steps: 500,
            sample_steps: 4_000,
            time_step: 0.002,
            target_error: None,
            max_rounds: 4,
            zero_temperature_tol: None,
            seed: 1,
            stats: StatsConfig::default(),
            translation_invariance: None,
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
    with_zero_temperature_tol => zero_temperature_tol: opt f64,
    with_seed => seed: val u64,
    with_stats => stats: val StatsConfig,
    with_translation_invariance => translation_invariance: opt [bool; 3],
});

/// Integrated free-energy profile along a discretized reaction path.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct FreeEnergyProfile {
    /// Plane coordinates `s_k`: the mass-weighted chord-length parameter of
    /// the spline, from `0` at the first image to the total path length at
    /// the last, at equal spacing. They are not scaled to `[0, 1]`.
    pub arc_lengths: Vec<f64>,
    /// Integrated free energy `F(s_k) - F(s_0)` at each plane, so the first
    /// entry is `0`.
    pub free_energy: Vec<f64>,
    /// Standard error of `free_energy` at each plane. The batch-means errors
    /// of the plane forces are propagated in log space through the Hermite
    /// weights and combined, by `hypot`, with the absolute difference between
    /// the Hermite integral and the trapezoid rule on the same nodes. The
    /// first entry is `0`. A zero entry means those two contributions are both
    /// zero.
    pub standard_error: Vec<f64>,
    /// Free-energy barrier `Delta F = F(s*) - F(s_0)` at the dividing plane,
    /// the global maximum of the interpolated profile.
    pub barrier: f64,
    /// Standard error of [`Self::barrier`]. The propagated plane-force error
    /// at `s*` is formed in log space and combined, by `hypot`, with the
    /// absolute difference between the Hermite barrier and the trapezoid
    /// integral at the same arc.
    pub barrier_error: f64,
    /// Index of the first sampled plane whose free energy is the largest.
    /// [`Self::dividing_arc`] can lie in an interval that does not touch this
    /// plane when the interpolant overshoots the samples.
    pub dividing_index: usize,
    /// Arc length `s*` of the dividing plane. It is not restricted to a
    /// sampled plane.
    pub dividing_arc: f64,
    /// Projected reactant frequency `omega_0 = (n^T H^-1 n)^(-1/2)` from the
    /// mass-weighted reactant Hessian `H` and the unit mass-weighted path
    /// normal `n` at the first image.
    pub reactant_frequency: f64,
    /// Standard error of [`Self::reactant_frequency`].
    ///
    /// Single profiles use zero because their reactant Hessian is deterministic.
    /// Pooled profiles estimate this value between replicates.
    pub reactant_frequency_error: f64,
    /// Whether the full mass-weighted reactant Hessian was certified positive
    /// definite off its zero modes by the dense eigensolver. This is `false`
    /// when the coordinate count exceeds 128 and dense certification is
    /// skipped; finite forces and positive projected curvature are still
    /// required.
    pub reactant_hessian_certified: bool,
    /// Reactant partition, when its finite logarithm has a representable
    /// positive exponential.
    pub reactant_partition: Option<f64>,
    /// Natural logarithm of the reactant partition,
    /// `ln Z_R = ln(2 * \int_0^{s_div} exp(-F(s) / k_B T) ds)`.
    pub reactant_log_partition: f64,
    /// Normalized PAFI rate
    /// `k = sqrt(k_B T / (2 pi)) * exp(-Delta F / k_B T) / Z_R`, with `Z_R`
    /// the doubled trapezoid reactant partition.
    ///
    /// This is a one-dimensional transition-state rate with no recrossing
    /// correction. It is not a dynamical rate.
    ///
    /// `Some` when [`Self::normalized_log_rate`] has a finite positive
    /// exponential. `None` when that finite log overflows or underflows.
    pub normalized_rate: Option<f64>,
    /// Natural logarithm of [`Self::normalized_rate`].
    /// This stays finite when [`Self::normalized_rate`] underflows.
    pub normalized_log_rate: f64,
    /// Propagated standard error of [`Self::normalized_rate`] when it is
    /// representable.
    pub normalized_rate_error: Option<f64>,
    /// Largest absolute discrepancy, over the sampled planes, between the
    /// Hermite integral of `dV/ds` and the anchor energy measured from the
    /// first plane. The acceptance check itself stays the trapezoid comparison
    /// of [`PafiConfig::zero_temperature_tol`].
    pub quadrature_residual: f64,
    gradients: Vec<f64>,
    gradient_errors: Vec<f64>,
}

impl FreeEnergyProfile {
    /// Sampled mean free-energy gradients `dF/ds` at each hyperplane in [`Self::arc_lengths`].
    #[must_use]
    pub fn gradients(&self) -> &[f64] {
        &self.gradients
    }

    /// Standard errors of [`Self::gradients`] at each hyperplane in [`Self::arc_lengths`].
    #[must_use]
    pub fn gradient_errors(&self) -> &[f64] {
        &self.gradient_errors
    }
}

#[derive(Clone, Debug)]
struct PlaneGeometry {
    anchor: Array1<f64>,
    normal: Array1<f64>,
    dn_ds: Array1<f64>,
    dc_ds: f64,
    tangent: Array1<f64>,
    arc: f64,
}

#[derive(Clone, Debug)]
struct PlaneEstimate {
    gradient: f64,
    standard_error: f64,
}

#[derive(Clone, Debug)]
struct NaturalSpline {
    knots: Vec<f64>,
    values: Vec<Array1<f64>>,
    second: Vec<Array1<f64>>,
}

impl NaturalSpline {
    fn new(
        path_images: ArrayView2<'_, f64>,
        masses: ArrayView1<'_, f64>,
    ) -> Result<Self, FluxError> {
        let rows = path_images.nrows();
        let dof = path_images.ncols();
        let mut knots = Vec::with_capacity(rows);
        knots.push(0.0);
        for i in 1..rows {
            let mut interval_squared = 0.0_f64;
            for j in 0..dof {
                let delta = path_images[[i, j]] - path_images[[i - 1, j]];
                interval_squared += masses[j] * delta * delta;
            }
            let interval = interval_squared.sqrt();
            if !interval.is_finite() || interval <= f64::MIN_POSITIVE {
                return Err(FluxError::InvalidConfig(
                    "PAFI path images must be distinct in mass-weighted arc length",
                ));
            }
            knots.push(knots[i - 1] + interval);
        }

        let values: Vec<Array1<f64>> = path_images
            .rows()
            .into_iter()
            .map(|row| row.to_owned())
            .collect();
        let mut second = vec![Array1::zeros(dof); rows];
        if rows == 3 {
            let left = knots[1] - knots[0];
            let right = knots[2] - knots[1];
            for j in 0..dof {
                let curvature = 2.0
                    * ((values[2][j] - values[1][j]) / right
                        - (values[1][j] - values[0][j]) / left)
                    / (left + right);
                for endpoint in &mut second {
                    endpoint[j] = curvature;
                }
            }
        } else if rows > 3 {
            let interior = rows - 2;
            let mut lower = vec![0.0_f64; interior];
            let mut diagonal = vec![0.0_f64; interior];
            let mut upper = vec![0.0_f64; interior];
            let mut rhs = vec![Array1::zeros(dof); interior];
            for k in 0..interior {
                let i = k + 1;
                let left = knots[i] - knots[i - 1];
                let right = knots[i + 1] - knots[i];
                lower[k] = left;
                diagonal[k] = 2.0 * (left + right);
                upper[k] = right;
                for j in 0..dof {
                    rhs[k][j] = 6.0
                        * ((values[i + 1][j] - values[i][j]) / right
                            - (values[i][j] - values[i - 1][j]) / left);
                }
            }
            let first = knots[1] - knots[0];
            let second_width = knots[2] - knots[1];
            diagonal[0] += first * (first + second_width) / second_width;
            upper[0] -= first * first / second_width;

            let penultimate = knots[rows - 2] - knots[rows - 3];
            let last = knots[rows - 1] - knots[rows - 2];
            lower[interior - 1] -= last * last / penultimate;
            diagonal[interior - 1] += last * (penultimate + last) / penultimate;

            for k in 1..interior {
                let factor = lower[k] / diagonal[k - 1];
                diagonal[k] -= factor * upper[k - 1];
                for j in 0..dof {
                    rhs[k][j] -= factor * rhs[k - 1][j];
                }
            }
            for k in (0..interior).rev() {
                let i = k + 1;
                for j in 0..dof {
                    let next = if k + 1 < interior {
                        upper[k] * second[i + 1][j]
                    } else {
                        0.0
                    };
                    second[i][j] = (rhs[k][j] - next) / diagonal[k];
                }
            }
            for j in 0..dof {
                second[0][j] =
                    ((first + second_width) * second[1][j] - first * second[2][j]) / second_width;
                second[rows - 1][j] = ((penultimate + last) * second[rows - 2][j]
                    - last * second[rows - 3][j])
                    / penultimate;
            }
        }
        Ok(Self {
            knots,
            values,
            second,
        })
    }

    fn end(&self) -> f64 {
        self.knots[self.knots.len() - 1]
    }

    fn evaluate(&self, s: f64) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let last = self.knots.len() - 1;
        let mut segment = last - 1;
        for i in 0..last {
            if s <= self.knots[i + 1] {
                segment = i;
                break;
            }
        }
        let left = self.knots[segment];
        let right = self.knots[segment + 1];
        let width = right - left;
        let a = (right - s) / width;
        let b = (s - left) / width;
        let dof = self.values[segment].len();
        let mut position = Array1::zeros(dof);
        let mut tangent = Array1::zeros(dof);
        let mut curvature = Array1::zeros(dof);
        for j in 0..dof {
            let y0 = self.values[segment][j];
            let y1 = self.values[segment + 1][j];
            let m0 = self.second[segment][j];
            let m1 = self.second[segment + 1][j];
            position[j] = a * y0
                + b * y1
                + ((a * a * a - a) * m0 + (b * b * b - b) * m1) * width * width / 6.0;
            tangent[j] = (y1 - y0) / width
                + width * ((-3.0 * a * a + 1.0) * m0 + (3.0 * b * b - 1.0) * m1) / 6.0;
            curvature[j] = a * m0 + b * m1;
        }
        (position, tangent, curvature)
    }
}

fn validate_path_and_build_spline(
    path_images: ArrayView2<'_, f64>,
    masses: ArrayView1<'_, f64>,
    planes: usize,
) -> Result<NaturalSpline, FluxError> {
    if path_images.nrows() < 2 {
        return Err(FluxError::InvalidConfig(
            "PAFI path requires at least two images",
        ));
    }
    if path_images.ncols() == 0 {
        return Err(FluxError::EmptyInput("PAFI path image coordinates"));
    }
    if planes < 2 {
        return Err(FluxError::InvalidConfig(
            "PafiConfig::planes must be at least 2",
        ));
    }
    validate_masses(masses)?;
    if masses.len() != path_images.ncols() {
        return Err(FluxError::Shape {
            expected: path_images.ncols(),
            got: masses.len(),
            context: "PAFI masses versus path coordinates",
        });
    }
    for row in path_images.rows() {
        validate_finite_slice(row, "PAFI path image")?;
    }

    let spline = NaturalSpline::new(path_images, masses)?;
    let total_arc = spline.end();
    if !total_arc.is_finite() || total_arc <= f64::MIN_POSITIVE {
        return Err(FluxError::InvalidConfig(
            "PAFI path total arc length must be positive",
        ));
    }
    Ok(spline)
}

fn evaluate_plane_geometry_at(
    spline: &NaturalSpline,
    masses: ArrayView1<'_, f64>,
    arc: f64,
) -> Result<PlaneGeometry, FluxError> {
    let dof = masses.len();
    let (anchor, tangent, curvature) = spline.evaluate(arc);
    let mut normal_raw = Array1::zeros(dof);
    let mut normal_derivative_raw = Array1::zeros(dof);
    for j in 0..dof {
        normal_raw[j] = masses[j] * tangent[j];
        normal_derivative_raw[j] = masses[j] * curvature[j];
    }
    let normal_norm = nrm2(normal_raw.view());
    if !normal_norm.is_finite() || normal_norm <= f64::MIN_POSITIVE.sqrt() {
        return Err(FluxError::InvalidConfig("PAFI normal norm underflowed"));
    }
    let normal = normal_raw.mapv(|v| v / normal_norm);
    if (nrm2(normal.view()) - 1.0).abs() > UNIT_NORMAL_TOL {
        return Err(FluxError::InvalidConfig(
            "PAFI unit normal construction failed",
        ));
    }
    let projection = dot(normal.view(), normal_derivative_raw.view());
    let mut dn_ds = Array1::zeros(dof);
    for j in 0..dof {
        dn_ds[j] = (normal_derivative_raw[j] - normal[j] * projection) / normal_norm;
    }
    let dc_ds = dot(normal.view(), tangent.view());
    if !dc_ds.is_finite() || dc_ds <= f64::MIN_POSITIVE.sqrt() {
        return Err(FluxError::InvalidConfig(
            "PAFI tangent projection to hyperplane normal is non-positive",
        ));
    }

    Ok(PlaneGeometry {
        anchor,
        normal,
        dn_ds,
        dc_ds,
        tangent,
        arc,
    })
}

fn build_plane_geometries_at(
    spline: &NaturalSpline,
    masses: ArrayView1<'_, f64>,
    arcs: &[f64],
) -> Result<Vec<PlaneGeometry>, FluxError> {
    if arcs.len() < 2 {
        return Err(FluxError::InvalidConfig(
            "PafiConfig::planes must be at least 2",
        ));
    }
    let mut geometries = Vec::with_capacity(arcs.len());
    for (index, &arc) in arcs.iter().enumerate() {
        if !arc.is_finite() {
            return Err(FluxError::NonFinite("PAFI plane arc length"));
        }
        if index > 0 && arc <= arcs[index - 1] {
            return Err(FluxError::InvalidConfig(
                "PAFI arc lengths must strictly increase",
            ));
        }
        geometries.push(evaluate_plane_geometry_at(spline, masses, arc)?);
    }
    Ok(geometries)
}

fn build_plane_geometries(
    path_images: ArrayView2<'_, f64>,
    masses: ArrayView1<'_, f64>,
    planes: usize,
) -> Result<Vec<PlaneGeometry>, FluxError> {
    let spline = validate_path_and_build_spline(path_images, masses, planes)?;
    let total_arc = spline.end();
    let arcs: Vec<f64> = (0..planes)
        .map(|k| total_arc * (k as f64) / ((planes - 1) as f64))
        .collect();
    build_plane_geometries_at(&spline, masses, &arcs)
}

fn sample_plane<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    geometry: &PlaneGeometry,
    config: &PafiConfig,
    plane_index: usize,
    round: usize,
    translation_modes: &[Array1<f64>],
) -> Result<Vec<f64>, FluxError> {
    let offset = dot(geometry.normal.view(), geometry.anchor.view());
    let plane = HyperplaneConstraint::new(geometry.normal.clone(), offset)?;
    engine.set_constraint(Some(plane))?;
    engine.set_positions(geometry.anchor.view())?;
    let seed = config
        .seed
        .wrapping_add((plane_index as u64).wrapping_mul(4_099))
        .wrapping_add((round as u64).wrapping_mul(1_000_003));
    engine.shoot_velocities(seed)?;
    for _ in 0..config.burn_in_steps {
        let report = engine.step(config.time_step)?;
        let lambda = report.constraint_force.ok_or(FluxError::InvalidConfig(
            "PAFI requires constraint force reporting on constrained steps",
        ))?;
        if !lambda.is_finite() {
            return Err(FluxError::NonFinite("PAFI burn-in constraint force"));
        }
    }

    let mut values = Vec::with_capacity(config.sample_steps);
    for _ in 0..config.sample_steps {
        let report = engine.step(config.time_step)?;
        let lambda = report.constraint_force.ok_or(FluxError::InvalidConfig(
            "PAFI requires constraint force reporting on constrained steps",
        ))?;
        values.push(projected_force_sample(
            engine,
            geometry,
            lambda,
            translation_modes,
        )?);
    }
    Ok(values)
}

fn projected_force_sample<E: DynamicsEngine + ?Sized>(
    engine: &E,
    geometry: &PlaneGeometry,
    lambda: f64,
    translation_modes: &[Array1<f64>],
) -> Result<f64, FluxError> {
    let positions = engine.positions();
    let masses = engine.masses();
    let mut weighted: Array1<f64> = positions
        .iter()
        .zip(geometry.anchor.iter())
        .zip(masses.iter())
        .map(|((position, anchor), mass)| (position - anchor) * mass.sqrt())
        .collect();
    for mode in translation_modes {
        let component = dot(weighted.view(), mode.view());
        weighted.scaled_add(-component, mode);
    }
    let jacobian_projection = weighted
        .iter()
        .zip(masses.iter())
        .zip(positions.iter().zip(geometry.anchor.iter()))
        .zip(geometry.dn_ds.iter())
        .map(|(((weighted, mass), (position, anchor)), dn_ds)| {
            let root = mass.sqrt();
            let displacement = if root > 0.0 {
                weighted / root
            } else {
                position - anchor
            };
            displacement * dn_ds
        })
        .sum::<f64>();
    let value = lambda * (geometry.dc_ds - jacobian_projection);
    if !value.is_finite() {
        return Err(FluxError::NonFinite("PAFI projected force sample"));
    }
    Ok(value)
}

fn estimate_plane(samples: &[f64], config: &PafiConfig) -> Result<PlaneEstimate, FluxError> {
    let estimate = batch_means(samples, &config.stats)?;
    if !estimate.mean.is_finite() {
        return Err(FluxError::NonFinite("PAFI plane force mean"));
    }
    if !estimate.standard_error.is_finite() || estimate.standard_error < 0.0 {
        return Err(FluxError::InvalidConfig(
            "PAFI plane standard error must be finite and non-negative",
        ));
    }
    Ok(PlaneEstimate {
        gradient: estimate.mean,
        standard_error: estimate.standard_error,
    })
}

/// Nodes used to differentiate the mean force at one plane.
const DERIVATIVE_WINDOW: usize = 5;

/// Central difference step as a fraction of the displacement.
///
/// The cube root of machine epsilon balances the `O(h^2)` truncation of a
/// second derivative against roundoff. The fraction multiplies the
/// displacement vector, so a translation of every coordinate leaves the
/// step, and therefore the Hessian product, unchanged.
fn hessian_displacement_fraction() -> f64 {
    f64::EPSILON.cbrt()
}

/// Eigenvalues no larger than this fraction of `1 +` the stiffest curvature
/// are zero modes (translation or rotation), not negative curvature.
///
/// A central difference with step [`hessian_displacement_fraction`] rounds
/// an exact zero eigenvalue down near `epsilon^(2/3)`. The square root of
/// machine epsilon sits above that roundoff and below a curvature of order one.
fn reactant_zero_mode_floor() -> f64 {
    f64::EPSILON.sqrt()
}

/// `rgsaddle::exact_eigh` uses cyclic Jacobi through this dimension when the
/// `faer` feature is off. Dense certification is skipped above this limit.
const REACTANT_HESSIAN_EIGEN_LIMIT: usize = 128;

/// Halvings that bring a bracket in `[0, 1]` below `f64` resolution at any
/// root that matters, and the cap when the root sits near zero.
const BISECTION_LIMIT: usize = 200;

/// Cubic Hermite quadrature of the plane mean forces.
///
/// Each interval uses the cubic Hermite interpolant of `g = dF/ds`. The
/// derivatives at the planes come from Lagrange differentiation on a window
/// of `min(n, 5)` neighbouring planes. Over an interval of width `h` the
/// integral is `h/2 (g_k + g_{k+1}) + h^2/12 (g'_k - g'_{k+1})`.
///
/// The integral is exact for a polynomial `g` that this window can
/// differentiate exactly, and [`PafiConfig::planes`] may be as small as 2:
///
/// - 2 planes. Both derivatives are the secant between the planes, the
///   correction vanishes, and the rule is the trapezoid rule. It is exact
///   for linear `g`.
/// - 3 planes. The window is quadratic, so the rule is exact for quadratic
///   `g` and not for a general cubic.
/// - 4 or more planes. The window has at least 4 nodes, so the rule is exact
///   for cubic `g`. For a smoother mean force the global error is `O(h^4)`.
///
/// Every weight is linear in the plane means, so the same matrix propagates
/// the plane standard errors.
#[derive(Clone, Debug)]
struct Quadrature {
    arcs: Vec<f64>,
    derivative: Vec<Vec<f64>>,
    cumulative: Vec<Vec<f64>>,
}

#[derive(Clone, Debug)]
struct DividingPlane {
    index: usize,
    arc: f64,
    free_energy: f64,
    weights: Vec<f64>,
    /// Last sample included in the reactant integral. This is the end of the
    /// segment that holds the interpolant maximum, or the maximum itself when
    /// that maximum sits on a sample.
    partition_end: usize,
}

fn checked_exp(log_value: f64, context: &'static str) -> Result<f64, FluxError> {
    if !log_value.is_finite() {
        return Err(FluxError::NonFinite(context));
    }
    let value = log_value.exp();
    if !value.is_finite() {
        return Err(FluxError::NonFinite(context));
    }
    Ok(value)
}

fn finite_exponential(log_value: f64) -> Option<f64> {
    let value = log_value.exp();
    (value.is_finite() && value > 0.0).then_some(value)
}

fn lagrange_derivative_weights(nodes: &[f64], at: usize) -> Result<Vec<f64>, FluxError> {
    let mut weights = vec![0.0_f64; nodes.len()];
    for j in 0..nodes.len() {
        if j == at {
            let mut diagonal = 0.0_f64;
            for m in 0..nodes.len() {
                if m == at {
                    continue;
                }
                let gap = nodes[at] - nodes[m];
                if !gap.is_finite() || gap == 0.0 {
                    return Err(FluxError::NonFinite("PAFI Lagrange derivative weight"));
                }
                diagonal += 1.0 / gap;
            }
            if !diagonal.is_finite() {
                return Err(FluxError::NonFinite("PAFI Lagrange derivative weight"));
            }
            weights[j] = diagonal;
        } else {
            // `l_j'(x_at) = prod_{m != j, m != at} (x_at - x_m) / prod_{m != j} (x_j - x_m)`.
            // The products overflow when the nodes span more than about `f64::MAX.sqrt()`,
            // while the ratio stays finite. Sum the logs, then exponentiate once.
            let mut log_magnitude = 0.0_f64;
            let mut negative = false;
            for m in 0..nodes.len() {
                if m == j {
                    continue;
                }
                let gap = nodes[j] - nodes[m];
                if !gap.is_finite() || gap == 0.0 {
                    return Err(FluxError::NonFinite("PAFI Lagrange derivative weight"));
                }
                log_magnitude -= gap.abs().ln();
                if gap.is_sign_negative() {
                    negative = !negative;
                }
                if m != at {
                    let span = nodes[at] - nodes[m];
                    if !span.is_finite() || span == 0.0 {
                        return Err(FluxError::NonFinite("PAFI Lagrange derivative weight"));
                    }
                    log_magnitude += span.abs().ln();
                    if span.is_sign_negative() {
                        negative = !negative;
                    }
                }
            }
            let magnitude = checked_exp(log_magnitude, "PAFI Lagrange derivative weight")?;
            weights[j] = if negative { -magnitude } else { magnitude };
        }
    }
    Ok(weights)
}

impl Quadrature {
    fn new(arcs: &[f64]) -> Result<Self, FluxError> {
        let n = arcs.len();
        if n < 2 {
            return Err(FluxError::Shape {
                expected: 2,
                got: n,
                context: "PAFI quadrature arc lengths",
            });
        }
        for pair in arcs.windows(2) {
            let ds = pair[1] - pair[0];
            if !ds.is_finite() || ds <= 0.0 {
                return Err(FluxError::InvalidConfig(
                    "PAFI arc lengths must strictly increase",
                ));
            }
        }
        let window = n.min(DERIVATIVE_WINDOW);
        let mut derivative = vec![vec![0.0_f64; n]; n];
        for (i, row) in derivative.iter_mut().enumerate() {
            let start = i.saturating_sub(window / 2).min(n - window);
            let local = lagrange_derivative_weights(&arcs[start..start + window], i - start)?;
            for (offset, weight) in local.into_iter().enumerate() {
                row[start + offset] = weight;
            }
        }
        let mut cumulative = vec![vec![0.0_f64; n]; n];
        for k in 0..n - 1 {
            let h = arcs[k + 1] - arcs[k];
            let mut previous = cumulative[k].clone();
            previous[k] += 0.5 * h;
            previous[k + 1] += 0.5 * h;
            for (j, weight) in previous.iter_mut().enumerate() {
                // `h * derivative` is of order one, so scaling it before `h / 12`
                // keeps a spacing above `sqrt(f64::MAX)` from overflowing `h * h`.
                *weight += h / 12.0 * (h * derivative[k][j] - h * derivative[k + 1][j]);
            }
            cumulative[k + 1] = previous;
        }
        Ok(Self {
            arcs: arcs.to_vec(),
            derivative,
            cumulative,
        })
    }

    fn len(&self) -> usize {
        self.arcs.len()
    }

    fn integrate(&self, gradients: &[f64]) -> Vec<f64> {
        self.cumulative
            .iter()
            .map(|row| {
                row.iter()
                    .zip(gradients)
                    .map(|(weight, gradient)| weight * gradient)
                    .sum()
            })
            .collect()
    }

    /// `(h s_k, h s_{k+1})`, the interpolant slopes at both ends of segment
    /// `k` times its width `h`.
    ///
    /// A derivative weight scales as `1 / h`, so `h * weight` is of order one
    /// and is formed before the gradient multiplies it. Summing the unscaled
    /// products first underflows to zero when `h` is near `sqrt(f64::MAX)`
    /// and overflows when `h` is near `sqrt(f64::MIN_POSITIVE)`.
    fn scaled_slopes(&self, k: usize, gradients: &[f64]) -> Result<(f64, f64), FluxError> {
        let h = self.arcs[k + 1] - self.arcs[k];
        let scaled = |row: &[f64]| -> f64 {
            row.iter()
                .zip(gradients)
                .map(|(weight, gradient)| (h * weight) * gradient)
                .sum()
        };
        let left = scaled(&self.derivative[k]);
        let right = scaled(&self.derivative[k + 1]);
        if left.is_finite() && right.is_finite() {
            Ok((left, right))
        } else {
            Err(FluxError::NonFinite(
                "PAFI dividing-plane interpolant slope",
            ))
        }
    }

    /// Cumulative-integral weights of `F(s_k + u h_k)`, with `u` in `[0, 1]`.
    fn interval_weights(&self, k: usize, u: f64) -> Vec<f64> {
        let h = self.arcs[k + 1] - self.arcs[k];
        let u2 = u * u;
        let u3 = u2 * u;
        let u4 = u2 * u2;
        let left_value = u - u3 + 0.5 * u4;
        let right_value = u3 - 0.5 * u4;
        let left_slope = 0.5 * u2 - 2.0 * u3 / 3.0 + 0.25 * u4;
        let right_slope = 0.25 * u4 - u3 / 3.0;
        let mut weights = self.cumulative[k].clone();
        weights[k] += h * left_value;
        weights[k + 1] += h * right_value;
        for (j, weight) in weights.iter_mut().enumerate() {
            *weight += h
                * (left_slope * (h * self.derivative[k][j])
                    + right_slope * (h * self.derivative[k + 1][j]));
        }
        weights
    }

    /// `F(s_k + u h)` on segment `k`, with `scaled` from [`Self::scaled_slopes`].
    fn interpolated_value(
        &self,
        k: usize,
        u: f64,
        gradients: &[f64],
        scaled: (f64, f64),
        free_energy: &[f64],
    ) -> f64 {
        let h = self.arcs[k + 1] - self.arcs[k];
        let u2 = u * u;
        let u3 = u2 * u;
        let u4 = u2 * u2;
        free_energy[k]
            + h * (gradients[k] * (u - u3 + 0.5 * u4) + gradients[k + 1] * (u3 - 0.5 * u4))
            + h * (scaled.0 * (0.5 * u2 - 2.0 * u3 / 3.0 + 0.25 * u4)
                + scaled.1 * (0.25 * u4 - u3 / 3.0))
    }

    /// Global maximum of the interpolated profile.
    ///
    /// Each segment is the integral of a cubic Hermite interpolant of the mean
    /// force, so its derivative is the cubic `a u^3 + b u^2 + c u + d`. The
    /// maximum on that segment is the larger of the endpoint values and the
    /// values at the real roots inside the segment. Those roots come from
    /// [`cubic_unit_interval_roots`], which brackets them between the
    /// critical points of the cubic, so a leading coefficient of any
    /// magnitude, including `a == 0`, is handled by the same path. The
    /// endpoint values are the integrated plane values themselves, so the
    /// barrier is never below the highest sampled plane. The highest sampled
    /// plane is recorded separately, because an overshoot can put `s*` on an
    /// interval that does not touch that plane.
    ///
    /// A slope or interior value that is not finite returns
    /// [`FluxError::NonFinite`] rather than dropping that segment.
    fn dividing_plane(
        &self,
        gradients: &[f64],
        free_energy: &[f64],
    ) -> Result<DividingPlane, FluxError> {
        let n = self.len();
        let mut index = 0usize;
        for i in 1..n {
            if free_energy[i] > free_energy[index] {
                index = i;
            }
        }
        let mut best_interval = 0usize;
        let mut best_u = 0.0_f64;
        let mut best_value = free_energy[0];
        for k in 0..n - 1 {
            let (u, value) = self.segment_maximum(k, gradients, free_energy)?;
            if value > best_value {
                best_interval = k;
                best_u = u;
                best_value = value;
            }
        }
        let width = self.arcs[best_interval + 1] - self.arcs[best_interval];
        let partition_end = if best_u == 0.0 {
            best_interval
        } else {
            best_interval + 1
        };
        Ok(DividingPlane {
            index,
            arc: self.arcs[best_interval] + best_u * width,
            free_energy: best_value,
            weights: self.interval_weights(best_interval, best_u),
            partition_end,
        })
    }

    /// `(u, F)` at the maximum of segment `k`, with `u` in `[0, 1]`.
    ///
    /// Ties keep the smaller `u`, so a flat segment reports its left endpoint.
    fn segment_maximum(
        &self,
        k: usize,
        gradients: &[f64],
        free_energy: &[f64],
    ) -> Result<(f64, f64), FluxError> {
        let scaled = self.scaled_slopes(k, gradients)?;
        let [a, b, c, d] = segment_force_coefficients(gradients[k], gradients[k + 1], scaled);
        let (nroots, roots) = cubic_unit_interval_roots(a, b, c, d);
        let value_at = |u: f64| -> Result<f64, FluxError> {
            let value = if u == 0.0 {
                free_energy[k]
            } else if u == 1.0 {
                free_energy[k + 1]
            } else {
                self.interpolated_value(k, u, gradients, scaled, free_energy)
            };
            if value.is_finite() {
                Ok(value)
            } else {
                Err(FluxError::NonFinite(
                    "PAFI dividing-plane interpolant value",
                ))
            }
        };
        let mut best_u = 0.0_f64;
        let mut best_value = value_at(0.0)?;
        for u in roots.iter().take(nroots).copied().chain([1.0]) {
            let value = value_at(u)?;
            if value > best_value || (value.to_bits() == best_value.to_bits() && u < best_u) {
                best_u = u;
                best_value = value;
            }
        }
        Ok((best_u, best_value))
    }
}

/// Coefficients `[a, b, c, d]` of a positive multiple of the mean force
/// `a u^3 + b u^2 + c u + d` on a segment, with `u` in `[0, 1]`.
///
/// `g0` and `g1` are the plane means at the ends and `scaled` holds the end
/// slopes times the segment width. Every input is divided by the largest
/// magnitude first, so `g0 - g1` cannot overflow. The roots, which are all the
/// dividing-plane search needs, do not depend on that factor. All zero inputs
/// give all zero coefficients.
fn segment_force_coefficients(g0: f64, g1: f64, scaled: (f64, f64)) -> [f64; 4] {
    let magnitude = g0
        .abs()
        .max(g1.abs())
        .max(scaled.0.abs())
        .max(scaled.1.abs());
    if magnitude == 0.0 {
        return [0.0; 4];
    }
    let (g0, g1) = (g0 / magnitude, g1 / magnitude);
    let (s0, s1) = (scaled.0 / magnitude, scaled.1 / magnitude);
    [
        (g0 - g1).mul_add(2.0, s0 + s1),
        (g1 - g0).mul_add(3.0, -s0.mul_add(2.0, s1)),
        s0,
        g0,
    ]
}

/// Real roots of `a x^2 + b x + c = 0`.
///
/// `a == 0` is the linear equation `b x + c = 0`. `a == 0` and `b == 0`
/// leave a constant, which has no isolated root. The nonzero case uses
/// `q = -1/2 (b + copysign(sqrt(b^2 - 4 a c), b))`, then the roots `q / a`
/// and `c / q`, so a small root is not formed by cancelling two large terms.
fn quadratic_real_roots(a: f64, b: f64, c: f64) -> (usize, [f64; 2]) {
    let mut roots = [0.0; 2];
    if !a.is_finite() || !b.is_finite() || !c.is_finite() {
        return (0, roots);
    }
    if a == 0.0 {
        if b == 0.0 {
            return (0, roots);
        }
        let root = -c / b;
        if root.is_finite() {
            roots[0] = root;
            return (1, roots);
        }
        return (0, roots);
    }
    let discriminant = b.mul_add(b, -4.0 * a * c);
    if !discriminant.is_finite() {
        return (0, roots);
    }
    if discriminant < 0.0 {
        let scale = b.abs().max((a.abs() * c.abs()).sqrt());
        if discriminant.abs() > 8.0 * f64::EPSILON * scale * scale {
            return (0, roots);
        }
        let root = -b / (2.0 * a);
        if root.is_finite() {
            roots[0] = root;
            return (1, roots);
        }
        return (0, roots);
    }
    let q = -0.5 * (b + discriminant.sqrt().copysign(b));
    if q == 0.0 || !q.is_finite() {
        let root = -b / (2.0 * a);
        if root.is_finite() {
            roots[0] = root;
            return (1, roots);
        }
        return (0, roots);
    }
    let first = q / a;
    let second = c / q;
    let mut count = 0usize;
    if first.is_finite() {
        roots[count] = first;
        count += 1;
    }
    if second.is_finite() && (count == 0 || second.to_bits() != roots[0].to_bits()) {
        roots[count] = second;
        count += 1;
    }
    (count, roots)
}

/// Real roots in `[0, 1]` of `a x^3 + b x^2 + c x + d = 0`, in ascending order.
///
/// The critical points of the cubic are the roots of its derivative, a
/// quadratic that [`quadratic_real_roots`] solves without cancellation. The
/// critical points inside `(0, 1)` cut the interval into at most three pieces
/// on which the cubic is monotone, so each piece holds at most one root and a
/// change of sign locates it by bisection. Nothing is divided by the leading
/// coefficient and no cube root is taken, so a leading coefficient that is
/// zero, tiny, or comparable to the others needs no separate case. The
/// coefficients are first divided by the largest magnitude, which leaves the
/// roots unchanged and keeps the derivative quadratic in range. A root that
/// is an endpoint, or a critical point where the cubic is exactly zero, is
/// reported once. A double root is a critical point where the cubic does not
/// change sign and is not a maximum of its antiderivative, so a touch without
/// a crossing is not reported.
fn cubic_unit_interval_roots(a: f64, b: f64, c: f64, d: f64) -> (usize, [f64; 3]) {
    let mut roots = [0.0; 3];
    if !a.is_finite() || !b.is_finite() || !c.is_finite() || !d.is_finite() {
        return (0, roots);
    }
    let scale = a.abs().max(b.abs()).max(c.abs()).max(d.abs());
    if scale == 0.0 {
        return (0, roots);
    }
    let (a, b, c, d) = (a / scale, b / scale, c / scale, d / scale);
    let value = |x: f64| a.mul_add(x, b).mul_add(x, c).mul_add(x, d);

    let (turning_count, turning) = quadratic_real_roots(3.0 * a, 2.0 * b, c);
    let mut breaks = [0.0_f64; 4];
    let mut break_count = 1usize;
    let mut inner = [0.0_f64; 2];
    let mut inner_count = 0usize;
    for &point in turning.iter().take(turning_count) {
        if point > 0.0 && point < 1.0 {
            inner[inner_count] = point;
            inner_count += 1;
        }
    }
    inner[..inner_count].sort_by(f64::total_cmp);
    for &point in inner.iter().take(inner_count) {
        breaks[break_count] = point;
        break_count += 1;
    }
    breaks[break_count] = 1.0;
    break_count += 1;

    let mut count = 0usize;
    let mut push = |x: f64| {
        if count < roots.len() && (count == 0 || roots[count - 1] != x) {
            roots[count] = x;
            count += 1;
        }
    };
    let mut previous_value = value(breaks[0]);
    if previous_value == 0.0 {
        push(breaks[0]);
    }
    for piece in breaks[..break_count].windows(2) {
        let (low, high) = (piece[0], piece[1]);
        let high_value = value(high);
        if high_value == 0.0 {
            push(high);
        } else if previous_value != 0.0
            && !previous_value.is_nan()
            && !high_value.is_nan()
            && (previous_value < 0.0) != (high_value < 0.0)
        {
            push(bisect_sign_change(&value, low, high, previous_value < 0.0));
        }
        previous_value = high_value;
    }
    (count, roots)
}

/// Root of `value` in `[low, high]` where it is negative at `low` exactly when
/// `low_negative`, and has the opposite sign at `high`.
///
/// The bracket halves until its ends are adjacent floats. The end with the
/// smaller residual is returned.
fn bisect_sign_change(
    value: &impl Fn(f64) -> f64,
    mut low: f64,
    mut high: f64,
    low_negative: bool,
) -> f64 {
    for _ in 0..BISECTION_LIMIT {
        let middle = low + 0.5 * (high - low);
        if middle <= low || middle >= high {
            break;
        }
        let at_middle = value(middle);
        if at_middle == 0.0 {
            return middle;
        }
        if (at_middle < 0.0) == low_negative {
            low = middle;
        } else {
            high = middle;
        }
    }
    if value(low).abs() <= value(high).abs() {
        low
    } else {
        high
    }
}

/// Cumulative integral of a mean force, with the truncation the rule itself
/// can see.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct IntegralEstimate {
    /// Integral from the first node to each node. The first entry is `0`.
    pub values: Vec<f64>,
    /// Truncation at each node: the absolute difference against an embedded
    /// lower-order rule on the same nodes. The first entry is `0`. A later
    /// zero means the two rules agree there.
    pub error: Vec<f64>,
}

/// Integrates plane mean forces and reports a truncation taken from the rule.
///
/// A backend implements this for its own nodes and weights. The profile uses
/// [`HermiteIntegral`]. This crate does not provide a Gaussian-process rule.
pub trait MeanForceIntegral {
    /// Cumulative integral of `gradients` along strictly increasing `arcs`.
    ///
    /// `arcs` and `gradients` have the same length, at least 2.
    fn cumulative(&self, arcs: &[f64], gradients: &[f64]) -> Result<IntegralEstimate, FluxError>;
}

/// Cubic Hermite integral of the mean force.
///
/// The point values are the profile quadrature. The truncation is the
/// absolute difference from the trapezoid rule on the same nodes.
#[derive(Clone, Copy, Debug, Default)]
pub struct HermiteIntegral;

/// Composite Simpson integral of the mean force.
///
/// Each pair of intervals uses the quadratic interpolant of the three plane
/// values, including unequal spacing. An even plane count has an odd number
/// of intervals and is rejected, rather than integrated by another rule. The
/// truncation is the absolute difference from the trapezoid rule.
#[derive(Clone, Copy, Debug, Default)]
pub struct SimpsonIntegral;

fn same_length(arcs: &[f64], gradients: &[f64]) -> Result<(), FluxError> {
    if arcs.len() != gradients.len() {
        return Err(FluxError::Shape {
            expected: arcs.len(),
            got: gradients.len(),
            context: "PAFI mean-force integral",
        });
    }
    Ok(())
}

fn trapezoid_to(
    arcs: &[f64],
    gradients: &[f64],
    arc: f64,
    context: &'static str,
) -> Result<f64, FluxError> {
    if !arc.is_finite() {
        return Err(FluxError::NonFinite(context));
    }
    let mut accumulated = 0.0_f64;
    for k in 0..arcs.len() - 1 {
        let left = arcs[k];
        if arc <= left {
            break;
        }
        let width = arcs[k + 1] - left;
        let fraction = ((arc.min(arcs[k + 1]) - left) / width).clamp(0.0, 1.0);
        accumulated += width
            * fraction
            * (gradients[k] * (1.0 - 0.5 * fraction) + gradients[k + 1] * (0.5 * fraction));
        if arc <= arcs[k + 1] {
            break;
        }
    }
    if accumulated.is_finite() {
        Ok(accumulated)
    } else {
        Err(FluxError::NonFinite(context))
    }
}

fn trapezoid_cumulative(
    arcs: &[f64],
    gradients: &[f64],
    context: &'static str,
) -> Result<Vec<f64>, FluxError> {
    let mut values = vec![0.0_f64; arcs.len()];
    for k in 0..arcs.len() - 1 {
        values[k + 1] = trapezoid_to(arcs, gradients, arcs[k + 1], context)?;
    }
    Ok(values)
}

fn integral_estimate(
    values: Vec<f64>,
    lower: Vec<f64>,
    context: &'static str,
) -> Result<IntegralEstimate, FluxError> {
    if values.iter().chain(&lower).any(|value| !value.is_finite()) {
        return Err(FluxError::NonFinite(context));
    }
    let error: Vec<f64> = values
        .iter()
        .zip(&lower)
        .map(|(value, trapezoid)| (value - trapezoid).abs())
        .collect();
    if error.iter().any(|value| !value.is_finite()) {
        return Err(FluxError::NonFinite(context));
    }
    Ok(IntegralEstimate { values, error })
}

/// Integral of the quadratic through `(x0, g0)`, `(x1, g1)`, `(x2, g2)` from
/// `x0` to `end`.
fn quadratic_integral(x0: f64, x1: f64, x2: f64, g0: f64, g1: f64, g2: f64, end: f64) -> f64 {
    let left = x1 - x0;
    let right = x2 - x1;
    let divided_left = (g1 - g0) / left;
    let divided_right = (g2 - g1) / right;
    let second = (divided_right - divided_left) / (x2 - x0);
    let h = end - x0;
    g0 * h + divided_left * h * h / 2.0 + second * (h * h * h / 3.0 - left * h * h / 2.0)
}

impl MeanForceIntegral for HermiteIntegral {
    fn cumulative(&self, arcs: &[f64], gradients: &[f64]) -> Result<IntegralEstimate, FluxError> {
        same_length(arcs, gradients)?;
        if gradients.iter().any(|gradient| !gradient.is_finite()) {
            return Err(FluxError::NonFinite("PAFI Hermite mean force"));
        }
        let quadrature = Quadrature::new(arcs)?;
        let values = quadrature.integrate(gradients);
        let lower = trapezoid_cumulative(arcs, gradients, "PAFI Hermite mean-force integral")?;
        integral_estimate(values, lower, "PAFI Hermite mean-force integral")
    }
}

impl MeanForceIntegral for SimpsonIntegral {
    fn cumulative(&self, arcs: &[f64], gradients: &[f64]) -> Result<IntegralEstimate, FluxError> {
        same_length(arcs, gradients)?;
        let n = arcs.len();
        if n < 3 || n.is_multiple_of(2) {
            return Err(FluxError::InvalidConfig(
                "Simpson mean-force integration needs an odd number of planes, at least 3",
            ));
        }
        for pair in arcs.windows(2) {
            let ds = pair[1] - pair[0];
            if !ds.is_finite() || ds <= 0.0 {
                return Err(FluxError::InvalidConfig(
                    "PAFI arc lengths must strictly increase",
                ));
            }
        }
        if gradients.iter().any(|gradient| !gradient.is_finite()) {
            return Err(FluxError::NonFinite("PAFI Simpson mean force"));
        }
        let mut values = vec![0.0_f64; n];
        let mut accumulated = 0.0_f64;
        for pair in 0..(n - 1) / 2 {
            let i = pair * 2;
            let to_middle = quadratic_integral(
                arcs[i],
                arcs[i + 1],
                arcs[i + 2],
                gradients[i],
                gradients[i + 1],
                gradients[i + 2],
                arcs[i + 1],
            );
            let to_end = quadratic_integral(
                arcs[i],
                arcs[i + 1],
                arcs[i + 2],
                gradients[i],
                gradients[i + 1],
                gradients[i + 2],
                arcs[i + 2],
            );
            values[i + 1] = accumulated + to_middle;
            accumulated += to_end;
            values[i + 2] = accumulated;
        }
        let lower = trapezoid_cumulative(arcs, gradients, "PAFI Simpson mean-force integral")?;
        integral_estimate(values, lower, "PAFI Simpson mean-force integral")
    }
}

/// `Z_R = 2 \int_0^{s_div} exp(-F / kT) ds` by the trapezoid rule, in log space.
///
/// `s_div` is the right endpoint `s_{k+1}` of the segment `[s_k, s_{k+1}]` when
/// `s*` lies in `(s_k, s_{k+1}]`, or `s_k` when `s* = s_k`. `d_ln_zr_d_f[i]` is
/// `d ln Z_R / d F_i`.
fn reactant_log_partition_and_sensitivity(
    arc_lengths: &[f64],
    free_energy: &[f64],
    dividing: usize,
    kbt: f64,
) -> Result<(f64, Vec<f64>), FluxError> {
    let n = free_energy.len();
    let mut exponent_shift = f64::NEG_INFINITY;
    for &value in &free_energy[..=dividing] {
        let exponent = -value / kbt;
        if !exponent.is_finite() {
            return Err(FluxError::NonFinite("PAFI reactant partition exponent"));
        }
        exponent_shift = exponent_shift.max(exponent);
    }

    let mut scaled_half_partition = 0.0_f64;
    let mut scaled_numerator = vec![0.0_f64; n];
    for i in 1..=dividing {
        let ds = arc_lengths[i] - arc_lengths[i - 1];
        let left_exp = (-free_energy[i - 1] / kbt - exponent_shift).exp();
        let right_exp = (-free_energy[i] / kbt - exponent_shift).exp();
        let left_weight = 0.5 * ds * left_exp;
        let right_weight = 0.5 * ds * right_exp;
        scaled_half_partition += left_weight + right_weight;
        scaled_numerator[i - 1] += left_weight;
        scaled_numerator[i] += right_weight;
    }
    if !scaled_half_partition.is_finite() || scaled_half_partition <= 0.0 {
        return Err(FluxError::InvalidConfig(
            "PAFI reactant partition must be finite and positive",
        ));
    }

    let reactant_log_partition = LN_2 + exponent_shift + scaled_half_partition.ln();
    if !reactant_log_partition.is_finite() {
        return Err(FluxError::NonFinite("PAFI reactant log partition"));
    }
    let mut d_ln_zr_d_f = vec![0.0_f64; n];
    for i in 0..=dividing {
        // The ratio lies in `[0, 1]`, so dividing by `kbt` last cannot
        // underflow the product `kbt * scaled_half_partition`.
        d_ln_zr_d_f[i] = -(scaled_numerator[i] / scaled_half_partition) / kbt;
    }
    Ok((reactant_log_partition, d_ln_zr_d_f))
}

/// `ln(|left * right|)` without forming the product.
///
/// A zero factor yields negative infinity. A non-finite factor yields `NaN`.
fn ln_abs_product(left: f64, right: f64) -> f64 {
    if left == 0.0 || right == 0.0 {
        f64::NEG_INFINITY
    } else if left.is_finite() && right.is_finite() {
        left.abs().ln() + right.abs().ln()
    } else {
        f64::NAN
    }
}

/// Natural log of the Euclidean norm of terms given by log magnitude.
///
/// `0.5 * logsumexp(2 ln|t_i|)`. Scaling by the largest magnitude before the
/// exponential keeps a square that would overflow in linear space. Every term
/// zero yields negative infinity.
fn ln_euclidean_norm(ln_abs_terms: &[f64]) -> f64 {
    let mut max_log_square = f64::NEG_INFINITY;
    for &ln_abs in ln_abs_terms {
        if !ln_abs.is_finite() && ln_abs != f64::NEG_INFINITY {
            return f64::NAN;
        }
        let log_square = ln_abs + ln_abs;
        if log_square > max_log_square {
            max_log_square = log_square;
        }
    }
    if max_log_square == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    if !max_log_square.is_finite() {
        return f64::INFINITY;
    }
    let mut scaled_sum = 0.0_f64;
    for &ln_abs in ln_abs_terms {
        scaled_sum += (ln_abs + ln_abs - max_log_square).exp();
    }
    0.5 * (max_log_square + scaled_sum.ln())
}

/// `sqrt(sum (weight_i * sigma_i)^2)` without forming the squares.
///
/// A finite factor above about `1e154` makes `sigma * sigma` overflow. The
/// log-space norm keeps that finite value. Every term zero yields `0`. A
/// norm that does not fit in `f64` yields `+inf`.
fn linear_euclidean_norm(weights: &[f64], sigmas: &[f64]) -> f64 {
    let ln_abs_terms: Vec<f64> = weights
        .iter()
        .copied()
        .zip(sigmas.iter().copied())
        .map(|(weight, sigma)| ln_abs_product(weight, sigma))
        .collect();
    let ln_norm = ln_euclidean_norm(&ln_abs_terms);
    if ln_norm == f64::NEG_INFINITY {
        0.0
    } else {
        ln_norm.exp()
    }
}

/// Linear standard error `exp(log_rate + ln_sigma)`.
///
/// The linear rate is not an input. `Some(0.0)` when `ln_sigma` is negative
/// infinity or the exponential underflows to zero. `None` only when the
/// exponential is not finite.
fn linear_rate_error(log_rate: f64, ln_sigma: f64) -> Option<f64> {
    if ln_sigma == f64::NEG_INFINITY && log_rate.is_finite() {
        return Some(0.0);
    }
    let error = (log_rate + ln_sigma).exp();
    if error.is_finite() { Some(error) } else { None }
}

fn integrate_profile(
    quadrature: &Quadrature,
    gradients: &[f64],
    gradient_errors: &[f64],
    kbt: f64,
    reactant_curvature: f64,
    reactant_hessian_certified: bool,
) -> Result<FreeEnergyProfile, FluxError> {
    let n = quadrature.len();
    if gradients.len() != n || gradient_errors.len() != n {
        return Err(FluxError::Shape {
            expected: n,
            got: gradients.len().max(gradient_errors.len()),
            context: "PAFI profile integration vectors",
        });
    }
    if !kbt.is_finite() || kbt <= 0.0 {
        return Err(FluxError::InvalidConfig(
            "PAFI requires positive thermal energy",
        ));
    }
    for &sigma in gradient_errors {
        if !sigma.is_finite() || sigma < 0.0 {
            return Err(FluxError::InvalidConfig(
                "PAFI gradient standard errors must be finite and non-negative",
            ));
        }
    }
    if !reactant_curvature.is_finite() || reactant_curvature <= 0.0 {
        return Err(FluxError::InvalidConfig(
            "PAFI reactant curvature must be finite and positive",
        ));
    }
    let free_energy = quadrature.integrate(gradients);
    if free_energy.iter().any(|value| !value.is_finite()) {
        return Err(FluxError::NonFinite("PAFI integrated free energy"));
    }
    let lower = trapezoid_cumulative(&quadrature.arcs, gradients, "PAFI trapezoid free energy")?;
    let dividing = quadrature.dividing_plane(gradients, &free_energy)?;
    let standard_error: Vec<f64> = quadrature
        .cumulative
        .iter()
        .zip(&free_energy)
        .zip(&lower)
        .map(|((weights, hermite), trapezoid)| {
            linear_euclidean_norm(weights, gradient_errors).hypot((hermite - trapezoid).abs())
        })
        .collect();
    if standard_error
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(FluxError::NonFinite("PAFI free-energy standard error"));
    }
    let barrier = dividing.free_energy;
    let lower_barrier = trapezoid_to(
        &quadrature.arcs,
        gradients,
        dividing.arc,
        "PAFI trapezoid barrier",
    )?;
    let barrier_error = linear_euclidean_norm(&dividing.weights, gradient_errors)
        .hypot((barrier - lower_barrier).abs());
    if !barrier.is_finite() || !barrier_error.is_finite() || barrier_error < 0.0 {
        return Err(FluxError::NonFinite("PAFI barrier"));
    }

    let reactant_frequency = reactant_curvature.sqrt();
    if !reactant_frequency.is_finite() || reactant_frequency <= 0.0 {
        return Err(FluxError::InvalidConfig(
            "PAFI reactant frequency must be finite and positive",
        ));
    }
    // Z_R is the doubled trapezoid of exp(-F / kT) up to the dividing segment.
    // k = sqrt(kT / 2 pi) exp(-Delta F / kT) / Z_R. omega_0 is reported and is
    // not this normalization.
    let (reactant_log_partition, d_ln_zr_d_f) = reactant_log_partition_and_sensitivity(
        &quadrature.arcs,
        &free_energy,
        dividing.partition_end,
        kbt,
    )?;
    let reactant_partition = finite_exponential(reactant_log_partition);
    let normalized_log_rate = 0.5 * (kbt / TAU).ln() - barrier / kbt - reactant_log_partition;
    if !normalized_log_rate.is_finite() {
        return Err(FluxError::InvalidConfig(
            "PAFI normalized log rate must be finite",
        ));
    }
    let normalized_rate = finite_exponential(normalized_log_rate);
    let mut ln_abs_terms = Vec::with_capacity(n);
    for (gradient_index, sigma) in gradient_errors.iter().copied().enumerate() {
        let barrier_sensitivity = dividing.weights[gradient_index];
        let mut partition_log_sensitivity = 0.0_f64;
        for (sensitivity, cumulative) in d_ln_zr_d_f
            .iter()
            .zip(&quadrature.cumulative)
            .take(dividing.partition_end + 1)
        {
            partition_log_sensitivity += sensitivity * cumulative[gradient_index];
        }
        let log_rate_sensitivity = -barrier_sensitivity / kbt - partition_log_sensitivity;
        if !log_rate_sensitivity.is_finite() {
            return Err(FluxError::InvalidConfig(
                "PAFI normalized-rate logarithmic error must be finite and non-negative",
            ));
        }
        ln_abs_terms.push(ln_abs_product(log_rate_sensitivity, sigma));
    }
    let ln_sigma = ln_euclidean_norm(&ln_abs_terms);
    if ln_sigma.is_nan() || (ln_sigma.is_infinite() && ln_sigma.is_sign_positive()) {
        return Err(FluxError::InvalidConfig(
            "PAFI normalized-rate logarithmic error must be finite and non-negative",
        ));
    }
    let normalized_rate_error = linear_rate_error(normalized_log_rate, ln_sigma);

    Ok(FreeEnergyProfile {
        arc_lengths: quadrature.arcs.clone(),
        free_energy,
        standard_error,
        barrier,
        barrier_error,
        dividing_index: dividing.index,
        dividing_arc: dividing.arc,
        reactant_frequency,
        reactant_frequency_error: 0.0,
        reactant_hessian_certified,
        reactant_partition,
        reactant_log_partition,
        normalized_rate,
        normalized_log_rate,
        normalized_rate_error,
        quadrature_residual: 0.0,
        gradients: gradients.to_vec(),
        gradient_errors: gradient_errors.to_vec(),
    })
}

#[derive(Debug)]
struct HessianProductEstimate {
    value: Array1<f64>,
    roundoff_floor: f64,
}

/// Central difference of the gradient along `direction`, with the probe moved
/// `fraction` length units along the unit displacement.
///
/// `roundoff_floor` is the norm of the floating-point error that the
/// subtraction of the two gradients can carry, per unit of the product.
fn finite_difference_hessian_product<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    origin: ArrayView1<'_, f64>,
    root_mass: ArrayView1<'_, f64>,
    direction: ArrayView1<'_, f64>,
    fraction: f64,
    hessian_frobenius: f64,
) -> Result<HessianProductEstimate, FluxError> {
    let dof = origin.len();
    let mut displacement = Array1::zeros(dof);
    for i in 0..dof {
        displacement[i] = direction[i] / root_mass[i];
    }
    let length = nrm2(displacement.view());
    if !length.is_finite() || length <= f64::MIN_POSITIVE.sqrt() {
        return Err(FluxError::InvalidConfig(
            "PAFI reactant Hessian direction underflowed",
        ));
    }
    let scale = fraction / length;
    let mut gradients = Vec::with_capacity(2);
    let mut position_error_norms = Vec::with_capacity(2);
    let unit_roundoff = 0.5 * f64::EPSILON;
    for sign in [1.0_f64, -1.0] {
        let delta = displacement.mapv(|value| sign * scale * value);
        let probe = &origin.to_owned() + &delta;
        let mut mass_weighted_position_error_sq = 0.0;
        for i in 0..dof {
            // `delta[i]` incurs one multiplication rounding and adding it to
            // `origin[i]` incurs one addition rounding. The standard
            // round-to-nearest bound uses u = eps/2 for each operation.
            let delta_error = unit_roundoff * delta[i].abs();
            let addition_error =
                unit_roundoff * (origin[i].abs() + (1.0 + unit_roundoff) * delta[i].abs());
            let mass_weighted_error = root_mass[i] * (delta_error + addition_error);
            mass_weighted_position_error_sq += mass_weighted_error * mass_weighted_error;
        }
        position_error_norms.push(mass_weighted_position_error_sq.sqrt());
        engine.set_positions(probe.view())?;
        let (_, forces) = engine.energy_and_forces()?;
        gradients.push(forces.mapv(|force| -force));
    }
    let mut product = Array1::zeros(dof);
    let mut component_roundoff = Array1::zeros(dof);
    for i in 0..dof {
        product[i] = (gradients[0][i] - gradients[1][i]) / (2.0 * scale) / root_mass[i];
        component_roundoff[i] = f64::EPSILON * (gradients[0][i].abs() + gradients[1][i].abs())
            / (2.0 * scale)
            / root_mass[i];
    }
    if product.iter().any(|value| !value.is_finite()) {
        return Err(FluxError::NonFinite("PAFI reactant Hessian product"));
    }
    let gradient_roundoff = nrm2(component_roundoff.view());
    let position_roundoff =
        hessian_frobenius * (position_error_norms[0] + position_error_norms[1]) / (2.0 * scale);
    Ok(HessianProductEstimate {
        value: product,
        roundoff_floor: gradient_roundoff + position_roundoff,
    })
}

/// Mass-weighted Hessian-vector product `M^{-1/2} (d grad V) M^{-1/2} direction`.
///
/// The probe moves `fraction` length units along the unit displacement, with
/// `fraction` equal to [`hessian_displacement_fraction`]. The same step is
/// used for every direction, and it does not depend on the absolute
/// coordinates, so translating every coordinate leaves the product unchanged.
///
/// [`rgsaddle::numerical_hvp`] is not used. It needs a shared
/// [`rgsaddle::PointSurface`], and its step sign follows the absolute
/// position through `direction · x`. This path only has a [`DynamicsEngine`],
/// and the central difference above has no absolute position in the step.
fn mass_weighted_hessian_product<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    origin: ArrayView1<'_, f64>,
    root_mass: ArrayView1<'_, f64>,
    direction: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, FluxError> {
    let estimate = finite_difference_hessian_product(
        engine,
        origin,
        root_mass,
        direction,
        hessian_displacement_fraction(),
        0.0,
    )?;
    Ok(estimate.value)
}

fn mass_weighted_hessian<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    origin: ArrayView1<'_, f64>,
    root_mass: ArrayView1<'_, f64>,
) -> Result<Array2<f64>, FluxError> {
    let dof = origin.len();
    if dof == 0 || dof > REACTANT_HESSIAN_EIGEN_LIMIT {
        return Err(FluxError::InvalidConfig(
            "PAFI reactant Hessian dimension is outside the dense eigen range",
        ));
    }
    let mut hessian = Array2::zeros((dof, dof));
    for column in 0..dof {
        let mut direction = Array1::zeros(dof);
        direction[column] = 1.0;
        let product = mass_weighted_hessian_product(engine, origin, root_mass, direction.view())?;
        for row in 0..dof {
            hessian[(row, column)] = product[row];
        }
    }
    // Column `j` differences the gradient along coordinate `j`, so on an
    // anharmonic surface `H[(i, j)]` and `H[(j, i)]` carry different `O(h^2)`
    // truncation terms. The Hessian is symmetric, so average them.
    for row in 0..dof {
        for column in row + 1..dof {
            let mean = 0.5 * (hessian[(row, column)] + hessian[(column, row)]);
            hessian[(row, column)] = mean;
            hessian[(column, row)] = mean;
        }
    }
    Ok(hessian)
}

/// Check the first image and report whether its full Hessian was certified.
///
/// The forces at `origin` must be finite, and for `dof <= 128`
/// ([`REACTANT_HESSIAN_EIGEN_LIMIT`]) the mass-weighted Hessian is
/// diagonalized with [`rgsaddle::exact_eigh`]. Eigenvalues no larger than
/// [`reactant_zero_mode_floor`] times `1 +` the stiffest curvature are zero
/// modes. Every other eigenvalue must be positive, and at least one positive
/// eigenvalue must remain. Above the limit, finite forces are the cheap check
/// here and positive projected curvature is checked by [`curvature_with_modes`].
fn ensure_reactant_minimum<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    origin: ArrayView1<'_, f64>,
) -> Result<bool, FluxError> {
    engine.set_positions(origin)?;
    let (_, forces) = engine.energy_and_forces()?;
    for force in forces.iter().copied() {
        if !force.is_finite() {
            return Err(FluxError::NonFinite("PAFI reactant force"));
        }
    }
    if origin.len() > REACTANT_HESSIAN_EIGEN_LIMIT {
        return Ok(false);
    }
    let root_mass = engine.masses().mapv(f64::sqrt);
    let hessian = mass_weighted_hessian(engine, origin, root_mass.view())?;
    let (values, _) = rgsaddle::exact_eigh(hessian.view())
        .map_err(|_| FluxError::InvalidConfig("PAFI reactant Hessian eigensolve failed"))?;
    let stiffest = values
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let zero_floor = reactant_zero_mode_floor() * (1.0 + stiffest);
    let mut positive = 0usize;
    for value in values.iter().copied() {
        if !value.is_finite() {
            return Err(FluxError::NonFinite("PAFI reactant Hessian eigenvalue"));
        }
        if value.abs() <= zero_floor {
            continue;
        }
        if value < 0.0 {
            return Err(FluxError::InvalidConfig(
                "PAFI reactant Hessian is not positive definite off the zero modes",
            ));
        }
        positive += 1;
    }
    if positive == 0 {
        return Err(FluxError::InvalidConfig(
            "PAFI reactant Hessian has no positive curvature off the zero modes",
        ));
    }
    Ok(true)
}

/// Richardson-extrapolated Hessian product, its truncation uncertainty, and the
/// roundoff floor of the fine probe.
///
/// A centered derivative has `O(h^2)` truncation error, so the fine-grid error
/// estimate is `(fine - coarse) / (2^2 - 1)`. Adding it to the fine product
/// gives the extrapolated value.
fn estimate_hessian_product<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    origin: ArrayView1<'_, f64>,
    root_mass: ArrayView1<'_, f64>,
    direction: ArrayView1<'_, f64>,
    fraction: f64,
    hessian_frobenius: f64,
) -> Result<(Array1<f64>, f64, f64), FluxError> {
    let coarse = finite_difference_hessian_product(
        engine,
        origin,
        root_mass,
        direction,
        fraction,
        hessian_frobenius,
    )?;
    let fine = finite_difference_hessian_product(
        engine,
        origin,
        root_mass,
        direction,
        0.5 * fraction,
        hessian_frobenius,
    )?;
    let correction = (&fine.value - &coarse.value).mapv(|value| value / 3.0);
    let uncertainty = nrm2(correction.view());
    let value = &fine.value + &correction;
    // Richardson extrapolation is (4 * fine - coarse) / 3, so independent
    // absolute roundoff bounds propagate with those same absolute weights.
    let roundoff_floor = (4.0 * fine.roundoff_floor + coarse.roundoff_floor) / 3.0;
    Ok((value, uncertainty, roundoff_floor))
}

fn classify_translation_mode(
    axis: usize,
    response: f64,
    uncertainty: f64,
    roundoff_floor: f64,
    declaration: Option<bool>,
) -> Result<bool, FluxError> {
    if let Some(invariant) = declaration {
        return Ok(invariant);
    }
    let upper = response + uncertainty;
    let lower = (response - uncertainty).max(0.0);
    if upper <= roundoff_floor {
        return Ok(true);
    }
    if lower > roundoff_floor {
        return Ok(false);
    }
    Err(FluxError::AmbiguousTranslationMode {
        axis: ["x", "y", "z"][axis],
        response,
        uncertainty,
        roundoff_floor,
        ambiguity: uncertainty - (response - roundoff_floor).abs(),
    })
}

/// Mass-weighted unit translations accepted at `origin`.
///
/// `Some` on [`PafiConfig::translation_invariance`] declares each Cartesian
/// axis and skips the probe. `None` probes with the same centered
/// Hessian-vector products, position-roundoff floor, and uncertainty band as
/// the reactant curvature. Fewer than two atoms, or a degree-of-freedom count
/// that is not a multiple of three, returns no modes.
fn classified_translation_modes<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    origin: &Array1<f64>,
    root_mass: &Array1<f64>,
    translation_invariance: Option<[bool; 3]>,
) -> Result<Vec<Array1<f64>>, FluxError> {
    let dof = origin.len();
    let hessian_frobenius = if dof >= 6 && dof % 3 == 0 && translation_invariance.is_none() {
        let mut squared_bound = 0.0;
        for i in 0..dof {
            let mut basis = Array1::zeros(dof);
            basis[i] = 1.0;
            let (column, uncertainty, roundoff_floor) = estimate_hessian_product(
                engine,
                origin.view(),
                root_mass.view(),
                basis.view(),
                hessian_displacement_fraction(),
                0.0,
            )?;
            let column_bound = nrm2(column.view()) + uncertainty + roundoff_floor;
            squared_bound += column_bound * column_bound;
        }
        squared_bound.sqrt()
    } else {
        0.0
    };

    let mut translation_modes = Vec::new();
    if dof >= 6 && dof % 3 == 0 {
        for axis in 0..3 {
            let mut translation = Array1::zeros(dof);
            for i in (axis..dof).step_by(3) {
                translation[i] = root_mass[i];
            }
            let norm = nrm2(translation.view());
            translation.mapv_inplace(|value| value / norm);
            let declaration = translation_invariance.map(|axes| axes[axis]);
            let is_invariant = if declaration.is_some() {
                classify_translation_mode(axis, 0.0, 0.0, 0.0, declaration)?
            } else {
                let (product, uncertainty, roundoff_floor) = estimate_hessian_product(
                    engine,
                    origin.view(),
                    root_mass.view(),
                    translation.view(),
                    hessian_displacement_fraction(),
                    hessian_frobenius,
                )?;
                classify_translation_mode(
                    axis,
                    nrm2(product.view()),
                    uncertainty,
                    roundoff_floor,
                    None,
                )?
            };
            if is_invariant {
                translation_modes.push(translation);
            }
        }
    }
    Ok(translation_modes)
}

/// Squared projected reactant frequency `omega_0^2 = 1 / (n^T H^-1 n)`.
///
/// `H` is the mass-weighted Hessian at the first path image and `n` the unit
/// mass-weighted path tangent there. `H u = n` is solved by conjugate
/// gradients. The products are central differences of
/// [`DynamicsEngine::energy_and_forces`], so the solve stays generic over the
/// engine trait. [`rgsaddle::numerical_hvp`] is not used: it requires a
/// shared [`rgsaddle::PointSurface`], and this path only has a
/// [`DynamicsEngine`]. The translation modes removed from `n` and from each
/// Hessian product are the modes [`classified_translation_modes`] accepts, and
/// the mean-force Jacobian removes that same list.
#[cfg(test)]
fn reactant_curvature<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    geometry: &PlaneGeometry,
    translation_invariance: Option<[bool; 3]>,
) -> Result<f64, FluxError> {
    let root_mass: Array1<f64> = engine.masses().mapv(f64::sqrt);
    let origin = geometry.anchor.clone();
    let translation_modes =
        classified_translation_modes(engine, &origin, &root_mass, translation_invariance)?;
    curvature_with_modes(engine, geometry, &translation_modes)
}

fn curvature_with_modes<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    geometry: &PlaneGeometry,
    translation_modes: &[Array1<f64>],
) -> Result<f64, FluxError> {
    let dof = engine.dof();
    let root_mass: Array1<f64> = engine.masses().mapv(f64::sqrt);
    let origin = geometry.anchor.clone();
    let remove_translations = |vector: &mut Array1<f64>| {
        for translation in translation_modes {
            let component = dot(vector.view(), translation.view());
            vector.scaled_add(-component, translation);
        }
    };

    let mut normal = Array1::zeros(dof);
    for i in 0..dof {
        normal[i] = root_mass[i] * geometry.tangent[i];
    }
    let raw_normal_norm = nrm2(normal.view());
    remove_translations(&mut normal);
    let normal_norm = nrm2(normal.view());
    if !normal_norm.is_finite()
        || normal_norm
            <= (256.0 * f64::EPSILON.sqrt() * raw_normal_norm).max(f64::MIN_POSITIVE.sqrt())
    {
        let context = if translation_modes.is_empty() {
            "PAFI reactant normal underflowed"
        } else {
            "PAFI reactant tangent lies entirely in the translational zero-mode subspace"
        };
        return Err(FluxError::InvalidConfig(context));
    }
    normal.mapv_inplace(|value| value / normal_norm);

    let mut apply = |direction: &Array1<f64>| -> Result<Array1<f64>, FluxError> {
        let (mut product, _, _) = estimate_hessian_product(
            engine,
            origin.view(),
            root_mass.view(),
            direction.view(),
            hessian_displacement_fraction(),
            0.0,
        )?;
        remove_translations(&mut product);
        Ok(product)
    };

    let tolerance = f64::EPSILON.sqrt();
    let mut solution = Array1::<f64>::zeros(dof);
    let mut residual = normal.clone();
    let mut search = residual.clone();
    let mut residual_sq = dot(residual.view(), residual.view());
    let iterations = (2 * dof).max(8);
    for _ in 0..iterations {
        if residual_sq.sqrt() <= tolerance {
            break;
        }
        let image = apply(&search)?;
        let curvature = dot(search.view(), image.view());
        if !curvature.is_finite() || curvature <= 0.0 {
            return Err(FluxError::InvalidConfig(
                "PAFI reactant Hessian is not positive definite along the path normal",
            ));
        }
        let alpha = residual_sq / curvature;
        solution.scaled_add(alpha, &search);
        residual.scaled_add(-alpha, &image);
        let next_sq = dot(residual.view(), residual.view());
        if !next_sq.is_finite() {
            return Err(FluxError::NonFinite("PAFI reactant Hessian residual"));
        }
        let beta = next_sq / residual_sq;
        residual_sq = next_sq;
        search = &residual + &search.mapv(|value| beta * value);
    }
    if residual_sq.sqrt() > tolerance {
        return Err(FluxError::InvalidConfig(
            "PAFI reactant Hessian solve did not converge",
        ));
    }
    let inverse_curvature = dot(normal.view(), solution.view());
    if !inverse_curvature.is_finite() || inverse_curvature <= 0.0 {
        return Err(FluxError::InvalidConfig(
            "PAFI reactant curvature along the path normal must be positive",
        ));
    }
    Ok(1.0 / inverse_curvature)
}

/// Relative disagreement between the trapezoid integral of `dV/ds` over the
/// planes and the end-to-end energy change, in units of `max(1, |Delta E_0|)`.
fn path_force_relative_error(arcs: &[f64], energies: &[f64], gradients: &[f64]) -> f64 {
    let mut integrated_energy = 0.0_f64;
    for i in 1..arcs.len() {
        let ds = arcs[i] - arcs[i - 1];
        integrated_energy += 0.5 * ds * (gradients[i - 1] + gradients[i]);
    }
    let endpoint_energy = energies[energies.len() - 1] - energies[0];
    let discrepancy = (integrated_energy - endpoint_energy).abs();
    discrepancy / 1.0_f64.max(endpoint_energy.abs())
}

/// Integrate the PAFI free-energy profile and normalized rate along `path_images`.
///
/// `path_images` has shape `(images, dof)`, one row per image, in the length
/// units of the engine, and `dof` must equal the number of degrees of freedom
/// of `engine`. The module documentation describes the method, the units and
/// the restoration of the engine on return.
///
/// # Errors
///
/// Returns [`FluxError::InvalidConfig`] when
///
/// - `time_step` is not finite and positive, `burn_in_steps` is `0`,
///   `sample_steps` is below `2`, `max_rounds` is `0`, `planes` is below `2`,
///   `zero_temperature_tol` is `Some` of a negative or non-finite value, or
///   `target_error` is negative or not finite,
/// - `path_images` has fewer than two rows, or two neighbouring images
///   coincide in the mass-weighted metric,
/// - the thermal energy `k_B T` of the engine is not positive,
/// - the thermostat friction is `0`, which is NVE dynamics and does not
///   sample the canonical ensemble the plane means are defined over,
/// - the zero-temperature check of [`PafiConfig::zero_temperature_tol`] fails,
/// - a plane normal degenerates, the mass-weighted Hessian at the first image
///   is not positive definite once zero modes are excluded when `dof <= 128`,
///   the reactant curvature along the first path normal is not finite and
///   positive, or the dividing segment has zero width.
///
/// [`FreeEnergyProfile::normalized_log_rate`] and
/// [`FreeEnergyProfile::reactant_log_partition`] are authoritative for every
/// finite value. A finite log whose exponential overflows or underflows does
/// not reject the profile: [`FreeEnergyProfile::normalized_rate`] and
/// [`FreeEnergyProfile::reactant_partition`] are then `None`, and
/// [`FreeEnergyProfile::normalized_rate_error`] is `None` only when
/// `exp(normalized_log_rate + ln(sigma_ln))` is not finite.
///
/// Returns [`FluxError::EmptyInput`] for an empty coordinate axis and
/// [`FluxError::Shape`] when the engine masses do not match the path
/// coordinates. [`FluxError::NonFinite`] reports non-finite path coordinates,
/// constraint forces, force samples, profile values, plane means, Lagrange
/// weights, dividing-plane interpolant slopes or values, or reactant
/// partition exponents `-F(s_k) / k_B T`. Errors
/// from the engine and from the batch-means estimator are passed through, for
/// example an invalid [`StatsConfig`] or an engine that does not report the
/// constraint force on a constrained step.
pub fn pafi_profile<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    path_images: ArrayView2<'_, f64>,
    config: &PafiConfig,
) -> Result<FreeEnergyProfile, FluxError> {
    if !config.time_step.is_finite() || config.time_step <= 0.0 {
        return Err(FluxError::InvalidConfig(
            "PafiConfig::time_step must be finite and positive",
        ));
    }
    if config.burn_in_steps == 0 {
        return Err(FluxError::InvalidConfig(
            "PafiConfig::burn_in_steps must be at least 1",
        ));
    }
    if config.sample_steps < 2 {
        return Err(FluxError::InvalidConfig(
            "PafiConfig::sample_steps must be at least 2",
        ));
    }
    if config.max_rounds == 0 {
        return Err(FluxError::InvalidConfig(
            "PafiConfig::max_rounds must be at least 1",
        ));
    }
    if let Some(tol) = config.zero_temperature_tol
        && (!tol.is_finite() || tol < 0.0)
    {
        return Err(FluxError::InvalidConfig(
            "PafiConfig::zero_temperature_tol must be finite and non-negative",
        ));
    }
    if let Some(target) = config.target_error
        && (!target.is_finite() || target < 0.0)
    {
        return Err(FluxError::InvalidConfig(
            "PafiConfig::target_error must be finite and non-negative",
        ));
    }

    let kbt = engine.thermostat().require_canonical_sampling()?;
    if kbt <= 0.0 {
        return Err(FluxError::InvalidConfig(
            "PAFI requires positive thermal energy",
        ));
    }

    let snapshot = EngineSnapshot::capture(engine);
    let result = (|| {
        let masses = engine.masses().to_owned();
        let geometries = build_plane_geometries(path_images, masses.view(), config.planes)?;
        let arcs: Vec<f64> = geometries.iter().map(|geometry| geometry.arc).collect();

        engine.set_constraint(None)?;
        let mut path_gradients = Vec::with_capacity(geometries.len());
        let mut path_energies = Vec::with_capacity(geometries.len());
        for geometry in &geometries {
            engine.set_positions(geometry.anchor.view())?;
            let (energy, forces) = engine.energy_and_forces()?;
            path_energies.push(energy);
            path_gradients.push(-dot(geometry.tangent.view(), forces.view()));
        }
        let relative_error = path_force_relative_error(&arcs, &path_energies, &path_gradients);
        if relative_error > effective_zero_temperature_tol(config) {
            return Err(FluxError::InvalidConfig(
                "PAFI zero-temperature path-force quadrature exceeds tolerance",
            ));
        }
        let quadrature = Quadrature::new(&arcs)?;
        let integrated_energy = quadrature.integrate(&path_gradients);
        let mut quadrature_residual = 0.0_f64;
        for (integral, energy) in integrated_energy.iter().zip(path_energies.iter()) {
            let discrepancy = (integral - (energy - path_energies[0])).abs();
            if !discrepancy.is_finite() {
                return Err(FluxError::NonFinite("PAFI path-force quadrature residual"));
            }
            quadrature_residual = quadrature_residual.max(discrepancy);
        }
        let reactant_hessian_certified =
            ensure_reactant_minimum(engine, geometries[0].anchor.view())?;
        let root_mass = engine.masses().mapv(f64::sqrt);
        let origin = geometries[0].anchor.clone();
        let translation_modes = classified_translation_modes(
            engine,
            &origin,
            &root_mass,
            config.translation_invariance,
        )?;
        let curvature = curvature_with_modes(engine, &geometries[0], &translation_modes)?;

        let mut plane_samples = vec![Vec::new(); geometries.len()];
        let mut final_profile = None;
        for round in 0..config.max_rounds {
            for (plane_index, geometry) in geometries.iter().enumerate() {
                let samples = sample_plane(
                    engine,
                    geometry,
                    config,
                    plane_index,
                    round,
                    &translation_modes,
                )?;
                plane_samples[plane_index].extend(samples);
            }
            let estimates: Vec<PlaneEstimate> = plane_samples
                .iter()
                .map(|samples| estimate_plane(samples, config))
                .collect::<Result<_, _>>()?;
            let gradients: Vec<f64> = estimates.iter().map(|estimate| estimate.gradient).collect();
            let errors: Vec<f64> = estimates
                .iter()
                .map(|estimate| estimate.standard_error)
                .collect();
            let mut profile = integrate_profile(
                &quadrature,
                &gradients,
                &errors,
                kbt,
                curvature,
                reactant_hessian_certified,
            )?;
            profile.quadrature_residual = quadrature_residual;
            let target_reached = match config.target_error {
                Some(target) => profile.barrier_error <= target,
                None => true,
            };
            final_profile = Some(profile);
            if target_reached {
                break;
            }
        }
        match final_profile {
            Some(profile) => Ok(profile),
            None => Err(FluxError::InvalidConfig(
                "PAFI completed no sampling rounds",
            )),
        }
    })();
    snapshot.restore_result(engine, result)
}

/// Trait-object entry point for [`pafi_profile`] over [`DynamicsEngine`].
///
/// The algorithm, errors and restoration of the engine are those of
/// [`pafi_profile`]. This function only fixes the engine type to the trait
/// object, so one call site serves native, `rgpot` and session engines.
pub fn pafi_profile_dyn(
    engine: &mut dyn DynamicsEngine,
    path_images: ArrayView2<'_, f64>,
    config: &PafiConfig,
) -> Result<FreeEnergyProfile, FluxError> {
    pafi_profile(engine, path_images, config)
}

/// Monitor observable driving [`EquidistributionMap`] plane placement.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum MonitorKind {
    /// Magnitude and local variation of the free-energy gradient (`|dF/ds|` and `|d^2F/ds^2|`).
    #[default]
    ForceGradient,
    /// Euclidean norm of the hyperplane normal curvature (`||dn/ds||_2`).
    PathCurvature,
    /// Integrated-autocorrelation-weighted variance (`2 * tau_int * Var(f)`) of the constraint force.
    AutocorrelatedVariance,
    /// Equal-weight combination of normalized force gradient, path curvature, and `tau_int`-weighted variance.
    Combined,
}

/// Per-knot observable series from which [`MonitorKind`] builds a non-negative monitor.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct MonitorInputs<'a> {
    /// Knot arc lengths `s_0 < s_1 < ... < s_{M-1}`.
    pub arc_lengths: &'a [f64],
    /// Free-energy or potential gradients `dF/ds` at each knot.
    pub force_gradients: Option<&'a [f64]>,
    /// Normal-curvature magnitudes `||dn/ds||_2` at each knot.
    pub path_curvatures: Option<&'a [f64]>,
    /// Autocorrelation-weighted constraint-force variances `2 * tau_int * Var(f)` at each knot.
    pub tau_weighted_variances: Option<&'a [f64]>,
}

impl<'a> MonitorInputs<'a> {
    /// Create monitor inputs on `arc_lengths`.
    #[must_use]
    pub const fn new(arc_lengths: &'a [f64]) -> Self {
        Self {
            arc_lengths,
            force_gradients: None,
            path_curvatures: None,
            tau_weighted_variances: None,
        }
    }

    /// Attach free-energy or potential gradients `dF/ds` at each knot.
    #[must_use]
    pub const fn with_force_gradients(mut self, force_gradients: &'a [f64]) -> Self {
        self.force_gradients = Some(force_gradients);
        self
    }

    /// Attach hyperplane normal-curvature norms `||dn/ds||_2` at each knot.
    #[must_use]
    pub const fn with_path_curvatures(mut self, path_curvatures: &'a [f64]) -> Self {
        self.path_curvatures = Some(path_curvatures);
        self
    }

    /// Attach autocorrelation-weighted constraint-force variances `2 * tau_int * Var(f)` at each knot.
    #[must_use]
    pub const fn with_tau_weighted_variances(mut self, tau_weighted_variances: &'a [f64]) -> Self {
        self.tau_weighted_variances = Some(tau_weighted_variances);
        self
    }

    /// Evaluate the non-negative monitor values for `kind` at each knot.
    pub fn evaluate(&self, kind: MonitorKind) -> Result<Vec<f64>, FluxError> {
        let m = self.arc_lengths.len();
        if m < 2 {
            return Err(FluxError::InvalidConfig(
                "MonitorInputs::arc_lengths requires at least two knots",
            ));
        }
        for (i, &s) in self.arc_lengths.iter().enumerate() {
            if !s.is_finite() {
                return Err(FluxError::NonFinite("MonitorInputs::arc_lengths"));
            }
            if i > 0 && s <= self.arc_lengths[i - 1] {
                return Err(FluxError::InvalidConfig(
                    "MonitorInputs::arc_lengths must strictly increase",
                ));
            }
        }
        let total = self.arc_lengths[m - 1] - self.arc_lengths[0];
        if !total.is_finite() || total <= f64::MIN_POSITIVE {
            return Err(FluxError::InvalidConfig(
                "MonitorInputs::arc_lengths total span must be positive",
            ));
        }

        let compute_force = |grads: &[f64]| -> Result<Vec<f64>, FluxError> {
            if grads.len() != m {
                return Err(FluxError::Shape {
                    expected: m,
                    got: grads.len(),
                    context: "MonitorInputs::force_gradients",
                });
            }
            for &g in grads {
                if !g.is_finite() {
                    return Err(FluxError::NonFinite("MonitorInputs::force_gradients"));
                }
            }
            let mut out = Vec::with_capacity(m);
            for k in 0..m {
                let slope = if k == 0 {
                    (grads[1] - grads[0]).abs() / (self.arc_lengths[1] - self.arc_lengths[0])
                } else if k == m - 1 {
                    (grads[m - 1] - grads[m - 2]).abs()
                        / (self.arc_lengths[m - 1] - self.arc_lengths[m - 2])
                } else {
                    let left = (grads[k] - grads[k - 1]).abs()
                        / (self.arc_lengths[k] - self.arc_lengths[k - 1]);
                    let right = (grads[k + 1] - grads[k]).abs()
                        / (self.arc_lengths[k + 1] - self.arc_lengths[k]);
                    0.5 * (left + right)
                };
                let local_ds = if k == 0 {
                    self.arc_lengths[1] - self.arc_lengths[0]
                } else if k == m - 1 {
                    self.arc_lengths[m - 1] - self.arc_lengths[m - 2]
                } else {
                    0.5 * (self.arc_lengths[k + 1] - self.arc_lengths[k - 1])
                };
                out.push(grads[k].abs() + 0.5 * local_ds * slope);
            }
            Ok(out)
        };

        let compute_nonneg =
            |slice: &[f64], context: &'static str| -> Result<Vec<f64>, FluxError> {
                if slice.len() != m {
                    return Err(FluxError::Shape {
                        expected: m,
                        got: slice.len(),
                        context,
                    });
                }
                for &v in slice {
                    if !v.is_finite() {
                        return Err(FluxError::NonFinite(context));
                    }
                    if v < 0.0 {
                        return Err(FluxError::InvalidConfig(
                            "MonitorInputs values must be non-negative",
                        ));
                    }
                }
                Ok(slice.to_vec())
            };

        match kind {
            MonitorKind::ForceGradient => {
                let grads = self.force_gradients.ok_or(FluxError::InvalidConfig(
                    "MonitorKind::ForceGradient requires force_gradients",
                ))?;
                compute_force(grads)
            }
            MonitorKind::PathCurvature => {
                let curv = self.path_curvatures.ok_or(FluxError::InvalidConfig(
                    "MonitorKind::PathCurvature requires path_curvatures",
                ))?;
                compute_nonneg(curv, "MonitorInputs::path_curvatures")
            }
            MonitorKind::AutocorrelatedVariance => {
                let var = self.tau_weighted_variances.ok_or(FluxError::InvalidConfig(
                    "MonitorKind::AutocorrelatedVariance requires tau_weighted_variances",
                ))?;
                compute_nonneg(var, "MonitorInputs::tau_weighted_variances")
            }
            MonitorKind::Combined => {
                let mut channels: Vec<Vec<f64>> = Vec::new();
                if let Some(grads) = self.force_gradients {
                    channels.push(compute_force(grads)?);
                }
                if let Some(curv) = self.path_curvatures {
                    channels.push(compute_nonneg(curv, "MonitorInputs::path_curvatures")?);
                }
                if let Some(var) = self.tau_weighted_variances {
                    channels.push(compute_nonneg(
                        var,
                        "MonitorInputs::tau_weighted_variances",
                    )?);
                }
                if channels.is_empty() {
                    return Err(FluxError::InvalidConfig(
                        "MonitorKind::Combined requires at least one monitor channel",
                    ));
                }
                let mut combined = vec![0.0_f64; m];
                let mut active = 0usize;
                for ch in &channels {
                    let mut integral = 0.0_f64;
                    for j in 0..(m - 1) {
                        let ds = self.arc_lengths[j + 1] - self.arc_lengths[j];
                        integral += 0.5 * (ch[j] + ch[j + 1]) * ds;
                    }
                    let mean = integral / total;
                    if mean > f64::MIN_POSITIVE {
                        active += 1;
                        for k in 0..m {
                            combined[k] += ch[k] / mean;
                        }
                    }
                }
                if active > 0 {
                    let scale = 1.0 / (active as f64);
                    for v in &mut combined {
                        *v *= scale;
                    }
                }
                Ok(combined)
            }
        }
    }
}

/// Monotone arc-length reparametrization `s -> sigma(s)` in `[0, 1]` that
/// equidistributes a regularized positive monitor function.
///
/// Following the equal-measure defensive-mixture principle (de Boor 1973),
/// the monitor floor is set equal to the path-average monitor
/// `w_bar = (1 / L) \int_0^L w(s) ds`. Half of the total coordinate measure
/// `sigma` comes from uniform arc length `ds / L` and half from the normalized
/// monitor `w(s) ds / (w_bar L)` (reducing to pure uniform arc length when
/// `w_bar == 0`).
#[derive(Clone, Debug, PartialEq)]
pub struct EquidistributionMap {
    knots: Vec<f64>,
    sigmas: Vec<f64>,
    raw_monitor: Vec<f64>,
    floor: f64,
    max_monitor: f64,
}

impl EquidistributionMap {
    /// Construct an equidistribution map from strictly increasing `arc_lengths`
    /// and non-negative `raw_monitor` values at those knots.
    pub fn new(arc_lengths: &[f64], raw_monitor: &[f64]) -> Result<Self, FluxError> {
        let m = arc_lengths.len();
        if m < 2 {
            return Err(FluxError::InvalidConfig(
                "EquidistributionMap requires at least two knots",
            ));
        }
        if raw_monitor.len() != m {
            return Err(FluxError::Shape {
                expected: m,
                got: raw_monitor.len(),
                context: "EquidistributionMap monitor versus arc_lengths",
            });
        }
        for i in 0..m {
            if !arc_lengths[i].is_finite() {
                return Err(FluxError::NonFinite("EquidistributionMap arc_lengths"));
            }
            if i > 0 && arc_lengths[i] <= arc_lengths[i - 1] {
                return Err(FluxError::InvalidConfig(
                    "EquidistributionMap arc_lengths must strictly increase",
                ));
            }
            if !raw_monitor[i].is_finite() {
                return Err(FluxError::NonFinite("EquidistributionMap raw_monitor"));
            }
            if raw_monitor[i] < 0.0 {
                return Err(FluxError::InvalidConfig(
                    "EquidistributionMap monitor values must be non-negative",
                ));
            }
        }
        let s0 = arc_lengths[0];
        let total = arc_lengths[m - 1] - s0;
        if !total.is_finite() || total <= f64::MIN_POSITIVE {
            return Err(FluxError::InvalidConfig(
                "EquidistributionMap total arc length must be positive",
            ));
        }

        let mut integral = 0.0_f64;
        let mut max_monitor = 0.0_f64;
        for i in 0..m {
            if raw_monitor[i] > max_monitor {
                max_monitor = raw_monitor[i];
            }
            if i + 1 < m {
                let ds = arc_lengths[i + 1] - arc_lengths[i];
                integral += 0.5 * (raw_monitor[i] + raw_monitor[i + 1]) * ds;
            }
        }
        let mean_monitor = integral / total;
        let floor = if mean_monitor > f64::MIN_POSITIVE {
            mean_monitor
        } else {
            1.0
        };

        let mut sigmas = vec![0.0_f64; m];
        if mean_monitor <= f64::MIN_POSITIVE {
            for i in 1..(m - 1) {
                sigmas[i] = (arc_lengths[i] - s0) / total;
            }
            sigmas[m - 1] = 1.0;
        } else {
            let mut cum = 0.0_f64;
            for i in 0..(m - 1) {
                let ds = arc_lengths[i + 1] - arc_lengths[i];
                let w_mid = 0.5 * (raw_monitor[i] + raw_monitor[i + 1]);
                cum += 0.5 * (ds / total) + 0.5 * (w_mid * ds / integral);
                sigmas[i + 1] = cum;
            }
            let norm = sigmas[m - 1];
            for val in &mut sigmas[1..(m - 1)] {
                *val /= norm;
            }
            sigmas[m - 1] = 1.0;
        }

        Ok(Self {
            knots: arc_lengths.to_vec(),
            sigmas,
            raw_monitor: raw_monitor.to_vec(),
            floor,
            max_monitor,
        })
    }

    /// Construct an equidistribution map from [`MonitorInputs`] and [`MonitorKind`].
    pub fn from_inputs(inputs: &MonitorInputs<'_>, kind: MonitorKind) -> Result<Self, FluxError> {
        let monitor = inputs.evaluate(kind)?;
        Self::new(inputs.arc_lengths, &monitor)
    }

    /// Evaluate the normalized coordinate `sigma(s)` in `[0, 1]`.
    pub fn sigma(&self, s: f64) -> Result<f64, FluxError> {
        if !s.is_finite() {
            return Err(FluxError::NonFinite("EquidistributionMap::sigma input"));
        }
        let m = self.knots.len();
        let s0 = self.knots[0];
        let s_end = self.knots[m - 1];
        let tol = 64.0 * f64::EPSILON * (1.0 + s0.abs() + s_end.abs());
        if s < s0 - tol || s > s_end + tol {
            return Err(FluxError::InvalidConfig(
                "EquidistributionMap::sigma argument is outside knot range",
            ));
        }
        if s <= s0 {
            return Ok(0.0);
        }
        if s >= s_end {
            return Ok(1.0);
        }
        let mut seg = m - 2;
        for i in 0..(m - 1) {
            if s <= self.knots[i + 1] {
                seg = i;
                break;
            }
        }
        let ds = self.knots[seg + 1] - self.knots[seg];
        let frac = (s - self.knots[seg]) / ds;
        Ok(self.sigmas[seg] + frac * (self.sigmas[seg + 1] - self.sigmas[seg]))
    }

    /// Invert `sigma` in `[0, 1]` back to the mass-weighted arc length `s(sigma)`.
    pub fn inverse(&self, sigma: f64) -> Result<f64, FluxError> {
        if !sigma.is_finite() {
            return Err(FluxError::NonFinite("EquidistributionMap::inverse input"));
        }
        let tol = 64.0 * f64::EPSILON;
        if sigma < -tol || sigma > 1.0 + tol {
            return Err(FluxError::InvalidConfig(
                "EquidistributionMap::inverse argument must lie in [0, 1]",
            ));
        }
        let m = self.knots.len();
        if sigma <= 0.0 {
            return Ok(self.knots[0]);
        }
        if sigma >= 1.0 {
            return Ok(self.knots[m - 1]);
        }
        let mut seg = m - 2;
        for i in 0..(m - 1) {
            if sigma <= self.sigmas[i + 1] {
                seg = i;
                break;
            }
        }
        let dsig = self.sigmas[seg + 1] - self.sigmas[seg];
        let frac = if dsig > 0.0 {
            (sigma - self.sigmas[seg]) / dsig
        } else {
            0.0
        };
        Ok(self.knots[seg] + frac * (self.knots[seg + 1] - self.knots[seg]))
    }

    fn is_reflection_symmetric(&self) -> bool {
        let m = self.knots.len();
        let s0 = self.knots[0];
        let s_end = self.knots[m - 1];
        let span = s_end - s0;
        let knot_tol = 64.0 * f64::EPSILON * (1.0 + span);
        let mon_tol = 64.0 * f64::EPSILON * (1.0 + self.max_monitor);
        for i in 0..(m / 2) {
            let left_s = self.knots[i] - s0;
            let right_s = s_end - self.knots[m - 1 - i];
            if (left_s - right_s).abs() > knot_tol {
                return false;
            }
            if (self.raw_monitor[i] - self.raw_monitor[m - 1 - i]).abs() > mon_tol {
                return false;
            }
        }
        true
    }

    /// Place `planes` (`>= 2`) hyperplanes at equal increments of `sigma` in `[0, 1]`.
    pub fn place_planes(&self, planes: usize) -> Result<Vec<f64>, FluxError> {
        if planes < 2 {
            return Err(FluxError::InvalidConfig(
                "EquidistributionMap::place_planes requires at least 2 planes",
            ));
        }
        let m = self.knots.len();
        let s0 = self.knots[0];
        let s_end = self.knots[m - 1];
        let mut out = vec![0.0_f64; planes];
        out[0] = s0;
        out[planes - 1] = s_end;
        for (k, slot) in out.iter_mut().enumerate().take(planes - 1).skip(1) {
            let u = (k as f64) / ((planes - 1) as f64);
            *slot = self.inverse(u)?;
        }
        if self.is_reflection_symmetric() {
            for k in 1..=(planes / 2) {
                let mirror = planes - 1 - k;
                if k == mirror {
                    out[k] = 0.5 * (s0 + s_end);
                } else {
                    let left = out[k] - s0;
                    let right = s_end - out[mirror];
                    let avg = 0.5 * (left + right);
                    out[k] = s0 + avg;
                    out[mirror] = s_end - avg;
                }
            }
        }
        Ok(out)
    }

    /// Principled monitor floor `w_bar = (1 / L) \int_0^L w(s) ds` (or `1.0` when `w_bar == 0`).
    #[must_use]
    pub const fn floor(&self) -> f64 {
        self.floor
    }

    /// Certified lower bound on the spacing `min_k (s_k - s_{k-1})` for `planes` planes.
    pub fn min_spacing_bound(&self, planes: usize) -> Result<f64, FluxError> {
        if planes < 2 {
            return Err(FluxError::InvalidConfig(
                "EquidistributionMap::min_spacing_bound requires at least 2 planes",
            ));
        }
        let total = self.knots[self.knots.len() - 1] - self.knots[0];
        let uniform_step = total / ((planes - 1) as f64);
        if self.max_monitor <= f64::MIN_POSITIVE {
            return Ok(uniform_step);
        }
        Ok(uniform_step * (2.0 * self.floor) / (self.floor + self.max_monitor))
    }

    /// Input arc-length knots `s_j`.
    #[must_use]
    pub fn knots(&self) -> &[f64] {
        &self.knots
    }

    /// Normalized coordinates `sigma(s_j)` at the input knots.
    #[must_use]
    pub fn knot_sigmas(&self) -> &[f64] {
        &self.sigmas
    }

    /// Raw monitor values at [`Self::knots`] before defensive regularization.
    #[must_use]
    pub fn monitor_values(&self) -> &[f64] {
        &self.raw_monitor
    }
}

/// Chebyshev / cosine spectral expansion and heuristic resolution indicator.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct SpectralTailReport {
    /// Chebyshev expansion coefficients `a_m` of the mean constraint force `dF/ds`
    /// on `x = 2 * (s - s_0) / L - 1` (`T_m(x) = cos(m * arccos(x))`).
    pub force_coefficients: Vec<f64>,
    /// Chebyshev expansion coefficients `c_m` of the integrated free energy `F(s)`.
    pub profile_coefficients: Vec<f64>,
    /// Heuristic indicator of unresolved spectral / quadrature error in `F*`.
    ///
    /// This value is not a rigorous upper bound on interpolation error.
    pub tail_indicator: f64,
    /// Per-interval resolution error indicator `epsilon_k` (`k = 0..planes - 1`).
    pub interval_errors: Vec<f64>,
    /// Indices of intervals `[s_k, s_{k+1}]` flagged as needing additional planes.
    pub flagged_intervals: Vec<usize>,
}

fn chebyshev_eval(coeffs: &[f64], x: f64) -> f64 {
    if coeffs.is_empty() {
        return 0.0;
    }
    if coeffs.len() == 1 {
        return coeffs[0];
    }
    let x_clamped = x.clamp(-1.0, 1.0);
    let mut b_k1 = 0.0_f64;
    let mut b_k2 = 0.0_f64;
    for &c in coeffs[1..].iter().rev() {
        let next = 2.0 * x_clamped * b_k1 - b_k2 + c;
        b_k2 = b_k1;
        b_k1 = next;
    }
    x_clamped * b_k1 - b_k2 + coeffs[0]
}

fn hermite_slope_basis(arc_lengths: &[f64]) -> Vec<Vec<f64>> {
    let n = arc_lengths.len();
    let mut slopes = vec![vec![0.0_f64; n]; n];
    if n == 2 {
        let h = arc_lengths[1] - arc_lengths[0];
        slopes[0][0] = -1.0 / h;
        slopes[0][1] = 1.0 / h;
        slopes[1][0] = -1.0 / h;
        slopes[1][1] = 1.0 / h;
        return slopes;
    }
    let h0 = arc_lengths[1] - arc_lengths[0];
    let h1 = arc_lengths[2] - arc_lengths[1];
    slopes[0][0] = -(2.0 * h0 + h1) / (h0 * (h0 + h1));
    slopes[0][1] = (h0 + h1) / (h0 * h1);
    slopes[0][2] = -h0 / (h1 * (h0 + h1));

    for i in 1..(n - 1) {
        let hl = arc_lengths[i] - arc_lengths[i - 1];
        let hr = arc_lengths[i + 1] - arc_lengths[i];
        slopes[i][i - 1] = -hr / (hl * (hl + hr));
        slopes[i][i] = (hr - hl) / (hl * hr);
        slopes[i][i + 1] = hl / (hr * (hl + hr));
    }

    let hr = arc_lengths[n - 1] - arc_lengths[n - 2];
    let hl = arc_lengths[n - 2] - arc_lengths[n - 3];
    slopes[n - 1][n - 3] = hr / (hl * (hl + hr));
    slopes[n - 1][n - 2] = -(hl + hr) / (hl * hr);
    slopes[n - 1][n - 1] = (2.0 * hr + hl) / (hr * (hl + hr));
    slopes
}

fn eval_hermite_1d(arc_lengths: &[f64], values: &[f64], slopes: &[f64], s: f64) -> f64 {
    let n = arc_lengths.len();
    let mut seg = n - 2;
    for i in 0..(n - 1) {
        if s <= arc_lengths[i + 1] {
            seg = i;
            break;
        }
    }
    let h = arc_lengths[seg + 1] - arc_lengths[seg];
    let t = ((s - arc_lengths[seg]) / h).clamp(0.0, 1.0);
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    h00 * values[seg] + h10 * h * slopes[seg] + h01 * values[seg + 1] + h11 * h * slopes[seg + 1]
}

/// Fit the mean constraint force and free energy with a Chebyshev / cosine expansion
/// on `arc_lengths`, compute a heuristic resolution indicator for `F*`, and flag intervals
/// whose local error exceeds the uniform share when `tail_indicator > target_se`.
pub fn estimate_spectral_tail(
    arc_lengths: &[f64],
    gradients: &[f64],
    target_se: f64,
) -> Result<SpectralTailReport, FluxError> {
    let zero_errors = vec![0.0_f64; arc_lengths.len()];
    estimate_spectral_tail_with_errors(arc_lengths, gradients, &zero_errors, target_se)
}

/// Fit the mean constraint force and free energy with a Chebyshev / cosine expansion
/// on `arc_lengths` while attenuating finite-difference slope noise by the Wiener
/// signal-to-noise factor of `gradient_errors`.
///
/// When `gradient_errors` are all zero, this is identical to [`estimate_spectral_tail`].
pub fn estimate_spectral_tail_with_errors(
    arc_lengths: &[f64],
    gradients: &[f64],
    gradient_errors: &[f64],
    target_se: f64,
) -> Result<SpectralTailReport, FluxError> {
    let n = arc_lengths.len();
    if n < 2 {
        return Err(FluxError::InvalidConfig(
            "estimate_spectral_tail requires at least 2 planes",
        ));
    }
    if gradients.len() != n || gradient_errors.len() != n {
        return Err(FluxError::Shape {
            expected: n,
            got: gradients.len().max(gradient_errors.len()),
            context: "PAFI spectral-tail inputs",
        });
    }
    if !target_se.is_finite() || target_se < 0.0 {
        return Err(FluxError::InvalidConfig(
            "estimate_spectral_tail target_se must be finite and non-negative",
        ));
    }
    for i in 0..n {
        if !arc_lengths[i].is_finite() {
            return Err(FluxError::NonFinite("estimate_spectral_tail arc_lengths"));
        }
        if i > 0 && arc_lengths[i] <= arc_lengths[i - 1] {
            return Err(FluxError::InvalidConfig(
                "estimate_spectral_tail arc_lengths must strictly increase",
            ));
        }
        if !gradients[i].is_finite() {
            return Err(FluxError::NonFinite("estimate_spectral_tail gradients"));
        }
        if !gradient_errors[i].is_finite() || gradient_errors[i] < 0.0 {
            return Err(FluxError::InvalidConfig(
                "PAFI gradient standard errors must be finite and non-negative",
            ));
        }
    }

    let s0 = arc_lengths[0];
    let s_end = arc_lengths[n - 1];
    let total_length = s_end - s0;
    if !total_length.is_finite() || total_length <= f64::MIN_POSITIVE {
        return Err(FluxError::InvalidConfig(
            "estimate_spectral_tail total arc length must be positive",
        ));
    }

    let d_basis = hermite_slope_basis(arc_lengths);
    let raw_slopes: Vec<f64> = (0..n)
        .map(|i| (0..n).map(|j| d_basis[i][j] * gradients[j]).sum())
        .collect();
    let p = n - 1;
    let mut alphas = vec![1.0_f64; p];
    for k in 0..p {
        let diff_k = raw_slopes[k] - raw_slopes[k + 1];
        let mut var_diff = 0.0_f64;
        for (j, &gradient_error) in gradient_errors.iter().enumerate() {
            let db = d_basis[k][j] - d_basis[k + 1][j];
            var_diff += db * db * gradient_error * gradient_error;
        }
        if var_diff > 0.0 {
            let sig2 = diff_k * diff_k;
            alphas[k] = if sig2 + var_diff > 0.0 {
                sig2 / (sig2 + var_diff)
            } else {
                0.0
            };
        }
    }
    let slopes: Vec<f64> = (0..n)
        .map(|i| {
            let w_i = if i == 0 {
                alphas[0]
            } else if i == n - 1 {
                alphas[p - 1]
            } else {
                0.5 * (alphas[i - 1] + alphas[i])
            };
            w_i * raw_slopes[i]
        })
        .collect();
    let p_f64 = p as f64;
    let mut lobatto_forces = vec![0.0_f64; p + 1];
    for (j, slot) in lobatto_forces.iter_mut().enumerate() {
        let theta_j = ((p - j) as f64) * PI / p_f64;
        let x_j = theta_j.cos();
        let s_j = (s0 + 0.5 * total_length * (1.0 + x_j)).clamp(s0, s_end);
        *slot = eval_hermite_1d(arc_lengths, gradients, &slopes, s_j);
    }

    let mut force_coefficients = vec![0.0_f64; p + 1];
    for (m, coeff) in force_coefficients.iter_mut().enumerate() {
        let gamma_m = if m == 0 || m == p { 2.0 } else { 1.0 };
        let mut sum = 0.0_f64;
        for (j, &f_val) in lobatto_forces.iter().enumerate() {
            let gamma_j = if j == 0 || j == p { 2.0 } else { 1.0 };
            let theta_j = ((p - j) as f64) * PI / p_f64;
            sum += (f_val / gamma_j) * ((m as f64) * theta_j).cos();
        }
        *coeff = (2.0 / (p_f64 * gamma_m)) * sum;
    }

    let mut profile_coefficients = vec![0.0_f64; p + 2];
    for m in 1..=(p + 1) {
        let gamma_prev = if m == 1 { 2.0 } else { 1.0 };
        let a_prev = force_coefficients[m - 1];
        let a_next = if m < p {
            force_coefficients[m + 1]
        } else {
            0.0
        };
        profile_coefficients[m] =
            (total_length / (4.0 * (m as f64))) * (gamma_prev * a_prev - a_next);
    }
    let mut c0 = 0.0_f64;
    for (m, &cm) in profile_coefficients.iter().enumerate().skip(1) {
        let sign = if m % 2 == 0 { 1.0 } else { -1.0 };
        c0 -= sign * cm;
    }
    profile_coefficients[0] = c0;

    let to_x = |s: f64| -> f64 { 2.0 * (s - s0) / total_length - 1.0 };

    let mean_alpha = alphas.iter().copied().sum::<f64>() / p_f64;
    let tail_start = 3.max(p);
    let mut spectral_tail_sum = 0.0_f64;
    for &cm in &profile_coefficients[tail_start..=(p + 1)] {
        spectral_tail_sum += mean_alpha * cm.abs();
    }

    let trap_barrier =
        trapezoid_cumulative(arc_lengths, gradients, "PAFI spectral-tail trapezoid")?
            .into_iter()
            .fold(0.0_f64, f64::max);
    let mut spec_barrier = 0.0_f64;
    for &s in arc_lengths {
        spec_barrier = spec_barrier.max(chebyshev_eval(&profile_coefficients, to_x(s)));
    }

    let mut interval_errors = vec![0.0_f64; p];
    let mut max_peak_gap = 0.0_f64;
    let mut total_quad_diff = 0.0_f64;
    for k in 0..p {
        let sa = arc_lengths[k];
        let sb = arc_lengths[k + 1];
        let ds = sb - sa;
        let hermite_quad_diff = (ds * ds / 12.0) * (raw_slopes[k] - raw_slopes[k + 1]).abs();
        let trap_inc = 0.5 * (gradients[k] + gradients[k + 1]) * ds;
        let fa = chebyshev_eval(&profile_coefficients, to_x(sa));
        let fb = chebyshev_eval(&profile_coefficients, to_x(sb));
        let spec_inc = fb - fa;
        let quad_diff = alphas[k] * hermite_quad_diff.max((spec_inc - trap_inc).abs());
        total_quad_diff += quad_diff;

        let endpoint_max = fa.max(fb);
        let mut peak_gap = 0.0_f64;
        for sub in 1..4 {
            let s_sub = sa + 0.25 * (sub as f64) * ds;
            let x_sub = to_x(s_sub);
            let f_sub = chebyshev_eval(&profile_coefficients, x_sub);
            spec_barrier = spec_barrier.max(f_sub);
            if f_sub > endpoint_max {
                peak_gap = peak_gap.max(f_sub - endpoint_max);
            }
        }
        let damped_peak_gap = alphas[k] * peak_gap;
        // Weight the global Chebyshev tail across intervals by local force variation |dg| * ds.
        let local_variation = (gradients[k + 1] - gradients[k]).abs() * ds / total_length;
        let local_tail_var = spectral_tail_sum * local_variation;
        max_peak_gap = max_peak_gap.max(damped_peak_gap);
        interval_errors[k] = quad_diff + damped_peak_gap + local_tail_var;
    }

    let raw_tail_indicator = (mean_alpha * (spec_barrier - trap_barrier).abs())
        .max(total_quad_diff)
        + spectral_tail_sum
        + max_peak_gap;
    let scale = 1.0 + trap_barrier.abs().max(spec_barrier.abs());
    let rounding_floor = 64.0 * f64::EPSILON * scale;
    let tail_indicator = if raw_tail_indicator <= rounding_floor {
        0.0
    } else {
        raw_tail_indicator
    };

    let mut flagged_intervals = Vec::new();
    if tail_indicator > target_se.max(rounding_floor) {
        let mean_err = interval_errors.iter().copied().sum::<f64>() / (p as f64);
        let max_err = interval_errors.iter().copied().fold(0.0_f64, f64::max);
        let per_interval_target = target_se / (p as f64);
        let threshold = if per_interval_target > rounding_floor {
            per_interval_target.min(mean_err)
        } else {
            mean_err
        };
        let sym_tol = 64.0 * f64::EPSILON * (1.0 + max_err);
        for (k, &err) in interval_errors.iter().enumerate() {
            if err + sym_tol >= threshold && err > rounding_floor {
                flagged_intervals.push(k);
            }
        }
        if flagged_intervals.is_empty() && max_err > rounding_floor {
            for (k, &err) in interval_errors.iter().enumerate() {
                if err + sym_tol >= mean_err {
                    flagged_intervals.push(k);
                }
            }
        }
    }

    Ok(SpectralTailReport {
        force_coefficients,
        profile_coefficients,
        tail_indicator,
        interval_errors,
        flagged_intervals,
    })
}

/// Size per-plane `(burn_in_steps, sample_steps)` from a measured integrated
/// autocorrelation time `tau_int` (in steps) and [`StatsConfig`].
pub fn size_window_from_tau_int(
    tau_int: f64,
    min_burn_in_steps: usize,
    min_sample_steps: usize,
    stats: &StatsConfig,
) -> Result<(usize, usize), FluxError> {
    if !tau_int.is_finite() || tau_int <= 0.0 {
        return Err(FluxError::InvalidConfig(
            "tau_int must be finite and positive",
        ));
    }
    if stats.min_batches < 2 {
        return Err(FluxError::InvalidConfig(
            "StatsConfig::min_batches must be at least 2",
        ));
    }
    let corr_block = (2.0 * tau_int).ceil().max(1.0) as usize;
    let burn_blocks = (stats.min_batches / 2).max(2);
    let burn_in = min_burn_in_steps.max(burn_blocks.saturating_mul(corr_block));
    let sample_steps = min_sample_steps.max(stats.min_batches.saturating_mul(corr_block));
    Ok((burn_in, sample_steps))
}

/// Configuration for adaptive Projected Average Force Integration with
/// monitor equidistribution and spectral-tail refinement.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct AdaptivePafiConfig {
    /// Base PAFI configuration (`base.planes` sets the initial plane count `>= 3`).
    ///
    /// Default: [`PafiConfig::default()`] with `planes = 5`.
    pub base: PafiConfig,
    /// Maximum total hyperplanes placed across all refinement iterations (`>= base.planes`).
    ///
    /// Default: `33`.
    pub max_planes: usize,
    /// Maximum spectral-tail refinement iterations (`>= 1`).
    ///
    /// Default: `6`.
    pub max_refinements: usize,
    /// Monitor function used by [`EquidistributionMap`] to place and bisect planes.
    ///
    /// Default: [`MonitorKind::Combined`].
    pub monitor: MonitorKind,
}

impl Default for AdaptivePafiConfig {
    fn default() -> Self {
        Self {
            base: PafiConfig::default().with_planes(5),
            max_planes: 33,
            max_refinements: 6,
            monitor: MonitorKind::Combined,
        }
    }
}

with_setters!(AdaptivePafiConfig {
    with_base => base: val PafiConfig,
    with_max_planes => max_planes: val usize,
    with_max_refinements => max_refinements: val usize,
    with_monitor => monitor: val MonitorKind,
});

/// Result of [`pafi_profile_adaptive`], pairing the converged [`FreeEnergyProfile`]
/// with its [`SpectralTailReport`], [`EquidistributionMap`], refinement count, and force calls.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct AdaptiveFreeEnergyProfile {
    /// Integrated free-energy profile on the final adaptively placed planes.
    pub profile: FreeEnergyProfile,
    /// Final Chebyshev / cosine spectral-tail report on the placed planes.
    pub tail: SpectralTailReport,
    /// Final monitor equidistribution map `s -> sigma(s)`.
    pub equidistribution: EquidistributionMap,
    /// Final geometric path-curvature channel `||dn/ds||_2` at each plane,
    /// before monitor selection or normalization.
    pub final_path_curvatures: Vec<f64>,
    /// Final autocorrelation-weighted force-variance channel at each plane,
    /// before monitor selection or normalization.
    pub final_tau_weighted_variances: Vec<f64>,
    /// Number of plane-insertion refinement rounds executed after the initial placement.
    pub refinement_rounds: usize,
    /// Total force calls performed across zero-temperature checks and constrained MD steps.
    pub force_calls: usize,
}

#[derive(Clone, Debug)]
struct SampledPlaneRecord {
    geometry: PlaneGeometry,
    samples: Vec<f64>,
    uid: usize,
    rounds_completed: usize,
}

fn adaptive_refinement_should_stop(
    tail: &SpectralTailReport,
    stop_threshold: f64,
    refinement_rounds: usize,
    max_refinements: usize,
    plane_count: usize,
    max_planes: usize,
) -> bool {
    tail.tail_indicator <= stop_threshold
        || refinement_rounds >= max_refinements
        || plane_count >= max_planes
        || tail.flagged_intervals.is_empty()
}

/// Integrate the PAFI free-energy profile with monitor equidistribution and
/// iterative spectral-tail plane refinement.
///
/// The validation, the canonical-sampling requirement, the reactant checks, and
/// the rate normalization are those of [`pafi_profile`]. A coarse grid whose
/// integrated profile never rises above the first plane has no reactant
/// segment to normalize over. Such a grid is refined while planes remain, and
/// the error of [`pafi_profile`] is returned if it is still unresolved when
/// refinement stops.
///
/// # Errors
///
/// Returns every error of [`pafi_profile`], and [`FluxError::InvalidConfig`]
/// when `base.planes < 3`, `max_planes < base.planes`, or
/// `max_refinements == 0`.
pub fn pafi_profile_adaptive<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    path_images: ArrayView2<'_, f64>,
    config: &AdaptivePafiConfig,
) -> Result<AdaptiveFreeEnergyProfile, FluxError> {
    if !config.base.time_step.is_finite() || config.base.time_step <= 0.0 {
        return Err(FluxError::InvalidConfig(
            "PafiConfig::time_step must be finite and positive",
        ));
    }
    if config.base.burn_in_steps == 0 {
        return Err(FluxError::InvalidConfig(
            "PafiConfig::burn_in_steps must be at least 1",
        ));
    }
    if config.base.sample_steps < 2 {
        return Err(FluxError::InvalidConfig(
            "PafiConfig::sample_steps must be at least 2",
        ));
    }
    if config.base.max_rounds == 0 {
        return Err(FluxError::InvalidConfig(
            "PafiConfig::max_rounds must be at least 1",
        ));
    }
    if config.base.planes < 3 {
        return Err(FluxError::InvalidConfig(
            "AdaptivePafiConfig requires base.planes >= 3",
        ));
    }
    if config.max_planes < config.base.planes {
        return Err(FluxError::InvalidConfig(
            "AdaptivePafiConfig::max_planes must be at least base.planes",
        ));
    }
    if config.max_refinements == 0 {
        return Err(FluxError::InvalidConfig(
            "AdaptivePafiConfig::max_refinements must be at least 1",
        ));
    }
    if let Some(tol) = config.base.zero_temperature_tol
        && (!tol.is_finite() || tol < 0.0)
    {
        return Err(FluxError::InvalidConfig(
            "PafiConfig::zero_temperature_tol must be finite and non-negative",
        ));
    }
    if let Some(target) = config.base.target_error
        && (!target.is_finite() || target < 0.0)
    {
        return Err(FluxError::InvalidConfig(
            "PafiConfig::target_error must be finite and non-negative",
        ));
    }

    let snapshot = EngineSnapshot::capture(engine);
    let result = (|| {
        let kbt = engine.thermostat().require_canonical_sampling()?;
        if kbt <= 0.0 {
            return Err(FluxError::InvalidConfig(
                "PAFI requires positive thermal energy",
            ));
        }
        let masses = engine.masses().to_owned();
        let spline =
            validate_path_and_build_spline(path_images, masses.view(), config.base.planes)?;
        let total_arc = spline.end();

        // Evaluate zero-temperature path force on config.base.planes uniform anchors
        // for the zero-temperature check and initial monitor construction.
        let ref_count = config.base.planes;
        let ref_arcs: Vec<f64> = (0..ref_count)
            .map(|k| total_arc * (k as f64) / ((ref_count - 1) as f64))
            .collect();
        let ref_geometries = build_plane_geometries_at(&spline, masses.view(), &ref_arcs)?;

        engine.set_constraint(None)?;
        let mut ref_gradients = Vec::with_capacity(ref_geometries.len());
        let mut ref_energies = Vec::with_capacity(ref_geometries.len());
        let mut ref_curvatures = Vec::with_capacity(ref_geometries.len());
        let mut force_calls = 0usize;
        for geometry in &ref_geometries {
            engine.set_positions(geometry.anchor.view())?;
            let (energy, forces) = engine.energy_and_forces()?;
            force_calls += 1;
            ref_energies.push(energy);
            ref_gradients.push(-dot(geometry.tangent.view(), forces.view()));
            ref_curvatures.push(nrm2(geometry.dn_ds.view()));
        }
        let relative_error = path_force_relative_error(&ref_arcs, &ref_energies, &ref_gradients);
        if relative_error > effective_zero_temperature_tol(&config.base) {
            return Err(FluxError::InvalidConfig(
                "PAFI zero-temperature path-force quadrature exceeds tolerance",
            ));
        }
        let reactant_hessian_certified =
            ensure_reactant_minimum(engine, ref_geometries[0].anchor.view())?;
        let root_mass = engine.masses().mapv(f64::sqrt);
        let origin = ref_geometries[0].anchor.clone();
        let translation_modes = classified_translation_modes(
            engine,
            &origin,
            &root_mass,
            config.base.translation_invariance,
        )?;
        let reactant_curvature =
            curvature_with_modes(engine, &ref_geometries[0], &translation_modes)?;

        let initial_map = match config.monitor {
            MonitorKind::ForceGradient => EquidistributionMap::from_inputs(
                &MonitorInputs::new(&ref_arcs).with_force_gradients(&ref_gradients),
                MonitorKind::ForceGradient,
            )?,
            MonitorKind::PathCurvature => EquidistributionMap::from_inputs(
                &MonitorInputs::new(&ref_arcs).with_path_curvatures(&ref_curvatures),
                MonitorKind::PathCurvature,
            )?,
            MonitorKind::AutocorrelatedVariance => {
                EquidistributionMap::new(&ref_arcs, &vec![0.0; ref_count])?
            }
            MonitorKind::Combined => EquidistributionMap::from_inputs(
                &MonitorInputs::new(&ref_arcs)
                    .with_force_gradients(&ref_gradients)
                    .with_path_curvatures(&ref_curvatures),
                MonitorKind::Combined,
            )?,
        };

        let path_symmetric = initial_map.is_reflection_symmetric();
        let initial_arcs = initial_map.place_planes(config.base.planes)?;
        let initial_geoms = build_plane_geometries_at(&spline, masses.view(), &initial_arcs)?;
        let mut next_uid = 0usize;
        let mut records: Vec<SampledPlaneRecord> = initial_geoms
            .into_iter()
            .map(|geometry| {
                let uid = next_uid;
                next_uid += 1;
                SampledPlaneRecord {
                    geometry,
                    samples: Vec::new(),
                    uid,
                    rounds_completed: 0,
                }
            })
            .collect();

        let mut refinement_rounds = 0usize;
        loop {
            // Sample any plane that has not yet completed at least one round.
            for record in &mut records {
                while record.rounds_completed == 0 {
                    let samples = sample_plane(
                        engine,
                        &record.geometry,
                        &config.base,
                        record.uid,
                        record.rounds_completed,
                        &translation_modes,
                    )?;
                    force_calls += 1 + config.base.burn_in_steps + config.base.sample_steps;
                    record.samples.extend(samples);
                    record.rounds_completed += 1;
                }
            }

            // If target_error is set on a stochastic system, add rounds up to max_rounds
            // when statistical SE exceeds target_error.
            loop {
                let mut gradients = Vec::with_capacity(records.len());
                let mut errors = Vec::with_capacity(records.len());
                let mut variances = Vec::with_capacity(records.len());
                let mut curvatures = Vec::with_capacity(records.len());
                let mut arcs = Vec::with_capacity(records.len());
                for record in &records {
                    let est = batch_means(&record.samples, &config.base.stats)?;
                    gradients.push(est.mean);
                    errors.push(est.standard_error);
                    variances.push(est.variance);
                    curvatures.push(nrm2(record.geometry.dn_ds.view()));
                    arcs.push(record.geometry.arc);
                }
                let quadrature = Quadrature::new(&arcs)?;
                let free_energy = quadrature.integrate(&gradients);
                // A coarse grid can integrate to a profile that never rises
                // above the first plane. That has no reactant segment, so it is
                // refined rather than reported while planes remain.
                let unresolved = free_energy.iter().all(|value| value.is_finite())
                    && quadrature
                        .dividing_plane(&gradients, &free_energy)?
                        .partition_end
                        == 0;
                let profile = match integrate_profile(
                    &quadrature,
                    &gradients,
                    &errors,
                    kbt,
                    reactant_curvature,
                    reactant_hessian_certified,
                ) {
                    Ok(profile) => Ok(profile),
                    Err(error) if unresolved => Err(error),
                    Err(error) => return Err(error),
                };
                let need_more_stat_rounds = match (&profile, config.base.target_error) {
                    (Ok(profile), Some(target)) => {
                        profile.barrier_error > target
                            && records
                                .iter()
                                .any(|r| r.rounds_completed < config.base.max_rounds)
                    }
                    _ => false,
                };
                if need_more_stat_rounds {
                    for record in &mut records {
                        if record.rounds_completed < config.base.max_rounds {
                            let samples = sample_plane(
                                engine,
                                &record.geometry,
                                &config.base,
                                record.uid,
                                record.rounds_completed,
                                &translation_modes,
                            )?;
                            force_calls += 1 + config.base.burn_in_steps + config.base.sample_steps;
                            record.samples.extend(samples);
                            record.rounds_completed += 1;
                        }
                    }
                    continue;
                }

                let (barrier, barrier_error) = match &profile {
                    Ok(profile) => (profile.barrier, profile.barrier_error),
                    Err(_) => (free_energy.iter().fold(0.0_f64, |a, v| a.max(v.abs())), 0.0),
                };
                let stop_threshold = if barrier_error > 0.0 {
                    match config.base.target_error {
                        Some(target) => barrier_error.min(target),
                        None => barrier_error,
                    }
                } else {
                    config
                        .base
                        .target_error
                        .unwrap_or(64.0 * f64::EPSILON * (1.0 + barrier.abs()))
                };

                let tail =
                    estimate_spectral_tail_with_errors(&arcs, &gradients, &errors, stop_threshold)?;
                let monitor_inputs = MonitorInputs::new(&arcs)
                    .with_force_gradients(&gradients)
                    .with_path_curvatures(&curvatures)
                    .with_tau_weighted_variances(&variances);
                let mut raw_mon = monitor_inputs.evaluate(config.monitor)?;
                if path_symmetric {
                    let m_len = raw_mon.len();
                    for i in 0..(m_len / 2) {
                        let avg = 0.5 * (raw_mon[i] + raw_mon[m_len - 1 - i]);
                        raw_mon[i] = avg;
                        raw_mon[m_len - 1 - i] = avg;
                    }
                }
                let current_map = EquidistributionMap::new(&arcs, &raw_mon)?;

                if adaptive_refinement_should_stop(
                    &tail,
                    stop_threshold,
                    refinement_rounds,
                    config.max_refinements,
                    records.len(),
                    config.max_planes,
                ) {
                    return Ok(AdaptiveFreeEnergyProfile {
                        profile: profile?,
                        tail,
                        equidistribution: current_map,
                        final_path_curvatures: curvatures,
                        final_tau_weighted_variances: variances,
                        refinement_rounds,
                        force_calls,
                    });
                }

                let remaining_budget = config.max_planes - records.len();
                let n_intervals = arcs.len() - 1;
                let symmetric = path_symmetric && current_map.is_reflection_symmetric();
                let d_basis = hermite_slope_basis(&arcs);
                let slopes: Vec<f64> = (0..arcs.len())
                    .map(|i| (0..arcs.len()).map(|j| d_basis[i][j] * gradients[j]).sum())
                    .collect();

                let min_gap = current_map
                    .min_spacing_bound(config.max_planes)?
                    .max(64.0 * f64::EPSILON * (1.0 + total_arc));

                // Rank flagged intervals by error while skipping intervals already below
                // the certified spacing floor and keeping mirror pairs together when symmetric.
                let mut ordered_flagged: Vec<usize> = tail
                    .flagged_intervals
                    .iter()
                    .copied()
                    .filter(|&k| arcs[k + 1] - arcs[k] > min_gap)
                    .collect();
                ordered_flagged.sort_by(|&a, &b| {
                    let ea = if symmetric {
                        tail.interval_errors[a].max(tail.interval_errors[n_intervals - 1 - a])
                    } else {
                        tail.interval_errors[a]
                    };
                    let eb = if symmetric {
                        tail.interval_errors[b].max(tail.interval_errors[n_intervals - 1 - b])
                    } else {
                        tail.interval_errors[b]
                    };
                    eb.total_cmp(&ea).then_with(|| a.cmp(&b))
                });

                let mut chosen_intervals = Vec::new();
                let mut used = vec![false; n_intervals];
                // Refine the highest-error interval (or symmetric mirror pair) per iteration
                // so the spectral fit updates before the next plane placement.
                let step_budget = remaining_budget.min(2);
                for k in ordered_flagged {
                    if used[k] {
                        continue;
                    }
                    if symmetric {
                        let mirror = n_intervals - 1 - k;
                        if k == mirror {
                            if chosen_intervals.len() < step_budget {
                                used[k] = true;
                                chosen_intervals.push(k);
                            }
                        } else if chosen_intervals.len() + 2 <= step_budget {
                            used[k] = true;
                            used[mirror] = true;
                            chosen_intervals.push(k.min(mirror));
                            chosen_intervals.push(k.max(mirror));
                        }
                        if chosen_intervals.len() >= step_budget {
                            break;
                        }
                    } else if chosen_intervals.len() < step_budget {
                        used[k] = true;
                        chosen_intervals.push(k);
                        if chosen_intervals.len() >= step_budget {
                            break;
                        }
                    }
                }

                if chosen_intervals.is_empty() {
                    return Ok(AdaptiveFreeEnergyProfile {
                        profile: profile?,
                        tail,
                        equidistribution: current_map,
                        final_path_curvatures: curvatures,
                        final_tau_weighted_variances: variances,
                        refinement_rounds,
                        force_calls,
                    });
                }

                let mut new_arcs = Vec::with_capacity(chosen_intervals.len());
                for &k in &chosen_intervals {
                    if symmetric && k > n_intervals - 1 - k && used[n_intervals - 1 - k] {
                        continue;
                    }
                    let sa = arcs[k];
                    let sb = arcs[k + 1];
                    let sig_a = current_map.sigma(sa)?;
                    let sig_b = current_map.sigma(sb)?;
                    let sig_mid = 0.5 * (sig_a + sig_b);
                    let dsig = sig_b - sig_a;

                    // If the interval brackets a downward zero-crossing of the force (a barrier peak),
                    // bisect the spline root and clamp inside the middle half of the sigma interval.
                    let s_candidate = if gradients[k] > 2.0 * errors[k]
                        && gradients[k + 1] < -2.0 * errors[k + 1]
                    {
                        let mut left = sa;
                        let mut right = sb;
                        for _ in 0..40 {
                            let mid = 0.5 * (left + right);
                            if eval_hermite_1d(&arcs, &gradients, &slopes, mid) > 0.0 {
                                left = mid;
                            } else {
                                right = mid;
                            }
                        }
                        let s_root = 0.5 * (left + right);
                        let sig_root = current_map
                            .sigma(s_root)?
                            .clamp(sig_a + 0.25 * dsig, sig_b - 0.25 * dsig);
                        current_map.inverse(sig_root)?
                    } else {
                        current_map.inverse(sig_mid)?
                    };

                    if symmetric {
                        let mirror = n_intervals - 1 - k;
                        if k == mirror {
                            new_arcs.push(0.5 * (arcs[0] + total_arc));
                        } else if used[mirror] {
                            let offset = s_candidate - arcs[0];
                            new_arcs.push(arcs[0] + offset);
                            new_arcs.push(total_arc - offset);
                        } else {
                            new_arcs.push(s_candidate);
                        }
                    } else {
                        new_arcs.push(s_candidate);
                    }
                }

                let min_gap = current_map
                    .min_spacing_bound(config.max_planes)?
                    .max(64.0 * f64::EPSILON * (1.0 + total_arc));
                let mut added_any = false;
                for s_new in new_arcs {
                    if records
                        .iter()
                        .all(|r| (r.geometry.arc - s_new).abs() > 0.25 * min_gap)
                    {
                        let geometry = evaluate_plane_geometry_at(&spline, masses.view(), s_new)?;
                        let uid = next_uid;
                        next_uid += 1;
                        records.push(SampledPlaneRecord {
                            geometry,
                            samples: Vec::new(),
                            uid,
                            rounds_completed: 0,
                        });
                        added_any = true;
                    }
                }
                if !added_any {
                    return Ok(AdaptiveFreeEnergyProfile {
                        profile: profile?,
                        tail,
                        equidistribution: current_map,
                        final_path_curvatures: curvatures,
                        final_tau_weighted_variances: variances,
                        refinement_rounds,
                        force_calls,
                    });
                }
                records.sort_by(|a, b| a.geometry.arc.total_cmp(&b.geometry.arc));
                refinement_rounds += 1;
                break;
            }
        }
    })();
    snapshot.restore_result(engine, result)
}

/// Trait-object entry point for [`pafi_profile_adaptive`] over [`DynamicsEngine`].
pub fn pafi_profile_adaptive_dyn(
    engine: &mut dyn DynamicsEngine,
    path_images: ArrayView2<'_, f64>,
    config: &AdaptivePafiConfig,
) -> Result<AdaptiveFreeEnergyProfile, FluxError> {
    pafi_profile_adaptive(engine, path_images, config)
}

/// Pool `replicates` (`>= 2`) [`FreeEnergyProfile`] estimates at thermal energy
/// `kbt` by averaging per-plane mean constraint forces across replicates before
/// integrating `F(s)` and evaluating `exp(-F(s*) / k_B T) / Z_R`.
///
/// Averaging per-replicate rates `k^(r) \propto exp(-F*^(r) / k_B T)` introduces
/// positive Jensen bias `E[exp(-F* / k_B T)] > exp(-E[F*] / k_B T)` when
/// per-replicate barrier noise `sd(F*)` is not negligible compared to `k_B T`.
/// Pooling the per-plane mean forces `g_k^(r)` across replicates first produces
/// a consistent estimator of `F(s)` and `k_TST`, with between-replicate standard
/// errors `SE(\bar{g}_k) = sqrt(\sum_r (g_k^(r) - \bar{g}_k)^2 / (n (n - 1)))`
/// propagated to `barrier_error` and `normalized_rate_error` by the delta method.
///
/// When all replicates share the same `arc_lengths` knots (within floating-point
/// tolerance), per-plane gradients are averaged directly at each knot. When
/// adaptive replicates have distinct knot vectors, each replicate's gradient is
/// evaluated on the modal reference knot grid via piecewise cubic Hermite
/// interpolation before pooling. The pooled result uses the same Hermite
/// integration, interior maximum search, doubled-trapezoid reactant
/// normalization, and log-space rate calculation as [`pafi_profile`]. The
/// reactant frequency does not enter the rate, so its between-replicate
/// spread is reported in [`FreeEnergyProfile::reactant_frequency_error`] and
/// not folded into [`FreeEnergyProfile::normalized_rate_error`]. The pooled
/// profile is certified only when every replicate was.
///
/// # Errors
///
/// Returns [`FluxError::InvalidConfig`] when `replicates.len() < 2`, `kbt <= 0`
/// or non-finite, any replicate has fewer than 2 knots, non-increasing knots,
/// or mismatched path endpoints. Returns [`FluxError::Shape`] when a replicate's
/// `gradients` length does not match its `arc_lengths` length, and
/// [`FluxError::NonFinite`] when any knot or gradient is non-finite.
pub fn pool_pafi_replicates(
    replicates: &[FreeEnergyProfile],
    kbt: f64,
) -> Result<FreeEnergyProfile, FluxError> {
    let n_rep = replicates.len();
    if n_rep < 2 {
        return Err(FluxError::InvalidConfig(
            "pool_pafi_replicates requires at least 2 replicates",
        ));
    }
    if !kbt.is_finite() || kbt <= 0.0 {
        return Err(FluxError::InvalidConfig(
            "PAFI requires positive thermal energy",
        ));
    }

    let first_arcs = &replicates[0].arc_lengths;
    if first_arcs.len() < 2 {
        return Err(FluxError::InvalidConfig(
            "pool_pafi_replicates requires at least 2 planes per replicate",
        ));
    }
    let s0_ref = first_arcs[0];
    let s_end_ref = first_arcs[first_arcs.len() - 1];
    let span_ref = s_end_ref - s0_ref;
    if !span_ref.is_finite() || span_ref <= 0.0 {
        return Err(FluxError::InvalidConfig(
            "pool_pafi_replicates requires positive total arc length",
        ));
    }
    let knot_tol = 64.0 * f64::EPSILON * (1.0 + s0_ref.abs() + s_end_ref.abs());

    for rep in replicates {
        let m = rep.arc_lengths.len();
        if m < 2 {
            return Err(FluxError::InvalidConfig(
                "pool_pafi_replicates requires at least 2 planes per replicate",
            ));
        }
        if rep.gradients.len() != m {
            return Err(FluxError::Shape {
                expected: m,
                got: rep.gradients.len(),
                context: "pool_pafi_replicates replicate gradients versus arc_lengths",
            });
        }
        if !rep.reactant_frequency.is_finite() || rep.reactant_frequency <= 0.0 {
            return Err(FluxError::InvalidConfig(
                "pool_pafi_replicates reactant frequencies must be finite and positive",
            ));
        }
        for i in 0..m {
            if !rep.arc_lengths[i].is_finite() {
                return Err(FluxError::NonFinite("pool_pafi_replicates arc_lengths"));
            }
            if i > 0 && rep.arc_lengths[i] <= rep.arc_lengths[i - 1] {
                return Err(FluxError::InvalidConfig(
                    "pool_pafi_replicates arc_lengths must strictly increase",
                ));
            }
            if !rep.gradients[i].is_finite() {
                return Err(FluxError::NonFinite("pool_pafi_replicates gradients"));
            }
        }
        if (rep.arc_lengths[0] - s0_ref).abs() > knot_tol
            || (rep.arc_lengths[m - 1] - s_end_ref).abs() > knot_tol
        {
            return Err(FluxError::InvalidConfig(
                "pool_pafi_replicates replicates must share path endpoints",
            ));
        }
    }

    let knots_match = |a: &[f64], b: &[f64]| -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b.iter())
                .all(|(&sa, &sb)| (sa - sb).abs() <= knot_tol)
    };

    // Select the modal knot vector across replicates (breaking ties by larger knot count).
    let mut best_idx = 0usize;
    let mut best_count = 0usize;
    let mut best_len = 0usize;
    for (i, cand) in replicates.iter().enumerate() {
        let count = replicates
            .iter()
            .filter(|other| knots_match(&cand.arc_lengths, &other.arc_lengths))
            .count();
        let len = cand.arc_lengths.len();
        if count > best_count || (count == best_count && len > best_len) {
            best_idx = i;
            best_count = count;
            best_len = len;
        }
    }
    let ref_arcs = &replicates[best_idx].arc_lengths;
    let m = ref_arcs.len();

    let mut replicate_grads_on_ref: Vec<Vec<f64>> = Vec::with_capacity(n_rep);
    for rep in replicates {
        if knots_match(&rep.arc_lengths, ref_arcs) {
            replicate_grads_on_ref.push(rep.gradients.clone());
        } else {
            let d_basis = hermite_slope_basis(&rep.arc_lengths);
            let slopes: Vec<f64> = (0..rep.arc_lengths.len())
                .map(|i| {
                    (0..rep.arc_lengths.len())
                        .map(|j| d_basis[i][j] * rep.gradients[j])
                        .sum()
                })
                .collect();
            let interp: Vec<f64> = ref_arcs
                .iter()
                .map(|&s| eval_hermite_1d(&rep.arc_lengths, &rep.gradients, &slopes, s))
                .collect();
            replicate_grads_on_ref.push(interp);
        }
    }

    let n_f64 = n_rep as f64;
    let mut mean_gradients = vec![0.0_f64; m];
    let mut se_gradients = vec![0.0_f64; m];
    for k in 0..m {
        let mean = replicate_grads_on_ref.iter().map(|g| g[k]).sum::<f64>() / n_f64;
        let var_sum: f64 = replicate_grads_on_ref
            .iter()
            .map(|g| {
                let diff = g[k] - mean;
                diff * diff
            })
            .sum();
        let se = (var_sum / (n_f64 * (n_f64 - 1.0))).sqrt();
        mean_gradients[k] = mean;
        se_gradients[k] = se;
    }

    let reactant_frequency = replicates
        .iter()
        .map(|rep| rep.reactant_frequency)
        .sum::<f64>()
        / n_f64;
    let frequency_var_sum = replicates
        .iter()
        .map(|rep| {
            let diff = rep.reactant_frequency - reactant_frequency;
            diff * diff
        })
        .sum::<f64>();
    let reactant_frequency_error = (frequency_var_sum / (n_f64 * (n_f64 - 1.0))).sqrt();
    let quadrature = Quadrature::new(ref_arcs)?;
    let reactant_hessian_certified = replicates.iter().all(|rep| rep.reactant_hessian_certified);
    let mut pooled = integrate_profile(
        &quadrature,
        &mean_gradients,
        &se_gradients,
        kbt,
        reactant_frequency * reactant_frequency,
        reactant_hessian_certified,
    )?;
    pooled.reactant_frequency_error = reactant_frequency_error;
    Ok(pooled)
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, Array2, ArrayView1};

    use super::*;
    use crate::engine::native::{NativeConfig, NativeEngine};
    use crate::engine::{DynamicsEngine, StepReport, SurfaceVelocityPolicy, ThermostatConfig};

    struct Quartic;

    impl rgsaddle::PointSurface for Quartic {
        fn eval(
            &self,
            x: ArrayView1<'_, f64>,
        ) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
            Ok((x[0].powi(4), Array1::from_vec(vec![4.0 * x[0].powi(3)])))
        }
    }

    struct IsotropicPair {
        stiffness: f64,
    }

    impl rgsaddle::PointSurface for IsotropicPair {
        fn eval(
            &self,
            x: ArrayView1<'_, f64>,
        ) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
            let mut gradient = Array1::zeros(6);
            let mut energy = 0.0;
            for axis in 0..3 {
                let extension = x[3 + axis] - x[axis];
                energy += 0.5 * self.stiffness * extension * extension;
                gradient[axis] = -self.stiffness * extension;
                gradient[3 + axis] = self.stiffness * extension;
            }
            Ok((energy, gradient))
        }
    }

    struct TetheredPair {
        first_stiffness: f64,
        second_stiffness: f64,
        coupling: f64,
    }

    impl rgsaddle::PointSurface for TetheredPair {
        fn eval(
            &self,
            x: ArrayView1<'_, f64>,
        ) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
            let mut energy = 0.0;
            let mut gradient = Array1::zeros(6);
            for axis in 0..3 {
                let first = x[axis];
                let second = x[3 + axis];
                let extension = second - first;
                energy += 0.5 * self.first_stiffness * first * first
                    + 0.5 * self.second_stiffness * second * second
                    + 0.5 * self.coupling * extension * extension;
                gradient[axis] = self.first_stiffness * first - self.coupling * extension;
                gradient[3 + axis] = self.second_stiffness * second + self.coupling * extension;
            }
            Ok((energy, gradient))
        }
    }

    struct AnisotropicTetheredPair {
        tether: [f64; 3],
        coupling: [f64; 3],
    }

    impl rgsaddle::PointSurface for AnisotropicTetheredPair {
        fn eval(
            &self,
            x: ArrayView1<'_, f64>,
        ) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
            let mut energy = 0.0;
            let mut gradient = Array1::zeros(6);
            for axis in 0..3 {
                let first = x[axis];
                let second = x[3 + axis];
                let extension = second - first;
                energy += 0.5 * self.tether[axis] * (first * first + second * second)
                    + 0.5 * self.coupling[axis] * extension * extension;
                gradient[axis] = self.tether[axis] * first - self.coupling[axis] * extension;
                gradient[3 + axis] = self.tether[axis] * second + self.coupling[axis] * extension;
            }
            Ok((energy, gradient))
        }
    }

    struct SingleParticleTrap {
        stiffness: [f64; 3],
    }

    impl rgsaddle::PointSurface for SingleParticleTrap {
        fn eval(
            &self,
            x: ArrayView1<'_, f64>,
        ) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
            let gradient = Array1::from_vec(
                self.stiffness
                    .iter()
                    .zip(x)
                    .map(|(&k, &coordinate)| k * coordinate)
                    .collect(),
            );
            Ok((0.5 * dot(x, gradient.view()), gradient))
        }
    }

    #[derive(Clone)]
    struct InconsistentLinearGradient {
        shortfall: f64,
    }

    impl rgsaddle::PointSurface for InconsistentLinearGradient {
        fn eval(
            &self,
            x: ArrayView1<'_, f64>,
        ) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
            // The energy rises by 1 on `[0, 1]`. A constant gradient of
            // `1 - shortfall` leaves a relative discrepancy of `shortfall`.
            // Adding that same shortfall as an antisymmetric slope about 0.5
            // keeps the trapezoid integral at `1 - shortfall` while giving
            // positive reactant curvature once the zero-temperature check passes.
            let gradient = (1.0 - self.shortfall) + self.shortfall * (x[0] - 0.5);
            Ok((x[0], Array1::from_vec(vec![gradient])))
        }
    }

    struct CoordinateLinear {
        index: usize,
    }

    impl rgsaddle::PointSurface for CoordinateLinear {
        fn eval(
            &self,
            x: ArrayView1<'_, f64>,
        ) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
            let mut gradient = Array1::zeros(x.len());
            gradient[self.index] = 1.0;
            Ok((x[self.index], gradient))
        }
    }

    struct AnharmonicOneDim {
        harmonic: f64,
        quartic: f64,
    }

    impl rgsaddle::PointSurface for AnharmonicOneDim {
        fn eval(
            &self,
            x: ArrayView1<'_, f64>,
        ) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
            let energy = 0.5 * self.harmonic * x[0] * x[0] + 0.25 * self.quartic * x[0].powi(4);
            let gradient = self.harmonic * x[0] + self.quartic * x[0].powi(3);
            Ok((energy, Array1::from_vec(vec![gradient])))
        }
    }

    struct AmbiguousTranslationPair {
        coupling: f64,
        background_gradient: f64,
        tether: f64,
        quartic: f64,
    }

    impl rgsaddle::PointSurface for AmbiguousTranslationPair {
        fn eval(
            &self,
            x: ArrayView1<'_, f64>,
        ) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
            let mut energy = 0.0;
            let mut gradient = Array1::zeros(6);
            for axis in 0..3 {
                let extension = x[3 + axis] - x[axis];
                energy += 0.5 * self.coupling * extension * extension;
                gradient[axis] = -self.coupling * extension;
                gradient[3 + axis] = self.coupling * extension;
            }
            let inv_sqrt_two = 0.5_f64.sqrt();
            let center = (x[0] + x[3]) * inv_sqrt_two;
            energy += self.background_gradient * center
                + 0.5 * self.tether * center * center
                + 0.25 * self.quartic * center.powi(4);
            let center_gradient =
                self.background_gradient + self.tether * center + self.quartic * center.powi(3);
            gradient[0] += center_gradient * inv_sqrt_two;
            gradient[3] += center_gradient * inv_sqrt_two;
            Ok((energy, gradient))
        }
    }

    struct AllToAllHarmonic {
        atoms: usize,
        stiffness: f64,
        tether: f64,
    }

    impl rgsaddle::PointSurface for AllToAllHarmonic {
        fn eval(
            &self,
            x: ArrayView1<'_, f64>,
        ) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
            let mut energy = 0.0;
            let mut gradient = Array1::zeros(x.len());
            for i in 0..x.len() {
                energy += 0.5 * self.tether * x[i] * x[i];
                gradient[i] += self.tether * x[i];
            }
            for first in 0..self.atoms {
                for second in (first + 1)..self.atoms {
                    for axis in 0..3 {
                        let i = 3 * first + axis;
                        let j = 3 * second + axis;
                        let extension = x[j] - x[i];
                        energy += 0.5 * self.stiffness * extension * extension;
                        gradient[i] -= self.stiffness * extension;
                        gradient[j] += self.stiffness * extension;
                    }
                }
            }
            Ok((energy, gradient))
        }
    }

    #[test]
    fn mass_metric_sets_normal_and_multiplier_conversion() {
        let path = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).expect("shape");
        let masses = Array1::from_vec(vec![1.0, 4.0]);
        let geometry = build_plane_geometries(path.view(), masses.view(), 3).expect("geometry");
        let middle = &geometry[1];
        assert_abs_diff_eq!(
            middle.normal[1] / middle.normal[0],
            4.0,
            epsilon = f64::EPSILON
        );

        let mut engine = NativeEngine::new(
            Quartic,
            NativeConfig::default()
                .with_masses(masses)
                .with_thermostat(ThermostatConfig::default().with_temperature(1.0)),
        )
        .expect("engine");
        engine
            .set_positions(middle.anchor.view())
            .expect("set anchor");
        let projected = projected_force_sample(&engine, middle, 2.0, &[]).expect("projected force");
        assert_abs_diff_eq!(projected, 2.0 * middle.dc_ds, epsilon = f64::EPSILON);
        assert!((middle.dc_ds - 1.0).abs() > 0.1);
    }

    #[test]
    fn reactant_curvature_projects_all_translation_zero_modes_with_or_without_cell() {
        let stiffness = 3.25;
        let positions = Array1::zeros(6);
        let mut tangent = Array1::from_vec(vec![0.3, 0.2, -0.1, 0.9, 0.2, 0.4]);
        let tangent_norm = nrm2(tangent.view());
        tangent.mapv_inplace(|value| value / tangent_norm);
        let geometry = PlaneGeometry {
            anchor: positions,
            normal: tangent.clone(),
            dn_ds: Array1::zeros(6),
            dc_ds: 1.0,
            tangent: tangent.clone(),
            arc: 0.0,
        };
        for cell in [Some([10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0]), None] {
            let mut config = NativeConfig::default().with_masses(Array1::ones(6));
            if let Some(cell) = cell {
                config = config.with_cell(cell);
            }
            let mut engine =
                NativeEngine::new(IsotropicPair { stiffness }, config).expect("pair engine");
            let curvature =
                reactant_curvature(&mut engine, &geometry, None).expect("projected curvature");
            assert_abs_diff_eq!(curvature, 2.0 * stiffness, epsilon = 1.0e-7);
        }
    }

    #[test]
    fn invariant_pair_is_projected_across_far_from_origin_probe_grid() {
        let stiffness = 3.25;
        let mut tangent = Array1::from_vec(vec![0.3, 0.2, -0.1, 0.9, 0.2, 0.4]);
        let tangent_norm = nrm2(tangent.view());
        tangent.mapv_inplace(|value| value / tangent_norm);
        let mut engine = NativeEngine::new(
            IsotropicPair { stiffness },
            NativeConfig::default().with_masses(Array1::ones(6)),
        )
        .expect("pair engine");
        for anchor_index in 0..60 {
            let anchor = 0.25 * anchor_index as f64;
            for extension in [0.0, 0.25, 0.75, 2.0] {
                let positions = Array1::from_vec(vec![
                    anchor,
                    anchor,
                    anchor,
                    anchor + extension,
                    anchor + extension,
                    anchor + extension,
                ]);
                let geometry = PlaneGeometry {
                    anchor: positions,
                    normal: tangent.clone(),
                    dn_ds: Array1::zeros(6),
                    dc_ds: 1.0,
                    tangent: tangent.clone(),
                    arc: 0.0,
                };
                let curvature = reactant_curvature(&mut engine, &geometry, None)
                    .unwrap_or_else(|error| {
                        panic!(
                            "invariant probe failed at anchor {anchor} and extension {extension}: {error}"
                        )
                    });
                assert_abs_diff_eq!(curvature, 2.0 * stiffness, epsilon = 1.0e-7);
            }
        }
    }

    #[test]
    fn dense_invariant_hessian_requires_full_frobenius_bound() {
        let atoms = 64;
        let stiffness = 0.25;
        let tether = 1.3e-8;
        let dof = 3 * atoms;
        let mut positions = Array1::zeros(dof);
        let mut tangent = Array1::zeros(dof);
        for atom in 0..atoms {
            for axis in 0..3 {
                let i = 3 * atom + axis;
                positions[i] = 14.75 + (atom as f64) / 3.0 + 0.1 * axis as f64;
                tangent[i] = ((atom + 2 * axis) % 5) as f64 - 2.0;
            }
        }
        let tangent_norm = nrm2(tangent.view());
        tangent.mapv_inplace(|value| value / tangent_norm);
        let geometry = PlaneGeometry {
            anchor: positions,
            normal: tangent.clone(),
            dn_ds: Array1::zeros(dof),
            dc_ds: 1.0,
            tangent,
            arc: 0.0,
        };
        let mut engine = NativeEngine::new(
            AllToAllHarmonic {
                atoms,
                stiffness,
                tether,
            },
            NativeConfig::default().with_masses(Array1::ones(dof)),
        )
        .expect("dense invariant engine");
        let curvature =
            reactant_curvature(&mut engine, &geometry, None).expect("projected curvature");
        assert_abs_diff_eq!(
            curvature,
            atoms as f64 * stiffness + tether,
            epsilon = 1.0e-7
        );
    }

    #[test]
    fn reactant_curvature_keeps_nonzero_translation_modes_in_a_periodic_trap() {
        let mut tangent = Array1::from_vec(vec![0.3, 0.2, -0.1, 0.9, 0.2, 0.4]);
        let tangent_norm = nrm2(tangent.view());
        tangent.mapv_inplace(|value| value / tangent_norm);
        let geometry = PlaneGeometry {
            anchor: Array1::zeros(6),
            normal: tangent.clone(),
            dn_ds: Array1::zeros(6),
            dc_ds: 1.0,
            tangent: tangent.clone(),
            arc: 0.0,
        };
        let mut engine = NativeEngine::new(
            TetheredPair {
                first_stiffness: 1.0,
                second_stiffness: 3.0,
                coupling: 0.25,
            },
            NativeConfig::default()
                .with_masses(Array1::ones(6))
                .with_cell([10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0]),
        )
        .expect("tethered pair engine");
        let curvature = reactant_curvature(&mut engine, &geometry, None).expect("trap curvature");
        let hessian_blocks = [
            [[1.25, -0.25], [-0.25, 3.25]],
            [[1.25, -0.25], [-0.25, 3.25]],
            [[1.25, -0.25], [-0.25, 3.25]],
        ];
        let mut inverse_quadratic_form = 0.0;
        for axis in 0..3 {
            let block = hessian_blocks[axis];
            let determinant = block[0][0] * block[1][1] - block[0][1] * block[1][0];
            let first = tangent[axis];
            let second = tangent[3 + axis];
            inverse_quadratic_form += (block[1][1] * first * first
                - 2.0 * block[0][1] * first * second
                + block[0][0] * second * second)
                / determinant;
        }
        let expected = 1.0 / inverse_quadratic_form;
        assert_abs_diff_eq!(curvature, expected, epsilon = 1.0e-7);
    }

    #[test]
    fn weak_center_of_mass_tether_is_not_silently_projected() {
        let inv_sqrt_two = 0.5_f64.sqrt();
        let tangent = Array1::from_vec(vec![inv_sqrt_two, 0.0, 0.0, inv_sqrt_two, 0.0, 0.0]);
        let geometry = PlaneGeometry {
            anchor: Array1::zeros(6),
            normal: tangent.clone(),
            dn_ds: Array1::zeros(6),
            dc_ds: 1.0,
            tangent,
            arc: 0.0,
        };
        let tether = 1.0e-8;
        let mut engine = NativeEngine::new(
            AnisotropicTetheredPair {
                tether: [tether, 0.0, 0.0],
                coupling: [3.25, 3.25, 3.25],
            },
            NativeConfig::default().with_masses(Array1::ones(6)),
        )
        .expect("weakly tethered pair");
        let curvature = reactant_curvature(
            &mut engine,
            &geometry,
            PafiConfig::default().translation_invariance,
        )
        .expect("weak tether must be retained");
        assert_abs_diff_eq!(curvature, tether, epsilon = 1.0e-12);
    }

    #[test]
    fn declared_translation_invariance_bypasses_numerical_probe_per_axis() {
        let inv_sqrt_two = 0.5_f64.sqrt();
        let tangent = Array1::from_vec(vec![0.0, inv_sqrt_two, 0.0, 0.0, inv_sqrt_two, 0.0]);
        let geometry = PlaneGeometry {
            anchor: Array1::zeros(6),
            normal: tangent.clone(),
            dn_ds: Array1::zeros(6),
            dc_ds: 1.0,
            tangent,
            arc: 0.0,
        };
        let mut engine = NativeEngine::new(
            AnisotropicTetheredPair {
                tether: [1.0, 0.0, 2.0],
                coupling: [3.25, 3.25, 3.25],
            },
            NativeConfig::default().with_masses(Array1::ones(6)),
        )
        .expect("anisotropic pair");
        let error = reactant_curvature(&mut engine, &geometry, Some([false, true, false]))
            .expect_err("declared invariant y translation must be projected");
        assert_eq!(
            error,
            FluxError::InvalidConfig(
                "PAFI reactant tangent lies entirely in the translational zero-mode subspace"
            )
        );
    }

    #[test]
    fn declared_false_keeps_an_invariant_translation_unprojected() {
        let mut tangent = Array1::from_vec(vec![0.8, 0.0, 0.0, 0.6, 0.0, 0.0]);
        let tangent_norm = nrm2(tangent.view());
        tangent.mapv_inplace(|value| value / tangent_norm);
        let geometry = PlaneGeometry {
            anchor: Array1::zeros(6),
            normal: tangent.clone(),
            dn_ds: Array1::zeros(6),
            dc_ds: 1.0,
            tangent,
            arc: 0.0,
        };
        let mut engine = NativeEngine::new(
            IsotropicPair { stiffness: 3.25 },
            NativeConfig::default().with_masses(Array1::ones(6)),
        )
        .expect("invariant pair");
        let error = reactant_curvature(&mut engine, &geometry, Some([false, true, true]))
            .expect_err("declared false x translation must remain in the solve");
        assert_eq!(
            error,
            FluxError::InvalidConfig(
                "PAFI reactant Hessian is not positive definite along the path normal"
            )
        );
    }

    #[test]
    fn translation_probe_is_independent_for_each_anisotropic_axis() {
        let inv_sqrt_two = 0.5_f64.sqrt();
        for (tether, translation_components, expected) in [
            ([0.0, 0.0, 2.0], [0.0, 0.0, 1.0], 2.0),
            ([2.0, 0.0, 0.0], [0.5_f64.sqrt(), 0.5_f64.sqrt(), 0.0], 2.0),
        ] {
            let mut tangent = Array1::zeros(6);
            for axis in 0..3 {
                tangent[axis] = inv_sqrt_two * translation_components[axis];
                tangent[3 + axis] = inv_sqrt_two * translation_components[axis];
            }
            let geometry = PlaneGeometry {
                anchor: Array1::zeros(6),
                normal: tangent.clone(),
                dn_ds: Array1::zeros(6),
                dc_ds: 1.0,
                tangent,
                arc: 0.0,
            };
            let mut engine = NativeEngine::new(
                AnisotropicTetheredPair {
                    tether,
                    coupling: [0.5, 3.25, 1.0e8],
                },
                NativeConfig::default().with_masses(Array1::ones(6)),
            )
            .expect("anisotropic pair");
            let curvature = reactant_curvature(
                &mut engine,
                &geometry,
                PafiConfig::default().translation_invariance,
            )
            .expect("tethered axis must not be projected");
            assert_abs_diff_eq!(curvature, expected, epsilon = 1.0e-7);
        }
    }

    #[test]
    fn finite_difference_uncertainty_reports_ambiguous_axis_and_values() {
        let error = classify_translation_mode(2, 1.5e-8, 0.75e-8, 1.0e-8, None)
            .expect_err("uncertainty interval straddles the roundoff floor");
        let message = error.to_string();
        assert!(message.contains("z axis"));
        assert!(message.contains("2.5000e-9"));
        assert!(message.contains("1.5000e-8"));
        assert!(message.contains("7.5000e-9"));
        assert!(message.contains("1.0000e-8"));
    }

    #[test]
    fn finite_difference_roundoff_uses_every_response_component() {
        let root_mass = Array1::ones(6);
        let origin = Array1::zeros(6);
        let mut direction = Array1::zeros(6);
        direction[0] = 1.0;
        let step = f64::EPSILON.cbrt();
        for index in [0, 5] {
            let mut engine = NativeEngine::new(
                CoordinateLinear { index },
                NativeConfig::default().with_masses(Array1::ones(6)),
            )
            .expect("linear surface");
            let estimate = finite_difference_hessian_product(
                &mut engine,
                origin.view(),
                root_mass.view(),
                direction.view(),
                step,
                0.0,
            )
            .expect("finite-difference estimate");
            assert_eq!(estimate.value, Array1::zeros(6));
            assert_abs_diff_eq!(
                estimate.roundoff_floor,
                f64::EPSILON / step,
                epsilon = f64::EPSILON
            );
        }
    }

    #[test]
    fn position_rounding_floor_matches_operation_error_bound() {
        let stiffness = 3.25;
        let origin = Array1::from_elem(6, 10.0);
        let root_mass = Array1::ones(6);
        let inv_sqrt_two = 0.5_f64.sqrt();
        let direction = Array1::from_vec(vec![inv_sqrt_two, 0.0, 0.0, inv_sqrt_two, 0.0, 0.0]);
        let step = f64::EPSILON.cbrt() * 11.0;
        let hessian_frobenius = 12.0_f64.sqrt() * stiffness;
        let mut engine = NativeEngine::new(
            IsotropicPair { stiffness },
            NativeConfig::default().with_masses(Array1::ones(6)),
        )
        .expect("pair engine");
        let without_position = finite_difference_hessian_product(
            &mut engine,
            origin.view(),
            root_mass.view(),
            direction.view(),
            step,
            0.0,
        )
        .expect("gradient-only floor");
        let with_position = finite_difference_hessian_product(
            &mut engine,
            origin.view(),
            root_mass.view(),
            direction.view(),
            step,
            hessian_frobenius,
        )
        .expect("position-aware floor");

        let unit_roundoff = 0.5 * f64::EPSILON;
        let position_error_norm = |sign: f64| {
            let mut squared = 0.0;
            for i in 0..6 {
                let delta = sign * step * direction[i];
                let multiplication = unit_roundoff * delta.abs();
                let addition =
                    unit_roundoff * (origin[i].abs() + (1.0 + unit_roundoff) * delta.abs());
                squared += (multiplication + addition).powi(2);
            }
            squared.sqrt()
        };
        let expected_position = hessian_frobenius
            * (position_error_norm(1.0) + position_error_norm(-1.0))
            / (2.0 * step);
        assert_abs_diff_eq!(
            with_position.roundoff_floor - without_position.roundoff_floor,
            expected_position,
            epsilon = f64::EPSILON * expected_position
        );
    }

    #[test]
    fn position_rounding_floor_uses_masses_and_adds_gradient_roundoff() {
        let root_mass = Array1::from_vec(vec![1.0, 2.0, 3.0, 1.5, 2.5, 4.0]);
        let masses = root_mass.mapv(|value| value * value);
        let origin = Array1::from_vec(vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0]);
        let mut direction = Array1::from_vec(vec![0.3, -0.2, 0.1, 0.4, -0.5, 0.6]);
        let direction_norm = nrm2(direction.view());
        direction.mapv_inplace(|value| value / direction_norm);
        let step = f64::EPSILON.cbrt() * 7.0;
        let hessian_frobenius = 7.0;
        let mut engine = NativeEngine::new(
            CoordinateLinear { index: 0 },
            NativeConfig::default().with_masses(masses),
        )
        .expect("linear surface");
        let gradient_only = finite_difference_hessian_product(
            &mut engine,
            origin.view(),
            root_mass.view(),
            direction.view(),
            step,
            0.0,
        )
        .expect("gradient-only floor");
        let combined = finite_difference_hessian_product(
            &mut engine,
            origin.view(),
            root_mass.view(),
            direction.view(),
            step,
            hessian_frobenius,
        )
        .expect("combined floor");

        let displacement = &direction / &root_mass;
        let scale = step / nrm2(displacement.view());
        let unit_roundoff = 0.5 * f64::EPSILON;
        let position_error_norm = |sign: f64| {
            let mut squared = 0.0;
            for i in 0..origin.len() {
                let delta = sign * scale * displacement[i];
                let multiplication = unit_roundoff * delta.abs();
                let addition =
                    unit_roundoff * (origin[i].abs() + (1.0 + unit_roundoff) * delta.abs());
                let mass_weighted = root_mass[i] * (multiplication + addition);
                squared += mass_weighted * mass_weighted;
            }
            squared.sqrt()
        };
        let expected_position = hessian_frobenius
            * (position_error_norm(1.0) + position_error_norm(-1.0))
            / (2.0 * scale);
        assert!(gradient_only.roundoff_floor > 0.0);
        assert!(expected_position > 0.0);
        assert_abs_diff_eq!(
            combined.roundoff_floor,
            gradient_only.roundoff_floor + expected_position,
            epsilon = f64::EPSILON * combined.roundoff_floor
        );
    }

    #[test]
    fn richardson_roundoff_floor_uses_both_step_sizes() {
        let origin = Array1::zeros(6);
        let root_mass = Array1::ones(6);
        let direction = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let step = f64::EPSILON.cbrt();
        let mut engine = NativeEngine::new(
            CoordinateLinear { index: 0 },
            NativeConfig::default().with_masses(Array1::ones(6)),
        )
        .expect("linear surface");
        let (_, _, roundoff_floor) = estimate_hessian_product(
            &mut engine,
            origin.view(),
            root_mass.view(),
            direction.view(),
            step,
            0.0,
        )
        .expect("Richardson floor");
        assert_abs_diff_eq!(
            roundoff_floor,
            3.0 * f64::EPSILON / step,
            epsilon = f64::EPSILON
        );
    }

    #[test]
    fn richardson_correction_recovers_anharmonic_hessian() {
        let harmonic = 2.75;
        let quartic = 8.0;
        let origin = Array1::zeros(1);
        let root_mass = Array1::ones(1);
        let direction = Array1::ones(1);
        let step = 0.1;
        let mut engine = NativeEngine::new(
            AnharmonicOneDim { harmonic, quartic },
            NativeConfig::default().with_masses(Array1::ones(1)),
        )
        .expect("anharmonic engine");
        let fine = finite_difference_hessian_product(
            &mut engine,
            origin.view(),
            root_mass.view(),
            direction.view(),
            0.5 * step,
            0.0,
        )
        .expect("fine Hessian product");
        assert_abs_diff_eq!(
            fine.value[0],
            harmonic + quartic * step * step / 4.0,
            epsilon = 1.0e-12
        );
        let (corrected, _, _) = estimate_hessian_product(
            &mut engine,
            origin.view(),
            root_mass.view(),
            direction.view(),
            step,
            0.0,
        )
        .expect("Richardson Hessian product");
        assert_abs_diff_eq!(corrected[0], harmonic, epsilon = 1.0e-12);
    }

    #[test]
    fn reactant_curvature_uses_richardson_product_in_cg_solve() {
        let harmonic = 2.75;
        let quartic = 8.0;
        let anchor = 0.1;
        let geometry = PlaneGeometry {
            anchor: Array1::from_vec(vec![anchor]),
            normal: Array1::ones(1),
            dn_ds: Array1::zeros(1),
            dc_ds: 1.0,
            tangent: Array1::ones(1),
            arc: 0.0,
        };
        let mut engine = NativeEngine::new(
            AnharmonicOneDim { harmonic, quartic },
            NativeConfig::default().with_masses(Array1::ones(1)),
        )
        .expect("anharmonic engine");
        let curvature =
            reactant_curvature(&mut engine, &geometry, None).expect("reactant curvature");
        let expected = harmonic + 3.0 * quartic * anchor * anchor;
        assert_abs_diff_eq!(curvature, expected, epsilon = 1.0e-10);
    }

    #[test]
    fn engine_probe_reports_thin_ambiguous_translation_band() {
        let tangent = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let geometry = PlaneGeometry {
            anchor: Array1::zeros(6),
            normal: tangent.clone(),
            dn_ds: Array1::zeros(6),
            dc_ds: 1.0,
            tangent,
            arc: 0.0,
        };
        let mut engine = NativeEngine::new(
            AmbiguousTranslationPair {
                coupling: 3.25,
                background_gradient: 300.0,
                tether: 1.0e-8,
                quartic: 2_000.0,
            },
            NativeConfig::default().with_masses(Array1::ones(6)),
        )
        .expect("ambiguous translation engine");
        let error =
            reactant_curvature(&mut engine, &geometry, None).expect_err("ambiguous x translation");
        match error {
            FluxError::AmbiguousTranslationMode {
                axis,
                response,
                uncertainty,
                roundoff_floor,
                ..
            } => {
                assert_eq!(axis, "x");
                assert!(
                    response < roundoff_floor && roundoff_floor < response + uncertainty,
                    "expected response < floor < response + uncertainty, got {response:e} < {roundoff_floor:e} < {:e}",
                    response + uncertainty
                );
            }
            other => panic!("expected ambiguous translation mode, got {other}"),
        }
    }

    #[test]
    fn pure_center_of_mass_tangent_reports_translational_zero_mode() {
        let inv_sqrt_two = 0.5_f64.sqrt();
        let tangent = Array1::from_vec(vec![inv_sqrt_two, 0.0, 0.0, inv_sqrt_two, 0.0, 0.0]);
        let geometry = PlaneGeometry {
            anchor: Array1::zeros(6),
            normal: tangent.clone(),
            dn_ds: Array1::zeros(6),
            dc_ds: 1.0,
            tangent,
            arc: 0.0,
        };
        let mut engine = NativeEngine::new(
            IsotropicPair { stiffness: 3.25 },
            NativeConfig::default().with_masses(Array1::ones(6)),
        )
        .expect("invariant pair engine");
        let error =
            reactant_curvature(&mut engine, &geometry, None).expect_err("translation-only path");
        assert_eq!(
            error,
            FluxError::InvalidConfig(
                "PAFI reactant tangent lies entirely in the translational zero-mode subspace"
            )
        );
    }

    #[test]
    fn single_particle_cartesian_trap_remains_unprojected() {
        let mut tangent = Array1::from_vec(vec![1.0, 2.0, -1.0]);
        let tangent_norm = nrm2(tangent.view());
        tangent.mapv_inplace(|value| value / tangent_norm);
        let geometry = PlaneGeometry {
            anchor: Array1::zeros(3),
            normal: tangent.clone(),
            dn_ds: Array1::zeros(3),
            dc_ds: 1.0,
            tangent: tangent.clone(),
            arc: 0.0,
        };
        let stiffness = [1.0, 2.0, 4.0];
        let expected = 1.0
            / tangent
                .iter()
                .zip(stiffness)
                .map(|(&n, k)| n * n / k)
                .sum::<f64>();
        let mut engine = NativeEngine::new(
            SingleParticleTrap { stiffness },
            NativeConfig::default().with_masses(Array1::ones(3)),
        )
        .expect("single-particle engine");
        let curvature = reactant_curvature(&mut engine, &geometry, None).expect("trap curvature");
        assert_abs_diff_eq!(curvature, expected, epsilon = 1.0e-7);
    }

    #[test]
    fn spline_geometry_has_distinct_endpoint_normals() {
        let path =
            Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.2, 1.4, 1.0]).expect("shape");
        let masses = Array1::from_vec(vec![1.0, 3.0]);
        let geometry = build_plane_geometries(path.view(), masses.view(), 7).expect("geometry");
        let difference = (&geometry[0].normal - &geometry[1].normal)
            .mapv(|value| value * value)
            .sum()
            .sqrt();
        assert!(difference > f64::EPSILON.sqrt());
    }

    fn circular_path_in_mass_coordinates(
        images: usize,
        radius: f64,
        theta_start: f64,
        theta_end: f64,
        masses: &Array1<f64>,
    ) -> Array2<f64> {
        let mut path = Array2::zeros((images, 2));
        for i in 0..images {
            let fraction = (i as f64) / ((images - 1) as f64);
            let theta = theta_start + fraction * (theta_end - theta_start);
            path[[i, 0]] = radius * theta.cos() / masses[0].sqrt();
            path[[i, 1]] = radius * theta.sin() / masses[1].sqrt();
        }
        path
    }

    fn circular_normal_derivative(
        radius: f64,
        theta: f64,
        direction: f64,
        masses: &Array1<f64>,
    ) -> Array1<f64> {
        let tangent_mass =
            Array1::from_vec(vec![direction * -theta.sin(), direction * theta.cos()]);
        let curvature_mass = Array1::from_vec(vec![-theta.cos() / radius, -theta.sin() / radius]);
        let mut raw_normal = Array1::zeros(2);
        let mut raw_derivative = Array1::zeros(2);
        for i in 0..2 {
            raw_normal[i] = masses[i].sqrt() * tangent_mass[i];
            raw_derivative[i] = masses[i].sqrt() * curvature_mass[i];
        }
        let norm = nrm2(raw_normal.view());
        let normal = raw_normal.mapv(|value| value / norm);
        let projection = dot(normal.view(), raw_derivative.view());
        raw_derivative.mapv(|value| value / norm) - normal.mapv(|value| value * projection / norm)
    }

    #[test]
    fn circular_spline_preserves_endpoint_normal_curvature() {
        let images = 33usize;
        let radius = 1.3_f64;
        let theta_start = std::f64::consts::FRAC_PI_2;
        let theta_end = 0.0_f64;
        let masses = Array1::from_vec(vec![1.0, 4.0]);
        let path =
            circular_path_in_mass_coordinates(images, radius, theta_start, theta_end, &masses);
        let geometry =
            build_plane_geometries(path.view(), masses.view(), images).expect("geometry");
        let direction = (theta_end - theta_start).signum();
        let tolerance = 2.0 / (images as f64);
        for (index, theta) in [(0, theta_start), (images - 1, theta_end)] {
            let expected = circular_normal_derivative(radius, theta, direction, &masses);
            let error = (&geometry[index].dn_ds - &expected)
                .mapv(|value| value * value)
                .sum()
                .sqrt();
            assert!(
                error < tolerance,
                "endpoint {index} normal-curvature error {error} exceeds {tolerance}"
            );
        }
    }

    fn profile_from(
        arcs: &[f64],
        gradients: &[f64],
        errors: &[f64],
        kbt: f64,
        curvature: f64,
    ) -> Result<FreeEnergyProfile, FluxError> {
        let quadrature = Quadrature::new(arcs)?;
        integrate_profile(&quadrature, gradients, errors, kbt, curvature, true)
    }

    #[test]
    fn profile_uses_interior_maximum_and_accepts_exact_errors() {
        // The four means lie on the cubic g(s) = 2 - 4s/3 - s^2 + s^3/3, which
        // the Hermite rule integrates exactly: F(s) = 2s - 2s^2/3 - s^3/3 +
        // s^4/12 peaks at s = 1 with F = 13/12.
        let arcs = [0.0, 1.0, 2.0, 3.0];
        let gradients = [2.0, 0.0, -2.0, -2.0];
        let errors = [0.0; 4];
        let profile = profile_from(&arcs, &gradients, &errors, 1.0, 4.0).expect("profile");
        assert_eq!(profile.dividing_index, 1);
        assert_abs_diff_eq!(profile.dividing_arc, 1.0, epsilon = 1.0e-12);
        assert_abs_diff_eq!(profile.barrier, 13.0 / 12.0, epsilon = 1.0e-12);
        let trapezoid_barrier = 0.5 * (gradients[0] + gradients[1]);
        let truncation = (profile.barrier - trapezoid_barrier).abs();
        assert!(truncation > 0.0);
        assert_abs_diff_eq!(profile.barrier_error, truncation, epsilon = 1.0e-12);
        assert_eq!(profile.normalized_rate_error, Some(0.0));
        assert_abs_diff_eq!(profile.reactant_frequency, 2.0, epsilon = 1.0e-12);
        // The maximum is the node s = 1, so the reactant integral is the one
        // trapezoid on [0, 1]. F(1) = 13/12.
        let log_partition = (1.0 + (-13.0 / 12.0_f64).exp()).ln();
        assert_abs_diff_eq!(
            profile.reactant_log_partition,
            log_partition,
            epsilon = 1.0e-12
        );
        assert_abs_diff_eq!(
            profile.normalized_log_rate,
            0.5 * (1.0_f64 / TAU).ln() - profile.barrier - log_partition,
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn replicate_pooling_matches_analytic_force_mean_and_standard_error() {
        let arcs = [0.0, 1.0, 2.0];
        let zero_errors = [0.0; 3];
        let kbt = 0.4;
        let curvature = 4.0;
        let replicate_gradients = [[2.0, 0.0, -2.0], [0.0, 1.0, 0.0], [1.0, -1.0, -1.0]];
        let mut replicates: Vec<_> = replicate_gradients
            .iter()
            .map(|gradients| {
                profile_from(&arcs, gradients, &zero_errors, kbt, curvature)
                    .expect("replicate profile")
            })
            .collect();
        for (profile, frequency) in replicates.iter_mut().zip([1.0, 2.0, 3.0]) {
            profile.reactant_frequency = frequency;
        }

        let pooled = pool_pafi_replicates(&replicates, kbt).expect("pooled profile");
        let expected_gradients = [1.0, 0.0, -1.0];
        let expected_gradient_se = (1.0_f64 / 3.0).sqrt();
        assert_eq!(pooled.gradients(), expected_gradients);
        for &error in pooled.gradient_errors() {
            assert_abs_diff_eq!(error, expected_gradient_se, epsilon = 1.0e-15);
        }

        // The pooled force is g(s) = 1 - s, hence F(s) = s - s^2/2 and F*=1/2.
        // Integrating the quadratic interpolant from 0 to 1 has nodal weights
        // (5/12, 2/3, -1/12), which gives SE(F*) = sqrt(30)/12 here.
        let expected_barrier = 0.5;
        let expected_barrier_se = 30.0_f64.sqrt() / 12.0;
        // The maximum is the node s = 1, so Z_R is the doubled trapezoid on
        // [0, 1] and ln k = ln sqrt(kT / 2 pi) - F1 / kT - ln(1 + exp(-F1 / kT)),
        // with F1 the barrier. The reactant frequency does not enter the rate.
        let boltzmann = (-expected_barrier / kbt).exp();
        let expected_rate = (kbt / TAU).sqrt() * boltzmann / (1.0 + boltzmann);
        let expected_rate_se = expected_rate * expected_barrier_se / (kbt * (1.0 + boltzmann));
        let expected_frequency_se = (1.0_f64 / 3.0).sqrt();
        assert_abs_diff_eq!(
            pooled.reactant_frequency_error,
            expected_frequency_se,
            epsilon = 1.0e-15
        );
        assert_abs_diff_eq!(pooled.dividing_arc, 1.0, epsilon = 1.0e-14);
        assert_abs_diff_eq!(pooled.barrier, expected_barrier, epsilon = 1.0e-14);
        assert_abs_diff_eq!(pooled.barrier_error, expected_barrier_se, epsilon = 1.0e-14);
        assert_abs_diff_eq!(
            pooled.normalized_rate.expect("representable pooled rate"),
            expected_rate,
            epsilon = 1.0e-14
        );
        assert_abs_diff_eq!(
            pooled
                .normalized_rate_error
                .expect("representable pooled rate error"),
            expected_rate_se,
            epsilon = 1.0e-14
        );

        let replicate_rates: Vec<f64> = replicates
            .iter()
            .map(|profile| profile.normalized_rate.expect("representable rate"))
            .collect();
        let arithmetic_rate = replicate_rates.iter().sum::<f64>() / 3.0;
        let geometric_rate = (replicates
            .iter()
            .map(|profile| profile.normalized_log_rate)
            .sum::<f64>()
            / 3.0)
            .exp();
        assert!((arithmetic_rate - expected_rate).abs() > 1.0e-3);
        assert!((geometric_rate - expected_rate).abs() > 1.0e-3);
        assert!((replicate_rates[0] - expected_rate).abs() > 1.0e-3);
    }

    #[test]
    fn adaptive_driver_stops_when_no_intervals_are_flagged() {
        let tail = SpectralTailReport {
            force_coefficients: vec![0.0],
            profile_coefficients: vec![0.0],
            tail_indicator: 1.0,
            interval_errors: vec![1.0],
            flagged_intervals: Vec::new(),
        };
        assert!(adaptive_refinement_should_stop(&tail, 0.1, 0, 4, 5, 20));
        let mut flagged = tail;
        flagged.flagged_intervals.push(0);
        assert!(!adaptive_refinement_should_stop(&flagged, 0.1, 0, 4, 5, 20));
    }

    fn quartic_gradient_slope(x: f64) -> f64 {
        12.0 * x * x
    }

    struct AnisotropicHarmonic {
        ax: f64,
        ay: f64,
    }

    impl rgsaddle::PointSurface for AnisotropicHarmonic {
        fn eval(
            &self,
            r: ArrayView1<'_, f64>,
        ) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
            let energy = 0.5 * (self.ax * r[0] * r[0] + 4.0 * self.ay * r[1] * r[1]);
            let gradient = Array1::from_vec(vec![self.ax * r[0], 4.0 * self.ay * r[1]]);
            Ok((energy, gradient))
        }
    }

    const ARC_RADIUS: f64 = 1.2;
    const ARC_FORCE_CONSTANTS: (f64, f64) = (3.0, 1.0);

    fn quarter_arc_engine() -> (NativeEngine<AnisotropicHarmonic>, Array1<f64>) {
        let masses = Array1::from_vec(vec![1.0, 4.0]);
        let engine = NativeEngine::new(
            AnisotropicHarmonic {
                ax: ARC_FORCE_CONSTANTS.0,
                ay: ARC_FORCE_CONSTANTS.1,
            },
            NativeConfig::default()
                .with_masses(masses.clone())
                .with_thermostat(ThermostatConfig::default().with_temperature(0.2)),
        )
        .expect("engine");
        (engine, masses)
    }

    fn quarter_arc_path(images: usize) -> Array2<f64> {
        let mut path = Array2::zeros((images, 2));
        for i in 0..images {
            let fraction = (i as f64) / ((images - 1) as f64);
            let theta = std::f64::consts::FRAC_PI_2 * (1.0 - fraction);
            path[[i, 0]] = ARC_RADIUS * theta.cos();
            path[[i, 1]] = 0.5 * ARC_RADIUS * theta.sin();
        }
        path
    }

    fn quarter_arc_relative_error(images: usize, planes: usize) -> f64 {
        let (mut engine, masses) = quarter_arc_engine();
        let path = quarter_arc_path(images);
        let geometry =
            build_plane_geometries(path.view(), masses.view(), planes).expect("geometry");
        let mut arcs = Vec::with_capacity(planes);
        let mut energies = Vec::with_capacity(planes);
        let mut gradients = Vec::with_capacity(planes);
        for plane in &geometry {
            engine.set_positions(plane.anchor.view()).expect("anchor");
            let (energy, forces) = engine.energy_and_forces().expect("forces");
            arcs.push(plane.arc);
            energies.push(energy);
            gradients.push(-dot(plane.tangent.view(), forces.view()));
        }
        path_force_relative_error(&arcs, &energies, &gradients)
    }

    #[test]
    fn quarter_arc_trapezoid_error_follows_inverse_square_law() {
        // The coefficient is a measured, converged quantity, so the scaled
        // error may differ from the analytic limit only by the spline and
        // higher-order Euler-Maclaurin terms, which stay below a few percent
        // from 9 planes on.
        let agreement = 0.05_f64;
        let images = 33usize;
        for planes in [9usize, 17, 33, 65] {
            let intervals = (planes - 1) as f64;
            let scaled = quarter_arc_relative_error(images, planes) * intervals * intervals;
            assert!(
                (scaled - QUARTER_ARC_TRAPEZOID_COEFFICIENT).abs()
                    <= agreement * QUARTER_ARC_TRAPEZOID_COEFFICIENT,
                "{planes} planes: scaled error {scaled} versus {QUARTER_ARC_TRAPEZOID_COEFFICIENT}"
            );
        }
    }

    #[test]
    fn hermite_rejects_non_finite_gradients() {
        let nan = HermiteIntegral
            .cumulative(&[0.0, 1.0, 2.0], &[0.0, f64::NAN, 1.0])
            .expect_err("nan gradient");
        assert_eq!(nan, FluxError::NonFinite("PAFI Hermite mean force"));

        let inf = HermiteIntegral
            .cumulative(&[0.0, 1.0, 2.0], &[0.0, f64::INFINITY, 1.0])
            .expect_err("infinite gradient");
        assert_eq!(inf, FluxError::NonFinite("PAFI Hermite mean force"));
    }

    #[test]
    fn mean_force_integrals_reject_finite_input_overflow() {
        let hermite = HermiteIntegral
            .cumulative(&[0.0, f64::MAX], &[f64::MAX, f64::MAX])
            .expect_err("hermite overflow");
        assert_eq!(
            hermite,
            FluxError::NonFinite("PAFI Hermite mean-force integral")
        );

        let simpson = SimpsonIntegral
            .cumulative(
                &[0.0, f64::MAX / 2.0, f64::MAX],
                &[f64::MAX, f64::MAX, f64::MAX],
            )
            .expect_err("simpson overflow");
        assert_eq!(
            simpson,
            FluxError::NonFinite("PAFI Simpson mean-force integral")
        );
    }

    #[test]
    fn simpson_rejects_an_even_plane_count_and_integrates_cubics() {
        let even = SimpsonIntegral
            .cumulative(&[0.0, 1.0, 2.0, 3.0], &[0.0, 1.0, 2.0, 3.0])
            .expect_err("even plane count");
        assert_eq!(
            even,
            FluxError::InvalidConfig(
                "Simpson mean-force integration needs an odd number of planes, at least 3"
            )
        );

        let arcs: Vec<f64> = (0..5).map(|k| 0.5 * f64::from(k)).collect();
        let cubic = |s: f64| 1.0 - 2.0 * s + 0.5 * s * s + 0.75 * s * s * s;
        let integral = |s: f64| s - s * s + s * s * s / 6.0 + 0.1875 * s * s * s * s;
        let gradients: Vec<f64> = arcs.iter().map(|&s| cubic(s)).collect();
        let estimate = SimpsonIntegral
            .cumulative(&arcs, &gradients)
            .expect("simpson");
        for (k, (value, &s)) in estimate.values.iter().zip(&arcs).enumerate() {
            if k % 2 == 0 {
                assert_abs_diff_eq!(*value, integral(s), epsilon = 1.0e-12);
            }
        }
        assert_eq!(estimate.error[0], 0.0);
        assert!(estimate.error[4] > 0.0);
        assert!(estimate.error.iter().all(|error| error.is_finite()));
    }

    #[test]
    fn default_tolerance_accepts_quarter_arc_at_default_planes() {
        let config = PafiConfig::default();
        let tolerance = effective_zero_temperature_tol(&config);
        let error = quarter_arc_relative_error(33, config.planes);
        assert!(
            error < tolerance,
            "relative error {error} versus default tolerance {tolerance}"
        );
        assert!(
            tolerance < 2.0 * (1.0 + ZERO_TEMPERATURE_SAFETY_MARGIN) * error,
            "default tolerance {tolerance} is no longer calibrated to {error}"
        );

        let (mut engine, _) = quarter_arc_engine();
        let path = quarter_arc_path(config.planes);
        let result = pafi_profile(
            &mut engine,
            path.view(),
            &config
                .with_burn_in_steps(1)
                .with_sample_steps(16)
                .with_stats(StatsConfig::default().with_min_batches(2)),
        );
        assert!(result.is_ok(), "default configuration returned {result:?}");
    }

    #[test]
    fn log_rate_stays_finite_when_the_barrier_underflows() {
        let arcs = [0.0, 1.0, 2.0, 3.0];
        let gradients = [800.0, 800.0, 800.0, 800.0];
        let errors = [0.0; 4];
        let curvature = 4.0;
        let profile = profile_from(&arcs, &gradients, &errors, 1.0, curvature).expect("profile");
        assert!(profile.barrier > 700.0);
        assert!(profile.normalized_rate.is_none());
        assert!(profile.reactant_log_partition.is_finite());
        let log_rate =
            0.5 * (1.0_f64 / TAU).ln() - profile.barrier - profile.reactant_log_partition;
        assert_abs_diff_eq!(profile.normalized_log_rate, log_rate, epsilon = 1.0e-9);
        assert!(
            profile
                .reactant_partition
                .is_some_and(|partition| partition.is_finite() && partition > 0.0)
        );
        assert_abs_diff_eq!(
            profile.reactant_frequency,
            curvature.sqrt(),
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn log_fields_stay_finite_for_deep_wells_large_beta_and_soft_modes() {
        // `(label, mean force, kT, reactant curvature)`.
        // A deep well makes `exp(-ΔF / kT)` underflow. A large beta is a small
        // `kT`. A soft mode is a tiny projected frequency. The log partition
        // and the log rate stay finite in every combination.
        let cases = [
            ("deep well", 2_500.0, 1.0, 4.0),
            ("large beta", 25.0, 1.0e-4, 4.0),
            ("soft mode", 4.0, 1.0, 1.0e-16),
            (
                "deep well, large beta, and soft mode",
                800.0,
                1.0e-3,
                1.0e-12,
            ),
        ];
        let arcs = [0.0, 0.5, 1.25, 2.0];
        for (label, slope, kbt, curvature) in cases {
            let gradients = [slope, slope, slope, slope];
            let profile = profile_from(&arcs, &gradients, &[0.0; 4], kbt, curvature)
                .unwrap_or_else(|err| panic!("{label}: {err}"));
            let frequency = curvature.sqrt();
            assert_abs_diff_eq!(
                profile.reactant_frequency,
                frequency,
                epsilon = 16.0 * f64::EPSILON * frequency.max(1.0)
            );
            assert!(
                profile.barrier.is_finite() && profile.barrier > 0.0,
                "{label}"
            );
            assert!(profile.reactant_log_partition.is_finite(), "{label}");
            let log_rate =
                0.5 * (kbt / TAU).ln() - profile.barrier / kbt - profile.reactant_log_partition;
            assert!(log_rate.is_finite(), "{label}");
            let rate_scale = log_rate.abs().max(1.0);
            assert_abs_diff_eq!(
                profile.normalized_log_rate,
                log_rate,
                epsilon = 16.0 * f64::EPSILON * rate_scale
            );
            assert!(
                profile
                    .normalized_rate
                    .is_none_or(|rate| rate.is_finite() && rate >= 0.0),
                "{label}"
            );
            assert!(
                profile
                    .reactant_partition
                    .is_some_and(|partition| partition.is_finite() && partition > 0.0),
                "{label}"
            );
        }
    }

    #[test]
    fn log_fields_match_the_line_integral_rate() {
        let arcs = [0.0, 0.5, 1.0, 1.5];
        let gradients = [0.3, 0.1, -0.4, -0.6];
        let errors = [0.02, 0.03, 0.01, 0.02];
        let kbt = 1.2;
        let curvature = 5.0;
        let profile = profile_from(&arcs, &gradients, &errors, kbt, curvature).expect("profile");
        let frequency = curvature.sqrt();
        assert_abs_diff_eq!(
            profile.reactant_frequency,
            frequency,
            epsilon = f64::EPSILON * frequency
        );
        assert_abs_diff_eq!(
            profile.reactant_log_partition,
            profile
                .reactant_partition
                .expect("representable reactant partition")
                .ln(),
            epsilon = 1.0e-12
        );
        let flux = (kbt / TAU).sqrt() * (-profile.barrier / kbt).exp()
            / profile
                .reactant_partition
                .expect("representable reactant partition");
        assert_abs_diff_eq!(
            profile
                .normalized_rate
                .expect("representable normalized rate"),
            flux,
            epsilon = f64::EPSILON * flux.abs()
        );
        assert_abs_diff_eq!(
            profile.normalized_log_rate,
            flux.ln(),
            epsilon = 16.0 * f64::EPSILON * flux.ln().abs().max(1.0)
        );
        assert!(
            profile
                .normalized_rate_error
                .is_some_and(|error| error.is_finite() && error > 0.0)
        );
    }

    fn polynomial_integral_error(planes: usize, degree: u32) -> f64 {
        let arcs: Vec<f64> = (0..planes)
            .map(|k| f64::from(k as u32) / f64::from((planes - 1) as u32))
            .collect();
        let gradients: Vec<f64> = arcs.iter().map(|s| s.powi(degree as i32)).collect();
        let quadrature = Quadrature::new(&arcs).expect("quadrature");
        let integrated = quadrature.integrate(&gradients);
        arcs.iter()
            .zip(integrated)
            .map(|(s, value)| {
                let exact = if degree == 0 {
                    *s
                } else {
                    s.powi(degree as i32 + 1) / f64::from(degree + 1)
                };
                (value - exact).abs()
            })
            .fold(0.0_f64, f64::max)
    }

    #[test]
    fn hermite_exactness_follows_the_plane_count() {
        let roundoff = 1.0e-12;
        assert!(polynomial_integral_error(2, 1) <= roundoff);
        assert!(polynomial_integral_error(2, 2) > roundoff);
        assert!(polynomial_integral_error(3, 2) <= roundoff);
        assert!(polynomial_integral_error(3, 3) > roundoff);
        assert!(polynomial_integral_error(4, 3) <= roundoff);
        assert!(polynomial_integral_error(4, 4) > roundoff);
        assert!(polynomial_integral_error(7, 3) <= roundoff);
    }

    #[test]
    fn dividing_plane_search_finds_an_overshoot_away_from_the_highest_sample() {
        // Seven samples. Plane 0 holds the highest sample, while the cubic
        // Hermite interpolant on the interval between planes 2 and 3 rises
        // above every sample. A search that only inspects the intervals next
        // to the highest sample stays on the first interval.
        let arcs: Vec<f64> = (0..7).map(f64::from).collect();
        let gradients = [1.0, -1.0, 4.0, -4.0, -3.0, -2.0, -1.0];
        let errors = [0.0; 7];
        let profile = profile_from(&arcs, &gradients, &errors, 1.0, 1.0).expect("profile");
        let highest_sample = profile
            .free_energy
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .expect("a sample")
            .0;
        assert_eq!(highest_sample, 0);
        assert_eq!(profile.dividing_index, 0);
        assert!(profile.dividing_arc > arcs[2] && profile.dividing_arc < arcs[3]);
        assert!(profile.barrier > profile.free_energy[0]);
    }

    #[test]
    fn dividing_plane_between_planes_removes_the_grid_maximum_bias() {
        let arcs: Vec<f64> = (0..8).map(|k| f64::from(k) * 0.5).collect();
        let peak = 1.7_f64;
        let gradients: Vec<f64> = arcs.iter().map(|s| -(s - peak)).collect();
        let profile = profile_from(&arcs, &gradients, &[0.0; 8], 1.0, 1.0).expect("profile");
        assert_abs_diff_eq!(profile.dividing_arc, peak, epsilon = 1.0e-10);
        assert_abs_diff_eq!(profile.barrier, peak * peak / 2.0, epsilon = 1.0e-10);
        assert!(
            profile
                .free_energy
                .iter()
                .all(|value| *value < profile.barrier)
        );
    }

    fn dense_profile_maximum(arcs: &[f64], gradients: &[f64], samples: usize) -> f64 {
        let quadrature = Quadrature::new(arcs).expect("quadrature");
        let free_energy = quadrature.integrate(gradients);
        let mut best = f64::NEG_INFINITY;
        for k in 0..arcs.len() - 1 {
            let scaled = quadrature.scaled_slopes(k, gradients).expect("slopes");
            for j in 0..=samples {
                let u = f64::from(u32::try_from(j).expect("sample index"))
                    / f64::from(u32::try_from(samples).expect("sample count"));
                let value = quadrature.interpolated_value(k, u, gradients, scaled, &free_energy);
                if value > best {
                    best = value;
                }
            }
        }
        best
    }

    fn sampled_local_maxima(arcs: &[f64], gradients: &[f64]) -> usize {
        let quadrature = Quadrature::new(arcs).expect("quadrature");
        let free_energy = quadrature.integrate(gradients);
        let steps = 4_000usize;
        let mut count = 0usize;
        let mut previous = f64::NEG_INFINITY;
        let mut current = free_energy[0];
        for k in 0..arcs.len() - 1 {
            let scaled = quadrature.scaled_slopes(k, gradients).expect("slopes");
            for j in 1..=steps {
                let u = f64::from(u32::try_from(j).expect("step"))
                    / f64::from(u32::try_from(steps).expect("steps"));
                let next = quadrature.interpolated_value(k, u, gradients, scaled, &free_energy);
                if current > previous && current > next {
                    count += 1;
                }
                previous = current;
                current = next;
            }
        }
        count
    }

    fn assert_exact_maximum_covers_dense_sample(arcs: &[f64], gradients: &[f64]) {
        let quadrature = Quadrature::new(arcs).expect("quadrature");
        let free_energy = quadrature.integrate(gradients);
        let exact = quadrature
            .dividing_plane(gradients, &free_energy)
            .expect("dividing plane")
            .free_energy;
        let samples = 1_000_000usize;
        let brute = dense_profile_maximum(arcs, gradients, samples);
        let scale = exact.abs().max(brute.abs()).max(1.0);
        let rounding = 128.0 * f64::EPSILON * scale;
        assert!(
            exact + rounding >= brute,
            "exact maximum {exact:.16e} is below the {samples}-point sample {brute:.16e}"
        );
        assert!(
            exact - brute <= rounding.max(1.0e-9),
            "exact maximum {exact:.16e} and the {samples}-point sample {brute:.16e} differ by more than rounding"
        );
    }

    #[test]
    fn derivative_roots_cover_quadratic_degeneracies_and_three_real_roots() {
        let (count, roots) = quadratic_real_roots(0.0, 2.0, -0.5);
        assert_eq!(count, 1);
        assert_abs_diff_eq!(roots[0], 0.25, epsilon = 1.0e-15);

        let (count, _) = quadratic_real_roots(0.0, 0.0, 3.0);
        assert_eq!(count, 0);

        let (count, _) = quadratic_real_roots(1.0, 0.0, 1.0);
        assert_eq!(count, 0);

        let (count, mut roots) = quadratic_real_roots(1.0, 0.0, -0.25);
        assert_eq!(count, 2);
        roots[..count].sort_by(f64::total_cmp);
        assert_abs_diff_eq!(roots[0], -0.5, epsilon = 1.0e-15);
        assert_abs_diff_eq!(roots[1], 0.5, epsilon = 1.0e-15);

        let (count, mut roots) = quadratic_real_roots(1.0, -1.0, 0.16);
        assert_eq!(count, 2);
        roots[..count].sort_by(f64::total_cmp);
        assert_abs_diff_eq!(roots[0], 0.2, epsilon = 1.0e-12);
        assert_abs_diff_eq!(roots[1], 0.8, epsilon = 1.0e-12);

        let (count, mut roots) = cubic_unit_interval_roots(0.0, 1.0, -1.0, 0.16);
        assert_eq!(count, 2);
        roots[..count].sort_by(f64::total_cmp);
        assert_abs_diff_eq!(roots[0], 0.2, epsilon = 1.0e-12);
        assert_abs_diff_eq!(roots[1], 0.8, epsilon = 1.0e-12);

        let (count, roots) = cubic_unit_interval_roots(0.0, 0.0, 2.0, -0.5);
        assert_eq!(count, 1);
        assert_abs_diff_eq!(roots[0], 0.25, epsilon = 1.0e-15);

        let (count, _) = cubic_unit_interval_roots(0.0, 0.0, 0.0, 4.0);
        assert_eq!(count, 0);

        let left = 0.2_f64;
        let middle = 0.5_f64;
        let right = 0.8_f64;
        let (count, mut roots) = cubic_unit_interval_roots(
            1.0,
            -(left + middle + right),
            left * middle + left * right + middle * right,
            -left * middle * right,
        );
        assert_eq!(count, 3);
        roots[..count].sort_by(f64::total_cmp);
        assert_abs_diff_eq!(roots[0], left, epsilon = 1.0e-12);
        assert_abs_diff_eq!(roots[1], middle, epsilon = 1.0e-12);
        assert_abs_diff_eq!(roots[2], right, epsilon = 1.0e-12);
    }

    /// Roots near `0.8` and `0.2` of `a s^3 - s^2 + s - 0.16`, and the value
    /// of its antiderivative `a s^4 / 4 - s^3 / 3 + s^2 / 2 - 0.16 s` at the
    /// larger root, for each leading coefficient `a`. They come from Newton's
    /// method in 80-digit decimal arithmetic.
    const NEAR_QUADRATIC_REFERENCE: [(f64, f64, f64, f64); 8] = [
        (0.0, 8e-1, 2e-1, 2.133_333_333_333_333_3e-2),
        (1e-16, 8e-1, 2e-1, 2.133_333_333_333_334_3e-2),
        (
            1e-15,
            8.000_000_000_000_008e-1,
            1.999_999_999_999_999_8e-1,
            2.133_333_333_333_343_7e-2,
        ),
        (
            1e-12,
            8.000_000_000_008_534e-1,
            1.999_999_999_999_866_6e-1,
            2.133_333_333_343_573_3e-2,
        ),
        (
            1e-10,
            8.000_000_000_853_333e-1,
            1.999_999_999_986_666_6e-1,
            2.133_333_334_357_333_2e-2,
        ),
        (
            1e-8,
            8.000_000_085_333_335e-1,
            1.999_999_998_666_666_7e-1,
            2.133_333_435_733_335_6e-2,
        ),
        (
            1e-6,
            8.000_008_533_348_504e-1,
            1.999_999_866_666_696_3e-1,
            2.133_343_573_355_178_7e-2,
        ),
        (
            1e-3,
            8.008_548_538_312_802e-1,
            1.999_866_696_287_278_8e-1,
            2.143_595_227_978_177_2e-2,
        ),
    ];

    #[test]
    fn cubic_roots_match_a_high_precision_reference_for_a_vanishing_cubic_term() {
        let root_tolerance = 64.0 * f64::EPSILON;
        for (a, upper, lower, _) in NEAR_QUADRATIC_REFERENCE {
            let (count, roots) = cubic_unit_interval_roots(a, -1.0, 1.0, -0.16);
            let mut inside: Vec<f64> = roots[..count]
                .iter()
                .copied()
                .filter(|root| (0.0..=1.0).contains(root))
                .collect();
            inside.sort_by(f64::total_cmp);
            assert_eq!(inside.len(), 2, "a = {a:e}: roots {inside:?}");
            assert_abs_diff_eq!(inside[0], lower, epsilon = root_tolerance);
            assert_abs_diff_eq!(inside[1], upper, epsilon = root_tolerance);
        }
    }

    #[test]
    fn dividing_plane_matches_a_high_precision_reference_for_a_vanishing_cubic_term() {
        // The mean force `a s^3 - s^2 + s - 0.16` on four planes is a cubic that
        // the Hermite rule integrates exactly, so the barrier is its
        // antiderivative at the larger root.
        let arcs = [0.0, 1.0, 2.0, 3.0];
        for (a, upper, _, barrier) in NEAR_QUADRATIC_REFERENCE {
            let gradients: Vec<f64> = arcs
                .iter()
                .map(|s| a * s * s * s - s * s + s - 0.16)
                .collect();
            let quadrature = Quadrature::new(&arcs).expect("quadrature");
            let free_energy = quadrature.integrate(&gradients);
            let plane = quadrature
                .dividing_plane(&gradients, &free_energy)
                .expect("dividing plane");
            assert_abs_diff_eq!(plane.arc, upper, epsilon = 1.0e-12);
            assert_abs_diff_eq!(plane.free_energy, barrier, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn segment_derivative_matches_the_cubic_hermite_force() {
        let arcs = [0.0, 0.3, 1.1, 1.7];
        let gradients = [0.4, -1.2, 0.7, 0.2];
        let quadrature = Quadrature::new(&arcs).expect("quadrature");
        for k in 0..arcs.len() - 1 {
            let g0 = gradients[k];
            let g1 = gradients[k + 1];
            let (hs0, hs1) = quadrature.scaled_slopes(k, &gradients).expect("slopes");
            let magnitude = g0.abs().max(g1.abs()).max(hs0.abs()).max(hs1.abs());
            let [a, b, c, d] = segment_force_coefficients(g0, g1, (hs0, hs1));
            for j in 0..=8 {
                let u = f64::from(j) / 8.0;
                let u2 = u * u;
                let u3 = u2 * u;
                let force = g0 * (2.0 * u3 - 3.0 * u2 + 1.0)
                    + g1 * (-2.0 * u3 + 3.0 * u2)
                    + hs0 * (u3 - 2.0 * u2 + u)
                    + hs1 * (u3 - u2);
                let polynomial = a.mul_add(u, b).mul_add(u, c).mul_add(u, d);
                assert_abs_diff_eq!(polynomial * magnitude, force, epsilon = 1.0e-12);
            }
        }
    }

    #[test]
    fn exact_segment_maximum_covers_a_million_point_sample() {
        let oscillating_arcs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let oscillating = [8.0, -6.0, 7.0, -9.0, 5.0, -4.0];
        assert!(
            sampled_local_maxima(&oscillating_arcs, &oscillating) >= 2,
            "the oscillating profile should have several local maxima"
        );
        assert_exact_maximum_covers_dense_sample(&oscillating_arcs, &oscillating);

        let descending_arcs = [0.0, 0.4, 1.1, 2.3];
        let descending = [-1.0, -1.0, -1.0, -1.0];
        let descending_quadrature = Quadrature::new(&descending_arcs).expect("quadrature");
        let descending_energy = descending_quadrature.integrate(&descending);
        let descending_plane = descending_quadrature
            .dividing_plane(&descending, &descending_energy)
            .expect("dividing plane");
        assert_eq!(descending_plane.arc, descending_arcs[0]);
        assert_eq!(descending_plane.free_energy, 0.0);
        assert_exact_maximum_covers_dense_sample(&descending_arcs, &descending);

        let ascending_arcs = [0.0, 0.5, 1.25, 2.0, 3.5];
        let ascending = [1.5, 1.5, 1.5, 1.5, 1.5];
        let ascending_profile =
            profile_from(&ascending_arcs, &ascending, &[0.0; 5], 1.0, 1.0).expect("profile");
        let end = *ascending_arcs.last().expect("end");
        assert_abs_diff_eq!(ascending_profile.dividing_arc, end, epsilon = 1.0e-12);
        assert_abs_diff_eq!(ascending_profile.barrier, 1.5 * end, epsilon = 1.0e-12);
        assert_exact_maximum_covers_dense_sample(&ascending_arcs, &ascending);
    }

    #[test]
    fn non_finite_profile_input_is_rejected() {
        let arcs = [0.0, 1.0, 2.0];
        let gradients = [0.0, f64::NAN, 1.0];
        let errors = [0.0, 0.0, 0.0];
        let result = profile_from(&arcs, &gradients, &errors, 1.0, 1.0);
        assert_eq!(
            result,
            Err(FluxError::NonFinite("PAFI integrated free energy"))
        );
    }

    fn slope_form_trapezoid_to(arcs: &[f64], gradients: &[f64], arc: f64) -> f64 {
        let mut accumulated = 0.0_f64;
        for k in 0..arcs.len() - 1 {
            let left = arcs[k];
            if arc <= left {
                break;
            }
            let width = arcs[k + 1] - left;
            let fraction = ((arc.min(arcs[k + 1]) - left) / width).clamp(0.0, 1.0);
            let slope = gradients[k + 1] - gradients[k];
            accumulated += width * (gradients[k] * fraction + slope * fraction * fraction / 2.0);
            if arc <= arcs[k + 1] {
                break;
            }
        }
        accumulated
    }

    #[test]
    fn trapezoid_to_matches_the_slope_form_on_ordinary_input() {
        let arcs = [0.0, 0.7, 1.9, 2.0, 3.5, 5.25];
        let gradients = [2.0, -0.3, 4.5, 1.25e3, -7.0e2, 0.125];
        let mut probes = vec![-1.0, 0.0, 6.0, 1.0e3];
        probes.extend_from_slice(&arcs);
        for window in arcs.windows(2) {
            for fraction in [0.1, 0.25, 0.5, 0.9] {
                probes.push(window[0] + fraction * (window[1] - window[0]));
            }
        }
        for arc in probes {
            let new = trapezoid_to(&arcs, &gradients, arc, "test").expect("finite");
            let old = slope_form_trapezoid_to(&arcs, &gradients, arc);
            assert_abs_diff_eq!(new, old, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn trapezoid_to_rejects_non_finite_input() {
        let arcs = [0.0, 1.0, 2.0];
        let gradients = [1.0, 2.0, 3.0];
        assert!(slope_form_trapezoid_to(&arcs, &gradients, f64::NAN).is_finite());
        for arc in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                trapezoid_to(&arcs, &gradients, arc, "test arc"),
                Err(FluxError::NonFinite("test arc"))
            );
        }
        assert_eq!(
            trapezoid_to(&arcs, &[1.0, f64::NAN, 3.0], 2.0, "test gradient"),
            Err(FluxError::NonFinite("test gradient"))
        );
    }

    #[test]
    fn trapezoid_to_survives_a_gradient_difference_that_overflows() {
        let arcs = [0.0, 1.0];
        let gradients = [-1.0e308_f64, 1.0e308];
        assert!((gradients[1] - gradients[0]).is_infinite());
        assert!(!slope_form_trapezoid_to(&arcs, &gradients, 1.0).is_finite());
        assert_abs_diff_eq!(
            trapezoid_to(&arcs, &gradients, 1.0, "test").expect("finite"),
            0.0,
            epsilon = 1.0e-12
        );
        let half = trapezoid_to(&arcs, &gradients, 0.5, "test").expect("finite");
        assert_abs_diff_eq!(half / 1.0e308, -0.25, epsilon = 1.0e-12);
    }

    #[test]
    fn trapezoid_to_rejects_a_result_that_overflows() {
        let arcs = [0.0, 2.0];
        let gradients = [1.0e308, 1.0e308];
        assert_eq!(
            trapezoid_to(&arcs, &gradients, 2.0, "test result"),
            Err(FluxError::NonFinite("test result"))
        );
    }

    #[test]
    fn non_positive_reactant_curvature_is_rejected() {
        let arcs = [0.0, 1.0, 2.0];
        let gradients = [1.0, 0.0, -1.0];
        let errors = [0.0, 0.0, 0.0];
        let result = profile_from(&arcs, &gradients, &errors, 1.0, 0.0);
        assert_eq!(
            result,
            Err(FluxError::InvalidConfig(
                "PAFI reactant curvature must be finite and positive"
            ))
        );
    }

    #[test]
    fn a_descending_profile_has_no_reactant_interval() {
        let arcs = [0.0, 1.0, 2.0];
        let gradients = [-1.0, -1.0, -1.0];
        let errors = [0.0, 0.0, 0.0];
        let result = profile_from(&arcs, &gradients, &errors, 1.0, 2.0);
        assert_eq!(
            result,
            Err(FluxError::InvalidConfig(
                "PAFI reactant partition must be finite and positive"
            ))
        );
    }

    /// Relative tolerance for a log-space rate error against the delta method.
    const NORMALIZED_RATE_ERROR_RELATIVE_TOLERANCE: f64 = 16.0 * f64::EPSILON;

    /// The rounded delta-method error `5.26e-156` sits this far from the
    /// value carried through `f64` operations.
    const FAR_BELOW_RANGE_RATE_ERROR_TOLERANCE: f64 = 256.0 * f64::EPSILON;

    /// `exp(ln|weight| + ln|sigma|)` near the top of the finite range.
    const EXTREME_SIGMA_NORM_RELATIVE_TOLERANCE: f64 = 1024.0 * f64::EPSILON;

    fn assert_relative_close(actual: f64, expected: f64, tolerance: f64) {
        let scale = expected.abs();
        let discrepancy = (actual - expected).abs();
        assert!(
            scale > 0.0 && discrepancy <= tolerance * scale,
            "actual {actual} expected {expected} discrepancy {discrepancy} tolerance {tolerance}"
        );
    }

    #[test]
    fn reactant_log_partition_stays_finite_for_deep_free_energy_wells() {
        let arcs = [0.0, 1.0, 2.0, 3.0];
        let gradients = [-800.0, -800.0, 800.0, 820.0];
        let errors = [0.0; 4];
        let profile = profile_from(&arcs, &gradients, &errors, 1.0, 4.0).expect("profile");

        let mut naive_half_partition = 0.0_f64;
        for i in 1..=profile.dividing_index {
            let ds = arcs[i] - arcs[i - 1];
            let left = (-profile.free_energy[i - 1]).exp();
            let right = (-profile.free_energy[i]).exp();
            naive_half_partition += 0.5 * ds * (left + right);
        }
        assert!(!naive_half_partition.is_finite());

        let shift = profile.free_energy[..=profile.dividing_index]
            .iter()
            .map(|&fe| -fe)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut scaled_half = 0.0_f64;
        for i in 1..=profile.dividing_index {
            let ds = arcs[i] - arcs[i - 1];
            let left = (-profile.free_energy[i - 1] - shift).exp();
            let right = (-profile.free_energy[i] - shift).exp();
            scaled_half += 0.5 * ds * (left + right);
        }
        let expected_ln_partition = LN_2 + shift + scaled_half.ln();
        assert_abs_diff_eq!(
            profile.reactant_log_partition,
            expected_ln_partition,
            epsilon = f64::EPSILON * expected_ln_partition.abs()
        );
        assert!(profile.normalized_log_rate.is_finite());
        assert!(profile.reactant_log_partition > f64::MAX.ln());
        assert!(profile.reactant_partition.is_none());
        assert_eq!(profile.normalized_log_rate.exp(), 0.0);
        assert!(profile.normalized_rate.is_none());
        assert_eq!(profile.normalized_rate_error, Some(0.0));
    }

    #[test]
    fn underflowing_linear_rate_is_none_while_its_error_stays_zero() {
        let arcs = [0.0, 1.0, 2.0, 3.0];
        let gradients = [-800.0, -800.0, 800.0, 820.0];
        let errors = [0.2, 0.2, 0.2, 0.2];
        let profile = profile_from(&arcs, &gradients, &errors, 1.0, 4.0).expect("profile");
        assert!(profile.normalized_log_rate.is_finite());
        assert_eq!(profile.normalized_log_rate.exp(), 0.0);
        assert!(profile.normalized_rate.is_none());
        assert_eq!(profile.normalized_rate_error, Some(0.0));
        assert!(profile.reactant_log_partition > f64::MAX.ln());
        assert!(profile.reactant_partition.is_none());
    }

    #[test]
    fn log_rate_above_the_f64_range_is_accepted() {
        let arcs = [0.0, (-400.0_f64).exp()];
        let gradients = [2.0, 2.0];
        let errors = [1.0, 1.0];
        let profile = profile_from(&arcs, &gradients, &errors, f64::MAX, 4.0).expect("profile");
        assert!(profile.normalized_log_rate.is_finite());
        assert!(profile.normalized_log_rate > f64::MAX.ln());
        assert!(profile.normalized_rate.is_none());
        assert_eq!(profile.normalized_rate_error, Some(0.0));
        assert!(profile.reactant_partition.is_some());
    }

    #[test]
    fn reactant_partition_sensitivity_matches_finite_differences() {
        let arcs = [0.0, 0.5, 1.0, 1.5];
        let gradients = [0.3, 0.1, -0.4, -0.6];
        let kbt = 1.2;
        let quadrature = Quadrature::new(&arcs).expect("quadrature");
        let free_energy = quadrature.integrate(&gradients);
        let dividing = quadrature
            .dividing_plane(&gradients, &free_energy)
            .expect("dividing plane");
        let (_, sensitivity) = reactant_log_partition_and_sensitivity(
            &arcs,
            &free_energy,
            dividing.partition_end,
            kbt,
        )
        .expect("partition");
        let step = f64::EPSILON.cbrt();
        for index in 0..free_energy.len() {
            let mut up = free_energy.clone();
            let mut down = free_energy.clone();
            up[index] += step;
            down[index] -= step;
            let (ln_up, _) =
                reactant_log_partition_and_sensitivity(&arcs, &up, dividing.partition_end, kbt)
                    .expect("up");
            let (ln_down, _) =
                reactant_log_partition_and_sensitivity(&arcs, &down, dividing.partition_end, kbt)
                    .expect("down");
            let fd = (ln_up - ln_down) / (2.0 * step);
            let scale = sensitivity[index].abs().max(1.0);
            let discrepancy = (fd - sensitivity[index]).abs();
            assert!(
                discrepancy <= step * scale,
                "d ln Z / d F[{index}] analytic {} fd {fd} discrepancy {discrepancy}",
                sensitivity[index]
            );
        }
    }

    #[test]
    fn normalized_rate_error_matches_finite_differences() {
        let arcs = [0.0, 0.5, 1.0, 1.5];
        let gradients = [0.3, 0.1, -0.4, -0.6];
        let errors = [0.02, 0.03, 0.01, 0.02];
        let kbt = 1.2;
        let curvature = 4.0;
        let profile = profile_from(&arcs, &gradients, &errors, kbt, curvature).expect("profile");
        let rate = profile.normalized_rate.expect("finite rate");
        let reported = profile.normalized_rate_error.expect("finite rate error");
        let step = f64::EPSILON.cbrt();
        let mut variance = 0.0_f64;
        for index in 0..gradients.len() {
            let mut up = gradients;
            let mut down = gradients;
            up[index] += step;
            down[index] -= step;
            let ln_up = profile_from(&arcs, &up, &errors, kbt, curvature)
                .expect("up")
                .normalized_log_rate;
            let ln_down = profile_from(&arcs, &down, &errors, kbt, curvature)
                .expect("down")
                .normalized_log_rate;
            let slope = (ln_up - ln_down) / (2.0 * step);
            variance += slope * slope * errors[index] * errors[index];
        }
        let fd_error = rate * variance.sqrt();
        let scale = reported.abs().max(rate);
        let discrepancy = (fd_error - reported).abs();
        assert!(
            discrepancy <= step * scale,
            "normalized_rate_error {reported} fd {fd_error} discrepancy {discrepancy}"
        );
    }

    fn rate_far_below_linear_range_inputs() -> (f64, f64, f64) {
        let plane_error: f64 = 1.0e200;
        let target_log_rate = -400.0 - f64::MAX.ln();
        let ln_sigma = 5.26e-156_f64.ln() - target_log_rate;
        let ln_term = ln_sigma - 0.5 * 2.0_f64.ln();
        let kbt = (0.5_f64.ln() + plane_error.ln() - ln_term).exp();
        let barrier_over_kbt = 0.5 * (kbt / TAU).ln() + 400.0 + f64::MAX.ln();
        let gradient = barrier_over_kbt * kbt;
        (kbt, gradient, plane_error)
    }

    fn previous_linear_rate_error(
        arcs: &[f64],
        gradients: &[f64],
        errors: &[f64],
        kbt: f64,
        rate: f64,
    ) -> f64 {
        let quadrature = Quadrature::new(arcs).expect("quadrature");
        let free_energy = quadrature.integrate(gradients);
        let dividing = quadrature
            .dividing_plane(gradients, &free_energy)
            .expect("dividing plane");
        let (_, d_ln_zr_d_f) =
            reactant_log_partition_and_sensitivity(arcs, &free_energy, dividing.partition_end, kbt)
                .expect("partition");
        let mut variance = 0.0_f64;
        for (gradient_index, sigma) in errors.iter().copied().enumerate() {
            let mut partition_log_sensitivity = 0.0_f64;
            for (sensitivity, cumulative) in d_ln_zr_d_f
                .iter()
                .zip(&quadrature.cumulative)
                .take(dividing.partition_end + 1)
            {
                partition_log_sensitivity += sensitivity * cumulative[gradient_index];
            }
            let sensitivity = -dividing.weights[gradient_index] / kbt - partition_log_sensitivity;
            variance += sensitivity * sensitivity * sigma * sigma;
        }
        rate * variance.sqrt()
    }

    #[test]
    fn normalized_rate_error_matches_delta_method_below_linear_range() {
        let (kbt, gradient, plane_error) = rate_far_below_linear_range_inputs();
        let arcs = [0.0, 1.0];
        let gradients = [gradient, gradient];
        let errors = [plane_error, plane_error];
        let profile = profile_from(&arcs, &gradients, &errors, kbt, 4.0).expect("profile");
        let target_log_rate = -400.0 - f64::MAX.ln();
        assert_eq!(profile.normalized_log_rate, target_log_rate);
        assert_eq!(profile.normalized_log_rate.exp(), 0.0);
        assert!(profile.normalized_rate.is_none());
        let reported = profile
            .normalized_rate_error
            .expect("representable rate error");
        let ln_sigma = (0.5 / kbt).ln() + plane_error.ln() + 0.5 * 2.0_f64.ln();
        let expected = (profile.normalized_log_rate + ln_sigma).exp();
        assert_relative_close(expected, 5.26e-156, FAR_BELOW_RANGE_RATE_ERROR_TOLERANCE);
        assert_relative_close(reported, expected, NORMALIZED_RATE_ERROR_RELATIVE_TOLERANCE);
    }

    #[test]
    fn normalized_rate_error_matches_the_in_range_linear_product() {
        let arcs = [0.0, 0.5, 1.0, 1.5];
        let gradients = [0.3, 0.1, -0.4, -0.6];
        let errors = [0.02, 0.03, 0.01, 0.02];
        let kbt = 1.2;
        let profile = profile_from(&arcs, &gradients, &errors, kbt, 4.0).expect("profile");
        let rate = profile.normalized_rate.expect("finite rate");
        let reported = profile.normalized_rate_error.expect("finite rate error");
        let previous = previous_linear_rate_error(&arcs, &gradients, &errors, kbt, rate);
        assert_relative_close(reported, previous, NORMALIZED_RATE_ERROR_RELATIVE_TOLERANCE);
    }

    #[test]
    fn profile_error_stays_finite_for_an_extreme_finite_plane_uncertainty() {
        let sigma: f64 = 1.0e200;
        let linear_square = sigma * sigma;
        assert!(
            !linear_square.is_finite(),
            "this sigma must overflow the linear square the old norm formed"
        );
        let arcs = [0.0, 1.0];
        let gradients = [2.0, 2.0];
        let errors = [0.0, sigma];
        let kbt = 1.0;
        let profile = profile_from(&arcs, &gradients, &errors, kbt, 4.0).expect("profile");
        let expected_profile_error = 0.5 * sigma;
        assert!(expected_profile_error.is_finite());
        assert_relative_close(
            profile.standard_error[1],
            expected_profile_error,
            EXTREME_SIGMA_NORM_RELATIVE_TOLERANCE,
        );
        assert_eq!(profile.standard_error[0], 0.0);
        assert_eq!(
            profile.barrier_error.to_bits(),
            profile.standard_error[1].to_bits()
        );

        let quadrature = Quadrature::new(&arcs).expect("quadrature");
        let free_energy = quadrature.integrate(&gradients);
        let dividing = quadrature
            .dividing_plane(&gradients, &free_energy)
            .expect("dividing plane");
        let (_, d_ln_zr) = reactant_log_partition_and_sensitivity(
            &arcs,
            &free_energy,
            dividing.partition_end,
            kbt,
        )
        .expect("partition");
        let weight = 0.5;
        let sensitivity = -weight / kbt - d_ln_zr[dividing.partition_end] * weight;
        let expected_rate_error =
            (profile.normalized_log_rate + sensitivity.abs().ln() + sigma.ln()).exp();
        assert!(
            expected_rate_error.is_finite(),
            "normalized rate error {expected_rate_error} is outside the finite range"
        );
        let reported = profile
            .normalized_rate_error
            .expect("finite normalized rate error");
        assert_relative_close(
            reported,
            expected_rate_error,
            NORMALIZED_RATE_ERROR_RELATIVE_TOLERANCE,
        );
    }

    #[test]
    fn an_overflowing_free_energy_standard_error_is_rejected() {
        // The last plane carries weight `h / 2 = 2` in `F(s_1)`, so its
        // standard error overflows. The barrier at `s = 2` weights the same
        // plane by `0.375`, so every later error stays finite.
        let arcs = [0.0, 4.0];
        let gradients = [2.0, -2.0];
        let sigma = 1.0e308_f64;
        assert!((2.0 * sigma).is_infinite() && (0.375 * sigma).is_finite());
        assert_eq!(
            profile_from(&arcs, &gradients, &[0.0, sigma], 1.0, 4.0).err(),
            Some(FluxError::NonFinite("PAFI free-energy standard error"))
        );
    }

    #[test]
    fn non_finite_reactant_partition_exponent_is_rejected() {
        let arcs = [0.0, 1.0];
        let gradients = [1.0, 1.0];
        let thermal = f64::MIN_POSITIVE * 0.25;
        assert!(thermal.is_finite() && thermal > 0.0);
        assert!(!(1.0 / thermal).is_finite());
        let quadrature = Quadrature::new(&arcs).expect("quadrature");
        let free_energy = quadrature.integrate(&gradients);
        assert!(free_energy.iter().all(|value| value.is_finite()));
        let result = profile_from(&arcs, &gradients, &[0.0, 0.0], thermal, 4.0);
        assert_eq!(
            result,
            Err(FluxError::NonFinite("PAFI reactant partition exponent"))
        );
    }

    #[test]
    fn default_tolerance_scales_with_planes_for_consistency_check() {
        let planes = 33usize;
        let scaled = default_zero_temperature_tol(planes);
        let old_fixed = 0.01_f64;
        assert!(
            scaled < old_fixed,
            "scaled tolerance {scaled} is not stricter than the old fixed default {old_fixed}"
        );
        let relative_error = 0.5 * (scaled + old_fixed);
        let surface = InconsistentLinearGradient {
            shortfall: relative_error,
        };
        let path = Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).expect("shape");
        let mut engine = NativeEngine::new(
            surface.clone(),
            NativeConfig::default()
                .with_masses(Array1::from_vec(vec![1.0]))
                .with_thermostat(ThermostatConfig::default().with_temperature(0.2))
                .with_seed(23),
        )
        .expect("engine");
        let config = PafiConfig::default()
            .with_planes(planes)
            .with_burn_in_steps(1)
            .with_sample_steps(2);
        let result = pafi_profile(&mut engine, path.view(), &config);
        assert_eq!(
            result,
            Err(FluxError::InvalidConfig(
                "PAFI zero-temperature path-force quadrature exceeds tolerance"
            ))
        );

        let mut fixed_engine = NativeEngine::new(
            surface,
            NativeConfig::default()
                .with_masses(Array1::from_vec(vec![1.0]))
                .with_thermostat(ThermostatConfig::default().with_temperature(0.2))
                .with_seed(23),
        )
        .expect("engine");
        let fixed_default = PafiConfig::default()
            .with_planes(planes)
            .with_zero_temperature_tol(old_fixed)
            .with_burn_in_steps(1)
            .with_sample_steps(2)
            .with_stats(StatsConfig::default().with_min_batches(2));
        assert!(
            pafi_profile(&mut fixed_engine, path.view(), &fixed_default).is_ok(),
            "the old fixed default {old_fixed} should accept relative error {relative_error}"
        );
    }

    #[test]
    fn documented_nine_plane_tolerance_matches_the_effective_default() {
        let reference = include_str!("../docs/orgmode/reference.org");
        let row = reference
            .lines()
            .find(|line| line.contains("| =zero_temperature_tol= |"))
            .expect("zero_temperature_tol row");
        let marker = "\\approx ";
        let rest = row.split_once(marker).expect("approximate value").1;
        let numeric: String = rest
            .chars()
            .take_while(|ch| ch.is_ascii_digit() || *ch == '.')
            .collect();
        let documented: f64 = numeric.parse().expect("documented tolerance");
        let decimals = numeric.split('.').nth(1).map(str::len).unwrap_or(0);
        let unit = 10_f64.powi(-(decimals as i32));
        let effective = effective_zero_temperature_tol(&PafiConfig::default());
        let difference = (effective - documented).abs();
        assert!(
            difference <= 0.5 * unit,
            "effective {effective} documented {documented} difference {difference}"
        );
    }

    #[test]
    fn explicit_tolerance_equal_to_the_nine_plane_value_stays_fixed() {
        let planes = 33usize;
        let nine_plane = default_zero_temperature_tol(DEFAULT_PLANES);
        let explicit = PafiConfig::default()
            .with_planes(planes)
            .with_zero_temperature_tol(nine_plane);
        assert_eq!(explicit.zero_temperature_tol, Some(nine_plane));
        assert_eq!(
            effective_zero_temperature_tol(&explicit).to_bits(),
            nine_plane.to_bits()
        );
        assert_ne!(
            nine_plane.to_bits(),
            default_zero_temperature_tol(planes).to_bits()
        );

        let unset = PafiConfig::default().with_planes(planes);
        assert!(unset.zero_temperature_tol.is_none());
        assert_eq!(
            effective_zero_temperature_tol(&unset).to_bits(),
            default_zero_temperature_tol(planes).to_bits()
        );
    }

    #[test]
    fn explicit_nine_plane_tolerance_is_not_scaled_with_planes() {
        // A 2% path-force discrepancy sits under the 9-plane tolerance and
        // over the tolerance scaled to 33 planes. An explicit 9-plane value
        // must keep the looser check; leaving the tolerance unset must not.
        let path = Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).expect("shape");
        let nine_plane_tol = 2.0 * (std::f64::consts::PI.powi(2) / 12.0) / 64.0;
        let config = |tol: Option<f64>| {
            PafiConfig::default()
                .with_planes(33)
                .with_zero_temperature_tol(tol)
                .with_burn_in_steps(1)
                .with_sample_steps(2)
                .with_stats(StatsConfig::default().with_min_batches(2))
        };
        let run = |config: &PafiConfig| {
            let mut engine = NativeEngine::new(
                InconsistentLinearGradient { shortfall: 0.02 },
                NativeConfig::default()
                    .with_masses(Array1::from_vec(vec![1.0]))
                    .with_thermostat(ThermostatConfig::default().with_temperature(0.2))
                    .with_seed(23),
            )
            .expect("engine");
            pafi_profile(&mut engine, path.view(), config)
        };

        assert!(run(&config(Some(nine_plane_tol))).is_ok());
        assert_eq!(
            run(&config(None)),
            Err(FluxError::InvalidConfig(
                "PAFI zero-temperature path-force quadrature exceeds tolerance"
            ))
        );
    }

    #[test]
    fn dyn_entrypoint_restores_engine_after_error() {
        let mut engine = NativeEngine::new(
            Quartic,
            NativeConfig::default()
                .with_masses(Array1::from_vec(vec![2.0]))
                .with_thermostat(ThermostatConfig::default().with_temperature(1.0)),
        )
        .expect("engine");
        let original_positions = Array1::from_vec(vec![0.25]);
        let requested_velocities = Array1::from_vec(vec![-0.75]);
        let original_constraint =
            HyperplaneConstraint::from_direction(Array1::from_vec(vec![1.0]).view(), 0.25)
                .expect("constraint");
        engine
            .set_constraint(Some(original_constraint.clone()))
            .expect("constraint");
        engine
            .set_positions(original_positions.view())
            .expect("positions");
        engine
            .set_velocities(requested_velocities.view())
            .expect("velocities");
        let original_velocities = engine.velocities().to_owned();

        let path = Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).expect("shape");
        let planes = 3usize;
        let spacing = 1.0 / ((planes - 1) as f64);
        // For `V(x) = x^4` the path force `g(x) = V'(x) = 4 x^3` is a cubic, so
        // the trapezoid error is exactly the leading Euler-Maclaurin term
        // `spacing^2 / 12 * (g'(1) - g'(0))`, with `g'(x) = 12 x^2`. A tolerance
        // of half that error makes the check reject this path.
        let trapezoid_error =
            spacing * spacing / 12.0 * (quartic_gradient_slope(1.0) - quartic_gradient_slope(0.0));
        let config = PafiConfig::default()
            .with_planes(planes)
            .with_burn_in_steps(1)
            .with_sample_steps(2)
            .with_zero_temperature_tol(trapezoid_error / 2.0);
        let result = pafi_profile_dyn(&mut engine, path.view(), &config);
        assert_eq!(
            result,
            Err(FluxError::InvalidConfig(
                "PAFI zero-temperature path-force quadrature exceeds tolerance"
            ))
        );
        assert_eq!(engine.positions(), original_positions.view());
        assert_eq!(engine.velocities(), original_velocities.view());
        assert_eq!(engine.constraint(), Some(&original_constraint));
    }

    #[test]
    fn reactant_partition_exponential_overflow_keeps_the_logarithm() {
        // Linear `g(s) = a + b s` is integrated exactly. `F(1) = -1e6` and
        // `F(2) = 1`, so the dividing sample is the last plane and the shifted
        // trapezoid is finite while `exp(ln Z_R)` is not.
        let profile = profile_from(
            &[0.0, 1.0, 2.0],
            &[-2_000_000.5, 0.5, 2_000_001.5],
            &[0.0, 0.0, 0.0],
            1.0,
            1.0,
        )
        .expect("finite logarithms");
        assert!(profile.reactant_log_partition.is_finite());
        assert!(profile.reactant_partition.is_none());
    }

    #[test]
    fn lagrange_weights_stay_finite_on_a_wide_grid() {
        // Four node gaps of this size have a product `24 * spacing^4 = 16 * MAX`,
        // so the running product overflows, while `spacing^2` stays finite and
        // the derivative ratio does not.
        let spacing = (f64::MAX / 24.0).sqrt().sqrt() * 2.0;
        let unit: Vec<f64> = (0..DERIVATIVE_WINDOW)
            .map(|node| f64::from(node as u32))
            .collect();
        let wide: Vec<f64> = unit.iter().map(|node| node * spacing).collect();
        let gradients = [1.0, 2.0, 3.0, 2.0, 1.0];
        let unit_integral = Quadrature::new(&unit)
            .expect("unit quadrature")
            .integrate(&gradients);
        let wide_integral = Quadrature::new(&wide)
            .expect("wide quadrature")
            .integrate(&gradients);
        // Each weight sums about two logs per other node, and the integral
        // adds a window of weights. This many ulps covers that chain and is
        // far tighter than an overflowing product.
        let lagrange_log_weight_tol =
            (DERIVATIVE_WINDOW * DERIVATIVE_WINDOW) as f64 * 64.0 * f64::EPSILON;
        for (unit_value, wide_value) in unit_integral.iter().zip(wide_integral) {
            let scaled = unit_value * spacing;
            let scale = scaled.abs().max(1.0);
            assert!(
                wide_value.is_finite()
                    && (wide_value - scaled).abs() <= lagrange_log_weight_tol * scale,
                "wide integral {wide_value} versus scaled unit integral {scaled}"
            );
        }
    }

    #[test]
    fn quadrature_weights_do_not_square_a_huge_spacing() {
        // `spacing^2` exceeds `f64::MAX` while the derivative weights scale as
        // `1 / spacing`, so every product `spacing * weight` is of order one.
        let spacing = 1.0e160_f64;
        assert!((spacing * spacing).is_infinite());
        let unit: Vec<f64> = (0..5).map(f64::from).collect();
        let wide: Vec<f64> = unit.iter().map(|node| node * spacing).collect();
        let gradients = [1.0, 2.0, 3.0, 2.0, 1.0];
        let unit_integral = Quadrature::new(&unit)
            .expect("unit quadrature")
            .integrate(&gradients);
        let wide_integral = Quadrature::new(&wide)
            .expect("wide quadrature")
            .integrate(&gradients);
        let lagrange_log_weight_tol =
            (DERIVATIVE_WINDOW * DERIVATIVE_WINDOW) as f64 * 64.0 * f64::EPSILON;
        for (unit_value, wide_value) in unit_integral.iter().zip(wide_integral) {
            let scaled = unit_value * spacing;
            assert!(
                wide_value.is_finite()
                    && (wide_value - scaled).abs()
                        <= lagrange_log_weight_tol * scaled.abs().max(spacing),
                "wide integral {wide_value} versus scaled unit integral {scaled}"
            );
        }
    }

    #[test]
    fn dividing_plane_on_a_huge_spacing_matches_the_scaled_unit_grid() {
        let spacing = 0.5e160_f64;
        let peak = 1.7_f64;
        let unit: Vec<f64> = (0..8).map(|k| f64::from(k) * 0.5).collect();
        let wide: Vec<f64> = unit.iter().map(|node| node * (spacing / 0.5)).collect();
        let gradients: Vec<f64> = unit.iter().map(|s| -(s - peak)).collect();
        let quadrature = Quadrature::new(&wide).expect("quadrature");
        let free_energy = quadrature.integrate(&gradients);
        let plane = quadrature
            .dividing_plane(&gradients, &free_energy)
            .expect("dividing plane");
        let scale = spacing / 0.5;
        assert!(
            (plane.arc - peak * scale).abs() <= 1.0e-9 * scale,
            "dividing arc {} versus {}",
            plane.arc,
            peak * scale
        );
        let barrier = peak * peak / 2.0 * scale;
        assert!(
            (plane.free_energy - barrier).abs() <= 1.0e-9 * scale,
            "barrier {} versus {barrier}",
            plane.free_energy
        );
    }

    /// Mean force `E (0.34 - x)(1 + 3x) / L` on four planes at `x L`, with
    /// `x = 0, 1/3, 2/3, 1`. The Hermite rule is exact for this quadratic, so
    /// the dividing plane is `x = 0.34` and the barrier is `0.077452 E`.
    fn scaled_quadratic_profile(length: f64, energy: f64) -> FreeEnergyProfile {
        let unit = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0];
        let arcs: Vec<f64> = unit.iter().map(|x| x * length).collect();
        let gradients: Vec<f64> = unit
            .iter()
            .map(|x| energy * (0.34 - x) * 3.0_f64.mul_add(*x, 1.0) / length)
            .collect();
        profile_from(&arcs, &gradients, &[0.0; 4], energy, 1.0).expect("profile")
    }

    #[test]
    fn interpolant_slopes_do_not_underflow_on_a_huge_spacing() {
        // The derivative weights are about `1 / L` and the means about
        // `E / L`, so their product is near `1e-325` and flushes to zero
        // unless the weight is scaled by `h` first.
        let (length, energy) = (1.0e155_f64, 1.0e-15_f64);
        assert_eq!((energy / length) * (1.0 / length), 0.0);
        let profile = scaled_quadratic_profile(length, energy);
        assert_abs_diff_eq!(profile.dividing_arc / length, 0.34, epsilon = 1.0e-12);
        assert_abs_diff_eq!(profile.barrier / energy, 0.077_452, epsilon = 1.0e-12);
    }

    #[test]
    fn interpolant_slopes_do_not_overflow_on_a_compact_grid() {
        // Here the same product is near `1e309` and overflows unless the
        // weight is scaled by `h` first.
        let (length, energy) = (3.0e-150_f64, 1.0e10_f64);
        assert!(((energy / length) * (1.0 / length)).is_infinite());
        let profile = scaled_quadratic_profile(length, energy);
        assert_abs_diff_eq!(profile.dividing_arc / length, 0.34, epsilon = 1.0e-12);
        assert_abs_diff_eq!(profile.barrier / energy, 0.077_452, epsilon = 1.0e-12);
    }

    #[test]
    fn barrier_is_not_below_the_highest_plane_on_a_mixed_scale_grid() {
        // Gaps of about 3e-6, 1e-16 and 2e-5. Re-evaluating the interpolant
        // at the right end of a segment lands a few ulps below the integrated
        // plane value there.
        let arcs = [
            0.0,
            3.328_878_653_758_490_4e-6,
            3.328_878_653_861_533_3e-6,
            1.922_735_806_438_615_6e-5,
        ];
        let gradients = [
            -1.390_499_491_346_321e-4,
            -5_202.970_370_744_257,
            -12.102_897_034_268_363,
            -9.627_726_547_885_606,
        ];
        let profile = profile_from(&arcs, &gradients, &[0.0; 4], 1.0, 1.0).expect("profile");
        let highest = profile.free_energy[profile.dividing_index];
        assert!(
            profile.barrier >= highest,
            "barrier {:e} is below the highest plane {highest:e}",
            profile.barrier
        );
    }

    #[test]
    fn dividing_plane_rejects_a_non_finite_slope_or_value() {
        let quadrature = Quadrature::new(&[0.0, 1.0, 2.0, 3.0]).expect("quadrature");
        let alternating = [f64::MAX, -f64::MAX, f64::MAX, -f64::MAX];
        assert_eq!(
            quadrature.dividing_plane(&alternating, &[0.0; 4]).err(),
            Some(FluxError::NonFinite(
                "PAFI dividing-plane interpolant slope"
            ))
        );
        // The slopes are finite, but an interior root adds about `1e299` to a
        // plane value at `f64::MAX`.
        let gradients = [1.0e300, -1.0e300, 1.0e300, -1.0e300];
        assert_eq!(
            quadrature.dividing_plane(&gradients, &[f64::MAX; 4]).err(),
            Some(FluxError::NonFinite(
                "PAFI dividing-plane interpolant value"
            ))
        );
    }

    #[test]
    fn reactant_sensitivity_keeps_a_ratio_when_kbt_times_the_partition_underflows() {
        // The scaled half partition is `1e-200`, so multiplying it by `kbt`
        // underflows to zero. Each node holds half of it, so
        // `d ln Z_R / d F_i` is `-0.5 / kbt`.
        let kbt = 1.0e-200_f64;
        let arcs = [0.0, 1.0e-200];
        assert_eq!(kbt * 1.0e-200, 0.0);
        let (_, sensitivity) =
            reactant_log_partition_and_sensitivity(&arcs, &[0.0, 0.0], 1, kbt).expect("partition");
        for value in sensitivity {
            assert!(
                (value - (-0.5 / kbt)).abs() <= 4.0 * f64::EPSILON * (0.5 / kbt),
                "sensitivity {value} versus {}",
                -0.5 / kbt
            );
        }
        let profile = profile_from(&arcs, &[1.0, 1.0], &[0.0, 0.0], kbt, 4.0).expect("profile");
        assert!(profile.reactant_log_partition.is_finite());
        assert!(profile.normalized_log_rate.is_finite());
    }

    struct BondQuartic;

    impl rgsaddle::PointSurface for BondQuartic {
        fn eval(
            &self,
            x: ArrayView1<'_, f64>,
        ) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
            const QUARTIC: f64 = 1.0;
            let gap = x[1] - x[0] - 1.0;
            let slope = gap + 4.0 * QUARTIC * gap.powi(3);
            let energy = 0.5 * gap * gap + QUARTIC * gap.powi(4);
            Ok((energy, Array1::from_vec(vec![-slope, slope])))
        }
    }

    #[test]
    fn translated_bond_hessian_matches_the_unshifted_hessian() {
        const TRANSLATION: f64 = 10_000.0;
        // Central-difference truncation of the quartic is `4 * h^2`. A step
        // proportional to `|x|` at this translation sits far above a few
        // displacement fractions. The displacement-relative step does not.
        let hessian_translation_tol = 64.0 * hessian_displacement_fraction();
        let mut engine = NativeEngine::new(
            BondQuartic,
            NativeConfig::default()
                .with_masses(Array1::from_vec(vec![1.0, 1.0]))
                .with_thermostat(ThermostatConfig::default().with_temperature(0.2)),
        )
        .expect("engine");
        let root = engine.masses().mapv(f64::sqrt);
        let origin = Array1::from_vec(vec![0.0, 1.0]);
        let shifted = Array1::from_vec(vec![TRANSLATION, TRANSLATION + 1.0]);
        ensure_reactant_minimum(&mut engine, origin.view()).expect("origin minimum");
        ensure_reactant_minimum(&mut engine, shifted.view()).expect("shifted minimum");
        let reference =
            mass_weighted_hessian(&mut engine, origin.view(), root.view()).expect("hessian");
        let translated = mass_weighted_hessian(&mut engine, shifted.view(), root.view())
            .expect("shifted hessian");
        for (left, right) in reference.iter().zip(translated.iter()) {
            assert!(
                (left - right).abs() <= hessian_translation_tol,
                "hessian entries {left} and {right} differ by more than {hessian_translation_tol}"
            );
        }
    }

    struct CoupledCubic;

    impl rgsaddle::PointSurface for CoupledCubic {
        fn eval(
            &self,
            r: ArrayView1<'_, f64>,
        ) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
            // `V = x^2 / 2 + y^2 / 2 + 0.4 x y^3`. The mixed derivative of the
            // gradient differs by `0.4 h^2` between the two difference orders.
            let (x, y) = (r[0], r[1]);
            let energy = 0.5 * x * x + 0.5 * y * y + 0.4 * x * y.powi(3);
            Ok((
                energy,
                Array1::from_vec(vec![x + 0.4 * y.powi(3), y + 1.2 * x * y * y]),
            ))
        }
    }

    #[test]
    fn mass_weighted_hessian_is_symmetric_for_unequal_masses() {
        let masses = [1.0, 4.0];
        let mut engine = NativeEngine::new(
            CoupledCubic,
            NativeConfig::default()
                .with_masses(Array1::from_vec(masses.to_vec()))
                .with_thermostat(ThermostatConfig::default().with_temperature(0.2)),
        )
        .expect("engine");
        let root = engine.masses().mapv(f64::sqrt);
        let (x, y) = (0.3_f64, 0.7_f64);
        let origin = Array1::from_vec(vec![x, y]);
        let hessian =
            mass_weighted_hessian(&mut engine, origin.view(), root.view()).expect("hessian");
        assert_eq!(
            hessian[(0, 1)].to_bits(),
            hessian[(1, 0)].to_bits(),
            "off-diagonal entries {} and {}",
            hessian[(0, 1)],
            hessian[(1, 0)]
        );
        let exact = [
            [
                1.0 / masses[0],
                1.2 * y * y / (masses[0] * masses[1]).sqrt(),
            ],
            [
                1.2 * y * y / (masses[0] * masses[1]).sqrt(),
                (1.0 + 2.4 * x * y) / masses[1],
            ],
        ];
        // Symmetrizing halves the `0.4 h^2 / sqrt(m_x m_y)` truncation of the
        // one asymmetric entry. Round-off in the force differences is the
        // remaining error.
        let hessian_tol = 4.0 * hessian_displacement_fraction().powi(2);
        for (row, exact_row) in exact.iter().enumerate() {
            for (column, expected) in exact_row.iter().enumerate() {
                assert!(
                    (hessian[(row, column)] - expected).abs() <= hessian_tol,
                    "entry ({row}, {column}) is {} against {expected}",
                    hessian[(row, column)]
                );
            }
        }
    }

    #[test]
    fn projected_curvature_agrees_with_the_symmetrized_hessian() {
        // The conjugate-gradient solve applies unsymmetrized directional
        // products. Each differs from the symmetric Hessian product by the
        // same `O(h^2)` truncation that symmetrizing the dense matrix halves,
        // so both curvatures agree with the analytic one to that order.
        let masses = [1.0, 4.0];
        let mut engine = NativeEngine::new(
            CoupledCubic,
            NativeConfig::default()
                .with_masses(Array1::from_vec(masses.to_vec()))
                .with_thermostat(ThermostatConfig::default().with_temperature(0.2)),
        )
        .expect("engine");
        let root = engine.masses().mapv(f64::sqrt);
        let (x, y) = (0.3_f64, 0.7_f64);
        let origin = Array1::from_vec(vec![x, y]);
        let tangent = Array1::from_vec(vec![0.6, 0.8]);
        let geometry = PlaneGeometry {
            anchor: origin.clone(),
            normal: tangent.clone(),
            dn_ds: Array1::zeros(2),
            dc_ds: 0.0,
            tangent: tangent.clone(),
            arc: 0.0,
        };
        let projected = reactant_curvature(&mut engine, &geometry, None).expect("curvature");

        let normal_norm = (root[0] * tangent[0]).hypot(root[1] * tangent[1]);
        let n = [
            root[0] * tangent[0] / normal_norm,
            root[1] * tangent[1] / normal_norm,
        ];
        let inverse_curvature = |h: [[f64; 2]; 2]| {
            let determinant = h[0][0] * h[1][1] - h[0][1] * h[1][0];
            (h[1][1] * n[0] * n[0] - (h[0][1] + h[1][0]) * n[0] * n[1] + h[0][0] * n[1] * n[1])
                / determinant
        };
        let dense =
            mass_weighted_hessian(&mut engine, origin.view(), root.view()).expect("hessian");
        let symmetrized = 1.0
            / inverse_curvature([
                [dense[(0, 0)], dense[(0, 1)]],
                [dense[(1, 0)], dense[(1, 1)]],
            ]);
        let coupling = 1.2 * y * y / (masses[0] * masses[1]).sqrt();
        let analytic = 1.0
            / inverse_curvature([
                [1.0 / masses[0], coupling],
                [coupling, (1.0 + 2.4 * x * y) / masses[1]],
            ]);
        let curvature_tol = 16.0 * hessian_displacement_fraction().powi(2);
        assert!(
            (projected - symmetrized).abs() <= curvature_tol,
            "projected {projected} against symmetrized {symmetrized}"
        );
        assert!(
            (projected - analytic).abs() <= curvature_tol,
            "projected {projected} against analytic {analytic}"
        );
    }

    struct SaddleWell;

    impl rgsaddle::PointSurface for SaddleWell {
        fn eval(
            &self,
            r: ArrayView1<'_, f64>,
        ) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
            // Stationary at the origin, positive curvature along `y`, and a
            // negative mode along `x`.
            Ok((
                -0.5 * r[0] * r[0] + 2.0 * r[1] * r[1],
                Array1::from_vec(vec![-r[0], 4.0 * r[1]]),
            ))
        }
    }

    #[test]
    fn a_saddle_is_rejected_as_a_reactant_minimum() {
        let mut engine = NativeEngine::new(
            SaddleWell,
            NativeConfig::default()
                .with_masses(Array1::from_vec(vec![1.0, 1.0]))
                .with_thermostat(ThermostatConfig::default().with_temperature(0.2)),
        )
        .expect("engine");
        let path = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 0.0, 1.0]).expect("shape");
        let config = PafiConfig::default()
            .with_planes(2)
            .with_burn_in_steps(1)
            .with_sample_steps(2);
        let result = pafi_profile(&mut engine, path.view(), &config);
        assert_eq!(
            result,
            Err(FluxError::InvalidConfig(
                "PAFI reactant Hessian is not positive definite off the zero modes"
            ))
        );
    }

    struct AnisotropicWell;

    impl rgsaddle::PointSurface for AnisotropicWell {
        fn eval(
            &self,
            r: ArrayView1<'_, f64>,
        ) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
            Ok((
                2.0 * r[0] * r[0] + 0.5 * r[1] * r[1],
                Array1::from_vec(vec![4.0 * r[0], r[1]]),
            ))
        }
    }

    #[test]
    fn anisotropic_well_starting_off_stationary_point_is_accepted() {
        let mut engine = NativeEngine::new(
            AnisotropicWell,
            NativeConfig::default()
                .with_masses(Array1::from_vec(vec![1.0, 4.0]))
                .with_thermostat(ThermostatConfig::default().with_temperature(0.2)),
        )
        .expect("engine");
        let path = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 0.0]).expect("shape");
        let config = PafiConfig::default()
            .with_planes(2)
            .with_burn_in_steps(1)
            .with_sample_steps(2)
            .with_zero_temperature_tol(1.0);
        let profile = pafi_profile(&mut engine, path.view(), &config).expect("accepted");
        assert!(profile.reactant_frequency > 0.0);
    }

    #[test]
    fn projected_force_drops_a_rigid_translation() {
        let mut engine = NativeEngine::new(
            IsotropicPair { stiffness: 3.25 },
            NativeConfig::default().with_masses(Array1::ones(6)),
        )
        .expect("engine");
        let mut dn_ds = Array1::zeros(6);
        dn_ds[0] = 0.4;
        let mut tangent = Array1::zeros(6);
        tangent[0] = 1.0;
        let geometry = PlaneGeometry {
            anchor: Array1::zeros(6),
            normal: tangent.clone(),
            dn_ds,
            dc_ds: 1.0,
            tangent,
            arc: 0.0,
        };
        let mut local = Array1::zeros(6);
        local[0] = 0.3;
        local[4] = 0.05;
        let root_mass = engine.masses().mapv(f64::sqrt);
        let origin = geometry.anchor.clone();
        let modes = classified_translation_modes(&mut engine, &origin, &root_mass, None)
            .expect("translation modes");
        assert_eq!(modes.len(), 3);
        engine.set_positions(local.view()).expect("local");
        let local_force =
            projected_force_sample(&engine, &geometry, 1.0, &modes).expect("local force");
        let mut shifted = local.clone();
        for atom in 0..2 {
            shifted[3 * atom] += 4.0;
            shifted[3 * atom + 1] -= 2.5;
        }
        engine.set_positions(shifted.view()).expect("shifted");
        let shifted_force =
            projected_force_sample(&engine, &geometry, 1.0, &modes).expect("shifted force");
        let raw = |positions: &Array1<f64>| -> f64 {
            positions
                .iter()
                .zip(geometry.dn_ds.iter())
                .map(|(left, right)| left * right)
                .sum()
        };
        assert!(
            (raw(&local) - raw(&shifted)).abs() > 1.0,
            "the unprojected Jacobian must see the rigid shift"
        );
        assert!(
            (local_force - shifted_force).abs() <= 1.0e-12,
            "rigid translation changed the projected force: local {local_force} shifted {shifted_force}"
        );
        assert_abs_diff_eq!(local_force, 0.94, epsilon = 1.0e-12);
    }

    #[test]
    fn projected_force_keeps_a_tethered_shift() {
        let mut engine = NativeEngine::new(
            TetheredPair {
                first_stiffness: 1.0,
                second_stiffness: 3.0,
                coupling: 0.25,
            },
            NativeConfig::default().with_masses(Array1::ones(6)),
        )
        .expect("engine");
        let shift = 0.3_f64;
        let mut positions = Array1::zeros(6);
        for atom in 0..2 {
            positions[3 * atom] = shift;
        }
        let mut dn_ds = Array1::zeros(6);
        dn_ds[0] = 0.4;
        let mut tangent = Array1::zeros(6);
        tangent[0] = 1.0;
        let geometry = PlaneGeometry {
            anchor: Array1::zeros(6),
            normal: tangent.clone(),
            dn_ds,
            dc_ds: 1.0,
            tangent,
            arc: 0.0,
        };
        let root_mass = engine.masses().mapv(f64::sqrt);
        let origin = geometry.anchor.clone();
        let modes = classified_translation_modes(&mut engine, &origin, &root_mass, None)
            .expect("translation modes");
        assert!(
            modes.is_empty(),
            "tethers to fixed points are not zero modes"
        );
        engine.set_positions(positions.view()).expect("shift");
        let force = projected_force_sample(&engine, &geometry, 1.0, &modes).expect("force");
        let correct = 1.0 - shift * 0.4;
        assert_abs_diff_eq!(force, correct, epsilon = 1.0e-12);
        assert!((force - 1.0).abs() > 0.1);
        let declared = classified_translation_modes(
            &mut engine,
            &origin,
            &root_mass,
            Some([true, true, true]),
        )
        .expect("declared modes");
        assert_eq!(declared.len(), 3);
        let stripped =
            projected_force_sample(&engine, &geometry, 1.0, &declared).expect("declared force");
        assert_abs_diff_eq!(stripped, 1.0, epsilon = 1.0e-12);
        let curvature = reactant_curvature(&mut engine, &geometry, None).expect("soft tether");
        assert_abs_diff_eq!(curvature, 16.0 / 13.0, epsilon = 1.0e-6);
        assert!((curvature - 2.0).abs() > 0.4);
    }

    const RIGID_SHIFT: [f64; 3] = [4.0, -2.5, 0.0];

    struct CurvedPairSampler {
        positions: Array1<f64>,
        base: Array1<f64>,
        velocities: Array1<f64>,
        masses: Array1<f64>,
        thermostat: ThermostatConfig,
        constraint: Option<HyperplaneConstraint>,
        stiffness: f64,
    }

    impl CurvedPairSampler {
        fn new(stiffness: f64) -> Self {
            Self {
                positions: Array1::zeros(6),
                base: Array1::zeros(6),
                velocities: Array1::zeros(6),
                masses: Array1::ones(6),
                thermostat: ThermostatConfig::default().with_temperature(1.0),
                constraint: None,
                stiffness,
            }
        }
    }

    impl DynamicsEngine for CurvedPairSampler {
        fn dof(&self) -> usize {
            6
        }

        fn masses(&self) -> ArrayView1<'_, f64> {
            self.masses.view()
        }

        fn positions(&self) -> ArrayView1<'_, f64> {
            self.positions.view()
        }

        fn set_positions(&mut self, positions: ArrayView1<'_, f64>) -> Result<(), FluxError> {
            if positions.len() != 6 {
                return Err(FluxError::Shape {
                    expected: 6,
                    got: positions.len(),
                    context: "curved pair positions",
                });
            }
            self.base.assign(&positions);
            self.positions.assign(&positions);
            Ok(())
        }

        fn velocities(&self) -> ArrayView1<'_, f64> {
            self.velocities.view()
        }

        fn set_velocities(&mut self, velocities: ArrayView1<'_, f64>) -> Result<(), FluxError> {
            if velocities.len() != 6 {
                return Err(FluxError::Shape {
                    expected: 6,
                    got: velocities.len(),
                    context: "curved pair velocities",
                });
            }
            self.velocities.assign(&velocities);
            Ok(())
        }

        fn cell(&self) -> Option<&[f64; 9]> {
            None
        }

        fn set_cell(&mut self, _cell: Option<[f64; 9]>) -> Result<(), FluxError> {
            Ok(())
        }

        fn thermostat(&self) -> &ThermostatConfig {
            &self.thermostat
        }

        fn set_thermostat(&mut self, thermostat: ThermostatConfig) -> Result<(), FluxError> {
            self.thermostat = thermostat;
            Ok(())
        }

        fn constraint(&self) -> Option<&HyperplaneConstraint> {
            self.constraint.as_ref()
        }

        fn set_constraint(
            &mut self,
            constraint: Option<HyperplaneConstraint>,
        ) -> Result<(), FluxError> {
            self.constraint = constraint;
            Ok(())
        }

        fn energy_and_forces(&mut self) -> Result<(f64, Array1<f64>), FluxError> {
            let mut energy = 0.0;
            let mut force = Array1::zeros(6);
            for axis in 0..3 {
                let extension = self.positions[3 + axis] - self.positions[axis];
                energy += 0.5 * self.stiffness * extension * extension;
                force[axis] = self.stiffness * extension;
                force[3 + axis] = -self.stiffness * extension;
            }
            Ok((energy, force))
        }

        fn step(&mut self, _dt: f64) -> Result<StepReport, FluxError> {
            for atom in 0..2 {
                for (axis, shift) in RIGID_SHIFT.iter().enumerate() {
                    self.positions[3 * atom + axis] = self.base[3 * atom + axis] + shift;
                }
            }
            Ok(StepReport::new(0.0, 0.0, Some(1.0)))
        }

        fn reseed(&mut self, _seed: u64) -> Result<(), FluxError> {
            Ok(())
        }

        fn shoot_velocities(&mut self, _seed: u64) -> Result<(), FluxError> {
            Ok(())
        }

        fn reflect_velocity(&mut self, _normal: ArrayView1<'_, f64>) -> Result<f64, FluxError> {
            Ok(0.0)
        }

        fn apply_dividing_surface(
            &mut self,
            _normal: ArrayView1<'_, f64>,
            _policy: SurfaceVelocityPolicy,
        ) -> Result<f64, FluxError> {
            Ok(0.0)
        }
    }

    fn curved_six_dof_path() -> Array2<f64> {
        Array2::from_shape_vec(
            (3, 6),
            vec![
                -0.5, 0.0, 0.0, 0.5, 0.0, 0.0, -1.0, 0.15, 0.0, 1.0, 0.15, 0.0, -1.6, 0.0, 0.0,
                1.6, 0.5, 0.0,
            ],
        )
        .expect("path shape")
    }

    fn profile_config() -> PafiConfig {
        PafiConfig::default()
            .with_planes(3)
            .with_burn_in_steps(1)
            .with_sample_steps(2)
            .with_zero_temperature_tol(100.0)
            .with_stats(StatsConfig::default().with_min_batches(2))
    }

    fn assert_sampled_gradients_drop_the_rigid_shift(
        engine: &mut CurvedPairSampler,
        path: &Array2<f64>,
        arcs: &[f64],
        gradients: &[f64],
    ) {
        assert_eq!(engine.dof(), 6);
        let origin = path.row(0).to_owned();
        let root_mass = engine.masses().to_owned().mapv(f64::sqrt);
        let modes = classified_translation_modes(engine, &origin, &root_mass, None)
            .expect("translation modes");
        assert_eq!(modes.len(), 3, "a free pair has three translation modes");
        let masses = engine.masses().to_owned();
        let spline =
            validate_path_and_build_spline(path.view(), masses.view(), arcs.len()).expect("spline");
        let geometries =
            build_plane_geometries_at(&spline, masses.view(), arcs).expect("geometries");
        assert!(
            geometries
                .iter()
                .any(|geometry| nrm2(geometry.dn_ds.view()) > 1.0e-3),
            "the path has to bend so the Jacobian sees a rigid shift"
        );
        let mut saw_gap = false;
        for (geometry, gradient) in geometries.iter().zip(gradients) {
            let mut shifted = geometry.anchor.clone();
            for atom in 0..2 {
                for (axis, shift) in RIGID_SHIFT.iter().enumerate() {
                    shifted[3 * atom + axis] += shift;
                }
            }
            engine.set_positions(shifted.view()).expect("shift");
            let kept = projected_force_sample(engine, geometry, 1.0, &modes).expect("kept");
            let dropped = projected_force_sample(engine, geometry, 1.0, &[]).expect("dropped");
            if (kept - dropped).abs() > 0.05 {
                saw_gap = true;
            }
            assert!(
                (gradient - kept).abs() <= 1.0e-12,
                "plane gradient {gradient} left the translation in the Jacobian; projected {kept}, raw {dropped}"
            );
        }
        assert!(
            saw_gap,
            "dropping the translation modes did not move any plane"
        );
    }

    #[test]
    fn pafi_profile_wires_translation_modes_on_a_curved_pair() {
        let path = curved_six_dof_path();
        let mut engine = CurvedPairSampler::new(3.25);
        let profile = pafi_profile(&mut engine, path.view(), &profile_config()).expect("profile");
        assert_sampled_gradients_drop_the_rigid_shift(
            &mut engine,
            &path,
            &profile.arc_lengths,
            profile.gradients(),
        );
    }

    #[test]
    fn pafi_profile_adaptive_wires_translation_modes_on_a_curved_pair() {
        let path = curved_six_dof_path();
        let mut engine = CurvedPairSampler::new(3.25);
        let config = AdaptivePafiConfig::default()
            .with_base(profile_config())
            .with_max_planes(3)
            .with_max_refinements(1)
            .with_monitor(MonitorKind::ForceGradient);
        let adaptive =
            pafi_profile_adaptive(&mut engine, path.view(), &config).expect("adaptive profile");
        assert_sampled_gradients_drop_the_rigid_shift(
            &mut engine,
            &path,
            &adaptive.profile.arc_lengths,
            adaptive.profile.gradients(),
        );
    }
}
