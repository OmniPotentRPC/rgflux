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
//! sampled plane. The barrier is `Delta F = F(s*) - F(s_0)`.
//! The reported error at each plane, and at the dividing arc, is the
//! batch-means error combined with the absolute difference between that
//! Hermite integral and the trapezoid rule.
//!
//! The normalized rate uses the Gaussian reactant integral. With the first
//! image at the reactant minimum, `F(s_0) = 0`, and `omega_0` the projected
//! frequency of the mass-weighted reactant Hessian `H` along the unit
//! mass-weighted path normal `n`, `omega_0^2 = 1 / (n^T H^-1 n)`,
//!
//! `Z_R = sqrt(2 pi k_B T) / omega_0`,
//! `k = (omega_0 / 2 pi) exp(-Delta F / k_B T)`.
//!
//! `Z_R` and `k` are also returned as [`FreeEnergyProfile::reactant_log_partition`]
//! and [`FreeEnergyProfile::normalized_log_rate`], so a large barrier does not
//! underflow the reported logarithm. In the harmonic limit this is the
//! Vineyard rate: the transverse entropy sits in `Delta F`, and `omega_0` is
//! the curvature of the potential along the path.
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
//!    A mismatch returns
//!    [`FluxError::InvalidConfig`] before any sampling.
//! 4. On each plane the engine is constrained, placed at the anchor, given
//!    Maxwell-Boltzmann velocities, run for `burn_in_steps`, and then run for
//!    `sample_steps` steps that each record
//!    `lambda (n . X' - (x - X) . n')`. Batch means over the samples give the
//!    mean force and its standard error on each plane.
//! 5. The profile, barrier and rate follow from the plane means. The
//!    standard errors of the planes are independent and propagate through the
//!    Hermite weights, then combine with the absolute difference between the
//!    Hermite integral and the trapezoid rule on the same nodes. The reactant
//!    Hessian is deterministic, so the rate error comes from the barrier
//!    alone, by the delta method.
//! 6. With `target_error` set, steps 4 and 5 repeat for up to `max_rounds`
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
//! - The first image is the reactant minimum. The rate normalization uses the
//!   reactant Hessian there. A non-positive curvature along the path normal
//!   makes `Z_R` undefined and the call fails.
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

use std::f64::consts::{PI, TAU};

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
    /// of the plane forces are propagated through the Hermite weights and
    /// combined, by `hypot`, with the absolute difference between the Hermite
    /// integral and the trapezoid rule on the same nodes. The first entry is
    /// `0`. A zero entry means those two contributions are both zero.
    pub standard_error: Vec<f64>,
    /// Free-energy barrier `Delta F = F(s*) - F(s_0)` at the dividing plane,
    /// the global maximum of the interpolated profile.
    pub barrier: f64,
    /// Standard error of [`Self::barrier`]. The propagated plane-force error
    /// at `s*` is combined, by `hypot`, with the absolute difference between
    /// the Hermite barrier and the trapezoid integral at the same arc.
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
    /// Whether the full mass-weighted reactant Hessian was certified positive
    /// definite off its zero modes by the dense eigensolver. This is `false`
    /// when the coordinate count exceeds 128 and dense certification is
    /// skipped; finite forces and positive projected curvature are still
    /// required.
    pub reactant_hessian_certified: bool,
    /// Standard error of [`Self::reactant_frequency`].
    ///
    /// Single profiles use zero because their reactant Hessian is deterministic.
    /// Pooled profiles estimate this value between replicates.
    pub reactant_frequency_error: f64,
    /// Gaussian reactant normalization
    /// `Z_R = sqrt(2 pi k_B T) / omega_0`, in units of `s`.
    ///
    /// `Some` when [`Self::reactant_log_partition`] has a finite positive
    /// exponential. `None` when that finite log overflows or underflows.
    pub reactant_partition: Option<f64>,
    /// Natural logarithm of the reactant normalization,
    /// `ln Z_R = ln(sqrt(2 pi k_B T) / omega_0)`.
    pub reactant_log_partition: f64,
    /// Normalized PAFI rate
    /// `k = sqrt(k_B T / (2 pi)) * exp(-Delta F / k_B T) / Z_R
    ///    = (omega_0 / 2 pi) * exp(-Delta F / k_B T)`.
    ///
    /// `Some` when [`Self::normalized_log_rate`] has a finite positive
    /// exponential. `None` when that finite log overflows or underflows.
    pub normalized_rate: Option<f64>,
    /// Natural logarithm of [`Self::normalized_rate`],
    /// `ln k = ln(omega_0 / 2 pi) - Delta F / k_B T`.
    /// This stays finite when [`Self::normalized_rate`] underflows.
    pub normalized_log_rate: f64,
    /// Propagated standard error of the linear normalized rate. The reactant
    /// Hessian is deterministic, so only [`Self::barrier`] contributes.
    ///
    /// Holds `exp(normalized_log_rate + ln(sigma))` with
    /// `sigma = barrier_error / k_B T`. `Some(0.0)` when `sigma == 0`.
    /// Otherwise `Some` when that exponential is finite and positive.
    /// `None` when `sigma != 0` and the exponential overflows to infinity,
    /// underflows to zero, or is not finite (for example non-finite `sigma`).
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
fn hessian_displacement_fraction() -> f64 {
    f64::EPSILON.cbrt()
}

/// Relative displacement for a Richardson-extrapolated centered derivative.
///
/// Balancing its `O(h^4)` truncation error against `O(epsilon / h)` roundoff
/// gives `h = epsilon^(1/5)`. The displacement is relative to the direction
/// norm and deliberately independent of the absolute coordinates.
fn richardson_displacement_fraction() -> f64 {
    f64::EPSILON.powf(1.0 / 5.0)
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

/// Dense reactant-Hessian certification is skipped above this dimension.
const REACTANT_HESSIAN_EIGEN_LIMIT: usize = 128;

/// Samples of the interpolant taken inside one plane interval before the
/// sign change of the force is refined.
const MAXIMUM_SCAN_POINTS: usize = 128;

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
}

fn lagrange_derivative_weights(nodes: &[f64], at: usize) -> Vec<f64> {
    let mut weights = vec![0.0_f64; nodes.len()];
    for j in 0..nodes.len() {
        if j == at {
            weights[j] = (0..nodes.len())
                .filter(|&m| m != at)
                .map(|m| 1.0 / (nodes[at] - nodes[m]))
                .sum();
        } else {
            let mut numerator = 1.0_f64;
            let mut denominator = 1.0_f64;
            for m in 0..nodes.len() {
                if m != j {
                    denominator *= nodes[j] - nodes[m];
                    if m != at {
                        numerator *= nodes[at] - nodes[m];
                    }
                }
            }
            weights[j] = numerator / denominator;
        }
    }
    weights
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
            let local = lagrange_derivative_weights(&arcs[start..start + window], i - start);
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
                *weight += h * h / 12.0 * (derivative[k][j] - derivative[k + 1][j]);
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

    fn slopes(&self, gradients: &[f64]) -> Vec<f64> {
        self.derivative
            .iter()
            .map(|row| {
                row.iter()
                    .zip(gradients)
                    .map(|(weight, gradient)| weight * gradient)
                    .sum()
            })
            .collect()
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
                * h
                * (left_slope * self.derivative[k][j] + right_slope * self.derivative[k + 1][j]);
        }
        weights
    }

    fn interpolated_value(
        &self,
        k: usize,
        u: f64,
        gradients: &[f64],
        slopes: &[f64],
        free_energy: &[f64],
    ) -> f64 {
        let h = self.arcs[k + 1] - self.arcs[k];
        let u2 = u * u;
        let u3 = u2 * u;
        let u4 = u2 * u2;
        free_energy[k]
            + h * (gradients[k] * (u - u3 + 0.5 * u4) + gradients[k + 1] * (u3 - 0.5 * u4))
            + h * h
                * (slopes[k] * (0.5 * u2 - 2.0 * u3 / 3.0 + 0.25 * u4)
                    + slopes[k + 1] * (0.25 * u4 - u3 / 3.0))
    }

    fn interpolated_force(&self, k: usize, u: f64, gradients: &[f64], slopes: &[f64]) -> f64 {
        let h = self.arcs[k + 1] - self.arcs[k];
        let u2 = u * u;
        let u3 = u2 * u;
        gradients[k] * (2.0 * u3 - 3.0 * u2 + 1.0)
            + gradients[k + 1] * (-2.0 * u3 + 3.0 * u2)
            + h * (slopes[k] * (u3 - 2.0 * u2 + u) + slopes[k + 1] * (u3 - u2))
    }

    /// Global maximum of the interpolated profile.
    ///
    /// Every interval is scanned. The highest sampled plane is recorded
    /// separately, because an overshoot can put `s*` in an interval that
    /// does not touch that plane.
    fn dividing_plane(&self, gradients: &[f64], free_energy: &[f64]) -> DividingPlane {
        let n = self.len();
        let mut index = 0usize;
        for i in 1..n {
            if free_energy[i] > free_energy[index] {
                index = i;
            }
        }
        let slopes = self.slopes(gradients);
        let mut best_interval = 0usize;
        let mut best_u = 0.0_f64;
        let mut best_value = free_energy[0];
        for k in 0..n - 1 {
            let mut scan_best = 0usize;
            let mut scan_value = self.interpolated_value(k, 0.0, gradients, &slopes, free_energy);
            for j in 1..=MAXIMUM_SCAN_POINTS {
                let u = j as f64 / MAXIMUM_SCAN_POINTS as f64;
                let value = self.interpolated_value(k, u, gradients, &slopes, free_energy);
                if value > scan_value {
                    scan_value = value;
                    scan_best = j;
                }
            }
            let mut u = scan_best as f64 / MAXIMUM_SCAN_POINTS as f64;
            if scan_best > 0 && scan_best < MAXIMUM_SCAN_POINTS {
                let mut lo = (scan_best - 1) as f64 / MAXIMUM_SCAN_POINTS as f64;
                let mut hi = (scan_best + 1) as f64 / MAXIMUM_SCAN_POINTS as f64;
                let force_lo = self.interpolated_force(k, lo, gradients, &slopes);
                let force_hi = self.interpolated_force(k, hi, gradients, &slopes);
                if force_lo > 0.0 && force_hi < 0.0 {
                    for _ in 0..80 {
                        let mid = 0.5 * (lo + hi);
                        if self.interpolated_force(k, mid, gradients, &slopes) > 0.0 {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }
                    u = 0.5 * (lo + hi);
                }
            }
            let value = self.interpolated_value(k, u, gradients, &slopes, free_energy);
            if value > best_value {
                best_interval = k;
                best_u = u;
                best_value = value;
            }
        }
        let width = self.arcs[best_interval + 1] - self.arcs[best_interval];
        DividingPlane {
            index,
            arc: self.arcs[best_interval] + best_u * width,
            free_energy: best_value,
            weights: self.interval_weights(best_interval, best_u),
        }
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
    let estimate = HermiteIntegral.cumulative(&quadrature.arcs, gradients)?;
    let free_energy = estimate.values;
    if free_energy.iter().any(|value| !value.is_finite())
        || estimate.error.iter().any(|value| !value.is_finite())
    {
        return Err(FluxError::NonFinite("PAFI integrated free energy"));
    }
    let dividing = quadrature.dividing_plane(gradients, &free_energy);
    let propagate = |weights: &[f64]| linear_euclidean_norm(weights, gradient_errors);
    let standard_error: Vec<f64> = quadrature
        .cumulative
        .iter()
        .zip(&estimate.error)
        .map(|(weights, truncation)| propagate(weights).hypot(*truncation))
        .collect();
    let barrier = dividing.free_energy;
    let lower_barrier = trapezoid_to(
        &quadrature.arcs,
        gradients,
        dividing.arc,
        "PAFI trapezoid barrier",
    )?;
    let barrier_error = propagate(&dividing.weights).hypot((barrier - lower_barrier).abs());
    if !barrier.is_finite() || !barrier_error.is_finite() || barrier_error < 0.0 {
        return Err(FluxError::NonFinite("PAFI barrier"));
    }

    let reactant_frequency = reactant_curvature.sqrt();
    if !reactant_frequency.is_finite() || reactant_frequency <= 0.0 {
        return Err(FluxError::InvalidConfig(
            "PAFI reactant frequency must be finite and positive",
        ));
    }
    // F(s_0) = 0 and a Gaussian reactant well give
    // Z_R = sqrt(2 pi kT) / omega_0, so
    // sqrt(kT / 2 pi) exp(-F* / kT) / Z_R = (omega_0 / 2 pi) exp(-F* / kT).
    let reactant_log_partition = 0.5 * (TAU.ln() + kbt.ln()) - reactant_frequency.ln();
    if !reactant_log_partition.is_finite() {
        return Err(FluxError::NonFinite("PAFI reactant log partition"));
    }
    let reactant_partition = finite_exponential(reactant_log_partition);
    let normalized_log_rate = reactant_frequency.ln() - TAU.ln() - barrier / kbt;
    if !normalized_log_rate.is_finite() {
        return Err(FluxError::InvalidConfig(
            "PAFI normalized log rate must be finite",
        ));
    }
    let normalized_rate = finite_exponential(normalized_log_rate);
    let normalized_rate_error = linear_rate_error(normalized_log_rate, barrier_error, kbt);

    Ok(FreeEnergyProfile {
        arc_lengths: quadrature.arcs.clone(),
        free_energy,
        standard_error,
        barrier,
        barrier_error,
        dividing_index: dividing.index,
        dividing_arc: dividing.arc,
        reactant_frequency,
        reactant_hessian_certified,
        reactant_frequency_error: 0.0,
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

/// Exponential of a finite logarithm when the result is a finite positive `f64`.
///
/// Overflow to positive infinity and underflow to zero are `None`.
fn finite_exponential(log_value: f64) -> Option<f64> {
    let linear = log_value.exp();
    if linear.is_finite() && linear > 0.0 {
        Some(linear)
    } else {
        None
    }
}

/// `sqrt(sum (weight_i * sigma_i)^2)` without forming the squares.
///
/// A finite factor above about `1e154` makes `sigma * sigma` overflow. The
/// scaled norm keeps a finite result whenever the mathematical norm fits in a
/// `f64`.
fn linear_euclidean_norm(weights: &[f64], sigmas: &[f64]) -> f64 {
    let mut scale = 0.0_f64;
    let mut sum = 0.0_f64;
    for (&weight, &sigma) in weights.iter().zip(sigmas) {
        let term = weight.abs() * sigma;
        if !term.is_finite() {
            return f64::INFINITY;
        }
        if term > scale {
            sum = 1.0 + sum * (scale / term).powi(2);
            scale = term;
        } else if scale > 0.0 {
            sum += (term / scale).powi(2);
        }
    }
    scale * sum.sqrt()
}

/// Linear standard error `exp(log_rate) * error / scale`.
///
/// The ratio is formed as `ln(error) - ln(scale)` so an intermediate linear
/// division cannot overflow or underflow. `Some(0.0)` when `error == 0`, even
/// when `exp(log_rate)` underflows. Otherwise `Some` when the final
/// exponential is finite and positive. `None` when that exponential
/// overflows, underflows, or is not finite.
fn linear_rate_error(log_rate: f64, error: f64, scale: f64) -> Option<f64> {
    if error == 0.0 {
        return Some(0.0);
    }
    finite_exponential(log_rate + error.ln() - scale.ln())
}

/// Mass-weighted Hessian-vector product `M^{-1/2} (d grad V) M^{-1/2} direction`.
fn mass_weighted_hessian_product<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    origin: ArrayView1<'_, f64>,
    root_mass: ArrayView1<'_, f64>,
    direction: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, FluxError> {
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
    let scale = hessian_displacement_fraction() / length;
    let mut gradients = Vec::with_capacity(2);
    for sign in [1.0_f64, -1.0] {
        let probe = &origin.to_owned() + &displacement.mapv(|value| sign * scale * value);
        engine.set_positions(probe.view())?;
        let (_, forces) = engine.energy_and_forces()?;
        gradients.push(forces.mapv(|force| -force));
    }
    let mut product = Array1::zeros(dof);
    for i in 0..dof {
        product[i] = (gradients[0][i] - gradients[1][i]) / (2.0 * scale) / root_mass[i];
    }
    if product.iter().any(|value| !value.is_finite()) {
        return Err(FluxError::NonFinite("PAFI reactant Hessian product"));
    }
    Ok(product)
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

#[derive(Debug)]
struct HessianProductEstimate {
    value: Array1<f64>,
    roundoff_floor: f64,
}

fn finite_difference_hessian_product<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    origin: &Array1<f64>,
    root_mass: &Array1<f64>,
    direction: &Array1<f64>,
    step: f64,
    hessian_frobenius: f64,
) -> Result<HessianProductEstimate, FluxError> {
    let dof = direction.len();
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
    let scale = step / length;
    let mut gradients = Vec::with_capacity(2);
    let mut position_error_norms = Vec::with_capacity(2);
    let unit_roundoff = 0.5 * f64::EPSILON;
    for sign in [1.0_f64, -1.0] {
        let delta = displacement.mapv(|value| sign * scale * value);
        let probe = origin + &delta;
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

fn estimate_hessian_product<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    origin: &Array1<f64>,
    root_mass: &Array1<f64>,
    direction: &Array1<f64>,
    step: f64,
    hessian_frobenius: f64,
) -> Result<(Array1<f64>, f64, f64), FluxError> {
    let coarse = finite_difference_hessian_product(
        engine,
        origin,
        root_mass,
        direction,
        step,
        hessian_frobenius,
    )?;
    let fine = finite_difference_hessian_product(
        engine,
        origin,
        root_mass,
        direction,
        0.5 * step,
        hessian_frobenius,
    )?;
    // A centered derivative has O(h^2) truncation error. The fine-grid error
    // estimate is therefore (fine - coarse) / (2^2 - 1), and adding it gives
    // the Richardson-extrapolated Hessian product.
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
    if !roundoff_floor.is_finite() {
        return Ok(false);
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

fn translation_probe_step(origin: &Array1<f64>) -> f64 {
    let reach = origin
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    // Central difference of a gradient: the step that balances truncation
    // and roundoff is the cube root of machine epsilon, scaled by the
    // coordinate magnitude.
    f64::EPSILON.cbrt() * (1.0 + reach)
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
    step: f64,
    translation_invariance: Option<[bool; 3]>,
) -> Result<Vec<Array1<f64>>, FluxError> {
    let dof = origin.len();
    let hessian_frobenius = if dof >= 6 && dof.is_multiple_of(3) && translation_invariance.is_none()
    {
        let mut squared_bound = 0.0;
        for i in 0..dof {
            let mut basis = Array1::zeros(dof);
            basis[i] = 1.0;
            let (column, uncertainty, roundoff_floor) =
                estimate_hessian_product(engine, origin, root_mass, &basis, step, 0.0)?;
            let column_bound = nrm2(column.view()) + uncertainty + roundoff_floor;
            squared_bound += column_bound * column_bound;
        }
        squared_bound.sqrt()
    } else {
        0.0
    };

    let mut translation_modes = Vec::new();
    if dof >= 6 && dof.is_multiple_of(3) {
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
                    origin,
                    root_mass,
                    &translation,
                    step,
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
    let step = translation_probe_step(&origin);
    let translation_modes =
        classified_translation_modes(engine, &origin, &root_mass, step, translation_invariance)?;
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
    // The CG Hessian product must be invariant under a rigid coordinate
    // translation. `finite_difference_hessian_product` divides this fixed
    // displacement by the mass-weighted direction norm, so the probe scale is
    // `h / |v|` and never depends on `|origin|`.
    let cg_step = richardson_displacement_fraction();
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
        let (mut image, _, _) =
            estimate_hessian_product(engine, &origin, &root_mass, &search, cg_step, 0.0)?;
        remove_translations(&mut image);
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

fn path_force_quadrature_residual(
    arcs: &[f64],
    energies: &[f64],
    gradients: &[f64],
) -> Result<f64, FluxError> {
    let integrated_energy = Quadrature::new(arcs)?.integrate(gradients);
    let mut residual = 0.0_f64;
    for (integral, energy) in integrated_energy.iter().zip(energies.iter()) {
        let discrepancy = (integral - (energy - energies[0])).abs();
        if !discrepancy.is_finite() {
            return Err(FluxError::NonFinite("PAFI path-force quadrature residual"));
        }
        residual = residual.max(discrepancy);
    }
    Ok(residual)
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
/// - the zero-temperature check of [`PafiConfig::zero_temperature_tol`] fails,
/// - a plane normal degenerates, or the reactant curvature along the first
///   path normal is not finite and positive, so `Z_R` is not defined.
///
/// Returns [`FluxError::EmptyInput`] for an empty coordinate axis and
/// [`FluxError::Shape`] when the engine masses do not match the path
/// coordinates. [`FluxError::NonFinite`] reports non-finite path coordinates,
/// constraint forces, force samples, profile values or plane means. Errors
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

    let snapshot = EngineSnapshot::capture(engine);
    let result = (|| {
        let kbt = engine.thermostat().thermal_energy()?;
        if kbt <= 0.0 {
            return Err(FluxError::InvalidConfig(
                "PAFI requires positive thermal energy",
            ));
        }
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
        let quadrature_residual =
            path_force_quadrature_residual(&arcs, &path_energies, &path_gradients)?;
        let reactant_hessian_certified =
            ensure_reactant_minimum(engine, geometries[0].anchor.view())?;
        let root_mass = engine.masses().mapv(f64::sqrt);
        let origin = geometries[0].anchor.clone();
        let translation_modes = classified_translation_modes(
            engine,
            &origin,
            &root_mass,
            translation_probe_step(&origin),
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

    let trap_barrier = trapezoid_cumulative(
        arc_lengths,
        gradients,
        "PAFI spectral-tail trapezoid integral",
    )?
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

fn respects_minimum_spacing(
    records: &[SampledPlaneRecord],
    candidate: f64,
    min_spacing: f64,
) -> bool {
    records.iter().all(|record| {
        let existing = record.geometry.arc;
        let tolerance =
            64.0 * f64::EPSILON * (1.0 + existing.abs() + candidate.abs() + min_spacing);
        (existing - candidate).abs() + tolerance >= min_spacing
    })
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

fn adaptive_spectral_stop_threshold(barrier_error: f64, barrier: f64) -> f64 {
    if barrier_error > 0.0 {
        barrier_error
    } else {
        64.0 * f64::EPSILON * (1.0 + barrier.abs())
    }
}

/// Integrate the PAFI free-energy profile with monitor equidistribution and
/// iterative spectral-tail plane refinement.
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
        let kbt = engine.thermostat().thermal_energy()?;
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
        let quadrature_residual =
            path_force_quadrature_residual(&ref_arcs, &ref_energies, &ref_gradients)?;
        let reactant_hessian_certified =
            ensure_reactant_minimum(engine, ref_geometries[0].anchor.view())?;
        let root_mass = engine.masses().mapv(f64::sqrt);
        let origin = ref_geometries[0].anchor.clone();
        let translation_modes = classified_translation_modes(
            engine,
            &origin,
            &root_mass,
            translation_probe_step(&origin),
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
                let mut profile = integrate_profile(
                    &quadrature,
                    &gradients,
                    &errors,
                    kbt,
                    reactant_curvature,
                    reactant_hessian_certified,
                )?;
                profile.quadrature_residual = quadrature_residual;
                let need_more_stat_rounds = match config.base.target_error {
                    Some(target) => {
                        profile.barrier_error > target
                            && records
                                .iter()
                                .any(|r| r.rounds_completed < config.base.max_rounds)
                    }
                    None => false,
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

                // Plane resolution is judged against the measured statistical
                // error. `target_error` controls additional sampling rounds;
                // it must not independently tighten spectral refinement.
                let stop_threshold =
                    adaptive_spectral_stop_threshold(profile.barrier_error, profile.barrier);

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
                        profile,
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
                        profile,
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
                    if respects_minimum_spacing(&records, s_new, min_gap) {
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
                        profile,
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
/// integration, interior maximum search, harmonic reactant normalization, and
/// log-space rate calculation as [`pafi_profile`].
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
    let reactant_hessian_certified = replicates
        .iter()
        .all(|replicate| replicate.reactant_hessian_certified);
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

    #[derive(Clone)]
    struct InconsistentLinearGradient {
        gradient: f64,
        slope: f64,
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

    struct SexticIsotropicPair {
        stiffness: f64,
        sextic: f64,
    }

    impl rgsaddle::PointSurface for SexticIsotropicPair {
        fn eval(
            &self,
            x: ArrayView1<'_, f64>,
        ) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
            let mut gradient = Array1::zeros(6);
            let mut energy = 0.0;
            for axis in 0..3 {
                let extension = x[3 + axis] - x[axis];
                let slope = self.stiffness * extension + 6.0 * self.sextic * extension.powi(5);
                energy +=
                    0.5 * self.stiffness * extension * extension + self.sextic * extension.powi(6);
                gradient[axis] = -slope;
                gradient[3 + axis] = slope;
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

    struct ShiftedTetheredPair {
        inner: AnisotropicTetheredPair,
        offset: f64,
    }

    impl rgsaddle::PointSurface for ShiftedTetheredPair {
        fn eval(
            &self,
            x: ArrayView1<'_, f64>,
        ) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
            let shifted = x.mapv(|value| value - self.offset);
            self.inner.eval(shifted.view())
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

    impl rgsaddle::PointSurface for InconsistentLinearGradient {
        fn eval(
            &self,
            x: ArrayView1<'_, f64>,
        ) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
            let gradient = self.gradient + self.slope * (x[0] - 0.5);
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
    fn richardson_cg_curvature_is_invariant_under_a_hundred_angstrom_shift() {
        let stiffness = 50.0;
        let mut tangent = Array1::from_vec(vec![0.3, 0.2, -0.1, 0.9, 0.2, 0.4]);
        let tangent_norm = nrm2(tangent.view());
        tangent.mapv_inplace(|value| value / tangent_norm);
        let mut reference_curvature = None;
        for shift in [0.0, 100.0] {
            let geometry = PlaneGeometry {
                anchor: Array1::from_elem(6, shift),
                normal: tangent.clone(),
                dn_ds: Array1::zeros(6),
                dc_ds: 1.0,
                tangent: tangent.clone(),
                arc: 0.0,
            };
            let mut engine = NativeEngine::new(
                SexticIsotropicPair {
                    stiffness,
                    sextic: 1.0e6,
                },
                NativeConfig::default().with_masses(Array1::ones(6)),
            )
            .expect("sextic pair engine");
            let curvature =
                reactant_curvature(&mut engine, &geometry, None).expect("curvature and modes");
            if let Some(expected_curvature) = reference_curvature {
                assert_abs_diff_eq!(curvature, expected_curvature, epsilon = 1.0e-8);
            } else {
                assert_abs_diff_eq!(curvature, 2.0 * stiffness, epsilon = 1.0e-6);
                reference_curvature = Some(curvature);
            }
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
    fn weak_tether_classification_is_invariant_under_rigid_offsets() {
        let tether = 1.0e-8;
        let coupling = 3.25;
        let first = 0.8;
        let second = 0.6;
        let determinant = tether * (tether + 2.0 * coupling);
        let inverse_numerator = (tether + coupling) * (first * first + second * second)
            + 2.0 * coupling * first * second;
        let expected = determinant / inverse_numerator;

        for offset in [0.0, 1.0e2, 1.0e4] {
            let tangent = Array1::from_vec(vec![first, 0.0, 0.0, second, 0.0, 0.0]);
            let geometry = PlaneGeometry {
                anchor: Array1::from_elem(6, offset),
                normal: tangent.clone(),
                dn_ds: Array1::zeros(6),
                dc_ds: 1.0,
                tangent,
                arc: 0.0,
            };
            let mut engine = NativeEngine::new(
                ShiftedTetheredPair {
                    inner: AnisotropicTetheredPair {
                        tether: [tether, 0.0, 0.0],
                        coupling: [coupling, coupling, coupling],
                    },
                    offset,
                },
                NativeConfig::default().with_masses(Array1::ones(6)),
            )
            .expect("shifted weakly tethered pair");
            let curvature = reactant_curvature(
                &mut engine,
                &geometry,
                PafiConfig::default().translation_invariance,
            )
            .expect("mixed tangent must retain the weak tether");
            assert_abs_diff_eq!(curvature, expected, epsilon = 1.0e-3 * tether);
        }
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
                &origin,
                &root_mass,
                &direction,
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
            &origin,
            &root_mass,
            &direction,
            step,
            0.0,
        )
        .expect("gradient-only floor");
        let with_position = finite_difference_hessian_product(
            &mut engine,
            &origin,
            &root_mass,
            &direction,
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
    fn non_finite_roundoff_floor_does_not_certify_translation_invariance() {
        let invariant = classify_translation_mode(0, 1.0, 0.0, f64::INFINITY, None)
            .expect("a finite nonzero response must fail closed");
        assert!(!invariant);
    }

    #[test]
    fn adaptive_insertions_respect_the_certified_minimum_spacing() {
        let map = EquidistributionMap::new(&[0.0, 1.0], &[0.0, 0.0]).expect("uniform map");
        let min_spacing = map.min_spacing_bound(3).expect("spacing certificate");
        assert_abs_diff_eq!(min_spacing, 0.5, epsilon = f64::EPSILON);
        let record = |arc| SampledPlaneRecord {
            geometry: PlaneGeometry {
                anchor: Array1::zeros(1),
                normal: Array1::ones(1),
                dn_ds: Array1::zeros(1),
                dc_ds: 1.0,
                tangent: Array1::ones(1),
                arc,
            },
            samples: Vec::new(),
            uid: 0,
            rounds_completed: 1,
        };
        let mut records = vec![record(0.0), record(1.0)];
        assert!(!respects_minimum_spacing(&records, 0.25, min_spacing));
        assert!(respects_minimum_spacing(&records, 0.5, min_spacing));
        records.push(record(0.5));
        records.sort_by(|a, b| a.geometry.arc.total_cmp(&b.geometry.arc));
        for pair in records.windows(2) {
            assert!(
                pair[1].geometry.arc - pair[0].geometry.arc >= min_spacing,
                "placed gap is below its certificate"
            );
        }
    }

    #[test]
    fn target_error_does_not_tighten_the_spectral_stop_threshold() {
        let barrier_error = 2.5e-3;
        let barrier = 1.25;
        let threshold = adaptive_spectral_stop_threshold(barrier_error, barrier);
        assert_abs_diff_eq!(threshold, barrier_error, epsilon = f64::EPSILON);
    }

    #[test]
    fn zero_barrier_error_uses_a_roundoff_scaled_spectral_threshold() {
        let barrier = 1.25;
        let threshold = adaptive_spectral_stop_threshold(0.0, barrier);
        let expected = 64.0 * f64::EPSILON * (1.0 + barrier);
        assert_abs_diff_eq!(threshold, expected, epsilon = f64::EPSILON * expected);
        assert!(threshold < 0.05);
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
            &origin,
            &root_mass,
            &direction,
            step,
            0.0,
        )
        .expect("gradient-only floor");
        let combined = finite_difference_hessian_product(
            &mut engine,
            &origin,
            &root_mass,
            &direction,
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
        let (_, _, roundoff_floor) =
            estimate_hessian_product(&mut engine, &origin, &root_mass, &direction, step, 0.0)
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
            &origin,
            &root_mass,
            &direction,
            0.5 * step,
            0.0,
        )
        .expect("fine Hessian product");
        assert_abs_diff_eq!(
            fine.value[0],
            harmonic + quartic * step * step / 4.0,
            epsilon = 1.0e-12
        );
        let (corrected, _, _) =
            estimate_hessian_product(&mut engine, &origin, &root_mass, &direction, step, 0.0)
                .expect("Richardson Hessian product");
        assert_abs_diff_eq!(corrected[0], harmonic, epsilon = 1.0e-12);
    }

    #[test]
    fn reactant_curvature_uses_richardson_product_in_cg_solve() {
        let harmonic = 2.75;
        let quartic = 8.0;
        let anchor = 0.7;
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
        assert_abs_diff_eq!(
            profile
                .normalized_rate_error
                .expect("representable rate error"),
            profile.normalized_rate.expect("representable rate") * profile.barrier_error,
            epsilon = 1.0e-12
        );
        let frequency = 4.0_f64.sqrt();
        let log_partition = 0.5 * (TAU * 1.0).ln() - frequency.ln();
        assert_abs_diff_eq!(
            profile.reactant_log_partition,
            log_partition,
            epsilon = 1.0e-12
        );
        assert_abs_diff_eq!(
            profile.normalized_log_rate,
            frequency.ln() - TAU.ln() - profile.barrier,
            epsilon = 1.0e-12
        );
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
        replicates[1].reactant_hessian_certified = false;

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
        let expected_profile = profile_from(
            &arcs,
            &expected_gradients,
            &[expected_gradient_se; 3],
            kbt,
            curvature,
        )
        .expect("analytic pooled profile");
        let expected_rate = expected_profile
            .normalized_rate
            .expect("representable analytic pooled rate");
        let expected_frequency_se = (1.0_f64 / 3.0).sqrt();
        let expected_rate_se = expected_profile
            .normalized_rate_error
            .expect("representable analytic pooled rate error");
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
            epsilon = 1.0e-15
        );
        assert_abs_diff_eq!(
            pooled
                .normalized_rate_error
                .expect("representable pooled rate error"),
            expected_rate_se,
            epsilon = 1.0e-15
        );
        assert!(
            !pooled.reactant_hessian_certified,
            "one uncertified replicate must make the pooled Hessian uncertified"
        );

        let arithmetic_rate = replicates
            .iter()
            .map(|profile| {
                profile
                    .normalized_rate
                    .expect("representable replicate rate")
            })
            .sum::<f64>()
            / 3.0;
        let geometric_rate = (replicates
            .iter()
            .map(|profile| profile.normalized_log_rate)
            .sum::<f64>()
            / 3.0)
            .exp();
        assert!((arithmetic_rate - expected_rate).abs() > 1.0e-3);
        assert!((geometric_rate - expected_rate).abs() > 1.0e-3);
        assert!(
            (replicates[0]
                .normalized_rate
                .expect("representable first rate")
                - expected_rate)
                .abs()
                > 1.0e-3
        );
    }

    struct HighDimensionalHarmonic;

    impl rgsaddle::PointSurface for HighDimensionalHarmonic {
        fn eval(
            &self,
            positions: ArrayView1<'_, f64>,
        ) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
            Ok((0.5 * dot(positions, positions), positions.to_owned()))
        }
    }

    #[test]
    fn adaptive_profile_preserves_an_uncertified_large_reactant_hessian() {
        let dof = REACTANT_HESSIAN_EIGEN_LIMIT + 1;
        let mut engine = NativeEngine::new(
            HighDimensionalHarmonic,
            NativeConfig::default()
                .with_masses(Array1::ones(dof))
                .with_thermostat(ThermostatConfig::default().with_temperature(0.2)),
        )
        .expect("large harmonic engine");
        let mut path = Array2::zeros((2, dof));
        path[(1, 0)] = 0.1;
        let base = PafiConfig::default()
            .with_planes(3)
            .with_burn_in_steps(1)
            .with_sample_steps(2)
            .with_zero_temperature_tol(1.0)
            .with_stats(StatsConfig::default().with_min_batches(2));
        let adaptive = pafi_profile_adaptive(
            &mut engine,
            path.view(),
            &AdaptivePafiConfig::default()
                .with_base(base)
                .with_max_planes(3)
                .with_max_refinements(1),
        )
        .expect("large adaptive profile");
        assert!(
            !adaptive.profile.reactant_hessian_certified,
            "the adaptive driver must not claim dense certification above its dimension limit"
        );
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
        let frequency = curvature.sqrt();
        let log_partition = 0.5 * (TAU).ln() - frequency.ln();
        let log_rate = frequency.ln() - TAU.ln() - profile.barrier;
        assert!(profile.barrier > 700.0);
        assert!(profile.normalized_rate.is_none());
        assert_abs_diff_eq!(
            profile.reactant_log_partition,
            log_partition,
            epsilon = f64::EPSILON * log_partition.abs()
        );
        assert_abs_diff_eq!(profile.normalized_log_rate, log_rate, epsilon = 1.0e-9);
        assert!(
            profile
                .reactant_partition
                .is_some_and(|partition| partition.is_finite() && partition > 0.0)
        );
    }

    #[test]
    fn normalized_rate_error_is_none_when_the_linear_value_underflows() {
        let arcs = [0.0, 1.0, 2.0, 3.0];
        let gradients = [800.0, 800.0, 800.0, 800.0];
        let errors = [0.01, 0.01, 0.01, 0.01];
        let profile = profile_from(&arcs, &gradients, &errors, 1.0, 4.0).expect("profile");
        assert!(profile.barrier > 700.0);
        assert!(profile.normalized_rate.is_none());
        assert!(profile.normalized_rate_error.is_none());
        assert!(profile.barrier_error > 0.0);
    }

    #[test]
    fn log_fields_match_the_projected_frequency_formula() {
        let arcs = [0.0, 0.5, 1.0, 1.5];
        let gradients = [0.3, 0.1, -0.4, -0.6];
        let errors = [0.02, 0.03, 0.01, 0.02];
        let kbt = 1.2;
        let curvature = 5.0;
        let profile = profile_from(&arcs, &gradients, &errors, kbt, curvature).expect("profile");
        let frequency = curvature.sqrt();
        let partition = (TAU * kbt).sqrt() / frequency;
        let rate = frequency / TAU * (-profile.barrier / kbt).exp();
        assert_abs_diff_eq!(
            profile.reactant_partition.expect("representable partition"),
            partition,
            epsilon = f64::EPSILON * partition.abs()
        );
        assert_abs_diff_eq!(
            profile.reactant_log_partition,
            partition.ln(),
            epsilon = 16.0 * f64::EPSILON * partition.ln().abs().max(1.0)
        );
        assert_abs_diff_eq!(
            profile.normalized_rate.expect("representable rate"),
            rate,
            epsilon = f64::EPSILON * rate.abs()
        );
        assert_abs_diff_eq!(
            profile.normalized_log_rate,
            rate.ln(),
            epsilon = 16.0 * f64::EPSILON * rate.ln().abs().max(1.0)
        );
        let flux = (kbt / TAU).sqrt() * (-profile.barrier / kbt).exp()
            / profile.reactant_partition.expect("representable partition");
        assert_abs_diff_eq!(
            profile.normalized_rate.expect("representable rate"),
            flux,
            epsilon = f64::EPSILON * rate.abs()
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

    #[test]
    fn non_finite_profile_input_is_rejected() {
        let arcs = [0.0, 1.0, 2.0];
        let gradients = [0.0, f64::NAN, 1.0];
        let errors = [0.0, 0.0, 0.0];
        let result = profile_from(&arcs, &gradients, &errors, 1.0, 1.0);
        assert_eq!(result, Err(FluxError::NonFinite("PAFI Hermite mean force")));
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
    fn finite_exponential_rejects_linear_overflow_and_underflow() {
        assert_eq!(finite_exponential(f64::MAX.ln() + 1.0), None);
        assert_eq!(finite_exponential(f64::from_bits(1).ln() - 1.0), None);
    }

    #[test]
    fn linear_rate_error_rejects_underflow_after_error_scaling() {
        assert_eq!(linear_rate_error(-2401.0, 2400.0, 1.0), None);
    }

    #[test]
    fn linear_rate_error_handles_extreme_ratios_in_log_space() {
        let overflowed_ratio = f64::MAX / f64::MIN_POSITIVE;
        assert!(overflowed_ratio.is_infinite());
        let overflow_result = linear_rate_error(-740.0, f64::MAX, f64::MIN_POSITIVE)
            .expect("log-space ratio remains representable");
        assert!(overflow_result.is_finite() && overflow_result > 0.0);

        let underflowed_ratio = f64::MIN_POSITIVE / f64::MAX;
        assert_eq!(underflowed_ratio, 0.0);
        let underflow_result = linear_rate_error(740.0, f64::MIN_POSITIVE, f64::MAX)
            .expect("log-space ratio remains representable");
        assert!(underflow_result.is_finite() && underflow_result > 0.0);

        assert_eq!(
            linear_rate_error(-1000.0, 0.0, f64::MIN_POSITIVE),
            Some(0.0)
        );
    }

    #[test]
    fn linear_rate_error_matches_linear_ratio_in_normal_range() {
        let log_rate = -3.25_f64;
        let barrier_error = 0.75_f64;
        let kbt = 1.5_f64;
        let expected = finite_exponential(log_rate + (barrier_error / kbt).ln());
        let actual = linear_rate_error(log_rate, barrier_error, kbt);
        assert_abs_diff_eq!(
            actual.expect("representable scaled error"),
            expected.expect("representable linear-ratio error"),
            epsilon = f64::EPSILON
        );
    }

    #[test]
    fn profile_error_stays_finite_for_an_extreme_plane_uncertainty() {
        let sigma = 1.0e200_f64;
        assert!(!(sigma * sigma).is_finite());
        let profile =
            profile_from(&[0.0, 1.0], &[2.0, 2.0], &[0.0, sigma], 1.0, 4.0).expect("profile");
        let expected = 0.5 * sigma;
        assert_abs_diff_eq!(
            profile.standard_error[1],
            expected,
            epsilon = f64::EPSILON * expected
        );
        assert!(profile.normalized_rate_error.is_some());
    }

    #[test]
    fn a_descending_profile_keeps_the_gaussian_partition() {
        let arcs = [0.0, 1.0, 2.0];
        let gradients = [-1.0, -1.0, -1.0];
        let errors = [0.0, 0.0, 0.0];
        let curvature = 2.0_f64;
        let profile = profile_from(&arcs, &gradients, &errors, 1.0, curvature).expect("profile");
        assert_eq!(profile.dividing_index, 0);
        assert_eq!(profile.barrier, 0.0);
        let partition = (TAU / curvature).sqrt();
        assert_abs_diff_eq!(
            profile.reactant_partition.expect("representable partition"),
            partition,
            epsilon = f64::EPSILON * partition
        );
        assert!(profile.normalized_log_rate.is_finite());
    }

    #[test]
    fn reactant_normalization_accepts_maximum_thermal_energy() {
        let curvature = 2.0_f64;
        let profile = profile_from(
            &[0.0, 1.0, 2.0],
            &[-1.0, -1.0, -1.0],
            &[0.0, 0.0, 0.0],
            f64::MAX,
            curvature,
        )
        .expect("profile at maximum thermal energy");
        let expected_log_partition = 0.5 * (TAU.ln() + f64::MAX.ln()) - curvature.sqrt().ln();
        assert_abs_diff_eq!(
            profile.reactant_log_partition,
            expected_log_partition,
            epsilon = f64::EPSILON * expected_log_partition.abs()
        );
        assert!(
            profile
                .reactant_partition
                .is_some_and(|partition| partition.is_finite() && partition > 0.0)
        );
    }

    #[test]
    fn reactant_normalization_accepts_subnormal_thermal_energy() {
        let kbt = f64::from_bits(1);
        assert!(kbt.is_subnormal());
        let profile = profile_from(
            &[0.0, 1.0, 2.0],
            &[-1.0, -1.0, -1.0],
            &[0.0, 0.0, 0.0],
            kbt,
            2.0,
        )
        .expect("profile at subnormal thermal energy");
        assert!(profile.reactant_log_partition.is_finite());
        assert!(
            profile
                .reactant_partition
                .is_some_and(|partition| partition.is_finite() && partition > 0.0)
        );
    }

    #[test]
    fn default_tolerance_scales_with_planes_for_consistency_check() {
        // A 2% path-force error is already above the old fixed default of 0.01,
        // so that profile does not tell scaling apart from the fixed default.
        // The error here is halfway between 0.01 and the scaled tolerance.
        let planes = 33usize;
        let scaled = default_zero_temperature_tol(planes);
        let old_fixed = 0.01_f64;
        assert!(
            scaled < old_fixed,
            "scaled tolerance {scaled} is not stricter than the old fixed default {old_fixed}"
        );
        let relative_error = 0.5 * (scaled + old_fixed);
        let surface = InconsistentLinearGradient {
            gradient: 1.0 - relative_error,
            slope: relative_error,
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
        let modes = classified_translation_modes(
            &mut engine,
            &origin,
            &root_mass,
            translation_probe_step(&origin),
            None,
        )
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
        let step = translation_probe_step(&origin);
        let modes = classified_translation_modes(&mut engine, &origin, &root_mass, step, None)
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
            step,
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
        step_calls: usize,
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
                step_calls: 0,
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
            self.step_calls += 1;
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
        let modes = classified_translation_modes(
            engine,
            &origin,
            &root_mass,
            translation_probe_step(&origin),
            None,
        )
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
        let base = profile_config().with_target_error(0.0).with_max_rounds(2);
        let config = AdaptivePafiConfig::default()
            .with_base(base)
            .with_max_planes(3)
            .with_max_refinements(1)
            .with_monitor(MonitorKind::ForceGradient);
        let adaptive =
            pafi_profile_adaptive(&mut engine, path.view(), &config).expect("adaptive profile");
        assert_eq!(
            engine.step_calls, 18,
            "three planes must each complete burn-in plus sampling in two statistics rounds"
        );
        assert_sampled_gradients_drop_the_rigid_shift(
            &mut engine,
            &path,
            &adaptive.profile.arc_lengths,
            adaptive.profile.gradients(),
        );
    }
}
