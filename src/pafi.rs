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
//! on a curved path gives a wrong profile.
//!
//! The plane means `g(s) = dF/ds` are integrated with a cubic Hermite rule.
//! With two planes the derivative correction cancels and the rule is the
//! trapezoid rule, exact for linear `g`. With three planes the derivative
//! window is quadratic, so the rule is exact for quadratic `g`. With four or
//! more planes the window has at least four nodes, the rule is exact for
//! cubic `g`, and the global error is `O(h^4)` for smoother `g`. The dividing
//! plane `s*` is the global maximum of that interpolant, not the highest
//! sampled plane: on each segment the derivative is a cubic, solved in closed
//! form, and the barrier is the largest value at those roots and the
//! endpoints. The barrier is `Delta F = F(s*) - F(s_0)`.
//!
//! The reactant partition is the doubled trapezoid of the Boltzmann weight up
//! to the last sample of the segment that holds `s*`,
//!
//! `Z_R = 2 * \int_0^{s_div} exp(-F(s) / k_B T) ds`,
//! `k = sqrt(k_B T / (2 pi)) exp(-Delta F / k_B T) / Z_R`.
//!
//! The integral is formed in log space. `Z_R` and `k` are also returned as
//! [`FreeEnergyProfile::reactant_log_partition`] and
//! [`FreeEnergyProfile::normalized_log_rate`], so a large barrier does not
//! underflow the reported logarithm. The value of `Z_R` itself is the
//! exponential of that logarithm, and the call fails when the exponential is
//! not finite. The Hermite derivative weights are products of node gaps and
//! are formed in log space for the same reason. The projected frequency
//! `omega_0` of the mass-weighted reactant Hessian is reported separately and
//! is not this normalization.
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
//!    with [`PafiConfig::zero_temperature_tol`]. A mismatch returns
//!    [`FluxError::InvalidConfig`] before any sampling.
//! 4. The first image is certified as the reactant minimum before any
//!    sampling. The gradient norm must sit at or below
//!    [`crate::stats::SERIES_FLAT_TOL`], and the mass-weighted Hessian must be
//!    positive definite once eigenvalues on the zero-mode floor are set
//!    aside. The Hessian products are central differences whose step is a
//!    fixed fraction of the displacement, so a translation of every coordinate
//!    leaves the Hessian unchanged. `omega_0` is then solved by conjugate
//!    gradients on those products.
//! 5. On each plane the engine is constrained, placed at the anchor, given
//!    Maxwell-Boltzmann velocities, run for `burn_in_steps`, and then run for
//!    `sample_steps` steps that each record
//!    `lambda (n . X' - (x - X) . n')`. Batch means over the samples give the
//!    mean force and its standard error on each plane.
//! 6. The profile, barrier and rate follow from the plane means. The
//!    standard errors of the planes are independent and propagate through the
//!    Hermite weights of the barrier and, for the rate, through `Z_R`. Each
//!    profile and barrier error combines that sampling contribution, by
//!    hypot, with the Hermite-minus-trapezoid truncation error, taken on the
//!    plane nodes for the profile and at `s*` for the barrier.
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
//! - The first image is the reactant minimum. The call checks the gradient and
//!   the positive definite character of the mass-weighted Hessian there,
//!   excluding zero modes, and reports the projected frequency. A failing
//!   check, or a non-positive curvature along the path normal, makes that
//!   frequency undefined and the call fails. A profile whose maximum is the
//!   first sample has an empty reactant interval, so `Z_R` is undefined and
//!   the call fails.
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
use crate::stats::{SERIES_FLAT_TOL, StatsConfig, batch_means};

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

/// Default [`PafiConfig::zero_temperature_tol`]: the quarter-arc trapezoid
/// error at [`DEFAULT_PLANES`] (1.29%) plus the safety margin.
const DEFAULT_ZERO_TEMPERATURE_TOL: f64 = default_zero_temperature_tol(DEFAULT_PLANES);

fn effective_zero_temperature_tol(config: &PafiConfig) -> f64 {
    let tol = config.zero_temperature_tol;
    if tol.to_bits() == DEFAULT_ZERO_TEMPERATURE_TOL.to_bits() {
        default_zero_temperature_tol(config.planes)
    } else {
        tol
    }
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
    /// The stored default is `2 * (pi^2 / 12) / 8^2`, about `0.0257`, for the
    /// default 9 planes. If this field retains that default value, changing
    /// `planes` changes the effective tolerance to
    /// `2 * (pi^2 / 12) / (planes - 1)^2`. An explicitly different value is
    /// used unchanged and does not scale with `planes`.
    ///
    /// The scaling follows the trapezoid error on a quarter-turn circular arc
    /// through a harmonic well. Paths that turn through more than a quarter
    /// circle can need a larger tolerance. Larger values weaken the guard
    /// against inconsistent forces and tangents. Must be finite and `>= 0`.
    ///
    /// Default: `2 * (pi^2 / 12) / 8^2`, about `0.0257`.
    pub zero_temperature_tol: f64,
    /// Base of the deterministic per-plane seeds `seed + 4099 * plane + 1000003 * round`
    /// (wrapping).
    ///
    /// Default: `1`.
    pub seed: u64,
    /// Batch-means uncertainty settings for the per-plane force series.
    ///
    /// Default: [`StatsConfig::default`].
    pub stats: StatsConfig,
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
            zero_temperature_tol: DEFAULT_ZERO_TEMPERATURE_TOL,
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
    /// Plane coordinates `s_k`: the mass-weighted chord-length parameter of
    /// the spline, from `0` at the first image to the total path length at
    /// the last, at equal spacing. They are not scaled to `[0, 1]`.
    pub arc_lengths: Vec<f64>,
    /// Integrated free energy `F(s_k) - F(s_0)` at each plane, so the first
    /// entry is `0`.
    pub free_energy: Vec<f64>,
    /// Standard error of `free_energy` at each plane. It combines, by hypot,
    /// the batch-means errors of the plane forces propagated through the
    /// Hermite weights with the truncation error: the absolute difference
    /// between the Hermite and trapezoid integrals of the plane means from
    /// the first plane to this plane, on the plane nodes. The first entry is
    /// `0`.
    pub standard_error: Vec<f64>,
    /// Free-energy barrier `Delta F = F(s*) - F(s_0)` at the dividing plane,
    /// the global maximum of the interpolated profile.
    pub barrier: f64,
    /// Standard error of [`Self::barrier`]. It combines, by hypot, the
    /// batch-means errors of the plane forces propagated through the Hermite
    /// weights of the barrier with the truncation error: the absolute
    /// difference between the Hermite integral at [`Self::dividing_arc`] and
    /// the trapezoid integral of the plane means up to that point, a partial
    /// trapezoid when the point lies between plane nodes.
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
    /// Reactant partition, when its finite logarithm has a representable
    /// positive exponential.
    pub reactant_partition: Option<f64>,
    /// Natural logarithm of the reactant partition,
    /// `ln Z_R = ln(2 * \int_0^{s_div} exp(-F(s) / k_B T) ds)`.
    pub reactant_log_partition: f64,
    /// Normalized PAFI rate, when its finite logarithm has a representable
    /// positive exponential.
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

fn build_plane_geometries(
    path_images: ArrayView2<'_, f64>,
    masses: ArrayView1<'_, f64>,
    planes: usize,
) -> Result<Vec<PlaneGeometry>, FluxError> {
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

    let mut geometries = Vec::with_capacity(planes);
    for k in 0..planes {
        let arc = total_arc * (k as f64) / ((planes - 1) as f64);
        let (anchor, tangent, curvature) = spline.evaluate(arc);
        let mut normal_raw = Array1::zeros(path_images.ncols());
        let mut normal_derivative_raw = Array1::zeros(path_images.ncols());
        for j in 0..path_images.ncols() {
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
        let mut dn_ds = Array1::zeros(path_images.ncols());
        for j in 0..path_images.ncols() {
            dn_ds[j] = (normal_derivative_raw[j] - normal[j] * projection) / normal_norm;
        }
        let dc_ds = dot(normal.view(), tangent.view());
        if !dc_ds.is_finite() || dc_ds <= f64::MIN_POSITIVE.sqrt() {
            return Err(FluxError::InvalidConfig(
                "PAFI tangent projection to hyperplane normal is non-positive",
            ));
        }

        geometries.push(PlaneGeometry {
            anchor,
            normal,
            dn_ds,
            dc_ds,
            tangent,
            arc,
        });
    }
    Ok(geometries)
}

fn sample_plane<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    geometry: &PlaneGeometry,
    config: &PafiConfig,
    plane_index: usize,
    round: usize,
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
        values.push(projected_force_sample(engine, geometry, lambda)?);
    }
    Ok(values)
}

fn projected_force_sample<E: DynamicsEngine + ?Sized>(
    engine: &E,
    geometry: &PlaneGeometry,
    lambda: f64,
) -> Result<f64, FluxError> {
    let positions = engine.positions();
    let mut jacobian_projection = 0.0_f64;
    for i in 0..engine.dof() {
        jacobian_projection += (positions[i] - geometry.anchor[i]) * geometry.dn_ds[i];
    }
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
/// `faer` feature is off. A larger reactant is rejected instead of panicking.
const REACTANT_HESSIAN_EIGEN_LIMIT: usize = 128;

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

    /// Global maximum of the interpolated profile.
    ///
    /// Each segment is the integral of a cubic Hermite interpolant of the mean
    /// force, so its derivative is the cubic `a u^3 + b u^2 + c u + d`. The
    /// maximum on that segment is the larger of the endpoint values and the
    /// values at the real roots inside the segment. A vanishing leading
    /// coefficient leaves a quadratic, including the `a == 0` and `b == 0`
    /// degeneracies. The highest sampled plane is recorded separately,
    /// because an overshoot can put `s*` on an interval that does not touch
    /// that plane.
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
            let (u, value) = self.segment_maximum(k, gradients, &slopes, free_energy);
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
        DividingPlane {
            index,
            arc: self.arcs[best_interval] + best_u * width,
            free_energy: best_value,
            weights: self.interval_weights(best_interval, best_u),
            partition_end,
        }
    }

    /// `(u, F)` at the maximum of segment `k`, with `u` in `[0, 1]`.
    ///
    /// Ties keep the smaller `u`, so a flat segment reports its left endpoint.
    fn segment_maximum(
        &self,
        k: usize,
        gradients: &[f64],
        slopes: &[f64],
        free_energy: &[f64],
    ) -> (f64, f64) {
        let h = self.arcs[k + 1] - self.arcs[k];
        let g0 = gradients[k];
        let g1 = gradients[k + 1];
        let s0 = slopes[k];
        let s1 = slopes[k + 1];
        // `dF/ds` on this segment. `h > 0`, so its roots are the critical
        // points of `F`. Coefficients of `a u^3 + b u^2 + c u + d`.
        let a = (g0 - g1).mul_add(2.0, h * (s0 + s1));
        let b = (g1 - g0).mul_add(3.0, -h * s0.mul_add(2.0, s1));
        let c = h * s0;
        let d = g0;
        let (nroots, roots) = cubic_real_roots(a, b, c, d);
        let mut best_u = 0.0_f64;
        let mut best_value = self.interpolated_value(k, 0.0, gradients, slopes, free_energy);
        let mut consider = |u: f64| {
            let Some(u) = root_in_unit_interval(u) else {
                return;
            };
            let value = self.interpolated_value(k, u, gradients, slopes, free_energy);
            if !value.is_finite() {
                return;
            }
            if value > best_value || (value.to_bits() == best_value.to_bits() && u < best_u) {
                best_u = u;
                best_value = value;
            }
        };
        for &root in roots.iter().take(nroots) {
            consider(root);
        }
        consider(1.0);
        (best_u, best_value)
    }
}

/// Clamp a root that landed just outside `[0, 1]` from rounding.
fn root_in_unit_interval(u: f64) -> Option<f64> {
    if !u.is_finite() {
        return None;
    }
    let pad = 64.0 * f64::EPSILON;
    if !(-pad..=1.0 + pad).contains(&u) {
        return None;
    }
    Some(u.clamp(0.0, 1.0))
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

/// Real roots of `a x^3 + b x^2 + c x + d = 0`.
///
/// A leading coefficient that is zero, or too small to change the polynomial
/// on `[0, 1]` by more than a rounding of the remaining terms, is solved by
/// [`quadratic_real_roots`]. Three real roots use the trigonometric form.
/// One real root uses Cardano's formula with the product identity
/// `u v = -p / 3`, which avoids cancellation in the sum of the two cube
/// roots. Each accepted root is polished by two Newton steps.
fn cubic_real_roots(a: f64, b: f64, c: f64, d: f64) -> (usize, [f64; 3]) {
    let mut roots = [0.0; 3];
    if !a.is_finite() || !b.is_finite() || !c.is_finite() || !d.is_finite() {
        return (0, roots);
    }
    let rest = b.abs() + c.abs() + d.abs();
    if a == 0.0 || a.abs() <= rest * f64::EPSILON {
        let (count, quadratic) = quadratic_real_roots(b, c, d);
        roots[..count].copy_from_slice(&quadratic[..count]);
        return (count, roots);
    }
    let monic_b = b / a;
    let monic_c = c / a;
    let monic_d = d / a;
    if !monic_b.is_finite() || !monic_c.is_finite() || !monic_d.is_finite() {
        let (count, quadratic) = quadratic_real_roots(b, c, d);
        roots[..count].copy_from_slice(&quadratic[..count]);
        return (count, roots);
    }
    let shift = monic_b / 3.0;
    let p = monic_c - monic_b * shift;
    let q = monic_d + shift * (2.0 * shift * shift - monic_c);
    let p_over_3 = p / 3.0;
    let half_q = 0.5 * q;
    let discriminant = half_q * half_q + p_over_3 * p_over_3 * p_over_3;
    if !p.is_finite() || !q.is_finite() || !discriminant.is_finite() {
        return (0, roots);
    }
    let mut count = 0usize;
    let mut push = |y: f64| {
        let x = polish_cubic_root(a, b, c, d, y - shift);
        if x.is_finite() {
            roots[count] = x;
            count += 1;
        }
    };
    if discriminant > 0.0 {
        let sqrt_discriminant = discriminant.sqrt();
        let mut u = (-half_q + sqrt_discriminant).cbrt();
        let mut v = (-half_q - sqrt_discriminant).cbrt();
        let product = -p_over_3;
        if u.abs() >= v.abs() {
            if u != 0.0 {
                v = product / u;
            }
        } else if v != 0.0 {
            u = product / v;
        }
        push(u + v);
    } else if discriminant == 0.0 {
        let u = (-half_q).cbrt();
        push(2.0 * u);
        push(-u);
    } else {
        let radius = (-p_over_3).sqrt();
        if !radius.is_finite() || radius == 0.0 {
            push(0.0);
        } else {
            let cosine = (-half_q / (radius * radius * radius)).clamp(-1.0, 1.0);
            let angle = cosine.acos();
            let diameter = 2.0 * radius;
            for k in 0..3 {
                let y = diameter * (angle / 3.0 - TAU * f64::from(k) / 3.0).cos();
                push(y);
            }
        }
    }
    (count, roots)
}

/// Two Newton steps from a closed-form seed. A step that leaves the
/// neighbourhood of the seed is rejected, so a poor seed stays put.
fn polish_cubic_root(a: f64, b: f64, c: f64, d: f64, mut x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    for _ in 0..2 {
        let slope = (3.0 * a).mul_add(x, 2.0 * b).mul_add(x, c);
        if !slope.is_finite() || slope == 0.0 {
            break;
        }
        let value = a.mul_add(x, b).mul_add(x, c).mul_add(x, d);
        if !value.is_finite() {
            break;
        }
        let step = value / slope;
        if !step.is_finite() || step.abs() > 1.0 {
            break;
        }
        x -= step;
    }
    x
}

/// `Z_R = 2 \int_0^{s_div} exp(-F / kT) ds` by the trapezoid rule, in log space.
///
/// `s_div` is the last sample of the segment that holds the interpolant
/// maximum. `d_ln_zr_d_f[i]` is `d ln Z_R / d F_i`.
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

    let reactant_log_partition = 2.0_f64.ln() + exponent_shift + scaled_half_partition.ln();
    if !reactant_log_partition.is_finite() {
        return Err(FluxError::NonFinite("PAFI reactant log partition"));
    }
    let mut d_ln_zr_d_f = vec![0.0_f64; n];
    for i in 0..=dividing {
        d_ln_zr_d_f[i] = -scaled_numerator[i] / (kbt * scaled_half_partition);
    }
    Ok((reactant_log_partition, d_ln_zr_d_f))
}

fn trapezoid_to(arcs: &[f64], gradients: &[f64], arc: f64) -> f64 {
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

fn trapezoid_cumulative(arcs: &[f64], gradients: &[f64]) -> Vec<f64> {
    (0..arcs.len())
        .map(|k| trapezoid_to(arcs, gradients, arcs[k]))
        .collect()
}

fn integrate_profile(
    quadrature: &Quadrature,
    gradients: &[f64],
    gradient_errors: &[f64],
    kbt: f64,
    reactant_curvature: f64,
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
    let dividing = quadrature.dividing_plane(gradients, &free_energy);
    let lower = trapezoid_cumulative(&quadrature.arcs, gradients);
    if lower.iter().any(|value| !value.is_finite()) {
        return Err(FluxError::NonFinite("PAFI trapezoid free energy"));
    }
    let propagate = |weights: &[f64]| {
        weights
            .iter()
            .zip(gradient_errors)
            .map(|(weight, sigma)| weight * weight * sigma * sigma)
            .sum::<f64>()
            .sqrt()
    };
    let standard_error: Vec<f64> = quadrature
        .cumulative
        .iter()
        .zip(&free_energy)
        .zip(&lower)
        .map(|((weights, hermite), trapezoid)| {
            propagate(weights).hypot((hermite - trapezoid).abs())
        })
        .collect();
    let barrier = dividing.free_energy;
    let lower_barrier = trapezoid_to(&quadrature.arcs, gradients, dividing.arc);
    if !lower_barrier.is_finite() {
        return Err(FluxError::NonFinite("PAFI trapezoid barrier"));
    }
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
    let mut variance_ln_rate = 0.0_f64;
    for (gradient_index, sigma) in gradient_errors.iter().copied().enumerate() {
        let barrier_sensitivity = dividing.weights[gradient_index];
        let mut partition_log_sensitivity = 0.0_f64;
        for profile_index in 0..=dividing.partition_end {
            partition_log_sensitivity +=
                d_ln_zr_d_f[profile_index] * quadrature.cumulative[profile_index][gradient_index];
        }
        let log_rate_sensitivity = -barrier_sensitivity / kbt - partition_log_sensitivity;
        variance_ln_rate += log_rate_sensitivity * log_rate_sensitivity * sigma * sigma;
    }
    let sigma_ln_k = variance_ln_rate.sqrt();
    if !sigma_ln_k.is_finite() || sigma_ln_k < 0.0 {
        return Err(FluxError::InvalidConfig(
            "PAFI normalized-rate logarithmic error must be finite and non-negative",
        ));
    }
    let normalized_rate_error = {
        let value = (normalized_log_rate + sigma_ln_k.ln()).exp();
        value.is_finite().then_some(value)
    };

    Ok(FreeEnergyProfile {
        arc_lengths: quadrature.arcs.clone(),
        free_energy,
        standard_error,
        barrier,
        barrier_error,
        dividing_index: dividing.index,
        dividing_arc: dividing.arc,
        reactant_frequency,
        reactant_partition,
        reactant_log_partition,
        normalized_rate,
        normalized_log_rate,
        normalized_rate_error,
        quadrature_residual: 0.0,
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
    Ok(hessian)
}

/// Reject a first image that is not a reactant minimum.
///
/// The gradient norm uses [`SERIES_FLAT_TOL`], the same stationary-force test
/// as OH-TST. The mass-weighted Hessian is diagonalized with
/// [`rgsaddle::exact_eigh`]. Eigenvalues no larger than
/// [`reactant_zero_mode_floor`] times `1 +` the stiffest curvature are zero
/// modes. Every other eigenvalue must be positive, and at least one positive
/// eigenvalue must remain.
fn ensure_reactant_minimum<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    origin: ArrayView1<'_, f64>,
) -> Result<(), FluxError> {
    engine.set_positions(origin)?;
    let (_, forces) = engine.energy_and_forces()?;
    for force in forces.iter().copied() {
        if !force.is_finite() {
            return Err(FluxError::NonFinite("PAFI reactant force"));
        }
    }
    let gradient_norm = nrm2(forces.view());
    if !gradient_norm.is_finite() || gradient_norm > SERIES_FLAT_TOL {
        return Err(FluxError::InvalidConfig(
            "PAFI reactant gradient is not zero",
        ));
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
    Ok(())
}

/// Squared projected reactant frequency `omega_0^2 = 1 / (n^T H^-1 n)`.
///
/// `H` is the mass-weighted Hessian at the first path image and `n` the unit
/// mass-weighted path tangent there. `H u = n` is solved by conjugate
/// gradients. The products are central differences of
/// [`DynamicsEngine::energy_and_forces`], so the solve stays generic over the
/// engine trait.
fn reactant_curvature<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    geometry: &PlaneGeometry,
) -> Result<f64, FluxError> {
    let dof = engine.dof();
    let root_mass: Array1<f64> = engine.masses().mapv(f64::sqrt);
    let origin = geometry.anchor.clone();

    let mut normal = Array1::zeros(dof);
    for i in 0..dof {
        normal[i] = root_mass[i] * geometry.tangent[i];
    }
    let normal_norm = nrm2(normal.view());
    if !normal_norm.is_finite() || normal_norm <= f64::MIN_POSITIVE.sqrt() {
        return Err(FluxError::InvalidConfig("PAFI reactant normal underflowed"));
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
        let image =
            mass_weighted_hessian_product(engine, origin.view(), root_mass.view(), search.view())?;
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
///   `zero_temperature_tol` is negative or not finite, or `target_error` is
///   negative or not finite,
/// - `path_images` has fewer than two rows, or two neighbouring images
///   coincide in the mass-weighted metric,
/// - the thermal energy `k_B T` of the engine is not positive,
/// - the zero-temperature check of [`PafiConfig::zero_temperature_tol`] fails,
/// - a plane normal degenerates, the first image is not a minimum (the
///   gradient norm exceeds [`crate::stats::SERIES_FLAT_TOL`], or the
///   mass-weighted Hessian is not positive definite once zero modes are
///   excluded), or the reactant curvature along the first path normal is not
///   finite and positive, so `Z_R` is not defined.
///
/// Returns [`FluxError::EmptyInput`] for an empty coordinate axis and
/// [`FluxError::Shape`] when the engine masses do not match the path
/// coordinates. [`FluxError::NonFinite`] reports non-finite path coordinates,
/// constraint forces, force samples, profile values, plane means, Lagrange
/// weights, or a reactant partition whose exponential overflowed. Errors
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
    if !config.zero_temperature_tol.is_finite() || config.zero_temperature_tol < 0.0 {
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
        let integrated_energy = quadrature.integrate(&path_gradients);
        let mut quadrature_residual = 0.0_f64;
        for (integral, energy) in integrated_energy.iter().zip(path_energies.iter()) {
            let discrepancy = (integral - (energy - path_energies[0])).abs();
            if !discrepancy.is_finite() {
                return Err(FluxError::NonFinite("PAFI path-force quadrature residual"));
            }
            quadrature_residual = quadrature_residual.max(discrepancy);
        }
        ensure_reactant_minimum(engine, geometries[0].anchor.view())?;
        let curvature = reactant_curvature(engine, &geometries[0])?;

        let mut plane_samples = vec![Vec::new(); geometries.len()];
        let mut final_profile = None;
        for round in 0..config.max_rounds {
            for (plane_index, geometry) in geometries.iter().enumerate() {
                let samples = sample_plane(engine, geometry, config, plane_index, round)?;
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
            let mut profile = integrate_profile(&quadrature, &gradients, &errors, kbt, curvature)?;
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

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, Array2, ArrayView1};

    use super::*;
    use crate::engine::native::{NativeConfig, NativeEngine};
    use crate::engine::{DynamicsEngine, ThermostatConfig};

    struct Quartic;

    impl rgsaddle::PointSurface for Quartic {
        fn eval(
            &self,
            x: ArrayView1<'_, f64>,
        ) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
            Ok((x[0].powi(4), Array1::from_vec(vec![4.0 * x[0].powi(3)])))
        }
    }

    struct InconsistentQuadraticGradient;

    impl rgsaddle::PointSurface for InconsistentQuadraticGradient {
        fn eval(
            &self,
            x: ArrayView1<'_, f64>,
        ) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
            // `V(x) = x^2 / 2` is a minimum at the reactant `x = 0` and rises
            // by 1/2 on `[0, 1]`. The reported gradient matches `V'` at 0 and
            // adds `QUADRATURE_SHORTFALL * 6 * x * (x - 1)` in integrated form:
            // `g(x) = (1 - 6 s) x + 6 s x^2` with `s = QUADRATURE_SHORTFALL`,
            // and `∫_0^1 g = 1/2 - s`. The trapezoid of that quadratic on 33
            // planes stays near the shortfall, above the scaled default
            // tolerance and below the explicit tolerance 0.0257.
            const QUADRATURE_SHORTFALL: f64 = 0.02;
            let bend = QUADRATURE_SHORTFALL * 6.0;
            let reported = (1.0 - bend) * x[0] + bend * x[0] * x[0];
            Ok((0.5 * x[0] * x[0], Array1::from_vec(vec![reported])))
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
        let projected = projected_force_sample(&engine, middle, 2.0).expect("projected force");
        assert_abs_diff_eq!(projected, 2.0 * middle.dc_ds, epsilon = f64::EPSILON);
        assert!((middle.dc_ds - 1.0).abs() > 0.1);
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
        integrate_profile(&quadrature, gradients, errors, kbt, curvature)
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
        let error = quarter_arc_relative_error(33, config.planes);
        assert!(
            error < config.zero_temperature_tol,
            "relative error {error} versus default tolerance {}",
            config.zero_temperature_tol
        );
        assert!(
            config.zero_temperature_tol < 2.0 * (1.0 + ZERO_TEMPERATURE_SAFETY_MARGIN) * error,
            "default tolerance {} is no longer calibrated to {error}",
            config.zero_temperature_tol
        );

        // The quarter-arc anchor is not a stationary point, so the profile
        // call starts at the harmonic minimum and runs out along `x`.
        let (mut engine, _) = quarter_arc_engine();
        let mut path = Array2::zeros((config.planes, 2));
        for k in 0..config.planes {
            path[[k, 0]] = (k as f64) / ((config.planes - 1) as f64);
        }
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
        let slopes = quadrature.slopes(gradients);
        let mut best = f64::NEG_INFINITY;
        for k in 0..arcs.len() - 1 {
            for j in 0..=samples {
                let u = f64::from(u32::try_from(j).expect("sample index"))
                    / f64::from(u32::try_from(samples).expect("sample count"));
                let value = quadrature.interpolated_value(k, u, gradients, &slopes, &free_energy);
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
        let slopes = quadrature.slopes(gradients);
        let steps = 4_000usize;
        let mut count = 0usize;
        let mut previous = f64::NEG_INFINITY;
        let mut current = quadrature.interpolated_value(0, 0.0, gradients, &slopes, &free_energy);
        for k in 0..arcs.len() - 1 {
            for j in 1..=steps {
                let u = f64::from(u32::try_from(j).expect("step"))
                    / f64::from(u32::try_from(steps).expect("steps"));
                let next = quadrature.interpolated_value(k, u, gradients, &slopes, &free_energy);
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

        let (count, mut roots) = cubic_real_roots(0.0, 1.0, -1.0, 0.16);
        assert_eq!(count, 2);
        roots[..count].sort_by(f64::total_cmp);
        assert_abs_diff_eq!(roots[0], 0.2, epsilon = 1.0e-12);
        assert_abs_diff_eq!(roots[1], 0.8, epsilon = 1.0e-12);

        let (count, roots) = cubic_real_roots(0.0, 0.0, 2.0, -0.5);
        assert_eq!(count, 1);
        assert_abs_diff_eq!(roots[0], 0.25, epsilon = 1.0e-15);

        let (count, _) = cubic_real_roots(0.0, 0.0, 0.0, 4.0);
        assert_eq!(count, 0);

        let left = 0.2_f64;
        let middle = 0.5_f64;
        let right = 0.8_f64;
        let (count, mut roots) = cubic_real_roots(
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

    #[test]
    fn segment_derivative_matches_the_cubic_hermite_force() {
        let arcs = [0.0, 0.3, 1.1, 1.7];
        let gradients = [0.4, -1.2, 0.7, 0.2];
        let quadrature = Quadrature::new(&arcs).expect("quadrature");
        let slopes = quadrature.slopes(&gradients);
        for k in 0..arcs.len() - 1 {
            let h = arcs[k + 1] - arcs[k];
            let g0 = gradients[k];
            let g1 = gradients[k + 1];
            let s0 = slopes[k];
            let s1 = slopes[k + 1];
            let a = (g0 - g1).mul_add(2.0, h * (s0 + s1));
            let b = (g1 - g0).mul_add(3.0, -h * s0.mul_add(2.0, s1));
            let c = h * s0;
            let d = g0;
            for j in 0..=8 {
                let u = f64::from(j) / 8.0;
                let u2 = u * u;
                let u3 = u2 * u;
                let force = g0 * (2.0 * u3 - 3.0 * u2 + 1.0)
                    + g1 * (-2.0 * u3 + 3.0 * u2)
                    + h * (s0 * (u3 - 2.0 * u2 + u) + s1 * (u3 - u2));
                let polynomial = a.mul_add(u, b).mul_add(u, c).mul_add(u, d);
                assert_abs_diff_eq!(polynomial, force, epsilon = 1.0e-12);
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
        let descending_plane =
            descending_quadrature.dividing_plane(&descending, &descending_energy);
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

    #[test]
    fn default_tolerance_scales_with_planes_for_consistency_check() {
        let path = Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).expect("shape");
        let mut engine = NativeEngine::new(
            InconsistentQuadraticGradient,
            NativeConfig::default()
                .with_masses(Array1::from_vec(vec![1.0]))
                .with_thermostat(ThermostatConfig::default().with_temperature(0.2))
                .with_seed(23),
        )
        .expect("engine");
        let config = PafiConfig::default()
            .with_planes(33)
            .with_burn_in_steps(1)
            .with_sample_steps(2);
        let result = pafi_profile(&mut engine, path.view(), &config);
        assert_eq!(
            result,
            Err(FluxError::InvalidConfig(
                "PAFI zero-temperature path-force quadrature exceeds tolerance"
            ))
        );

        let mut legacy_engine = NativeEngine::new(
            InconsistentQuadraticGradient,
            NativeConfig::default()
                .with_masses(Array1::from_vec(vec![1.0]))
                .with_thermostat(ThermostatConfig::default().with_temperature(0.2))
                .with_seed(23),
        )
        .expect("engine");
        let legacy_like = PafiConfig::default()
            .with_planes(33)
            .with_zero_temperature_tol(0.0257)
            .with_burn_in_steps(1)
            .with_sample_steps(2)
            .with_stats(StatsConfig::default().with_min_batches(2));
        assert!(pafi_profile(&mut legacy_engine, path.view(), &legacy_like).is_ok());
    }

    #[test]
    fn only_the_exact_default_tolerance_scales_with_planes() {
        let exact = PafiConfig::default().with_planes(33);
        assert_eq!(
            effective_zero_temperature_tol(&exact).to_bits(),
            default_zero_temperature_tol(33).to_bits()
        );

        let next_up = f64::from_bits(DEFAULT_ZERO_TEMPERATURE_TOL.to_bits() + 1);
        let near = PafiConfig::default()
            .with_planes(33)
            .with_zero_temperature_tol(next_up);
        assert_eq!(
            effective_zero_temperature_tol(&near).to_bits(),
            next_up.to_bits()
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
}
