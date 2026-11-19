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
//! The profile is the trapezoid integral of the per-plane mean forces,
//! `F(s_k) - F(s_0) = sum_j 0.5 (g_{j-1} + g_j) (s_j - s_{j-1})`. The barrier is
//! `Delta F = max_k F(s_k)` at the dividing plane `s*`, and the normalized rate
//! is
//!
//! `k = sqrt(k_B T / (2 pi)) exp(-F(s*) / k_B T) / Z_R`
//!
//! with `Z_R = 2 * \int_0^{s*} exp(-F(s) / k_B T) ds`, also evaluated by the
//! trapezoid rule.
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
//! 4. On each plane the engine is constrained, placed at the anchor, given
//!    Maxwell-Boltzmann velocities, run for `burn_in_steps`, and then run for
//!    `sample_steps` steps that each record
//!    `lambda (n . X' - (x - X) . n')`. Batch means over the samples give the
//!    mean force and its standard error on each plane.
//! 5. The profile, barrier and rate follow from the plane means. The
//!    standard errors of the planes are independent and propagate through the
//!    trapezoid weights and, for the rate, through `Z_R` by the delta method.
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
//! - The free energy rises somewhere along the path, so the dividing plane is
//!   not the first one. Otherwise `Z_R` is zero and the call fails.
//!
//! ## Units
//!
//! - Positions, path images and masses use the units of the engine. Arc
//!   lengths `s` are mass-weighted, `sqrt(mass) * length`, and equal lengths
//!   for unit masses.
//! - Energies, `free_energy`, `standard_error`, `barrier`, `barrier_error`,
//!   `target_error` and the energy part of the tolerance are in engine energy
//!   units. `time_step` is in engine time units.
//! - [`FreeEnergyProfile::normalized_rate`] has units of inverse engine time
//!   and [`FreeEnergyProfile::reactant_partition`] has the units of `s`.
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
//! check.
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
//! assert!(profile.normalized_rate > 0.0);
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

use ndarray::{Array1, ArrayView1, ArrayView2};
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
    /// Standard error of `free_energy` at each plane, propagated from the
    /// batch-means errors of the plane forces through the trapezoid weights.
    /// The first entry is `0`.
    pub standard_error: Vec<f64>,
    /// Free-energy barrier `Delta F = max_k F(s_k)`, attained at the dividing
    /// plane `s*`.
    pub barrier: f64,
    /// Standard error of [`Self::barrier`].
    pub barrier_error: f64,
    /// Index of the dividing plane `s*`: the first plane where
    /// [`Self::barrier`] is attained.
    pub dividing_index: usize,
    /// Reactant-side normalization
    /// `Z_R = 2 * \int_0^{s*} exp(-F(s) / k_B T) ds`.
    pub reactant_partition: f64,
    /// Natural logarithm of the reactant-side normalization
    /// `ln Z_R = ln(2 * \int_0^{s*} exp(-F(s) / k_B T) ds)`.
    pub reactant_log_partition: f64,
    /// Normalized PAFI rate
    /// `k = sqrt(k_B T / (2 pi)) * exp(-F(s*) / k_B T) / Z_R`.
    pub normalized_rate: f64,
    /// Natural logarithm of [`Self::normalized_rate`].
    pub normalized_log_rate: f64,
    /// Propagated standard error of [`Self::normalized_rate`].
    pub normalized_rate_error: f64,
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

fn integrate_means(
    arc_lengths: &[f64],
    gradients: &[f64],
) -> Result<(Vec<f64>, f64, usize), FluxError> {
    let n = arc_lengths.len();
    if n < 2 || gradients.len() != n {
        return Err(FluxError::Shape {
            expected: n,
            got: gradients.len(),
            context: "PAFI mean-profile integration vectors",
        });
    }
    let mut free_energy = vec![0.0_f64; n];
    for i in 1..n {
        let ds = arc_lengths[i] - arc_lengths[i - 1];
        if !ds.is_finite() || ds <= 0.0 {
            return Err(FluxError::InvalidConfig(
                "PAFI arc lengths must strictly increase",
            ));
        }
        free_energy[i] = free_energy[i - 1] + 0.5 * (gradients[i - 1] + gradients[i]) * ds;
        if !free_energy[i].is_finite() {
            return Err(FluxError::NonFinite("PAFI integrated free energy"));
        }
    }
    let mut dividing = 0usize;
    for i in 1..n {
        if free_energy[i] > free_energy[dividing] {
            dividing = i;
        }
    }
    let barrier = free_energy[dividing];
    Ok((free_energy, barrier, dividing))
}

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
        d_ln_zr_d_f[i] = -scaled_numerator[i] / (kbt * scaled_half_partition);
    }

    Ok((reactant_log_partition, d_ln_zr_d_f))
}

fn integrate_profile(
    arc_lengths: &[f64],
    gradients: &[f64],
    gradient_errors: &[f64],
    kbt: f64,
) -> Result<FreeEnergyProfile, FluxError> {
    let n = arc_lengths.len();
    if n < 2 || gradients.len() != n || gradient_errors.len() != n {
        return Err(FluxError::Shape {
            expected: n,
            got: gradients.len().max(gradient_errors.len()),
            context: "PAFI profile integration vectors",
        });
    }
    let (free_energy, barrier, dividing) = integrate_means(arc_lengths, gradients)?;

    let mut integration_weights = vec![vec![0.0_f64; n]; n];
    for i in 1..n {
        let previous = integration_weights[i - 1].clone();
        integration_weights[i] = previous;
        let ds = arc_lengths[i] - arc_lengths[i - 1];
        integration_weights[i][i - 1] += 0.5 * ds;
        integration_weights[i][i] += 0.5 * ds;
    }
    for &sigma in gradient_errors {
        if !sigma.is_finite() || sigma < 0.0 {
            return Err(FluxError::InvalidConfig(
                "PAFI gradient standard errors must be finite and non-negative",
            ));
        }
    }
    let mut standard_error = vec![0.0_f64; n];
    for i in 1..n {
        let mut variance = 0.0_f64;
        for (weight, sigma) in integration_weights[i].iter().zip(gradient_errors) {
            variance += weight * weight * sigma * sigma;
        }
        standard_error[i] = variance.sqrt();
    }

    let (reactant_log_partition, d_ln_zr_d_f) =
        reactant_log_partition_and_sensitivity(arc_lengths, &free_energy, dividing, kbt)?;
    let reactant_partition = reactant_log_partition.exp();
    let barrier_error = standard_error[dividing];
    let normalized_log_rate = 0.5 * (kbt / TAU).ln() - barrier / kbt - reactant_log_partition;
    if !normalized_log_rate.is_finite() {
        return Err(FluxError::InvalidConfig(
            "PAFI normalized log rate must be finite",
        ));
    }
    let normalized_rate = normalized_log_rate.exp();
    if !normalized_rate.is_finite() || normalized_rate < 0.0 {
        return Err(FluxError::InvalidConfig(
            "PAFI normalized rate must be finite",
        ));
    }
    let mut variance_ln_rate = 0.0_f64;
    for (gradient_index, sigma) in gradient_errors.iter().copied().enumerate() {
        let barrier_sensitivity = integration_weights[dividing][gradient_index];
        let mut partition_log_sensitivity = 0.0_f64;
        for profile_index in 0..=dividing {
            partition_log_sensitivity +=
                d_ln_zr_d_f[profile_index] * integration_weights[profile_index][gradient_index];
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
    let normalized_rate_error = normalized_rate * sigma_ln_k;
    if !normalized_rate_error.is_finite() || normalized_rate_error < 0.0 {
        return Err(FluxError::InvalidConfig(
            "PAFI normalized-rate standard error must be finite and non-negative",
        ));
    }

    Ok(FreeEnergyProfile {
        arc_lengths: arc_lengths.to_vec(),
        free_energy,
        standard_error,
        barrier,
        barrier_error,
        dividing_index: dividing,
        reactant_partition,
        reactant_log_partition,
        normalized_rate,
        normalized_log_rate,
        normalized_rate_error,
    })
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
/// - a plane normal degenerates, or the dividing plane is the first plane so
///   that the reactant partition `Z_R` is not finite and positive.
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
            let profile = integrate_profile(&arcs, &gradients, &errors, kbt)?;
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
/// Following the equal-measure defensive-mixture principle (de Boor 1973;
/// Goodson, Goswami, Rabitz, and Warren 2000), the monitor floor is set equal
/// to the path-average monitor `w_bar = (1 / L) \int_0^L w(s) ds`. Half of the
/// total coordinate measure `sigma` comes from uniform arc length `ds / L` and
/// half from the normalized monitor `w(s) ds / (w_bar L)` (reducing to pure
/// uniform arc length when `w_bar == 0`).
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
}

/// Chebyshev / cosine spectral expansion and certified tail bound for a PAFI profile.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct SpectralTailReport {
    /// Chebyshev expansion coefficients `a_m` of the mean constraint force `dF/ds`
    /// on `x = 2 * (s - s_0) / L - 1` (`T_m(x) = cos(m * arccos(x))`).
    pub force_coefficients: Vec<f64>,
    /// Chebyshev expansion coefficients `c_m` of the integrated free energy `F(s)`.
    pub profile_coefficients: Vec<f64>,
    /// Certified upper bound on the unresolved spectral / quadrature error of `F*`.
    pub tail_bound: f64,
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

fn spline_second_derivative_basis(arc_lengths: &[f64]) -> Vec<Vec<f64>> {
    let n = arc_lengths.len();
    let mut second = vec![vec![0.0_f64; n]; n];
    if n == 3 {
        let left = arc_lengths[1] - arc_lengths[0];
        let right = arc_lengths[2] - arc_lengths[1];
        let scale = 2.0 / (left + right);
        let c0 = scale / left;
        let c1 = -scale * (1.0 / left + 1.0 / right);
        let c2 = scale / right;
        for row in &mut second {
            row[0] = c0;
            row[1] = c1;
            row[2] = c2;
        }
    } else if n > 3 {
        let interior = n - 2;
        let mut lower = vec![0.0_f64; interior];
        let mut diagonal = vec![0.0_f64; interior];
        let mut upper = vec![0.0_f64; interior];
        let mut rhs = vec![vec![0.0_f64; n]; interior];
        for k in 0..interior {
            let i = k + 1;
            let left = arc_lengths[i] - arc_lengths[i - 1];
            let right = arc_lengths[i + 1] - arc_lengths[i];
            lower[k] = left;
            diagonal[k] = 2.0 * (left + right);
            upper[k] = right;
            rhs[k][i - 1] = 6.0 / left;
            rhs[k][i] = -6.0 * (1.0 / left + 1.0 / right);
            rhs[k][i + 1] = 6.0 / right;
        }
        let first = arc_lengths[1] - arc_lengths[0];
        let second_width = arc_lengths[2] - arc_lengths[1];
        diagonal[0] += first * (first + second_width) / second_width;
        upper[0] -= first * first / second_width;

        let penultimate = arc_lengths[n - 2] - arc_lengths[n - 3];
        let last = arc_lengths[n - 1] - arc_lengths[n - 2];
        lower[interior - 1] -= last * last / penultimate;
        diagonal[interior - 1] += last * (penultimate + last) / penultimate;

        for k in 1..interior {
            let factor = lower[k] / diagonal[k - 1];
            diagonal[k] -= factor * upper[k - 1];
            for j in 0..n {
                rhs[k][j] -= factor * rhs[k - 1][j];
            }
        }
        for k in (0..interior).rev() {
            let i = k + 1;
            for j in 0..n {
                let next = if k + 1 < interior {
                    upper[k] * second[i + 1][j]
                } else {
                    0.0
                };
                second[i][j] = (rhs[k][j] - next) / diagonal[k];
            }
        }
        for j in 0..n {
            second[0][j] =
                ((first + second_width) * second[1][j] - first * second[2][j]) / second_width;
            second[n - 1][j] =
                ((penultimate + last) * second[n - 2][j] - last * second[n - 3][j]) / penultimate;
        }
    }
    second
}

fn integrate_spectral_profile(
    arc_lengths: &[f64],
    gradients: &[f64],
    gradient_errors: &[f64],
    kbt: f64,
) -> Result<FreeEnergyProfile, FluxError> {
    let n = arc_lengths.len();
    if n < 2 || gradients.len() != n || gradient_errors.len() != n {
        return Err(FluxError::Shape {
            expected: n,
            got: gradients.len().max(gradient_errors.len()),
            context: "PAFI spectral profile integration vectors",
        });
    }
    for &sigma in gradient_errors {
        if !sigma.is_finite() || sigma < 0.0 {
            return Err(FluxError::InvalidConfig(
                "PAFI gradient standard errors must be finite and non-negative",
            ));
        }
    }
    let m_basis = spline_second_derivative_basis(arc_lengths);
    let mut integration_weights = vec![vec![0.0_f64; n]; n];
    let mut free_energy = vec![0.0_f64; n];
    for i in 1..n {
        let ds = arc_lengths[i] - arc_lengths[i - 1];
        if !ds.is_finite() || ds <= 0.0 {
            return Err(FluxError::InvalidConfig(
                "PAFI arc lengths must strictly increase",
            ));
        }
        let prev = integration_weights[i - 1].clone();
        integration_weights[i] = prev;
        integration_weights[i][i - 1] += 0.5 * ds;
        integration_weights[i][i] += 0.5 * ds;
        let cubic_scale = ds * ds * ds / 24.0;
        let mut delta_i = 0.0_f64;
        let mut var_delta_i = 0.0_f64;
        for j in 0..n {
            let coeff = -cubic_scale * (m_basis[i - 1][j] + m_basis[i][j]);
            delta_i += coeff * gradients[j];
            var_delta_i += coeff * coeff * gradient_errors[j] * gradient_errors[j];
        }
        // Only apply the high-order spline correction when its statistical uncertainty
        // is strictly smaller than the trapezoid statistical noise on the interval.
        let trap_var =
            0.25 * ds * ds * (gradient_errors[i - 1].powi(2) + gradient_errors[i].powi(2));
        let alpha = if var_delta_i == 0.0 {
            1.0
        } else if var_delta_i <= trap_var && (delta_i * delta_i + var_delta_i) > 0.0 {
            (delta_i * delta_i) / (delta_i * delta_i + var_delta_i)
        } else {
            0.0
        };
        for j in 0..n {
            integration_weights[i][j] -= alpha * cubic_scale * (m_basis[i - 1][j] + m_basis[i][j]);
        }
        let mut fi = 0.0_f64;
        for j in 0..n {
            fi += integration_weights[i][j] * gradients[j];
        }
        if !fi.is_finite() {
            return Err(FluxError::NonFinite("PAFI integrated free energy"));
        }
        free_energy[i] = fi;
    }

    let mut dividing = 0usize;
    for i in 1..n {
        if free_energy[i] > free_energy[dividing] {
            dividing = i;
        }
    }
    let barrier = free_energy[dividing];

    let mut standard_error = vec![0.0_f64; n];
    for i in 1..n {
        let mut variance = 0.0_f64;
        for (weight, sigma) in integration_weights[i].iter().zip(gradient_errors) {
            variance += weight * weight * sigma * sigma;
        }
        standard_error[i] = variance.sqrt();
    }

    let (reactant_log_partition, d_ln_zr_d_f) =
        reactant_log_partition_and_sensitivity(arc_lengths, &free_energy, dividing, kbt)?;
    let reactant_partition = reactant_log_partition.exp();
    let barrier_error = standard_error[dividing];
    let normalized_log_rate = 0.5 * (kbt / TAU).ln() - barrier / kbt - reactant_log_partition;
    if !normalized_log_rate.is_finite() {
        return Err(FluxError::InvalidConfig(
            "PAFI normalized log rate must be finite",
        ));
    }
    let normalized_rate = normalized_log_rate.exp();
    if !normalized_rate.is_finite() || normalized_rate < 0.0 {
        return Err(FluxError::InvalidConfig(
            "PAFI normalized rate must be finite",
        ));
    }
    let mut variance_ln_rate = 0.0_f64;
    for (gradient_index, sigma) in gradient_errors.iter().copied().enumerate() {
        let barrier_sensitivity = integration_weights[dividing][gradient_index];
        let mut partition_log_sensitivity = 0.0_f64;
        for profile_index in 0..=dividing {
            partition_log_sensitivity +=
                d_ln_zr_d_f[profile_index] * integration_weights[profile_index][gradient_index];
        }
        let log_rate_sensitivity = -barrier_sensitivity / kbt - partition_log_sensitivity;
        variance_ln_rate += log_rate_sensitivity * log_rate_sensitivity * sigma * sigma;
    }
    let sigma_ln_k = variance_ln_rate.sqrt();
    let normalized_rate_error = normalized_rate * sigma_ln_k;

    Ok(FreeEnergyProfile {
        arc_lengths: arc_lengths.to_vec(),
        free_energy,
        standard_error,
        barrier,
        barrier_error,
        dividing_index: dividing,
        reactant_partition,
        reactant_log_partition,
        normalized_rate,
        normalized_log_rate,
        normalized_rate_error,
    })
}

/// Fit the mean constraint force and free energy with a Chebyshev / cosine expansion
/// on `arc_lengths`, compute the certified tail bound on `F*`, and flag intervals
/// whose local error exceeds the uniform share when `tail_bound > target_se`.
pub fn estimate_spectral_tail(
    arc_lengths: &[f64],
    gradients: &[f64],
    target_se: f64,
) -> Result<SpectralTailReport, FluxError> {
    let n = arc_lengths.len();
    if n < 2 {
        return Err(FluxError::InvalidConfig(
            "estimate_spectral_tail requires at least 2 planes",
        ));
    }
    if gradients.len() != n {
        return Err(FluxError::Shape {
            expected: n,
            got: gradients.len(),
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
    let slopes: Vec<f64> = (0..n)
        .map(|i| (0..n).map(|j| d_basis[i][j] * gradients[j]).sum())
        .collect();
    let p = n - 1;
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

    let tail_start = 3.max(p);
    let mut spectral_tail_sum = 0.0_f64;
    for &cm in &profile_coefficients[tail_start..=(p + 1)] {
        spectral_tail_sum += cm.abs();
    }

    let (_, trap_barrier, _) = integrate_means(arc_lengths, gradients)?;
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
        let hermite_quad_diff = (ds * ds / 12.0) * (slopes[k] - slopes[k + 1]).abs();
        let trap_inc = 0.5 * (gradients[k] + gradients[k + 1]) * ds;
        let fa = chebyshev_eval(&profile_coefficients, to_x(sa));
        let fb = chebyshev_eval(&profile_coefficients, to_x(sb));
        let spec_inc = fb - fa;
        let quad_diff = hermite_quad_diff.max((spec_inc - trap_inc).abs());
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
        // Weight the global Chebyshev tail across intervals by local force variation |dg| * ds.
        let local_variation = (gradients[k + 1] - gradients[k]).abs() * ds / total_length;
        let local_tail_var = spectral_tail_sum * local_variation;
        max_peak_gap = max_peak_gap.max(peak_gap);
        interval_errors[k] = quad_diff + peak_gap + local_tail_var;
    }

    let raw_tail_bound =
        (spec_barrier - trap_barrier).abs().max(total_quad_diff) + spectral_tail_sum + max_peak_gap;
    let scale = 1.0 + trap_barrier.abs().max(spec_barrier.abs());
    let rounding_floor = 64.0 * f64::EPSILON * scale;
    let tail_bound = if raw_tail_bound <= rounding_floor {
        0.0
    } else {
        raw_tail_bound
    };

    let mut flagged_intervals = Vec::new();
    if tail_bound > target_se.max(rounding_floor) {
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
        tail_bound,
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
    if !config.base.zero_temperature_tol.is_finite() || config.base.zero_temperature_tol < 0.0 {
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
                let profile = integrate_spectral_profile(&arcs, &gradients, &errors, kbt)?;
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
                            )?;
                            force_calls += 1 + config.base.burn_in_steps + config.base.sample_steps;
                            record.samples.extend(samples);
                            record.rounds_completed += 1;
                        }
                    }
                    continue;
                }

                let stop_threshold = if profile.barrier_error > 0.0 {
                    match config.base.target_error {
                        Some(target) => profile.barrier_error.min(target),
                        None => profile.barrier_error,
                    }
                } else {
                    config
                        .base
                        .target_error
                        .unwrap_or(64.0 * f64::EPSILON * (1.0 + profile.barrier.abs()))
                };

                let tail = estimate_spectral_tail(&arcs, &gradients, stop_threshold)?;
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

                if tail.tail_bound <= stop_threshold
                    || refinement_rounds >= config.max_refinements
                    || records.len() >= config.max_planes
                    || tail.flagged_intervals.is_empty()
                {
                    return Ok(AdaptiveFreeEnergyProfile {
                        profile,
                        tail,
                        equidistribution: current_map,
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
                    let s_candidate = if gradients[k] > 0.0 && gradients[k + 1] < 0.0 {
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
                        profile,
                        tail,
                        equidistribution: current_map,
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

    struct InconsistentLinearGradient;

    impl rgsaddle::PointSurface for InconsistentLinearGradient {
        fn eval(
            &self,
            x: ArrayView1<'_, f64>,
        ) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
            Ok((x[0], Array1::from_vec(vec![0.98])))
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

    #[test]
    fn profile_uses_interior_maximum_and_accepts_exact_errors() {
        let arcs = [0.0, 1.0, 2.0, 3.0];
        let gradients = [2.0, 0.0, -2.0, -2.0];
        let errors = [0.0; 4];
        let profile = integrate_profile(&arcs, &gradients, &errors, 1.0).expect("profile");
        assert_eq!(profile.dividing_index, 1);
        assert_eq!(profile.barrier, 1.0);
        assert_eq!(profile.barrier_error, 0.0);
        assert_eq!(profile.normalized_rate_error, 0.0);
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
    fn reactant_log_partition_stays_finite_for_deep_free_energy_wells() {
        let arcs = [0.0, 1.0, 2.0, 3.0];
        let gradients = [-800.0, -800.0, 800.0, 820.0];
        let errors = [0.0; 4];
        let profile = integrate_profile(&arcs, &gradients, &errors, 1.0).expect("profile");

        let mut naive_half_partition = 0.0_f64;
        for i in 1..=profile.dividing_index {
            let ds = arcs[i] - arcs[i - 1];
            let left = (-profile.free_energy[i - 1]).exp();
            let right = (-profile.free_energy[i]).exp();
            naive_half_partition += 0.5 * ds * (left + right);
        }
        assert!(!naive_half_partition.is_finite());

        let expected_ln_partition = 800.0 + (4.0 + (-800.0_f64).exp() + (-810.0_f64).exp()).ln();
        assert_abs_diff_eq!(
            profile.reactant_log_partition,
            expected_ln_partition,
            epsilon = f64::EPSILON * expected_ln_partition.abs()
        );
        assert!(profile.normalized_log_rate.is_finite());
    }

    #[test]
    fn log_partition_matches_legacy_path_for_well_conditioned_profile() {
        let arcs = [0.0, 0.5, 1.0, 1.5];
        let gradients = [0.3, 0.1, -0.4, -0.6];
        let errors = [0.02, 0.03, 0.01, 0.02];
        let profile = integrate_profile(&arcs, &gradients, &errors, 1.2).expect("profile");

        let mut legacy_half_partition = 0.0_f64;
        for i in 1..=profile.dividing_index {
            let ds = arcs[i] - arcs[i - 1];
            let left = (-profile.free_energy[i - 1] / 1.2).exp();
            let right = (-profile.free_energy[i] / 1.2).exp();
            legacy_half_partition += 0.5 * ds * (left + right);
        }
        let legacy_partition = 2.0 * legacy_half_partition;
        let legacy_rate = (1.2 / TAU).sqrt() * (-profile.barrier / 1.2).exp() / legacy_partition;

        assert_abs_diff_eq!(
            profile.reactant_partition,
            legacy_partition,
            epsilon = f64::EPSILON * legacy_partition.abs()
        );
        assert_abs_diff_eq!(
            profile.normalized_rate,
            legacy_rate,
            epsilon = f64::EPSILON * legacy_rate.abs()
        );
        assert_abs_diff_eq!(
            profile.reactant_log_partition,
            legacy_partition.ln(),
            epsilon = 16.0 * f64::EPSILON * legacy_partition.ln().abs().max(1.0)
        );
    }

    #[test]
    fn non_finite_profile_input_is_rejected() {
        let arcs = [0.0, 1.0, 2.0];
        let gradients = [0.0, f64::NAN, 1.0];
        let errors = [0.0, 0.0, 0.0];
        let result = integrate_profile(&arcs, &gradients, &errors, 1.0);
        assert_eq!(
            result,
            Err(FluxError::NonFinite("PAFI integrated free energy"))
        );
    }

    #[test]
    fn zero_width_reactant_partition_is_rejected() {
        let arcs = [0.0, 1.0, 2.0];
        let gradients = [-1.0, -1.0, -1.0];
        let errors = [0.0, 0.0, 0.0];
        let result = integrate_profile(&arcs, &gradients, &errors, 1.0);
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
            InconsistentLinearGradient,
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
            InconsistentLinearGradient,
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
}
