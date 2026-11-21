//! OH-TST with a fixed hyperplane normal.
//!
//! [`optimize_hyperplane`] holds the seed normal fixed and places the offset on
//! the dividing surface that is stationary along that normal. The reported
//! rate is the harmonic hyperplanar TST rate
//!
//! `k = sqrt(κ / μ) / 2π * exp(-ΔV / kT)`,
//!
//! where `κ` is the reactant curvature inferred from the sampled constraint
//! force, `μ = 1 / (nᵀ M⁻¹ n)` is the flux mass, and `ΔV` is the potential
//! difference from the reactant minimum to the dividing surface. On a harmonic
//! reactant this is the closed-form harmonic TST rate of Vineyard (1957). The
//! constraint force is the mean Lagrange multiplier `⟨λ⟩ = dA/dc` of Sprik and
//! Ciccotti (1998), which is `κ (c - s_min)` on a harmonic reactant whatever
//! the sign of `c - s_min`. The standard error is the first-order delta-method
//! image of the curvature sample's batch-means error: `κ` is proportional to
//! `⟨λ⟩` and the rate to `sqrt(κ)`, so `σ_ln k = σ_λ / (2 |⟨λ⟩|)`.
//!
//! ## Theory and workflow
//!
//! Given a unit seed hyperplane `sigma(x) = n · x - c_0 = 0`, [`optimize_hyperplane`]
//! executes four stages:
//!
//! 1. **Reactant relaxation**: Clears any active constraint and relaxes the
//!    engine's current positions with [`rgmin::Lbfgs`] (the eOn step capped at
//!    `max_translation` per atom, up to `8 * max_outer_steps` iterations)
//!    until `max_i |F_i| <= reactant_fmax`. A start already within the
//!    tolerance is not moved. The minimum must also have a directional slope
//!    `-n · F` that rises across `x_min` (`s_min = n · x_min`,
//!    `V_min = V(x_min)`); stationary maxima or plateaus along `n` are escaped
//!    by stepping `-max_translation * n` and relaxing again, up to
//!    `8 * max_outer_steps` times. A relaxation that ends above the tolerance
//!    returns [`FluxError::ReactantNotRelaxed`].
//! 2. **Dividing-surface placement**: If the seed point `x_min + (c_0 - s_min) n`
//!    already has vanishing normal slope (`|-n · F| <= SERIES_FLAT_TOL`) and
//!    `V(x_seed) > V_min`, `c^* = c_0` is kept. Otherwise, the offset walks from
//!    `s_min` in direction `sign(c_0 - s_min)` (`+1` when `c_0 >= s_min`) in
//!    steps of `max_translation` (up to `max_outer_steps` steps) to bracket the
//!    first sign change of the normal slope, then bisects `[lo, hi]` to
//!    `SERIES_FLAT_TOL * (1 + hi)` to locate `c^*` with barrier
//!    `ΔV = V(x_min + (c^* - s_min) n) - V_min > 0`.
//! 3. **Curvature probe sampling**: Places a parallel probe plane at distance
//!    `δ = min(max_translation, 0.5 * |c^* - s_min|)` from `s_min` toward `c^*`
//!    (`c_probe = s_min + sign(c^* - s_min) * δ`), calls
//!    [`crate::hyperplane::sample_hyperplane`] for
//!    `max(max_outer_steps * sample_step_factor, min_sample_steps)` steps after
//!    `burn_in_steps`, and infers the reactant curvature
//!    `κ = ⟨λ⟩ / (c_probe - s_min) > 0`.
//! 4. **Harmonic rate and uncertainty**: Evaluates the flux mass
//!    `μ = (sum_i n_i^2 / m_i)^-1` via [`rgsaddle::flux_mass`] (keeping a length
//!    mismatch on [`FluxError::Shape`] and non-finite or non-positive masses on
//!    [`FluxError::InvalidConfig`]), angular frequency `ω = sqrt(κ / μ)`, rate
//!    `k = (ω / 2π) exp(-ΔV / k_B T)`, and delta-method standard error
//!    `σ_k = k * σ_λ / (2 |⟨λ⟩|)`.
//!
//! ## Assumptions
//!
//! - Single reaction coordinate along the fixed unit normal `n` (`||n||_2 = 1`);
//!   variational rotation of `n` on the unit sphere is not yet implemented.
//! - Positive barrier `ΔV > 0` above the reactant minimum and positive harmonic
//!   reactant curvature `κ > 0` at the probe offset `c_probe`.
//! - Strictly positive thermal energy `k_B * T > 0` and masses `m_i > 0`.
//!
//! ## Tolerances
//!
//! The reactant is a physical minimum: it is accepted once its largest force
//! component is at most [`OhtstConfig::reactant_fmax`]. Forces near a relaxed
//! minimum carry rounding noise that grows with the system size and the
//! coordinate magnitude, so a fixed multiple of machine epsilon is not a
//! reachable force criterion. [`SERIES_FLAT_TOL`] is kept where the
//! comparison is between binary64 numbers of known scale: the bisection
//! stops when the bracket is `SERIES_FLAT_TOL * (1 + hi)` wide, and a seed
//! is used verbatim only when its normal slope is zero to that precision.
//! Neither can fail on rounding; a seed with a larger slope is bisected.
//!
//! ## Units
//!
//! - `seed_plane.offset` and `OhtstConfig::max_translation` have length unit `[L]`.
//! - `OhtstConfig::max_rotation` is in radians (dimensionless).
//! - `OhtstConfig::time_step` has engine time unit `[T]`.
//! - `OhtstEstimate::rate` and `OhtstEstimate::standard_error` have inverse time
//!   units `[T]^-1` (`sqrt([E] / ([M] [L]^2))`).
//!
//! ## Engine wiring
//!
//! [`optimize_hyperplane`] accepts `&mut E` for any `E: DynamicsEngine + ?Sized`
//! (`NativeEngine` over `PointSurface`, `EindirSurface`, or `RgpotSurface`,
//! `LammpcEngine` under the `lammpc` feature, or `RgpotSessionEngine`). It
//! captures an `EngineSnapshot` on entry and restores `positions`, `velocities`,
//! `cell`, `thermostat`, and `constraint` before returning.
//!
//! ## Example
//!
//! ```rust
//! use ndarray::{Array1, ArrayView1};
//! use rgflux::engine::native::{NativeConfig, NativeEngine};
//! use rgflux::engine::{DynamicsEngine, HyperplaneConstraint, ThermostatConfig};
//! use rgflux::ohtst::{OhtstConfig, optimize_hyperplane};
//!
//! struct DoubleWell1D;
//! impl rgsaddle::PointSurface for DoubleWell1D {
//!     fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
//!         let u = x[0];
//!         let w = u * u - 1.0;
//!         Ok((0.25 * w * w, Array1::from_vec(vec![u * w])))
//!     }
//! }
//!
//! let cfg = NativeConfig::default()
//!     .with_masses(Array1::from_vec(vec![1.0]))
//!     .with_thermostat(ThermostatConfig::default().with_gamma(1.0).with_temperature(0.1))
//!     .with_seed(17);
//! let mut engine = NativeEngine::new(DoubleWell1D, cfg)?;
//! engine.set_positions(Array1::from_vec(vec![-1.0]).view())?;
//!
//! let seed = HyperplaneConstraint::from_direction(Array1::from_vec(vec![1.0]).view(), 0.0)?;
//! let ohtst_cfg = OhtstConfig::default()
//!     .with_burn_in_steps(32)
//!     .with_min_sample_steps(64)
//!     .with_sample_step_factor(2)
//!     .with_seed(17);
//! let est = optimize_hyperplane(&mut engine, &seed, &ohtst_cfg)?;
//! assert_eq!(est.plane.offset, 0.0);
//! assert!(est.rate > 0.0 && est.rate.is_finite());
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
//! - Goswami, R., Gunde, M., and Jónsson, H. (2026). *Enhanced climbing image
//!   nudged elastic band method with Hessian eigenmode alignment*. Frontiers
//!   in Chemistry, 14, 1807063. <https://doi.org/10.3389/fchem.2026.1807063>
//! - Jóhannesson, G. H., and Jónsson, H. (2001). *Optimization of hyperplanar
//!   transition states*. The Journal of Chemical Physics, 115, 9644-9656.
//!   <https://doi.org/10.1063/1.1415499>
//! - Bligaard, T., and Jónsson, H. (2005). *Optimization of hyperplanar
//!   transition states: Application to 2D test problems*. Computer Physics
//!   Communications, 169, 284-288. <https://doi.org/10.1016/j.cpc.2005.03.064>
//! - Sprik, M., and Ciccotti, G. (1998). *Free energy from constrained
//!   molecular dynamics*. The Journal of Chemical Physics, 109, 7737-7744.
//!   <https://doi.org/10.1063/1.477419>
//! - Vineyard, G. H. (1957). *Frequency factors and isotope effects in solid
//!   state rate processes*. Journal of Physics and Chemistry of Solids, 3,
//!   121-127. <https://doi.org/10.1016/0022-3697(57)90059-8>
//! - Bennett, C. H. (1976). *Efficient estimation of free energy differences
//!   from Monte Carlo data*. Journal of Computational Physics, 22, 245-268.
//!   <https://doi.org/10.1016/0021-9991(76)90078-4>
//! - Shirts, M. R., and Chodera, J. D. (2008). *Statistically optimal analysis
//!   of samples from multiple equilibrium states*. The Journal of Chemical
//!   Physics, 129, 124105. <https://doi.org/10.1063/1.2978177>

use std::f64::consts::TAU;

use ndarray::{Array1, ArrayView1};
use rgmin::vecops::dot;
use rgmin::{GradNorm, Lbfgs};
use rgsaddle::flux_mass as saddle_flux_mass;

use crate::engine::{DynamicsEngine, EngineSnapshot, HyperplaneConstraint};
use crate::error::FluxError;
use crate::hyperplane::{HyperplaneConfig, sample_hyperplane};
use crate::stats::{SERIES_FLAT_TOL, StatsConfig};

/// Configuration for fixed-normal hyperplane placement and OH-TST rate evaluation.
///
/// [`optimize_hyperplane`] reads [`Self::max_outer_steps`],
/// [`Self::max_translation`], [`Self::seed`], [`Self::stats`],
/// [`Self::reactant_fmax`] and the curvature-sample fields from
/// [`Self::burn_in_steps`] to [`Self::time_step`], and checks
/// [`Self::max_rotation`]. [`Self::patience`], [`Self::min_overlap`] and
/// [`Self::use_mbar`] belong to the variational normal optimization and are
/// not read by the fixed-normal evaluation.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct OhtstConfig {
    /// Maximum Riemannian trust-region / bracketing outer iterations (`>= 1`).
    ///
    /// Default: `32`.
    pub max_outer_steps: usize,
    /// Consecutive accepted planes within gradient noise required to stop
    /// (reserved for variational normal optimization).
    ///
    /// Default: `3`.
    pub patience: usize,
    /// Trust-region cap on normal translation per step and curvature probe
    /// step, and the per-atom displacement cap of the reactant L-BFGS step
    /// (`> 0`).
    ///
    /// Default: `0.25`.
    pub max_translation: f64,
    /// Trust-region cap on normal rotation angle (radians) per step (`>= 0`).
    ///
    /// Default: `0.25`.
    pub max_rotation: f64,
    /// Minimum two-state BAR overlap before emitting a low-overlap warning
    /// (reserved for variational normal optimization).
    ///
    /// Default: `0.03`.
    pub min_overlap: f64,
    /// Combine reversible work across all plane nodes with MBAR when `true`
    /// (reserved for variational normal optimization).
    ///
    /// Default: `true`.
    pub use_mbar: bool,
    /// Deterministic PRNG seed.
    ///
    /// Default: `1`.
    pub seed: u64,
    /// Batch-means uncertainty settings.
    ///
    /// Default: [`StatsConfig::default()`].
    pub stats: StatsConfig,
    /// Equilibration steps discarded before the curvature sample.
    ///
    /// Default: `256`.
    pub burn_in_steps: usize,
    /// Lower bound on the curvature-sample length (`>= 2`).
    ///
    /// The length used is the greater of this bound and
    /// [`Self::max_outer_steps`] times [`Self::sample_step_factor`].
    /// Default: `128`.
    pub min_sample_steps: usize,
    /// Multiplier on [`Self::max_outer_steps`] when sizing the curvature sample.
    ///
    /// Default: `64`, so a default optimization samples at least `2048` steps.
    pub sample_step_factor: usize,
    /// Steps between retained configurations in the curvature sample (`>= 1`).
    ///
    /// Default: `16`.
    pub retention_stride: usize,
    /// Integration timestep of the curvature sample, in the engine time unit (`> 0`).
    ///
    /// Default: `0.01`.
    pub time_step: f64,
    /// Physical force tolerance on the reactant minimum, in the engine force
    /// unit (eV/Å in metal units; finite and `> 0`).
    ///
    /// The reactant is a minimum once its largest absolute force component
    /// `max_i |F_i|` is at most this value. The absolute separation from the
    /// exact minimum is then about `reactant_fmax / κ`, so the default moves
    /// the curvature `κ` inferred over a probe of length `δ` by a relative
    /// `reactant_fmax / (κ δ)` and the barrier by about
    /// `reactant_fmax² / (2 κ)`.
    ///
    /// Default: `1e-6`, the gradient tolerance of [`rgmin::Lbfgs::default`]
    /// on the same infinity norm.
    pub reactant_fmax: f64,
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
            burn_in_steps: 256,
            min_sample_steps: 128,
            sample_step_factor: 64,
            retention_stride: 16,
            time_step: 0.01,
            reactant_fmax: 1.0e-6,
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
    with_burn_in_steps => burn_in_steps: val usize,
    with_min_sample_steps => min_sample_steps: val usize,
    with_sample_step_factor => sample_step_factor: val usize,
    with_retention_stride => retention_stride: val usize,
    with_time_step => time_step: val f64,
    with_reactant_fmax => reactant_fmax: val f64,
});

/// Fixed-normal OH-TST rate estimate and dividing hyperplane.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct OhtstEstimate {
    /// Optimized dividing hyperplane.
    pub plane: HyperplaneConstraint,
    /// Fixed-normal hyperplanar transition-state rate constant `k_OHTST`.
    pub rate: f64,
    /// Propagated standard error of [`Self::rate`].
    pub standard_error: f64,
}

/// Optimize the offset of `seed_plane` and return the harmonic OH-TST rate.
///
/// The normal is held at the seed. A seed whose normal force already vanishes,
/// and whose potential lies above the reactant minimum, is the dividing
/// surface. Otherwise the offset walks along the normal to the first
/// normal-force zero. The reactant curvature is `κ = ⟨λ⟩ / δ` from a
/// constrained sample a distance `δ` from the minimum toward that surface.
///
/// # Errors
///
/// - [`FluxError::InvalidConfig`] when a [`OhtstConfig`] field violates its
///   bounds, thermal energy `k_B * T <= 0`, the flux-mass metric from
///   [`rgsaddle::flux_mass`] is non-finite or non-positive, no reactant
///   minimum along `seed_plane.normal` is found within the escape budget, no
///   stationary dividing surface with positive barrier is found along
///   `seed_plane.normal`, or the sampled reactant curvature `κ <= 0`.
/// - [`FluxError::ReactantNotRelaxed`] when a reactant relaxation stops, on
///   its iteration budget or otherwise, with `max_i |F_i| > reactant_fmax`.
/// - [`FluxError::Shape`] when `seed_plane.normal.len()` does not match
///   `engine.dof()` or `engine.masses().len()`.
/// - [`FluxError::NonFinite`] when the evaluated rate or standard error
///   overflows or underflows to zero.
/// - Any [`FluxError`] propagated by `engine` or [`sample_hyperplane`].
pub fn optimize_hyperplane<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    seed_plane: &HyperplaneConstraint,
    config: &OhtstConfig,
) -> Result<OhtstEstimate, FluxError> {
    validate_config(config)?;
    seed_plane.validate()?;
    if seed_plane.normal.len() != engine.dof() {
        return Err(FluxError::Shape {
            expected: engine.dof(),
            got: seed_plane.normal.len(),
            context: "OH-TST seed plane",
        });
    }
    let snapshot = EngineSnapshot::capture(engine);
    let result = evaluate_harmonic_rate(engine, seed_plane, config);
    snapshot.restore_result(engine, result)
}

fn validate_config(config: &OhtstConfig) -> Result<(), FluxError> {
    if config.max_outer_steps == 0 {
        return Err(FluxError::InvalidConfig(
            "OhtstConfig::max_outer_steps must be at least 1",
        ));
    }
    if !config.max_translation.is_finite() || config.max_translation <= 0.0 {
        return Err(FluxError::InvalidConfig(
            "OhtstConfig::max_translation must be finite and positive",
        ));
    }
    if !config.max_rotation.is_finite() || config.max_rotation < 0.0 {
        return Err(FluxError::InvalidConfig(
            "OhtstConfig::max_rotation must be finite and non-negative",
        ));
    }
    if config.min_sample_steps < 2 {
        return Err(FluxError::InvalidConfig(
            "OhtstConfig::min_sample_steps must be at least 2",
        ));
    }
    if config.retention_stride == 0 {
        return Err(FluxError::InvalidConfig(
            "OhtstConfig::retention_stride must be at least 1",
        ));
    }
    if !config.time_step.is_finite() || config.time_step <= 0.0 {
        return Err(FluxError::InvalidConfig(
            "OhtstConfig::time_step must be finite and positive",
        ));
    }
    if !config.reactant_fmax.is_finite() || config.reactant_fmax <= 0.0 {
        return Err(FluxError::InvalidConfig(
            "OhtstConfig::reactant_fmax must be finite and positive",
        ));
    }
    Ok(())
}

fn evaluate_harmonic_rate<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    seed_plane: &HyperplaneConstraint,
    config: &OhtstConfig,
) -> Result<OhtstEstimate, FluxError> {
    let kt = engine.thermostat().thermal_energy()?;
    if kt <= 0.0 {
        return Err(FluxError::InvalidConfig(
            "OH-TST requires a positive thermal energy k_B * T",
        ));
    }
    let mu = flux_mass(seed_plane.normal.view(), engine.masses())?;
    engine.set_constraint(None)?;

    let minimum = reactant_minimum(engine, seed_plane, config)?;
    engine.set_positions(minimum.view())?;
    let (energy_min, _) = engine.energy_and_forces()?;

    let dividing = dividing_surface(engine, &minimum, energy_min, seed_plane, config)?;
    engine.set_positions(dividing.point.view())?;
    let (energy_dividing, _) = engine.energy_and_forces()?;
    let barrier = energy_dividing - energy_min;
    if !(barrier.is_finite() && barrier > 0.0) {
        return Err(FluxError::InvalidConfig(
            "OH-TST dividing surface must lie above the reactant minimum",
        ));
    }

    let s_min = dot(seed_plane.normal.view(), minimum.view());
    let signed_separation = dividing.offset - s_min;
    let separation = signed_separation.abs();
    let probe = config.max_translation.min(0.5 * separation);
    if !(probe.is_finite() && probe > 0.0) {
        return Err(FluxError::InvalidConfig(
            "OH-TST curvature probe length vanished",
        ));
    }
    let toward = signed_separation.signum();
    let probe_offset = s_min + toward * probe;
    let probe_point = shifted(minimum.view(), seed_plane.normal.view(), toward * probe);
    engine.set_positions(probe_point.view())?;
    let probe_plane = HyperplaneConstraint::from_direction(seed_plane.normal.view(), probe_offset)?;
    let sampled = sample_hyperplane(engine, &probe_plane, &probe_sampling(config))?;
    let lambda = sampled.mean_force;
    // The constraint force `λ = dA/dc` takes the sign of `c - s_min`, so the
    // signed offset from the minimum, as the constraint holds it in binary64,
    // makes `κ` positive on either side of the minimum.
    let kappa = lambda / (probe_offset - s_min);
    if !(kappa.is_finite() && kappa > 0.0) {
        return Err(FluxError::InvalidConfig(
            "OH-TST reactant curvature must be finite and positive",
        ));
    }

    let omega = (kappa / mu).sqrt();
    let rate = omega / TAU * (-barrier / kt).exp();
    if !(rate.is_finite() && rate > 0.0) {
        return Err(FluxError::NonFinite("OH-TST rate"));
    }
    let sigma_ln = 0.5 * sampled.standard_error / lambda.abs();
    let standard_error = rate * sigma_ln;
    if !standard_error.is_finite() {
        return Err(FluxError::NonFinite("OH-TST rate error"));
    }
    Ok(OhtstEstimate {
        plane: dividing.plane,
        rate,
        standard_error,
    })
}

struct DividingSurface {
    plane: HyperplaneConstraint,
    point: Array1<f64>,
    offset: f64,
}

fn dividing_surface<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    minimum: &Array1<f64>,
    energy_min: f64,
    seed_plane: &HyperplaneConstraint,
    config: &OhtstConfig,
) -> Result<DividingSurface, FluxError> {
    let normal = seed_plane.normal.view();
    let s_min = dot(normal, minimum.view());
    let seed_point = shifted(minimum.view(), normal, seed_plane.offset - s_min);
    let seed_slope = normal_slope(engine, seed_point.view(), normal)?;
    let (seed_energy, _) = {
        engine.set_positions(seed_point.view())?;
        engine.energy_and_forces()?
    };
    if slope_vanishes(seed_slope) && seed_energy > energy_min {
        return Ok(DividingSurface {
            plane: seed_plane.clone(),
            point: seed_point,
            offset: seed_plane.offset,
        });
    }

    let direction = if seed_plane.offset >= s_min {
        1.0
    } else {
        -1.0
    };
    let mut hi = config.max_translation;
    let mut crossed = false;
    for _ in 0..config.max_outer_steps {
        let point = shifted(minimum.view(), normal, direction * hi);
        let slope = direction * normal_slope(engine, point.view(), normal)?;
        if slope <= 0.0 {
            crossed = true;
            break;
        }
        hi += config.max_translation;
    }
    if !crossed {
        return Err(FluxError::InvalidConfig(
            "OH-TST found no stationary dividing surface along the seed normal",
        ));
    }
    let mut lo = hi - config.max_translation;
    let tol = SERIES_FLAT_TOL * (1.0 + hi);
    while hi - lo > tol {
        let mid = 0.5 * (lo + hi);
        let point = shifted(minimum.view(), normal, direction * mid);
        let slope = direction * normal_slope(engine, point.view(), normal)?;
        if slope > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let offset = s_min + direction * hi;
    let point = shifted(minimum.view(), normal, direction * hi);
    let plane = HyperplaneConstraint::from_direction(normal, offset)?;
    Ok(DividingSurface {
        plane,
        point,
        offset,
    })
}

fn reactant_minimum<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    seed_plane: &HyperplaneConstraint,
    config: &OhtstConfig,
) -> Result<Array1<f64>, FluxError> {
    let normal = seed_plane.normal.view();
    for _ in 0..minimization_budget(config) {
        relax(engine, config)?;
        let position = engine.positions().to_owned();
        if slope_rises_through(engine, position.view(), normal, config.max_translation)? {
            return Ok(position);
        }
        displace(engine, normal, -config.max_translation)?;
    }
    Err(FluxError::InvalidConfig(
        "OH-TST reactant minimization exceeded its iteration budget",
    ))
}

/// Iterations of one descent, and descents of one reactant search.
fn minimization_budget(config: &OhtstConfig) -> usize {
    // Allow several minimization updates per dividing-surface search iteration.
    const MINIMIZATION_BUDGET_FACTOR: usize = 8;
    config
        .max_outer_steps
        .saturating_mul(MINIMIZATION_BUDGET_FACTOR)
}

/// Relax the engine positions with the rgmin L-BFGS solver until
/// `max_i |F_i| <= reactant_fmax`, leave the engine there, and return that
/// force. A start already within the tolerance is not moved.
///
/// The eOn step of [`Lbfgs`] takes no line search and compares no energies,
/// so rounding in the energy near the minimum cannot stall it.
fn relax<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    config: &OhtstConfig,
) -> Result<f64, FluxError> {
    let (_, force) = engine.energy_and_forces()?;
    let initial = max_force(force.view());
    if initial <= config.reactant_fmax {
        return Ok(initial);
    }
    let start = engine.positions().to_owned();
    let mut solver = Lbfgs::builder()
        .capped(config.max_translation)
        .tolerance(config.reactant_fmax)
        .norm(GradNorm::Infinity)
        .build();
    let mut failure = None;
    let (_, relaxed, _) =
        solver.minimize(
            start.view(),
            minimization_budget(config),
            |x| match energy_and_gradient(engine, x) {
                Ok(value) => Some(value),
                Err(err) => {
                    failure = Some(err);
                    None
                }
            },
        );
    if let Some(err) = failure {
        return Err(err);
    }
    engine.set_positions(relaxed.view())?;
    let (_, force) = engine.energy_and_forces()?;
    let fmax = max_force(force.view());
    if fmax <= config.reactant_fmax {
        Ok(fmax)
    } else {
        Err(FluxError::ReactantNotRelaxed {
            fmax,
            tolerance: config.reactant_fmax,
        })
    }
}

fn energy_and_gradient<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    positions: ArrayView1<'_, f64>,
) -> Result<(f64, Array1<f64>), FluxError> {
    engine.set_positions(positions)?;
    let (energy, force) = engine.energy_and_forces()?;
    Ok((energy, force.mapv_into(|component| -component)))
}

/// Whether the slope along `normal` rises through `point`, so that `point` is
/// a minimum along the normal and not a maximum or a plateau.
///
/// The curvature `(ahead - behind) / (2 step)` is positive exactly when the
/// slope `step` ahead exceeds the slope `step` behind, because `step > 0`.
fn slope_rises_through<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    point: ArrayView1<'_, f64>,
    normal: ArrayView1<'_, f64>,
    step: f64,
) -> Result<bool, FluxError> {
    let ahead = normal_slope(engine, shifted(point, normal, step).view(), normal)?;
    let behind = normal_slope(engine, shifted(point, normal, -step).view(), normal)?;
    engine.set_positions(point)?;
    Ok(ahead > behind)
}

fn normal_slope<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    point: ArrayView1<'_, f64>,
    normal: ArrayView1<'_, f64>,
) -> Result<f64, FluxError> {
    engine.set_positions(point)?;
    let (_, force) = engine.energy_and_forces()?;
    Ok(-dot(normal, force.view()))
}

fn displace<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    normal: ArrayView1<'_, f64>,
    distance: f64,
) -> Result<(), FluxError> {
    let next = shifted(engine.positions(), normal, distance);
    engine.set_positions(next.view())
}

fn shifted(origin: ArrayView1<'_, f64>, direction: ArrayView1<'_, f64>, scale: f64) -> Array1<f64> {
    let mut out = origin.to_owned();
    for (slot, component) in out.iter_mut().zip(direction.iter()) {
        *slot += scale * *component;
    }
    out
}

/// Largest absolute force component, or infinity when any component is not
/// finite, so that no tolerance accepts a broken force.
fn max_force(force: ArrayView1<'_, f64>) -> f64 {
    force
        .iter()
        .try_fold(0.0_f64, |peak, component| {
            component.is_finite().then(|| peak.max(component.abs()))
        })
        .unwrap_or(f64::INFINITY)
}

fn slope_vanishes(slope: f64) -> bool {
    slope.abs() <= SERIES_FLAT_TOL
}

fn flux_mass(normal: ArrayView1<'_, f64>, masses: ArrayView1<'_, f64>) -> Result<f64, FluxError> {
    match saddle_flux_mass(normal, masses) {
        Ok(mass) => Ok(mass),
        Err(rgsaddle::SaddleError::Shape(_)) if normal.len() != masses.len() => {
            Err(FluxError::Shape {
                expected: masses.len(),
                got: normal.len(),
                context: "OH-TST flux mass",
            })
        }
        Err(_) => Err(FluxError::InvalidConfig(
            "OH-TST flux mass metric must be finite and positive",
        )),
    }
}

fn curvature_sample_steps(config: &OhtstConfig) -> usize {
    config
        .max_outer_steps
        .saturating_mul(config.sample_step_factor)
        .max(config.min_sample_steps)
}

fn probe_sampling(config: &OhtstConfig) -> HyperplaneConfig {
    HyperplaneConfig::default()
        .with_burn_in_steps(config.burn_in_steps)
        .with_sample_steps(curvature_sample_steps(config))
        .with_retention_stride(config.retention_stride)
        .with_time_step(config.time_step)
        .with_seed(config.seed)
        .with_stats(config.stats.clone())
}
