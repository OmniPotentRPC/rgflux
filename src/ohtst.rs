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
//! - Sprik, M., and Ciccotti, G. (1998). *Free energy from constrained
//!   molecular dynamics*. The Journal of Chemical Physics, 109, 7737–7744.
//!   <https://doi.org/10.1063/1.477419>
//! - Vineyard, G. H. (1957). *Frequency factors and isotope effects in solid
//!   state rate processes*. Journal of Physics and Chemistry of Solids, 3,
//!   121–127. <https://doi.org/10.1016/0022-3697(57)90059-8>
//! - Bennett, C. H. (1976). *Efficient estimation of free energy differences
//!   from Monte Carlo data*. Journal of Computational Physics, 22, 245–268.
//!   <https://doi.org/10.1016/0021-9991(76)90078-4>
//! - Shirts, M. R., and Chodera, J. D. (2008). *Statistically optimal analysis
//!   of samples from multiple equilibrium states*. The Journal of Chemical
//!   Physics, 129, 124105. <https://doi.org/10.1063/1.2978177>

use std::f64::consts::TAU;

use ndarray::{Array1, ArrayView1};
use rgmin::vecops::dot;
use rgsaddle::flux_mass as saddle_flux_mass;

use crate::engine::{DynamicsEngine, EngineSnapshot, HyperplaneConstraint};
use crate::error::FluxError;
use crate::hyperplane::{HyperplaneConfig, sample_hyperplane};
use crate::stats::{SERIES_FLAT_TOL, StatsConfig};

/// Configuration for fixed-normal hyperplane placement and OH-TST rate evaluation.
///
/// [`optimize_hyperplane`] reads [`Self::max_outer_steps`],
/// [`Self::max_translation`], [`Self::seed`], [`Self::stats`] and the
/// curvature-sample fields from [`Self::burn_in_steps`] on, and checks
/// [`Self::max_rotation`]. [`Self::patience`], [`Self::min_overlap`] and
/// [`Self::use_mbar`] belong to the variational normal optimization and are
/// not read by the fixed-normal evaluation.
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
    /// Equilibration steps discarded before the curvature sample.
    ///
    /// Default: 256.
    pub burn_in_steps: usize,
    /// Lower bound on the curvature-sample length.
    ///
    /// The length used is the greater of this bound and
    /// [`Self::max_outer_steps`] times [`Self::sample_step_factor`].
    /// Default: 128.
    pub min_sample_steps: usize,
    /// Multiplier on [`Self::max_outer_steps`] when sizing the curvature sample.
    ///
    /// Default: 64, so a default optimization samples at least 2048 steps.
    pub sample_step_factor: usize,
    /// Steps between retained configurations in the curvature sample.
    ///
    /// Default: 16.
    pub retention_stride: usize,
    /// Integration timestep of the curvature sample, in the engine time unit.
    ///
    /// Default: 0.01.
    pub time_step: f64,
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
    let mu = saddle_flux_mass(seed_plane.normal.view(), engine.masses())?;
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
    let probe = curvature_probe_step(separation, config);
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
    let normal = seed_plane.normal.clone();
    // Allow several minimization updates per dividing-surface search iteration.
    const MINIMIZATION_BUDGET_FACTOR: usize = 8;
    let limit = config
        .max_outer_steps
        .saturating_mul(MINIMIZATION_BUDGET_FACTOR);
    for _ in 0..limit {
        let position = engine.positions().to_owned();
        let (energy, force) = engine.energy_and_forces()?;
        if force_is_stationary(force.view()) {
            let confined = slope_rises_through(
                engine,
                position.view(),
                normal.view(),
                config.max_translation,
            )?;
            if confined {
                return Ok(engine.positions().to_owned());
            }
            displace(engine, normal.view(), -config.max_translation)?;
            continue;
        }
        if !descend_once(engine, energy, force.view())? {
            return Err(FluxError::InvalidConfig(
                "OH-TST reactant minimization stalled before the force vanished",
            ));
        }
    }
    Err(FluxError::InvalidConfig(
        "OH-TST reactant minimization exceeded its iteration budget",
    ))
}

fn descend_once<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    energy: f64,
    force: ArrayView1<'_, f64>,
) -> Result<bool, FluxError> {
    // Start with a full force-vector displacement before backtracking.
    const INITIAL_LINE_SEARCH_STEP: f64 = 1.0;
    // Twenty-four halvings span the useful binary64 step range for this search.
    const MAX_LINE_SEARCH_HALVINGS: usize = 24;

    let start = engine.positions().to_owned();
    let mut step = INITIAL_LINE_SEARCH_STEP;
    for _ in 0..MAX_LINE_SEARCH_HALVINGS {
        let trial = shifted(start.view(), force, step);
        engine.set_positions(trial.view())?;
        let (trial_energy, _) = engine.energy_and_forces()?;
        if trial_energy < energy {
            return Ok(true);
        }
        step *= 0.5;
    }
    engine.set_positions(start.view())?;
    Ok(false)
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

fn force_is_stationary(force: ArrayView1<'_, f64>) -> bool {
    let peak = force
        .iter()
        .fold(0.0_f64, |peak, component| peak.max(component.abs()));
    peak <= SERIES_FLAT_TOL
}

fn slope_vanishes(slope: f64) -> bool {
    slope.abs() <= SERIES_FLAT_TOL
}

fn curvature_probe_step(separation: f64, config: &OhtstConfig) -> f64 {
    f64::min(config.max_translation, 0.5 * separation)
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
