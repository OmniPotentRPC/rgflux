//! Bennett-Chandler reactive-flux transmission coefficient estimator and trajectory shooter.
//!
//! Given `N` reactive-flux trajectories initialized on a dividing surface with
//! normal velocities `v_{n,i}(0)` and side histories `theta_i(t) in {0, 1}`
//! (`1` on the product side, `0` on the reactant side), the Bennett-Chandler
//! correlation ratio is
//!
//! `kappa(t) = < v_n(0) * theta(s(t)) > / < v_n(0) * theta(v_n(0)) >`
//!
//! (Bennett 1977; Chandler 1978). [`estimate_bennett_chandler`] evaluates
//! `kappa(t)` at every recorded step, verifies stationarity of the configured
//! plateau window, and reports the plateau average alongside a block-averaged
//! standard error from [`crate::stats::batch_means`]. No clamping into `[0, 1]`
//! is performed.
//!
//! [`shoot_bennett_chandler`] generates reactive-flux trajectories generically
//! over any [`crate::engine::DynamicsEngine`] from configurations on a
//! [`crate::engine::HyperplaneConstraint`] and passes them to
//! [`estimate_bennett_chandler`]. Each half-trajectory stops once
//! [`KappaConfig::commit_checkpoints`] consecutive checkpoints lie at least
//! [`KappaConfig::commitment_distance`] into the same side, copying that side
//! into every remaining checkpoint of `kappa(t)`. This optimization is disabled
//! by default because a dividing-surface distance alone cannot prove that a
//! trajectory will not return.
//!
//! ## Example
//!
//! ```rust
//! use rgflux::kappa::bennett_chandler::{
//!     BennettChandlerConfig, ReactiveTrajectory, Side, estimate_bennett_chandler,
//! };
//!
//! let trajectories = vec![
//!     ReactiveTrajectory::new(1.4, vec![Side::Product; 16])?,
//!     ReactiveTrajectory::new(-0.9, vec![Side::Reactant; 16])?,
//!     ReactiveTrajectory::new(0.6, vec![Side::Product; 16])?,
//! ];
//! let est = estimate_bennett_chandler(&trajectories, &BennettChandlerConfig::default())?;
//! assert_eq!(est.plateau, 1.0);
//! assert_eq!(est.standard_error, 0.0);
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
//! - Bennett, C. H. (1977). *Molecular Dynamics and Transition State Theory:
//!   The Simulation of Infrequent Events*. In *Algorithms for Chemical
//!   Computations*, ACS Symposium Series, 46, 63-97.
//!   <https://doi.org/10.1021/bk-1977-0046.ch004>
//! - Chandler, D. (1978). *Statistical mechanics of isomerization dynamics in
//!   liquids and the transition state approximation*. The Journal of Chemical
//!   Physics, 68, 2959-2970. <https://doi.org/10.1063/1.436049>
//! - Kramers, H. A. (1940). *Brownian motion in a field of force and the
//!   diffusion model of chemical reactions*. Physica, 7, 284-304.
//!   <https://doi.org/10.1016/S0031-8914(40)90098-2>

use ndarray::ArrayView2;
use rgmin::vecops::dot;

use crate::engine::{DynamicsEngine, HyperplaneConstraint, validate_finite_slice};
use crate::error::FluxError;
pub use crate::kappa::{DEFAULT_COMMIT_CHECKPOINTS, KappaConfig};
use crate::stats::{
    MIN_STATIONARITY_SAMPLES, SERIES_FLAT_TOL, StationarityReport, StatsConfig, batch_means,
    check_stationarity,
};

/// Default start fraction of the trajectory length defining the plateau window.
pub const DEFAULT_PLATEAU_START_FRACTION: f64 = 0.5;

/// Default end fraction of the trajectory length defining the plateau window.
pub const DEFAULT_PLATEAU_END_FRACTION: f64 = 1.0;

/// Default integration steps per shot in [`ShootingConfig`].
pub const DEFAULT_SHOOTING_STEPS: usize = 200;

/// Default integration timestep in [`ShootingConfig`].
pub const DEFAULT_SHOOTING_TIME_STEP: f64 = 0.01;

/// SplitMix64 golden-ratio increment separating per-configuration PRNG seeds.
const SHOT_SEED_INCREMENT: u64 = 0x9E37_79B9_7F4A_7C15;

/// SplitMix64 mixing constant separating antithetic `(-v)` Langevin noise streams.
const ANTITHETIC_SEED_MIX: u64 = 0xBF58_476D_1CE4_E5B9;

/// Side of the dividing surface occupied by a trajectory at a given step.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Side {
    /// Reactant side (`s(t) <= 0`, indicator `theta = 0`).
    Reactant,
    /// Product side (`s(t) > 0`, indicator `theta = 1`).
    Product,
}

impl Side {
    /// Classify a signed reaction coordinate `s = n · x - c` (`s > 0` is [`Self::Product`]).
    #[must_use]
    pub fn from_signed_distance(signed_distance: f64) -> Self {
        if signed_distance > 0.0 {
            Self::Product
        } else {
            Self::Reactant
        }
    }

    /// Classify a signed reaction coordinate `s = n · x - c` into a committed side
    /// (`s < 0` is `Some(Self::Reactant)`, `s > 0` is `Some(Self::Product)`, and
    /// `s == 0` on the dividing surface is `None`).
    #[must_use]
    pub fn committed_from_signed_distance(signed_distance: f64) -> Option<Self> {
        if signed_distance < 0.0 {
            Some(Self::Reactant)
        } else if signed_distance > 0.0 {
            Some(Self::Product)
        } else {
            None
        }
    }

    /// Heaviside indicator `theta(s)` (`1.0` on [`Self::Product`], `0.0` on [`Self::Reactant`]).
    #[must_use]
    pub const fn indicator(self) -> f64 {
        match self {
            Self::Reactant => 0.0,
            Self::Product => 1.0,
        }
    }
}

/// Recorded reactive-flux trajectory launched from the dividing surface.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct ReactiveTrajectory {
    /// Initial velocity component normal to the dividing surface, `v_n(0) = n · v(0)`.
    pub initial_normal_velocity: f64,
    /// Dividing-surface side occupied at each recorded post-launch time index `t = 0..T`.
    pub sides: Vec<Side>,
}

impl ReactiveTrajectory {
    /// Construct a validated [`ReactiveTrajectory`].
    pub fn new(initial_normal_velocity: f64, sides: Vec<Side>) -> Result<Self, FluxError> {
        if !initial_normal_velocity.is_finite() {
            return Err(FluxError::NonFinite(
                "ReactiveTrajectory::initial_normal_velocity",
            ));
        }
        if sides.is_empty() {
            return Err(FluxError::EmptyInput(
                "ReactiveTrajectory::sides must not be empty",
            ));
        }
        Ok(Self {
            initial_normal_velocity,
            sides,
        })
    }
}

/// Configuration for the engine-free Bennett-Chandler plateau estimator.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct BennettChandlerConfig {
    /// Inclusive start fraction `[0, 1)` of the trajectory length for the plateau window.
    ///
    /// Default: [`DEFAULT_PLATEAU_START_FRACTION`] (`0.5`).
    pub plateau_start_fraction: f64,
    /// Exclusive end fraction `(0, 1]` of the trajectory length for the plateau window.
    ///
    /// Default: [`DEFAULT_PLATEAU_END_FRACTION`] (`1.0`).
    pub plateau_end_fraction: f64,
    /// Batch-means and stationarity configuration applied to the plateau window.
    ///
    /// Default: [`StatsConfig::default()`].
    pub stats: StatsConfig,
}

impl Default for BennettChandlerConfig {
    fn default() -> Self {
        Self {
            plateau_start_fraction: DEFAULT_PLATEAU_START_FRACTION,
            plateau_end_fraction: DEFAULT_PLATEAU_END_FRACTION,
            stats: StatsConfig::default(),
        }
    }
}

with_setters!(BennettChandlerConfig {
    with_plateau_start_fraction => plateau_start_fraction: val f64,
    with_plateau_end_fraction => plateau_end_fraction: val f64,
    with_stats => stats: val StatsConfig,
});

/// Configuration for shooting reactive-flux trajectories through a [`DynamicsEngine`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct ShootingConfig {
    /// Number of integration steps per trajectory (`>= 4`).
    ///
    /// Default: [`DEFAULT_SHOOTING_STEPS`] (`200`).
    pub steps: usize,
    /// Integration timestep `dt > 0`.
    ///
    /// Default: [`DEFAULT_SHOOTING_TIME_STEP`] (`0.01`).
    pub time_step: f64,
    /// Shoot antithetic velocity pairs `(+v, -v)` from each dividing-surface configuration.
    ///
    /// Default: `true`.
    pub antithetic: bool,
    /// Deterministic base PRNG seed.
    ///
    /// Default: `1`.
    pub seed: u64,
    /// Plateau and uncertainty configuration passed to [`estimate_bennett_chandler`].
    ///
    /// Default: [`BennettChandlerConfig::default()`].
    pub estimator: BennettChandlerConfig,
    /// Basin-commitment configuration (`commit_checkpoints <= steps`; `0`
    /// disables early commitment).
    ///
    /// Default: [`KappaConfig::default()`].
    pub kappa: KappaConfig,
}

impl Default for ShootingConfig {
    fn default() -> Self {
        Self {
            steps: DEFAULT_SHOOTING_STEPS,
            time_step: DEFAULT_SHOOTING_TIME_STEP,
            antithetic: true,
            seed: 1,
            estimator: BennettChandlerConfig::default(),
            kappa: KappaConfig::default(),
        }
    }
}

with_setters!(ShootingConfig {
    with_steps => steps: val usize,
    with_time_step => time_step: val f64,
    with_antithetic => antithetic: val bool,
    with_seed => seed: val u64,
    with_estimator => estimator: val BennettChandlerConfig,
    with_kappa => kappa: val KappaConfig,
});

impl ShootingConfig {
    /// Set [`KappaConfig::commit_checkpoints`] on [`Self::kappa`].
    ///
    /// A nonzero value opts into early commitment and also requires a finite,
    /// positive [`KappaConfig::commitment_distance`].
    #[must_use]
    pub fn with_commit_checkpoints(mut self, commit_checkpoints: usize) -> Self {
        self.kappa.commit_checkpoints = commit_checkpoints;
        self
    }
}

/// Output of [`estimate_bennett_chandler`] and [`shoot_bennett_chandler`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct BennettChandlerEstimate {
    /// Time-dependent reactive-flux correlation `kappa(t)` at each recorded step.
    pub kappa_t: Vec<f64>,
    /// Unclamped mean of `kappa(t)` across [`Self::plateau_window`].
    pub plateau: f64,
    /// Block-averaged standard error of [`Self::plateau`] from [`crate::stats::batch_means`].
    pub standard_error: f64,
    /// Effective trajectory sample size `n_eff` from [`crate::stats::batch_means`].
    pub n_eff: f64,
    /// Half-open index range `(start, end)` of the plateau window in [`Self::kappa_t`].
    pub plateau_window: (usize, usize),
    /// Stationarity report over [`Self::plateau_window`].
    pub stationarity: StationarityReport,
    /// Number of trajectories with `initial_normal_velocity > 0`.
    pub positive_flux_count: usize,
}

fn resolve_plateau_window(
    n_steps: usize,
    config: &BennettChandlerConfig,
) -> Result<(usize, usize), FluxError> {
    if !config.plateau_start_fraction.is_finite() || !config.plateau_end_fraction.is_finite() {
        return Err(FluxError::NonFinite(
            "BennettChandlerConfig plateau fractions",
        ));
    }
    if config.plateau_start_fraction < 0.0
        || config.plateau_end_fraction > 1.0
        || config.plateau_start_fraction >= config.plateau_end_fraction
    {
        return Err(FluxError::InvalidConfig(
            "BennettChandlerConfig requires 0 <= plateau_start_fraction < plateau_end_fraction <= 1",
        ));
    }
    let start = ((n_steps as f64) * config.plateau_start_fraction).floor() as usize;
    let raw_end = ((n_steps as f64) * config.plateau_end_fraction).ceil() as usize;
    let end = if raw_end > n_steps { n_steps } else { raw_end };
    if end <= start || end - start < MIN_STATIONARITY_SAMPLES {
        return Err(FluxError::InvalidConfig(
            "BennettChandlerConfig plateau window must span at least 4 steps",
        ));
    }
    Ok((start, end))
}

/// Estimate `kappa(t)` and its stationary plateau with block-averaged uncertainty from
/// recorded dividing-surface trajectories.
///
/// # Errors
///
/// - [`FluxError::EmptyInput`] when `trajectories` or any trajectory's side history is empty.
/// - [`FluxError::NoPositiveFlux`] when no trajectory has `initial_normal_velocity > 0`.
/// - [`FluxError::NonStationaryPlateau`] when `kappa(t)` exhibits statistically significant
///   drift across the plateau window.
pub fn estimate_bennett_chandler(
    trajectories: &[ReactiveTrajectory],
    config: &BennettChandlerConfig,
) -> Result<BennettChandlerEstimate, FluxError> {
    if trajectories.is_empty() {
        return Err(FluxError::EmptyInput(
            "no reactive-flux trajectories supplied",
        ));
    }

    let n_steps = trajectories[0].sides.len();
    if n_steps == 0 {
        return Err(FluxError::EmptyInput(
            "ReactiveTrajectory::sides must not be empty",
        ));
    }

    let mut pos_flux_sum = 0.0_f64;
    let mut positive_flux_count = 0_usize;
    for traj in trajectories {
        if !traj.initial_normal_velocity.is_finite() {
            return Err(FluxError::NonFinite(
                "ReactiveTrajectory::initial_normal_velocity",
            ));
        }
        if traj.sides.len() != n_steps {
            return Err(FluxError::Shape {
                expected: n_steps,
                got: traj.sides.len(),
                context: "ReactiveTrajectory::sides length",
            });
        }
        if traj.initial_normal_velocity > 0.0 {
            pos_flux_sum += traj.initial_normal_velocity;
            positive_flux_count += 1;
        }
    }

    if positive_flux_count == 0 || pos_flux_sum <= f64::MIN_POSITIVE {
        return Err(FluxError::NoPositiveFlux);
    }

    let (win_start, win_end) = resolve_plateau_window(n_steps, config)?;
    let win_len = win_end - win_start;

    let mut kappa_t = Vec::with_capacity(n_steps);
    for t in 0..n_steps {
        let mut numer = 0.0_f64;
        for traj in trajectories {
            numer += traj.initial_normal_velocity * traj.sides[t].indicator();
        }
        kappa_t.push(numer / pos_flux_sum);
    }

    let plateau_slice = &kappa_t[win_start..win_end];
    let raw_time_stat = check_stationarity(plateau_slice, &config.stats)?;

    // Combine temporal sub-window SE with the paired trajectory-ensemble SE of the
    // early-minus-late shift by the law of total variance.
    let n_traj = trajectories.len();
    let raw_first = ((win_len as f64) * config.stats.stationarity_first_fraction).floor() as usize;
    let n_first = if raw_first < 2 { 2 } else { raw_first };
    let raw_second = ((win_len as f64) * config.stats.stationarity_last_fraction).floor() as usize;
    let max_second = win_len - n_first;
    let n_second = if raw_second < 2 {
        2
    } else if raw_second > max_second {
        max_second
    } else {
        raw_second
    };

    let mut paired_shifts = Vec::with_capacity(n_traj);
    for traj in trajectories {
        let early_avg = traj.sides[win_start..(win_start + n_first)]
            .iter()
            .map(|s| s.indicator())
            .sum::<f64>()
            / (n_first as f64);
        let late_avg = traj.sides[(win_end - n_second)..win_end]
            .iter()
            .map(|s| s.indicator())
            .sum::<f64>()
            / (n_second as f64);
        let delta_i =
            (n_traj as f64) * traj.initial_normal_velocity * (early_avg - late_avg) / pos_flux_sum;
        paired_shifts.push(delta_i);
    }
    let shift_est = batch_means(&paired_shifts, &config.stats)?;

    let plateau = plateau_slice.iter().copied().sum::<f64>() / (win_len as f64);
    let scale = 1.0 + plateau.abs();
    let diff = (raw_time_stat.first_mean - raw_time_stat.second_mean).abs();

    let sub_cfg = StatsConfig::default().with_min_batches(config.stats.min_batches);
    let first_time_se = batch_means(&plateau_slice[..n_first], &sub_cfg)?.standard_error;
    let second_time_se =
        batch_means(&plateau_slice[(win_len - n_second)..], &sub_cfg)?.standard_error;
    let total_se = first_time_se
        .hypot(second_time_se)
        .hypot(shift_est.standard_error);

    let z_score = if diff <= SERIES_FLAT_TOL * scale {
        0.0
    } else if total_se <= SERIES_FLAT_TOL * scale {
        f64::INFINITY
    } else {
        diff / total_se
    };
    let is_stationary = z_score <= config.stats.max_z_score;
    let stationarity = StationarityReport {
        is_stationary,
        z_score,
        max_z_score: config.stats.max_z_score,
        drift_slope: raw_time_stat.drift_slope,
        first_mean: raw_time_stat.first_mean,
        second_mean: raw_time_stat.second_mean,
    };

    if !stationarity.is_stationary {
        return Err(FluxError::NonStationaryPlateau {
            z_score: stationarity.z_score,
            max_z_score: stationarity.max_z_score,
            drift_slope: stationarity.drift_slope,
        });
    }

    // Linearized ratio influence values around `plateau`, whose full-sample mean is
    // `plateau` and whose batch-means SE accounts for numerator-denominator covariance
    // and inter-trajectory correlation.
    let mut linearized = Vec::with_capacity(n_traj);
    for traj in trajectories {
        let theta_bar = traj.sides[win_start..win_end]
            .iter()
            .map(|s| s.indicator())
            .sum::<f64>()
            / (win_len as f64);
        let pos_ind = if traj.initial_normal_velocity > 0.0 {
            1.0
        } else {
            0.0
        };
        let influence =
            (n_traj as f64) * traj.initial_normal_velocity * (theta_bar - plateau * pos_ind)
                / pos_flux_sum;
        linearized.push(plateau + influence);
    }

    // When consecutive trajectories are antithetic pairs or correlated configurations,
    // align blocks to even widths whenever `n_traj >= 4` and no explicit batch_size was set.
    let ensemble_stats_cfg = if config.stats.batch_size.is_none() && n_traj >= 4 {
        let base_est = batch_means(&linearized, &config.stats)?;
        let bs = base_est.batch_size;
        let even_bs = if bs.is_multiple_of(2) { bs } else { bs + 1 };
        if n_traj / even_bs >= 2 {
            config.stats.clone().with_batch_size(even_bs)
        } else {
            config.stats.clone()
        }
    } else {
        config.stats.clone()
    };
    let batch_est = batch_means(&linearized, &ensemble_stats_cfg)?;

    Ok(BennettChandlerEstimate {
        kappa_t,
        plateau,
        standard_error: batch_est.standard_error,
        n_eff: batch_est.n_eff,
        plateau_window: (win_start, win_end),
        stationarity,
        positive_flux_count,
    })
}

fn restore_constraint_and_fail<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    saved: Option<HyperplaneConstraint>,
    err: FluxError,
) -> FluxError {
    let _ = engine.set_constraint(saved);
    err
}

fn shoot_half<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    dividing_surface: &HyperplaneConstraint,
    steps: usize,
    time_step: f64,
    commit_checkpoints: usize,
    commitment_distance: f64,
) -> Result<Vec<Side>, FluxError> {
    let mut sides = Vec::with_capacity(steps);
    let mut run: (Option<Side>, usize) = (None, 0);
    for _ in 0..steps {
        engine.step(time_step)?;
        let s = dividing_surface.signed_distance(engine.positions())?;
        let side = Side::from_signed_distance(s);
        sides.push(side);
        let state = if s <= -commitment_distance {
            Some(Side::Reactant)
        } else if s >= commitment_distance {
            Some(Side::Product)
        } else {
            None
        };
        run = match (state, run.0) {
            (Some(curr), Some(prev)) if curr == prev => (state, run.1 + 1),
            (Some(_), _) => (state, 1),
            (None, _) => (None, 0),
        };
        if commit_checkpoints > 0 && run.1 >= commit_checkpoints {
            sides.resize(steps, side);
            break;
        }
    }
    Ok(sides)
}

/// Shoot reactive-flux trajectories from `initial_positions` on `dividing_surface`
/// using any [`DynamicsEngine`].
///
/// Early commitment is disabled by default. When explicitly enabled, each
/// half-trajectory ends once [`KappaConfig::commit_checkpoints`] consecutive
/// checkpoints are at least [`KappaConfig::commitment_distance`] into the same
/// side, copying that side into every remaining checkpoint up to
/// [`ShootingConfig::steps`].
///
/// This optimization is safe only when the distance identifies a thermalized
/// basin and the checkpoint window exceeds its relevant recrossing time. It can
/// otherwise hide a later return and bias the transmission coefficient.
///
/// Clears any active hyperplane constraint on `engine` for the free-flight shots
/// and restores it before returning.
///
/// # Errors
///
/// - [`FluxError::EmptyInput`] when `initial_positions` has zero rows.
/// - [`FluxError::Shape`] when `initial_positions.ncols()` or `dividing_surface.normal.len()`
///   differs from `engine.dof()`.
/// - [`FluxError::InvalidConfig`] when `dividing_surface` is not a unit plane,
///   `config.steps < 4`, `config.time_step <= 0` or non-finite, or
///   `config.kappa.commit_checkpoints > config.steps`, or early commitment is
///   enabled without a finite, positive `config.kappa.commitment_distance`.
/// - [`FluxError::NonFinite`] when `initial_positions` contains non-finite values.
/// - Any [`FluxError`] propagated by `engine`.
pub fn shoot_reactive_trajectories<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    dividing_surface: &HyperplaneConstraint,
    initial_positions: ArrayView2<'_, f64>,
    config: &ShootingConfig,
) -> Result<Vec<ReactiveTrajectory>, FluxError> {
    if initial_positions.nrows() == 0 {
        return Err(FluxError::EmptyInput(
            "shoot_reactive_trajectories requires at least one initial configuration",
        ));
    }
    let dof = engine.dof();
    if initial_positions.ncols() != dof {
        return Err(FluxError::Shape {
            expected: dof,
            got: initial_positions.ncols(),
            context: "shoot_reactive_trajectories initial_positions columns",
        });
    }
    dividing_surface.validate()?;
    if dividing_surface.normal.len() != dof {
        return Err(FluxError::Shape {
            expected: dof,
            got: dividing_surface.normal.len(),
            context: "shoot_reactive_trajectories dividing_surface normal",
        });
    }
    if config.steps < MIN_STATIONARITY_SAMPLES {
        return Err(FluxError::InvalidConfig(
            "ShootingConfig::steps must be at least 4",
        ));
    }
    if !config.time_step.is_finite() || config.time_step <= 0.0 {
        return Err(FluxError::InvalidConfig(
            "ShootingConfig::time_step must be finite and positive",
        ));
    }
    let commit_checkpoints = config.kappa.commit_checkpoints;
    if commit_checkpoints > config.steps {
        return Err(FluxError::InvalidConfig(
            "KappaConfig::commit_checkpoints must not exceed ShootingConfig::steps",
        ));
    }
    let commitment_distance = config.kappa.commitment_distance;
    if !commitment_distance.is_finite() {
        return Err(FluxError::NonFinite("KappaConfig::commitment_distance"));
    }
    if commit_checkpoints > 0 && commitment_distance <= 0.0 {
        return Err(FluxError::InvalidConfig(
            "KappaConfig::commitment_distance must be strictly positive when early commitment is enabled",
        ));
    }
    if commit_checkpoints == 0 && commitment_distance < 0.0 {
        return Err(FluxError::InvalidConfig(
            "KappaConfig::commitment_distance must be non-negative",
        ));
    }
    for row in initial_positions.rows() {
        validate_finite_slice(row, "shoot_reactive_trajectories initial_positions")?;
    }

    let saved_constraint = engine.constraint().cloned();
    engine.set_constraint(None)?;

    let capacity = if config.antithetic {
        2 * initial_positions.nrows()
    } else {
        initial_positions.nrows()
    };
    let mut trajectories = Vec::with_capacity(capacity);

    for (idx, x0) in initial_positions.rows().into_iter().enumerate() {
        let shot_seed = config
            .seed
            .wrapping_add((idx as u64).wrapping_mul(SHOT_SEED_INCREMENT));

        if let Err(e) = engine.set_positions(x0) {
            return Err(restore_constraint_and_fail(engine, saved_constraint, e));
        }
        if let Err(e) = engine.shoot_velocities(shot_seed) {
            return Err(restore_constraint_and_fail(engine, saved_constraint, e));
        }
        let v_plus = engine.velocities().to_owned();
        let v0_normal = dot(dividing_surface.normal.view(), v_plus.view());

        let sides_plus = match shoot_half(
            engine,
            dividing_surface,
            config.steps,
            config.time_step,
            commit_checkpoints,
            commitment_distance,
        ) {
            Ok(sides) => sides,
            Err(e) => return Err(restore_constraint_and_fail(engine, saved_constraint, e)),
        };
        match ReactiveTrajectory::new(v0_normal, sides_plus) {
            Ok(t) => trajectories.push(t),
            Err(e) => return Err(restore_constraint_and_fail(engine, saved_constraint, e)),
        }

        if config.antithetic {
            let v_minus = v_plus.mapv(|c| -c);
            if let Err(e) = engine.set_positions(x0) {
                return Err(restore_constraint_and_fail(engine, saved_constraint, e));
            }
            // Seed an independent thermal-noise stream before installing `-v_plus`.
            if let Err(e) = engine.shoot_velocities(shot_seed ^ ANTITHETIC_SEED_MIX) {
                return Err(restore_constraint_and_fail(engine, saved_constraint, e));
            }
            if let Err(e) = engine.set_velocities(v_minus.view()) {
                return Err(restore_constraint_and_fail(engine, saved_constraint, e));
            }

            let sides_minus = match shoot_half(
                engine,
                dividing_surface,
                config.steps,
                config.time_step,
                commit_checkpoints,
                commitment_distance,
            ) {
                Ok(sides) => sides,
                Err(e) => return Err(restore_constraint_and_fail(engine, saved_constraint, e)),
            };
            match ReactiveTrajectory::new(-v0_normal, sides_minus) {
                Ok(t) => trajectories.push(t),
                Err(e) => return Err(restore_constraint_and_fail(engine, saved_constraint, e)),
            }
        }
    }

    engine.set_constraint(saved_constraint)?;
    Ok(trajectories)
}

/// Shoot reactive-flux trajectories through `engine` and compute the
/// Bennett-Chandler transmission coefficient estimate.
pub fn shoot_bennett_chandler<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    dividing_surface: &HyperplaneConstraint,
    initial_positions: ArrayView2<'_, f64>,
    config: &ShootingConfig,
) -> Result<BennettChandlerEstimate, FluxError> {
    let trajectories =
        shoot_reactive_trajectories(engine, dividing_surface, initial_positions, config)?;
    estimate_bennett_chandler(&trajectories, &config.estimator)
}
