//! Constrained hyperplane sampling, mean force, torque, and control variates.
//!
//! Samples configurations restricted to a hyperplane `sigma(x) = n · x - c = 0`
//! through [`crate::engine::DynamicsEngine`] to estimate the hyperplane
//! free-energy derivative `dA/dc = <lambda>_c` and collect dividing-surface
//! configurations for reactive-flux shooting or free-energy integration.
//!
//! Note: the mean holonomic constraint force `<lambda>` and retained
//! configurations are implemented here via [`sample_hyperplane`]; the in-plane
//! virial, rotational torque `dA/dn`, and Assaraf-Caffarel / Stein zero-variance
//! harmonic control variates are not yet implemented.
//!
//! ## Theory
//!
//! For a unit normal `n` (`||n||_2 = 1`) and scalar offset `c`, the hyperplane
//! `sigma(x) = n · x - c = 0` carries the canonical constrained free energy
//! `A(c) = -k_B T ln int delta(n · x - c) exp(-V(x) / k_B T) dx`. Under mass
//! matrix `M = diag(m_i)` with inverse-mass metric norm
//! `z = n^T M^-1 n = sum_i n_i^2 / m_i`, the holonomic Lagrange multiplier
//! enforcing `sigma(x) = 0` and `n · v = 0` is (Sprik and Ciccotti 1998)
//!
//! `lambda(x) = -(n^T M^-1 F(x)) / (n^T M^-1 n) = (n^T M^-1 grad V(x)) / (n^T M^-1 n)`,
//!
//! whose constrained canonical average equals `dA/dc = <lambda>_c` because
//! `||grad sigma||_{M^-1} = sqrt(z)` is constant across the flat hyperplane.
//! [`sample_hyperplane`] discards `burn_in_steps`, records `lambda` over
//! `sample_steps` steps, and computes its mean and block-averaged standard
//! error via [`crate::stats::batch_means`], while retaining configurations
//! every `retention_stride` steps.
//!
//! ## Assumptions
//!
//! - The hyperplane normal `n` has unit Euclidean norm (`||n||_2 = 1` within
//!   [`crate::engine::UNIT_NORMAL_TOL`]) and length matching `engine.dof()`.
//! - The constrained Markov chain reaches stationarity within `burn_in_steps`
//!   at positive thermal energy `k_B * T` and friction `gamma > 0`.
//! - Every constrained [`crate::engine::DynamicsEngine::step`] call populates
//!   [`crate::engine::StepReport::constraint_force`] with a finite `Some(lambda)`.
//!
//! ## Units
//!
//! - `plane.normal` is dimensionless; `plane.offset` has engine length unit `[L]`.
//! - `time_step` has engine time unit `[T]`.
//! - `mean_force` and `standard_error` have force units `[E] / [L]` (energy per
//!   unit offset along `n`, e.g. `eV / Angstrom` or reduced units).
//! - `configurations` has shape `(n_retained, dof)` in `[L]`.
//!
//! ## Engine wiring
//!
//! [`sample_hyperplane`] accepts `&mut E` for any `E: DynamicsEngine + ?Sized`.
//! Before mutating engine state, it captures an internal `EngineSnapshot`
//! (`positions`, `velocities`, `cell`, `thermostat`, and `constraint`), runs
//! `engine.reseed(config.seed)`, installs `Some(plane.clone())` via
//! `engine.set_constraint`, draws tangent-projected Maxwell-Boltzmann velocities
//! via `engine.shoot_velocities(config.seed)`, integrates `burn_in_steps` +
//! `sample_steps` steps of duration `config.time_step`, and restores the five
//! captured state fields before returning (whether sampling succeeded or failed).
//!
//! - [`crate::engine::native::NativeEngine`]: enforces `sigma(x) = 0` with
//!   two-pass linear SHAKE position projection and RATTLE tangent velocity
//!   projection in the mass metric over any [`crate::engine::ForceSurface`]
//!   ([`rgsaddle::PointSurface`], [`crate::engine::EindirSurface`], or
//!   `engine::rgpot::RgpotSurface`).
//! - `engine::lammpc::LammpcEngine` (feature `lammpc`): drives constrained
//!   Langevin or zero-friction projected velocity-Verlet steps through
//!   `lammpc_session_md_*` in LAMMPS `metal` units and evaluates the
//!   mass-weighted multiplier `lambda` from `lammpc_session_calculate_result`.
//! - `engine::rgpot_session::RgpotSessionEngine` (feature `rgpot-session`):
//!   forwards `(plane_point, plane_normal)` to `rgpot_md_session_step_langevin`
//!   once `rgpot_core::md_session` is linked (`OpenMm`, `NbLib`, `Lammps`).
//!
//! ## Example
//!
//! ```rust
//! use ndarray::{Array1, ArrayView1};
//! use rgflux::engine::native::{NativeConfig, NativeEngine};
//! use rgflux::engine::{DynamicsEngine, HyperplaneConstraint, ThermostatConfig};
//! use rgflux::hyperplane::{HyperplaneConfig, sample_hyperplane};
//!
//! struct Harmonic2D;
//! impl rgsaddle::PointSurface for Harmonic2D {
//!     fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
//!         Ok((
//!             0.5 * (x[0] * x[0] + 2.0 * x[1] * x[1]),
//!             Array1::from_vec(vec![x[0], 2.0 * x[1]]),
//!         ))
//!     }
//! }
//!
//! let cfg = NativeConfig::default()
//!     .with_masses(Array1::from_vec(vec![1.0, 1.0]))
//!     .with_thermostat(ThermostatConfig::default().with_gamma(1.5).with_temperature(0.2))
//!     .with_seed(11);
//! let mut engine = NativeEngine::new(Harmonic2D, cfg)?;
//! engine.set_positions(Array1::from_vec(vec![0.4, 0.0]).view())?;
//!
//! let plane = HyperplaneConstraint::from_direction(Array1::from_vec(vec![1.0, 0.0]).view(), 0.4)?;
//! let hp_cfg = HyperplaneConfig::default()
//!     .with_burn_in_steps(50)
//!     .with_sample_steps(200)
//!     .with_retention_stride(20)
//!     .with_time_step(0.01)
//!     .with_seed(11);
//! let sample = sample_hyperplane(&mut engine, &plane, &hp_cfg)?;
//! assert!((sample.mean_force - 0.4).abs() < 1e-12);
//! assert_eq!(sample.configurations.dim(), (10, 2));
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
//! - Jóhannesson, G. H., and Jónsson, H. (2001). *Optimization of hyperplanar
//!   transition states*. The Journal of Chemical Physics, 115, 9644-9656.
//!   <https://doi.org/10.1063/1.1415499>
//! - Bligaard, T., and Jónsson, H. (2005). *Optimization of hyperplanar
//!   transition states: Application to 2D test problems*. Computer Physics
//!   Communications, 169, 284-288. <https://doi.org/10.1016/j.cpc.2005.03.064>
//! - Sprik, M., and Ciccotti, G. (1998). *Free energy from constrained
//!   molecular dynamics*. The Journal of Chemical Physics, 109, 7737-7744.
//!   <https://doi.org/10.1063/1.477419>
//! - Leimkuhler, B., and Matthews, C. (2012). *Rational Construction of
//!   Stochastic Numerical Methods for Molecular Sampling*. Applied Mathematics
//!   Research eXpress. <https://doi.org/10.1093/amrx/abs010>
//! - Leimkuhler, B., and Matthews, C. (2016). *Efficient molecular dynamics
//!   using geodesic integration and solvent-solute splitting*. Proceedings of
//!   the Royal Society A, 472, 20160138. <https://doi.org/10.1098/rspa.2016.0138>
//! - Assaraf, R., and Caffarel, M. (1999). *Zero-Variance Principle for Monte
//!   Carlo Algorithms*. Physical Review Letters, 83, 4682-4685.
//!   <https://doi.org/10.1103/PhysRevLett.83.4682>

pub use crate::engine::HyperplaneConstraint;

use ndarray::Array2;

use crate::engine::{DynamicsEngine, EngineSnapshot};
use crate::error::FluxError;
use crate::stats::{StatsConfig, batch_means};

/// Configuration for sampling free-energy gradients and configurations on a hyperplane.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct HyperplaneConfig {
    /// Equilibration steps discarded before recording observables.
    ///
    /// Default: `500`.
    pub burn_in_steps: usize,
    /// Production steps integrated on the hyperplane (`>= 2`).
    ///
    /// Default: `4_000`.
    pub sample_steps: usize,
    /// Steps between retained dividing-surface configurations (`>= 1`).
    ///
    /// Default: `10`.
    pub retention_stride: usize,
    /// Integration timestep `dt > 0`.
    ///
    /// Default: `0.005`.
    pub time_step: f64,
    /// Deterministic PRNG seed.
    ///
    /// Default: `1`.
    pub seed: u64,
    /// Batch-means and stationarity controls for the sampled mean force.
    ///
    /// Default: [`StatsConfig::default()`].
    pub stats: StatsConfig,
}

impl Default for HyperplaneConfig {
    fn default() -> Self {
        Self {
            burn_in_steps: 500,
            sample_steps: 4_000,
            retention_stride: 10,
            time_step: 0.005,
            seed: 1,
            stats: StatsConfig::default(),
        }
    }
}

with_setters!(HyperplaneConfig {
    with_burn_in_steps => burn_in_steps: val usize,
    with_sample_steps => sample_steps: val usize,
    with_retention_stride => retention_stride: val usize,
    with_time_step => time_step: val f64,
    with_seed => seed: val u64,
    with_stats => stats: val StatsConfig,
});

/// Summary of a constrained hyperplane sampling trajectory.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct HyperplaneSample {
    /// Sampled hyperplane definition.
    pub plane: HyperplaneConstraint,
    /// Mean holonomic constraint force `<lambda> = dA/dc`.
    pub mean_force: f64,
    /// Block-averaged standard error of [`Self::mean_force`].
    pub standard_error: f64,
    /// Retained configurations on the hyperplane (`(n_retained, dof)`).
    pub configurations: Array2<f64>,
}

/// Sample a hyperplane through `engine`.
///
/// # Errors
///
/// - [`FluxError::InvalidConfig`] when `retention_stride == 0`, `sample_steps < 2`,
///   `time_step <= 0` or non-finite, or [`crate::engine::StepReport::constraint_force`]
///   is `None`.
/// - [`FluxError::Shape`] when `plane.normal.len() != engine.dof()`.
/// - [`FluxError::NonFinite`] when a sampled constraint force is NaN or infinite.
/// - Any [`FluxError`] propagated by `engine` during setup, stepping, or snapshot
///   restoration.
pub fn sample_hyperplane<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    plane: &HyperplaneConstraint,
    config: &HyperplaneConfig,
) -> Result<HyperplaneSample, FluxError> {
    if config.retention_stride == 0 {
        return Err(FluxError::InvalidConfig(
            "HyperplaneConfig::retention_stride must be at least 1",
        ));
    }
    if config.sample_steps < 2 {
        return Err(FluxError::InvalidConfig(
            "HyperplaneConfig::sample_steps must be at least 2",
        ));
    }
    if !config.time_step.is_finite() || config.time_step <= 0.0 {
        return Err(FluxError::InvalidConfig(
            "HyperplaneConfig::time_step must be finite and positive",
        ));
    }
    if plane.normal.len() != engine.dof() {
        return Err(FluxError::Shape {
            expected: engine.dof(),
            got: plane.normal.len(),
            context: "hyperplane dimension",
        });
    }

    let snapshot = EngineSnapshot::capture(engine);

    let result = (|| -> Result<HyperplaneSample, FluxError> {
        engine.reseed(config.seed)?;
        engine.set_constraint(Some(plane.clone()))?;
        engine.shoot_velocities(config.seed)?;

        for _ in 0..config.burn_in_steps {
            engine.step(config.time_step)?;
        }

        let mut force_series = Vec::with_capacity(config.sample_steps);
        let mut retained_flat = Vec::new();
        let mut retained_count = 0_usize;
        for step_idx in 0..config.sample_steps {
            let report = engine.step(config.time_step)?;
            let lambda = report.constraint_force.ok_or(FluxError::InvalidConfig(
                "DynamicsEngine::step must report constraint_force when constrained",
            ))?;
            if !lambda.is_finite() {
                return Err(FluxError::NonFinite("hyperplane constraint force"));
            }
            force_series.push(lambda);

            if step_idx % config.retention_stride == 0 {
                retained_flat.extend(engine.positions().iter().copied());
                retained_count += 1;
            }
        }

        let estimate = batch_means(&force_series, &config.stats)?;
        let configurations = Array2::from_shape_vec((retained_count, engine.dof()), retained_flat)
            .map_err(|_| FluxError::InvalidConfig("failed to reshape retained configurations"))?;
        Ok(HyperplaneSample {
            plane: plane.clone(),
            mean_force: estimate.mean,
            standard_error: estimate.standard_error,
            configurations,
        })
    })();

    snapshot.restore_result(engine, result)
}

#[cfg(test)]
mod restore_tests {
    use ndarray::{Array1, ArrayView1};

    use super::{HyperplaneConfig, sample_hyperplane};
    use crate::engine::{
        DynamicsEngine, HyperplaneConstraint, StepReport, SurfaceVelocityPolicy, ThermostatConfig,
    };
    use crate::error::FluxError;

    struct ScriptedEngine {
        positions: Array1<f64>,
        velocities: Array1<f64>,
        masses: Array1<f64>,
        thermostat: ThermostatConfig,
        constraint: Option<HyperplaneConstraint>,
        fail_step: bool,
        reject_position_writes: bool,
    }

    impl ScriptedEngine {
        fn new(fail_step: bool, reject_position_writes: bool) -> Self {
            Self {
                positions: Array1::from_vec(vec![0.2]),
                velocities: Array1::zeros(1),
                masses: Array1::from_vec(vec![1.0]),
                thermostat: ThermostatConfig::default(),
                constraint: None,
                fail_step,
                reject_position_writes,
            }
        }
    }

    impl DynamicsEngine for ScriptedEngine {
        fn dof(&self) -> usize {
            self.positions.len()
        }

        fn masses(&self) -> ArrayView1<'_, f64> {
            self.masses.view()
        }

        fn positions(&self) -> ArrayView1<'_, f64> {
            self.positions.view()
        }

        fn set_positions(&mut self, positions: ArrayView1<'_, f64>) -> Result<(), FluxError> {
            if self.reject_position_writes {
                return Err(FluxError::InvalidConfig("position restore rejected"));
            }
            if positions.len() != self.positions.len() {
                return Err(FluxError::Shape {
                    expected: self.positions.len(),
                    got: positions.len(),
                    context: "ScriptedEngine positions",
                });
            }
            self.positions.assign(&positions);
            Ok(())
        }

        fn velocities(&self) -> ArrayView1<'_, f64> {
            self.velocities.view()
        }

        fn set_velocities(&mut self, velocities: ArrayView1<'_, f64>) -> Result<(), FluxError> {
            if velocities.len() != self.velocities.len() {
                return Err(FluxError::Shape {
                    expected: self.velocities.len(),
                    got: velocities.len(),
                    context: "ScriptedEngine velocities",
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
            Ok((0.0, Array1::zeros(self.dof())))
        }

        fn step(&mut self, _dt: f64) -> Result<StepReport, FluxError> {
            self.positions[0] += 1.0;
            if self.fail_step {
                return Err(FluxError::InvalidConfig("scripted step failed"));
            }
            Ok(StepReport {
                potential_energy: 0.0,
                kinetic_energy: 0.0,
                constraint_force: Some(0.0),
            })
        }

        fn reseed(&mut self, _seed: u64) -> Result<(), FluxError> {
            Ok(())
        }

        fn shoot_velocities(&mut self, _seed: u64) -> Result<(), FluxError> {
            self.velocities[0] = 1.0;
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

    fn state(
        engine: &ScriptedEngine,
    ) -> (
        Array1<f64>,
        Array1<f64>,
        Option<HyperplaneConstraint>,
        ThermostatConfig,
    ) {
        (
            engine.positions.clone(),
            engine.velocities.clone(),
            engine.constraint.clone(),
            engine.thermostat,
        )
    }

    #[test]
    fn sample_restores_on_success_and_on_step_failure() {
        let plane =
            HyperplaneConstraint::from_direction(Array1::from_vec(vec![1.0]).view(), 0.2).unwrap();
        let config = HyperplaneConfig::default()
            .with_burn_in_steps(0)
            .with_sample_steps(2)
            .with_retention_stride(1)
            .with_time_step(0.01)
            .with_seed(9);

        let mut ok_engine = ScriptedEngine::new(false, false);
        let before_ok = state(&ok_engine);
        let sampled = sample_hyperplane(&mut ok_engine, &plane, &config).unwrap();
        assert!(sampled.mean_force.is_finite());
        assert_eq!(state(&ok_engine), before_ok);

        let mut err_engine = ScriptedEngine::new(true, false);
        let before_err = state(&err_engine);
        let err = sample_hyperplane(&mut err_engine, &plane, &config).unwrap_err();
        assert_eq!(err, FluxError::InvalidConfig("scripted step failed"));
        assert_eq!(state(&err_engine), before_err);
    }

    #[test]
    fn sample_returns_restore_error_when_sampling_succeeds() {
        let plane =
            HyperplaneConstraint::from_direction(Array1::from_vec(vec![1.0]).view(), 0.2).unwrap();
        let config = HyperplaneConfig::default()
            .with_burn_in_steps(0)
            .with_sample_steps(2)
            .with_retention_stride(1)
            .with_time_step(0.01)
            .with_seed(9);
        let mut engine = ScriptedEngine::new(false, true);
        let err = sample_hyperplane(&mut engine, &plane, &config).unwrap_err();
        assert_eq!(err, FluxError::InvalidConfig("position restore rejected"));
    }

    #[test]
    fn sample_preserves_sampling_error_when_restore_also_fails() {
        let plane =
            HyperplaneConstraint::from_direction(Array1::from_vec(vec![1.0]).view(), 0.2).unwrap();
        let config = HyperplaneConfig::default()
            .with_burn_in_steps(0)
            .with_sample_steps(2)
            .with_retention_stride(1)
            .with_time_step(0.01)
            .with_seed(9);
        let mut engine = ScriptedEngine::new(true, true);
        let err = sample_hyperplane(&mut engine, &plane, &config).unwrap_err();
        assert_eq!(err, FluxError::InvalidConfig("scripted step failed"));
    }
}
