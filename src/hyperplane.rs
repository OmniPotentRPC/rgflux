//! Constrained hyperplane sampling, mean force, torque, and control variates.
//!
//! Samples configurations restricted to a hyperplane `n · x - c = 0` through
//! [`crate::engine::DynamicsEngine`] to estimate the hyperplane free-energy
//! derivatives `dA/dc` and `dA/dn` along with in-plane virial and zero-variance
//! harmonic control variates.
//!
//! Note: implementation moves from rgsaddle.
//!
//! ## References
//!
//! - Sprik, M., and Ciccotti, G. (1998). *Free energy from constrained
//!   molecular dynamics*. The Journal of Chemical Physics, 109, 7737–7744.
//!   <https://doi.org/10.1063/1.477419>
//! - Jóhannesson, G. H., and Jónsson, H. (2001). *Optimization of hyperplanar
//!   transition states*. The Journal of Chemical Physics, 115, 9644–9656.
//!   <https://doi.org/10.1063/1.1415499>
//! - Leimkuhler, B., and Matthews, C. (2012). *Rational Construction of
//!   Stochastic Numerical Methods for Molecular Sampling*. Applied Mathematics
//!   Research eXpress. <https://doi.org/10.1093/amrx/abs010>
//! - Leimkuhler, B., and Matthews, C. (2016). *Efficient molecular dynamics
//!   using geodesic integration and solvent–solute splitting*. Proceedings of
//!   the Royal Society A, 472, 20160138. <https://doi.org/10.1098/rspa.2016.0138>
//! - Assaraf, R., and Caffarel, M. (1999). *Zero-Variance Principle for Monte
//!   Carlo Algorithms*. Physical Review Letters, 83, 4682–4685.
//!   <https://doi.org/10.1103/PhysRevLett.83.4682>

pub use crate::engine::HyperplaneConstraint;

use ndarray::{Array1, Array2, ArrayView1, Axis};

use crate::engine::{DynamicsEngine, EngineSnapshot, validate_finite_slice};
use crate::error::FluxError;
use crate::stats::{StatsConfig, batch_means};

/// Configuration for sampling free-energy gradients and configurations on a hyperplane.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct HyperplaneConfig {
    /// Equilibration steps discarded before recording observables.
    pub burn_in_steps: usize,
    /// Production steps integrated on the hyperplane.
    pub sample_steps: usize,
    /// Steps between retained dividing-surface configurations (`>= 1`).
    pub retention_stride: usize,
    /// Integration timestep `dt > 0`.
    pub time_step: f64,
    /// Deterministic PRNG seed.
    pub seed: u64,
    /// Batch-means and stationarity controls for the sampled mean force.
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
    /// Mean physical force projected into the sampled plane.
    pub mean_projected_force: Array1<f64>,
    /// Mean production position.
    pub mean_position: Array1<f64>,
    /// Population covariance of production positions.
    pub position_covariance: Array2<f64>,
}

/// Build the plane through a path `image`, orthogonal to its `tangent`.
pub fn plane_from_path_image(
    image: ArrayView1<'_, f64>,
    tangent: ArrayView1<'_, f64>,
) -> Result<HyperplaneConstraint, FluxError> {
    if image.len() != tangent.len() {
        return Err(FluxError::Shape {
            expected: image.len(),
            got: tangent.len(),
            context: "path image tangent",
        });
    }
    validate_finite_slice(image, "path image")?;
    let mut plane = HyperplaneConstraint::from_direction(tangent, 0.0)?;
    plane.offset = plane.normal.dot(&image);
    if !plane.offset.is_finite() {
        return Err(FluxError::NonFinite("path plane offset"));
    }
    Ok(plane)
}

/// Sample a hyperplane through `engine`.
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
        let mut mean_projected_force = Array1::zeros(engine.dof());
        let mut mean_position = Array1::zeros(engine.dof());
        let mut covariance_accumulator = Array2::zeros((engine.dof(), engine.dof()));
        for step_idx in 0..config.sample_steps {
            let report = engine.step(config.time_step)?;
            let lambda = report.constraint_force.ok_or(FluxError::InvalidConfig(
                "DynamicsEngine::step must report constraint_force when constrained",
            ))?;
            if !lambda.is_finite() {
                return Err(FluxError::NonFinite("hyperplane constraint force"));
            }
            force_series.push(lambda);

            let (_, mut projected_force) = engine.energy_and_forces()?;
            if projected_force.len() != engine.dof() {
                return Err(FluxError::Shape {
                    expected: engine.dof(),
                    got: projected_force.len(),
                    context: "hyperplane physical force",
                });
            }
            validate_finite_slice(projected_force.view(), "hyperplane physical force")?;
            let normal_component = projected_force.dot(&plane.normal);
            projected_force.scaled_add(-normal_component, &plane.normal);
            let sample_count = (step_idx + 1) as f64;
            let projected_force_delta = &projected_force - &mean_projected_force;
            mean_projected_force.scaled_add(sample_count.recip(), &projected_force_delta);

            let position = engine.positions().to_owned();
            validate_finite_slice(position.view(), "hyperplane position")?;
            let position_delta = &position - &mean_position;
            mean_position.scaled_add(sample_count.recip(), &position_delta);
            let updated_delta = &position - &mean_position;
            covariance_accumulator += &position_delta
                .view()
                .insert_axis(Axis(1))
                .dot(&updated_delta.view().insert_axis(Axis(0)));

            if step_idx % config.retention_stride == 0 {
                retained_flat.extend(position.iter().copied());
                retained_count += 1;
            }
        }

        let estimate = batch_means(&force_series, &config.stats)?;
        let configurations = Array2::from_shape_vec((retained_count, engine.dof()), retained_flat)
            .map_err(|_| FluxError::InvalidConfig("failed to reshape retained configurations"))?;
        let position_covariance = covariance_accumulator / config.sample_steps as f64;
        Ok(HyperplaneSample {
            plane: plane.clone(),
            mean_force: estimate.mean,
            standard_error: estimate.standard_error,
            configurations,
            mean_projected_force,
            mean_position,
            position_covariance,
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
