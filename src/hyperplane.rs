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

use ndarray::Array2;

use crate::engine::DynamicsEngine;
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

    let restored = snapshot.restore(engine);
    match (result, restored) {
        (Ok(sample), Ok(())) => Ok(sample),
        (Err(err), _) => Err(err),
        (Ok(_), Err(err)) => Err(err),
    }
}

struct EngineSnapshot {
    constraint: Option<crate::engine::HyperplaneConstraint>,
    thermostat: crate::engine::ThermostatConfig,
    positions: ndarray::Array1<f64>,
    velocities: ndarray::Array1<f64>,
}

impl EngineSnapshot {
    fn capture<E: DynamicsEngine + ?Sized>(engine: &E) -> Self {
        Self {
            constraint: engine.constraint().cloned(),
            thermostat: *engine.thermostat(),
            positions: engine.positions().to_owned(),
            velocities: engine.velocities().to_owned(),
        }
    }

    fn restore<E: DynamicsEngine + ?Sized>(self, engine: &mut E) -> Result<(), FluxError> {
        engine.set_constraint(self.constraint)?;
        engine.set_thermostat(self.thermostat)?;
        engine.set_positions(self.positions.view())?;
        engine.set_velocities(self.velocities.view())?;
        Ok(())
    }
}
