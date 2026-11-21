use ndarray::{Array1, ArrayView1};

use rgflux::FluxError;
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::engine::{
    DynamicsEngine, ForceSurface, HyperplaneConstraint, StepReport, SurfaceVelocityPolicy,
    ThermostatConfig,
};
use rgflux::hyperplane::{HyperplaneConfig, sample_hyperplane};
use rgflux::ohtst::{OhtstConfig, optimize_hyperplane};

struct DoubleWell2D;

impl ForceSurface for DoubleWell2D {
    fn energy_and_forces(
        &self,
        positions: ArrayView1<'_, f64>,
        _cell: Option<&[f64; 9]>,
    ) -> Result<(f64, Array1<f64>), FluxError> {
        if positions.len() != 2 {
            return Err(FluxError::Shape {
                expected: 2,
                got: positions.len(),
                context: "DoubleWell2D positions",
            });
        }
        let x = positions[0];
        let y = positions[1];
        let well = x * x - 1.0;
        Ok((
            0.25 * well * well + 0.5 * y * y,
            Array1::from_vec(vec![-x * well, -y]),
        ))
    }
}

fn engine_at(position: [f64; 2], seed: u64) -> Result<NativeEngine<DoubleWell2D>, FluxError> {
    let mut engine = NativeEngine::new(
        DoubleWell2D,
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0, 1.0]))
            .with_seed(seed),
    )?;
    engine.set_positions(Array1::from_vec(position.to_vec()).view())?;
    Ok(engine)
}

#[test]
fn hyperplane_sampling_returns_finite_force_and_configurations() -> Result<(), FluxError> {
    let plane = HyperplaneConstraint::from_direction(Array1::from_vec(vec![1.0, 0.0]).view(), 0.2)?;
    let config = HyperplaneConfig::default()
        .with_burn_in_steps(150)
        .with_sample_steps(800)
        .with_retention_stride(20)
        .with_time_step(0.01)
        .with_seed(11);

    let mut engine = engine_at([0.2, 0.0], 11)?;
    let sampled = sample_hyperplane(&mut engine, &plane, &config)?;

    assert!(sampled.mean_force.is_finite());
    assert!(sampled.standard_error.is_finite());
    assert!(sampled.configurations.nrows() > 0);
    assert_eq!(sampled.configurations.ncols(), 2);
    Ok(())
}

#[test]
fn hyperplane_mean_force_changes_with_plane_offset() -> Result<(), FluxError> {
    let config = HyperplaneConfig::default()
        .with_burn_in_steps(120)
        .with_sample_steps(600)
        .with_retention_stride(30)
        .with_time_step(0.01)
        .with_seed(21);

    let mut left = engine_at([-0.3, 0.0], 21)?;
    let mut right = engine_at([0.3, 0.0], 21)?;

    let left_plane =
        HyperplaneConstraint::from_direction(Array1::from_vec(vec![1.0, 0.0]).view(), -0.3)?;
    let right_plane =
        HyperplaneConstraint::from_direction(Array1::from_vec(vec![1.0, 0.0]).view(), 0.3)?;

    let left_sample = sample_hyperplane(&mut left, &left_plane, &config)?;
    let right_sample = sample_hyperplane(&mut right, &right_plane, &config)?;

    assert!((left_sample.mean_force - right_sample.mean_force).abs() > 1.0e-6);
    Ok(())
}

#[test]
fn ohtst_optimization_returns_positive_finite_rate() -> Result<(), FluxError> {
    let mut engine = engine_at([0.0, 0.0], 33)?;
    let seed_plane =
        HyperplaneConstraint::from_direction(Array1::from_vec(vec![1.0, 0.0]).view(), 0.0)?;
    let config = OhtstConfig::default().with_seed(33);

    let estimate = optimize_hyperplane(&mut engine, &seed_plane, &config)?;

    assert!(estimate.rate.is_finite() && estimate.rate > 0.0);
    assert!(estimate.standard_error.is_finite() && estimate.standard_error >= 0.0);
    Ok(())
}

fn engine_state<E: DynamicsEngine>(
    engine: &E,
) -> (
    Array1<f64>,
    Array1<f64>,
    Option<HyperplaneConstraint>,
    ThermostatConfig,
) {
    (
        engine.positions().to_owned(),
        engine.velocities().to_owned(),
        engine.constraint().cloned(),
        *engine.thermostat(),
    )
}

struct RestoreRejectingEngine<E> {
    inner: E,
    initial_constraint: Option<HyperplaneConstraint>,
    operation_started: bool,
    reject_next_position_write: bool,
}

impl<E: DynamicsEngine> RestoreRejectingEngine<E> {
    fn new(inner: E) -> Self {
        Self {
            initial_constraint: inner.constraint().cloned(),
            inner,
            operation_started: false,
            reject_next_position_write: false,
        }
    }
}

impl<E: DynamicsEngine> DynamicsEngine for RestoreRejectingEngine<E> {
    fn dof(&self) -> usize {
        self.inner.dof()
    }

    fn masses(&self) -> ArrayView1<'_, f64> {
        self.inner.masses()
    }

    fn positions(&self) -> ArrayView1<'_, f64> {
        self.inner.positions()
    }

    fn set_positions(&mut self, positions: ArrayView1<'_, f64>) -> Result<(), FluxError> {
        if self.reject_next_position_write {
            self.reject_next_position_write = false;
            return Err(FluxError::InvalidConfig("position restore rejected"));
        }
        self.inner.set_positions(positions)
    }

    fn velocities(&self) -> ArrayView1<'_, f64> {
        self.inner.velocities()
    }

    fn set_velocities(&mut self, velocities: ArrayView1<'_, f64>) -> Result<(), FluxError> {
        self.inner.set_velocities(velocities)
    }

    fn cell(&self) -> Option<&[f64; 9]> {
        self.inner.cell()
    }

    fn set_cell(&mut self, cell: Option<[f64; 9]>) -> Result<(), FluxError> {
        self.inner.set_cell(cell)
    }

    fn thermostat(&self) -> &ThermostatConfig {
        self.inner.thermostat()
    }

    fn set_thermostat(&mut self, thermostat: ThermostatConfig) -> Result<(), FluxError> {
        self.inner.set_thermostat(thermostat)
    }

    fn constraint(&self) -> Option<&HyperplaneConstraint> {
        self.inner.constraint()
    }

    fn set_constraint(
        &mut self,
        constraint: Option<HyperplaneConstraint>,
    ) -> Result<(), FluxError> {
        if constraint.is_none() {
            self.operation_started = true;
        } else if self.operation_started && constraint == self.initial_constraint {
            self.reject_next_position_write = true;
        }
        self.inner.set_constraint(constraint)
    }

    fn energy_and_forces(&mut self) -> Result<(f64, Array1<f64>), FluxError> {
        self.inner.energy_and_forces()
    }

    fn step(&mut self, dt: f64) -> Result<StepReport, FluxError> {
        self.inner.step(dt)
    }

    fn reseed(&mut self, seed: u64) -> Result<(), FluxError> {
        self.inner.reseed(seed)
    }

    fn shoot_velocities(&mut self, seed: u64) -> Result<(), FluxError> {
        self.inner.shoot_velocities(seed)
    }

    fn reflect_velocity(&mut self, normal: ArrayView1<'_, f64>) -> Result<f64, FluxError> {
        self.inner.reflect_velocity(normal)
    }

    fn apply_dividing_surface(
        &mut self,
        normal: ArrayView1<'_, f64>,
        policy: SurfaceVelocityPolicy,
    ) -> Result<f64, FluxError> {
        self.inner.apply_dividing_surface(normal, policy)
    }
}

struct RisingLine;

impl ForceSurface for RisingLine {
    fn energy_and_forces(
        &self,
        positions: ArrayView1<'_, f64>,
        _cell: Option<&[f64; 9]>,
    ) -> Result<(f64, Array1<f64>), FluxError> {
        if positions.len() != 1 {
            return Err(FluxError::Shape {
                expected: 1,
                got: positions.len(),
                context: "RisingLine positions",
            });
        }
        Ok((positions[0], Array1::from_vec(vec![-1.0])))
    }
}

#[test]
fn optimize_hyperplane_restores_engine_when_the_search_fails() -> Result<(), FluxError> {
    let mut engine = NativeEngine::new(
        RisingLine,
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0]))
            .with_seed(5),
    )?;
    engine.set_constraint(Some(HyperplaneConstraint::from_direction(
        Array1::from_vec(vec![1.0]).view(),
        0.4,
    )?))?;
    engine.set_positions(Array1::from_vec(vec![0.2]).view())?;
    engine.shoot_velocities(5)?;
    let before = engine_state(&engine);
    let err = optimize_hyperplane(
        &mut engine,
        &HyperplaneConstraint::from_direction(Array1::from_vec(vec![1.0]).view(), 0.0)?,
        &OhtstConfig::default().with_max_outer_steps(1).with_seed(5),
    )
    .unwrap_err();
    assert!(matches!(err, FluxError::ReactantNotRelaxed { .. }));
    assert_eq!(engine_state(&engine), before);
    Ok(())
}

#[test]
fn optimize_hyperplane_returns_restore_error_after_success() -> Result<(), FluxError> {
    let mut inner = engine_at([-1.0, 0.0], 33)?;
    inner.set_constraint(Some(HyperplaneConstraint::from_direction(
        Array1::from_vec(vec![1.0, 0.0]).view(),
        -1.0,
    )?))?;
    let mut engine = RestoreRejectingEngine::new(inner);
    let seed = HyperplaneConstraint::from_direction(Array1::from_vec(vec![1.0, 0.0]).view(), 0.0)?;

    let err =
        optimize_hyperplane(&mut engine, &seed, &OhtstConfig::default().with_seed(33)).unwrap_err();
    assert_eq!(err, FluxError::InvalidConfig("position restore rejected"));
    Ok(())
}

#[test]
fn optimize_hyperplane_preserves_operation_error_when_restore_fails() -> Result<(), FluxError> {
    let mut inner = NativeEngine::new(
        RisingLine,
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0]))
            .with_seed(5),
    )?;
    inner.set_constraint(Some(HyperplaneConstraint::from_direction(
        Array1::from_vec(vec![1.0]).view(),
        0.4,
    )?))?;
    let mut engine = RestoreRejectingEngine::new(inner);
    let seed = HyperplaneConstraint::from_direction(Array1::from_vec(vec![1.0]).view(), 0.0)?;

    let err = optimize_hyperplane(
        &mut engine,
        &seed,
        &OhtstConfig::default().with_max_outer_steps(1).with_seed(5),
    )
    .unwrap_err();
    assert_eq!(
        err,
        FluxError::ReactantNotRelaxed {
            fmax: 1.0,
            tolerance: OhtstConfig::default().reactant_fmax,
        }
    );
    Ok(())
}
