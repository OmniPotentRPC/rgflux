//! Scripted engine and closed-form helpers shared by the `ohtst_*` integration tests.

#![allow(dead_code)]

use ndarray::{Array1, ArrayView1};
use rgflux::FluxError;
use rgflux::engine::{
    DynamicsEngine, HyperplaneConstraint, StepReport, SurfaceVelocityPolicy, ThermostatConfig,
};
use rgflux::ohtst::{OhtstConfig, OhtstEstimate, optimize_hyperplane};
use std::cell::Cell;
use std::f64::consts::TAU;

pub const HALF_WIDTH: f64 = 2.0;
pub const CURVATURE: f64 = 2.0;
pub const BARRIER: f64 = HALF_WIDTH * HALF_WIDTH / 4.0;
pub const THERMAL: f64 = 0.25;
pub const RELATIVE: f64 = 1.0e-9;

pub type Potential = Box<dyn Fn(ArrayView1<'_, f64>) -> (f64, Array1<f64>)>;

pub type ConstraintForce = Box<dyn Fn(&HyperplaneConstraint, usize) -> f64>;

#[derive(Default)]
pub struct Log {
    pub reseeds: Vec<u64>,
    pub shoots: Vec<u64>,
    pub step_sizes: Vec<f64>,
    pub step_offsets: Vec<f64>,
    pub first_step_positions: Vec<Array1<f64>>,
}

pub struct Scripted {
    pub positions: Array1<f64>,
    pub velocities: Array1<f64>,
    pub masses: Array1<f64>,
    pub thermostat: ThermostatConfig,
    pub constraint: Option<HyperplaneConstraint>,
    pub potential: Potential,
    pub constraint_force: ConstraintForce,
    pub steps_since_reseed: usize,
    pub log: Log,
    pub sampling: Cell<bool>,
    pub retained_reads: Cell<usize>,
}

impl Scripted {
    pub fn new(
        positions: Vec<f64>,
        masses: Vec<f64>,
        potential: Potential,
        constraint_force: ConstraintForce,
    ) -> Self {
        let dof = positions.len();
        Self {
            positions: Array1::from_vec(positions),
            velocities: Array1::zeros(dof),
            masses: Array1::from_vec(masses),
            thermostat: ThermostatConfig::default().with_temperature(THERMAL),
            constraint: None,
            potential,
            constraint_force,
            steps_since_reseed: 0,
            log: Log::default(),
            sampling: Cell::new(false),
            retained_reads: Cell::new(0),
        }
    }

    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.thermostat = self.thermostat.with_temperature(temperature);
        self
    }
}

impl DynamicsEngine for Scripted {
    fn dof(&self) -> usize {
        self.positions.len()
    }

    fn masses(&self) -> ArrayView1<'_, f64> {
        self.masses.view()
    }

    fn positions(&self) -> ArrayView1<'_, f64> {
        if self.sampling.get() {
            self.retained_reads.set(self.retained_reads.get() + 1);
        }
        self.positions.view()
    }

    fn set_positions(&mut self, positions: ArrayView1<'_, f64>) -> Result<(), FluxError> {
        self.positions.assign(&positions);
        Ok(())
    }

    fn velocities(&self) -> ArrayView1<'_, f64> {
        self.velocities.view()
    }

    fn set_velocities(&mut self, velocities: ArrayView1<'_, f64>) -> Result<(), FluxError> {
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
        self.sampling.set(false);
        self.constraint = constraint;
        Ok(())
    }

    fn energy_and_forces(&mut self) -> Result<(f64, Array1<f64>), FluxError> {
        Ok((self.potential)(self.positions.view()))
    }

    fn step(&mut self, dt: f64) -> Result<StepReport, FluxError> {
        let plane = self
            .constraint
            .clone()
            .ok_or(FluxError::InvalidConfig("scripted step needs a constraint"))?;
        if self.steps_since_reseed == 0 {
            self.log.first_step_positions.push(self.positions.clone());
        }
        self.log.step_sizes.push(dt);
        self.log.step_offsets.push(plane.offset);
        let lambda = (self.constraint_force)(&plane, self.steps_since_reseed);
        self.steps_since_reseed += 1;
        Ok(StepReport::new(0.0, 0.0, Some(lambda)))
    }

    fn reseed(&mut self, seed: u64) -> Result<(), FluxError> {
        self.log.reseeds.push(seed);
        self.steps_since_reseed = 0;
        Ok(())
    }

    fn shoot_velocities(&mut self, seed: u64) -> Result<(), FluxError> {
        self.log.shoots.push(seed);
        self.sampling.set(true);
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

pub fn well(half_width: f64) -> Potential {
    Box::new(move |x| {
        let u = x[0];
        let gap = u * u - half_width * half_width;
        let scale = half_width * half_width;
        (
            gap * gap / (4.0 * scale),
            Array1::from_vec(vec![-u * gap / scale]),
        )
    })
}

pub fn harmonic_force(kappa: f64, minimum: f64) -> ConstraintForce {
    Box::new(move |plane, _| kappa * (plane.offset - minimum))
}

pub fn well_engine(start: f64) -> Scripted {
    let minimum = if start < 0.0 { -HALF_WIDTH } else { HALF_WIDTH };
    Scripted::new(
        vec![start],
        vec![1.0],
        well(HALF_WIDTH),
        harmonic_force(CURVATURE, minimum),
    )
}

pub fn plane(direction: &[f64], offset: f64) -> HyperplaneConstraint {
    HyperplaneConstraint::from_direction(Array1::from_vec(direction.to_vec()).view(), offset)
        .expect("valid plane")
}

pub fn harmonic_rate(kappa: f64, mu: f64, barrier: f64, thermal: f64) -> f64 {
    (kappa / mu).sqrt() / TAU * (-barrier / thermal).exp()
}

pub fn assert_close(actual: f64, expected: f64, relative: f64, what: &str) {
    let allowed = relative * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= allowed,
        "{what}: {actual} against {expected} (allowed {allowed})"
    );
}

pub fn invalid(message: &'static str) -> FluxError {
    FluxError::InvalidConfig(message)
}

pub fn quiet() -> OhtstConfig {
    OhtstConfig::default().with_seed(3)
}

pub fn estimate(
    engine: &mut Scripted,
    seed: &HyperplaneConstraint,
    config: &OhtstConfig,
) -> OhtstEstimate {
    optimize_hyperplane(engine, seed, config).expect("OH-TST estimate")
}

pub fn error(
    engine: &mut Scripted,
    seed: &HyperplaneConstraint,
    config: &OhtstConfig,
) -> FluxError {
    optimize_hyperplane(engine, seed, config).expect_err("OH-TST must reject this input")
}

pub fn oblique_well() -> Potential {
    let inner = well(HALF_WIDTH);
    Box::new(move |x| {
        let u = 0.6 * x[0] + 0.8 * x[1];
        let (energy, slope) = inner(Array1::from_vec(vec![u]).view());
        (
            energy,
            Array1::from_vec(vec![0.6 * slope[0], 0.8 * slope[0]]),
        )
    })
}

pub fn shifted_well(top: f64) -> Potential {
    let inner = well(HALF_WIDTH);
    Box::new(move |x| inner(Array1::from_vec(vec![x[0] - top]).view()))
}
