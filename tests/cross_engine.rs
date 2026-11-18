//! Cross-engine harness and compatibility checks.

use ndarray::{Array1, ArrayView1};
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::engine::{
    DynamicsEngine, HyperplaneConstraint, StepReport, SurfaceVelocityPolicy, ThermostatConfig,
};

#[cfg(feature = "rgpot-session")]
use rgflux::FluxError;

struct QuarticDoubleWell1D;

impl rgsaddle::PointSurface for QuarticDoubleWell1D {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
        let dx = x[0] * x[0] - 1.0;
        let energy = dx * dx;
        let grad = Array1::from_vec(vec![4.0 * x[0] * dx]);
        Ok((energy, grad))
    }
}

#[derive(Clone, Debug)]
struct ReferenceVerlet {
    masses: Array1<f64>,
    positions: Array1<f64>,
    velocities: Array1<f64>,
    thermostat: ThermostatConfig,
}

impl ReferenceVerlet {
    fn new() -> Self {
        Self {
            masses: Array1::from_vec(vec![1.0]),
            positions: Array1::from_vec(vec![-0.8]),
            velocities: Array1::from_vec(vec![0.15]),
            thermostat: ThermostatConfig::default()
                .with_gamma(0.0)
                .with_temperature(0.0),
        }
    }
}

impl DynamicsEngine for ReferenceVerlet {
    fn dof(&self) -> usize {
        1
    }

    fn masses(&self) -> ArrayView1<'_, f64> {
        self.masses.view()
    }

    fn positions(&self) -> ArrayView1<'_, f64> {
        self.positions.view()
    }

    fn set_positions(&mut self, positions: ArrayView1<'_, f64>) -> Result<(), rgflux::FluxError> {
        self.positions.assign(&positions);
        Ok(())
    }

    fn velocities(&self) -> ArrayView1<'_, f64> {
        self.velocities.view()
    }

    fn set_velocities(&mut self, velocities: ArrayView1<'_, f64>) -> Result<(), rgflux::FluxError> {
        self.velocities.assign(&velocities);
        Ok(())
    }

    fn cell(&self) -> Option<&[f64; 9]> {
        None
    }

    fn set_cell(&mut self, _cell: Option<[f64; 9]>) -> Result<(), rgflux::FluxError> {
        Ok(())
    }

    fn thermostat(&self) -> &ThermostatConfig {
        &self.thermostat
    }

    fn set_thermostat(&mut self, thermostat: ThermostatConfig) -> Result<(), rgflux::FluxError> {
        self.thermostat = thermostat;
        Ok(())
    }

    fn constraint(&self) -> Option<&HyperplaneConstraint> {
        None
    }

    fn set_constraint(
        &mut self,
        _constraint: Option<HyperplaneConstraint>,
    ) -> Result<(), rgflux::FluxError> {
        Ok(())
    }

    fn energy_and_forces(&mut self) -> Result<(f64, Array1<f64>), rgflux::FluxError> {
        let x = self.positions[0];
        let dx = x * x - 1.0;
        let energy = dx * dx;
        let force = -4.0 * x * dx;
        Ok((energy, Array1::from_vec(vec![force])))
    }

    fn step(&mut self, dt: f64) -> Result<StepReport, rgflux::FluxError> {
        let x = self.positions[0];
        let v = self.velocities[0];
        let f0 = -4.0 * x * (x * x - 1.0);
        let v_half = v + 0.5 * dt * f0;
        let x1 = x + dt * v_half;
        let f1 = -4.0 * x1 * (x1 * x1 - 1.0);
        let v1 = v_half + 0.5 * dt * f1;
        self.positions[0] = x1;
        self.velocities[0] = v1;
        Ok(StepReport::new(
            (x1 * x1 - 1.0).powi(2),
            0.5 * v1 * v1,
            None,
        ))
    }

    fn reseed(&mut self, _seed: u64) -> Result<(), rgflux::FluxError> {
        Ok(())
    }

    fn shoot_velocities(&mut self, _seed: u64) -> Result<(), rgflux::FluxError> {
        self.velocities[0] = 0.0;
        Ok(())
    }

    fn reflect_velocity(&mut self, _normal: ArrayView1<'_, f64>) -> Result<f64, rgflux::FluxError> {
        Ok(self.velocities[0])
    }

    fn apply_dividing_surface(
        &mut self,
        _normal: ArrayView1<'_, f64>,
        _policy: SurfaceVelocityPolicy,
    ) -> Result<f64, rgflux::FluxError> {
        Ok(self.velocities[0])
    }
}

#[test]
fn native_engine_matches_scripted_reference_on_deterministic_system() {
    const TRAJECTORY_TOLERANCE: f64 = 256.0 * f64::EPSILON;
    let cfg = NativeConfig::default()
        .with_masses(Array1::from_vec(vec![1.0]))
        .with_thermostat(
            ThermostatConfig::default()
                .with_gamma(0.0)
                .with_temperature(0.0),
        )
        .with_seed(77);
    let mut native = NativeEngine::new(QuarticDoubleWell1D, cfg).expect("native engine");
    native
        .set_positions(Array1::from_vec(vec![-0.8]).view())
        .expect("set state");
    native
        .set_velocities(Array1::from_vec(vec![0.15]).view())
        .expect("set velocities");

    let mut reference = ReferenceVerlet::new();

    for _ in 0..200 {
        let nr = native.step(0.01).expect("native step");
        let rr = reference.step(0.01).expect("ref step");
        assert!((native.positions()[0] - reference.positions()[0]).abs() <= TRAJECTORY_TOLERANCE);
        assert!((native.velocities()[0] - reference.velocities()[0]).abs() <= TRAJECTORY_TOLERANCE);
        assert!((nr.potential_energy - rr.potential_energy).abs() <= TRAJECTORY_TOLERANCE);
        assert!((nr.kinetic_energy - rr.kinetic_energy).abs() <= TRAJECTORY_TOLERANCE);
    }
}

#[cfg(feature = "rgpot-session")]
#[test]
fn session_engine_contract_is_compatible_with_cross_harness_glue() {
    use rgflux::engine::rgpot_session::{
        RgpotSessionConfig, RgpotSessionEngine, SessionEngineKind,
    };

    let cfg = RgpotSessionConfig::default()
        .with_engine(SessionEngineKind::OpenMm)
        .with_masses(Array1::from_vec(vec![12.0, 12.0, 12.0]));
    let mut engine = RgpotSessionEngine::new(cfg).expect("session config");
    assert!(matches!(
        engine.reseed(123),
        Err(FluxError::BackendUnavailable(message)) if message.contains("reseed is unsupported")
    ));

    match engine.step(0.001) {
        Ok(rep) => {
            assert!(rep.potential_energy.is_finite());
            assert!(rep.kinetic_energy.is_finite());
        }
        Err(FluxError::BackendUnavailable(_)) => {}
        Err(err) => panic!("unexpected session step error: {err:?}"),
    }
}
