//! Cross-engine harness and compatibility checks.

use ndarray::{Array1, ArrayView1};
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::engine::{
    DynamicsEngine, HyperplaneConstraint, StepReport, SurfaceVelocityPolicy, ThermostatConfig,
};
#[cfg(feature = "lammpc")]
use rgflux::stats::{StatsConfig, batch_means};

#[cfg(feature = "rgpot-session")]
use rgflux::FluxError;
#[cfg(feature = "lammpc")]
use rgflux::engine::lammpc::{LammpcConfig, LammpcEngine};

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

#[cfg(feature = "lammpc")]
struct LjDimerSurface {
    epsilon_ev: f64,
    sigma_angstrom: f64,
}

#[cfg(feature = "lammpc")]
impl rgsaddle::PointSurface for LjDimerSurface {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
        let dx = x[3] - x[0];
        let dy = x[4] - x[1];
        let dz = x[5] - x[2];
        let r2 = dx * dx + dy * dy + dz * dz;
        let r = r2.sqrt();
        let sr = self.sigma_angstrom / r;
        let sr2 = sr * sr;
        let sr6 = sr2 * sr2 * sr2;
        let sr12 = sr6 * sr6;
        let energy = 4.0 * self.epsilon_ev * (sr12 - sr6);
        let prefactor = 24.0 * self.epsilon_ev * (2.0 * sr12 - sr6) / (r * r);
        let mut gradient = Array1::zeros(6);
        gradient[0] = prefactor * dx;
        gradient[1] = prefactor * dy;
        gradient[2] = prefactor * dz;
        gradient[3] = -gradient[0];
        gradient[4] = -gradient[1];
        gradient[5] = -gradient[2];
        Ok((energy, gradient))
    }
}

#[cfg(feature = "lammpc")]
#[test]
fn lammpc_langevin_temperature_agrees_with_target_and_native() {
    const TARGET_TEMPERATURE: f64 = 0.25;
    const BOLTZMANN: f64 = 1.0;
    const DT: f64 = 0.001;
    const BURN_IN: usize = 1_000;
    const SAMPLES: usize = 1_200;
    const ARGON_MASS_AMU: f64 = 39.948;
    const ARGON_Z: i32 = 18;
    const EPSILON_EV: f64 = 0.010_323;
    const SIGMA_ANGSTROM: f64 = 3.405;
    const SIGMA_BAND: f64 = 3.0;

    let plane = HyperplaneConstraint::from_direction(
        Array1::from_vec(vec![-1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).view(),
        4.2,
    )
    .expect("plane");

    let mk_native = || {
        NativeEngine::new(
            LjDimerSurface {
                epsilon_ev: EPSILON_EV,
                sigma_angstrom: SIGMA_ANGSTROM,
            },
            NativeConfig::default()
                .with_masses(Array1::from_vec(vec![
                    ARGON_MASS_AMU,
                    ARGON_MASS_AMU,
                    ARGON_MASS_AMU,
                    ARGON_MASS_AMU,
                    ARGON_MASS_AMU,
                    ARGON_MASS_AMU,
                ]))
                .with_constraint(plane.clone())
                .with_seed(100)
                .with_thermostat(
                    ThermostatConfig::default()
                        .with_gamma(1.0)
                        .with_temperature(TARGET_TEMPERATURE)
                        .with_boltzmann(BOLTZMANN),
                ),
        )
        .expect("native")
    };

    let mk_lammpc = || {
        LammpcEngine::new(
            LammpcConfig::default()
                .with_atomic_numbers(vec![ARGON_Z, ARGON_Z])
                .with_type_to_atomic_number(vec![ARGON_Z])
                .with_masses_by_type(vec![ARGON_MASS_AMU])
                .with_pair_style("lj/cut 8.5".to_string())
                .with_pair_coeffs(vec![format!("1 1 {} {}", EPSILON_EV, SIGMA_ANGSTROM)])
                .with_constraint(plane.clone())
                .with_seed(100)
                .with_thermostat(
                    ThermostatConfig::default()
                        .with_gamma(1.0)
                        .with_temperature(TARGET_TEMPERATURE)
                        .with_boltzmann(BOLTZMANN),
                ),
        )
        .expect("lammpc")
    };

    let mut native = mk_native();
    let mut lammpc = mk_lammpc();
    let start = Array1::from_vec(vec![0.0, 0.0, 0.0, 4.2, 0.0, 0.0]);
    native.set_positions(start.view()).expect("native pos");
    lammpc.set_positions(start.view()).expect("lammpc pos");
    native.shoot_velocities(100).expect("native shoot");
    lammpc.shoot_velocities(100).expect("lammpc shoot");

    for _ in 0..BURN_IN {
        native.step(DT).expect("native burn");
        lammpc.step(DT).expect("lammpc burn");
    }

    // lammpc's constrained thermostat path removes rigid translation plus one
    // plane mode for a two-atom system, leaving an effective thermal count of 3.
    let dof = 3.0;
    let mut native_t = Vec::with_capacity(SAMPLES);
    let mut lammpc_t = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let nr = native.step(DT).expect("native sample");
        let lr = lammpc.step(DT).expect("lammpc sample");
        native_t.push((2.0 * nr.kinetic_energy) / (dof * BOLTZMANN));
        lammpc_t.push((2.0 * lr.kinetic_energy) / (dof * BOLTZMANN));
    }

    let stats_cfg = StatsConfig::default().with_batch_size(200);
    let native_est = batch_means(&native_t, &stats_cfg).expect("native stats");
    let lammpc_est = batch_means(&lammpc_t, &stats_cfg).expect("lammpc stats");
    println!(
        "lammpc_mean_temperature={} lammpc_standard_error={} native_mean_temperature={} native_standard_error={}",
        lammpc_est.mean, lammpc_est.standard_error, native_est.mean, native_est.standard_error
    );

    assert!(
        (lammpc_est.mean - TARGET_TEMPERATURE).abs() <= SIGMA_BAND * lammpc_est.standard_error,
        "lammpc mean={} se={} target={}",
        lammpc_est.mean,
        lammpc_est.standard_error,
        TARGET_TEMPERATURE
    );
    let combined = (native_est.standard_error.powi(2) + lammpc_est.standard_error.powi(2)).sqrt();
    assert!(
        (lammpc_est.mean - native_est.mean).abs() <= SIGMA_BAND * combined,
        "lammpc mean={} se={} native mean={} se={} combined={}",
        lammpc_est.mean,
        lammpc_est.standard_error,
        native_est.mean,
        native_est.standard_error,
        combined
    );
}
