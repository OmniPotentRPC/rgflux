#![cfg(feature = "lammpc")]

use ndarray::{Array1, ArrayView1};
use rgflux::engine::lammpc::{LammpcConfig, LammpcEngine};
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::engine::{DynamicsEngine, HyperplaneConstraint, ThermostatConfig};

const ARGON_EPSILON_EV: f64 = 0.010_323;
const ARGON_SIGMA_ANGSTROM: f64 = 3.405;
const ARGON_MASS_AMU: f64 = 39.948;
const ARGON_ATOMIC_NUMBER: i32 = 18;
const SIGMA_TOLERANCE_FACTOR: f64 = 64.0;

struct LjDimerSurface {
    epsilon_ev: f64,
    sigma_angstrom: f64,
}

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

fn dimer_plane() -> HyperplaneConstraint {
    let normal = Array1::from_vec(vec![-1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    HyperplaneConstraint::from_direction(normal.view(), 4.2).expect("unit normal")
}

fn make_lammpc_engine(seed: u64) -> LammpcEngine {
    let cfg = LammpcConfig::default()
        .with_atomic_numbers(vec![ARGON_ATOMIC_NUMBER, ARGON_ATOMIC_NUMBER])
        .with_type_to_atomic_number(vec![ARGON_ATOMIC_NUMBER])
        .with_masses_by_type(vec![ARGON_MASS_AMU])
        .with_pair_style("lj/cut 8.5".to_string())
        .with_pair_coeffs(vec![format!(
            "1 1 {} {}",
            ARGON_EPSILON_EV, ARGON_SIGMA_ANGSTROM
        )])
        .with_constraint(dimer_plane())
        .with_seed(seed)
        .with_thermostat(
            ThermostatConfig::default()
                .with_gamma(1.0)
                .with_temperature(0.25)
                .with_boltzmann(1.0),
        );
    let mut engine = LammpcEngine::new(cfg).expect("lammpc engine");
    let x = Array1::from_vec(vec![0.0, 0.0, 0.0, 4.2, 0.0, 0.0]);
    engine.set_positions(x.view()).expect("set positions");
    engine.shoot_velocities(seed).expect("shoot");
    engine
}

#[test]
fn same_seed_replays_bitwise_and_different_seed_diverges() {
    let collect = |seed: u64| -> Vec<u64> {
        let mut engine = make_lammpc_engine(seed);
        let mut out = Vec::new();
        for _ in 0..128 {
            let rep = engine.step(0.001).expect("step");
            out.push(rep.potential_energy.to_bits());
            out.push(rep.kinetic_energy.to_bits());
            out.push(rep.constraint_force.expect("constraint force").to_bits());
            out.extend(engine.positions().iter().map(|v| v.to_bits()));
            out.extend(engine.velocities().iter().map(|v| v.to_bits()));
        }
        out
    };

    let a = collect(9_001);
    let b = collect(9_001);
    let c = collect(9_002);
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn constrained_coordinate_and_constraint_force_match_native_surface() {
    let mut lammpc = make_lammpc_engine(77);
    let plane = dimer_plane();
    let mut native = NativeEngine::new(
        LjDimerSurface {
            epsilon_ev: ARGON_EPSILON_EV,
            sigma_angstrom: ARGON_SIGMA_ANGSTROM,
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
            .with_seed(77)
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(1.0)
                    .with_temperature(0.25),
            ),
    )
    .expect("native engine");

    let x = Array1::from_vec(vec![0.0, 0.0, 0.0, 4.2, 0.0, 0.0]);
    let v = Array1::from_vec(vec![0.0, 0.1, -0.05, 0.0, -0.1, 0.05]);
    lammpc.set_positions(x.view()).expect("set positions");
    lammpc.set_velocities(v.view()).expect("set velocities");
    native.set_positions(x.view()).expect("set positions");
    native.set_velocities(v.view()).expect("set velocities");

    for _ in 0..1_000 {
        let rep = lammpc.step(0.001).expect("lammpc step");
        let sigma = plane
            .signed_distance(lammpc.positions())
            .expect("signed distance");
        let scale = 1.0 + plane.offset.abs() + rgmin::vecops::nrm2(lammpc.positions());
        assert!(
            sigma.abs() <= SIGMA_TOLERANCE_FACTOR * f64::EPSILON * scale,
            "constraint drift exceeded tolerance: {sigma}"
        );
        assert!(rep.constraint_force.is_some());
    }

    let native_report = native.step(0.001).expect("native step");
    let lammpc_report = lammpc.step(0.001).expect("lammpc step");
    let native_force = native_report.constraint_force.expect("native lambda");
    let lammpc_force = lammpc_report.constraint_force.expect("lammpc lambda");
    let agreement_tol = 5.0e-3_f64;
    assert!(
        (native_force - lammpc_force).abs() <= agreement_tol,
        "native={native_force} lammpc={lammpc_force}"
    );
}
