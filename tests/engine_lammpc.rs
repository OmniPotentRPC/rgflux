#![cfg(feature = "lammpc")]

use ndarray::{Array1, ArrayView1};
use rgflux::FluxError;
use rgflux::engine::lammpc::{LammpcConfig, LammpcEngine, MASS_VELOCITY_TO_EV};
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::engine::{DynamicsEngine, HyperplaneConstraint, ThermostatConfig};
use rgflux::stats::{StatsConfig, batch_means};
use rgsaddle::PointSurface;

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

fn metal_masses(amu: &[f64]) -> Array1<f64> {
    let mut masses = Vec::with_capacity(amu.len() * 3);
    for &mass in amu {
        masses.extend_from_slice(&[
            mass * MASS_VELOCITY_TO_EV,
            mass * MASS_VELOCITY_TO_EV,
            mass * MASS_VELOCITY_TO_EV,
        ]);
    }
    Array1::from_vec(masses)
}

fn lambda_from_surface(
    surface: &LjDimerSurface,
    positions: ArrayView1<'_, f64>,
    masses_amu: ArrayView1<'_, f64>,
    plane: &HyperplaneConstraint,
) -> f64 {
    let (_energy, gradient) = surface.eval(positions).expect("surface");
    let mut normal_accel = 0.0_f64;
    let mut metric = 0.0_f64;
    for i in 0..masses_amu.len() {
        normal_accel += plane.normal[i] * (-gradient[i]) / masses_amu[i];
        metric += plane.normal[i] * plane.normal[i] / masses_amu[i];
    }
    -normal_accel / metric
}

fn assert_mean_within_sampling_error(samples: &[f64]) {
    let estimate = batch_means(samples, &StatsConfig::default()).expect("batch means");
    let mut scale = 1.0_f64;
    for value in samples {
        if value.abs() > scale {
            scale = value.abs();
        }
    }
    let machine = 64.0 * f64::EPSILON * scale * (samples.len() as f64).sqrt();
    let statistical = 3.0 * estimate.standard_error;
    let bound = if statistical > machine {
        statistical
    } else {
        machine
    };
    assert!(
        estimate.mean.abs() <= bound,
        "mean {} exceeds bound {} (se {}, machine {})",
        estimate.mean,
        bound,
        estimate.standard_error,
        machine
    );
}

fn make_force_free_constrained_engine(seed: u64) -> LammpcEngine {
    let cfg = LammpcConfig::default()
        .with_atomic_numbers(vec![ARGON_ATOMIC_NUMBER, ARGON_ATOMIC_NUMBER])
        .with_type_to_atomic_number(vec![ARGON_ATOMIC_NUMBER])
        .with_masses_by_type(vec![ARGON_MASS_AMU])
        .with_pair_style("lj/cut 8.5".to_string())
        .with_pair_coeffs(vec![format!("1 1 0 {}", ARGON_SIGMA_ANGSTROM)])
        .with_constraint(dimer_plane())
        .with_seed(seed)
        .with_thermostat(
            ThermostatConfig::default()
                .with_gamma(0.0)
                .with_temperature(0.25)
                .with_boltzmann(1.0),
        );
    let mut engine = LammpcEngine::new(cfg).expect("lammpc engine");
    let x = Array1::from_vec(vec![0.0, 0.0, 0.0, 4.2, 0.0, 0.0]);
    engine.set_positions(x.view()).expect("set positions");
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
    let surface = LjDimerSurface {
        epsilon_ev: ARGON_EPSILON_EV,
        sigma_angstrom: ARGON_SIGMA_ANGSTROM,
    };
    let amu = Array1::from_vec(vec![
        ARGON_MASS_AMU,
        ARGON_MASS_AMU,
        ARGON_MASS_AMU,
        ARGON_MASS_AMU,
        ARGON_MASS_AMU,
        ARGON_MASS_AMU,
    ]);
    let x = Array1::from_vec(vec![0.0, 0.0, 0.0, 4.2, 0.0, 0.0]);
    let v = Array1::from_vec(vec![0.0, 0.1, -0.05, 0.0, -0.1, 0.05]);
    lammpc.set_positions(x.view()).expect("set positions");
    lammpc.set_velocities(v.view()).expect("set velocities");

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

    // Zero friction keeps the trajectory on one side of the sign error. A
    // flipped multiplier sits near -1.945e-3 instead of +1.945e-3 here, and
    // the mean of that bias is many sampling errors, not a 5e-3 window.
    let mut lammpc = LammpcEngine::new(
        LammpcConfig::default()
            .with_atomic_numbers(vec![ARGON_ATOMIC_NUMBER, ARGON_ATOMIC_NUMBER])
            .with_type_to_atomic_number(vec![ARGON_ATOMIC_NUMBER])
            .with_masses_by_type(vec![ARGON_MASS_AMU])
            .with_pair_style("lj/cut 8.5".to_string())
            .with_pair_coeffs(vec![format!(
                "1 1 {} {}",
                ARGON_EPSILON_EV, ARGON_SIGMA_ANGSTROM
            )])
            .with_constraint(plane.clone())
            .with_seed(77)
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(0.0)
                    .with_temperature(0.0)
                    .with_boltzmann(1.0),
            ),
    )
    .expect("lammpc engine");
    let mut native = NativeEngine::new(
        LjDimerSurface {
            epsilon_ev: ARGON_EPSILON_EV,
            sigma_angstrom: ARGON_SIGMA_ANGSTROM,
        },
        NativeConfig::default()
            .with_masses(metal_masses(&[ARGON_MASS_AMU, ARGON_MASS_AMU]))
            .with_constraint(plane.clone())
            .with_seed(77)
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(0.0)
                    .with_temperature(0.0)
                    .with_boltzmann(1.0),
            ),
    )
    .expect("native engine");
    lammpc.set_positions(x.view()).expect("set positions");
    lammpc.set_velocities(v.view()).expect("set velocities");
    let mut difference = Vec::with_capacity(64);
    for _ in 0..32 {
        let pos = lammpc.positions().to_owned();
        let vel = lammpc.velocities().to_owned();
        native.set_positions(pos.view()).expect("sync positions");
        native.set_velocities(vel.view()).expect("sync velocities");
        let native_force = native
            .step(0.001)
            .expect("native")
            .constraint_force
            .expect("native lambda");
        let lammpc_force = lammpc
            .step(0.001)
            .expect("lammpc")
            .constraint_force
            .expect("lammpc lambda");
        let expected = lambda_from_surface(&surface, lammpc.positions(), amu.view(), &plane);
        difference.push(lammpc_force - native_force);
        difference.push(lammpc_force - expected);
    }
    assert_mean_within_sampling_error(&difference);
}

#[test]
fn unequal_mass_constraint_force_matches_native_multiplier() {
    let heavy = 80.0_f64;
    let light = 16.0_f64;
    let plane = dimer_plane();
    let mut lammpc = LammpcEngine::new(
        LammpcConfig::default()
            .with_atomic_numbers(vec![ARGON_ATOMIC_NUMBER, 10])
            .with_type_to_atomic_number(vec![ARGON_ATOMIC_NUMBER, 10])
            .with_masses_by_type(vec![light, heavy])
            .with_pair_style("lj/cut 8.5".to_string())
            .with_pair_coeffs(vec![format!(
                "* * {} {}",
                ARGON_EPSILON_EV, ARGON_SIGMA_ANGSTROM
            )])
            .with_constraint(plane.clone())
            .with_seed(11)
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(0.0)
                    .with_temperature(0.0)
                    .with_boltzmann(1.0),
            ),
    )
    .expect("lammpc");
    let mut native = NativeEngine::new(
        LjDimerSurface {
            epsilon_ev: ARGON_EPSILON_EV,
            sigma_angstrom: ARGON_SIGMA_ANGSTROM,
        },
        NativeConfig::default()
            .with_masses(metal_masses(&[light, heavy]))
            .with_constraint(plane.clone())
            .with_seed(11)
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(0.0)
                    .with_temperature(0.0)
                    .with_boltzmann(1.0),
            ),
    )
    .expect("native");
    let x = Array1::from_vec(vec![0.0, 0.0, 0.0, 4.2, 0.0, 0.0]);
    let v = Array1::from_vec(vec![0.0, 0.2, -0.1, 0.0, -0.05, 0.1]);
    lammpc.set_positions(x.view()).expect("positions");
    lammpc.set_velocities(v.view()).expect("velocities");
    let amu = Array1::from_vec(vec![light, light, light, heavy, heavy, heavy]);
    let surface = LjDimerSurface {
        epsilon_ev: ARGON_EPSILON_EV,
        sigma_angstrom: ARGON_SIGMA_ANGSTROM,
    };
    let mut difference = Vec::with_capacity(32);
    for _ in 0..32 {
        let pos = lammpc.positions().to_owned();
        let vel = lammpc.velocities().to_owned();
        native.set_positions(pos.view()).expect("sync positions");
        native.set_velocities(vel.view()).expect("sync velocities");
        let native_force = native
            .step(0.001)
            .expect("native")
            .constraint_force
            .expect("native lambda");
        let lammpc_force = lammpc
            .step(0.001)
            .expect("lammpc")
            .constraint_force
            .expect("lammpc lambda");
        let expected = lambda_from_surface(&surface, lammpc.positions(), amu.view(), &plane);
        difference.push(lammpc_force - native_force);
        difference.push(lammpc_force - expected);
    }
    assert_mean_within_sampling_error(&difference);
}

#[test]
fn constrained_step_keeps_caller_velocity() {
    let mut engine = make_force_free_constrained_engine(17);
    let caller_velocities = Array1::from_vec(vec![0.0, 0.2, -0.1, 0.0, -0.2, 0.1]);
    engine
        .set_velocities(caller_velocities.view())
        .expect("set caller velocities");

    engine.step(0.001).expect("constrained step");

    assert_eq!(
        engine
            .velocities()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        caller_velocities
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn reflected_velocity_is_used_by_next_constrained_step() {
    let mut engine = make_force_free_constrained_engine(23);
    let initial = Array1::from_vec(vec![0.0, -0.2, 0.1, 0.0, 0.2, -0.1]);
    engine.set_velocities(initial.view()).expect("set velocity");
    let normal = Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);

    engine
        .reflect_velocity(normal.view())
        .expect("reflect velocity");
    engine.step(0.001).expect("constrained step");

    assert_eq!(engine.velocities()[1].to_bits(), 0.2_f64.to_bits());
    assert_eq!(engine.velocities()[4].to_bits(), 0.2_f64.to_bits());
}

#[test]
fn constraint_projection_overflow_is_reported() {
    let normal = Array1::from_vec(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    let plane = HyperplaneConstraint::from_direction(normal.view(), 0.0).expect("plane");
    let cfg = LammpcConfig::default()
        .with_atomic_numbers(vec![ARGON_ATOMIC_NUMBER, ARGON_ATOMIC_NUMBER])
        .with_type_to_atomic_number(vec![ARGON_ATOMIC_NUMBER])
        .with_masses_by_type(vec![ARGON_MASS_AMU])
        .with_constraint(plane);
    let mut engine = LammpcEngine::new(cfg).expect("lammpc engine");
    let original_positions = engine.positions().to_owned();
    let original_velocities = engine.velocities().to_owned();
    let positions = Array1::from_vec(vec![f64::MAX, 0.0, 0.0, f64::MAX, 0.0, 0.0]);

    assert!(matches!(
        engine.set_positions(positions.view()),
        Err(FluxError::ConstraintViolation { .. })
    ));
    assert_eq!(engine.positions(), original_positions.view());
    assert_eq!(engine.velocities(), original_velocities.view());
}

#[test]
fn rejected_constraint_update_preserves_the_previous_engine_state() {
    let mut engine = LammpcEngine::new(LammpcConfig::default()).expect("lammpc engine");
    let positions = Array1::from_vec(vec![f64::MAX, 0.0, 0.0, f64::MAX, 0.0, 0.0]);
    engine
        .set_positions(positions.view())
        .expect("unconstrained positions");
    let normal = Array1::from_vec(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    let plane = HyperplaneConstraint::from_direction(normal.view(), 0.0).expect("plane");

    assert!(matches!(
        engine.set_constraint(Some(plane)),
        Err(FluxError::ConstraintViolation { .. })
    ));
    assert_eq!(engine.constraint(), None);
    assert_eq!(engine.positions(), positions.view());
}

#[test]
fn command_injection_text_is_rejected_with_the_field_name() {
    let cases = [
        (
            LammpcConfig::default().with_pair_style("lj/cut 8.5; clear".to_string()),
            "LammpcConfig::pair_style",
        ),
        (
            LammpcConfig::default().with_boundary("p p p\nclear".to_string()),
            "LammpcConfig::boundary",
        ),
        (
            LammpcConfig::default().with_extra_setup(vec!["neighbor 2.0 bin\0clear".to_string()]),
            "LammpcConfig::extra_setup",
        ),
    ];

    for (config, expected_field) in cases {
        let error = match LammpcEngine::new(config) {
            Ok(_) => panic!("unsafe command text was accepted"),
            Err(error) => error,
        };
        assert_eq!(
            error,
            FluxError::InvalidLammpcCommand {
                field: expected_field
            }
        );
    }
}

fn velocity_bits(engine: &LammpcEngine) -> Vec<u64> {
    engine
        .velocities()
        .iter()
        .map(|value| value.to_bits())
        .collect()
}

#[test]
fn caller_velocity_survives_reinitialization_and_the_first_step() {
    let mut fresh = make_force_free_constrained_engine(17);
    fresh
        .step(0.001)
        .expect("first step keeps the zero velocity");
    assert_eq!(velocity_bits(&fresh), vec![0.0_f64.to_bits(); 6]);

    let mut engine = make_force_free_constrained_engine(17);
    let caller = Array1::from_vec(vec![0.0, 0.2, -0.1, 0.0, -0.2, 0.1]);
    engine
        .set_velocities(caller.view())
        .expect("set caller velocities");
    let caller_bits: Vec<u64> = caller.iter().map(|value| value.to_bits()).collect();
    engine.step(0.001).expect("first constrained step");
    assert_eq!(velocity_bits(&engine), caller_bits);

    let positions = engine.positions().to_owned();
    engine
        .set_positions(positions.view())
        .expect("reinit via positions");
    engine.step(0.001).expect("step after set_positions");
    assert_eq!(velocity_bits(&engine), caller_bits);

    engine
        .set_cell(engine.cell().copied())
        .expect("reinit via cell");
    engine.step(0.001).expect("step after set_cell");
    assert_eq!(velocity_bits(&engine), caller_bits);

    engine
        .set_thermostat(*engine.thermostat())
        .expect("reinit via thermostat");
    engine.step(0.001).expect("step after set_thermostat");
    assert_eq!(velocity_bits(&engine), caller_bits);

    engine
        .set_constraint(Some(dimer_plane()))
        .expect("reinit via constraint");
    engine.step(0.001).expect("step after set_constraint");
    assert_eq!(velocity_bits(&engine), caller_bits);

    engine.reseed(17).expect("reinit via reseed");
    engine.step(0.001).expect("step after reseed");
    assert_eq!(velocity_bits(&engine), caller_bits);

    engine.step(0.002).expect("step after a dt change");
    assert_eq!(velocity_bits(&engine), caller_bits);
}

#[test]
fn unconstrained_nve_positions_stay_unwrapped_across_a_periodic_face() {
    let mut engine = LammpcEngine::new(
        LammpcConfig::default()
            .with_atomic_numbers(vec![ARGON_ATOMIC_NUMBER, ARGON_ATOMIC_NUMBER])
            .with_type_to_atomic_number(vec![ARGON_ATOMIC_NUMBER])
            .with_masses_by_type(vec![ARGON_MASS_AMU])
            .with_pair_style("lj/cut 8.5".to_string())
            .with_pair_coeffs(vec![format!(
                "1 1 {} {}",
                ARGON_EPSILON_EV, ARGON_SIGMA_ANGSTROM
            )])
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(0.0)
                    .with_temperature(0.0)
                    .with_boltzmann(1.0),
            )
            .with_cell(Some([30.0, 0.0, 0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 30.0])),
    )
    .expect("nve engine");
    // 30.09 Å plus 100 Å/ps for 0.001 ps lands at 30.19. The wrapped image is 0.19.
    let start = Array1::from_vec(vec![30.09, 0.0, 0.0, 15.0, 15.0, 15.0]);
    engine.set_positions(start.view()).expect("positions");
    engine
        .set_velocities(Array1::from_vec(vec![100.0, 0.0, 0.0, 0.0, 0.0, 0.0]).view())
        .expect("velocities");
    engine.step(0.001).expect("crossing step");
    let crossed = engine.positions()[0];
    let expected = 30.09_f64 + 100.0 * 0.001;
    let tol = 4096.0 * f64::EPSILON * expected.abs();
    assert!(
        (crossed - expected).abs() <= tol,
        "wrapped coordinate {crossed}, unwrapped target {expected}"
    );
    assert!(crossed > 30.0, "position folded into the cell: {crossed}");
}

#[test]
fn single_atom_nve_kinetic_energy_follows_the_mass_and_velocity() {
    let mut engine = LammpcEngine::new(
        LammpcConfig::default()
            .with_atomic_numbers(vec![ARGON_ATOMIC_NUMBER])
            .with_type_to_atomic_number(vec![ARGON_ATOMIC_NUMBER])
            .with_masses_by_type(vec![ARGON_MASS_AMU])
            .with_pair_style("lj/cut 8.5".to_string())
            .with_pair_coeffs(vec![format!(
                "1 1 {} {}",
                ARGON_EPSILON_EV, ARGON_SIGMA_ANGSTROM
            )])
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(0.0)
                    .with_temperature(0.0)
                    .with_boltzmann(1.0),
            ),
    )
    .expect("single atom");
    engine
        .set_positions(Array1::from_vec(vec![1.0, 2.0, 3.0]).view())
        .expect("positions");
    let velocity = Array1::from_vec(vec![0.2, -0.1, 0.05]);
    engine.set_velocities(velocity.view()).expect("velocities");
    let report = engine.step(0.001).expect("step");
    let speed_sq = 0.2 * 0.2 + 0.1 * 0.1 + 0.05 * 0.05;
    let expected = 0.5 * ARGON_MASS_AMU * speed_sq * MASS_VELOCITY_TO_EV;
    let scale = 1.0 + expected.abs();
    assert!(report.kinetic_energy > 0.0, "single-atom KE was zero");
    assert!(
        (report.kinetic_energy - expected).abs() <= 64.0 * f64::EPSILON * scale,
        "kinetic {} expected {}",
        report.kinetic_energy,
        expected
    );
}

#[test]
fn coincident_atoms_return_non_finite_error() {
    let mut engine = LammpcEngine::new(
        LammpcConfig::default()
            .with_atomic_numbers(vec![ARGON_ATOMIC_NUMBER, ARGON_ATOMIC_NUMBER])
            .with_type_to_atomic_number(vec![ARGON_ATOMIC_NUMBER])
            .with_masses_by_type(vec![ARGON_MASS_AMU])
            .with_pair_style("lj/cut 8.5".to_string())
            .with_pair_coeffs(vec![format!(
                "1 1 {} {}",
                ARGON_EPSILON_EV, ARGON_SIGMA_ANGSTROM
            )])
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(0.0)
                    .with_temperature(0.0)
                    .with_boltzmann(1.0),
            ),
    )
    .expect("engine");
    engine
        .set_positions(Array1::zeros(6).view())
        .expect("coincident positions");
    let err = engine.energy_and_forces().expect_err("non-finite pair");
    assert!(
        matches!(err, FluxError::NonFinite(_)),
        "expected NonFinite, got {err:?}"
    );
}
