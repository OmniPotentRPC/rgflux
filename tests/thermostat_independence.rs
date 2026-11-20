//! Langevin half of the constrained-mean-force checks.
//!
//! Develop's native engine has one thermostat: white-noise Langevin, then
//! RATTLE. There is no GLE and no `ColoredNoise`, so a second thermostat
//! cannot be run through this code. The numbers below are that single
//! production path against the exact constrained force and the exact TST rate.

use ndarray::{Array1, Array2, ArrayView1};
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::engine::{DynamicsEngine, EindirSurface, HyperplaneConstraint, ThermostatConfig};
use rgflux::pafi::{PafiConfig, pafi_profile};
use rgflux::stats::{StatsConfig, batch_means};

struct TiltedWidening {
    theta: f64,
}

impl rgsaddle::PointSurface for TiltedWidening {
    fn eval(&self, r: ArrayView1<f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
        let (nx, ny) = (self.theta.cos(), self.theta.sin());
        let s = nx * r[0] + ny * r[1];
        let q = -ny * r[0] + nx * r[1];
        let well = 1.0 - s * s;
        let kappa = 1.5 + 0.5 * s;
        let energy = well * well + 0.5 * kappa * q * q;
        let d_ds = -4.0 * s * well + 0.25 * q * q;
        let d_dq = kappa * q;
        Ok((
            energy,
            Array1::from_vec(vec![d_ds * nx - d_dq * ny, d_ds * ny + d_dq * nx]),
        ))
    }
}

fn exact_widening_force(s: f64, kbt: f64) -> f64 {
    let well = 1.0 - s * s;
    let kappa = 1.5 + 0.5 * s;
    -4.0 * s * well + 0.25 * kbt / kappa
}

#[test]
fn langevin_harmonic_plane_matches_exact_df_dc() {
    let (k1, k2) = (1.4_f64, 3.1_f64);
    let offset = 0.65_f64;
    let plane =
        HyperplaneConstraint::from_direction(Array1::from_vec(vec![3.0, 4.0]).view(), offset)
            .expect("3-4-5 unit normal");
    let (n1, n2) = (plane.normal[0], plane.normal[1]);
    let exact = offset / ((n1 * n1) / k1 + (n2 * n2) / k2);
    let oracle = rgmin::Oracle::unbounded(2, move |x| {
        let e = 0.5 * k1 * x[0] * x[0] + 0.5 * k2 * x[1] * x[1];
        (e, Array1::from_vec(vec![k1 * x[0], k2 * x[1]]))
    });
    let cfg = NativeConfig::default()
        .with_masses(Array1::from_vec(vec![0.9, 2.1]))
        .with_thermostat(
            ThermostatConfig::default()
                .with_gamma(1.6)
                .with_temperature(0.5),
        )
        .with_constraint(plane)
        .with_seed(9001);
    let mut engine = NativeEngine::new(EindirSurface::new(oracle), cfg).expect("engine");
    engine.shoot_velocities(9001).expect("shoot");
    let dt = 0.02;
    for _ in 0..2_000 {
        engine.step(dt).expect("burn");
    }
    let mut lambdas = Vec::with_capacity(20_000);
    for _ in 0..20_000 {
        let report = engine.step(dt).expect("step");
        lambdas.push(report.constraint_force.expect("lambda"));
    }
    let est = batch_means(&lambdas, &StatsConfig::default()).expect("mean");
    let z = (est.mean - exact) / est.standard_error;
    eprintln!(
        "r9b-diag harmonic Langevin force={:.6} se={:.6} exact={exact:.6} z={z:.3}",
        est.mean, est.standard_error
    );
    assert!(est.standard_error > 0.0 && est.standard_error < 0.02);
    assert!(
        (est.mean - exact).abs() <= 2.0 * est.standard_error,
        "force {} exact {exact} se {}",
        est.mean,
        est.standard_error
    );
}

#[test]
fn langevin_widening_well_rate_matches_exact_tst() {
    let kbt = 0.25_f64;
    let theta = 0.4_f64;
    let direction = [theta.cos(), theta.sin()];
    let exact_barrier = 1.0 + 0.5 * kbt * 1.5_f64.ln();
    let exact_rate = 8.0_f64.sqrt() / std::f64::consts::TAU * (-exact_barrier / kbt).exp();
    let mut path = Array2::zeros((7, 2));
    for i in 0..7 {
        let s = -1.0 + i as f64 / 6.0;
        path[[i, 0]] = s * direction[0];
        path[[i, 1]] = s * direction[1];
    }
    let config = PafiConfig::default()
        .with_planes(7)
        .with_burn_in_steps(4_000)
        .with_sample_steps(8_000)
        .with_time_step(0.05)
        .with_stats(StatsConfig::default().with_min_batches(8));
    let mut engine = NativeEngine::new(
        TiltedWidening { theta },
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0, 1.0]))
            .with_thermostat(
                ThermostatConfig::default()
                    .with_temperature(kbt)
                    .with_gamma(4.0),
            ),
    )
    .expect("engine");
    let profile = pafi_profile(&mut engine, path.view(), &config).expect("profile");
    let barrier_z = (profile.barrier - exact_barrier) / profile.barrier_error;
    let rate_z = (profile.normalized_rate - exact_rate) / profile.normalized_rate_error;
    eprintln!(
        "r9b-diag 2d-well Langevin barrier={:.6} se={:.6} z={barrier_z:.3} rate={:.6e} se={:.6e} exact={exact_rate:.6e} z={rate_z:.3}",
        profile.barrier,
        profile.barrier_error,
        profile.normalized_rate,
        profile.normalized_rate_error
    );
    assert!((profile.barrier - exact_barrier).abs() <= 2.0 * profile.barrier_error);
    assert!((profile.normalized_rate - exact_rate).abs() <= 2.0 * profile.normalized_rate_error);
}

#[test]
fn langevin_constrained_kinetic_temperature_is_canonical() {
    let kbt = 0.3_f64;
    let theta = 0.7_f64;
    let (nx, ny) = (theta.cos(), theta.sin());
    let exact = exact_widening_force(0.0, kbt);
    let plane = HyperplaneConstraint::new(Array1::from_vec(vec![nx, ny]), 0.0).expect("plane");
    let cfg = NativeConfig::default()
        .with_masses(Array1::from_vec(vec![1.0, 2.5]))
        .with_thermostat(
            ThermostatConfig::default()
                .with_gamma(2.0)
                .with_temperature(kbt),
        )
        .with_constraint(plane)
        .with_seed(17);
    let mut engine = NativeEngine::new(TiltedWidening { theta }, cfg).expect("engine");
    engine
        .set_positions(Array1::from_vec(vec![0.0, 0.0]).view())
        .expect("anchor");
    engine.shoot_velocities(17).expect("shoot");
    let dt = 0.05;
    for _ in 0..8_000 {
        engine.step(dt).expect("burn");
    }
    let mut forces = Vec::with_capacity(40_000);
    let mut kinetic = Vec::with_capacity(40_000);
    for _ in 0..40_000 {
        let report = engine.step(dt).expect("step");
        forces.push(report.constraint_force.expect("lambda"));
        kinetic.push((2.0 * report.kinetic_energy) / kbt);
    }
    let force = batch_means(&forces, &StatsConfig::default()).expect("force");
    let kin = batch_means(&kinetic, &StatsConfig::default()).expect("ke");
    let force_z = (force.mean - exact) / force.standard_error;
    let temp_z = (kin.mean - 1.0) / kin.standard_error;
    eprintln!(
        "r9b-diag well Langevin force={:.6} se={:.6} exact={exact:.6} z={force_z:.3} T_dof={:.4} se={:.4} z={temp_z:.3}",
        force.mean, force.standard_error, kin.mean, kin.standard_error
    );
    assert!((force.mean - exact).abs() <= 2.0 * force.standard_error);
    assert!((kin.mean - 1.0).abs() <= 2.0 * kin.standard_error);

    let flat = HyperplaneConstraint::new(Array1::from_vec(vec![nx, ny]), 0.0).expect("flat");
    let flat_cfg = NativeConfig::default()
        .with_masses(Array1::from_vec(vec![1.0, 2.5]))
        .with_thermostat(
            ThermostatConfig::default()
                .with_gamma(2.0)
                .with_temperature(kbt),
        )
        .with_constraint(flat)
        .with_seed(19);
    let surface = EindirSurface::new(rgmin::Oracle::unbounded(2, |x| {
        (0.0, Array1::zeros(x.len()))
    }));
    let mut flat_engine = NativeEngine::new(surface, flat_cfg).expect("flat");
    flat_engine.shoot_velocities(19).expect("shoot");
    let mut flat_kinetic = Vec::new();
    for step in 0..80_000 {
        let report = flat_engine.step(dt).expect("step");
        if step >= 10_000 {
            flat_kinetic.push((2.0 * report.kinetic_energy) / kbt);
        }
    }
    let flat_est = batch_means(&flat_kinetic, &StatsConfig::default()).expect("flat");
    let flat_z = (flat_est.mean - 1.0) / flat_est.standard_error;
    eprintln!(
        "r9b-diag flat Langevin T_dof={:.4} se={:.4} z={flat_z:.3}",
        flat_est.mean, flat_est.standard_error
    );
    assert!((flat_est.mean - 1.0).abs() <= 2.0 * flat_est.standard_error);
}
