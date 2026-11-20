//! Constrained mean force for white-noise Langevin and the smart-sampling GLE.
//!
//! Both thermostats go through `NativeEngine::step` and `pafi_profile`. The
//! mean force is an equilibrium average, so the two have to agree with the
//! exact value and with each other.

use ndarray::{Array1, Array2, ArrayView1};
use rgflux::engine::gle::ColoredNoise;
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

fn noises() -> [ColoredNoise; 2] {
    [ColoredNoise::White, ColoredNoise::SmartSampling]
}

#[test]
fn both_thermostats_match_exact_harmonic_df_dc() {
    let (k1, k2) = (1.4_f64, 3.1_f64);
    let offset = 0.65_f64;
    let plane =
        HyperplaneConstraint::from_direction(Array1::from_vec(vec![3.0, 4.0]).view(), offset)
            .expect("3-4-5 unit normal");
    let (n1, n2) = (plane.normal[0], plane.normal[1]);
    let exact = offset / ((n1 * n1) / k1 + (n2 * n2) / k2);
    let mut previous: Option<(f64, f64)> = None;
    for noise in noises() {
        let oracle = rgmin::Oracle::unbounded(2, move |x| {
            let e = 0.5 * k1 * x[0] * x[0] + 0.5 * k2 * x[1] * x[1];
            (e, Array1::from_vec(vec![k1 * x[0], k2 * x[1]]))
        });
        let cfg = NativeConfig::default()
            .with_masses(Array1::from_vec(vec![0.9, 2.1]))
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(1.6)
                    .with_temperature(0.5)
                    .with_noise(noise),
            )
            .with_constraint(plane.clone())
            .with_seed(9001);
        let mut engine = NativeEngine::new(EindirSurface::new(oracle), cfg).expect("engine");
        engine.shoot_velocities(9001).expect("shoot");
        for _ in 0..2_000 {
            engine.step(0.02).expect("burn");
        }
        let mut lambdas = Vec::with_capacity(20_000);
        for _ in 0..20_000 {
            let report = engine.step(0.02).expect("step");
            lambdas.push(report.constraint_force.expect("lambda"));
        }
        let est = batch_means(&lambdas, &StatsConfig::default()).expect("mean");
        let z = (est.mean - exact) / est.standard_error;
        eprintln!(
            "r10-diag harmonic {noise:?} force={:.6} se={:.6} exact={exact:.6} z={z:.3}",
            est.mean, est.standard_error
        );
        assert!(est.standard_error > 0.0 && est.standard_error < 0.02);
        assert!(
            (est.mean - exact).abs() <= 2.0 * est.standard_error,
            "{noise:?} force {} exact {exact} se {}",
            est.mean,
            est.standard_error
        );
        if let Some((other, other_se)) = previous {
            let gap = (est.mean - other).abs();
            let gap_se = (est.standard_error.powi(2) + other_se.powi(2)).sqrt();
            eprintln!(
                "r10-diag harmonic gap={gap:.6e} se={gap_se:.6e} z={:.3}",
                gap / gap_se
            );
            assert!(gap <= 2.0 * gap_se, "harmonic gap {gap} se {gap_se}");
        }
        previous = Some((est.mean, est.standard_error));
    }
}

#[test]
fn both_thermostats_match_pafi_plane_forces_at_unequal_mass() {
    let kbt = 0.25_f64;
    let theta = 0.4_f64;
    let direction = [theta.cos(), theta.sin()];
    // Equal masses make the mass-weighted constraint parallel to the Cartesian
    // normal, so a GLE that thermostats the constrained momentum still lands
    // inside the barrier error. These masses keep that direction distinct.
    let masses = [0.35_f64, 4.2_f64];
    assert!(masses[0] / masses[1] < 0.1);
    let planes = 7_usize;
    let mut path = Array2::zeros((planes, 2));
    for i in 0..planes {
        let s = -1.0 + i as f64 / (planes - 1) as f64;
        path[[i, 0]] = s * direction[0];
        path[[i, 1]] = s * direction[1];
    }
    let config = PafiConfig::default()
        .with_planes(planes)
        .with_burn_in_steps(8_000)
        .with_sample_steps(16_000)
        .with_time_step(0.05)
        .with_stats(StatsConfig::default().with_min_batches(8));
    let mut sampled = Vec::with_capacity(2);
    for noise in noises() {
        let mut engine = NativeEngine::new(
            TiltedWidening { theta },
            NativeConfig::default()
                .with_masses(Array1::from_vec(masses.to_vec()))
                .with_thermostat(
                    ThermostatConfig::default()
                        .with_temperature(kbt)
                        .with_gamma(4.0)
                        .with_noise(noise),
                ),
        )
        .expect("engine");
        let profile = pafi_profile(&mut engine, path.view(), &config).expect("profile");
        let forces = profile.gradients().to_vec();
        let errors = profile.gradient_errors().to_vec();
        assert_eq!(forces.len(), planes);
        for (plane, (&force, &error)) in forces.iter().zip(&errors).enumerate() {
            eprintln!("r11-diag plane {plane} {noise:?} force={force:.6} se={error:.6}");
            assert!(error > 0.0, "{noise:?} plane {plane} has no error");
        }
        sampled.push((forces, errors));
    }
    for plane in 0..planes {
        let (gle, gle_se) = (sampled[1].0[plane], sampled[1].1[plane]);
        let (langevin, langevin_se) = (sampled[0].0[plane], sampled[0].1[plane]);
        let gap = (gle - langevin).abs();
        let gap_se = (gle_se.powi(2) + langevin_se.powi(2)).sqrt();
        eprintln!(
            "r11-diag plane {plane} gap={gap:.6e} se={gap_se:.6e} z={:.3}",
            gap / gap_se
        );
        assert!(
            gap <= 2.0 * gap_se,
            "plane {plane} GLE {gle} Langevin {langevin} gap {gap} se {gap_se}"
        );
    }
}

#[test]
fn both_thermostats_keep_canonical_constrained_kinetic_energy() {
    let kbt = 0.3_f64;
    let theta = 0.7_f64;
    let (nx, ny) = (theta.cos(), theta.sin());
    let exact = exact_widening_force(0.0, kbt);
    let mut previous: Option<(f64, f64)> = None;
    for noise in noises() {
        let plane = HyperplaneConstraint::new(Array1::from_vec(vec![nx, ny]), 0.0).expect("plane");
        let cfg = NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0, 2.5]))
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(2.0)
                    .with_temperature(kbt)
                    .with_noise(noise),
            )
            .with_constraint(plane)
            .with_seed(17);
        let mut engine = NativeEngine::new(TiltedWidening { theta }, cfg).expect("engine");
        engine
            .set_positions(Array1::from_vec(vec![0.0, 0.0]).view())
            .expect("anchor");
        engine.shoot_velocities(17).expect("shoot");
        for _ in 0..8_000 {
            engine.step(0.05).expect("burn");
        }
        let mut forces = Vec::with_capacity(40_000);
        let mut kinetic = Vec::with_capacity(40_000);
        for _ in 0..40_000 {
            let report = engine.step(0.05).expect("step");
            forces.push(report.constraint_force.expect("lambda"));
            kinetic.push((2.0 * report.kinetic_energy) / kbt);
        }
        let force = batch_means(&forces, &StatsConfig::default()).expect("force");
        let kin = batch_means(&kinetic, &StatsConfig::default()).expect("ke");
        eprintln!(
            "r10-diag well {noise:?} force={:.6} se={:.6} exact={exact:.6} z={:.3} T_dof={:.4} se={:.4} z={:.3}",
            force.mean,
            force.standard_error,
            (force.mean - exact) / force.standard_error,
            kin.mean,
            kin.standard_error,
            (kin.mean - 1.0) / kin.standard_error
        );
        assert!((force.mean - exact).abs() <= 2.0 * force.standard_error);
        assert!((kin.mean - 1.0).abs() <= 2.0 * kin.standard_error);
        if let Some((other, other_se)) = previous {
            let gap = (force.mean - other).abs();
            let gap_se = (force.standard_error.powi(2) + other_se.powi(2)).sqrt();
            assert!(gap <= 2.0 * gap_se, "well gap {gap} se {gap_se}");
        }
        previous = Some((force.mean, force.standard_error));

        let flat = HyperplaneConstraint::new(Array1::from_vec(vec![nx, ny]), 0.0).expect("flat");
        let flat_cfg = NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0, 2.5]))
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(2.0)
                    .with_temperature(kbt)
                    .with_noise(noise),
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
            let report = flat_engine.step(0.05).expect("step");
            if step >= 10_000 {
                flat_kinetic.push((2.0 * report.kinetic_energy) / kbt);
            }
        }
        let flat_est = batch_means(&flat_kinetic, &StatsConfig::default()).expect("flat");
        eprintln!(
            "r10-diag flat {noise:?} T_dof={:.4} se={:.4} z={:.3}",
            flat_est.mean,
            flat_est.standard_error,
            (flat_est.mean - 1.0) / flat_est.standard_error
        );
        assert!((flat_est.mean - 1.0).abs() <= 2.0 * flat_est.standard_error);
    }
}
