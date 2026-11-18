//! Analytic validation suite for `rgflux::engine::native`.

use ndarray::{Array1, ArrayView1};
use rgflux::FluxError;
use rgflux::engine::gle::ColoredNoise;
use rgflux::engine::native::{ConstraintProjection, NativeConfig, NativeEngine};
use rgflux::engine::{
    DynamicsEngine, EindirSurface, HYPERPLANE_PROJECTION_TOL, HyperplaneConstraint,
    ThermostatConfig,
};
use rgflux::stats::{StatsConfig, batch_means};

/// Three-sigma acceptance band for unbiased canonical Monte Carlo averages.
const CANONICAL_SIGMA_BOUND: f64 = 3.0;

/// Machine-precision relative bound `8 * f64::EPSILON` (`2^-49`) on `n · x - c`
/// across constrained trajectories on a tilted 2D double well.
const DOUBLE_WELL_CONSTRAINT_REL_TOL: f64 = 8.0 * f64::EPSILON;

struct AnisotropicHarmonic {
    stiffness: [f64; 3],
}

impl rgsaddle::PointSurface for AnisotropicHarmonic {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
        let mut energy = 0.0;
        let mut grad = Array1::zeros(3);
        for i in 0..3 {
            energy += 0.5 * self.stiffness[i] * x[i] * x[i];
            grad[i] = self.stiffness[i] * x[i];
        }
        Ok((energy, grad))
    }
}

struct DoubleWell2D {
    barrier: f64,
    transverse_k: f64,
    coupling: f64,
}

impl rgsaddle::PointSurface for DoubleWell2D {
    fn eval(&self, r: ArrayView1<f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
        let (x, y) = (r[0], r[1]);
        let dx = x * x - 1.0;
        let dy = y - self.coupling * x;
        let energy = self.barrier * dx * dx + 0.5 * self.transverse_k * dy * dy;
        let gx = 4.0 * self.barrier * x * dx - self.coupling * self.transverse_k * dy;
        let gy = self.transverse_k * dy;
        Ok((energy, Array1::from_vec(vec![gx, gy])))
    }
}

#[test]
fn harmonic_oscillator_equipartition_matches_temperature_within_batch_se() {
    let target_temp = 0.7;
    let boltzmann = 1.0;
    let dof = 3_usize;
    let dt = 0.04;
    let burn_in = 2_000_usize;
    let samples = 30_000_usize;

    let surface = AnisotropicHarmonic {
        stiffness: [0.8, 1.5, 2.6],
    };
    let config = NativeConfig::default()
        .with_masses(Array1::from_vec(vec![1.0, 1.7, 2.3]))
        .with_thermostat(
            ThermostatConfig::default()
                .with_gamma(1.4)
                .with_temperature(target_temp)
                .with_boltzmann(boltzmann),
        )
        .with_seed(20261118);

    let mut engine = NativeEngine::new(surface, config).expect("valid harmonic engine");
    engine.shoot_velocities(20261118).expect("shoot");

    for _ in 0..burn_in {
        engine.step(dt).expect("burn-in step");
    }

    let mut kinetic_temps = Vec::with_capacity(samples);
    let mut virial_temps = Vec::with_capacity(samples);
    for _ in 0..samples {
        let report = engine.step(dt).expect("sample step");
        let t_kin = (2.0 * report.kinetic_energy) / ((dof as f64) * boltzmann);
        let t_pot = (2.0 * report.potential_energy) / ((dof as f64) * boltzmann);
        kinetic_temps.push(t_kin);
        virial_temps.push(t_pot);
    }

    let kin_est = batch_means(&kinetic_temps, &StatsConfig::default()).expect("kinetic stats");
    let pot_est = batch_means(&virial_temps, &StatsConfig::default()).expect("potential stats");

    assert!(kin_est.standard_error > 0.0 && kin_est.standard_error < 0.02);
    assert!(pot_est.standard_error > 0.0 && pot_est.standard_error < 0.03);
    assert!(
        (kin_est.mean - target_temp).abs() <= CANONICAL_SIGMA_BOUND * kin_est.standard_error,
        "kinetic temperature {} outside {} * SE ({}) of {}",
        kin_est.mean,
        CANONICAL_SIGMA_BOUND,
        kin_est.standard_error,
        target_temp
    );
    assert!(
        (pot_est.mean - target_temp).abs() <= CANONICAL_SIGMA_BOUND * pot_est.standard_error,
        "configurational temperature {} outside {} * SE ({}) of {}",
        pot_est.mean,
        CANONICAL_SIGMA_BOUND,
        pot_est.standard_error,
        target_temp
    );
}

#[test]
fn double_well_2d_constraint_holds_machine_precision_zero_under_rattle_and_shake() {
    let offset = 0.35;
    let plane =
        HyperplaneConstraint::from_direction(Array1::from_vec(vec![0.6, 0.8]).view(), offset)
            .expect("unit plane");

    for (mode, seed) in [
        (ConstraintProjection::Rattle, 101_u64),
        (ConstraintProjection::Shake, 202_u64),
    ] {
        let surface = DoubleWell2D {
            barrier: 1.2,
            transverse_k: 2.5,
            coupling: 0.3,
        };
        let cfg = NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.1, 1.9]))
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(1.0)
                    .with_temperature(0.4),
            )
            .with_constraint(plane.clone())
            .with_projection(mode)
            .with_seed(seed);

        let mut engine = NativeEngine::new(surface, cfg).expect("constrained engine");
        engine
            .set_positions(Array1::from_vec(vec![-0.7, 0.9]).view())
            .expect("projected positions");
        engine.shoot_velocities(seed).expect("projected velocities");

        let dyn_engine: &mut dyn DynamicsEngine = &mut engine;
        for _ in 0..2_500 {
            let report = dyn_engine.step(0.01).expect("constrained step");
            assert!(report.constraint_force.is_some());

            let x = dyn_engine.positions();
            let v = dyn_engine.velocities();
            let x_norm = (x[0] * x[0] + x[1] * x[1]).sqrt();
            let v_norm = (v[0] * v[0] + v[1] * v[1]).sqrt();
            let pos_scale = 1.0 + offset.abs() + x_norm;
            let vel_scale = 1.0 + v_norm;

            let sigma = plane.signed_distance(x).expect("signed distance");
            let nv = plane.normal[0] * v[0] + plane.normal[1] * v[1];
            assert!(
                sigma.abs() <= DOUBLE_WELL_CONSTRAINT_REL_TOL * pos_scale,
                "mode {mode:?}: |n · x - c| = {:.3e} exceeds {:.3e}",
                sigma.abs(),
                DOUBLE_WELL_CONSTRAINT_REL_TOL * pos_scale
            );
            assert!(
                nv.abs() <= HYPERPLANE_PROJECTION_TOL * vel_scale,
                "mode {mode:?}: |n · v| = {:.3e} exceeds {:.3e}",
                nv.abs(),
                HYPERPLANE_PROJECTION_TOL * vel_scale
            );
        }
    }
}

#[test]
fn projection_overflow_reports_a_constraint_violation() {
    let plane =
        HyperplaneConstraint::from_direction(Array1::from_vec(vec![1.0, 1.0, 0.0]).view(), 0.0)
            .expect("plane");
    let mut engine = NativeEngine::new(
        AnisotropicHarmonic {
            stiffness: [1.0, 1.0, 1.0],
        },
        NativeConfig::default()
            .with_masses(Array1::from_elem(3, 1.0))
            .with_constraint(plane),
    )
    .expect("engine");
    let positions = Array1::from_vec(vec![f64::MAX, f64::MAX, 0.0]);

    let error = engine
        .set_positions(positions.view())
        .expect_err("non-finite projection residual");
    assert!(
        matches!(
            error,
            FluxError::ConstraintViolation {
                residual,
                tolerance,
            } if !residual.is_finite() && tolerance == HYPERPLANE_PROJECTION_TOL
        ),
        "{error:?}"
    );
}

#[test]
fn harmonic_well_constraint_force_mean_matches_analytic_df_dc() {
    // V(x1, x2) = (1/2) k1 x1^2 + (1/2) k2 x2^2 on n1 x1 + n2 x2 = c.
    // Exact marginal variance of xi = n · x is k_B T (n1^2 / k1 + n2^2 / k2),
    // so F(c) = (1/2) k_eff c^2 + const and dF/dc = k_eff * c with
    // k_eff = 1 / (n1^2 / k1 + n2^2 / k2).
    let (k1, k2) = (1.4_f64, 3.1_f64);
    let offset = 0.65_f64;
    let plane =
        HyperplaneConstraint::from_direction(Array1::from_vec(vec![3.0, 4.0]).view(), offset)
            .expect("3-4-5 unit normal");
    let (n1, n2) = (plane.normal[0], plane.normal[1]);
    let k_eff = 1.0 / ((n1 * n1) / k1 + (n2 * n2) / k2);
    let analytic_df_dc = k_eff * offset;

    let oracle = rgmin::Oracle::unbounded(2, move |x| {
        let e = 0.5 * k1 * x[0] * x[0] + 0.5 * k2 * x[1] * x[1];
        let g = Array1::from_vec(vec![k1 * x[0], k2 * x[1]]);
        (e, g)
    });
    let surface = EindirSurface::new(oracle);
    let cfg = NativeConfig::default()
        .with_masses(Array1::from_vec(vec![0.9, 2.1]))
        .with_thermostat(
            ThermostatConfig::default()
                .with_gamma(1.6)
                .with_temperature(0.5),
        )
        .with_constraint(plane)
        .with_seed(9001);

    let mut engine = NativeEngine::new(surface, cfg).expect("harmonic constrained engine");
    engine.shoot_velocities(9001).expect("shoot");

    let dt = 0.02;
    for _ in 0..2_000 {
        engine.step(dt).expect("burn-in");
    }

    let n_steps = 20_000_usize;
    let mut lambdas = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        let report = engine.step(dt).expect("step");
        lambdas.push(report.constraint_force.expect("constraint active"));
    }

    let est = batch_means(&lambdas, &StatsConfig::default()).expect("lambda batch means");
    assert!(est.standard_error > 0.0 && est.standard_error < 0.02);
    assert!(
        (est.mean - analytic_df_dc).abs() <= CANONICAL_SIGMA_BOUND * est.standard_error,
        "mean constraint force {} differs from analytic dF/dc = {} by more than {} * SE ({})",
        est.mean,
        analytic_df_dc,
        CANONICAL_SIGMA_BOUND,
        est.standard_error
    );
}

#[test]
fn seeded_runs_replay_deterministically_bit_for_bit() {
    let plane = HyperplaneConstraint::from_direction(Array1::from_vec(vec![1.0, -1.0]).view(), 0.2)
        .expect("unit plane");

    let run_with_seed = |seed: u64| -> (Vec<u64>, Vec<u64>, Vec<u64>) {
        let surface = DoubleWell2D {
            barrier: 1.0,
            transverse_k: 1.8,
            coupling: 0.25,
        };
        let cfg = NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0, 1.5]))
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(1.3)
                    .with_temperature(0.35),
            )
            .with_constraint(plane.clone())
            .with_seed(seed);
        let mut engine = NativeEngine::new(surface, cfg).expect("engine");
        engine
            .set_positions(Array1::from_vec(vec![0.4, 0.1]).view())
            .expect("set pos");
        engine.shoot_velocities(seed).expect("shoot");

        let mut pos_bits = Vec::new();
        let mut vel_bits = Vec::new();
        let mut lam_bits = Vec::new();
        for _ in 0..256 {
            let rep = engine.step(0.01).expect("step");
            pos_bits.extend(engine.positions().iter().map(|v| v.to_bits()));
            vel_bits.extend(engine.velocities().iter().map(|v| v.to_bits()));
            lam_bits.push(rep.constraint_force.expect("lambda").to_bits());
        }
        (pos_bits, vel_bits, lam_bits)
    };

    let first = run_with_seed(77_777);
    let replay = run_with_seed(77_777);
    let other = run_with_seed(88_888);

    assert_eq!(first, replay);
    assert_ne!(first, other);
}

#[test]
fn reseed_reproduces_trajectory_and_distinguishes_streams() {
    let mk_engine = || {
        let surface = DoubleWell2D {
            barrier: 1.0,
            transverse_k: 1.6,
            coupling: 0.2,
        };
        let cfg = NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0, 1.3]))
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(1.4)
                    .with_temperature(0.4),
            )
            .with_seed(5);
        let mut engine = NativeEngine::new(surface, cfg).expect("engine");
        engine
            .set_positions(Array1::from_vec(vec![0.15, -0.22]).view())
            .expect("set positions");
        engine
    };

    let sample_bits = |engine: &mut NativeEngine<_>, reseed: u64| -> Vec<u64> {
        engine.reseed(reseed).expect("reseed");
        let mut bits = Vec::with_capacity(3 * 128);
        for _ in 0..128 {
            let rep = engine.step(0.01).expect("step");
            bits.push(rep.potential_energy.to_bits());
            bits.push(rep.kinetic_energy.to_bits());
            bits.extend(engine.positions().iter().map(|v| v.to_bits()));
        }
        bits
    };

    let mut a = mk_engine();
    let mut b = mk_engine();
    let mut c = mk_engine();

    let seq_a = sample_bits(&mut a, 123_456_789);
    let seq_b = sample_bits(&mut b, 123_456_789);
    let seq_c = sample_bits(&mut c, 123_456_790);

    assert_eq!(seq_a, seq_b);
    assert_ne!(seq_a, seq_c);
}

#[test]
fn reseed_and_velocity_shoot_discard_gle_auxiliary_noise() {
    let mk_engine = || {
        NativeEngine::new(
            AnisotropicHarmonic {
                stiffness: [0.8, 1.5, 2.6],
            },
            NativeConfig::default()
                .with_masses(Array1::from_vec(vec![1.0, 1.7, 2.3]))
                .with_thermostat(
                    ThermostatConfig::default()
                        .with_temperature(0.7)
                        .with_noise(ColoredNoise::SmartSampling),
                ),
        )
        .expect("GLE engine")
    };
    let positions = Array1::from_vec(vec![0.2, -0.3, 0.4]);
    let velocities = Array1::from_vec(vec![0.5, -0.1, 0.25]);
    let state_bits = |engine: &NativeEngine<_>| {
        engine
            .positions()
            .iter()
            .chain(engine.velocities())
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    };

    let mut warmed = mk_engine();
    warmed
        .set_velocities(velocities.view())
        .expect("warm velocity");
    warmed.step(0.03).expect("populate GLE auxiliaries");
    warmed
        .set_positions(positions.view())
        .expect("reset position");
    warmed
        .set_velocities(velocities.view())
        .expect("reset velocity");
    warmed.reseed_rng(91);

    let mut fresh = mk_engine();
    fresh
        .set_positions(positions.view())
        .expect("fresh position");
    fresh
        .set_velocities(velocities.view())
        .expect("fresh velocity");
    fresh.reseed(91).expect("fresh reseed");

    warmed.step(0.03).expect("reseeded step");
    fresh.step(0.03).expect("fresh step");
    assert_eq!(state_bits(&warmed), state_bits(&fresh));

    let mut warmed = mk_engine();
    warmed
        .set_velocities(velocities.view())
        .expect("warm velocity");
    warmed.step(0.03).expect("populate GLE auxiliaries");
    warmed
        .set_positions(positions.view())
        .expect("reset position");
    warmed.shoot_velocities(177).expect("shoot warmed");

    let mut fresh = mk_engine();
    fresh
        .set_positions(positions.view())
        .expect("fresh position");
    fresh.shoot_velocities(177).expect("shoot fresh");

    warmed.step(0.03).expect("shot step");
    fresh.step(0.03).expect("fresh shot step");
    assert_eq!(state_bits(&warmed), state_bits(&fresh));
}
