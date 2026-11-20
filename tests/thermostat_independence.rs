//! Thermostat and constraint-algorithm independence of the constrained
//! observables.
//!
//! The native engine has one thermostat, white-noise Langevin integrated with
//! BAOAB. There is no GLE and no coloured-noise thermostat, so a second
//! thermostat family cannot be run through this code. The independence check
//! therefore varies the Langevin friction `gamma` over a factor of twenty,
//! from `0.3` (weakly damped, inertial) to `6.0` (strongly damped), and it
//! crosses every friction with both constraint algorithms the engine offers,
//! RATTLE and SHAKE. The constrained mean force, the free-degree kinetic
//! temperature and the PAFI barrier and rate are equilibrium quantities, so
//! every setting must reproduce the same exact value, and the estimates at
//! different settings must agree with each other within their combined
//! statistical error.
//!
//! Each test also runs negative controls: the same estimates are compared
//! with the exact value evaluated at a deliberately wrong temperature, and
//! the comparison must fail. That shows the checks can tell temperatures
//! apart.
//!
//! All runs are seeded and use no wall-clock or thread state, so the results
//! are reproducible. The file runs in under half a minute under the test
//! profile.

use ndarray::{Array1, Array2, ArrayView1};
use rgflux::engine::native::{ConstraintProjection, NativeConfig, NativeEngine};
use rgflux::engine::{DynamicsEngine, EindirSurface, HyperplaneConstraint, ThermostatConfig};
use rgflux::pafi::{PafiConfig, pafi_profile};
use rgflux::stats::{StatsConfig, batch_means};

/// Friction coefficients spanning a factor of twenty.
const GAMMAS: [f64; 3] = [0.3, 1.6, 6.0];

/// Constraint algorithms crossed with every friction.
const PROJECTIONS: [ConstraintProjection; 2] =
    [ConstraintProjection::Rattle, ConstraintProjection::Shake];

/// Largest accepted deviation, in standard errors, from the exact value and
/// between two independent settings.
const Z_LIMIT: f64 = 3.0;

#[derive(Clone, Copy, Debug)]
struct Estimate {
    mean: f64,
    error: f64,
}

#[derive(Clone, Copy, Debug)]
struct Setting {
    gamma: f64,
    projection: ConstraintProjection,
    index: usize,
}

impl Setting {
    fn label(&self) -> String {
        format!("gamma={} {:?}", self.gamma, self.projection)
    }
}

fn settings() -> Vec<Setting> {
    let mut all = Vec::new();
    for gamma in GAMMAS {
        for projection in PROJECTIONS {
            all.push(Setting {
                gamma,
                projection,
                index: all.len(),
            });
        }
    }
    all
}

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

fn thermostat(gamma: f64, temperature: f64) -> ThermostatConfig {
    ThermostatConfig::default()
        .with_gamma(gamma)
        .with_temperature(temperature)
}

/// Seed for the `index`-th setting of an observable, distinct for every
/// setting so the runs are statistically independent.
fn seed_for(base: u64, index: usize) -> u64 {
    base + 1_000 * (index as u64 + 1)
}

fn mean_error(series: &[f64]) -> Estimate {
    let est = batch_means(series, &StatsConfig::default()).expect("batch means");
    Estimate {
        mean: est.mean,
        error: est.standard_error,
    }
}

/// Check that every estimate matches `exact` and that all pairs agree.
///
/// `bias_bound` is a deterministic discretization bound shared by every
/// setting. It widens the comparison with `exact` only, because it cancels in
/// the comparison between two settings.
fn check_independent(
    label: &str,
    estimates: &[Estimate],
    exact: f64,
    bias_bound: f64,
    quiet: bool,
) -> Result<(), String> {
    let all = settings();
    assert_eq!(estimates.len(), all.len());
    for (setting, est) in all.iter().zip(estimates) {
        if !(est.error > 0.0 && est.error.is_finite()) {
            return Err(format!(
                "{label}: {} has no usable error {}",
                setting.label(),
                est.error
            ));
        }
        let z = (est.mean - exact) / est.error.hypot(bias_bound);
        if !quiet {
            eprintln!(
                "diag {label} {} mean={:.6e} se={:.3e} exact={exact:.6e} z={z:.2}",
                setting.label(),
                est.mean,
                est.error
            );
        }
        if z.abs() > Z_LIMIT {
            return Err(format!(
                "{label}: {} mean {} exact {exact} se {} z {z:.2}",
                setting.label(),
                est.mean,
                est.error
            ));
        }
    }
    for i in 0..estimates.len() {
        for j in (i + 1)..estimates.len() {
            let (a, b) = (estimates[i], estimates[j]);
            let z = (a.mean - b.mean) / a.error.hypot(b.error);
            if !quiet {
                eprintln!(
                    "diag {label} {} vs {} difference z={z:.2}",
                    all[i].label(),
                    all[j].label()
                );
            }
            if z.abs() > Z_LIMIT {
                return Err(format!(
                    "{label}: {} gives {} but {} gives {} (z {z:.2})",
                    all[i].label(),
                    a.mean,
                    all[j].label(),
                    b.mean
                ));
            }
        }
    }
    Ok(())
}

fn assert_independent(label: &str, estimates: &[Estimate], exact: f64, bias_bound: f64) {
    if let Err(message) = check_independent(label, estimates, exact, bias_bound, false) {
        panic!("{message}");
    }
}

/// The estimates must be rejected against an exact value computed at a
/// temperature that is wrong by the given factors.
fn assert_rejects_wrong_temperature(
    label: &str,
    estimates: &[Estimate],
    factors: &[f64],
    bias_bound: impl Fn(f64) -> f64,
    exact_at: impl Fn(f64) -> f64,
) {
    for &factor in factors {
        let exact = exact_at(factor);
        let outcome = check_independent(label, estimates, exact, bias_bound(factor), true);
        eprintln!(
            "diag {label} wrong kT x{factor}: {}",
            match &outcome {
                Ok(()) => "accepted".to_string(),
                Err(message) => format!("rejected ({message})"),
            }
        );
        assert!(
            outcome.is_err(),
            "{label}: an exact value at kT x{factor} was not rejected"
        );
    }
}

/// Mean over independent replicas with the error of that mean taken from the
/// spread of the replicas.
fn replica_estimate(values: &[f64]) -> Estimate {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    Estimate {
        mean,
        error: (variance / n).sqrt(),
    }
}

fn harmonic_plane_force(setting: &Setting, seed: u64) -> (Estimate, f64) {
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
        .with_thermostat(thermostat(setting.gamma, 0.5))
        .with_constraint(plane)
        .with_projection(setting.projection)
        .with_seed(seed);
    let mut engine = NativeEngine::new(EindirSurface::new(oracle), cfg).expect("engine");
    assert_eq!(engine.thermostat().gamma, setting.gamma);
    assert_eq!(engine.projection(), setting.projection);
    engine.shoot_velocities(seed).expect("shoot");
    let dt = 0.02;
    for _ in 0..5_000 {
        engine.step(dt).expect("burn");
    }
    let mut lambdas = Vec::with_capacity(120_000);
    for _ in 0..120_000 {
        let report = engine.step(dt).expect("step");
        lambdas.push(report.constraint_force.expect("lambda"));
    }
    (mean_error(&lambdas), exact)
}

#[test]
fn harmonic_plane_df_dc_is_independent_of_friction_and_projection() {
    let mut estimates = Vec::new();
    let mut exact = 0.0;
    for setting in settings() {
        let (est, e) = harmonic_plane_force(&setting, seed_for(9_000, setting.index));
        exact = e;
        estimates.push(est);
    }
    assert_independent("harmonic dF/dc", &estimates, exact, 0.0);
    // The harmonic constrained force does not depend on temperature, so it
    // cannot be used to detect a wrong kT. A wrong plane offset is the
    // negative control here.
    let wrong = check_independent("harmonic dF/dc", &estimates, exact * 1.02, 0.0, true);
    assert!(wrong.is_err(), "a 2% error in dF/dc was not rejected");
}

#[test]
fn widening_well_rate_is_independent_of_friction_and_projection() {
    const REPLICAS: usize = 8;
    let kbt = 0.25_f64;
    let theta = 0.4_f64;
    let direction = [theta.cos(), theta.sin()];
    let entropy_scale = 0.5 * 1.5_f64.ln();
    let exact_barrier = |kbt: f64| 1.0 + entropy_scale * kbt;
    let exact_rate =
        |kbt: f64| 8.0_f64.sqrt() / std::f64::consts::TAU * (-exact_barrier(kbt) / kbt).exp();
    let mut path = Array2::zeros((21, 2));
    for i in 0..21 {
        let s = -1.0 + i as f64 / 10.0;
        path[[i, 0]] = s * direction[0];
        path[[i, 1]] = s * direction[1];
    }
    let config = PafiConfig::default()
        .with_planes(21)
        .with_burn_in_steps(4_000)
        .with_sample_steps(15_000)
        .with_time_step(0.05)
        .with_stats(StatsConfig::default().with_min_batches(8));

    // The error `pafi_profile` reports is dominated by the gap between the
    // Hermite and trapezoid integrals, which is the same for every setting
    // and hides sampling noise. The statistical error is therefore taken from
    // the spread of independently seeded replicas. The comparison with the
    // exact value gets a fixed quadrature allowance on the barrier instead.
    let barrier_allowance = 1.0e-3_f64;
    let mut barriers = Vec::new();
    let mut rates = Vec::new();
    let mut entropic_temperatures = Vec::new();
    for setting in settings() {
        let mut engine = NativeEngine::new(
            TiltedWidening { theta },
            NativeConfig::default()
                .with_masses(Array1::from_vec(vec![1.0, 1.0]))
                .with_thermostat(thermostat(setting.gamma, kbt))
                .with_projection(setting.projection),
        )
        .expect("engine");
        assert_eq!(engine.thermostat().gamma, setting.gamma);
        assert_eq!(engine.projection(), setting.projection);
        let mut barrier_samples = Vec::new();
        let mut rate_samples = Vec::new();
        let mut entropic_samples = Vec::new();
        for replica in 0..REPLICAS {
            let seed = 100_000_000 * (setting.index * REPLICAS + replica + 1) as u64;
            let profile = pafi_profile(&mut engine, path.view(), &config.clone().with_seed(seed))
                .expect("profile");
            barrier_samples.push(profile.barrier);
            rate_samples.push(
                profile
                    .normalized_rate
                    .expect("test profile has a representable normalized rate"),
            );
            // The barrier is the potential barrier 1 plus the entropic term
            // `kT ln(kappa_ratio) / 2`, so this recovers the temperature.
            entropic_samples.push((profile.barrier - 1.0) / entropy_scale);
        }
        barriers.push(replica_estimate(&barrier_samples));
        rates.push(replica_estimate(&rate_samples));
        entropic_temperatures.push(replica_estimate(&entropic_samples));
    }

    let rate_allowance = |kbt: f64| exact_rate(kbt) * barrier_allowance / kbt;
    let entropic_allowance = barrier_allowance / entropy_scale;
    assert_independent(
        "widening-well barrier",
        &barriers,
        exact_barrier(kbt),
        barrier_allowance,
    );
    assert_independent(
        "widening-well rate",
        &rates,
        exact_rate(kbt),
        rate_allowance(kbt),
    );
    assert_independent(
        "widening-well entropic kT",
        &entropic_temperatures,
        kbt,
        entropic_allowance,
    );

    let wrong = [0.85, 1.15];
    assert_rejects_wrong_temperature(
        "widening-well barrier",
        &barriers,
        &wrong,
        |_| barrier_allowance,
        |f| exact_barrier(f * kbt),
    );
    assert_rejects_wrong_temperature(
        "widening-well rate",
        &rates,
        &wrong,
        |f| rate_allowance(f * kbt),
        |f| exact_rate(f * kbt),
    );
    assert_rejects_wrong_temperature(
        "widening-well entropic kT",
        &entropic_temperatures,
        &wrong,
        |_| entropic_allowance,
        |f| f * kbt,
    );
}

#[test]
fn free_degree_kinetic_temperature_is_independent_of_friction_and_projection() {
    let kbt = 0.3_f64;
    let theta = 0.7_f64;
    let (nx, ny) = (theta.cos(), theta.sin());
    let dt = 0.05;

    let mut forces = Vec::new();
    let mut temperatures = Vec::new();
    let mut flat_temperatures = Vec::new();
    for setting in settings() {
        let seed = seed_for(17, setting.index);
        let plane = HyperplaneConstraint::new(Array1::from_vec(vec![nx, ny]), 0.0).expect("plane");
        let cfg = NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0, 2.5]))
            .with_thermostat(thermostat(setting.gamma, kbt))
            .with_constraint(plane)
            .with_projection(setting.projection)
            .with_seed(seed);
        let mut engine = NativeEngine::new(TiltedWidening { theta }, cfg).expect("engine");
        assert_eq!(engine.thermostat().gamma, setting.gamma);
        assert_eq!(engine.projection(), setting.projection);
        engine
            .set_positions(Array1::from_vec(vec![0.0, 0.0]).view())
            .expect("anchor");
        engine.shoot_velocities(seed).expect("shoot");
        for _ in 0..8_000 {
            engine.step(dt).expect("burn");
        }
        let mut force_series = Vec::with_capacity(200_000);
        let mut kinetic_series = Vec::with_capacity(200_000);
        for _ in 0..200_000 {
            let report = engine.step(dt).expect("step");
            force_series.push(report.constraint_force.expect("lambda"));
            kinetic_series.push((2.0 * report.kinetic_energy) / kbt);
        }
        forces.push(mean_error(&force_series));
        temperatures.push(mean_error(&kinetic_series));

        let flat_seed = seed_for(19, setting.index);
        let flat = HyperplaneConstraint::new(Array1::from_vec(vec![nx, ny]), 0.0).expect("flat");
        let flat_cfg = NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0, 2.5]))
            .with_thermostat(thermostat(setting.gamma, kbt))
            .with_constraint(flat)
            .with_projection(setting.projection)
            .with_seed(flat_seed);
        let surface = EindirSurface::new(rgmin::Oracle::unbounded(2, |x| {
            (0.0, Array1::zeros(x.len()))
        }));
        let mut flat_engine = NativeEngine::new(surface, flat_cfg).expect("flat");
        flat_engine.shoot_velocities(flat_seed).expect("shoot");
        let mut flat_kinetic = Vec::with_capacity(200_000);
        for step in 0..210_000 {
            let report = flat_engine.step(dt).expect("step");
            if step >= 10_000 {
                flat_kinetic.push((2.0 * report.kinetic_energy) / kbt);
            }
        }
        flat_temperatures.push(mean_error(&flat_kinetic));
    }
    assert_independent(
        "well constraint force",
        &forces,
        exact_widening_force(0.0, kbt),
        0.0,
    );
    assert_independent("well free-degree temperature", &temperatures, 1.0, 0.0);
    assert_independent("flat free-degree temperature", &flat_temperatures, 1.0, 0.0);

    // The free-degree kinetic temperature is `2 KE / kT` with the true kT, so
    // measured against a wrong kT it sits at the ratio of the two.
    let wrong = [0.9, 1.1];
    assert_rejects_wrong_temperature(
        "well free-degree temperature",
        &temperatures,
        &wrong,
        |_| 0.0,
        |f| 1.0 / f,
    );
    assert_rejects_wrong_temperature(
        "flat free-degree temperature",
        &flat_temperatures,
        &wrong,
        |_| 0.0,
        |f| 1.0 / f,
    );
    assert_rejects_wrong_temperature(
        "well constraint force",
        &forces,
        &[0.7, 1.4],
        |_| 0.0,
        |f| exact_widening_force(0.0, f * kbt),
    );
}
