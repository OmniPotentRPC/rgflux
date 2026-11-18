//! Analytic PAFI checks through the native constrained dynamics engine.

use std::f64::consts::{PI, TAU};

use ndarray::{Array1, Array2, ArrayView1};
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::engine::{DynamicsEngine, ThermostatConfig};
use rgflux::pafi::{PafiConfig, pafi_profile};
use rgflux::stats::StatsConfig;

struct WideningDoubleWell;

impl rgsaddle::PointSurface for WideningDoubleWell {
    fn eval(&self, r: ArrayView1<'_, f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
        let x = r[0];
        let y = r[1];
        let well = (1.0 - x * x).powi(2);
        let k = 1.5 + 0.5 * x;
        let energy = well + 0.5 * k * y * y;
        let gx = -4.0 * x * (1.0 - x * x) + 0.25 * y * y;
        let gy = k * y;
        Ok((energy, Array1::from_vec(vec![gx, gy])))
    }
}

fn line_path(start: f64, end: f64, planes: usize) -> Array2<f64> {
    let mut path = Array2::<f64>::zeros((planes, 2));
    for i in 0..planes {
        let t = (i as f64) / ((planes - 1) as f64);
        path[[i, 0]] = start + (end - start) * t;
        path[[i, 1]] = 0.0;
    }
    path
}

fn tilted_path(start: f64, end: f64, planes: usize, slope: f64) -> Array2<f64> {
    let mut path = Array2::<f64>::zeros((planes, 2));
    for i in 0..planes {
        let t = (i as f64) / ((planes - 1) as f64);
        let x = start + (end - start) * t;
        path[[i, 0]] = x;
        path[[i, 1]] = slope * x;
    }
    path
}

fn normalized_rate_from_profile(
    arc_lengths: &[f64],
    free_energy: &[f64],
    standard_error: &[f64],
    kbt: f64,
) -> (f64, f64, usize) {
    let dividing = free_energy.len() - 1;
    let mut zr_half = 0.0_f64;
    for i in 1..=dividing {
        let ds = arc_lengths[i] - arc_lengths[i - 1];
        let w0 = (-free_energy[i - 1] / kbt).exp();
        let w1 = (-free_energy[i] / kbt).exp();
        zr_half += 0.5 * ds * (w0 + w1);
    }
    let zr = 2.0 * zr_half;
    let rate = (kbt / TAU).sqrt() * (-free_energy[dividing] / kbt).exp() / zr;

    let mut d_zr_d_f = vec![0.0_f64; free_energy.len()];
    for i in 1..=dividing {
        let ds = arc_lengths[i] - arc_lengths[i - 1];
        let w0 = (-free_energy[i - 1] / kbt).exp();
        let w1 = (-free_energy[i] / kbt).exp();
        d_zr_d_f[i - 1] += -ds * w0 / kbt;
        d_zr_d_f[i] += -ds * w1 / kbt;
    }
    let mut var_zr = 0.0_f64;
    for i in 0..=dividing {
        var_zr += d_zr_d_f[i] * d_zr_d_f[i] * standard_error[i] * standard_error[i];
    }
    let sigma_zr = var_zr.sqrt();
    let sigma_ln = ((standard_error[dividing] / kbt).powi(2) + (sigma_zr / zr).powi(2)).sqrt();
    let rate_se = rate * sigma_ln;
    (rate, rate_se, dividing)
}

fn run_barrier_case(masses: Vec<f64>, seed: u64) -> (f64, f64, f64) {
    let thermal = 0.05_f64;
    let analytic_barrier = 1.0 + 0.5 * thermal * 1.5_f64.ln();
    let planes = 41usize;
    let path = line_path(-1.0, 0.0, planes);
    let config = PafiConfig::default()
        .with_planes(planes)
        .with_burn_in_steps(20_000)
        .with_sample_steps(64)
        .with_time_step(0.001)
        .with_stats(StatsConfig::default().with_min_batches(2))
        .with_seed(seed);

    let mut engine = NativeEngine::new(
        WideningDoubleWell,
        NativeConfig::default()
            .with_masses(Array1::from_vec(masses))
            .with_seed(seed)
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(1.5)
                    .with_temperature(thermal)
                    .with_boltzmann(1.0),
            ),
    )
    .expect("native engine");
    engine.shoot_velocities(seed).expect("initial velocities");

    let profile_result = pafi_profile(&mut engine, path.view(), &config);
    assert!(
        profile_result.is_ok(),
        "pafi_profile returned {:?}",
        profile_result
    );
    let profile = profile_result.expect("checked above");
    let z = (profile.barrier - analytic_barrier).abs() / profile.barrier_error;
    (profile.barrier, profile.barrier_error, z)
}

#[test]
fn pafi_widening_barrier_matches_analytic_within_reported_se() {
    let (barrier, se, z) = run_barrier_case(vec![1.0, 1.0], 11);
    assert!(
        z <= 2.0,
        "barrier {:.8} ± {:.8} gives z={:.4}",
        barrier,
        se,
        z
    );
}

#[test]
fn pafi_widening_barrier_mass_anisotropic_matches_analytic_within_reported_se() {
    let (barrier, se, z) = run_barrier_case(vec![1.0, 4.0], 21);
    assert!(
        z <= 2.0,
        "barrier {:.8} ± {:.8} gives z={:.4}",
        barrier,
        se,
        z
    );
}

#[test]
fn normalized_pafi_rate_matches_harmonic_limit_htst_within_se() {
    let thermal = 0.02_f64;
    let planes = 41usize;
    let path = line_path(-1.0, 0.0, planes);
    let config = PafiConfig::default()
        .with_planes(planes)
        .with_burn_in_steps(20_000)
        .with_sample_steps(64)
        .with_time_step(0.001)
        .with_stats(StatsConfig::default().with_min_batches(2))
        .with_seed(7);

    let mut engine = NativeEngine::new(
        WideningDoubleWell,
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0, 4.0]))
            .with_seed(7)
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(1.3)
                    .with_temperature(thermal)
                    .with_boltzmann(1.0),
            ),
    )
    .expect("native engine");
    engine.shoot_velocities(7).expect("initial velocities");

    let profile_result = pafi_profile(&mut engine, path.view(), &config);
    assert!(
        profile_result.is_ok(),
        "pafi_profile returned {:?}",
        profile_result
    );
    let profile = profile_result.expect("checked above");
    let (rate, rate_error, dividing) = normalized_rate_from_profile(
        &profile.arc_lengths,
        &profile.free_energy,
        &profile.standard_error,
        thermal,
    );
    assert!(rate_error.is_finite() && rate_error > 0.0);
    assert!(dividing > 0);

    let lambda_x_reactant = 8.0_f64;
    let lambda_y_reactant = 1.0_f64;
    let lambda_y_saddle = 1.5_f64;
    let htst_prefactor =
        ((lambda_x_reactant * lambda_y_reactant / lambda_y_saddle).sqrt()) / (2.0 * PI);
    let htst_rate = htst_prefactor * (-1.0 / thermal).exp();
    let z = (rate - htst_rate).abs() / rate_error;
    assert!(
        z <= 2.0,
        "rate {:.8e} ± {:.8e} versus HTST {:.8e} gives z={:.4}",
        rate,
        rate_error,
        htst_rate,
        z
    );
}

#[test]
fn arc_lengths_reflect_mass_weighting() {
    let thermal = 0.03_f64;
    let planes = 9usize;
    let path = tilted_path(-1.0, 0.0, planes, 0.3);
    let config = PafiConfig::default()
        .with_planes(planes)
        .with_burn_in_steps(200)
        .with_sample_steps(64)
        .with_time_step(0.001)
        .with_stats(StatsConfig::default().with_min_batches(2))
        .with_seed(99);

    let make_engine = |masses: Vec<f64>, seed: u64| {
        let mut engine = NativeEngine::new(
            WideningDoubleWell,
            NativeConfig::default()
                .with_masses(Array1::from_vec(masses))
                .with_seed(seed)
                .with_thermostat(
                    ThermostatConfig::default()
                        .with_gamma(1.3)
                        .with_temperature(thermal)
                        .with_boltzmann(1.0),
                ),
        )
        .expect("native engine");
        engine.shoot_velocities(seed).expect("initial velocities");
        engine
    };

    let mut unit_mass = make_engine(vec![1.0, 1.0], 99);
    let mut skew_mass = make_engine(vec![1.0, 4.0], 100);
    let p1 = pafi_profile(&mut unit_mass, path.view(), &config).expect("profile unit");
    let p2 = pafi_profile(&mut skew_mass, path.view(), &config).expect("profile skew");
    let s1 = *p1.arc_lengths.last().expect("last arc");
    let s2 = *p2.arc_lengths.last().expect("last arc");
    assert!(
        (s1 - s2).abs() > 1.0e-3,
        "mass-weighted arc lengths should differ: s1={}, s2={}",
        s1,
        s2
    );
}
