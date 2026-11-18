//! Analytic PAFI checks through the native constrained dynamics engine.

use std::f64::consts::PI;

use ndarray::{Array1, Array2, ArrayView1};
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::engine::{DynamicsEngine, ThermostatConfig};
use rgflux::pafi::{PafiConfig, pafi_profile};
use rgflux::stats::StatsConfig;

struct WideningDoubleWell;
struct AnisotropicHarmonic {
    ax: f64,
    ay: f64,
}

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

impl rgsaddle::PointSurface for AnisotropicHarmonic {
    fn eval(&self, r: ArrayView1<'_, f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
        let x = r[0];
        let y = r[1];
        let energy = 0.5 * (self.ax * x * x + self.ay * y * y);
        let gx = self.ax * x;
        let gy = self.ay * y;
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

fn circular_path(radius: f64, theta0: f64, theta1: f64, planes: usize) -> Array2<f64> {
    let mut path = Array2::<f64>::zeros((planes, 2));
    for i in 0..planes {
        let t = (i as f64) / ((planes - 1) as f64);
        let theta = theta0 + (theta1 - theta0) * t;
        path[[i, 0]] = radius * theta.cos();
        path[[i, 1]] = radius * theta.sin();
    }
    path
}

fn integrate_reactant_partition(arc_lengths: &[f64], free_energy: &[f64], kbt: f64) -> f64 {
    let dividing = free_energy.len() - 1;
    let mut zr_half = 0.0_f64;
    for i in 1..=dividing {
        let ds = arc_lengths[i] - arc_lengths[i - 1];
        let w0 = (-free_energy[i - 1] / kbt).exp();
        let w1 = (-free_energy[i] / kbt).exp();
        zr_half += 0.5 * ds * (w0 + w1);
    }
    2.0 * zr_half
}

fn run_barrier_case(
    masses: Vec<f64>,
    seed: u64,
    planes: usize,
    burn_in_steps: usize,
    sample_steps: usize,
    time_step: f64,
) -> (f64, f64, f64) {
    let thermal = 0.05_f64;
    let analytic_barrier = 1.0 + 0.5 * thermal * 1.5_f64.ln();
    let path = line_path(-1.0, 0.0, planes);
    let config = PafiConfig::default()
        .with_planes(planes)
        .with_burn_in_steps(burn_in_steps)
        .with_sample_steps(sample_steps)
        .with_time_step(time_step)
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
    let (barrier, se, z) = run_barrier_case(vec![1.0, 1.0], 11, 121, 60_000, 64, 0.0005);
    eprintln!(
        "isotropic widening barrier={:.8} se={:.8} z={:.4}",
        barrier, se, z
    );
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
    let (barrier, se, z) = run_barrier_case(vec![1.0, 4.0], 21, 121, 60_000, 64, 0.0005);
    eprintln!(
        "anisotropic widening barrier={:.8} se={:.8} z={:.4}",
        barrier, se, z
    );
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
    let rate = profile.normalized_rate;
    let rate_error = profile.normalized_rate_error;
    let dividing = profile.dividing_index;
    assert!(rate_error.is_finite() && rate_error > 0.0);
    assert!(dividing > 0);
    let reconstructed_zr =
        integrate_reactant_partition(&profile.arc_lengths, &profile.free_energy, thermal);
    assert!(
        (profile.reactant_partition - reconstructed_zr).abs() <= 1.0e-12,
        "reactant partition mismatch: {:.16e} vs {:.16e}",
        profile.reactant_partition,
        reconstructed_zr
    );

    let lambda_x_reactant = 8.0_f64;
    let lambda_y_reactant = 1.0_f64;
    let lambda_y_saddle = 1.5_f64;
    let htst_prefactor =
        ((lambda_x_reactant * lambda_y_reactant / lambda_y_saddle).sqrt()) / (2.0 * PI);
    let htst_rate = htst_prefactor * (-1.0 / thermal).exp();
    let z = (rate - htst_rate).abs() / rate_error;
    eprintln!(
        "normalized rate={:.8e} se={:.8e} htst={:.8e} z={:.4}",
        rate, rate_error, htst_rate, z
    );
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
fn widening_barrier_bias_shrinks_with_refinement() {
    let analytic_barrier = 1.0 + 0.5 * 0.05_f64 * 1.5_f64.ln();
    let (coarse_barrier, coarse_se, coarse_z) =
        run_barrier_case(vec![1.0, 4.0], 91, 41, 20_000, 64, 0.001);
    let (fine_barrier, fine_se, fine_z) =
        run_barrier_case(vec![1.0, 4.0], 91, 121, 60_000, 64, 0.0005);
    let coarse_abs = (coarse_barrier - analytic_barrier).abs();
    let fine_abs = (fine_barrier - analytic_barrier).abs();
    eprintln!(
        "refinement coarse barrier={:.8} se={:.8} z={:.4}; fine barrier={:.8} se={:.8} z={:.4}",
        coarse_barrier, coarse_se, coarse_z, fine_barrier, fine_se, fine_z
    );
    assert!(
        fine_abs < coarse_abs,
        "refinement should reduce bias: coarse {:.8} ± {:.8} (z={:.4}), fine {:.8} ± {:.8} (z={:.4})",
        coarse_barrier,
        coarse_se,
        coarse_z,
        fine_barrier,
        fine_se,
        fine_z
    );
    assert!(
        fine_z <= 2.0,
        "refined barrier {:.8} ± {:.8} gives z={:.4}",
        fine_barrier,
        fine_se,
        fine_z
    );
}

#[test]
fn curved_harmonic_profile_requires_jacobian_term() {
    let thermal = 0.05_f64;
    let radius = 1.2_f64;
    let ax = 3.0_f64;
    let ay = 1.0_f64;
    let theta0 = PI * 0.5;
    let theta1 = 0.0_f64;
    let path = circular_path(radius, theta0, theta1, 81);
    let config = PafiConfig::default()
        .with_planes(81)
        .with_burn_in_steps(30_000)
        .with_sample_steps(64)
        .with_time_step(0.0005)
        .with_stats(StatsConfig::default().with_min_batches(2))
        .with_seed(1234);

    let mut engine = NativeEngine::new(
        AnisotropicHarmonic { ax, ay },
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0, 1.0]))
            .with_seed(1234)
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(1.0)
                    .with_temperature(thermal)
                    .with_boltzmann(1.0),
            ),
    )
    .expect("native engine");
    engine.shoot_velocities(1234).expect("initial velocities");

    let profile = pafi_profile(&mut engine, path.view(), &config).expect("curved profile");
    let a0 = ax * theta0.cos().powi(2) + ay * theta0.sin().powi(2);
    let a1 = ax * theta1.cos().powi(2) + ay * theta1.sin().powi(2);
    let analytic_barrier = 0.5 * thermal * (a1 / a0).ln();
    let z = (profile.barrier - analytic_barrier).abs() / profile.barrier_error;
    eprintln!(
        "curved harmonic barrier={:.8} se={:.8} analytic={:.8} z={:.4}",
        profile.barrier, profile.barrier_error, analytic_barrier, z
    );
    assert!(
        z <= 2.0,
        "curved harmonic barrier {:.8} ± {:.8} (analytic {:.8}) gives z={:.4}",
        profile.barrier,
        profile.barrier_error,
        analytic_barrier,
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
