//! Analytic PAFI checks through the native constrained dynamics engine.
//!
//! Every statistical bound is derived in `tests/pafi_support/mod.rs` from a
//! suite false failure rate of one in a thousand shared by all
//! `FAMILY_SIZE` statistical assertions. Coverage targets the Hermite
//! integral, which is the expectation of the estimator. Reproducible with
//! `cargo test --features rgpot --test pafi_native -- --nocapture`; set
//! `PAFI_SEED_OFFSET` to draw an independent suite.

mod pafi_support;

use std::f64::consts::{PI, TAU};

use ndarray::{Array1, Array2, ArrayView1};
use pafi_support::{FAMILY_SIZE, Z_CRIT, assert_below, assert_coverage, block_stats, se_cap, seed};
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::engine::{DynamicsEngine, ThermostatConfig};
use rgflux::pafi::{
    FreeEnergyProfile, HermiteIntegral, MeanForceIntegral, PafiConfig, pafi_profile,
};
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
        let energy = 0.5 * (self.ax * x * x + 4.0 * self.ay * y * y);
        let gradient = Array1::from_vec(vec![self.ax * x, 4.0 * self.ay * y]);
        Ok((energy, gradient))
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

fn mass_weighted_circular_path(
    radius: f64,
    theta_start: f64,
    theta_end: f64,
    images: usize,
) -> Array2<f64> {
    let mut path = Array2::zeros((images, 2));
    for i in 0..images {
        let fraction = (i as f64) / ((images - 1) as f64);
        let theta = theta_start + fraction * (theta_end - theta_start);
        path[[i, 0]] = radius * theta.cos();
        path[[i, 1]] = 0.5 * radius * theta.sin();
    }
    path
}

fn run_barrier_case(
    masses: Vec<f64>,
    seed: u64,
    planes: usize,
    burn_in_steps: usize,
    sample_steps: usize,
    time_step: f64,
) -> (f64, f64, f64) {
    let thermal = 0.2_f64;
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
        "pafi_profile returned {profile_result:?}",
    );
    let profile = profile_result.expect("checked above");
    let z = (profile.barrier - analytic_barrier).abs() / profile.barrier_error;
    (profile.barrier, profile.barrier_error, z)
}

#[test]
fn pafi_widening_barrier_matches_analytic_within_reported_se() {
    let (barrier, se, z) = run_barrier_case(vec![1.0, 1.0], 11, 31, 10_000, 10_000, 0.001);
    eprintln!("isotropic widening barrier={barrier:.8} se={se:.8} z={z:.4}");
    assert!(z <= 2.0, "barrier {barrier:.8} ± {se:.8} gives z={z:.4}");
}

#[test]
fn pafi_widening_barrier_mass_anisotropic_matches_analytic_within_reported_se() {
    let (barrier, se, z) = run_barrier_case(vec![1.0, 4.0], 21, 31, 20_000, 10_000, 0.001);
    eprintln!("anisotropic widening barrier={barrier:.8} se={se:.8} z={z:.4}");
    assert!(z <= 2.0, "barrier {barrier:.8} ± {se:.8} gives z={z:.4}");
}

const CURVED_THERMAL: f64 = 0.2;
const CURVED_RADIUS: f64 = 1.2;
const CURVED_FORCE_CONSTANT_X: f64 = 3.0;
const CURVED_FORCE_CONSTANT_Y: f64 = 1.0;
const CURVED_PLANES: usize = 9;
const CURVED_SEED: u64 = 1_234;

/// Largest disagreement, as a fraction of the analytic barrier, between an
/// end plane of the curved path and the same plane seen from a path that
/// continues past it. Common random numbers make the two runs share their
/// sampling noise, so the residual is not statistical: it is 0.03% of the
/// barrier with not-a-knot endpoints and 7.8% with natural ones.
const END_PLANE_AGREEMENT: f64 = 1.0 / 64.0;

/// Profile along the quarter circle `theta` in `[pi / 2, 0]`, sampled at
/// `CURVED_PLANES + extra_images` equally spaced images and planes. With
/// `extra_images = 1` the arc continues one image spacing past `theta = 0`,
/// so the plane at `theta = 0` is an interior plane of the longer path while
/// every plane before it keeps its position, tangent and per-plane seed.
fn curved_profile(extra_images: usize) -> FreeEnergyProfile {
    let theta_start = PI * 0.5;
    let theta_end = 0.0_f64;
    let images = CURVED_PLANES + extra_images;
    let spacing = (theta_end - theta_start) / ((CURVED_PLANES - 1) as f64);
    let path = mass_weighted_circular_path(
        CURVED_RADIUS,
        theta_start,
        theta_end + spacing * (extra_images as f64),
        images,
    );
    let config = PafiConfig::default()
        .with_planes(images)
        .with_burn_in_steps(3_000)
        .with_sample_steps(54_000)
        .with_stats(StatsConfig::default().with_min_batches(4))
        .with_seed(CURVED_SEED);
    let mut engine = NativeEngine::new(
        AnisotropicHarmonic {
            ax: CURVED_FORCE_CONSTANT_X,
            ay: CURVED_FORCE_CONSTANT_Y,
        },
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0, 4.0]))
            .with_seed(CURVED_SEED)
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(1.5)
                    .with_temperature(CURVED_THERMAL)
                    .with_boltzmann(1.0),
            ),
    )
    .expect("native engine");
    engine
        .shoot_velocities(CURVED_SEED)
        .expect("initial velocities");
    pafi_profile(&mut engine, path.view(), &config).expect("curved profile")
}

#[test]
fn curved_mass_weighted_harmonic_profile_requires_jacobian() {
    let (profile, longer) = std::thread::scope(|scope| {
        let base = scope.spawn(|| curved_profile(0));
        let longer = scope.spawn(|| curved_profile(1));
        (
            base.join().expect("curved profile thread"),
            longer.join().expect("longer curved profile thread"),
        )
    });
    let analytic_barrier =
        0.5 * CURVED_THERMAL * (CURVED_FORCE_CONSTANT_X / CURVED_FORCE_CONSTANT_Y).ln();
    let z = (profile.barrier - analytic_barrier).abs() / profile.barrier_error;
    eprintln!(
        "curved harmonic barrier={:.8} se={:.8} analytic={:.8} z={:.4}",
        profile.barrier, profile.barrier_error, analytic_barrier, z
    );
    assert_eq!(profile.dividing_index, CURVED_PLANES - 1);
    assert!(
        z <= 4.0,
        "curved barrier {:.8} ± {:.8} versus analytic {:.8} gives z={:.4}",
        profile.barrier,
        profile.barrier_error,
        analytic_barrier,
        z
    );

    // Natural spline endpoints zero the normal curvature `dn/ds` on the end
    // planes. That shifts the last free-energy increment by far less than the
    // sampling error, so compare against the longer path, which sees the same
    // plane with the curvature of an interior plane.
    let tolerance = END_PLANE_AGREEMENT * analytic_barrier;
    let worst = (1..CURVED_PLANES)
        .map(|k| {
            let increment = profile.free_energy[k] - profile.free_energy[k - 1];
            let reference = longer.free_energy[k] - longer.free_energy[k - 1];
            (increment - reference).abs()
        })
        .fold(0.0_f64, f64::max);
    eprintln!("end-plane disagreement={worst:.8} tolerance={tolerance:.8}");
    assert!(
        worst <= tolerance,
        "free-energy increments differ by up to {worst:.8} from those of a longer path, \
         above the tolerance {tolerance:.8}"
    );
}

#[test]
fn normalized_pafi_rate_matches_analytic_profile() {
    let thermal = 0.2_f64;
    let planes = 41usize;
    let path = line_path(-1.0, 0.0, planes);
    let config = PafiConfig::default()
        .with_planes(planes)
        .with_burn_in_steps(10_000)
        .with_sample_steps(10_000)
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
        "pafi_profile returned {profile_result:?}",
    );
    let profile = profile_result.expect("checked above");
    let rate = profile.normalized_rate;
    let rate_error = profile.normalized_rate_error;
    let dividing = profile.dividing_index;
    assert!(rate_error.is_finite() && rate_error >= 0.0);
    assert!(dividing > 0);

    // Reactant Hessian at x = -1: V_xx = 8 along the unit normal, mass 1.
    let omega_0 = 8.0_f64.sqrt();
    assert!((profile.reactant_frequency - omega_0).abs() <= 1.0e-6 * omega_0);
    let gaussian_partition = (2.0 * PI * thermal).sqrt() / omega_0;
    assert!((profile.reactant_partition - gaussian_partition).abs() <= 1.0e-6 * gaussian_partition);
    assert!(
        (profile.reactant_log_partition - gaussian_partition.ln()).abs()
            <= 1.0e-6 * gaussian_partition.ln().abs()
    );
    let formula = omega_0 / (2.0 * PI) * (-profile.barrier / thermal).exp();
    assert!((rate - formula).abs() <= 1.0e-6 * formula);
    assert!((profile.normalized_log_rate - formula.ln()).abs() <= 1.0e-6 * formula.ln().abs());

    let intervals = 20_000usize;
    let lower = -2.5_f64;
    let dx = (0.0 - lower) / (intervals as f64);
    let analytic_free_energy =
        |x: f64| (1.0 - x * x).powi(2) + 0.5 * thermal * (1.5 + 0.5 * x).ln();
    let reference = analytic_free_energy(-1.0);
    let mut partition_sum = 0.0_f64;
    for i in 0..=intervals {
        let x = lower + (i as f64) * dx;
        let weight = if i == 0 || i == intervals {
            1.0
        } else if i % 2 == 0 {
            2.0
        } else {
            4.0
        };
        partition_sum += weight * (-(analytic_free_energy(x) - reference) / thermal).exp();
    }
    let analytic_partition = dx * partition_sum / 3.0;
    let analytic_barrier = analytic_free_energy(0.0) - reference;
    let analytic_rate =
        (thermal / (2.0 * PI)).sqrt() * (-analytic_barrier / thermal).exp() / analytic_partition;
    // Gaussian normalization against the exact full line integral, plus two
    // reported standard errors, the same coverage the barrier tests use.
    let predicted = omega_0 / (2.0 * PI) * (-analytic_barrier / thermal).exp();
    let tolerance = (predicted - analytic_rate).abs() + 2.0 * rate_error;
    let absolute_error = (rate - analytic_rate).abs();
    let relative_error = absolute_error / analytic_rate;
    eprintln!(
        "normalized rate={rate:.8e} se={rate_error:.8e} analytic={analytic_rate:.8e} relative error={relative_error:.4} tolerance={tolerance:.4e}",
    );
    assert!(
        absolute_error <= tolerance,
        "rate {rate:.8e} ± {rate_error:.8e} versus analytic {analytic_rate:.8e} misses the Gaussian model by more than {tolerance:.4e}",
    );
}

#[test]
fn sampling_refinement_restores_analytic_coverage() {
    let (coarse_barrier, coarse_se, coarse_z) =
        run_barrier_case(vec![1.0, 4.0], 91, 31, 20_000, 2_500, 0.001);
    let (fine_barrier, fine_se, fine_z) =
        run_barrier_case(vec![1.0, 4.0], 91, 31, 20_000, 10_000, 0.001);
    eprintln!(
        "refinement coarse barrier={coarse_barrier:.8} se={coarse_se:.8} z={coarse_z:.4}; fine barrier={fine_barrier:.8} se={fine_se:.8} z={fine_z:.4}",
    );
    assert!(
        coarse_z > 4.0 && fine_z <= 2.0,
        "refinement should move the analytic value into two-SE coverage: coarse z={coarse_z:.4}, fine barrier {fine_barrier:.8} ± {fine_se:.8} gives z={fine_z:.4}",
    );
}

#[test]
fn arc_lengths_reflect_mass_weighting() {
    let thermal = 0.03_f64;
    let planes = 9usize;
    let slope = 0.3_f64;
    let start = -1.0_f64;
    let end = 0.0_f64;
    let path = tilted_path(start, end, planes, slope);
    let spacing = 1.0 / ((planes - 1) as f64);
    // Along the path the potential is
    // `V(x, slope * x) = (1 - x^2)^2 + 0.5 * (1.5 + 0.5 * x) * (slope * x)^2`.
    // The quartic part has third derivative `4 * 3 * 2 * x`, and the cubic part
    // `0.5 * 0.5 * slope^2 * x^3` has the constant `3 * 2 * 0.5 * 0.5 * slope^2`.
    // The path force `V'` therefore has curvature at most `third_derivative_bound`,
    // and the trapezoid error over `[start, end]` is at most
    // `spacing^2 * (end - start) * third_derivative_bound / 12`.
    // Masses only rescale arc length uniformly on a straight path, so the same
    // bound holds for both mass sets.
    let third_derivative_bound =
        4.0 * 3.0 * 2.0 * start.abs().max(end.abs()) + 3.0 * 2.0 * 0.5 * 0.5 * slope * slope;
    let trapezoid_error_bound = spacing * spacing * (end - start) * third_derivative_bound / 12.0;
    let config = PafiConfig::default()
        .with_planes(planes)
        .with_burn_in_steps(200)
        .with_sample_steps(64)
        .with_time_step(0.001)
        .with_zero_temperature_tol(trapezoid_error_bound)
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
        "mass-weighted arc lengths should differ: s1={s1}, s2={s2}",
    );
}

const THERMAL: f64 = 0.2;
const PLANES: usize = 21;
const TIME_STEP: f64 = 0.05;
const BURN_IN: usize = 200;

const ISO_SAMPLES: usize = 10_000;
const ANISO_SAMPLES: usize = 20_000;
const RATE_SAMPLES: usize = 20_000;
const REFINE_COARSE_SAMPLES: usize = 5_000;
const REFINE_FINE_SAMPLES: usize = 20_000;

const CURVED_BURN_IN: usize = 3_000;
const CURVED_SAMPLES: usize = 54_000;
// A block of `CURVED_SAMPLES / 40` steps has to be long compared with the
// autocorrelation time. At the default step `0.002` a block spans `2.7` time
// units. At `0.01` it spans `13.5`.
const CURVED_TIME_STEP: f64 = 0.01;
const CURVED_SE_CAP: f64 = se_cap(3.28e-3, 0.072);

/// Exact mean force `dF/dx` of the widening double well on the plane `x`.
fn widening_mean_force(x: f64) -> f64 {
    -4.0 * x * (1.0 - x * x) + 0.25 * THERMAL / (1.5 + 0.5 * x)
}

/// Continuum barrier `F(0) - F(-1)` of the widening double well.
fn widening_analytic_barrier() -> f64 {
    1.0 + 0.5 * THERMAL * 1.5_f64.ln()
}

/// Hermite integral of [`widening_mean_force`] from `x = -1` to `x = 0`.
fn widening_hermite_barrier(planes: usize) -> f64 {
    let denominator = (planes - 1) as f64;
    let arcs: Vec<f64> = (0..planes).map(|i| (i as f64) / denominator).collect();
    let gradients: Vec<f64> = arcs
        .iter()
        .map(|arc| widening_mean_force(-1.0 + *arc))
        .collect();
    HermiteIntegral
        .cumulative(&arcs, &gradients)
        .expect("widening hermite barrier")
        .values[planes - 1]
}

fn widening_profile(
    masses: Vec<f64>,
    gamma: f64,
    seed_base: u64,
    samples: usize,
) -> FreeEnergyProfile {
    let seed_value = seed(seed_base);
    let path = line_path(-1.0, 0.0, PLANES);
    let config = PafiConfig::default()
        .with_planes(PLANES)
        .with_burn_in_steps(BURN_IN)
        .with_sample_steps(samples)
        .with_time_step(TIME_STEP)
        .with_stats(block_stats(samples))
        .with_seed(seed_value);
    let mut engine = NativeEngine::new(
        WideningDoubleWell,
        NativeConfig::default()
            .with_masses(Array1::from_vec(masses))
            .with_seed(seed_value)
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(gamma)
                    .with_temperature(THERMAL)
                    .with_boltzmann(1.0),
            ),
    )
    .expect("native engine");
    engine
        .shoot_velocities(seed_value)
        .expect("initial velocities");
    pafi_profile(&mut engine, path.view(), &config).expect("widening profile")
}

/// End sample of a profile that rises to its last plane, with the interpolant
/// maximum agreeing to rounding.
fn end_point(profile: &FreeEnergyProfile) -> (f64, f64) {
    let last = profile.free_energy.len() - 1;
    assert_eq!(
        profile.dividing_index, last,
        "dividing plane should be the last sample"
    );
    let scale = profile.barrier.abs().max(1.0);
    let allowance = 1024.0 * f64::EPSILON * scale;
    let gap = (profile.barrier - profile.free_energy[last]).abs();
    assert!(
        gap <= allowance,
        "barrier {:.16e} and last sample {:.16e} differ by more than rounding",
        profile.barrier,
        profile.free_energy[last]
    );
    // The interpolant maximum and a stored sample can disagree by a unit in
    // the last place. The sample maximum is the barrier within that rounding.
    assert!(
        profile
            .free_energy
            .iter()
            .all(|&value| value <= profile.barrier + allowance),
        "barrier must be the profile maximum"
    );
    (profile.free_energy[last], profile.standard_error[last])
}

#[test]
fn widening_mean_force_integrates_to_analytic_barrier() {
    let intervals = 20_000usize;
    let dx = 1.0 / (intervals as f64);
    let mut sum = 0.0_f64;
    for i in 0..=intervals {
        let weight = if i == 0 || i == intervals {
            1.0
        } else if i % 2 == 0 {
            2.0
        } else {
            4.0
        };
        sum += weight * widening_mean_force(-1.0 + (i as f64) * dx);
    }
    let simpson = dx * sum / 3.0;
    let tolerance = 1024.0 * f64::EPSILON;
    assert!(
        (simpson - widening_analytic_barrier()).abs() <= tolerance,
        "mean force integral {simpson:.16} differs from analytic barrier {:.16}",
        widening_analytic_barrier()
    );
}

#[test]
fn error_budget_isotropic_barrier_covers_hermite_expectation() {
    let profile = widening_profile(vec![1.0, 1.0], 1.5, 11, ISO_SAMPLES);
    let (barrier, standard_error) = end_point(&profile);
    assert_coverage(
        "isotropic barrier",
        barrier,
        standard_error,
        widening_hermite_barrier(PLANES),
    );
}

#[test]
fn error_budget_anisotropic_barrier_covers_hermite_expectation() {
    let profile = widening_profile(vec![1.0, 4.0], 1.5, 21, ANISO_SAMPLES);
    let (barrier, standard_error) = end_point(&profile);
    assert_coverage(
        "anisotropic barrier",
        barrier,
        standard_error,
        widening_hermite_barrier(PLANES),
    );
}

fn budget_curved_profile(extra_images: usize, seed_value: u64) -> FreeEnergyProfile {
    let theta_start = PI * 0.5;
    let theta_end = 0.0_f64;
    let images = CURVED_PLANES + extra_images;
    let spacing = (theta_end - theta_start) / ((CURVED_PLANES - 1) as f64);
    let path = mass_weighted_circular_path(
        CURVED_RADIUS,
        theta_start,
        theta_end + spacing * (extra_images as f64),
        images,
    );
    let config = PafiConfig::default()
        .with_planes(images)
        .with_burn_in_steps(CURVED_BURN_IN)
        .with_sample_steps(CURVED_SAMPLES)
        .with_time_step(CURVED_TIME_STEP)
        .with_stats(block_stats(CURVED_SAMPLES))
        .with_seed(seed_value);
    let mut engine = NativeEngine::new(
        AnisotropicHarmonic {
            ax: CURVED_FORCE_CONSTANT_X,
            ay: CURVED_FORCE_CONSTANT_Y,
        },
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0, 4.0]))
            .with_seed(seed_value)
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(1.5)
                    .with_temperature(THERMAL)
                    .with_boltzmann(1.0),
            ),
    )
    .expect("native engine");
    engine
        .shoot_velocities(seed_value)
        .expect("initial velocities");
    pafi_profile(&mut engine, path.view(), &config).expect("curved profile")
}

/// Exact mean force `-dF/dtheta` on the plane at angle `theta`.
fn curved_mean_force(theta: f64) -> f64 {
    let (sin, cos) = theta.sin_cos();
    THERMAL * (CURVED_FORCE_CONSTANT_X - CURVED_FORCE_CONSTANT_Y) * sin * cos
        / (CURVED_FORCE_CONSTANT_X * cos * cos + CURVED_FORCE_CONSTANT_Y * sin * sin)
}

fn curved_hermite_barrier() -> f64 {
    let step = PI * 0.5 / ((CURVED_PLANES - 1) as f64);
    let arcs: Vec<f64> = (0..CURVED_PLANES).map(|k| (k as f64) * step).collect();
    let gradients: Vec<f64> = (0..CURVED_PLANES)
        .map(|k| curved_mean_force(PI * 0.5 - (k as f64) * step))
        .collect();
    HermiteIntegral
        .cumulative(&arcs, &gradients)
        .expect("curved hermite barrier")
        .values[CURVED_PLANES - 1]
}

#[test]
fn error_budget_curved_profile_covers_hermite_expectation() {
    let seed_value = seed(CURVED_SEED);
    let profile = budget_curved_profile(0, seed_value);
    let (barrier, standard_error) = end_point(&profile);
    let expected = curved_hermite_barrier();
    // Without the Jacobian term the canonical mean barrier is zero. The
    // coverage window around `expected`, using the standard error cap as an
    // upper bound, excludes that limit.
    assert!(
        expected > Z_CRIT * CURVED_SE_CAP,
        "expected barrier {expected:.8} is not separated from the zero Jacobian limit"
    );
    assert_below("curved barrier SE", standard_error, CURVED_SE_CAP);
    assert_coverage("curved barrier", barrier, standard_error, expected);
}

#[test]
fn error_budget_normalized_rate_covers_hermite_expectation() {
    let profile = widening_profile(vec![1.0, 4.0], 1.3, 7, RATE_SAMPLES);
    assert_eq!(profile.dividing_index, PLANES - 1);
    let omega_0 = 8.0_f64.sqrt();
    assert!((profile.reactant_frequency - omega_0).abs() <= 1.0e-6 * omega_0);

    let omega = profile.reactant_frequency;
    let gaussian = (TAU * THERMAL).sqrt() / omega;
    let rounding = 1024.0 * f64::EPSILON;
    assert!(
        (profile.reactant_partition - gaussian).abs() <= rounding * gaussian.abs().max(1.0),
        "reactant partition {:.16e} vs closed form {:.16e}",
        profile.reactant_partition,
        gaussian
    );
    assert!(
        (profile.reactant_log_partition - gaussian.ln()).abs()
            <= rounding * gaussian.ln().abs().max(1.0),
        "reactant log partition {:.16e} vs closed form {:.16e}",
        profile.reactant_log_partition,
        gaussian.ln()
    );
    let closed_rate = omega / TAU * (-profile.barrier / THERMAL).exp();
    assert!(
        (profile.normalized_rate - closed_rate).abs() <= rounding * closed_rate.abs().max(1.0),
        "normalized rate {:.16e} vs closed form {:.16e}",
        profile.normalized_rate,
        closed_rate
    );
    assert!(
        (profile.normalized_log_rate - closed_rate.ln()).abs()
            <= rounding * closed_rate.ln().abs().max(1.0),
        "normalized log rate {:.16e} vs closed form {:.16e}",
        profile.normalized_log_rate,
        closed_rate.ln()
    );

    let expected_barrier = widening_hermite_barrier(PLANES);
    let expected_rate = omega_0 / TAU * (-expected_barrier / THERMAL).exp();
    assert_coverage(
        "normalized rate",
        profile.normalized_rate,
        profile.normalized_rate_error,
        expected_rate,
    );
}

#[test]
fn error_budget_refinement_covers_hermite_expectation() {
    let coarse = widening_profile(vec![1.0, 1.0], 1.5, 91, REFINE_COARSE_SAMPLES);
    let fine = widening_profile(vec![1.0, 1.0], 1.5, 92, REFINE_FINE_SAMPLES);
    let (coarse_barrier, coarse_se) = end_point(&coarse);
    let (fine_barrier, fine_se) = end_point(&fine);
    let expected = widening_hermite_barrier(PLANES);
    assert_coverage("coarse barrier", coarse_barrier, coarse_se, expected);
    assert_coverage("fine barrier", fine_barrier, fine_se, expected);
}

#[test]
fn arc_lengths_match_the_mass_weighted_chord() {
    let thermal = 0.03_f64;
    let planes = 9usize;
    let slope = 0.3_f64;
    let path = tilted_path(-1.0, 0.0, planes, slope);
    let spacing = 1.0 / ((planes - 1) as f64);
    let third_derivative_bound = 4.0 * 3.0 * 2.0 * 1.0 + 3.0 * 2.0 * 0.5 * 0.5 * slope * slope;
    let trapezoid_error_bound = spacing * spacing * 1.0 * third_derivative_bound / 12.0;
    let config = PafiConfig::default()
        .with_planes(planes)
        .with_burn_in_steps(32)
        .with_sample_steps(32)
        .with_time_step(0.01)
        .with_zero_temperature_tol(trapezoid_error_bound)
        .with_stats(StatsConfig::default().with_min_batches(2))
        .with_seed(seed(99));

    let make_engine = |masses: Vec<f64>, seed_base: u64| {
        let seed_value = seed(seed_base);
        let mut engine = NativeEngine::new(
            WideningDoubleWell,
            NativeConfig::default()
                .with_masses(Array1::from_vec(masses))
                .with_seed(seed_value)
                .with_thermostat(
                    ThermostatConfig::default()
                        .with_gamma(1.3)
                        .with_temperature(thermal)
                        .with_boltzmann(1.0),
                ),
        )
        .expect("native engine");
        engine
            .shoot_velocities(seed_value)
            .expect("initial velocities");
        engine
    };

    let mut unit_mass = make_engine(vec![1.0, 1.0], 99);
    let mut skew_mass = make_engine(vec![1.0, 4.0], 100);
    let p1 = pafi_profile(&mut unit_mass, path.view(), &config).expect("profile unit");
    let p2 = pafi_profile(&mut skew_mass, path.view(), &config).expect("profile skew");
    let s1 = *p1.arc_lengths.last().expect("last arc");
    let s2 = *p2.arc_lengths.last().expect("last arc");
    let expected_s1 = (1.0 + slope * slope).sqrt();
    let expected_s2 = (1.0 + 4.0 * slope * slope).sqrt();
    let tolerance = 1024.0 * f64::EPSILON * expected_s2;
    assert!(
        (s1 - expected_s1).abs() <= tolerance,
        "unit mass arc length mismatch: s1={s1:.16}, expected={expected_s1:.16}"
    );
    assert!(
        (s2 - expected_s2).abs() <= tolerance,
        "skew mass arc length mismatch: s2={s2:.16}, expected={expected_s2:.16}"
    );
    assert!(
        (s1 - s2).abs() > 1.0e-3,
        "mass weighted arc lengths should differ: s1={s1}, s2={s2}"
    );
}

#[test]
fn statistical_family_size_matches_call_sites() {
    let native = include_str!("pafi_native.rs");
    let rgpot = include_str!("pafi_engine_trait_rgpot.rs");
    let coverage = concat!("assert_", "coverage(");
    let below = concat!("assert_", "below(");
    let count = |source: &str| -> usize {
        source
            .lines()
            .filter(|line| !line.trim_start().starts_with("//") && !line.contains("fn "))
            .map(|line| line.matches(coverage).count() + line.matches(below).count())
            .sum()
    };
    assert_eq!(
        count(native) + count(rgpot),
        FAMILY_SIZE,
        "FAMILY_SIZE must equal the number of statistical assertion call sites"
    );
}
