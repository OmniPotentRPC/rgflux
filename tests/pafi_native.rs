//! Analytic PAFI checks through the native constrained dynamics engine.
//!
//! Every statistical bound is derived in `tests/pafi_support/mod.rs` from a
//! suite false failure rate of one in a thousand shared by all
//! `FAMILY_SIZE` statistical assertions. Reproducible with
//! `cargo test --features rgpot --test pafi_native -- --nocapture`; set
//! `PAFI_SEED_OFFSET` to draw an independent suite.

mod pafi_support;

use std::f64::consts::PI;

use ndarray::{Array1, Array2, ArrayView1};
use pafi_support::{
    FAMILY_SIZE, Z_CRIT, assert_above, assert_below, assert_coverage, assert_dividing_plane,
    block_stats, se_cap, seed, t_cap,
};
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::engine::{DynamicsEngine, ThermostatConfig};
use rgflux::pafi::{FreeEnergyProfile, PafiConfig, pafi_profile};
use rgflux::stats::StatsConfig;

const THERMAL: f64 = 0.2;
const PLANES: usize = 21;
const TIME_STEP: f64 = 0.05;
const BURN_IN: usize = 200;

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

/// Exact mean force `dF/dx` of the widening double well on the plane `x`.
fn widening_mean_force(x: f64) -> f64 {
    -4.0 * x * (1.0 - x * x) + 0.25 * THERMAL / (1.5 + 0.5 * x)
}

/// Continuum barrier `F(0) - F(-1)` of the widening double well.
fn widening_analytic_barrier() -> f64 {
    1.0 + 0.5 * THERMAL * 1.5_f64.ln()
}

/// Expectation of `FreeEnergyProfile::free_energy`: the trapezoid sum of the
/// exact mean force on the same planes, so quadrature error is not a bias.
fn widening_expected_free_energy(planes: usize) -> Vec<f64> {
    let h = 1.0 / ((planes - 1) as f64);
    let mut free_energy = vec![0.0_f64; planes];
    for i in 1..planes {
        let x0 = -1.0 + ((i - 1) as f64) * h;
        let x1 = -1.0 + (i as f64) * h;
        free_energy[i] =
            free_energy[i - 1] + 0.5 * h * (widening_mean_force(x0) + widening_mean_force(x1));
    }
    free_energy
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

/// End point free energy and its SE: the barrier of a profile that rises
/// monotonically to its last plane.
fn end_point(profile: &FreeEnergyProfile) -> (f64, f64) {
    let last = profile.free_energy.len() - 1;
    assert_eq!(profile.barrier, profile.free_energy[profile.dividing_index]);
    assert!(
        profile.free_energy.iter().all(|&f| f <= profile.barrier),
        "barrier must be the profile maximum"
    );
    (profile.free_energy[last], profile.standard_error[last])
}

// Reported barrier SE caps: `se_cap(mean)` with the mean SE measured over
// 4096 seeds of each configuration.
const ISO_SAMPLES: usize = 10_000;
const ISO_SE_CAP: f64 = se_cap(7.66e-4);
const ANISO_SAMPLES: usize = 20_000;
const ANISO_SE_CAP: f64 = se_cap(9.10e-4);

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
fn pafi_widening_barrier_matches_analytic_within_reported_se() {
    let profile = widening_profile(vec![1.0, 1.0], 1.5, 11, ISO_SAMPLES);
    let (barrier, se) = end_point(&profile);
    let expected = widening_expected_free_energy(PLANES)[PLANES - 1];
    assert_below("isotropic barrier SE", se, ISO_SE_CAP);
    assert_coverage("isotropic barrier", barrier, se, expected);
}

#[test]
fn pafi_widening_barrier_mass_anisotropic_matches_analytic_within_reported_se() {
    let profile = widening_profile(vec![1.0, 4.0], 1.5, 21, ANISO_SAMPLES);
    let (barrier, se) = end_point(&profile);
    let expected = widening_expected_free_energy(PLANES)[PLANES - 1];
    assert_below("anisotropic barrier SE", se, ANISO_SE_CAP);
    assert_coverage("anisotropic barrier", barrier, se, expected);
}

const CURVED_RADIUS: f64 = 1.2;
const CURVED_FORCE_CONSTANT_X: f64 = 3.0;
const CURVED_FORCE_CONSTANT_Y: f64 = 1.0;
const CURVED_PLANES: usize = 9;
const CURVED_SEED: u64 = 1_234;
const CURVED_BURN_IN: usize = 3_000;
const CURVED_SAMPLES: usize = 54_000;
// A block of `CURVED_SAMPLES / 40` steps must be long compared with the
// autocorrelation time for the batch means SE to follow the Student `t` pivot
// of `tests/pafi_support/mod.rs`. At the default step `0.002` a block spans
// `2.7` time units and the SE had `cv = 0.117` over 256 seeds; at `0.01` it
// spans `13.5` and the SE had `cv = 0.072` over 4096 seeds.
const CURVED_TIME_STEP: f64 = 0.01;
const CURVED_SE_CAP: f64 = se_cap(3.28e-3);
// Cap on `end_plane_disagreement`, `t_cap(mean, cv)` with the mean (`3.29e-5`)
// and coefficient of variation (`0.169`) measured over 4096 seeds, tightened
// from `2.2966 * mean` (`7.5557e-5`) to the Student `t` bound `1.7645 * mean`
// (`5.8052e-5`) so that doubling the end-plane estimator error exceeds the cap.
const END_PLANE_CAP: f64 = t_cap(3.29e-5, 0.169);

/// Profile along the quarter circle `theta` in `[pi / 2, 0]`, sampled at
/// `CURVED_PLANES + extra_images` equally spaced images and planes. With
/// `extra_images = 1` the arc continues one image spacing past `theta = 0`,
/// so the plane at `theta = 0` is an interior plane of the longer path while
/// every plane before it keeps its position, tangent and per-plane seed.
fn curved_profile(extra_images: usize, seed_value: u64) -> FreeEnergyProfile {
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

/// Exact mean force `-dF/dtheta` on the plane at angle `theta`. In the mass
/// weighted coordinates `(x, 2 y)` the path is a circle, the potential is
/// `(ax u^2 + ay v^2) / 2` and the plane at `theta` is the line through the
/// origin along `(cos theta, sin theta)`, so `F(theta) = (kT / 2) ln(ax cos^2
/// theta + ay sin^2 theta)` up to a constant.
fn curved_mean_force(theta: f64) -> f64 {
    let (sin, cos) = theta.sin_cos();
    THERMAL * (CURVED_FORCE_CONSTANT_X - CURVED_FORCE_CONSTANT_Y) * sin * cos
        / (CURVED_FORCE_CONSTANT_X * cos * cos + CURVED_FORCE_CONSTANT_Y * sin * sin)
}

/// Expectation of the last free energy of `curved_profile`: the trapezoid sum
/// of the exact mean force over the planes. The spline parameter is the chord
/// length, uniform between images, and the images are uniform in `theta`, so
/// the parameter is linear in `theta` at the planes and the trapezoid weights
/// are `h / 2` with `h = (pi / 2) / (CURVED_PLANES - 1)`. The continuum barrier
/// `(kT / 2) ln(ax / ay)` differs from this sum by the trapezoid error.
fn curved_expected_barrier() -> f64 {
    let h = PI * 0.5 / ((CURVED_PLANES - 1) as f64);
    (1..CURVED_PLANES)
        .map(|k| {
            let theta0 = PI * 0.5 - ((k - 1) as f64) * h;
            let theta1 = PI * 0.5 - (k as f64) * h;
            0.5 * h * (curved_mean_force(theta0) + curved_mean_force(theta1))
        })
        .sum()
}

/// Largest change of one free-energy increment between the profile and the
/// profile of the longer path.
fn end_plane_disagreement(profile: &FreeEnergyProfile, longer: &FreeEnergyProfile) -> f64 {
    (1..CURVED_PLANES)
        .map(|k| {
            let increment = profile.free_energy[k] - profile.free_energy[k - 1];
            let reference = longer.free_energy[k] - longer.free_energy[k - 1];
            (increment - reference).abs()
        })
        .fold(0.0_f64, f64::max)
}

/// Regression-injection helper that doubles the end-plane estimator error in
/// `profile` relative to the interior-plane reference `longer`.
fn double_end_plane_estimator_error(
    profile: &FreeEnergyProfile,
    longer: &FreeEnergyProfile,
) -> FreeEnergyProfile {
    let mut regressed = profile.clone();
    let last = CURVED_PLANES - 1;
    let increment = profile.free_energy[last] - profile.free_energy[last - 1];
    let reference = longer.free_energy[last] - longer.free_energy[last - 1];
    regressed.free_energy[last] += increment - reference;
    regressed
}

#[test]
fn curved_mass_weighted_harmonic_profile_requires_jacobian() {
    let seed_value = seed(CURVED_SEED);
    let (profile, longer) = std::thread::scope(|scope| {
        let base = scope.spawn(|| curved_profile(0, seed_value));
        let longer = scope.spawn(|| curved_profile(1, seed_value));
        (
            base.join().expect("curved profile thread"),
            longer.join().expect("longer curved profile thread"),
        )
    });
    let expected = curved_expected_barrier();
    let (barrier, se) = end_point(&profile);

    // The last increment is `2.56e-3` with an ensemble standard deviation of
    // `1.7e-4`, fifteen standard deviations, so the dividing plane is the last
    // one in all 4096 ensemble seeds.
    assert_dividing_plane(
        "curved dividing plane",
        profile.dividing_index,
        CURVED_PLANES - 1,
    );

    // Without the `(r - r_0) . dn/ds` Jacobian term the projected force is
    // linear in the centred plane coordinate and its canonical mean barrier is
    // zero. The coverage and SE assertions below confine the estimate to
    // `expected +- Z_CRIT * CURVED_SE_CAP`, which excludes zero.
    assert!(
        expected > Z_CRIT * CURVED_SE_CAP,
        "expected barrier {expected:.8} is not separated from the zero-Jacobian limit"
    );
    assert_below("curved barrier SE", se, CURVED_SE_CAP);
    assert_coverage("curved barrier", barrier, se, expected);

    // Natural spline endpoints zero the normal curvature `dn/ds` on the end
    // planes and move the last increment far more than the sampling noise left
    // after sharing the seeds. The longer path sees the same plane with the
    // curvature of an interior plane.
    let worst = end_plane_disagreement(&profile, &longer);
    assert_below("end plane disagreement", worst, END_PLANE_CAP);
}

#[test]
fn end_plane_cap_rejects_doubled_estimator_error() {
    let (profile, longer) = std::thread::scope(|scope| {
        let base = scope.spawn(|| curved_profile(0, CURVED_SEED));
        let longer = scope.spawn(|| curved_profile(1, CURVED_SEED));
        (
            base.join().expect("curved profile thread"),
            longer.join().expect("longer curved profile thread"),
        )
    });
    let regressed = double_end_plane_estimator_error(&profile, &longer);
    let regressed_worst = end_plane_disagreement(&regressed, &longer);
    assert_above(
        "regressed end plane disagreement",
        regressed_worst,
        END_PLANE_CAP,
    );
}

const RATE_SAMPLES: usize = 20_000;
const RATE_SE_CAP: f64 = se_cap(7.70e-6);

#[test]
fn normalized_pafi_rate_matches_analytic_profile() {
    let profile = widening_profile(vec![1.0, 4.0], 1.3, 7, RATE_SAMPLES);
    let rate = profile.normalized_rate;
    let rate_error = profile.normalized_rate_error;
    // The last plane increment `h (g_{N-1} + g_N) / 2` is about `6.7e-3` with an
    // ensemble standard deviation of `1.1e-4`, sixty standard deviations, so the
    // dividing plane is the last one in all 4096 ensemble seeds.
    assert_dividing_plane("rate dividing plane", profile.dividing_index, PLANES - 1);

    let reconstructed_zr =
        integrate_reactant_partition(&profile.arc_lengths, &profile.free_energy, THERMAL);
    assert!(
        (profile.reactant_partition - reconstructed_zr).abs()
            <= 1024.0 * f64::EPSILON * reconstructed_zr,
        "reactant partition mismatch: {:.16e} vs {:.16e}",
        profile.reactant_partition,
        reconstructed_zr
    );
    // The relative rounding of `reconstructed_zr` bounds the absolute change of
    // its logarithm, and the log space sum adds rounding of `|ln Z_R|` ulps.
    let log_tolerance = 1024.0 * f64::EPSILON * reconstructed_zr.ln().abs().max(1.0);
    assert!(
        (profile.reactant_log_partition - reconstructed_zr.ln()).abs() <= log_tolerance,
        "reactant log partition mismatch: {:.16e} vs {:.16e}",
        profile.reactant_log_partition,
        reconstructed_zr.ln()
    );

    // Plug in rate at the exact mean force trapezoid profile: the expectation
    // of the estimator up to the second order bias in the exponential.
    let arcs: Vec<f64> = (0..PLANES)
        .map(|i| (i as f64) / ((PLANES - 1) as f64))
        .collect();
    let expected_free_energy = widening_expected_free_energy(PLANES);
    let expected_zr = integrate_reactant_partition(&arcs, &expected_free_energy, THERMAL);
    let expected_rate = (THERMAL / (2.0 * PI)).sqrt()
        * (-expected_free_energy[PLANES - 1] / THERMAL).exp()
        / expected_zr;

    assert_below("rate SE", rate_error, RATE_SE_CAP);
    assert_coverage("normalized rate", rate, rate_error, expected_rate);
}

const REFINE_COARSE_SAMPLES: usize = 5_000;
const REFINE_FINE_SAMPLES: usize = 20_000;
const REFINE_COARSE_SE_CAP: f64 = se_cap(1.053e-3);
const REFINE_FINE_SE_CAP: f64 = se_cap(5.49e-4);
const REFINE_RATIO_CAP: f64 = se_cap(0.523);

#[test]
fn sampling_refinement_restores_analytic_coverage() {
    let coarse = widening_profile(vec![1.0, 1.0], 1.5, 91, REFINE_COARSE_SAMPLES);
    let fine = widening_profile(vec![1.0, 1.0], 1.5, 92, REFINE_FINE_SAMPLES);
    let (coarse_barrier, coarse_se) = end_point(&coarse);
    let (fine_barrier, fine_se) = end_point(&fine);
    let expected = widening_expected_free_energy(PLANES)[PLANES - 1];
    assert_below("coarse barrier SE", coarse_se, REFINE_COARSE_SE_CAP);
    assert_below("fine barrier SE", fine_se, REFINE_FINE_SE_CAP);
    assert_below("fine over coarse SE", fine_se / coarse_se, REFINE_RATIO_CAP);
    assert_coverage("coarse barrier", coarse_barrier, coarse_se, expected);
    assert_coverage("fine barrier", fine_barrier, fine_se, expected);
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
        "unit-mass arc length mismatch: s1={s1:.16}, expected={expected_s1:.16}"
    );
    assert!(
        (s2 - expected_s2).abs() <= tolerance,
        "skew-mass arc length mismatch: s2={s2:.16}, expected={expected_s2:.16}"
    );
    assert!(
        s2 > s1,
        "mass-weighted arc lengths should differ: s1={s1}, s2={s2}"
    );
}

#[test]
fn statistical_family_size_matches_call_sites() {
    let native = include_str!("pafi_native.rs");
    let rgpot = include_str!("pafi_engine_trait_rgpot.rs");
    let coverage = concat!("assert_", "coverage(");
    let below = concat!("assert_", "below(");
    let dividing = concat!("assert_", "dividing_plane(");
    let raw_dividing = concat!(".dividing_", "index");
    let count = |source: &str| -> usize {
        source
            .lines()
            .filter(|line| !line.trim_start().starts_with("//") && !line.contains("fn "))
            .map(|line| {
                line.matches(coverage).count()
                    + line.matches(below).count()
                    + line.matches(dividing).count()
            })
            .sum()
    };
    let uncounted_dividing = |source: &str| -> usize {
        let total_raw = source
            .lines()
            .filter(|line| {
                let trimmed = line.trim_start();
                !trimmed.starts_with("//")
                    && line.contains(raw_dividing)
                    && !line.contains("free_energy[profile.dividing_index]")
                    && !line.contains("lhs.dividing_index, rhs.dividing_index")
            })
            .count();
        let counted_calls = source
            .lines()
            .filter(|line| !line.trim_start().starts_with("//") && !line.contains("fn "))
            .map(|line| line.matches(dividing).count())
            .sum::<usize>();
        total_raw.saturating_sub(counted_calls)
    };
    assert_eq!(
        uncounted_dividing(native) + uncounted_dividing(rgpot),
        0,
        "every stochastic dividing_index assertion must go through assert_dividing_plane"
    );
    assert_eq!(
        count(native) + count(rgpot),
        FAMILY_SIZE,
        "FAMILY_SIZE must equal the number of statistical assertion call sites"
    );
}
