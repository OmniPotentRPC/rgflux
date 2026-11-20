//! Analytic PAFI checks through the native constrained dynamics engine.
//!
//! Every statistical bound is derived in `tests/pafi_support/mod.rs` from a
//! suite false failure rate of one in a thousand shared by all
//! `FAMILY_SIZE` statistical assertions. Coverage targets the Hermite
//! integral of the analytic mean force. Reproducible with
//! `cargo test --test pafi_native -- --nocapture`; set `PAFI_SEED_OFFSET`
//! to draw an independent suite.

mod pafi_support;

use std::f64::consts::{PI, TAU};

use ndarray::{Array1, Array2, ArrayView1};
use pafi_support::{
    FAMILY_SIZE, Z_CRIT, assert_above, assert_below, assert_coverage, assert_dividing_plane,
    assert_sweep_within_budget, block_stats, curved_barrier_se_cap, end_plane_disagreement_cap,
    offset_sweep, seed, seed_with_offset, widening_barrier_se_cap, widening_rate_se_cap,
    widening_se_ratio_cap,
};
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::engine::{DynamicsEngine, ThermostatConfig};
use rgflux::pafi::{
    FreeEnergyProfile, HermiteIntegral, MeanForceIntegral, PafiConfig, pafi_profile,
};
use rgflux::stats::StatsConfig;

const THERMAL: f64 = 0.2;
const PLANES: usize = 21;
const TIME_STEP: f64 = 0.05;
const BURN_IN: usize = 200;

/// Offsets of `PAFI_SEED_OFFSET` in the curved standard-error and end-plane
/// sweeps. A curved profile costs about a second, so these stay small.
const CURVED_SE_SWEEP_OFFSETS: u64 = 64;

/// Offsets in the widening-well sweeps, whose profiles are far cheaper.
const WIDENING_SWEEP_OFFSETS: u64 = 128;

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

/// Exact mean force `dF/dx` of the widening double well on the plane `x`.
fn widening_mean_force(x: f64) -> f64 {
    -4.0 * x * (1.0 - x * x) + 0.25 * THERMAL / (1.5 + 0.5 * x)
}

/// Continuum barrier `F(0) - F(-1)` of the widening double well.
fn widening_analytic_barrier() -> f64 {
    1.0 + 0.5 * THERMAL * 1.5_f64.ln()
}

/// Hermite integral of [`widening_mean_force`] from `x = -1` to `x = 0`.
fn widening_expected_barrier(planes: usize) -> f64 {
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
    widening_profile_seeded(masses, gamma, seed(seed_base), samples)
}

fn widening_profile_seeded(
    masses: Vec<f64>,
    gamma: f64,
    seed_value: u64,
    samples: usize,
) -> FreeEnergyProfile {
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

/// Last-plane free energy and its reported standard error.
fn end_point(profile: &FreeEnergyProfile) -> (f64, f64) {
    let last = profile.free_energy.len() - 1;
    let gap = (profile.barrier - profile.free_energy[last]).abs();
    assert!(
        gap <= profile.standard_error[last],
        "endpoint and barrier differ by {gap:.8e}, above the reported standard error {:.8e}",
        profile.standard_error[last]
    );
    (profile.free_energy[last], profile.standard_error[last])
}

const ISO_SAMPLES: usize = 10_000;
const ISO_MASS_Y: f64 = 1.0;
const ISO_GAMMA: f64 = 1.5;
const ANISO_SAMPLES: usize = 20_000;
const ANISO_MASS_Y: f64 = 4.0;
const ANISO_GAMMA: f64 = 1.5;

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
    let profile = widening_profile(vec![1.0, 1.0], 1.5, ISO_SEED, ISO_SAMPLES);
    let (barrier, se) = end_point(&profile);
    let expected = widening_expected_barrier(PLANES);
    assert_below(
        "isotropic barrier SE",
        se,
        widening_barrier_se_cap(ISO_SAMPLES, ISO_MASS_Y, ISO_GAMMA),
    );
    assert_coverage("isotropic barrier", barrier, se, expected);
}

#[test]
fn pafi_widening_barrier_mass_anisotropic_matches_analytic_within_reported_se() {
    let profile = widening_profile(vec![1.0, 4.0], 1.5, ANISO_SEED, ANISO_SAMPLES);
    let (barrier, se) = end_point(&profile);
    let expected = widening_expected_barrier(PLANES);
    assert_below(
        "anisotropic barrier SE",
        se,
        widening_barrier_se_cap(ANISO_SAMPLES, ANISO_MASS_Y, ANISO_GAMMA),
    );
    assert_coverage("anisotropic barrier", barrier, se, expected);
}

const ISO_SEED: u64 = 11;
const ANISO_SEED: u64 = 21;

/// Reported barrier standard error of a widening profile at its last plane.
fn widening_barrier_se(profile: &FreeEnergyProfile) -> f64 {
    profile.standard_error[PLANES - 1]
}

#[test]
#[ignore = "opt-in sweep of PAFI_SEED_OFFSET for the widening barrier standard-error caps"]
fn widening_barrier_se_cap_offset_sweep() {
    let iso_cap = widening_barrier_se_cap(ISO_SAMPLES, ISO_MASS_Y, ISO_GAMMA);
    let aniso_cap = widening_barrier_se_cap(ANISO_SAMPLES, ANISO_MASS_Y, ANISO_GAMMA);
    let report = offset_sweep(WIDENING_SWEEP_OFFSETS, 2, |offset| {
        let iso = widening_profile_seeded(
            vec![1.0, ISO_MASS_Y],
            ISO_GAMMA,
            seed_with_offset(ISO_SEED, offset),
            ISO_SAMPLES,
        );
        let aniso = widening_profile_seeded(
            vec![1.0, ANISO_MASS_Y],
            ANISO_GAMMA,
            seed_with_offset(ANISO_SEED, offset),
            ANISO_SAMPLES,
        );
        vec![
            (widening_barrier_se(&iso), iso_cap),
            (widening_barrier_se(&aniso), aniso_cap),
        ]
    });
    assert_sweep_within_budget(
        "widening barrier SE",
        &["isotropic", "anisotropic"],
        &report,
    );
}

const CURVED_RADIUS: f64 = 1.2;
const CURVED_FORCE_CONSTANT_X: f64 = 3.0;
const CURVED_FORCE_CONSTANT_Y: f64 = 1.0;
const CURVED_PLANES: usize = 9;
const CURVED_SEED: u64 = 1_234;
const CURVED_BURN_IN: usize = 3_000;
const CURVED_SAMPLES: usize = 54_000;
// A block of `CURVED_SAMPLES / 40` steps has to be long compared with the
// autocorrelation time. At `0.01` a block spans `13.5` time units.
const CURVED_TIME_STEP: f64 = 0.01;

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

/// Exact mean force `-dF/dtheta` on the plane at angle `theta`.
fn curved_mean_force(theta: f64) -> f64 {
    let (sin, cos) = theta.sin_cos();
    THERMAL * (CURVED_FORCE_CONSTANT_X - CURVED_FORCE_CONSTANT_Y) * sin * cos
        / (CURVED_FORCE_CONSTANT_X * cos * cos + CURVED_FORCE_CONSTANT_Y * sin * sin)
}

/// Hermite integral of [`curved_mean_force`] over the quarter circle.
fn curved_expected_barrier() -> f64 {
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

/// Offsets the last increment away from `longer` by one disagreement cap.
///
/// Twice the paired discrepancy is a few times `1e-4`. One increment's
/// sampling standard deviation is about `1e-3`, so that doubled discrepancy
/// is not a tail event of the maximum. An added bias of the cap itself is.
fn offset_end_plane_by_its_cap(
    profile: &FreeEnergyProfile,
    longer: &FreeEnergyProfile,
) -> FreeEnergyProfile {
    let mut regressed = profile.clone();
    let last = CURVED_PLANES - 1;
    let increment = profile.free_energy[last] - profile.free_energy[last - 1];
    let reference = longer.free_energy[last] - longer.free_energy[last - 1];
    let direction = if increment >= reference { 1.0 } else { -1.0 };
    regressed.free_energy[last] += direction * end_plane_disagreement_cap();
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
    let cap = curved_barrier_se_cap();
    assert_dividing_plane(
        "curved dividing plane",
        profile.dividing_index,
        CURVED_PLANES - 1,
    );
    // Without the Jacobian term the canonical mean barrier is zero. The
    // coverage window around `expected`, using the standard error cap as an
    // upper bound, excludes that limit.
    assert!(
        expected > Z_CRIT * cap,
        "expected barrier {expected:.8} is not separated from the zero-Jacobian limit"
    );
    assert_below("curved barrier SE", se, cap);
    assert_coverage("curved barrier", barrier, se, expected);

    // Natural spline endpoints zero the normal curvature `dn/ds` on the end
    // plane. Extending the path makes that image interior and changes the
    // spline curvature everywhere. [`end_plane_disagreement_cap`] is the
    // union bound on the eight Hermite increment differences.
    let worst = end_plane_disagreement(&profile, &longer);
    assert_below(
        "end plane disagreement",
        worst,
        end_plane_disagreement_cap(),
    );
}

#[test]
fn end_plane_cap_rejects_an_added_cap_of_bias() {
    let seed_value = seed(CURVED_SEED);
    let (profile, longer) = std::thread::scope(|scope| {
        let base = scope.spawn(|| curved_profile(0, seed_value));
        let longer = scope.spawn(|| curved_profile(1, seed_value));
        (
            base.join().expect("curved profile thread"),
            longer.join().expect("longer curved profile thread"),
        )
    });
    let regressed = offset_end_plane_by_its_cap(&profile, &longer);
    let regressed_worst = end_plane_disagreement(&regressed, &longer);
    assert_above(
        "regressed end plane disagreement",
        regressed_worst,
        end_plane_disagreement_cap(),
    );
}

#[test]
#[ignore = "opt-in sweep of PAFI_SEED_OFFSET for the curved standard-error cap"]
fn curved_se_cap_offset_sweep() {
    let cap = curved_barrier_se_cap();
    let report = offset_sweep(CURVED_SE_SWEEP_OFFSETS, 1, |offset| {
        let profile = curved_profile(0, seed_with_offset(CURVED_SEED, offset));
        vec![(profile.standard_error[CURVED_PLANES - 1], cap)]
    });
    assert_sweep_within_budget("curved SE cap", &["barrier SE"], &report);
}

#[test]
#[ignore = "opt-in sweep of PAFI_SEED_OFFSET for the end-plane disagreement cap"]
fn end_plane_disagreement_cap_offset_sweep() {
    let cap = end_plane_disagreement_cap();
    let report = offset_sweep(CURVED_SE_SWEEP_OFFSETS, 1, |offset| {
        let seed_value = seed_with_offset(CURVED_SEED, offset);
        let profile = curved_profile(0, seed_value);
        let longer = curved_profile(1, seed_value);
        vec![(end_plane_disagreement(&profile, &longer), cap)]
    });
    assert_sweep_within_budget("end plane", &["disagreement"], &report);
}

const RATE_SAMPLES: usize = 20_000;
const RATE_MASS_Y: f64 = 4.0;
const RATE_GAMMA: f64 = 1.3;

#[test]
fn normalized_pafi_rate_matches_analytic_profile() {
    let profile = widening_profile(vec![1.0, 4.0], 1.3, RATE_SEED, RATE_SAMPLES);
    let rate = profile.normalized_rate.expect("representable rate");
    let rate_error = profile
        .normalized_rate_error
        .expect("representable rate error");
    assert_dividing_plane("rate dividing plane", profile.dividing_index, PLANES - 1);

    // Reactant Hessian at x = -1: V_xx = 8 along the unit normal, mass 1.
    // The reactant well is that Gaussian, so the partition and the rate are
    // the closed form of the barrier the profile reports.
    // The reactant curvature is a numerical Hessian, so the frequency matches
    // the analytic oscillator to the tolerance used for that projection. The
    // partition and the rate are then the closed form of the frequency and
    // the barrier the profile actually reports.
    let omega_0 = 8.0_f64.sqrt();
    assert!((profile.reactant_frequency - omega_0).abs() <= 1.0e-6 * omega_0);
    let gaussian = (TAU * THERMAL).sqrt() / profile.reactant_frequency;
    assert!(
        (profile.reactant_partition.expect("representable partition") - gaussian).abs()
            <= 1.0e-6 * gaussian
    );
    assert!((profile.reactant_log_partition - gaussian.ln()).abs() <= 1.0e-6 * gaussian.ln().abs());
    let closed_rate = profile.reactant_frequency / TAU * (-profile.barrier / THERMAL).exp();
    assert!((rate - closed_rate).abs() <= 1.0e-12 * closed_rate);
    assert!(
        (profile.normalized_log_rate - closed_rate.ln()).abs()
            <= 1.0e-9 * closed_rate.ln().abs().max(1.0)
    );

    let expected_barrier = widening_expected_barrier(PLANES);
    let expected_rate = omega_0 / TAU * (-expected_barrier / THERMAL).exp();
    assert_below(
        "rate SE",
        rate_error,
        widening_rate_se_cap(RATE_SAMPLES, RATE_MASS_Y, RATE_GAMMA),
    );
    assert_coverage("normalized rate", rate, rate_error, expected_rate);
}

const RATE_SEED: u64 = 7;

#[test]
#[ignore = "opt-in sweep of PAFI_SEED_OFFSET for the widening rate standard-error cap"]
fn widening_rate_se_cap_offset_sweep() {
    let cap = widening_rate_se_cap(RATE_SAMPLES, RATE_MASS_Y, RATE_GAMMA);
    let report = offset_sweep(WIDENING_SWEEP_OFFSETS, 1, |offset| {
        let profile = widening_profile_seeded(
            vec![1.0, RATE_MASS_Y],
            RATE_GAMMA,
            seed_with_offset(RATE_SEED, offset),
            RATE_SAMPLES,
        );
        vec![(
            profile
                .normalized_rate_error
                .expect("representable rate error"),
            cap,
        )]
    });
    assert_sweep_within_budget("widening rate SE", &["rate SE"], &report);
}

const REFINE_COARSE_SAMPLES: usize = 5_000;
const REFINE_FINE_SAMPLES: usize = 20_000;
const REFINE_MASS_Y: f64 = 1.0;
const REFINE_GAMMA: f64 = 1.5;
const REFINE_COARSE_SEED: u64 = 91;
const REFINE_FINE_SEED: u64 = 92;

#[test]
fn sampling_refinement_restores_analytic_coverage() {
    let coarse = widening_profile(
        vec![1.0, 1.0],
        1.5,
        REFINE_COARSE_SEED,
        REFINE_COARSE_SAMPLES,
    );
    let fine = widening_profile(vec![1.0, 1.0], 1.5, REFINE_FINE_SEED, REFINE_FINE_SAMPLES);
    let (coarse_barrier, coarse_se) = end_point(&coarse);
    let (fine_barrier, fine_se) = end_point(&fine);
    let expected = widening_expected_barrier(PLANES);
    assert_below(
        "coarse barrier SE",
        coarse_se,
        widening_barrier_se_cap(REFINE_COARSE_SAMPLES, REFINE_MASS_Y, REFINE_GAMMA),
    );
    assert_below(
        "fine barrier SE",
        fine_se,
        widening_barrier_se_cap(REFINE_FINE_SAMPLES, REFINE_MASS_Y, REFINE_GAMMA),
    );
    assert_below(
        "fine over coarse SE",
        fine_se / coarse_se,
        widening_se_ratio_cap(
            REFINE_COARSE_SAMPLES,
            REFINE_FINE_SAMPLES,
            REFINE_MASS_Y,
            REFINE_GAMMA,
        ),
    );
    assert_coverage("coarse barrier", coarse_barrier, coarse_se, expected);
    assert_coverage("fine barrier", fine_barrier, fine_se, expected);
}

#[test]
#[ignore = "opt-in sweep of PAFI_SEED_OFFSET for the refinement standard-error caps"]
fn sampling_refinement_cap_offset_sweep() {
    let coarse_cap = widening_barrier_se_cap(REFINE_COARSE_SAMPLES, REFINE_MASS_Y, REFINE_GAMMA);
    let fine_cap = widening_barrier_se_cap(REFINE_FINE_SAMPLES, REFINE_MASS_Y, REFINE_GAMMA);
    let ratio_cap = widening_se_ratio_cap(
        REFINE_COARSE_SAMPLES,
        REFINE_FINE_SAMPLES,
        REFINE_MASS_Y,
        REFINE_GAMMA,
    );
    let report = offset_sweep(WIDENING_SWEEP_OFFSETS, 3, |offset| {
        let coarse = widening_profile_seeded(
            vec![1.0, REFINE_MASS_Y],
            REFINE_GAMMA,
            seed_with_offset(REFINE_COARSE_SEED, offset),
            REFINE_COARSE_SAMPLES,
        );
        let fine = widening_profile_seeded(
            vec![1.0, REFINE_MASS_Y],
            REFINE_GAMMA,
            seed_with_offset(REFINE_FINE_SEED, offset),
            REFINE_FINE_SAMPLES,
        );
        let coarse_se = widening_barrier_se(&coarse);
        let fine_se = widening_barrier_se(&fine);
        vec![
            (coarse_se, coarse_cap),
            (fine_se, fine_cap),
            (fine_se / coarse_se, ratio_cap),
        ]
    });
    assert_sweep_within_budget(
        "refinement",
        &[
            "coarse barrier SE",
            "fine barrier SE",
            "fine over coarse SE",
        ],
        &report,
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
                    && !line.contains("generic.dividing_index, dynamic.dividing_index")
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
