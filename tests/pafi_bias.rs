//! PAFI bias checks against Vineyard and exact quadrature.
//!
//! Every potential is evaluated through the `rgpot` callback ABI and driven by
//! the native BAOAB engine, so the PAFI code under test runs unchanged on a
//! real engine. Reference rates come from the same potential:
//!
//! - exact quadrature of the transition-state-theory flux through the plane
//!   `x = 0`, `k = sqrt(kT / 2 pi m) exp(-F(0) / kT) / \int_{-\infty}^0 exp(-F(x) / kT) dx`,
//!   where `F(x)` integrates out the spectator coordinates numerically;
//! - Vineyard harmonic TST, `k = (prod omega_R / prod omega_S) exp(-dE / kT) / 2 pi`.
//!
//! Beta cancels in a relative rate error, so a miss that stays put when the
//! temperature changes is a fixed energy or normalization error. Equivalence
//! of `ln(k_PAFI / k_ref)` is a two one-sided test (TOST) against the
//! trapezoid bias this potential would leave after one uniform refinement.

#![cfg(feature = "rgpot")]

use std::f64::consts::PI;
use std::os::raw::c_void;

use ndarray::{Array1, Array2};
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::engine::rgpot::{RgpotConfig, RgpotSurface};
use rgflux::engine::{DynamicsEngine, ThermostatConfig};
use rgflux::pafi::{FreeEnergyProfile, PafiConfig, pafi_profile};
use rgflux::stats::StatsConfig;
use rgpot_core::potential::PotentialImpl;
use rgpot_core::status::rgpot_status_t;
use rgpot_core::tensor::rgpot_tensor_cpu_f64_2d;
use rgpot_core::types::{rgpot_force_input_t, rgpot_force_out_t};

const MASSES: [f64; 3] = [2.0, 4.0, 3.0];
const SPECTATOR_Z_STIFFNESS: f64 = 2.0;
const REPLICATES: usize = 8;
/// Nominal level of each one-sided test in the equivalence procedure.
const EQUIVALENCE_ALPHA: f64 = 0.05;
const TIME_STEP: f64 = 0.02;
const FRICTION: f64 = 2.0;

/// `V = (1 - x^2)^2 + (wy + widen x) y^2 / 2 + g y^4 / 4 + kz z^2 / 2`.
#[derive(Clone, Copy, Debug)]
struct Well {
    wy: f64,
    widen: f64,
    quartic: f64,
}

/// `V(x) = (1 - x^2)^2` with decoupled harmonic spectators: the free-energy
/// gradient is `V'(x)` exactly, so PAFI has no sampling noise.
const QUARTIC_1D: Well = Well {
    wy: 1.5,
    widen: 0.0,
    quartic: 0.0,
};

/// Reaction coordinate coupled to a spectator whose stiffness changes along
/// the path and which carries a quartic term: anharmonic in both modes.
const ANHARMONIC_2D: Well = Well {
    wy: 1.5,
    widen: 0.5,
    quartic: 1.0,
};

struct CallbackState {
    well: Well,
    forces: Vec<f64>,
}

unsafe extern "C" fn drop_state(ptr: *mut c_void) {
    if !ptr.is_null() {
        let _ = unsafe { Box::from_raw(ptr as *mut CallbackState) };
    }
}

unsafe extern "C" fn well_callback(
    user_data: *mut c_void,
    input: *const rgpot_force_input_t,
    output: *mut rgpot_force_out_t,
) -> rgpot_status_t {
    if user_data.is_null() || input.is_null() || output.is_null() {
        return rgpot_status_t::RGPOT_INVALID_PARAMETER;
    }
    let state = unsafe { &mut *(user_data as *mut CallbackState) };
    let inp = unsafe { &*input };
    let out = unsafe { &mut *output };
    if unsafe { inp.n_atoms() } != Some(1) {
        return rgpot_status_t::RGPOT_INVALID_PARAMETER;
    }
    let dl = unsafe { &(*inp.positions).dl_tensor };
    let pos = unsafe { std::slice::from_raw_parts(dl.data as *const f64, 3) };
    let (x, y, z) = (pos[0], pos[1], pos[2]);
    let well = state.well;
    let stiffness = well.wy + well.widen * x;
    out.energy = (1.0 - x * x).powi(2)
        + 0.5 * stiffness * y * y
        + 0.25 * well.quartic * y.powi(4)
        + 0.5 * SPECTATOR_Z_STIFFNESS * z * z;
    out.variance = 0.0;
    state.forces[0] = -(-4.0 * x * (1.0 - x * x) + 0.5 * well.widen * y * y);
    state.forces[1] = -(stiffness * y + well.quartic * y.powi(3));
    state.forces[2] = -(SPECTATOR_Z_STIFFNESS * z);
    out.forces = unsafe { rgpot_tensor_cpu_f64_2d(state.forces.as_mut_ptr(), 1, 3) };
    rgpot_status_t::RGPOT_SUCCESS
}

fn rgpot_engine(well: Well, kt: f64, seed: u64) -> NativeEngine<RgpotSurface> {
    let state = Box::new(CallbackState {
        well,
        forces: vec![0.0; 3],
    });
    let potential = PotentialImpl::new(
        well_callback,
        Box::into_raw(state).cast::<c_void>(),
        Some(drop_state),
    );
    let surface = RgpotSurface::new(
        potential,
        RgpotConfig::default().with_atomic_numbers(vec![29]),
    )
    .expect("rgpot surface");
    let mut engine = NativeEngine::new(
        surface,
        NativeConfig::default()
            .with_masses(Array1::from_vec(MASSES.to_vec()))
            .with_seed(seed)
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(FRICTION)
                    .with_temperature(kt)
                    .with_boltzmann(1.0),
            ),
    )
    .expect("native engine");
    engine.shoot_velocities(seed).expect("initial velocities");
    engine
}

fn simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, intervals: usize) -> f64 {
    let h = (b - a) / intervals as f64;
    let mut sum = f(a) + f(b);
    for i in 1..intervals {
        sum += f(a + i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    sum * h / 3.0
}

/// `F(x) - V_x(x)`: free energy of the `y` spectator (the `z` mode is
/// decoupled and identical on every plane).
fn spectator_free_energy(well: Well, x: f64, kt: f64) -> f64 {
    let stiffness = well.wy + well.widen * x;
    let half_width = 14.0 * (kt / stiffness).sqrt();
    let partition = simpson(
        |y| (-(0.5 * stiffness * y * y + 0.25 * well.quartic * y.powi(4)) / kt).exp(),
        -half_width,
        half_width,
        4_000,
    );
    -kt * partition.ln()
}

/// Exact TST rate and barrier for the dividing plane `x = 0`.
fn exact_quadrature(well: Well, kt: f64) -> (f64, f64) {
    let free_energy = |x: f64| (1.0 - x * x).powi(2) + spectator_free_energy(well, x, kt);
    let reference = free_energy(-1.0);
    let reactant = simpson(
        |x| (-(free_energy(x) - reference) / kt).exp(),
        -2.0,
        0.0,
        6_000,
    );
    let barrier = free_energy(0.0) - reference;
    let rate = (kt / (2.0 * PI * MASSES[0])).sqrt() * (-barrier / kt).exp() / reactant;
    (rate, barrier)
}

/// Vineyard harmonic TST: `V''` at the minimum `x = -1` is 8 and at the saddle
/// `x = 0` is -4; the `z` mode is the same in both states.
fn vineyard_rate(well: Well, kt: f64) -> f64 {
    let reactant_x = (8.0 / MASSES[0]).sqrt();
    let reactant_y = ((well.wy - well.widen) / MASSES[1]).sqrt();
    let saddle_y = (well.wy / MASSES[1]).sqrt();
    reactant_x * reactant_y / (2.0 * PI * saddle_y) * (-1.0 / kt).exp()
}

fn straight_path(start: f64, end: f64, planes: usize) -> Array2<f64> {
    let mut path = Array2::<f64>::zeros((planes, 3));
    for i in 0..planes {
        path[[i, 0]] = start + (end - start) * i as f64 / (planes - 1) as f64;
    }
    path
}

#[derive(Clone, Copy, Debug)]
struct Sampling {
    planes: usize,
    burn_in: usize,
    steps: usize,
    path_end: f64,
}

fn pafi_once(well: Well, kt: f64, sampling: Sampling, seed: u64) -> FreeEnergyProfile {
    let path = straight_path(-1.0, sampling.path_end, sampling.planes);
    let config = PafiConfig::default()
        .with_planes(sampling.planes)
        .with_burn_in_steps(sampling.burn_in)
        .with_sample_steps(sampling.steps)
        .with_time_step(TIME_STEP)
        .with_zero_temperature_tol(f64::MAX)
        .with_stats(StatsConfig::default().with_min_batches(8))
        .with_seed(seed);
    let mut engine = rgpot_engine(well, kt, seed);
    pafi_profile(&mut engine, path.view(), &config).expect("pafi profile")
}

fn replicates(well: Well, kt: f64, sampling: Sampling, first_seed: u64) -> Vec<FreeEnergyProfile> {
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..REPLICATES as u64)
            .map(|r| scope.spawn(move || pafi_once(well, kt, sampling, first_seed + 7_919 * r)))
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("replicate"))
            .collect()
    })
}

#[derive(Clone, Copy, Debug)]
struct Equivalence {
    /// Mean of `ln(k_PAFI / k_ref)` over the replicates.
    mean_log_ratio: f64,
    standard_error: f64,
    /// Half-width of the equivalence interval, in the same units.
    margin: f64,
    /// One-sided statistics `(mean + margin) / se` and `(margin - mean) / se`.
    lower: f64,
    upper: f64,
    critical: f64,
}

impl Equivalence {
    /// Schuirmann's procedure: both one-sided tests reject at `EQUIVALENCE_ALPHA`.
    fn equivalent(&self) -> bool {
        self.lower.is_finite()
            && self.upper.is_finite()
            && self.lower > self.critical
            && self.upper > self.critical
    }
}

/// Upper `alpha` quantile of the Student t with `df` degrees of freedom.
///
/// The density is integrated by Simpson's rule and the quantile is bisected,
/// so the threshold is not a transcribed table entry.
fn student_t_upper_quantile(df: usize, alpha: f64) -> f64 {
    let constant = t_density_constant(df);
    let density = |t: f64| {
        let df = df as f64;
        constant * (1.0 + t * t / df).powf(-0.5 * (df + 1.0))
    };
    let target = 0.5 - alpha;
    let panels = 2_048usize;
    let mut hi = 1.0_f64;
    while simpson(density, 0.0, hi, panels) < target {
        hi *= 2.0;
    }
    let mut lo = 0.0_f64;
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if simpson(density, 0.0, mid, panels) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

fn t_density_constant(df: usize) -> f64 {
    let half = ln_gamma_half_integer(df);
    let other = ln_gamma_half_integer(df + 1);
    // c = Γ((df+1)/2) / (sqrt(df π) Γ(df/2)).
    (other - half - 0.5 * ((df as f64) * PI).ln()).exp()
}

/// `ln Γ(n/2)` for a positive integer `n`.
fn ln_gamma_half_integer(n: usize) -> f64 {
    if n.is_multiple_of(2) {
        let factorial_of = n / 2 - 1;
        (1..=factorial_of).map(|k| (k as f64).ln()).sum()
    } else {
        let steps = n / 2;
        let mut value = 0.5 * PI.ln();
        for k in 1..=steps {
            value += ((2 * k - 1) as f64).ln() - (2.0_f64).ln();
        }
        value
    }
}

/// Leading trapezoid error of `∫ V'(x) dx` from the minimum `x = -1` to the
/// saddle `x = 0` of `V(x) = (1 - x^2)^2`, at the plane spacing of `sampling`.
///
/// The Euler-Maclaurin endpoint term is `h_x^2 / 12 * (V''(0) - V''(-1))`.
/// `V'' = -4 + 12 x^2`, so the gap is `-12` and the bias is exactly `-h_x^2`.
/// Mass cancels because both the free energy and the arc length carry it.
fn trapezoid_barrier_bias(sampling: Sampling) -> f64 {
    let path_start = -1.0;
    let spacing = (sampling.path_end - path_start) / (sampling.planes - 1) as f64;
    let second = |x: f64| -4.0 + 12.0 * x * x;
    let endpoint_gap = second(0.0) - second(path_start);
    spacing * spacing / 12.0 * endpoint_gap
}

/// Equivalence margin on `ln(k / k_ref)`.
///
/// One uniform refinement halves `h_x` and divides an `h_x^2` bias by four.
/// Dividing the leading bias by `kT` converts that energy error into a log
/// rate error. A corrected profile has to land inside the bias a once-refined
/// trapezoid would still have.
fn log_rate_margin(sampling: Sampling, kt: f64) -> f64 {
    let refined_spacing_factor = 2.0;
    let refined_bias =
        trapezoid_barrier_bias(sampling).abs() / (refined_spacing_factor * refined_spacing_factor);
    refined_bias / kt
}

fn log_rate_equivalence(
    profiles: &[FreeEnergyProfile],
    reference_rate: f64,
    margin: f64,
) -> Equivalence {
    assert_eq!(profiles.len(), REPLICATES);
    let logs: Vec<f64> = profiles
        .iter()
        .map(|profile| profile.normalized_log_rate - reference_rate.ln())
        .collect();
    let n = logs.len() as f64;
    let mean = logs.iter().sum::<f64>() / n;
    let variance = logs.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let standard_error = (variance / n).sqrt();
    let critical = student_t_upper_quantile(REPLICATES - 1, EQUIVALENCE_ALPHA);
    let (lower, upper) = if standard_error > 0.0 {
        (
            (mean + margin) / standard_error,
            (margin - mean) / standard_error,
        )
    } else {
        (f64::INFINITY, f64::INFINITY)
    };
    Equivalence {
        mean_log_ratio: mean,
        standard_error,
        margin,
        lower,
        upper,
        critical,
    }
}

const TEMPERATURES: [f64; 2] = [0.01, 0.005];

const PRODUCTION: Sampling = Sampling {
    planes: 17,
    burn_in: 1_000,
    steps: 8_000,
    path_end: 0.0,
};

fn report(label: &str, kt: f64, test: &Equivalence) {
    eprintln!(
        "{label} kT={kt}: ln(k/k_ref)={:+.5} se={:.5} margin={:.5} lower={:+.2} upper={:+.2} tcrit={:.4} equivalent={}",
        test.mean_log_ratio,
        test.standard_error,
        test.margin,
        test.lower,
        test.upper,
        test.critical,
        test.equivalent()
    );
}

fn assert_matches_references(well: Well, sampling: Sampling) {
    let mut failures = Vec::new();
    for (index, kt) in TEMPERATURES.into_iter().enumerate() {
        let (exact, _) = exact_quadrature(well, kt);
        let vineyard = vineyard_rate(well, kt);
        let profiles = replicates(well, kt, sampling, 100 + 1_000 * index as u64);
        let margin = log_rate_margin(sampling, kt);
        let against_exact = log_rate_equivalence(&profiles, exact, margin);
        let against_vineyard = log_rate_equivalence(&profiles, vineyard, margin);
        report("exact quadrature", kt, &against_exact);
        report("vineyard", kt, &against_vineyard);
        if !against_exact.equivalent() {
            failures.push(format!(
                "kT={kt} exact mean={:+.4} margin={:.4}",
                against_exact.mean_log_ratio, against_exact.margin
            ));
        }
        if !against_vineyard.equivalent() {
            failures.push(format!(
                "kT={kt} vineyard mean={:+.4} margin={:.4}",
                against_vineyard.mean_log_ratio, against_vineyard.margin
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "TOST does not find the PAFI rate equivalent to the reference: {}",
        failures.join("; ")
    );
}

#[test]
fn pafi_matches_vineyard_and_exact_quadrature_on_the_anharmonic_well() {
    assert_matches_references(ANHARMONIC_2D, PRODUCTION);
}

#[test]
fn an_even_plane_count_across_the_saddle_still_finds_the_barrier() {
    let across = Sampling {
        planes: 18,
        path_end: 1.0,
        ..PRODUCTION
    };
    assert_matches_references(ANHARMONIC_2D, across);
}

#[test]
fn a_deliberately_coarse_integral_is_outside_the_equivalence_margin() {
    let coarse = Sampling {
        planes: 2,
        ..PRODUCTION
    };
    let kt = TEMPERATURES[0];
    let (exact, _) = exact_quadrature(ANHARMONIC_2D, kt);
    let profiles = replicates(ANHARMONIC_2D, kt, coarse, 5_000);
    let test = log_rate_equivalence(&profiles, exact, log_rate_margin(coarse, kt));
    report("coarse exact quadrature", kt, &test);
    assert!(
        !test.equivalent(),
        "a two-plane integral should miss the refined-trapezoid margin {:.4}, mean {:+.4}",
        test.margin,
        test.mean_log_ratio
    );
}

#[test]
fn zero_noise_quartic_reproduces_the_barrier_energy() {
    for kt in TEMPERATURES {
        let sampling = Sampling {
            planes: 17,
            burn_in: 10,
            steps: 64,
            path_end: 0.0,
        };
        let profile = pafi_once(QUARTIC_1D, kt, sampling, 1);
        let residual = (profile.barrier - 1.0).abs();
        eprintln!(
            "quartic kT={kt}: barrier-1={:+.3e} se={:.1e}",
            profile.barrier - 1.0,
            profile.barrier_error
        );
        assert!(
            residual <= 1.0e-9,
            "path-force integral {:.12} misses dE = 1 by {residual:.3e} at kT={kt}",
            profile.barrier
        );
    }
}

#[test]
fn zero_noise_quartic_rate_matches_the_doubled_line_integral() {
    for kt in TEMPERATURES {
        let sampling = Sampling {
            planes: 257,
            burn_in: 10,
            steps: 64,
            path_end: 0.0,
        };
        let profile = pafi_once(QUARTIC_1D, kt, sampling, 1);
        assert!(
            (profile.barrier - 1.0).abs() <= 2.0e-5,
            "barrier {:.8} is not converged at 257 planes",
            profile.barrier
        );
        // Mass-weighted arc length is `sqrt(m) dx`. The reactant partition is
        // the doubled trapezoid from the minimum to the saddle.
        let mass = MASSES[0];
        let integral = simpson(|x| (-(1.0 - x * x).powi(2) / kt).exp(), -1.0, 0.0, 8_000);
        let partition = 2.0 * mass.sqrt() * integral;
        let expected = (kt / (2.0 * PI)).sqrt() * (-profile.barrier / kt).exp() / partition;
        let ratio = profile.normalized_rate / expected;
        eprintln!("quartic kT={kt}: k/k_line={ratio:.8}");
        assert!(
            (ratio - 1.0).abs() <= 1.0e-6,
            "converged quadrature must match the doubled line integral; ratio {ratio:.8} at kT={kt}"
        );
    }
}
