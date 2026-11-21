//! Constrained mean forces and PAFI profiles against the Langevin friction.
//!
//! The native engine has one thermostat family: white-noise Langevin with
//! friction `gamma > 0`, with `gamma == 0` giving NVE dynamics. There is no
//! second thermostat algorithm, so "independent of the thermostat" is tested
//! over what exists:
//!
//! - the friction, over a factor of sixteen (`0.5`, `2`, `8`),
//! - the constraint scheme, RATTLE or SHAKE, which change how the plane is
//!   enforced and not what is sampled,
//! - the temperature, where the reference moves with `k_B T` and the same
//!   estimator has to follow it,
//! - the NVE choice, `gamma == 0`. NVE samples an energy shell started at the
//!   anchor and not the canonical ensemble, so `sample_hyperplane` and
//!   `pafi_profile` must refuse it. Before that check they accepted it and
//!   returned a biased number; every test below asserts the refusal.
//!
//! What the references are:
//!
//! - Harmonic plane: `dF/dc = c / (n^T K^-1 n)`, exact for a quadratic
//!   potential and independent of the masses.
//! - Widening well: the mean force on the plane `x` is
//!   `-4x(1 - x^2) + k_B T / (4 k(x))` with `k(x) = 1.5 + 0.5x`. The barrier
//!   target is the Hermite integral of that force on the sampled planes, the
//!   expectation of the PAFI estimator, so quadrature error does not enter.
//!   The rate target is the estimator's own formula,
//!   `(omega_0 / 2 pi) exp(-Delta F / k_B T)`, with `omega_0^2 = 8` the
//!   curvature of `(1 - x^2)^2` at `x = -1`. It is the harmonic-reactant
//!   rate that PAFI documents. It is not the exact rate of the anharmonic
//!   one-dimensional reduction, because the reactant partition function is
//!   replaced by a Gaussian.
//! - Tilted plane: on `s = 0` the multiplier has mean `k_B T / (4 k(0))`, and
//!   with one holonomic constraint in two coordinates `2 KE / k_B T = 1`.
//!
//! Statistics: seeds are fixed (shift them with `PAFI_SEED_OFFSET`). Each
//! statistical assertion is a two sided `|x - target| <= Z * SE` check with
//! `SE` from 40 batch means, so `Z` is the Student `t` quantile at 39
//! degrees of freedom and level `alpha / (2 m)`, with suite level
//! `alpha = 1/1000` split by Bonferroni over the `m = FAMILY_SIZE` checks
//! counted by [`Checks`]. Two estimates from independent seeds agree when
//! `|a - b| <= Z * hypot(SE_a, SE_b)`. Power is asserted as well: the value
//! a wrong answer would give must lie outside the band, so a result that only
//! passes because its error bar is wide fails. Those values are zero force
//! for the two planes, `2` for the kinetic temperature (the coordinate count
//! before imposing one holonomic constraint), and the zero-temperature
//! barrier `1` for the widening well.

mod pafi_support;

use ndarray::{Array1, Array2, ArrayView1};
use pafi_support::{block_stats, seed};
use rgflux::FluxError;
use rgflux::engine::native::{ConstraintProjection, NativeConfig, NativeEngine};
use rgflux::engine::{DynamicsEngine, HyperplaneConstraint, ThermostatConfig};
use rgflux::hyperplane::{HyperplaneConfig, sample_hyperplane};
use rgflux::pafi::{HermiteIntegral, MeanForceIntegral, PafiConfig, pafi_profile};
use rgflux::stats::batch_means;

/// Statistical assertions over the three tests.
const FAMILY_SIZE: usize = 36;

/// Two sided Student `t` quantile `t_{39, 1 - alpha / (2 m)}` with
/// `alpha = 1/1000` and `m = 36`, so `1 - alpha / (2 m) = 1 - 1/72000`.
const Z_CRIT: f64 = 4.746_483;

const FRICTIONS: [f64; 3] = [0.5, 2.0, 8.0];

const HARMONIC_CHECKS: usize = 9;
const WIDENING_CHECKS: usize = 19;
const TILTED_CHECKS: usize = 8;

#[test]
fn family_size_matches_the_statistical_checks() {
    assert_eq!(
        HARMONIC_CHECKS + WIDENING_CHECKS + TILTED_CHECKS,
        FAMILY_SIZE
    );
}

/// Counts the statistical assertions a test makes, so the family size that
/// sets [`Z_CRIT`] is checked and not assumed.
struct Checks {
    name: &'static str,
    expected: usize,
    done: usize,
}

impl Checks {
    fn new(name: &'static str, expected: usize) -> Self {
        Self {
            name,
            expected,
            done: 0,
        }
    }

    /// `|estimate - target| <= Z * se`, and, when a `rival` is given, the
    /// rival lies outside the same band. The rival is the value a wrong
    /// answer would give, so a standard error too wide to tell them apart
    /// fails.
    fn covers(&mut self, label: &str, estimate: f64, se: f64, target: f64, rival: Option<f64>) {
        self.done += 1;
        assert!(se.is_finite() && se > 0.0, "{label}: bad se {se}");
        let z = (estimate - target).abs() / se;
        eprintln!("{label}: {estimate:.6e} se={se:.3e} target={target:.6e} z={z:.2}");
        assert!(
            z <= Z_CRIT,
            "{label}: {estimate:.6e} +- {se:.3e} versus {target:.6e} gives z={z:.2} > {Z_CRIT}"
        );
        if let Some(rival) = rival {
            let z_rival = (estimate - rival).abs() / se;
            eprintln!("{label}: rival={rival:.6e} z={z_rival:.2}");
            assert!(
                z_rival > Z_CRIT,
                "{label}: the band {estimate:.6e} +- {Z_CRIT} * {se:.3e} cannot exclude {rival:.6e}"
            );
        }
    }

    /// Two independent estimates agree within `Z * hypot(se_a, se_b)`.
    fn agree(&mut self, label: &str, a: (f64, f64), b: (f64, f64)) {
        self.done += 1;
        let spread = a.1.hypot(b.1);
        let z = (a.0 - b.0).abs() / spread;
        eprintln!("{label}: {:.6e} vs {:.6e} z={z:.2}", a.0, b.0);
        assert!(
            z <= Z_CRIT,
            "{label}: {:.6e} and {:.6e} differ by z={z:.2} > {Z_CRIT}",
            a.0,
            b.0
        );
    }

    /// `rival` lies outside `estimate +- Z * se`.
    fn excludes(&mut self, label: &str, estimate: f64, se: f64, rival: f64) {
        self.done += 1;
        let z = (estimate - rival).abs() / se;
        eprintln!("{label}: {estimate:.6e} se={se:.3e} rival={rival:.6e} z={z:.2}");
        assert!(
            z > Z_CRIT,
            "{label}: the band {estimate:.6e} +- {Z_CRIT} * {se:.3e} cannot exclude {rival:.6e}"
        );
    }

    fn finish(self) {
        assert_eq!(
            self.done, self.expected,
            "{}: the family size no longer matches the statistical assertions made",
            self.name
        );
    }
}

/// NVE (`gamma == 0`) has to be refused. `outcome` describes what a call
/// returned if it was accepted.
fn assert_nve_refused(label: &str, outcome: Result<String, FluxError>) {
    match outcome {
        Err(FluxError::InvalidConfig(message)) => {
            assert!(
                message.contains("gamma"),
                "{label}: unrelated error {message}"
            );
        }
        Err(other) => panic!("{label}: expected InvalidConfig, got {other}"),
        Ok(result) => panic!(
            "{label}: gamma == 0 is NVE and does not sample the canonical ensemble, \
             yet it was accepted and returned {result}"
        ),
    }
}

fn unit_masses() -> Array1<f64> {
    Array1::from_vec(vec![1.0, 1.0])
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

struct WideningWell;

impl rgsaddle::PointSurface for WideningWell {
    fn eval(&self, r: ArrayView1<f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
        let (x, y) = (r[0], r[1]);
        let well = 1.0 - x * x;
        let kappa = 1.5 + 0.5 * x;
        Ok((
            well * well + 0.5 * kappa * y * y,
            Array1::from_vec(vec![-4.0 * x * well + 0.25 * y * y, kappa * y]),
        ))
    }
}

struct Harmonic {
    k: [f64; 2],
}

impl rgsaddle::PointSurface for Harmonic {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
        Ok((
            0.5 * (self.k[0] * x[0] * x[0] + self.k[1] * x[1] * x[1]),
            Array1::from_vec(vec![self.k[0] * x[0], self.k[1] * x[1]]),
        ))
    }
}

fn thermostat(gamma: f64, kbt: f64) -> ThermostatConfig {
    ThermostatConfig::default()
        .with_gamma(gamma)
        .with_temperature(kbt)
}

#[test]
fn harmonic_plane_force_and_fluctuations_follow_friction_and_temperature() {
    const K: [f64; 2] = [2.2, 0.7];
    const OFFSET: f64 = -0.4;
    const KBT: f64 = 0.5;
    const SAMPLES: usize = 40_000;
    let plane =
        HyperplaneConstraint::from_direction(Array1::from_vec(vec![5.0, 12.0]).view(), OFFSET)
            .expect("5-12-13 unit normal");
    let (n1, n2) = (plane.normal[0], plane.normal[1]);
    let exact = OFFSET / (n1 * n1 / K[0] + n2 * n2 / K[1]);

    let tangent = [-n2, n1];
    let conditional_mean = {
        let denominator = n1 * n1 / K[0] + n2 * n2 / K[1];
        let x = [
            OFFSET * n1 / K[0] / denominator,
            OFFSET * n2 / K[1] / denominator,
        ];
        tangent[0] * x[0] + tangent[1] * x[1]
    };
    let tangent_stiffness = tangent[0] * tangent[0] * K[0] + tangent[1] * tangent[1] * K[1];

    let run = |gamma: f64,
               kbt: f64,
               projection: ConstraintProjection,
               base: u64|
     -> ((f64, f64), (f64, f64)) {
        let run_seed = seed(base);
        let config = NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.3, 0.8]))
            .with_thermostat(thermostat(gamma, kbt))
            .with_projection(projection)
            .with_seed(run_seed);
        let mut engine = NativeEngine::new(Harmonic { k: K }, config).expect("engine");
        let sampling = HyperplaneConfig::default()
            .with_burn_in_steps(2_000)
            .with_sample_steps(SAMPLES)
            .with_retention_stride(1)
            .with_time_step(0.02)
            .with_seed(run_seed)
            .with_stats(block_stats(SAMPLES));
        let sample = sample_hyperplane(&mut engine, &plane, &sampling).expect("sample");
        let squared_tangent_displacements: Vec<f64> = sample
            .configurations
            .rows()
            .into_iter()
            .map(|position| {
                let coordinate = tangent[0] * position[0] + tangent[1] * position[1];
                (coordinate - conditional_mean).powi(2)
            })
            .collect();
        let fluctuation =
            batch_means(&squared_tangent_displacements, &block_stats(SAMPLES)).expect("variance");
        (
            (sample.mean_force, sample.standard_error),
            (fluctuation.mean, fluctuation.standard_error),
        )
    };

    let mut checks = Checks::new("harmonic", HARMONIC_CHECKS);
    let slow = run(FRICTIONS[0], KBT, ConstraintProjection::Rattle, 101);
    let middle = run(FRICTIONS[1], KBT, ConstraintProjection::Rattle, 102);
    let fast = run(FRICTIONS[2], KBT, ConstraintProjection::Rattle, 103);
    let shake = run(FRICTIONS[1], KBT, ConstraintProjection::Shake, 104);
    for (label, estimate) in [
        ("gamma 0.5 rattle", slow.0),
        ("gamma 2 rattle", middle.0),
        ("gamma 8 rattle", fast.0),
        ("gamma 2 shake", shake.0),
    ] {
        checks.covers(label, estimate.0, estimate.1, exact, Some(0.0));
    }
    checks.agree("gamma 0.5 vs 2", slow.0, middle.0);
    checks.agree("gamma 8 vs 2", fast.0, middle.0);
    checks.agree("shake vs rattle", shake.0, middle.0);

    const LOW_KBT: f64 = 0.2;
    let low_temperature = run(FRICTIONS[1], LOW_KBT, ConstraintProjection::Rattle, 106);
    let variance = KBT / tangent_stiffness;
    let low_variance = LOW_KBT / tangent_stiffness;
    checks.covers(
        "kT 0.5 tangent variance",
        middle.1.0,
        middle.1.1,
        variance,
        Some(low_variance),
    );
    checks.covers(
        "kT 0.2 tangent variance",
        low_temperature.1.0,
        low_temperature.1.1,
        low_variance,
        Some(variance),
    );
    checks.finish();

    let nve_config = NativeConfig::default()
        .with_masses(Array1::from_vec(vec![1.3, 0.8]))
        .with_thermostat(thermostat(0.0, KBT))
        .with_seed(seed(105));
    let mut nve = NativeEngine::new(Harmonic { k: K }, nve_config).expect("engine");
    let sampling = HyperplaneConfig::default()
        .with_burn_in_steps(2_000)
        .with_sample_steps(SAMPLES)
        .with_time_step(0.02)
        .with_seed(seed(105))
        .with_stats(block_stats(SAMPLES));
    assert_nve_refused(
        "sample_hyperplane",
        sample_hyperplane(&mut nve, &plane, &sampling).map(|sample| {
            format!(
                "mean_force {:.4} +- {:.4} against the canonical {exact:.4}",
                sample.mean_force, sample.standard_error
            )
        }),
    );
}

const WIDENING_PLANES: usize = 21;
const WIDENING_BURN_IN: usize = 1_000;
const WIDENING_SAMPLES: usize = 16_000;
const WIDENING_TIME_STEP: f64 = 0.02;
const REACTANT_OMEGA_SQUARED: f64 = 8.0;

fn widening_mean_force(x: f64, kbt: f64) -> f64 {
    -4.0 * x * (1.0 - x * x) + 0.25 * kbt / (1.5 + 0.5 * x)
}

/// Hermite integral of the exact mean force over the sampled planes: the
/// expectation of the PAFI barrier on a path from `x = -1` to `x = 0`.
fn widening_expected_barrier(kbt: f64) -> f64 {
    let arcs: Vec<f64> = (0..WIDENING_PLANES)
        .map(|i| i as f64 / (WIDENING_PLANES - 1) as f64)
        .collect();
    let gradients: Vec<f64> = arcs
        .iter()
        .map(|arc| widening_mean_force(-1.0 + arc, kbt))
        .collect();
    HermiteIntegral
        .cumulative(&arcs, &gradients)
        .expect("hermite")
        .values[WIDENING_PLANES - 1]
}

/// Rate of the one-dimensional reduction with the dividing plane at `x = 0`
/// and the reactant partition function integrated, not replaced by a
/// Gaussian: `sqrt(k_B T / 2 pi) exp(-Delta F / k_B T) / Z_R` with
/// `Z_R = int exp(-(F(x) - F(-1)) / k_B T) dx` over `x < 0`.
///
/// `F(x) = (1 - x^2)^2 + (k_B T / 2) ln(1.5 + 0.5x)` is normalizable for
/// `x > -3`; the weight there is below `exp(-60)`, so the integral starts at
/// `-2.9`. Simpson's rule on `1_000` intervals is far below the statistical
/// error.
fn widening_reduced_rate(kbt: f64, barrier: f64) -> f64 {
    let free_energy = |x: f64| (1.0 - x * x).powi(2) + 0.5 * kbt * (1.5 + 0.5 * x).ln();
    let reference = free_energy(-1.0);
    let (lo, hi, intervals) = (-2.9_f64, 0.0_f64, 1_000_usize);
    let h = (hi - lo) / intervals as f64;
    let mut sum = 0.0;
    for i in 0..=intervals {
        let x = lo + h * i as f64;
        let weight = if i == 0 || i == intervals {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        sum += weight * (-(free_energy(x) - reference) / kbt).exp();
    }
    let reactant_partition = sum * h / 3.0;
    (kbt / std::f64::consts::TAU).sqrt() * (-barrier / kbt).exp() / reactant_partition
}

fn widening_path() -> Array2<f64> {
    let mut path = Array2::zeros((WIDENING_PLANES, 2));
    for i in 0..WIDENING_PLANES {
        path[[i, 0]] = -1.0 + i as f64 / (WIDENING_PLANES - 1) as f64;
    }
    path
}

fn widening_profile(
    gamma: f64,
    kbt: f64,
    base: u64,
) -> Result<rgflux::pafi::FreeEnergyProfile, FluxError> {
    let run_seed = seed(base);
    let config = PafiConfig::default()
        .with_planes(WIDENING_PLANES)
        .with_burn_in_steps(WIDENING_BURN_IN)
        .with_sample_steps(WIDENING_SAMPLES)
        .with_time_step(WIDENING_TIME_STEP)
        .with_stats(block_stats(WIDENING_SAMPLES))
        .with_seed(run_seed);
    let mut engine = NativeEngine::new(
        WideningWell,
        NativeConfig::default()
            .with_masses(unit_masses())
            .with_thermostat(thermostat(gamma, kbt))
            .with_seed(run_seed),
    )
    .expect("engine");
    pafi_profile(&mut engine, widening_path().view(), &config)
}

#[test]
fn widening_well_barrier_and_rate_are_independent_of_friction() {
    let mut checks = Checks::new("widening", WIDENING_CHECKS);
    let zero_temperature_barrier = 1.0;
    let mut barriers = Vec::new();
    for (index, (kbt, base)) in [(0.2, 200_u64), (0.35, 300)].into_iter().enumerate() {
        let target = widening_expected_barrier(kbt);
        let target_rate =
            REACTANT_OMEGA_SQUARED.sqrt() / std::f64::consts::TAU * (-target / kbt).exp();
        let reduced = widening_reduced_rate(kbt, target);
        assert!(
            (1.02..1.12).contains(&(target_rate / reduced)),
            "kT {kbt}: the Gaussian-reactant rate {target_rate:.4e} must sit a few percent above \
             the integrated-partition rate {reduced:.4e}"
        );
        for (slot, gamma) in FRICTIONS.into_iter().enumerate() {
            let profile =
                widening_profile(gamma, kbt, base + slot as u64).expect("widening profile");
            let label = format!("kT {kbt} gamma {gamma}");
            assert!(
                (profile.reactant_frequency - REACTANT_OMEGA_SQUARED.sqrt()).abs() < 1e-6,
                "{label}: reactant frequency {}",
                profile.reactant_frequency
            );
            checks.covers(
                &format!("{label} barrier"),
                profile.barrier,
                profile.barrier_error,
                target,
                Some(zero_temperature_barrier),
            );
            let rate = profile
                .normalized_rate
                .expect("widening rate must be representable");
            let rate_error = profile
                .normalized_rate_error
                .expect("widening rate error must be representable");
            checks.covers(
                &format!("{label} rate"),
                rate,
                rate_error,
                target_rate,
                None,
            );
            if kbt > 0.3 {
                checks.excludes(
                    &format!("{label} rate against the integrated-partition rate"),
                    rate,
                    rate_error,
                    widening_reduced_rate(kbt, target),
                );
            }
            barriers.push((profile.barrier, profile.barrier_error));
        }
        let first = 3 * index;
        checks.agree(
            &format!("kT {kbt} gamma 0.5 vs 2"),
            barriers[first],
            barriers[first + 1],
        );
        checks.agree(
            &format!("kT {kbt} gamma 0.5 vs 8"),
            barriers[first],
            barriers[first + 2],
        );
    }
    checks.finish();

    assert_nve_refused(
        "pafi_profile",
        widening_profile(0.0, 0.2, 400).map(|profile| {
            format!(
                "barrier {:.4} +- {:.4} against the canonical {:.4}",
                profile.barrier,
                profile.barrier_error,
                widening_expected_barrier(0.2)
            )
        }),
    );
}

#[test]
fn tilted_plane_entropic_force_and_kinetic_temperature_are_independent_of_friction() {
    const KBT: f64 = 0.3;
    const THETA: f64 = 0.7;
    const SAMPLES: usize = 40_000;
    let normal = Array1::from_vec(vec![THETA.cos(), THETA.sin()]);
    let plane = HyperplaneConstraint::new(normal, 0.0).expect("plane");
    let entropic = 0.25 * KBT / 1.5;

    let force = |gamma: f64, masses: Vec<f64>, base: u64| -> Result<(f64, f64), FluxError> {
        let run_seed = seed(base);
        let config = NativeConfig::default()
            .with_masses(Array1::from_vec(masses))
            .with_thermostat(thermostat(gamma, KBT))
            .with_seed(run_seed);
        let mut engine = NativeEngine::new(TiltedWidening { theta: THETA }, config)?;
        let sampling = HyperplaneConfig::default()
            .with_burn_in_steps(2_000)
            .with_sample_steps(SAMPLES)
            .with_time_step(0.02)
            .with_seed(run_seed)
            .with_stats(block_stats(SAMPLES));
        let sample = sample_hyperplane(&mut engine, &plane, &sampling)?;
        Ok((sample.mean_force, sample.standard_error))
    };
    let kinetic = |gamma: f64, base: u64| -> (f64, f64) {
        let run_seed = seed(base);
        let config = NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0, 2.5]))
            .with_thermostat(thermostat(gamma, KBT))
            .with_constraint(plane.clone())
            .with_seed(run_seed);
        let mut engine =
            NativeEngine::new(TiltedWidening { theta: THETA }, config).expect("engine");
        engine.shoot_velocities(run_seed).expect("velocities");
        for _ in 0..2_000 {
            engine.step(0.02).expect("burn-in");
        }
        let mut series = Vec::with_capacity(SAMPLES);
        for _ in 0..SAMPLES {
            let report = engine.step(0.02).expect("step");
            series.push(2.0 * report.kinetic_energy / KBT);
        }
        let estimate = batch_means(&series, &block_stats(SAMPLES)).expect("batch means");
        (estimate.mean, estimate.standard_error)
    };

    let mut checks = Checks::new("tilted", TILTED_CHECKS);
    let mut forces = Vec::new();
    for (slot, gamma) in FRICTIONS.into_iter().enumerate() {
        let estimate = force(gamma, vec![1.0, 1.0], 500 + slot as u64).expect("force");
        checks.covers(
            &format!("gamma {gamma} multiplier"),
            estimate.0,
            estimate.1,
            entropic,
            Some(0.0),
        );
        forces.push(estimate);
    }
    checks.agree("multiplier gamma 0.5 vs 2", forces[0], forces[1]);
    checks.agree("multiplier gamma 0.5 vs 8", forces[0], forces[2]);
    for (slot, gamma) in FRICTIONS.into_iter().enumerate() {
        let estimate = kinetic(gamma, 600 + slot as u64);
        checks.covers(
            &format!("gamma {gamma} 2KE/kT"),
            estimate.0,
            estimate.1,
            1.0,
            Some(2.0),
        );
    }
    checks.finish();

    assert_nve_refused(
        "sample_hyperplane",
        force(0.0, vec![1.0, 1.0], 700).map(|(mean, se)| {
            format!("multiplier {mean:.4} +- {se:.4} against the canonical {entropic:.4}")
        }),
    );
}
