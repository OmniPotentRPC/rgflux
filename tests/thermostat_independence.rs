//! Constrained mean forces and PAFI profiles against the Langevin friction.
//!
//! The native engine samples with white-noise Langevin dynamics at friction
//! `gamma > 0`, with `gamma == 0` giving NVE dynamics, and offers a
//! smart-sampling generalized Langevin equation as a second noise. The
//! Langevin tests vary what that family exposes:
//!
//! - the friction, over a factor of sixteen (`0.5`, `2`, `8`),
//! - the constraint scheme, RATTLE or SHAKE, which change how the plane is
//!   enforced and not what is sampled,
//! - the temperature, where the reference moves with `k_B T` and the same
//!   estimator has to follow it,
//! - the white-noise NVE choice, `gamma == 0`. NVE samples an energy shell
//!   started at the anchor and not the canonical ensemble, so
//!   `sample_hyperplane` and `pafi_profile` must refuse it. Before that check
//!   they accepted it and returned a biased number; every test below asserts
//!   the refusal. The smart-sampling GLE remains thermostatted at zero
//!   `gamma`.
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
//!   `sqrt(k_B T / 2 pi) exp(-Delta F / k_B T) / Z_R`, with `Z_R` the doubled
//!   trapezoid of `exp(-F(s) / k_B T)` over the expected profile. The
//!   harmonic-reactant rate `(omega_0 / 2 pi) exp(-Delta F / k_B T)`, with
//!   `omega_0^2 = 8` the curvature of `(1 - x^2)^2` at `x = -1`, lies a
//!   few percent above it because the reactant well is not a Gaussian, and
//!   the rate PAFI reports must exclude it.
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
//!
//! The last tests run white noise and the smart-sampling GLE through the
//! same engine and compare each against the exact equilibrium force and
//! against the other, with fixed seeds and a two standard error band.

mod pafi_support;

use ndarray::{Array1, Array2, ArrayView1};
use pafi_support::{block_stats, seed};
use rgflux::FluxError;
use rgflux::engine::gle::ColoredNoise;
use rgflux::engine::native::{ConstraintProjection, NativeConfig, NativeEngine};
use rgflux::engine::{DynamicsEngine, EindirSurface, HyperplaneConstraint, ThermostatConfig};
use rgflux::hyperplane::{HyperplaneConfig, sample_hyperplane};
use rgflux::pafi::{HermiteIntegral, MeanForceIntegral, PafiConfig, pafi_profile};
use rgflux::stats::{StatsConfig, batch_means};

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

/// Arc lengths and Hermite integral of the exact mean force over the sampled
/// planes: the expectation of the PAFI profile on a path from `x = -1` to
/// `x = 0`.
fn widening_expected_profile(kbt: f64) -> (Vec<f64>, Vec<f64>) {
    let arcs: Vec<f64> = (0..WIDENING_PLANES)
        .map(|i| i as f64 / (WIDENING_PLANES - 1) as f64)
        .collect();
    let gradients: Vec<f64> = arcs
        .iter()
        .map(|arc| widening_mean_force(-1.0 + arc, kbt))
        .collect();
    let values = HermiteIntegral
        .cumulative(&arcs, &gradients)
        .expect("hermite")
        .values;
    (arcs, values)
}

/// Hermite integral of the exact mean force over the sampled planes: the
/// expectation of the PAFI barrier on a path from `x = -1` to `x = 0`.
fn widening_expected_barrier(kbt: f64) -> f64 {
    let (_, values) = widening_expected_profile(kbt);
    values[WIDENING_PLANES - 1]
}

/// The normalized rate PAFI reports for the expected profile, with the
/// dividing plane at `x = 0`: `sqrt(k_B T / 2 pi) exp(-Delta F / k_B T) / Z_R`
/// and `Z_R` the doubled trapezoid of `exp(-F(s) / k_B T)` over the planes.
fn widening_expected_rate(kbt: f64) -> f64 {
    let (arcs, values) = widening_expected_profile(kbt);
    let weights: Vec<f64> = values.iter().map(|value| (-value / kbt).exp()).collect();
    let half_partition: f64 = arcs
        .windows(2)
        .zip(weights.windows(2))
        .map(|(arc, weight)| 0.5 * (arc[1] - arc[0]) * (weight[0] + weight[1]))
        .sum();
    let barrier = values[WIDENING_PLANES - 1];
    (kbt / std::f64::consts::TAU).sqrt() * (-barrier / kbt).exp() / (2.0 * half_partition)
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
        let target_rate = widening_expected_rate(kbt);
        let gaussian_rate =
            REACTANT_OMEGA_SQUARED.sqrt() / std::f64::consts::TAU * (-target / kbt).exp();
        assert!(
            (1.05..1.25).contains(&(gaussian_rate / target_rate)),
            "kT {kbt}: the Gaussian-reactant rate {gaussian_rate:.4e} must sit well above \
             the doubled-trapezoid rate {target_rate:.4e}"
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
                    &format!("{label} rate against the Gaussian-reactant rate"),
                    rate,
                    rate_error,
                    gaussian_rate,
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
            "gle harmonic {noise:?} force={:.6} se={:.6} exact={exact:.6} z={z:.3}",
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
                "gle harmonic gap={gap:.6e} se={gap_se:.6e} z={:.3}",
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
            eprintln!("gle plane {plane} {noise:?} force={force:.6} se={error:.6}");
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
            "gle plane {plane} gap={gap:.6e} se={gap_se:.6e} z={:.3}",
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
            "gle well {noise:?} force={:.6} se={:.6} exact={exact:.6} z={:.3} T_dof={:.4} se={:.4} z={:.3}",
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
            "gle flat {noise:?} T_dof={:.4} se={:.4} z={:.3}",
            flat_est.mean,
            flat_est.standard_error,
            (flat_est.mean - 1.0) / flat_est.standard_error
        );
        assert!((flat_est.mean - 1.0).abs() <= 2.0 * flat_est.standard_error);
    }
}
