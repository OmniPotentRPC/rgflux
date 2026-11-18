//! Value checks for the harmonic OH-TST rate and the hyperplane constraint force.
//!
//! The double well is harmonic about each minimum out to a seam, then an even
//! quartic matched in value, slope, and curvature. The Boltzmann weight of the
//! seam is far below double precision at the temperature used here, so the
//! harmonic TST rate is the OH-TST rate. A separate harmonic oscillator checks
//! the mean constraint force at more than one plane offset.
//!
//! Acceptance is `max_z_score` times the reported standard error. For a flat
//! sample, a 64-ULP relative bound covers the floating-point rate evaluation.
//! rgsaddle's [`rgsaddle::OhTstRate::error`] is only the statistical part;
//! the leave-one-out error of `ln ρ` is [`rgsaddle::BasinDensity::quadrature_error`]
//! and is added in quadrature. The temperature puts the anharmonic seam many
//! thermal widths past the basin window, so that interpolation error covers
//! the remaining difference from the harmonic rate.

use std::f64::consts::TAU;

use ndarray::{Array1, ArrayView1};
use rgsaddle::{
    BasinConfig, Hyperplane, LangevinConfig, PointSurface, ReversibleWorkConfig, SaddleError,
    basin_density, ohtst_rate, reversible_work,
};

use rgflux::FluxError;
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::engine::{DynamicsEngine, HyperplaneConstraint, ThermostatConfig};
use rgflux::hyperplane::{HyperplaneConfig, sample_hyperplane};
use rgflux::ohtst::{OhtstConfig, optimize_hyperplane};

const SEED: u64 = 17;
const THERMAL: f64 = 0.005;
const SEAM: f64 = 0.25;
const HARMONIC_RATE_ROUNDOFF_ULPS: f64 = 64.0;
const DIVIDING_OFFSET_TOLERANCE: f64 = 1_024.0 * f64::EPSILON;

struct HarmonicBarrier;

impl PointSurface for HarmonicBarrier {
    fn eval(&self, position: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        if position.len() != 1 {
            return Err(SaddleError::Shape(
                "HarmonicBarrier is one dimensional".into(),
            ));
        }
        let x = position[0];
        let (energy, slope) = barrier_energy_slope(x);
        Ok((energy, Array1::from_vec(vec![slope])))
    }
}

struct AsymmetricRightBarrier;

impl PointSurface for AsymmetricRightBarrier {
    fn eval(&self, position: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        if position.len() != 1 {
            return Err(SaddleError::Shape(
                "AsymmetricRightBarrier is one dimensional".into(),
            ));
        }
        let x = position[0];
        if x > 1.0 {
            let displacement = x - 1.0;
            return Ok((
                4.0 * displacement * displacement,
                Array1::from_vec(vec![8.0 * displacement]),
            ));
        }
        let (energy, slope) = barrier_energy_slope(x);
        Ok((energy, Array1::from_vec(vec![slope])))
    }
}

struct UnitHarmonic;

impl PointSurface for UnitHarmonic {
    fn eval(&self, position: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        if position.len() != 1 {
            return Err(SaddleError::Shape("UnitHarmonic is one dimensional".into()));
        }
        let x = position[0];
        Ok((0.5 * x * x, Array1::from_vec(vec![x])))
    }
}

fn barrier_polynomial() -> (f64, f64, f64) {
    let seam_value = (1.0 - SEAM) * (1.0 - SEAM);
    let quartic = 1.0 / (4.0 * SEAM * SEAM * SEAM);
    let quadratic = 1.0 - 1.5 / SEAM;
    let constant = seam_value - quadratic * SEAM * SEAM - quartic * SEAM * SEAM * SEAM * SEAM;
    (constant, quadratic, quartic)
}

fn barrier_energy_slope(x: f64) -> (f64, f64) {
    if x <= -SEAM {
        let displacement = x + 1.0;
        return (displacement * displacement, 2.0 * displacement);
    }
    if x >= SEAM {
        let displacement = x - 1.0;
        return (displacement * displacement, 2.0 * displacement);
    }
    let (constant, quadratic, quartic) = barrier_polynomial();
    let energy = constant + quadratic * x * x + quartic * x * x * x * x;
    let slope = 2.0 * quadratic * x + 4.0 * quartic * x * x * x;
    (energy, slope)
}

fn harmonic_tst_rate(barrier: f64, thermal: f64) -> f64 {
    2.0_f64.sqrt() / TAU * (-barrier / thermal).exp()
}

fn within_reported_error(actual: f64, reference: f64, standard_error: f64, coverage: f64) -> bool {
    (actual - reference).abs() <= coverage * standard_error
}

fn within_harmonic_rate_error(
    actual: f64,
    reference: f64,
    standard_error: f64,
    coverage: f64,
) -> bool {
    let roundoff_bound = HARMONIC_RATE_ROUNDOFF_ULPS * f64::EPSILON * reference.abs();
    (actual - reference).abs() <= coverage * standard_error + roundoff_bound
}

fn engine<S: PointSurface + Send + Sync>(
    surface: S,
    temperature: f64,
) -> Result<NativeEngine<S>, FluxError> {
    NativeEngine::new(
        surface,
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0]))
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(1.0)
                    .with_temperature(temperature),
            )
            .with_seed(SEED),
    )
}

#[test]
fn hyperplane_mean_force_matches_harmonic_offsets() -> Result<(), FluxError> {
    let offsets = [0.4, -0.7];
    let config = HyperplaneConfig::default()
        .with_burn_in_steps(4)
        .with_sample_steps(16)
        .with_retention_stride(1)
        .with_time_step(0.01)
        .with_seed(SEED);
    let coverage = config.stats.max_z_score;

    for offset in offsets {
        let plane =
            HyperplaneConstraint::from_direction(Array1::from_vec(vec![1.0]).view(), offset)?;
        let mut sampler = engine(UnitHarmonic, 0.5)?;
        sampler.set_positions(Array1::from_vec(vec![offset]).view())?;
        let positions = sampler.positions().to_owned();
        let velocities = sampler.velocities().to_owned();
        let constraint = sampler.constraint().cloned();
        let thermostat = *sampler.thermostat();
        let sampled = sample_hyperplane(&mut sampler, &plane, &config)?;
        assert_eq!(sampler.positions(), positions.view());
        assert_eq!(sampler.velocities(), velocities.view());
        assert_eq!(sampler.constraint(), constraint.as_ref());
        assert_eq!(sampler.thermostat(), &thermostat);
        assert!(
            within_reported_error(sampled.mean_force, offset, sampled.standard_error, coverage),
            "mean force {} ± {} against analytic {offset} at coverage {coverage}",
            sampled.mean_force,
            sampled.standard_error
        );
        assert_eq!(sampled.mean_projected_force, Array1::zeros(1));
        assert_eq!(sampled.mean_position, Array1::from_vec(vec![offset]));
        assert_eq!(sampled.position_covariance[[0, 0]], 0.0);
    }
    Ok(())
}

#[test]
fn ohtst_rate_matches_harmonic_tst_and_rgsaddle() -> Result<(), Box<dyn std::error::Error>> {
    let (barrier, _, _) = barrier_polynomial();
    let analytic = harmonic_tst_rate(barrier, THERMAL);
    let config = OhtstConfig::default()
        .with_max_translation(0.05)
        .with_seed(SEED);
    let coverage = config.stats.max_z_score;

    let mut sampler = engine(HarmonicBarrier, THERMAL)?;
    sampler.set_positions(Array1::from_vec(vec![-1.0]).view())?;
    let positions = sampler.positions().to_owned();
    let velocities = sampler.velocities().to_owned();
    let constraint = sampler.constraint().cloned();
    let thermostat = *sampler.thermostat();
    let seed = HyperplaneConstraint::from_direction(Array1::from_vec(vec![1.0]).view(), 0.0)?;
    let estimate = optimize_hyperplane(&mut sampler, &seed, &config)?;
    assert_eq!(sampler.positions(), positions.view());
    assert_eq!(sampler.velocities(), velocities.view());
    assert_eq!(sampler.constraint(), constraint.as_ref());
    assert_eq!(sampler.thermostat(), &thermostat);
    assert!(
        within_harmonic_rate_error(estimate.rate, analytic, estimate.standard_error, coverage),
        "rgflux rate {} ± {} against harmonic TST {analytic}",
        estimate.rate,
        estimate.standard_error
    );

    let rg = rgsaddle_rate()?;
    let combined = (estimate.standard_error * estimate.standard_error + rg.1 * rg.1).sqrt();
    assert!(
        within_reported_error(estimate.rate, rg.0, combined, coverage),
        "rgflux rate {} ± {} against rgsaddle {} ± {}",
        estimate.rate,
        estimate.standard_error,
        rg.0,
        rg.1
    );
    Ok(())
}

#[test]
fn ohtst_walks_from_both_reactant_wells_to_the_known_dividing_surface() -> Result<(), FluxError> {
    let config = OhtstConfig::default()
        .with_max_translation(0.05)
        .with_seed(SEED);
    let normal = Array1::from_vec(vec![1.0]);

    let mut left = engine(HarmonicBarrier, THERMAL)?;
    left.set_positions(Array1::from_vec(vec![-1.0]).view())?;
    let left_seed = HyperplaneConstraint::from_direction(normal.view(), 0.2)?;
    let left_estimate = optimize_hyperplane(&mut left, &left_seed, &config)?;
    assert!(
        left_estimate.plane.offset.abs() <= DIVIDING_OFFSET_TOLERANCE,
        "left-hand walk stopped at {}",
        left_estimate.plane.offset
    );

    let mut right = engine(HarmonicBarrier, THERMAL)?;
    right.set_positions(Array1::from_vec(vec![1.0]).view())?;
    let right_seed = HyperplaneConstraint::from_direction(normal.view(), -0.2)?;
    let right_estimate = optimize_hyperplane(&mut right, &right_seed, &config)?;
    assert!(
        right_estimate.plane.offset.abs() <= DIVIDING_OFFSET_TOLERANCE,
        "right-hand walk stopped at {}",
        right_estimate.plane.offset
    );
    assert!(
        right_estimate.rate.is_finite() && right_estimate.rate > 0.0,
        "toward=-1 should retain a positive reactant curvature"
    );
    Ok(())
}

#[test]
fn ohtst_negative_probe_direction_uses_the_reactant_side_curvature() -> Result<(), FluxError> {
    let (barrier, _, _) = barrier_polynomial();
    let expected_rate = harmonic_tst_rate(barrier, THERMAL);
    let config = OhtstConfig::default()
        .with_max_translation(0.05)
        .with_seed(SEED);
    let normal = Array1::from_vec(vec![1.0]);
    let seed = HyperplaneConstraint::from_direction(normal.view(), -0.2)?;
    let mut sampler = engine(AsymmetricRightBarrier, THERMAL)?;
    sampler.set_positions(Array1::from_vec(vec![1.0]).view())?;

    let estimate = optimize_hyperplane(&mut sampler, &seed, &config)?;

    assert!(
        estimate.plane.offset.abs() <= DIVIDING_OFFSET_TOLERANCE,
        "right-hand walk stopped at {}",
        estimate.plane.offset
    );
    assert!(
        within_harmonic_rate_error(
            estimate.rate,
            expected_rate,
            estimate.standard_error,
            config.stats.max_z_score
        ),
        "negative probe sampled rate {} ± {} instead of {expected_rate}",
        estimate.rate,
        estimate.standard_error
    );
    Ok(())
}

fn rgsaddle_rate() -> Result<(f64, f64), SaddleError> {
    let sampler = LangevinConfig::default()
        .with_thermal(THERMAL)
        .with_time_step(0.05)
        .with_burn_in(20)
        .with_records(80)
        .with_blocks(4)
        .with_stride(1)
        .with_seed(SEED);
    let basin = basin_density(
        &HarmonicBarrier,
        Array1::from_vec(vec![-1.0]).view(),
        Array1::from_vec(vec![1.0]).view(),
        &BasinConfig::default()
            .with_sampler(sampler.clone())
            .with_cap(40.0)
            .with_width(12.0),
    )?;
    let dividing = Hyperplane::new(Array1::from_vec(vec![0.0]), Array1::from_vec(vec![1.0]))?;
    let work = reversible_work(
        &HarmonicBarrier,
        &[basin.plane.clone(), dividing],
        Some(Array1::from_vec(vec![-1.0]).view()),
        &ReversibleWorkConfig::default()
            .with_sampler(sampler)
            .with_nodes(9)
            .with_bar_samples(0),
    )?;
    let rate = ohtst_rate(&basin, &work)?;
    let log_error = (rate.log_rate_error * rate.log_rate_error
        + basin.quadrature_error * basin.quadrature_error)
        .sqrt();
    Ok((rate.rate, rate.rate * log_error))
}
