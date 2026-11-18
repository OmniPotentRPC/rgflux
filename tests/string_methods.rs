//! Deterministic finite-temperature string checks on analytic surfaces.

use ndarray::{Array1, Array2, ArrayView1, array};
use rgsaddle::{PointSurface, SaddleError};

use rgflux::FluxError;
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::engine::{DynamicsEngine, ThermostatConfig};
use rgflux::hyperplane::HyperplaneConfig;
use rgflux::string::{
    FiniteTemperatureStringConfig, SwarmStringConfig, finite_temperature_string, swarm_string,
};

const STRING_SEED: u64 = 0x5eed;
const MULLER_BROWN_TEMPERATURE: f64 = 2.0;
const MULLER_BROWN_TIME_STEP: f64 = 0.000_05;

struct MullerBrown;

impl PointSurface for MullerBrown {
    fn eval(&self, position: ArrayView1<'_, f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        if position.len() != 2 {
            return Err(SaddleError::Shape("MullerBrown is two dimensional".into()));
        }
        let amplitudes = [-200.0, -100.0, -170.0, 15.0];
        let x_quadratic = [-1.0, -1.0, -6.5, 0.7];
        let xy_quadratic = [0.0, 0.0, 11.0, 0.6];
        let y_quadratic = [-10.0, -10.0, -6.5, 0.7];
        let x_centers = [1.0, 0.0, -0.5, -1.0];
        let y_centers = [0.0, 0.5, 1.5, 1.0];
        let mut energy = 0.0;
        let mut gradient = Array1::zeros(2);
        for term in 0..amplitudes.len() {
            let dx = position[0] - x_centers[term];
            let dy = position[1] - y_centers[term];
            let exponent = x_quadratic[term] * dx * dx
                + xy_quadratic[term] * dx * dy
                + y_quadratic[term] * dy * dy;
            let contribution = amplitudes[term] * exponent.exp();
            energy += contribution;
            gradient[0] += contribution * (2.0 * x_quadratic[term] * dx + xy_quadratic[term] * dy);
            gradient[1] += contribution * (xy_quadratic[term] * dx + 2.0 * y_quadratic[term] * dy);
        }
        Ok((energy, gradient))
    }
}

fn muller_brown_engine() -> Result<NativeEngine<MullerBrown>, FluxError> {
    NativeEngine::new(
        MullerBrown,
        NativeConfig::default()
            .with_masses(array![1.0, 1.0])
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(10.0)
                    .with_temperature(MULLER_BROWN_TEMPERATURE),
            )
            .with_seed(STRING_SEED),
    )
}

fn linear_path(start: ArrayView1<'_, f64>, end: ArrayView1<'_, f64>, count: usize) -> Array2<f64> {
    let mut images = Array2::zeros((count, start.len()));
    for index in 0..count {
        let fraction = index as f64 / (count - 1) as f64;
        images
            .row_mut(index)
            .assign(&(&start + &((&end - &start) * fraction)));
    }
    images
}

#[test]
fn finite_temperature_string_reaches_muller_brown_path_region() -> Result<(), FluxError> {
    let reactant = array![-0.558_224, 1.441_73];
    let product = array![0.623_499, 0.028_038];
    let known_saddle = array![-0.822_002, 0.624_313];
    let initial = linear_path(reactant.view(), product.view(), 17);
    let sampling = HyperplaneConfig::default()
        .with_burn_in_steps(64)
        .with_sample_steps(256)
        .with_retention_stride(16)
        .with_time_step(MULLER_BROWN_TIME_STEP)
        .with_seed(STRING_SEED);
    let config = FiniteTemperatureStringConfig::default()
        .with_max_iterations(80)
        .with_patience(2)
        .with_evolution_step(0.000_2)
        .with_force_tolerance(65.0)
        .with_sampling(sampling)
        .with_seed(STRING_SEED);
    let mut engine = muller_brown_engine()?;
    engine.set_positions(reactant.view())?;
    let before = engine.positions().to_owned();
    let result = finite_temperature_string(&mut engine, initial.view(), &config)?;
    assert_eq!(engine.positions(), before.view());
    for (actual, expected) in result.images.row(0).iter().zip(reactant.iter()) {
        assert!((actual - expected).abs() <= 16.0 * f64::EPSILON);
    }
    for (actual, expected) in result
        .images
        .row(result.images.nrows() - 1)
        .iter()
        .zip(product.iter())
    {
        assert!((actual - expected).abs() <= 16.0 * f64::EPSILON);
    }

    let saddle_distance = result
        .images
        .rows()
        .into_iter()
        .map(|image| {
            (&image - &known_saddle)
                .dot(&(&image - &known_saddle))
                .sqrt()
        })
        .fold(f64::INFINITY, f64::min);
    assert!(result.converged);
    // The fixed-seed calibration reaches 0.06969; this bound retains the
    // measured figure-level resolution of the 17-image path.
    assert!(saddle_distance < 0.08);
    assert!(
        result
            .images
            .column(0)
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
            < -0.9
    );
    Ok(())
}

#[test]
fn swarm_string_runs_shared_plane_trajectories() -> Result<(), FluxError> {
    let reactant = array![-0.558_224, 1.441_73];
    let product = array![0.623_499, 0.028_038];
    let initial = linear_path(reactant.view(), product.view(), 3);
    let sampling = HyperplaneConfig::default()
        .with_burn_in_steps(2)
        .with_sample_steps(4)
        .with_retention_stride(2)
        .with_time_step(MULLER_BROWN_TIME_STEP);
    let config = SwarmStringConfig::default()
        .with_max_iterations(1)
        .with_swarm_size(2)
        .with_sampling(sampling)
        .with_seed(STRING_SEED);
    let mut engine = muller_brown_engine()?;
    engine.set_positions(reactant.view())?;
    let before = engine.positions().to_owned();
    let result = swarm_string(&mut engine, initial.view(), &config)?;
    assert_eq!(result.iterations, 1);
    assert_eq!(engine.positions(), before.view());
    assert!(
        result
            .mean_displacements
            .iter()
            .all(|value| value.is_finite())
    );
    Ok(())
}

#[test]
fn public_string_validation_fails_before_old_stub_behavior() -> Result<(), FluxError> {
    let mut engine = muller_brown_engine()?;
    let coincident = array![[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]];
    let sampling = HyperplaneConfig::default()
        .with_burn_in_steps(0)
        .with_sample_steps(2);
    let finite = FiniteTemperatureStringConfig::default()
        .with_max_iterations(1)
        .with_sampling(sampling.clone());
    assert!(finite_temperature_string(&mut engine, coincident.view(), &finite).is_err());

    let swarm = SwarmStringConfig::default()
        .with_max_iterations(1)
        .with_swarm_size(1)
        .with_sampling(sampling);
    let valid = array![[0.0, 0.0], [0.5, 0.25], [1.0, 0.0]];
    assert!(swarm_string(&mut engine, valid.view(), &swarm).is_err());
    Ok(())
}
