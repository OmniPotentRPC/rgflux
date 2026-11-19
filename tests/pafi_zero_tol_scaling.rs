use ndarray::{Array1, Array2, ArrayView1};
use rgflux::FluxError;
use rgflux::engine::ThermostatConfig;
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::pafi::{PafiConfig, pafi_profile};
use rgflux::stats::StatsConfig;

struct InconsistentLinearGradient;

impl rgsaddle::PointSurface for InconsistentLinearGradient {
    fn eval(&self, x: ArrayView1<'_, f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
        Ok((x[0], Array1::from_vec(vec![0.98])))
    }
}

#[test]
fn default_tolerance_scales_with_planes_for_consistency_check() {
    let path = Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).expect("shape");
    let mut engine = NativeEngine::new(
        InconsistentLinearGradient,
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0]))
            .with_thermostat(ThermostatConfig::default().with_temperature(0.2))
            .with_seed(23),
    )
    .expect("engine");
    let config = PafiConfig::default()
        .with_planes(33)
        .with_burn_in_steps(1)
        .with_sample_steps(2);
    let result = pafi_profile(&mut engine, path.view(), &config);
    assert_eq!(
        result,
        Err(FluxError::InvalidConfig(
            "PAFI zero-temperature path-force quadrature exceeds tolerance"
        ))
    );

    let mut legacy_engine = NativeEngine::new(
        InconsistentLinearGradient,
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0]))
            .with_thermostat(ThermostatConfig::default().with_temperature(0.2))
            .with_seed(23),
    )
    .expect("engine");
    let legacy_like = PafiConfig::default()
        .with_planes(33)
        .with_zero_temperature_tol(0.0257)
        .with_burn_in_steps(1)
        .with_sample_steps(2)
        .with_stats(StatsConfig::default().with_min_batches(2));
    assert!(pafi_profile(&mut legacy_engine, path.view(), &legacy_like).is_ok());
}

fn inconsistent_engine() -> NativeEngine<InconsistentLinearGradient> {
    NativeEngine::new(
        InconsistentLinearGradient,
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0]))
            .with_thermostat(ThermostatConfig::default().with_temperature(0.2))
            .with_seed(23),
    )
    .expect("engine")
}

fn run_at_33_planes(config: &PafiConfig) -> Result<(), FluxError> {
    let path = Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).expect("shape");
    let config = config
        .clone()
        .with_planes(33)
        .with_burn_in_steps(1)
        .with_sample_steps(2)
        .with_stats(StatsConfig::default().with_min_batches(2));
    pafi_profile(&mut inconsistent_engine(), path.view(), &config).map(|_| ())
}

const REJECTED: Result<(), FluxError> = Err(FluxError::InvalidConfig(
    "PAFI zero-temperature path-force quadrature exceeds tolerance",
));

#[test]
fn unset_tolerance_is_rescaled_for_the_plane_count() {
    assert_eq!(run_at_33_planes(&PafiConfig::default()), REJECTED);
}

#[test]
fn setter_value_next_above_default_is_used_exactly() {
    let default_tol = PafiConfig::default().zero_temperature_tol;
    let next = default_tol.next_up();
    assert!(next > default_tol);
    let config = PafiConfig::default().with_zero_temperature_tol(next);
    assert_eq!(config.zero_temperature_tol, next);
    assert_eq!(run_at_33_planes(&config), Ok(()));
}

#[test]
fn setter_value_next_below_default_is_used_exactly() {
    let default_tol = PafiConfig::default().zero_temperature_tol;
    let config = PafiConfig::default().with_zero_temperature_tol(default_tol.next_down());
    assert_eq!(run_at_33_planes(&config), Ok(()));
}

#[test]
fn setter_value_equal_to_default_is_pinned_and_not_rescaled() {
    let default_tol = PafiConfig::default().zero_temperature_tol;
    let config = PafiConfig::default().with_zero_temperature_tol(default_tol);
    assert_eq!(run_at_33_planes(&config), Ok(()));
}

#[test]
fn setter_value_is_exact_in_either_call_order() {
    let default_tol = PafiConfig::default().zero_temperature_tol;
    let config = PafiConfig::default()
        .with_zero_temperature_tol(default_tol.next_up())
        .with_planes(33);
    assert_eq!(config.planes, 33);
    assert_eq!(run_at_33_planes(&config), Ok(()));
}

#[test]
fn directly_assigned_value_next_to_default_is_used_exactly() {
    let mut config = PafiConfig::default();
    config.zero_temperature_tol = config.zero_temperature_tol.next_up();
    assert_eq!(run_at_33_planes(&config), Ok(()));
}
