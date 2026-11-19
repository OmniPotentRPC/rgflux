use ndarray::{Array1, Array2, ArrayView1};
use rgflux::FluxError;
use rgflux::engine::ThermostatConfig;
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::pafi::{PafiConfig, pafi_profile};
use rgflux::stats::StatsConfig;

struct InconsistentLinearGradient;

impl rgsaddle::PointSurface for InconsistentLinearGradient {
    fn eval(&self, x: ArrayView1<'_, f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
        // The energy rises by 1. A constant gradient of 0.98 leaves a
        // discrepancy of 0.02. Adding that same shortfall as a slope
        // recenters the gradient so the trapezoid integral is still 0.98,
        // and it keeps the reactant curvature positive once the check passes.
        let shortfall = 0.02;
        let gradient = (1.0 - shortfall) + shortfall * (x[0] - 0.5);
        Ok((x[0], Array1::from_vec(vec![gradient])))
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
