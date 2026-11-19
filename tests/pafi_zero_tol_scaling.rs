use ndarray::{Array1, Array2, ArrayView1};
use rgflux::FluxError;
use rgflux::engine::ThermostatConfig;
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::pafi::{PafiConfig, pafi_profile};
use rgflux::stats::StatsConfig;

struct InconsistentQuadraticGradient;

impl rgsaddle::PointSurface for InconsistentQuadraticGradient {
    fn eval(&self, x: ArrayView1<'_, f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
        // `V(x) = x^2 / 2` is a minimum at the reactant `x = 0` and rises by
        // 1/2 on `[0, 1]`. The reported gradient matches `V'` at 0 and adds
        // `QUADRATURE_SHORTFALL * 6 * x * (x - 1)` in integrated form:
        // `g(x) = (1 - 6 s) x + 6 s x^2` with `s = QUADRATURE_SHORTFALL`, and
        // `∫_0^1 g = 1/2 - s`. The trapezoid of that quadratic on 33 planes
        // stays near the shortfall, above the scaled default tolerance and
        // below the explicit tolerance 0.0257.
        const QUADRATURE_SHORTFALL: f64 = 0.02;
        let bend = QUADRATURE_SHORTFALL * 6.0;
        let reported = (1.0 - bend) * x[0] + bend * x[0] * x[0];
        Ok((0.5 * x[0] * x[0], Array1::from_vec(vec![reported])))
    }
}

#[test]
fn default_tolerance_scales_with_planes_for_consistency_check() {
    let path = Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).expect("shape");
    let mut engine = NativeEngine::new(
        InconsistentQuadraticGradient,
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
        InconsistentQuadraticGradient,
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
