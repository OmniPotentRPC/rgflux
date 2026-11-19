//! Selectable PAFI quadrature rules on an analytic one-dimensional barrier.

use ndarray::{Array1, Array2, ArrayView1};
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::engine::{DynamicsEngine, ThermostatConfig};
use rgflux::pafi::{PafiConfig, PafiIntegration, pafi_profile};
use rgflux::stats::StatsConfig;

struct QuarticBarrier;

impl rgsaddle::PointSurface for QuarticBarrier {
    fn eval(
        &self,
        point: ArrayView1<'_, f64>,
    ) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
        let x = point[0];
        Ok((
            (1.0 - x * x).powi(2),
            Array1::from_vec(vec![-4.0 * x * (1.0 - x * x)]),
        ))
    }
}

fn profile(rule: PafiIntegration) -> rgflux::pafi::FreeEnergyProfile {
    let path =
        Array2::from_shape_vec((5, 1), vec![-1.0, -0.75, -0.5, -0.25, 0.0]).expect("path shape");
    let config = PafiConfig::default()
        .with_planes(9)
        .with_integration(rule)
        .with_burn_in_steps(2)
        .with_sample_steps(8)
        .with_time_step(0.001)
        .with_zero_temperature_tol(f64::MAX)
        .with_stats(StatsConfig::default().with_min_batches(2));
    let mut engine = NativeEngine::new(
        QuarticBarrier,
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0]))
            .with_thermostat(
                ThermostatConfig::default()
                    .with_temperature(0.01)
                    .with_boltzmann(1.0),
            ),
    )
    .expect("engine");
    engine.shoot_velocities(1).expect("velocities");
    pafi_profile(&mut engine, path.view(), &config).expect("profile")
}

#[test]
fn selectable_rules_change_plane_placement_and_return_finite_profiles() {
    let hermite = profile(PafiIntegration::Hermite);
    let legendre = profile(PafiIntegration::GaussLegendre);
    let jacobi = profile(PafiIntegration::GaussJacobi);
    let clenshaw_curtis = profile(PafiIntegration::ClenshawCurtis);
    let bayesian = profile(PafiIntegration::Bayesian);

    assert_ne!(legendre.arc_lengths, hermite.arc_lengths);
    assert_ne!(jacobi.arc_lengths, legendre.arc_lengths);
    assert_eq!(clenshaw_curtis.arc_lengths[0], 0.0);
    assert_eq!(
        *clenshaw_curtis.arc_lengths.last().expect("last plane"),
        1.0
    );
    assert_ne!(bayesian.arc_lengths, hermite.arc_lengths);
    for candidate in [&legendre, &jacobi, &clenshaw_curtis, &bayesian] {
        assert!(candidate.barrier.is_finite());
        assert!(candidate.barrier_error.is_finite());
        assert!(candidate.normalized_rate.is_finite());
    }
}
