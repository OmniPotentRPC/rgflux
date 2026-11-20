//! Reactant certification at and above the dense-Hessian dimension limit.

use ndarray::{Array1, Array2, ArrayView1};
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::engine::{DynamicsEngine, ThermostatConfig};
use rgflux::pafi::{PafiConfig, pafi_profile};
use rgflux::stats::StatsConfig;

struct SeparableHarmonic;

impl rgsaddle::PointSurface for SeparableHarmonic {
    fn eval(
        &self,
        point: ArrayView1<'_, f64>,
    ) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
        Ok((0.5 * point.dot(&point), point.to_owned()))
    }
}

fn sampled_profile(dof: usize) {
    let masses = Array1::from_elem(dof, 1.0);
    let mut engine = NativeEngine::new(
        SeparableHarmonic,
        NativeConfig::default()
            .with_masses(masses)
            .with_seed(19)
            .with_thermostat(
                ThermostatConfig::default()
                    .with_temperature(0.2)
                    .with_boltzmann(1.0),
            ),
    )
    .expect("native engine");
    engine.shoot_velocities(19).expect("initial velocities");

    let mut path = Array2::zeros((2, dof));
    path[[1, 0]] = 0.5;
    let config = PafiConfig::default()
        .with_planes(2)
        .with_burn_in_steps(1)
        .with_sample_steps(2)
        .with_time_step(0.01)
        .with_stats(StatsConfig::default().with_min_batches(2));

    let profile = pafi_profile(&mut engine, path.view(), &config).expect("sampled profile");
    assert_eq!(profile.free_energy.len(), 2);
    assert!(profile.barrier.is_finite());
}

#[test]
fn pafi_samples_above_dense_hessian_limit() {
    sampled_profile(150);
}

#[test]
fn dense_hessian_limit_boundary_accepts_128_and_129_dof() {
    sampled_profile(128);
    sampled_profile(129);
}
