//! Direct Forward Flux Sampling tests.

use ndarray::{Array1, ArrayView1};
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::engine::{DynamicsEngine, ThermostatConfig};
use rgflux::ffs::{FfsConfig, forward_flux_rate};

struct QuarticDoubleWell1D;

impl rgsaddle::PointSurface for QuarticDoubleWell1D {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
        let dx = x[0] * x[0] - 1.0;
        let energy = dx * dx;
        let grad = Array1::from_vec(vec![4.0 * x[0] * dx]);
        Ok((energy, grad))
    }
}

fn make_engine(seed: u64) -> NativeEngine<QuarticDoubleWell1D> {
    let cfg = NativeConfig::default()
        .with_masses(Array1::from_vec(vec![1.0]))
        .with_thermostat(
            ThermostatConfig::default()
                .with_gamma(2.0)
                .with_temperature(0.9),
        )
        .with_seed(seed);
    let mut engine = NativeEngine::new(QuarticDoubleWell1D, cfg).expect("engine");
    engine
        .set_positions(Array1::from_vec(vec![-1.0]).view())
        .expect("start in basin A");
    engine.shoot_velocities(seed).expect("shoot");
    engine
}

#[test]
fn forward_flux_sampling_returns_finite_rate_on_small_double_well() {
    let mut engine = make_engine(11);
    let cfg = FfsConfig::default()
        .with_interfaces(vec![-0.8, -0.4, -0.1, 0.2])
        .with_points_per_interface(8)
        .with_trials_per_interface(64)
        .with_max_trial_steps(2_000)
        .with_max_flux_steps(500_000)
        .with_time_step(0.01)
        .with_seed(11);
    let estimate = forward_flux_rate(&mut engine, |x| x[0], &cfg).expect("ffs estimate");
    assert!(estimate.initial_flux > 0.0);
    assert!(estimate.rate > 0.0);
    assert!(estimate.standard_error.is_finite());
    assert_eq!(estimate.interface_probabilities.len(), 3);
    assert!(estimate.interface_probabilities.iter().all(|p| *p > 0.0));
}

#[test]
fn forward_flux_sampling_is_reproducible_for_identical_seed() {
    let cfg = FfsConfig::default()
        .with_interfaces(vec![-0.8, -0.4, -0.1, 0.2])
        .with_points_per_interface(8)
        .with_trials_per_interface(48)
        .with_max_trial_steps(1_500)
        .with_max_flux_steps(500_000)
        .with_time_step(0.01)
        .with_seed(99);
    let mut a = make_engine(99);
    let mut b = make_engine(99);

    let ea = forward_flux_rate(&mut a, |x| x[0], &cfg).expect("estimate a");
    let eb = forward_flux_rate(&mut b, |x| x[0], &cfg).expect("estimate b");
    assert_eq!(ea.initial_flux.to_bits(), eb.initial_flux.to_bits());
    assert_eq!(ea.rate.to_bits(), eb.rate.to_bits());
    assert_eq!(ea.standard_error.to_bits(), eb.standard_error.to_bits());
    assert_eq!(
        ea.interface_probabilities
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        eb.interface_probabilities
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn invalid_ffs_configuration_is_rejected() {
    let mut engine = make_engine(7);
    let cfg = FfsConfig::default().with_interfaces(vec![0.1, 0.1]);
    let err = forward_flux_rate(&mut engine, |x| x[0], &cfg).expect_err("invalid interfaces");
    assert!(format!("{err}").contains("interfaces"));
}
