//! Cross-engine harness and compatibility checks.

use ndarray::{Array1, ArrayView1};
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::engine::{DynamicsEngine, ThermostatConfig};

#[cfg(feature = "rgpot-session")]
use rgflux::FluxError;

struct QuarticDoubleWell1D;

impl rgsaddle::PointSurface for QuarticDoubleWell1D {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
        let dx = x[0] * x[0] - 1.0;
        let energy = dx * dx;
        let grad = Array1::from_vec(vec![4.0 * x[0] * dx]);
        Ok((energy, grad))
    }
}

#[test]
fn native_engine_cross_harness_reports_finite_observables() {
    let cfg = NativeConfig::default()
        .with_masses(Array1::from_vec(vec![1.0]))
        .with_thermostat(
            ThermostatConfig::default()
                .with_gamma(1.5)
                .with_temperature(0.5),
        )
        .with_seed(77);
    let mut engine = NativeEngine::new(QuarticDoubleWell1D, cfg).expect("native engine");
    engine
        .set_positions(Array1::from_vec(vec![-1.0]).view())
        .expect("set state");
    engine.shoot_velocities(77).expect("seeded shot");

    let mut means = Vec::with_capacity(2_000);
    for _ in 0..2_000 {
        let report = engine.step(0.01).expect("step");
        means.push(report.potential_energy + report.kinetic_energy);
    }

    let mean = means.iter().sum::<f64>() / (means.len() as f64);
    assert!(mean.is_finite());
}

#[cfg(feature = "rgpot-session")]
#[test]
fn session_engine_contract_is_compatible_with_cross_harness_glue() {
    use rgflux::engine::rgpot_session::{
        RgpotSessionConfig, RgpotSessionEngine, SessionEngineKind,
    };

    let cfg = RgpotSessionConfig::default()
        .with_engine(SessionEngineKind::OpenMm)
        .with_masses(Array1::from_vec(vec![12.0, 12.0, 12.0]));
    let mut engine = RgpotSessionEngine::new(cfg).expect("session config");
    engine.reseed(123).expect("reseed");

    match engine.step(0.001) {
        Ok(rep) => {
            assert!(rep.potential_energy.is_finite());
            assert!(rep.kinetic_energy.is_finite());
        }
        Err(FluxError::BackendUnavailable(_)) => {}
        Err(err) => panic!("unexpected session step error: {err:?}"),
    }
}
