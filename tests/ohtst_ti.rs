//! Lambda thermodynamic integration against quadrature references.

use ndarray::array;
use rgflux::ohtst::{
    AnalyticThermodynamicSampler, HarmonicReference, Hyperplane, LangevinConfig,
    ThermodynamicSurfaceSampler, harmonic_energy_at,
};
use rgsaddle::{
    PointSurface, SaddleError, THERMODYNAMIC_DERIVATIVE, ThermodynamicIntegrationConfig,
    thermodynamic_integration,
};

include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/common/ohtst_surfaces.rs"
));

use ohtst_surfaces::{SkewedWell, anharmonic_line_free_energy};

#[test]
fn quartic_line_ti_matches_partition_quadrature() {
    struct Harmonic;

    impl PointSurface for Harmonic {
        fn eval(
            &self,
            q: ndarray::ArrayView1<f64>,
        ) -> Result<(f64, ndarray::Array1<f64>), SaddleError> {
            Ok((0.5 * q[1] * q[1], array![0.0, q[1]]))
        }
    }

    struct Quartic;

    impl PointSurface for Quartic {
        fn eval(
            &self,
            q: ndarray::ArrayView1<f64>,
        ) -> Result<(f64, ndarray::Array1<f64>), SaddleError> {
            Ok((0.25 * q[1].powi(4), array![0.0, q[1].powi(3)]))
        }
    }

    let thermal = 0.5;
    let dynamics = LangevinConfig::default()
        .with_thermal(thermal)
        .with_time_step(0.03)
        .with_friction(2.0);
    let mut sampler = AnalyticThermodynamicSampler::new(&Harmonic, &Quartic, dynamics);
    let plane = Hyperplane::new(array![0.0, 0.0], array![1.0, 0.0]).unwrap();
    let config = ThermodynamicIntegrationConfig::default()
        .with_thermal(thermal)
        .with_nodes(6)
        .with_thermalize(500)
        .with_stride(8)
        .with_samples(4_000)
        .with_blocks(16)
        .with_seed(41)
        .with_observable(THERMODYNAMIC_DERIVATIVE);
    let estimate = thermodynamic_integration(&mut sampler, &plane, &config).unwrap();
    let full = |y: f64| 0.25 * y.powi(4);
    let harmonic = |y: f64| 0.5 * y * y;
    let expected = anharmonic_line_free_energy(&full, &harmonic, thermal, 8.0);
    assert!(
        (estimate.free_energy - expected).abs() < 4.0 * estimate.error + 0.01,
        "{} ± {} against {expected}",
        estimate.free_energy,
        estimate.error
    );
}

#[test]
fn skewed_well_line_ti_matches_quadrature() {
    let thermal = 0.15;
    let reactant = array![-1.0, -1.0];
    let reference = HarmonicReference::new(reactant.clone());
    let dynamics = LangevinConfig::default()
        .with_thermal(thermal)
        .with_time_step(0.05)
        .with_friction(1.5)
        .with_burn_in(400);
    let mut sampler = ThermodynamicSurfaceSampler::new(&SkewedWell, reference, dynamics).unwrap();
    let plane = Hyperplane::new(array![0.0, 0.0], array![1.0, 0.0]).unwrap();
    let config = ThermodynamicIntegrationConfig::default()
        .with_thermal(thermal)
        .with_nodes(5)
        .with_thermalize(600)
        .with_stride(6)
        .with_samples(3_000)
        .with_blocks(12)
        .with_seed(17);
    let estimate = thermodynamic_integration(&mut sampler, &plane, &config).unwrap();
    let full = |y: f64| SkewedWell::energy(0.0, y).0;
    let harmonic = |y: f64| {
        let mut evaluations = 0;
        harmonic_energy_at(
            &SkewedWell,
            &HarmonicReference::new(reactant.clone()),
            array![0.0, y].view(),
            &mut evaluations,
        )
        .unwrap()
    };
    let expected = anharmonic_line_free_energy(&full, &harmonic, thermal, 6.0);
    assert!(
        (estimate.free_energy - expected).abs() < 5.0 * estimate.error + 0.02,
        "{} ± {} against {expected}",
        estimate.free_energy,
        estimate.error
    );
}
