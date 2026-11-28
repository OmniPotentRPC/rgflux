use ndarray::{Array1, ArrayView1, array};
use rgflux::pafi::cartesian::{PafiConfig, average_hyperplane};
use rgsaddle::{ArcPath, PointSurface, SaddleError};

struct CoupledQuadratic;

impl PointSurface for CoupledQuadratic {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let energy = 0.5 * (2.0 * x[0] * x[0] + x[1] * x[1]) + 0.25 * x[0] * x[1];
        Ok((energy, array![2.0 * x[0] + 0.25 * x[1], x[1] + 0.25 * x[0]]))
    }
}

fn bits(values: impl IntoIterator<Item = f64>) -> String {
    values
        .into_iter()
        .map(|value| format!("{:016x}", value.to_bits()))
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn projector_outputs_match_the_pre_unification_dump() {
    let path =
        ArcPath::from_positions(&[array![-0.8, -0.3], array![0.1, 0.4], array![0.9, 0.2]]).unwrap();
    let geometry = path.geometry(0.73).unwrap();
    let config = PafiConfig::default()
        .with_thermal(0.07)
        .with_mobility(0.8)
        .with_time_step(0.003)
        .with_thermalize(7)
        .with_stride(2)
        .with_samples(12)
        .with_blocks(3)
        .with_seed(0x5eed)
        .with_fix_com(false)
        .with_retain(4)
        .with_target_error(None)
        .with_max_rounds(1);
    let average = average_hyperplane(&CoupledQuadratic, &geometry, &config).unwrap();
    let perpendicular = rgsaddle::stencil::perp(
        array![1.25, -2.5, 0.75, 4.0].view(),
        array![0.5, -0.5, 0.5, -0.5].view(),
    );
    let mut dump = vec![
        average.gradient,
        average.error,
        average.psi,
        average.tau,
        average.n_eff,
    ];
    dump.extend(perpendicular);
    for point in average.retained {
        dump.extend(point);
    }
    let actual = bits(dump);
    let expected = include_str!("data/projector-before.bits").trim();
    if expected == "GENERATE" {
        panic!("pre-unification dump: {actual}");
    }
    assert_eq!(actual, expected);
}
