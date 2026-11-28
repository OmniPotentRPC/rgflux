//! Fast, potential-free part of the Swinburne reference-code replay.
//!
//! `tomswinburne/pafi` commit 1661a1d61b2ccbbd33ec4c9656d60947b8b74cc1
//! pins these nine Fe-SIA plane gradients in `tests/test_pafi.py`. The test
//! applies that repository's cubic-remesh integration without embedding a
//! second EAM implementation in rgsaddle. The ignored full test evaluates
//! the bundled W EAM through rgpot and lammpc.

use ndarray::{Array1, s};
use rgflux::pafi::cartesian::integrate_profile;
use rgsaddle::{ArcPath, PathFrame};

const FE_GRADIENT: [f64; 9] = [
    0.0, 0.614145, 0.846192, 0.59226, 0.089912, -0.426347, -0.853405, -0.792713, 0.0,
];

fn planes(count: usize) -> Vec<f64> {
    (0..count)
        .map(|index| index as f64 / (count - 1) as f64)
        .collect()
}

fn reference_remesh(arcs: &[f64], gradients: &[f64]) -> (f64, f64) {
    let count = arcs.len();
    let basis = ArcPath::new(
        arcs.iter()
            .enumerate()
            .map(|(index, arc)| {
                let mut positions = Array1::zeros(count + 1);
                positions[0] = *arc;
                positions[index + 1] = 1.0;
                PathFrame {
                    positions,
                    arc: *arc,
                }
            })
            .collect(),
    )
    .expect("remesh basis");
    let dense = 10 * count;
    let points: Vec<f64> = (0..dense)
        .map(|index| index as f64 / (dense - 1) as f64)
        .collect();
    let weights: Vec<Array1<f64>> = points
        .iter()
        .map(|point| {
            basis
                .geometry(*point)
                .expect("remesh point")
                .positions
                .slice(s![1..])
                .to_owned()
        })
        .collect();
    let mut cumulative = vec![Array1::<f64>::zeros(count)];
    for index in 1..dense {
        let width = 0.5 * (points[index] - points[index - 1]);
        cumulative
            .push(&cumulative[index - 1] + &((&weights[index - 1] + &weights[index]) * width));
    }
    let gradient = Array1::from_vec(gradients.to_vec());
    let barrier = cumulative
        .iter()
        .map(|weight| weight.dot(&gradient))
        .fold(f64::NEG_INFINITY, f64::max);
    let end = cumulative
        .last()
        .expect("last cumulative weight")
        .dot(&gradient);
    (barrier, end)
}

#[test]
fn fe_reference_gradients_match_the_reference_integral() {
    let arcs = planes(FE_GRADIENT.len());
    // Exercise rgsaddle's ordinary profile validation before applying the
    // public code's cubic-remesh postprocessor.
    integrate_profile(&arcs, &FE_GRADIENT, &[0.0; FE_GRADIENT.len()]).expect("validated profile");
    let (barrier, end) = reference_remesh(&arcs, &FE_GRADIENT);
    let roundoff = 4096.0 * f64::EPSILON;
    assert!(
        (barrier - 0.276_228_762_296_550_97).abs() <= roundoff,
        "{barrier}"
    );
    assert!((end - 0.003_824_869_324_197_502).abs() <= roundoff, "{end}");
}
