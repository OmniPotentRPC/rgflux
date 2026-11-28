//! Stored PAFI values at atom masses 1.008 and 15.999.
//!
//! The figures were read from the metric maps before those maps required
//! equal lengths. Each stored value is compared at a relative tolerance of
//! 1e-9 by the same rule as the mass-metric golden values of rgsaddle.

use std::collections::BTreeMap;

use ndarray::{Array1, ArrayView1, array};
use rgflux::pafi::cartesian::{PafiConfig, pafi_profile};
use rgsaddle::{ArcPath, MassWeighted, PointSurface, SaddleError};

const LIGHT: f64 = 1.008;
const HEAVY: f64 = 15.999;

fn energy_gradient(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let x0 = x[0];
    let well = 0.25 * (x0 * x0 - 1.0).powi(2);
    let mut transverse = 0.0;
    let mut gradient = Array1::zeros(x.len());
    gradient[0] = x0 * x0 * x0 - x0;
    for index in 1..x.len() {
        transverse += 0.5 * x[index] * x[index];
        gradient[index] = x[index];
    }
    (well + transverse, gradient)
}

struct SaddleWell;

impl PointSurface for SaddleWell {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        Ok(energy_gradient(x))
    }
}

fn expect_close(stored: &BTreeMap<&'static str, Vec<f64>>, name: &str, got: &[f64]) {
    let expected = stored
        .get(name)
        .unwrap_or_else(|| panic!("missing stored {name}"));
    assert_eq!(got.len(), expected.len(), "{name}");
    for (index, (got, expected)) in got.iter().zip(expected).enumerate() {
        let gap = (got - expected).abs();
        let tolerance = 1.0e-9 * expected.abs();
        assert!(
            gap <= tolerance,
            "{name}[{index}] got {got:.17e} stored {expected:.17e}"
        );
    }
}

fn pafi_values() -> Result<Vec<f64>, SaddleError> {
    let surface = MassWeighted::per_atom(&SaddleWell, &[LIGHT, HEAVY])?;
    let frames = vec![Array1::zeros(6), array![0.4, 0.0, 0.0, 0.0, 0.0, 0.0]];
    let path = ArcPath::from_positions(&frames)?;
    let arcs = path.arcs();
    let config = PafiConfig::default()
        .with_thermal(0.05)
        .with_mobility(1.0)
        .with_time_step(0.01)
        .with_thermalize(2)
        .with_stride(1)
        .with_samples(4)
        .with_blocks(2)
        .with_seed(5)
        .with_fix_com(false)
        .with_retain(1)
        .with_target_error(None)
        .with_max_rounds(1)
        .with_max_jump(None);
    let (profile, planes) = pafi_profile(&surface, &path, &arcs, &config)?;
    let mut out = Vec::new();
    out.push(profile.barrier);
    out.push(profile.barrier_error);
    out.extend(profile.free_energy.iter().copied());
    out.extend(profile.gradient.iter().copied());
    for plane in &planes {
        out.push(plane.gradient);
        out.push(plane.error);
        out.push(plane.psi);
    }
    Ok(out)
}

#[test]
fn pafi_matches_stored_values() {
    let values: Vec<f64> = include_str!("data/pafi_golden.txt")
        .lines()
        .map(|line| line.parse().expect("float"))
        .collect();
    let stored = BTreeMap::from([("pafi", values)]);
    expect_close(&stored, "pafi", &pafi_values().expect("PAFI"));
}
