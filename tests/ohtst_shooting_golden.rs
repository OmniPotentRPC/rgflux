//! Stored shooting values at atom masses 1.008 and 15.999.
//!
//! The figures were read from the metric maps before those maps required
//! equal lengths. Each stored value is compared at a relative tolerance of
//! 1e-9 by the same rule as the mass-metric golden values of rgsaddle.

use std::collections::BTreeMap;

use ndarray::{Array1, ArrayView1, array};
use rgflux::ohtst::{AnalyticSurfaceDynamics, TransmissionConfig, transmission_coefficient};
use rgsaddle::{Hyperplane, PointSurface, SaddleError};

const LIGHT: f64 = 1.008;
const HEAVY: f64 = 15.999;

fn coordinate_masses() -> Array1<f64> {
    Array1::from(vec![LIGHT, LIGHT, LIGHT, HEAVY, HEAVY, HEAVY])
}

struct Bowl;

impl PointSurface for Bowl {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let bond = x[0] - x[3];
        let center = x[0] + x[3];
        let mut energy = 0.5 * bond * bond + 0.15 * center * center;
        let mut gradient = Array1::zeros(x.len());
        gradient[0] = bond + 0.3 * center;
        gradient[3] = -bond + 0.3 * center;
        for index in [1, 2, 4, 5] {
            energy += 0.25 * x[index] * x[index];
            gradient[index] = 0.5 * x[index];
        }
        Ok((energy, gradient))
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

fn shooting() -> Result<Vec<f64>, SaddleError> {
    let plane = Hyperplane::new(
        array![0.12, 0.01, 0.0, 0.05, 0.0, -0.01],
        array![0.55, 0.0, 0.0, 0.35, 0.0, 0.0],
    )?;
    let start = array![0.12, 0.01, 0.0, 0.05, 0.0, -0.01];
    let config = TransmissionConfig::default()
        .with_thermal(0.4)
        .with_friction(0.05)
        .with_time_step(0.1)
        .with_steps(40)
        .with_check_every(10)
        .with_shots(8)
        .with_blocks(2)
        .with_seed(11)
        .with_masses(coordinate_masses().to_vec())
        .with_hold_excluded(false);
    let mut dynamics = AnalyticSurfaceDynamics::new(&Bowl);
    let coefficient = transmission_coefficient(&mut dynamics, &plane, &[start], &config)?;
    let mut out = vec![
        coefficient.kappa,
        coefficient.error,
        coefficient.shots as f64,
        coefficient.unassigned as f64,
        coefficient.reactive as f64,
        coefficient.recrossed as f64,
        coefficient.evaluations as f64,
    ];
    out.extend(coefficient.kappa_of_t.iter().copied());
    Ok(out)
}

#[test]
fn ohtst_shooting_matches_stored_values() {
    let values: Vec<f64> = include_str!("data/ohtst_shooting_golden.txt")
        .lines()
        .map(|line| line.parse().expect("float"))
        .collect();
    let stored = BTreeMap::from([("shooting", values)]);
    expect_close(&stored, "shooting", &shooting().expect("shooting"));
}
