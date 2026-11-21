//! Physical force tolerance of the OH-TST reactant minimum.

use ndarray::Array1;

use rgflux::FluxError;
use rgflux::ohtst::optimize_hyperplane;
use rgflux::stats::SERIES_FLAT_TOL;

mod ohtst_support;

use ohtst_support::*;

/// Unit in the last place of coordinates in `[4096, 8192)`.
const ULP: f64 = 4096.0 * f64::EPSILON;
/// Float nearest the reactant minimum of [`floor_well`].
const NEAR_MINIMUM: f64 = 4100.0;

/// The quartic well of [`well`] with its barrier top at `4102 + ULP / 2`.
///
/// The minimum `4100 + ULP / 2` lies halfway between two floats, so the
/// smallest force any representable position reaches is `CURVATURE * ULP / 2`,
/// about `9.1e-13`, far above [`SERIES_FLAT_TOL`]. Both subtractions are exact.
fn floor_well() -> Potential {
    let inner = well(HALF_WIDTH);
    Box::new(move |x| inner(Array1::from_vec(vec![(x[0] - 4102.0) - 0.5 * ULP]).view()))
}

fn floor_engine(start: f64) -> Scripted {
    Scripted::new(
        vec![start],
        vec![1.0],
        floor_well(),
        harmonic_force(CURVATURE, NEAR_MINIMUM),
    )
}

fn force_at(x: f64) -> f64 {
    floor_well()(Array1::from_vec(vec![x]).view()).1[0]
}

#[test]
fn the_floor_well_cannot_be_relaxed_below_its_rounding_floor() {
    let floor = 0.5 * CURVATURE * ULP;
    for step in -4_i32..=4 {
        let x = NEAR_MINIMUM + f64::from(step) * ULP;
        let force = force_at(x).abs();
        assert!(
            force >= 0.99 * floor,
            "|F({x})| = {force:e} under the floor {floor:e}"
        );
        assert!(force > 32.0 * SERIES_FLAT_TOL);
    }
    assert_close(force_at(NEAR_MINIMUM).abs(), floor, 1.0e-6, "floor force");
}

#[test]
fn a_reactant_relaxed_to_the_rounding_floor_gives_a_rate() {
    let mut engine = floor_engine(NEAR_MINIMUM);
    let found = optimize_hyperplane(&mut engine, &plane(&[1.0], 4102.0), &quiet())
        .expect("a reactant at fmax 9e-13 is a minimum");
    assert_close(
        found.rate,
        harmonic_rate(CURVATURE, 1.0, BARRIER, THERMAL),
        RELATIVE,
        "rate",
    );
    assert_eq!(
        engine.log.first_step_positions[0][0],
        NEAR_MINIMUM + 0.25,
        "the relaxed reactant is not moved before probing"
    );
    assert_eq!(engine.positions[0], NEAR_MINIMUM);
}

#[test]
fn a_distant_start_relaxes_to_a_tolerance_near_the_rounding_floor() {
    let mut engine = floor_engine(NEAR_MINIMUM - 0.6);
    let config = quiet().with_reactant_fmax(4.0 * CURVATURE * ULP);
    let found = optimize_hyperplane(&mut engine, &plane(&[1.0], 4102.0), &config)
        .expect("rgmin reaches a few units of the floor");
    assert_close(
        found.rate,
        harmonic_rate(CURVATURE, 1.0, BARRIER, THERMAL),
        RELATIVE,
        "rate",
    );
    let probe_start = engine.log.first_step_positions[0][0];
    assert!(
        (probe_start - (NEAR_MINIMUM + 0.25)).abs() <= 4.0 * ULP,
        "probe started at {probe_start}"
    );
    assert_eq!(engine.positions[0], NEAR_MINIMUM - 0.6);
}

#[test]
fn a_tolerance_below_the_rounding_floor_is_refused() {
    let tolerance = 0.1 * CURVATURE * ULP;
    let mut engine = floor_engine(NEAR_MINIMUM);
    let err = optimize_hyperplane(
        &mut engine,
        &plane(&[1.0], 4102.0),
        &quiet().with_reactant_fmax(tolerance),
    )
    .expect_err("no float meets the tolerance");
    match err {
        FluxError::ReactantNotRelaxed {
            fmax,
            tolerance: reported,
        } => {
            assert_eq!(reported, tolerance);
            assert!(fmax > tolerance && fmax.is_finite(), "fmax {fmax:e}");
        }
        other => panic!("unexpected error {other:?}"),
    }
    assert_eq!(engine.positions[0], NEAR_MINIMUM);
    assert!(
        engine.log.step_sizes.is_empty(),
        "no sampling after refusal"
    );
}
