//! Physical force tolerance of the OH-TST reactant minimum, caller-supplied
//! minima, and standalone reactant relaxation.

use ndarray::Array1;

use rgflux::FluxError;
use rgflux::ohtst::{
    ReactantMinimum, optimize_hyperplane, optimize_hyperplane_from_minimum, relax_reactant,
};
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

fn supplied(positions: &[f64], fmax: f64) -> ReactantMinimum {
    ReactantMinimum::new(Array1::from_vec(positions.to_vec()), fmax)
}

#[test]
fn a_supplied_minimum_within_tolerance_is_used_as_given() {
    let reactant = supplied(&[NEAR_MINIMUM], force_at(NEAR_MINIMUM).abs());
    let mut engine = floor_engine(NEAR_MINIMUM + 1.0);
    let found =
        optimize_hyperplane_from_minimum(&mut engine, &plane(&[1.0], 4102.0), &reactant, &quiet())
            .expect("a supplied minimum at fmax 9e-13 is accepted");
    assert_close(
        found.rate,
        harmonic_rate(CURVATURE, 1.0, BARRIER, THERMAL),
        RELATIVE,
        "rate",
    );
    assert_eq!(engine.log.first_step_positions[0][0], NEAR_MINIMUM + 0.25);
    assert_eq!(engine.positions[0], NEAR_MINIMUM + 1.0, "engine restored");

    let mut relaxed = floor_engine(NEAR_MINIMUM);
    let direct = optimize_hyperplane(&mut relaxed, &plane(&[1.0], 4102.0), &quiet())
        .expect("direct estimate");
    assert_eq!(found, direct);
}

#[test]
fn a_declared_force_above_the_tolerance_is_refused_before_the_engine_moves() {
    let config = quiet();
    let declared = 2.0 * config.reactant_fmax;
    let mut engine = floor_engine(NEAR_MINIMUM + 1.0);
    let err = optimize_hyperplane_from_minimum(
        &mut engine,
        &plane(&[1.0], 4102.0),
        &supplied(&[NEAR_MINIMUM], declared),
        &config,
    )
    .expect_err("declared force exceeds the tolerance");
    assert_eq!(
        err,
        FluxError::ReactantNotRelaxed {
            fmax: declared,
            tolerance: config.reactant_fmax,
        }
    );
    assert_eq!(engine.positions[0], NEAR_MINIMUM + 1.0);
    assert!(engine.log.step_sizes.is_empty());
}

#[test]
fn a_supplied_minimum_whose_measured_force_exceeds_the_tolerance_is_refused() {
    let config = quiet();
    let unrelaxed = NEAR_MINIMUM - 0.1;
    let mut engine = floor_engine(NEAR_MINIMUM + 1.0);
    let err = optimize_hyperplane_from_minimum(
        &mut engine,
        &plane(&[1.0], 4102.0),
        &supplied(&[unrelaxed], 1.0e-9),
        &config,
    )
    .expect_err("the engine measures a large force");
    assert_eq!(
        err,
        FluxError::ReactantNotRelaxed {
            fmax: force_at(unrelaxed).abs(),
            tolerance: config.reactant_fmax,
        }
    );
    assert_eq!(engine.positions[0], NEAR_MINIMUM + 1.0, "engine restored");
    assert!(engine.log.step_sizes.is_empty());
}

#[test]
fn a_supplied_minimum_must_declare_a_finite_non_negative_force() {
    for bad in [f64::NAN, -1.0e-9, f64::INFINITY, f64::NEG_INFINITY] {
        let mut engine = floor_engine(NEAR_MINIMUM);
        assert_eq!(
            optimize_hyperplane_from_minimum(
                &mut engine,
                &plane(&[1.0], 4102.0),
                &supplied(&[NEAR_MINIMUM], bad),
                &quiet(),
            ),
            Err(invalid(
                "ReactantMinimum::fmax must be finite and non-negative"
            ))
        );
    }
    let mut engine = floor_engine(NEAR_MINIMUM);
    assert!(
        optimize_hyperplane_from_minimum(
            &mut engine,
            &plane(&[1.0], 4102.0),
            &supplied(&[NEAR_MINIMUM], 0.0),
            &quiet(),
        )
        .is_ok(),
        "a declared zero is checked against the measured force"
    );
}

#[test]
fn a_supplied_minimum_must_match_the_engine_and_be_finite() {
    let mut engine = floor_engine(NEAR_MINIMUM);
    assert_eq!(
        optimize_hyperplane_from_minimum(
            &mut engine,
            &plane(&[1.0], 4102.0),
            &supplied(&[NEAR_MINIMUM, 0.0], 0.0),
            &quiet(),
        ),
        Err(FluxError::Shape {
            expected: 1,
            got: 2,
            context: "OH-TST reactant minimum",
        })
    );
    assert_eq!(
        optimize_hyperplane_from_minimum(
            &mut engine,
            &plane(&[1.0], 4102.0),
            &supplied(&[f64::NAN], 0.0),
            &quiet(),
        ),
        Err(FluxError::NonFinite("OH-TST reactant minimum positions"))
    );
}

#[test]
fn a_supplied_maximum_along_the_normal_is_refused() {
    // The barrier top of the double well has zero force.
    let mut engine = well_engine(-HALF_WIDTH);
    assert_eq!(
        optimize_hyperplane_from_minimum(
            &mut engine,
            &plane(&[1.0], 1.0),
            &supplied(&[0.0], 0.0),
            &quiet(),
        ),
        Err(invalid(
            "OH-TST reactant minimum is not a minimum along the seed normal"
        ))
    );
    assert_eq!(engine.positions[0], -HALF_WIDTH);
}

#[test]
fn an_unusable_force_tolerance_is_refused_with_a_supplied_minimum() {
    let message = "OhtstConfig::reactant_fmax must be finite and positive";
    for bad in [0.0, -1.0e-6, f64::NAN, f64::INFINITY] {
        let config = quiet().with_reactant_fmax(bad);
        let mut engine = floor_engine(NEAR_MINIMUM);
        assert_eq!(
            optimize_hyperplane_from_minimum(
                &mut engine,
                &plane(&[1.0], 4102.0),
                &supplied(&[NEAR_MINIMUM], 0.0),
                &config,
            ),
            Err(invalid(message))
        );
        assert_eq!(relax_reactant(&mut engine, &config), Err(invalid(message)));
    }
}

#[test]
fn relax_reactant_returns_a_minimum_within_tolerance_and_restores_the_engine() {
    let config = quiet().with_reactant_fmax(4.0 * CURVATURE * ULP);
    let start = NEAR_MINIMUM - 0.6;
    let mut engine = floor_engine(start);
    let reactant = relax_reactant(&mut engine, &config).expect("relaxed");
    assert!(reactant.fmax <= config.reactant_fmax, "{:e}", reactant.fmax);
    assert_eq!(reactant.fmax, force_at(reactant.positions[0]).abs());
    assert!((reactant.positions[0] - NEAR_MINIMUM).abs() <= 4.0 * ULP);
    assert_eq!(engine.positions[0], start);
    assert!(engine.log.step_sizes.is_empty());

    let found =
        optimize_hyperplane_from_minimum(&mut engine, &plane(&[1.0], 4102.0), &reactant, &config)
            .expect("rate from the relaxed reactant");
    let direct =
        optimize_hyperplane(&mut engine, &plane(&[1.0], 4102.0), &config).expect("direct rate");
    assert_eq!(found, direct);
}

#[test]
fn relax_reactant_leaves_a_relaxed_start_in_place() {
    let mut engine = floor_engine(NEAR_MINIMUM);
    let reactant = relax_reactant(&mut engine, &quiet()).expect("relaxed");
    assert_eq!(reactant.positions[0], NEAR_MINIMUM);
    assert_eq!(reactant.fmax, force_at(NEAR_MINIMUM).abs());
}

#[test]
fn relax_reactant_refuses_a_force_that_never_relaxes() {
    let mut engine = Scripted::new(
        vec![0.0],
        vec![1.0],
        Box::new(|_| (0.0, Array1::from_vec(vec![1.0]))),
        harmonic_force(1.0, 0.0),
    );
    assert_eq!(
        relax_reactant(&mut engine, &quiet()),
        Err(FluxError::ReactantNotRelaxed {
            fmax: 1.0,
            tolerance: quiet().reactant_fmax,
        })
    );
    assert_eq!(engine.positions[0], 0.0);
}

#[test]
fn one_relaxed_reactant_serves_every_exit_of_its_basin() {
    // Two independent double wells: the reactant at (-2, -2) has one exit
    // over each barrier. The scripted constraint force is harmonic about -2
    // along whichever unit axis the probe plane holds.
    let inner = well(HALF_WIDTH);
    let potential: Potential = Box::new(move |x| {
        let (ex, fx) = inner(Array1::from_vec(vec![x[0]]).view());
        let (ey, fy) = inner(Array1::from_vec(vec![x[1]]).view());
        (ex + ey, Array1::from_vec(vec![fx[0], fy[0]]))
    });
    let mut engine = Scripted::new(
        vec![-1.3 * HALF_WIDTH, -0.8 * HALF_WIDTH],
        vec![1.0, 1.0],
        potential,
        harmonic_force(CURVATURE, -HALF_WIDTH),
    );
    let config = analytic();
    let reactant = relax_reactant(&mut engine, &config).expect("relaxed once");
    for exit in [[1.0, 0.0], [0.0, 1.0]] {
        let found =
            optimize_hyperplane_from_minimum(&mut engine, &plane(&exit, 0.0), &reactant, &config)
                .expect("rate on this exit");
        assert_close(
            found.rate,
            harmonic_rate(CURVATURE, 1.0, BARRIER, THERMAL),
            RELATIVE,
            "rate",
        );
    }
}
