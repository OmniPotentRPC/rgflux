//! Bracketing and bisection of the OH-TST dividing surface along the seed normal.

use ndarray::{Array1, ArrayView1};

use rgflux::ohtst::OhtstEstimate;
use rgflux::stats::SERIES_FLAT_TOL;

mod ohtst_support;

use ohtst_support::*;

fn root_error(found: &OhtstEstimate, root: f64, reach: f64) -> f64 {
    assert!(
        (found.plane.offset - root).abs() <= SERIES_FLAT_TOL * (1.0 + reach),
        "dividing offset {} against root {root}",
        found.plane.offset
    );
    found.plane.offset - root
}

#[test]
fn bracketing_walks_past_a_seed_that_is_not_stationary() {
    // Minimum at -2, barrier top at 0. Steps of 0.8 reach x = -1.2, -0.4 and
    // 0.4, so the third step crosses the top and brackets it.
    for seed_offset in [-1.0, 1.0, 3.0] {
        let mut engine = well_engine(-HALF_WIDTH);
        let config = quiet().with_max_translation(0.8);
        let found = estimate(&mut engine, &plane(&[1.0], seed_offset), &config);
        let miss = root_error(&found, 0.0, 2.4);
        assert!(miss >= 0.0, "the bracket's far end has a non-rising slope");
        assert_close(
            found.rate,
            harmonic_rate(CURVATURE, 1.0, BARRIER, THERMAL),
            RELATIVE,
            "rate",
        );
    }
}

#[test]
fn bracketing_runs_toward_smaller_coordinates_below_the_minimum() {
    for seed_offset in [1.0, -1.0, -3.0] {
        let mut engine = well_engine(HALF_WIDTH);
        let config = quiet().with_max_translation(0.8);
        let found = estimate(&mut engine, &plane(&[1.0], seed_offset), &config);
        let miss = root_error(&found, 0.0, 2.4);
        assert!(miss <= 0.0, "the bracket's far end has a non-rising slope");
        assert_close(
            found.rate,
            harmonic_rate(CURVATURE, 1.0, BARRIER, THERMAL),
            RELATIVE,
            "rate",
        );
    }
}

#[test]
fn bisection_converges_to_the_stationary_point_of_a_shifted_well() {
    // The top of `(u^2 - 4)^2 / 16` moved to 0.3: the dividing surface is not
    // at the origin and not at a multiple of the translation limit.
    let shifted = Box::new(|x: ArrayView1<'_, f64>| {
        let (energy, force) = well(HALF_WIDTH)(Array1::from_vec(vec![x[0] - 0.3]).view());
        (energy, force)
    });
    let mut engine = Scripted::new(
        vec![-1.7],
        vec![1.0],
        shifted,
        harmonic_force(CURVATURE, -1.7),
    );
    let config = quiet().with_max_translation(0.45);
    let found = estimate(&mut engine, &plane(&[1.0], 1.0), &config);
    root_error(&found, 0.3, 2.0);
}

#[test]
fn bracketing_budget_counts_the_outer_steps() {
    // Steps of 0.8 cross the top on the third step.
    let config = quiet().with_max_translation(0.8);
    let mut engine = well_engine(-HALF_WIDTH);
    estimate(
        &mut engine,
        &plane(&[1.0], 1.0),
        &config.clone().with_max_outer_steps(3),
    );
    let mut engine = well_engine(-HALF_WIDTH);
    assert_eq!(
        error(
            &mut engine,
            &plane(&[1.0], 1.0),
            &config.with_max_outer_steps(2)
        ),
        invalid("OH-TST found no stationary dividing surface along the seed normal")
    );
}

#[test]
fn bracketing_is_not_attempted_beyond_a_surface_without_a_top() {
    let mut engine = Scripted::new(
        vec![0.0],
        vec![1.0],
        Box::new(|x| (0.5 * x[0] * x[0], Array1::from_vec(vec![-x[0]]))),
        harmonic_force(1.0, 0.0),
    );
    assert_eq!(
        error(&mut engine, &plane(&[1.0], 1.0), &quiet()),
        invalid("OH-TST found no stationary dividing surface along the seed normal")
    );
}

#[test]
fn a_seed_at_the_minimum_is_not_a_dividing_surface() {
    // The slope vanishes at either minimum and the energy there is not above
    // the minimum, so the search continues toward the barrier.
    for (seed_offset, start) in [(-HALF_WIDTH, -HALF_WIDTH), (HALF_WIDTH, -HALF_WIDTH)] {
        let mut engine = well_engine(start);
        let found = estimate(&mut engine, &plane(&[1.0], seed_offset), &quiet());
        let miss = root_error(&found, 0.0, 2.25);
        assert!(miss >= 0.0);
    }
}

#[test]
fn a_stationary_seed_above_the_minimum_is_the_dividing_surface() {
    let mut engine = well_engine(-HALF_WIDTH);
    let found = estimate(&mut engine, &plane(&[1.0], 0.0), &quiet());
    assert_eq!(found.plane.offset, 0.0);
}

fn ledge(minimum: f64) -> Potential {
    // Flat to the left of `minimum` and rising to its right. The energy to
    // the left is scripted to rise at first and fall back to the minimum
    // value half a unit away, so a seed there is not a dividing surface.
    Box::new(move |x| {
        let right = x[0] - minimum;
        if right >= 0.0 {
            (0.5 * right * right, Array1::from_vec(vec![-right]))
        } else {
            let left = -right;
            (left * (1.0 - 2.0 * left), Array1::from_vec(vec![0.0]))
        }
    })
}

fn flat_edge_bisection(max_translation: f64) -> (f64, OhtstEstimate) {
    let mut engine = Scripted::new(
        vec![1.0],
        vec![1.0],
        ledge(1.0),
        harmonic_force(CURVATURE, 1.0),
    );
    let config = quiet().with_max_translation(max_translation);
    let found = estimate(&mut engine, &plane(&[1.0], 0.5), &config);
    let tolerance = SERIES_FLAT_TOL * (1.0 + max_translation);
    let mut hi = max_translation;
    while hi > tolerance {
        hi *= 0.5;
    }
    (hi, found)
}

#[test]
fn bisection_halves_a_flat_bracket_down_to_the_flatness_tolerance() {
    // Every slope on the left ledge is exactly zero, so each bisection step
    // keeps the near end and the bracket shrinks to the tolerance. Looking
    // toward smaller coordinates, `direction * 0.0` is a positive zero.
    let (hi, found) = flat_edge_bisection(0.25);
    assert_eq!(found.plane.offset, 1.0 - hi);
    assert!(found.plane.offset < 1.0 && 1.0 - found.plane.offset < 1.0e-13);
}

#[test]
fn bisection_stops_when_the_bracket_equals_the_tolerance_exactly() {
    // The tolerance is `SERIES_FLAT_TOL * (1 + 8 w)` for an initial bracket of
    // `8 w`. Take `w` with `w == SERIES_FLAT_TOL * (1 + 8 w)`: the third
    // halving gives a bracket of exactly `w`, which is within tolerance and is
    // not halved again.
    let mut width = SERIES_FLAT_TOL;
    for _ in 0..16 {
        width = SERIES_FLAT_TOL * (1.0 + 8.0 * width);
    }
    assert_eq!(width, SERIES_FLAT_TOL * (1.0 + 8.0 * width));
    let (hi, found) = flat_edge_bisection(8.0 * width);
    assert_eq!(hi, width);
    assert_eq!(found.plane.offset, 1.0 - width);
}

#[test]
fn a_zero_slope_ends_the_bracketing_walk_in_either_direction() {
    // From a minimum on the right edge of the ledge the walk to the left sees
    // zero slope at once; the barrier then has no height above the minimum.
    let mut engine = Scripted::new(
        vec![1.0],
        vec![1.0],
        Box::new(|x| {
            (
                0.0,
                Array1::from_vec(vec![if x[0] > 1.0 { 1.0 - x[0] } else { 0.0 }]),
            )
        }),
        harmonic_force(CURVATURE, 1.0),
    );
    assert_eq!(
        error(&mut engine, &plane(&[1.0], 0.5), &quiet()),
        invalid("OH-TST dividing surface must lie above the reactant minimum")
    );
}

#[test]
fn a_seed_slope_at_the_flatness_tolerance_still_vanishes() {
    let tolerance = SERIES_FLAT_TOL;
    let inner = well(HALF_WIDTH);
    let potential: Potential = Box::new(move |x| {
        let (energy, force) = inner(x);
        if x[0] == 0.5 {
            (energy, Array1::from_vec(vec![-tolerance]))
        } else {
            (energy, force)
        }
    });
    let mut engine = Scripted::new(
        vec![-HALF_WIDTH],
        vec![1.0],
        potential,
        harmonic_force(CURVATURE, -HALF_WIDTH),
    );
    let found = estimate(&mut engine, &plane(&[1.0], 0.5), &quiet());
    assert_eq!(found.plane.offset, 0.5);
}
