//! Reactant minimization and the error paths of the fixed-normal OH-TST rate.

use ndarray::Array1;

use rgflux::FluxError;
use rgflux::engine::DynamicsEngine;
use rgflux::stats::SERIES_FLAT_TOL;

mod ohtst_support;

use ohtst_support::*;

fn plateau(top: f64) -> Potential {
    Box::new(move |x| {
        let reached = x[0].min(top);
        let force = if x[0] < top { 1.0 } else { 0.0 };
        (-reached, Array1::from_vec(vec![force]))
    })
}

fn plateau_engine(top: f64) -> Scripted {
    Scripted::new(vec![0.0], vec![1.0], plateau(top), harmonic_force(1.0, 0.0))
}

#[test]
fn minimization_budget_is_eight_updates_per_outer_step() {
    let config = quiet().with_max_outer_steps(1);
    // Seven unit descents and the stationary check use all eight iterations.
    let err = error(&mut plateau_engine(7.0), &plane(&[1.0], 7.0), &config);
    assert_eq!(
        err,
        invalid("OH-TST dividing surface must lie above the reactant minimum")
    );
    let err = error(&mut plateau_engine(8.0), &plane(&[1.0], 8.0), &config);
    assert_eq!(
        err,
        invalid("OH-TST reactant minimization exceeded its iteration budget")
    );
    let err = error(
        &mut plateau_engine(8.0),
        &plane(&[1.0], 8.0),
        &config.with_max_outer_steps(2),
    );
    assert_eq!(
        err,
        invalid("OH-TST dividing surface must lie above the reactant minimum")
    );
}

#[test]
fn a_force_that_never_relaxes_is_refused_with_its_size() {
    let mut engine = Scripted::new(
        vec![0.0],
        vec![1.0],
        Box::new(|_| (0.0, Array1::from_vec(vec![1.0]))),
        harmonic_force(1.0, 0.0),
    );
    assert_eq!(
        error(&mut engine, &plane(&[1.0], 0.0), &quiet()),
        FluxError::ReactantNotRelaxed {
            fmax: 1.0,
            tolerance: quiet().reactant_fmax,
        }
    );
    assert_eq!(
        engine.positions[0], 0.0,
        "the failed search restores the start"
    );
}

#[test]
fn minimization_backtracks_past_an_overshooting_step() {
    let mut engine = well_engine(-1.3 * HALF_WIDTH);
    let found = estimate(&mut engine, &plane(&[1.0], 0.0), &analytic());
    assert_close(
        found.rate,
        harmonic_rate(CURVATURE, 1.0, BARRIER, THERMAL),
        RELATIVE,
        "rate",
    );
    let first = engine.log.step_offsets[0];
    assert_close(first, -HALF_WIDTH + 0.25, RELATIVE, "probe offset");
}

#[test]
fn minimization_leaves_a_maximum_toward_smaller_coordinates() {
    // Started on the barrier top the search steps back along the normal and
    // settles in the well at -2, so the probe sits at -2 + 0.25.
    let mut engine = Scripted::new(
        vec![0.0],
        vec![1.0],
        well(HALF_WIDTH),
        harmonic_force(CURVATURE, -HALF_WIDTH),
    );
    let found = estimate(&mut engine, &plane(&[1.0], 0.0), &analytic());
    assert_close(
        found.rate,
        harmonic_rate(CURVATURE, 1.0, BARRIER, THERMAL),
        RELATIVE,
        "rate",
    );
    assert_close(
        engine.log.step_offsets[0],
        -HALF_WIDTH + 0.25,
        RELATIVE,
        "probe offset",
    );
}

#[test]
fn a_flat_stationary_point_is_not_a_minimum() {
    let mut engine = Scripted::new(
        vec![0.0],
        vec![1.0],
        Box::new(|_| (0.0, Array1::from_vec(vec![0.0]))),
        harmonic_force(1.0, 0.0),
    );
    assert_eq!(
        error(&mut engine, &plane(&[1.0], 0.0), &quiet()),
        invalid("OH-TST reactant minimization exceeded its iteration budget")
    );
}

#[test]
fn a_force_peak_in_any_component_is_not_stationary() {
    // Only the second component of the force is non-zero at the start.
    for start in [[-HALF_WIDTH, 0.5], [-HALF_WIDTH, -0.5]] {
        let mut engine = Scripted::new(
            start.to_vec(),
            vec![1.0, 1.0],
            Box::new(|x| {
                let (energy, slope) = well(HALF_WIDTH)(Array1::from_vec(vec![x[0]]).view());
                (
                    energy + 0.5 * x[1] * x[1],
                    Array1::from_vec(vec![slope[0], -x[1]]),
                )
            }),
            harmonic_force(CURVATURE, -HALF_WIDTH),
        );
        let found = estimate(&mut engine, &plane(&[1.0, 0.0], 0.0), &quiet());
        assert_close(
            found.rate,
            harmonic_rate(CURVATURE, 1.0, BARRIER, THERMAL),
            RELATIVE,
            "rate",
        );
        assert!(
            engine.log.first_step_positions[0][1].abs() < 1.0e-6,
            "the transverse force was relaxed before probing"
        );
    }
}

#[test]
fn force_at_the_flatness_tolerance_counts_as_stationary() {
    let tolerance = SERIES_FLAT_TOL;
    let potential: Potential = Box::new(move |x| {
        let (energy, slope) = well(HALF_WIDTH)(Array1::from_vec(vec![x[0]]).view());
        if x[0] == -HALF_WIDTH {
            (energy, Array1::from_vec(vec![tolerance]))
        } else {
            (energy, Array1::from_vec(vec![slope[0]]))
        }
    });
    let mut engine = Scripted::new(
        vec![-HALF_WIDTH],
        vec![1.0],
        potential,
        harmonic_force(CURVATURE, -HALF_WIDTH),
    );
    let found = estimate(&mut engine, &plane(&[1.0], 0.0), &quiet());
    assert_eq!(engine.log.first_step_positions[0][0], -HALF_WIDTH + 0.25);
    assert!(found.rate > 0.0);
}

fn seam_potential(top_energy: impl Fn(f64) -> f64 + 'static) -> Potential {
    Box::new(move |x| {
        let u = x[0];
        let force = if u <= 0.0 { -(u + 1.0) } else { 1.0 };
        (top_energy(u), Array1::from_vec(vec![force]))
    })
}

#[test]
fn a_dividing_surface_not_above_the_minimum_is_rejected() {
    let message = invalid("OH-TST dividing surface must lie above the reactant minimum");
    for energy in [
        Box::new(|_: f64| 0.0) as Box<dyn Fn(f64) -> f64>,
        Box::new(|u: f64| -0.5 * (u + 1.0)),
        Box::new(|u: f64| if u > -0.5 { f64::NAN } else { 0.0 }),
    ] {
        let mut engine = Scripted::new(
            vec![-1.0],
            vec![1.0],
            seam_potential(energy),
            harmonic_force(1.0, -1.0),
        );
        assert_eq!(error(&mut engine, &plane(&[1.0], 0.0), &quiet()), message);
    }
}

#[test]
fn a_barrier_that_rounds_the_probe_onto_the_minimum_is_rejected() {
    // The minimum sits at 0, so a dividing offset of 2^-1074 is representable.
    // Half of it underflows to zero.
    let tiny = f64::from_bits(1);
    let potential: Potential = Box::new(|x| (x[0].abs(), Array1::from_vec(vec![-x[0]])));
    let mut engine = Scripted::new(vec![0.0], vec![1.0], potential, harmonic_force(1.0, 0.0));
    assert_eq!(
        error(&mut engine, &plane(&[1.0], tiny), &quiet()),
        invalid("OH-TST curvature probe length vanished")
    );
}

#[test]
fn a_probe_that_cannot_move_the_plane_gives_no_curvature() {
    // The dividing offset is the float after the minimum at 1, so the probe is
    // half a unit in the last place and `1 + probe` rounds back to 1. The
    // plane the constraint holds is then the minimum itself.
    let next_after_one = 1.0 + f64::EPSILON;
    let potential: Potential = Box::new(|x| {
        let d = x[0] - 1.0;
        (d.abs(), Array1::from_vec(vec![-d]))
    });
    let mut engine = Scripted::new(vec![1.0], vec![1.0], potential, harmonic_force(1.0, 1.0));
    assert_eq!(
        error(&mut engine, &plane(&[1.0], next_after_one), &quiet()),
        invalid("OH-TST reactant curvature must be finite and positive")
    );
}

#[test]
fn a_negative_or_zero_mean_force_gives_no_curvature() {
    for sign in [-1.0, 0.0] {
        let mut engine = Scripted::new(
            vec![-HALF_WIDTH],
            vec![1.0],
            well(HALF_WIDTH),
            Box::new(move |plane, _| sign * (plane.offset + HALF_WIDTH)),
        );
        assert_eq!(
            error(&mut engine, &plane(&[1.0], 0.0), &quiet()),
            invalid("OH-TST reactant curvature must be finite and positive")
        );
    }
}

#[test]
fn engine_state_is_restored_after_success_and_failure() {
    let mut engine = well_engine(-1.3 * HALF_WIDTH);
    engine.velocities[0] = 0.75;
    engine
        .set_constraint(Some(plane(&[1.0], 0.5)))
        .expect("constraint");
    let positions = engine.positions.clone();
    let velocities = engine.velocities.clone();
    let constraint = engine.constraint.clone();
    estimate(&mut engine, &plane(&[1.0], 0.0), &quiet());
    assert_eq!(engine.positions, positions);
    assert_eq!(engine.velocities, velocities);
    assert_eq!(engine.constraint, constraint);
    let err = error(
        &mut engine,
        &plane(&[1.0], 1.0),
        &quiet().with_max_translation(0.8).with_max_outer_steps(2),
    );
    assert_eq!(
        err,
        invalid("OH-TST found no stationary dividing surface along the seed normal")
    );
    assert_eq!(engine.positions, positions);
    assert_eq!(engine.velocities, velocities);
    assert_eq!(engine.constraint, constraint);
}
