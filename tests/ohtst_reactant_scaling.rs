//! Reactant minimization cost on periodic fcc Lennard-Jones cells.
//!
//! The ignored `reactant_minimization_scaling` prints wall time, force
//! evaluations and the largest force component reached for cells of 32 to
//! 8788 atoms. Run it with
//! `cargo test --release --test ohtst_reactant_scaling -- --ignored --nocapture`.

mod ohtst_lj;

use ndarray::Array1;
use ohtst_lj::{LjCell, StopAtSampling, engine, lattice, perturbed};
use rgflux::FluxError;
use rgflux::engine::{DynamicsEngine, HyperplaneConstraint};
use rgflux::ohtst::{
    OhtstConfig, ReactantMinimum, optimize_hyperplane, optimize_hyperplane_from_minimum,
    relax_reactant,
};
use std::time::Instant;

const AMPLITUDE: f64 = 0.05;
const SIZES: [usize; 7] = [2, 3, 4, 6, 8, 10, 13];

fn max_force(engine: &mut impl DynamicsEngine) -> f64 {
    let (_, force) = engine.energy_and_forces().expect("forces");
    force.iter().fold(0.0_f64, |m, f| m.max(f.abs()))
}

/// A seed plane just ahead of the first atom along `x`.
fn seed_plane(start: &Array1<f64>) -> HyperplaneConstraint {
    let mut direction = Array1::zeros(start.len());
    direction[0] = 1.0;
    HyperplaneConstraint::from_direction(direction.view(), start[0] + 0.3).expect("plane")
}

struct Stop {
    seconds: f64,
    evaluations: usize,
    outcome: String,
}

/// Runs `run` until the engine is first asked to sample, which happens only
/// after the reactant minimization and the dividing-plane search.
fn until_sampling(
    cells: usize,
    start: &Array1<f64>,
    run: impl FnOnce(&mut StopAtSampling) -> Result<rgflux::ohtst::OhtstEstimate, FluxError>,
) -> Stop {
    let mut engine = StopAtSampling::new(engine(cells, start));
    let clock = Instant::now();
    let result = run(&mut engine);
    match (engine.stopped, result) {
        (Some((at, evaluations)), _) => Stop {
            seconds: at.duration_since(clock).as_secs_f64(),
            evaluations,
            outcome: "reached sampling".to_owned(),
        },
        (None, result) => Stop {
            seconds: clock.elapsed().as_secs_f64(),
            evaluations: engine.inner.surface().evaluations(),
            outcome: match result {
                Err(error) => error.to_string(),
                Ok(_) => "unexpected rate".to_owned(),
            },
        },
    }
}

#[test]
fn lattice_forces_are_the_negative_energy_gradient() {
    let start = perturbed(2, AMPLITUDE, 3);
    let surface = LjCell::new(2);
    let (_, gradient) = rgsaddle::PointSurface::eval(&surface, start.view()).expect("eval");
    let h = 1.0e-6;
    for k in [0, 7, 50, 95] {
        let mut ahead = start.clone();
        let mut behind = start.clone();
        ahead[k] += h;
        behind[k] -= h;
        let (e_ahead, _) = rgsaddle::PointSurface::eval(&surface, ahead.view()).expect("ahead");
        let (e_behind, _) = rgsaddle::PointSurface::eval(&surface, behind.view()).expect("behind");
        let slope = (e_ahead - e_behind) / (2.0 * h);
        assert!(
            (slope - gradient[k]).abs() < 1.0e-6 * (1.0 + gradient[k].abs()),
            "coordinate {k}: finite difference {slope:e}, gradient {:e}",
            gradient[k]
        );
    }
}

#[test]
fn perfect_lattice_forces_vanish_only_to_rounding() {
    let mut engine = engine(4, &lattice(4));
    let fmax = max_force(&mut engine);
    assert!(fmax < 1.0e-10, "perfect lattice fmax {fmax:e}");
}

#[test]
fn small_cells_relax_within_the_default_tolerance() {
    let config = OhtstConfig::default();
    for cells in [2, 3, 4] {
        for start in [lattice(cells), perturbed(cells, AMPLITUDE, 5)] {
            let mut engine = engine(cells, &start);
            let reactant = relax_reactant(&mut engine, &config).expect("relaxed reactant");
            assert!(reactant.fmax <= config.reactant_fmax);
            assert_eq!(engine.positions(), start.view());
            let stop = until_sampling(cells, &start, |engine| {
                optimize_hyperplane(engine, &seed_plane(&start), &config)
            });
            assert_eq!(stop.outcome, "reached sampling");
        }
    }
}

#[test]
#[ignore = "timing sweep up to 8788 atoms; run with --release --ignored --nocapture"]
fn reactant_minimization_scaling() {
    let config = OhtstConfig::default();
    println!(
        "| atoms | start | relax_reactant s | evals | fmax | to sampling s | evals | outcome | search only s | evals |"
    );
    for cells in SIZES {
        let atoms = 4 * cells * cells * cells;
        for (label, start) in [
            ("lattice", lattice(cells)),
            ("perturbed", perturbed(cells, AMPLITUDE, 5)),
        ] {
            let mut relax_engine = engine(cells, &start);
            let clock = Instant::now();
            let reactant: ReactantMinimum =
                relax_reactant(&mut relax_engine, &config).expect("relaxed reactant");
            let relax_seconds = clock.elapsed().as_secs_f64();
            let relax_evals = relax_engine.surface().evaluations();

            let plane = seed_plane(&start);
            let full = until_sampling(cells, &start, |engine| {
                optimize_hyperplane(engine, &plane, &config)
            });
            let search = until_sampling(cells, &start, |engine| {
                optimize_hyperplane_from_minimum(engine, &plane, &reactant, &config)
            });
            println!(
                "| {atoms} | {label} | {relax_seconds:.4} | {relax_evals} | {:.2e} | {:.4} | {} | {} | {:.4} | {} |",
                reactant.fmax,
                full.seconds,
                full.evaluations,
                full.outcome,
                search.seconds,
                search.evaluations,
            );
        }
    }
}
