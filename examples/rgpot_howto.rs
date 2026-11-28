//! How-to: wrap an rgpot potential as a [`PointSurface`].
//!
//! `rgpot-pair` is the zero-dependency half of the rgpot project: plain
//! Rust functions over flat position arrays, no RPC server and no
//! capnp. `lj_gradient` computes the energy and gradient of a whole
//! Lennard-Jones cluster, in 3-D, for every pair inside `cutoff` (a
//! non-positive cutoff keeps every pair).
//!
//! The system here is a migrating atom `B` between two fixed walls `A`
//! and `C` at `(-1.5, 0, 0)` and `(1.5, 0, 0)`, unit epsilon and sigma.
//! `A` and `C` are far enough apart that `B` prefers to sit near one
//! wall or the other, so the line between them is a double well with the
//! saddle at the midpoint by symmetry.
//!
//! Run:
//!
//! ```console
//! $ cargo run --release --example rgpot_howto
//! ```

use ndarray::{Array1, ArrayView1, array};
use rgflux::pafi::cartesian::zero_temperature_barrier;
use rgsaddle::{
    ArcPath, MinModeConfig, MinModeSession, MinModeStatus, PointSurface, RelaxConfig, SaddleError,
    relax_to_tolerance,
};

/// Half the A-C separation. Past `rmin = 2^(1/6)`, so the midpoint is a
/// barrier rather than the single minimum.
const WALL: f64 = 1.5;

/// `B`'s position in the fixed-wall frame: a thin [`PointSurface`] over
/// `rgpot_pair::lj_gradient`, which wants a flat array of 3 atoms.
struct MigratingAtom;

impl PointSurface for MigratingAtom {
    fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let positions = [-WALL, 0.0, 0.0, q[0], q[1], q[2], WALL, 0.0, 0.0];
        let mut gradient = [0.0; 9];
        let energy = rgpot_pair::lj_gradient(1.0, 1.0, -1.0, &positions, &mut gradient);
        Ok((energy, array![gradient[3], gradient[4], gradient[5]]))
    }
}

fn main() -> Result<(), SaddleError> {
    let surface = MigratingAtom;

    let cell = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let outcome = relax_to_tolerance(
        &surface,
        array![-0.6, 0.02, 0.0],
        cell,
        &RelaxConfig::default(),
    )?;
    let near_a = outcome.positions;
    println!(
        "minimum near A: ({:.6}, {:.6}, {:.6}), energy {:.6}, {} steps",
        near_a[0], near_a[1], near_a[2], outcome.energy, outcome.steps
    );

    let config = MinModeConfig::default().with_force_tol(1.0e-8);
    let mut session = MinModeSession::new(config, array![0.0, 0.05, 0.0], array![1.0, 0.0, 0.0])?;
    let mut converged = false;
    for _ in 0..2000 {
        if session.step(&surface)?.status == MinModeStatus::Converged {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(SaddleError::Solver("the dimer did not converge".into()));
    }
    let saddle = session.position().to_owned();
    println!(
        "saddle: ({:.6}, {:.6}, {:.6}), energy {:.6}",
        saddle[0],
        saddle[1],
        saddle[2],
        surface.eval(saddle.view())?.0
    );

    let saddle_energy = surface.eval(saddle.view())?.0;
    let barrier = saddle_energy - outcome.energy;
    println!("zero-temperature barrier from the relaxed minimum: {barrier:.6}");

    let mirror = array![-near_a[0], near_a[1], near_a[2]];
    let path = ArcPath::from_positions(&[near_a.clone(), saddle.clone(), mirror])?;
    let cold = zero_temperature_barrier(&surface, &path, &path.arcs())?;
    println!(
        "same barrier along a PAFI path over this surface: {:.6}",
        cold.barrier
    );

    Ok(())
}
