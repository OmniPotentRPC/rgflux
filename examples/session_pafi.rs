//! One straight-path PAFI plane on a live MD session.
//!
//! The path translates the solvent along the first lattice vector, so every
//! plane has the same normal and the mean multiplier is the free-energy
//! derivative. One process samples one plane, so planes run side by side.
//!
//! usage: session_pafi ENGINE SPEC SOLUTE AXIS_SHIFT_A BURN SAMPLE SEED [START.txt]
//!
//! SOLUTE is the number of leading atoms the path leaves in place. The solvent
//! is rigid three-site molecules. The plane sits AXIS_SHIFT_A from the starting
//! configuration; the system is first pulled there in 0.005 A stages of 50
//! steps. PAFI_DT_PS sets the time step, 0.001 ps when it is unset.
use std::io::Write;

use ndarray::Array1;
use rgflux::engine::md_session::MdSessionEngine;
use rgflux::engine::{DynamicsEngine, HyperplaneConstraint, ThermostatConfig};
use rgflux::stats::{StatsConfig, batch_means};
use rgpot_core::md_session::CAbiMdSession;

const KB: f64 = 8.617_333_262e-5;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    let solute: usize = a[3].parse()?;
    let shift: f64 = a[4].parse()?;
    let (burn, sample): (usize, usize) = (a[5].parse()?, a[6].parse()?);
    let seed: u64 = a[7].parse()?;
    let dt: f64 = std::env::var("PAFI_DT_PS")
        .map_or(Ok(0.001), |text| text.parse())?;
    let threads = std::env::var("OMP_NUM_THREADS").unwrap_or_else(|_| "1".into());
    // SAFETY: the engine is built from the rgpot tree this crate links.
    let session = unsafe {
        CAbiMdSession::open_engine(
            std::path::Path::new(&a[1]),
            a[2].as_bytes(),
            Some(&format!("threads={threads}")),
        )
    }?;
    let thermostat = ThermostatConfig::default()
        .with_gamma(1.0)
        .with_temperature(300.0)
        .with_boltzmann(KB);
    let mut engine = MdSessionEngine::new(session, thermostat)?;
    if let Some(path) = a.get(8) {
        let x: Array1<f64> = std::fs::read_to_string(path)?
            .split_whitespace()
            .map(str::parse)
            .collect::<Result<Vec<f64>, _>>()?
            .into();
        engine.set_positions(x.view())?;
    }
    let dof = engine.dof();
    let cell = *engine.cell().ok_or("the session has no cell")?;
    let edge = (cell[0].powi(2) + cell[1].powi(2) + cell[2].powi(2)).sqrt();
    let axis = [cell[0] / edge, cell[1] / edge, cell[2] / edge];
    let mut normal = Array1::<f64>::zeros(dof);
    for k in 3 * solute..dof {
        normal[k] = engine.masses()[k] * axis[k % 3];
    }
    let total: f64 = (solute..dof / 3).map(|i| engine.masses()[3 * i]).sum();
    let norm = normal.dot(&normal).sqrt();
    normal /= norm;
    let groups: Vec<Vec<usize>> = (solute..dof / 3)
        .step_by(3)
        .map(|i| vec![i, i + 1, i + 2])
        .collect();
    engine.check_rigid_translation(normal.view(), &groups)?;
    // A translation by s moves n . x by s * (n . e), with n . e = M / norm.
    let scale = total / norm;
    let start = normal.dot(&engine.positions());
    engine.shoot_velocities(seed)?;
    let stages = (shift.abs() / 0.005).ceil() as usize;
    for k in 1..=stages {
        let offset = start + scale * shift * k as f64 / stages as f64;
        engine.set_constraint(Some(HyperplaneConstraint::new(normal.clone(), offset)?))?;
        engine.run(dt, 50)?;
    }
    let offset = start + scale * shift;
    engine.set_constraint(Some(HyperplaneConstraint::new(normal.clone(), offset)?))?;
    engine.run(dt, burn)?;
    let mut series = Vec::with_capacity(sample);
    let mut heat = 0.0;
    let mut out = std::io::BufWriter::new(std::fs::File::create(format!(
        "plane_{shift:+.3}_{seed}.txt"
    ))?);
    let mut chunks = 0.0;
    while series.len() < sample {
        // dF/ds = lambda * dc/ds in eV per Angstrom of solvent displacement.
        let chunk = engine.run(dt, 1000)?;
        let mean = chunk.iter().sum::<f64>() / chunk.len() as f64;
        writeln!(out, "{} {}", series.len(), mean * scale)?;
        series.extend(chunk.iter().map(|l| l * scale));
        heat += engine.kinetic_energy();
        chunks += 1.0;
        out.flush()?;
    }
    out.flush()?;
    let mut last = std::fs::File::create(format!("last_{shift:+.3}_{seed}.txt"))?;
    for x in engine.positions() {
        write!(last, "{x:.17e} ")?;
    }
    let estimate = batch_means(&series, &StatsConfig::default())?;
    println!(
        "shift {shift} seed {seed} gradient {} error {} kinetic {} residual {}",
        estimate.mean,
        estimate.standard_error,
        heat / chunks,
        normal.dot(&engine.positions()) - offset
    );
    Ok(())
}
