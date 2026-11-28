//! OH-TST tutorial: plane optimisation, sampling, MBAR, transmission and
//! rate on the same surface as the PAFI tutorial.
//!
//! `V(x, y) = (x² - 1)² + ½ k(x) y²` with `k(x) = 2.5 + 1.5 x`; see
//! `examples/pafi_tutorial.rs` for the derivation. The valley floor is the
//! straight line `y = 0`, and the potential is symmetric under `y -> -y`:
//! the Hessian at the saddle has no `x`-`y` coupling, and the dividing
//! plane the dimer finds (normal to the unstable mode) is already the
//! optimal hyperplane. The plane optimiser runs and reports that there is
//! nothing to improve, which is itself a useful thing to see on a first
//! run. `docs/orgmode/howto/oh-tst.org` has a case where the optimiser
//! does move the plane.
//!
//! Run directly for the walkthrough in `tutorials/oh-tst`:
//!
//! ```console
//! $ cargo run --release --features ohtst --example ohtst_tutorial
//! ```
//!
//! Run with an output path for the JSON mode
//! `docs/orgmode/tutorials/oh-tst.org` uses from Python:
//!
//! ```console
//! $ cargo run --release --features ohtst --example ohtst_tutorial -- result.json
//! ```

use std::env;
use std::fs;

use ndarray::{Array1, ArrayView1, array};
use rgflux::ohtst::{
    AnalyticSurfaceDynamics, AnalyticSurfaceSampler, BasinConfig, GaussianTransport, Hyperplane,
    LangevinConfig, PlaneOptimizer, PlaneOptimizerConfig, PlanePath, ReversibleWorkConfig,
    TransmissionConfig, WorkEstimate, basin_density, ohtst_rate, reversible_work_along,
    sample_plane, transmission_coefficient,
};
use rgsaddle::{MinModeConfig, MinModeSession, MinModeStatus, PointSurface, SaddleError};

struct CurvedWell;

/// `k(x) = 2.5 + 1.5 x`, matching `examples/pafi_tutorial.rs`.
fn stiffness(x: f64) -> f64 {
    2.5 + 1.5 * x
}

impl PointSurface for CurvedWell {
    fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (x, y) = (q[0], q[1]);
        let k = stiffness(x);
        let well = x * x - 1.0;
        let energy = well * well + 0.5 * k * y * y;
        let dk_dx = 1.5;
        let dvdx = 4.0 * x * well + 0.5 * dk_dx * y * y;
        let dvdy = k * y;
        Ok((energy, array![dvdx, dvdy]))
    }
}

/// The exact transverse-Gaussian free energy of the plane at `x`, up to a
/// constant: `A(x) = (x² - 1)² - ½ kt ln(2π kt / k(x))`. The closed form
/// of the reversible work `A(0) - A(-1)` is the same as
/// `closed_form_barrier` in `examples/pafi_tutorial.rs`, since the plane
/// free energy here is the same quantity PAFI integrates along the
/// valley.
fn closed_form_work(kt: f64) -> f64 {
    1.0 + 0.5 * kt * (stiffness(0.0) / stiffness(-1.0)).ln()
}

struct TutorialResult {
    turn_degrees: f64,
    density: f64,
    density_error: f64,
    integrated_work: f64,
    integrated_error: f64,
    bar_work: f64,
    bar_error: f64,
    mbar_work: f64,
    mbar_error: f64,
    closed_form: f64,
    bare_rate: f64,
    bare_rate_error: f64,
    kappa: f64,
    kappa_error: f64,
    shots: usize,
    unassigned: usize,
    reactive: usize,
    recrossed: usize,
    final_rate: f64,
    final_error: f64,
}

fn run() -> Result<TutorialResult, SaddleError> {
    let kt = 0.1;
    let reactant = array![-1.0, 0.0];
    let product = array![1.0, 0.0];

    let config = MinModeConfig::default().with_force_tol(1.0e-8);
    let mut session = MinModeSession::new(config, array![0.05, -0.05], array![1.0, 0.2])?;
    let mut converged = false;
    for _ in 0..2000 {
        if session.step(&CurvedWell)?.status == MinModeStatus::Converged {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(SaddleError::Solver("the dimer did not converge".into()));
    }
    let seed = Hyperplane::from_min_mode(&session)?.oriented_toward(product.view());

    let sampler = LangevinConfig::default()
        .with_thermal(kt)
        .with_time_step(0.02)
        .with_friction(1.0)
        .with_burn_in(1000)
        .with_stride(5)
        .with_seed(11);
    let optimizer_config = PlaneOptimizerConfig::default()
        .with_sampler(sampler.clone().with_records(2000))
        .with_max_iterations(50);
    let optimum = PlaneOptimizer::new(&seed, optimizer_config)?.run(&CurvedWell)?;
    let turn_degrees = (optimum.plane.normal()[1].atan2(optimum.plane.normal()[0])
        - seed.normal()[1].atan2(seed.normal()[0]))
    .to_degrees();

    let sampled = sample_plane(
        &CurvedWell,
        &optimum.plane,
        None,
        &sampler.clone().with_records(8000).with_retain(256),
    )?;

    let basin = basin_density(
        &CurvedWell,
        reactant.view(),
        product.view(),
        &BasinConfig::default().with_sampler(sampler.clone().with_records(4000)),
    )?;

    let path = PlanePath::legs(&[basin.plane.clone(), optimum.plane.clone()])?;
    let work_config = ReversibleWorkConfig::default()
        .with_sampler(sampler.clone().with_records(2000))
        .with_nodes(9);
    let mut integrator = AnalyticSurfaceSampler::new(&CurvedWell, work_config.sampler.clone())
        .with_start(reactant.clone());
    let integrated = reversible_work_along(&mut integrator, &path, &work_config)?;
    let mbar_config = work_config
        .with_transport(GaussianTransport::default().into())
        .with_mbar(true)
        .with_estimate(WorkEstimate::Mbar);
    let mut mapped = AnalyticSurfaceSampler::new(&CurvedWell, mbar_config.sampler.clone())
        .with_start(reactant.clone());
    let work = reversible_work_along(&mut mapped, &path, &mbar_config)?;
    let closed_form = closed_form_work(kt);

    let rate = ohtst_rate(&basin, &work)?;

    let transmission_config = TransmissionConfig::default()
        .with_thermal(kt)
        .with_friction(1.0)
        .with_time_step(0.02)
        .with_steps(400)
        .with_check_every(40)
        .with_shots(sampled.retained.len().min(4096))
        .with_blocks(16)
        .with_seed(23);
    let transmission = transmission_coefficient(
        &mut AnalyticSurfaceDynamics::new(&CurvedWell),
        &optimum.plane,
        &sampled.retained,
        &transmission_config,
    )?;
    let (final_rate, final_error) = rate.transmitted(&transmission);

    Ok(TutorialResult {
        turn_degrees,
        density: basin.density,
        density_error: basin.density_error,
        integrated_work: integrated.integrated_work,
        integrated_error: integrated.integrated_error,
        bar_work: work.bar_work.unwrap_or(f64::NAN),
        bar_error: work.bar_error.unwrap_or(f64::NAN),
        mbar_work: work.mbar_work.unwrap_or(f64::NAN),
        mbar_error: work.mbar_error.unwrap_or(f64::NAN),
        closed_form,
        bare_rate: rate.rate,
        bare_rate_error: rate.error,
        kappa: transmission.kappa,
        kappa_error: transmission.error,
        shots: transmission.shots,
        unassigned: transmission.unassigned,
        reactive: transmission.reactive,
        recrossed: transmission.recrossed,
        final_rate,
        final_error,
    })
}

fn print_report(result: &TutorialResult) {
    println!(
        "optimal plane turned {:.4}° from the dimer's seed",
        result.turn_degrees
    );
    println!(
        "Z_H / Z_R = {:.4} ± {:.4}",
        result.density, result.density_error
    );
    println!(
        "reversible work reactant plane -> optimum: integration {:.5} ± {:.5}, BAR {:.5} ± {:.5}, MBAR {:.5} ± {:.5}; closed form {:.5}",
        result.integrated_work,
        result.integrated_error,
        result.bar_work,
        result.bar_error,
        result.mbar_work,
        result.mbar_error,
        result.closed_form
    );
    println!(
        "k_OH (no transmission correction) = {:.6e} ± {:.1e}",
        result.bare_rate, result.bare_rate_error
    );
    println!(
        "kappa = {:.5} ± {:.5} ({} reactive, {} recrossed of {} shots, {} unassigned)",
        result.kappa,
        result.kappa_error,
        result.reactive,
        result.recrossed,
        result.shots,
        result.unassigned
    );
    println!(
        "k = kappa * k_OH = {:.6e} ± {:.1e}",
        result.final_rate, result.final_error
    );
}

fn write_json(result: &TutorialResult, output: &str) -> Result<(), SaddleError> {
    let out = format!(
        "{{\n  \"turn_degrees\": {},\n  \"density\": {},\n  \"density_error\": {},\n  \"integrated_work\": {},\n  \"integrated_error\": {},\n  \"bar_work\": {},\n  \"bar_error\": {},\n  \"mbar_work\": {},\n  \"mbar_error\": {},\n  \"closed_form\": {},\n  \"bare_rate\": {},\n  \"bare_rate_error\": {},\n  \"kappa\": {},\n  \"kappa_error\": {},\n  \"final_rate\": {},\n  \"final_error\": {}\n}}\n",
        result.turn_degrees,
        result.density,
        result.density_error,
        result.integrated_work,
        result.integrated_error,
        result.bar_work,
        result.bar_error,
        result.mbar_work,
        result.mbar_error,
        result.closed_form,
        result.bare_rate,
        result.bare_rate_error,
        result.kappa,
        result.kappa_error,
        result.final_rate,
        result.final_error,
    );
    fs::write(output, out).map_err(|e| SaddleError::Invalid(format!("cannot write {output}: {e}")))
}

fn main() -> Result<(), SaddleError> {
    let result = run()?;
    let args: Vec<String> = env::args().collect();
    if let Some(output) = args.get(1) {
        return write_json(&result, output);
    }
    print_report(&result);
    Ok(())
}
