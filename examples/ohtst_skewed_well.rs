//! Optimal-hyperplane TST on a skewed double well, end to end.
//!
//! `V = (x² - 1)² + ½ k(x) (y - x)²` with `k(x) = 4 (1 + 0.8 tanh 2x)` has
//! minima at `±(1, 1)` and its saddle at the origin, with a valley that
//! stiffens toward the product. The coordinates are already mass-weighted
//! (unit masses). A dimer search seeds the plane, the plane optimizer
//! climbs the plane free energy, and the rate is
//!
//! `k = sqrt(kT / 2π) (Z_H / Z_R) exp(-ΔA / kT)`,
//!
//! with `Z_H / Z_R` from parallel planes through the reactant and `ΔA` the
//! reversible work from the reactant plane to the optimum, by
//! thermodynamic integration with BAR as a check. The first-order
//! correction to harmonic TST and the control variates are printed beside
//! it.
//!
//! The plane optimizer is rgmin's Riemannian trust region on the product
//! of the sphere of normals and the offset along the normal.

use ndarray::{Array1, ArrayView1, array};
use rgflux::ohtst::{
    AnalyticSurfaceSampler, AnharmonicConfig, BasinConfig, HarmonicReference, Hyperplane,
    LangevinConfig, PlaneOptimizer, PlaneOptimizerConfig, ReversibleWorkConfig,
    anharmonic_correction, basin_density, ohtst_rate, reversible_work, sample_plane,
};
use rgsaddle::{MinModeConfig, MinModeSession, MinModeStatus, PointSurface, SaddleError};

struct SkewedWell;

impl PointSurface for SkewedWell {
    fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (x, y) = (q[0], q[1]);
        let well = x * x - 1.0;
        let tanh = (2.0 * x).tanh();
        let stiffness = 4.0 * (1.0 + 0.8 * tanh);
        let valley = y - x;
        let energy = well * well + 0.5 * stiffness * valley * valley;
        let gx = 4.0 * x * well + 3.2 * (1.0 - tanh * tanh) * valley * valley - stiffness * valley;
        Ok((energy, array![gx, stiffness * valley]))
    }
}

fn main() -> Result<(), SaddleError> {
    let kt = 0.1;
    let reactant = array![-1.0, -1.0];

    let config = MinModeConfig::default().with_force_tol(1.0e-8);
    let mut session = MinModeSession::new(config, array![0.1, -0.05], array![1.0, 0.3])?;
    let mut converged = false;
    for _ in 0..2000 {
        if session.step(&SkewedWell)?.status == MinModeStatus::Converged {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(SaddleError::Solver("the dimer did not converge".into()));
    }
    let seed = Hyperplane::from_min_mode(&session)?.oriented_toward(array![1.0, 1.0].view());
    println!(
        "dimer plane: point ({:.6}, {:.6}), normal ({:.4}, {:.4})",
        seed.point()[0],
        seed.point()[1],
        seed.normal()[0],
        seed.normal()[1]
    );

    let sampler = LangevinConfig::default()
        .with_thermal(kt)
        .with_time_step(0.1)
        .with_friction(1.0)
        .with_burn_in(300)
        .with_stride(5)
        .with_seed(9);
    let config = PlaneOptimizerConfig::default()
        .with_sampler(sampler.clone().with_records(2000))
        .with_max_iterations(100);
    let optimum = PlaneOptimizer::new(&seed, config)?.run(&SkewedWell)?;
    let normal = optimum.plane.normal();
    let turn = (normal[1].atan2(normal[0]) - seed.normal()[1].atan2(seed.normal()[0])).to_degrees();
    println!(
        "optimal plane after {} trust-region iterations ({} inner) and {} force calls: offset {:.4} along the normal, turned {turn:.2}°",
        optimum.iterations,
        optimum.inner_iterations,
        optimum.evaluations,
        optimum.plane.normal().dot(&optimum.plane.point())
    );

    let reference = HarmonicReference::new(seed.point().to_owned());
    let controlled = sample_plane(
        &SkewedWell,
        &optimum.plane,
        None,
        &sampler.clone().with_records(4000).with_control(reference),
    )?;
    if let Some(raw) = &controlled.uncontrolled {
        println!(
            "mean force on the optimum: {:.5} ± {:.5} with control variates, {:.5} ± {:.5} without",
            controlled.normal_force,
            controlled.normal_force_error,
            raw.normal_force,
            raw.normal_force_error
        );
    }

    let basin = basin_density(
        &SkewedWell,
        reactant.view(),
        array![1.0, 1.0].view(),
        &BasinConfig::default().with_sampler(sampler.clone().with_records(4000)),
    )?;
    println!(
        "Z_H / Z_R = {:.4} ± {:.4} from {} parallel planes; harmonic {:.4}",
        basin.density,
        basin.density_error,
        basin.offsets.len(),
        basin.harmonic_density
    );
    let work_config = ReversibleWorkConfig::default().with_sampler(sampler.with_records(2000));
    let mut work_sampler = AnalyticSurfaceSampler::new(&SkewedWell, work_config.sampler.clone())
        .with_start(reactant.clone());
    let work = reversible_work(
        &mut work_sampler,
        &[basin.plane.clone(), optimum.plane.clone()],
        &work_config,
    )?;
    let (bar, bar_error) = (
        work.bar_work.unwrap_or(f64::NAN),
        work.bar_error.unwrap_or(f64::NAN),
    );
    println!(
        "ΔA = {:.4} ± {:.4} by integration, {bar:.4} ± {bar_error:.4} by BAR",
        work.work, work.error
    );
    let rate = ohtst_rate(&basin, &work)?;
    println!(
        "k_OH = {:.4e} ± {:.1e} (ln k = {:.4} ± {:.4}), with unit mass in mass-weighted coordinates",
        rate.rate, rate.error, rate.log_rate, rate.log_rate_error
    );

    let correction = anharmonic_correction(
        &SkewedWell,
        reactant.view(),
        &seed,
        &AnharmonicConfig::default().with_probes(2000),
    )?;
    let (ratio, ratio_error) = correction.log_ratio(kt);
    println!("first-order ln(k / k_HTST) at this kT: {ratio:.4} ± {ratio_error:.4}");
    Ok(())
}
