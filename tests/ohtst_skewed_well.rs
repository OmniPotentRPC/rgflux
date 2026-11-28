//! OH-TST on a skewed double well whose optimal plane shifts and turns
//! with temperature: `(x² - 1)² + ½ k₀ (1 + α tanh 2x) (y - x)²` with
//! `k₀ = 4`, `α = 0.8`, unit masses. The reactant minimum is `(-1, -1)`,
//! the saddle the origin one energy unit up, and the stiffness across the
//! valley rises toward the product, so the free energy of a line through
//! the saddle changes both along and across the barrier.
//!
//! The references are quadrature of line integrals and the third and
//! fourth derivatives of the surface at its stationary points. An
//! independent adaptive quadrature outside the crate (SciPy) maximizing
//! the full ratio `Z / Z_R` over lines gave offsets 0.0041, 0.0397 and
//! 0.1451 and turns -0.235°, -2.288° and -7.123° at `kT` = 0.01, 0.1 and
//! 0.4. Maximizing `A` alone, the criterion of Jóhannesson and Jónsson,
//! J. Chem. Phys. 115, 9644 (2001), agrees to four digits up to
//! `kT = 0.1` and gives 0.1374 and -7.359° at `kT = 0.4`. The rate differs
//! by 2e-4 in `ln k` there.

use ndarray::{Array1, array};
use rgflux::ohtst::{
    AnalyticSurfaceDynamics, AnalyticSurfaceSampler, AnharmonicConfig, BasinConfig, BasinEnd,
    Basins, EndState, GaussianTransport, Hyperplane, LangevinConfig, PlaneOptimizer,
    PlaneOptimizerConfig, PlanePath, ReversibleWorkConfig, TransmissionConfig, WorkEstimate,
    anharmonic_correction, basin_density, ohtst_rate, reactant_density, reversible_work,
    reversible_work_along, transmission_coefficient,
};
use rgsaddle::{MinModeConfig, MinModeSession, MinModeStatus};

include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/common/ohtst_surfaces.rs"
));

use ohtst_surfaces::{
    SKEWED_ALPHA, SKEWED_K0, SkewedWell, descend, descent_path, eigen2, half_plane_integral,
    hessian, line_draws, line_free_energy, line_integral, stationary,
};

const REACTANT: [f64; 2] = [-1.0, -1.0];
const SADDLE_ENERGY: f64 = 1.0;
const EXTENT: f64 = 8.0;

/// Saddle eigenvalues `2 ∓ 2√5`; the unstable mode is along
/// `(2, √5 - 1)`.
fn saddle_eigenvalues() -> (f64, f64) {
    let root = 2.0 * 5.0f64.sqrt();
    (2.0 - root, 2.0 + root)
}

fn saddle_angle() -> f64 {
    ((5.0f64.sqrt() - 1.0) / 2.0).atan()
}

/// Reactant Hessian `[[8 + k, -k], [-k, k]]` with `k = k₀ (1 - α tanh 2)`.
fn reactant_eigenvalues() -> (f64, f64) {
    let k = SKEWED_K0 * (1.0 - SKEWED_ALPHA * 2.0f64.tanh());
    let (trace, det) = (8.0 + 2.0 * k, 8.0 * k);
    let root = (trace * trace - 4.0 * det).sqrt();
    (0.5 * (trace - root), 0.5 * (trace + root))
}

fn unit(theta: f64) -> [f64; 2] {
    [theta.cos(), theta.sin()]
}

fn off_saddle(offset: f64, theta: f64) -> [f64; 2] {
    let n = unit(theta);
    [offset * n[0], offset * n[1]]
}

fn free_energy(point: [f64; 2], theta: f64, kt: f64) -> f64 {
    line_free_energy(
        SkewedWell::energy,
        point,
        unit(theta),
        kt,
        SADDLE_ENERGY,
        EXTENT,
    )
}

/// The line of largest free energy by Newton's method on quadrature: its
/// offset from the saddle along its normal, its angle, and the Hessian of
/// `A / kT` in those coordinates.
fn optimal_line(kt: f64) -> (f64, f64, [[f64; 2]; 2]) {
    let f = |d: f64, t: f64| free_energy(off_saddle(d, t), t, kt) / kt;
    let (mut d, mut t) = (0.0, saddle_angle());
    let h = 1.0e-3;
    let mut curvature = [[0.0; 2]; 2];
    for _ in 0..8 {
        let f0 = f(d, t);
        let (fdp, fdm, ftp, ftm) = (f(d + h, t), f(d - h, t), f(d, t + h), f(d, t - h));
        let cross =
            (f(d + h, t + h) - f(d + h, t - h) - f(d - h, t + h) + f(d - h, t - h)) / (4.0 * h * h);
        let (gd, gt) = ((fdp - fdm) / (2.0 * h), (ftp - ftm) / (2.0 * h));
        let (hdd, htt) = (
            (fdp - 2.0 * f0 + fdm) / (h * h),
            (ftp - 2.0 * f0 + ftm) / (h * h),
        );
        let det = hdd * htt - cross * cross;
        d -= (htt * gd - cross * gt) / det;
        t -= (hdd * gt - cross * gd) / det;
        curvature = [[hdd, cross], [cross, htt]];
    }
    (d, t, curvature)
}

/// `ln k` of a line by quadrature, `k = sqrt(kT/2π) Z / Z_R` with `Z_R`
/// the integral behind it.
fn log_rate(kt: f64, offset: f64, theta: f64) -> f64 {
    log_line_rate(kt, off_saddle(offset, theta), theta)
}

/// `ln k` of the line through `point` with its normal at `theta`.
fn log_line_rate(kt: f64, point: [f64; 2], theta: f64) -> f64 {
    let behind = half_plane_integral(
        SkewedWell::energy,
        point,
        unit(theta),
        kt,
        SADDLE_ENERGY,
        EXTENT,
    );
    0.5 * (kt / (2.0 * std::f64::consts::PI)).ln()
        - free_energy(point, theta, kt) / kt
        - behind.ln()
}

/// `ln k_HTST = ln(√(λ_R1 λ_R2) / (2π √λ‡)) - ΔV / kT`.
fn log_harmonic_rate(kt: f64) -> f64 {
    let (r1, r2) = reactant_eigenvalues();
    let (_, stable) = saddle_eigenvalues();
    ((r1 * r2).sqrt() / (2.0 * std::f64::consts::PI * stable.sqrt())).ln() - 1.0 / kt
}

fn dimer_plane() -> (Hyperplane, f64) {
    let config = MinModeConfig::default().with_force_tol(1.0e-8);
    let mut session = MinModeSession::new(config, array![0.1, -0.05], array![1.0, 0.3]).unwrap();
    for _ in 0..2000 {
        let report = session.step(&SkewedWell).unwrap();
        if report.status == MinModeStatus::Converged {
            return (
                Hyperplane::from_min_mode(&session)
                    .unwrap()
                    .oriented_toward(array![1.0, 1.0].view()),
                report.curvature,
            );
        }
    }
    panic!("the dimer did not converge");
}

fn angle_of(plane: &Hyperplane) -> f64 {
    plane.normal()[1].atan2(plane.normal()[0])
}

fn sampler(kt: f64, time_step: f64) -> LangevinConfig {
    LangevinConfig::default()
        .with_thermal(kt)
        .with_time_step(time_step)
        .with_friction(1.0)
        .with_burn_in(300)
        .with_stride(5)
        .with_seed(9)
}

/// The stationary points have their closed forms, and a dimer from near
/// the saddle finds the saddle plane OH-TST starts from.
#[test]
fn skewed_well_stationary_points_and_dimer() {
    let saddle = stationary(SkewedWell::energy, [0.05, -0.03]);
    assert!(saddle[0].abs() < 1e-9 && saddle[1].abs() < 1e-9);
    let (lowest, highest, mode) = eigen2(hessian(SkewedWell::energy, saddle));
    let (unstable, stable) = saddle_eigenvalues();
    assert!((lowest - unstable).abs() < 1e-5 && (highest - stable).abs() < 1e-5);
    let angle = mode[1].atan2(mode[0]);
    let folded = (angle - saddle_angle()).rem_euclid(std::f64::consts::PI);
    assert!(folded.min(std::f64::consts::PI - folded) < 1e-6);
    let minimum = descend(SkewedWell::energy, [-0.8, -1.2]);
    assert!((minimum[0] - REACTANT[0]).abs() < 1e-6 && (minimum[1] - REACTANT[1]).abs() < 1e-6);
    let (soft, stiff, _) = eigen2(hessian(SkewedWell::energy, REACTANT));
    let (r1, r2) = reactant_eigenvalues();
    assert!((soft - r1).abs() < 1e-5 && (stiff - r2).abs() < 1e-5);

    let (plane, curvature) = dimer_plane();
    assert!(plane.point()[0].abs() < 1e-6 && plane.point()[1].abs() < 1e-6);
    assert!((curvature - unstable).abs() < 1e-3 * unstable.abs());
    let along = (angle_of(&plane) - saddle_angle()).cos();
    assert!(1.0 - along < 1e-6, "normal off the mode by {along}");
}

/// The first-order coefficients from Hessian actions and gradients match
/// the contractions of the closed-form third and fourth derivatives in
/// the Hessian eigenbases (SymPy): `A‡ = 0.27552572`, `A_R = 0.09307962`,
/// `S = 2.05852212`, `c₁ = -0.03181767`. At `kT = 0.02` quadrature puts
/// the optimal line 0.008252 from the saddle with
/// `ln(k/k_HTST) = -0.0006827`, `+0.0037572` for the saddle plane; the
/// first-order terms, `kT S / (2|λ_u|)`, `c₁ kT` and `(A‡ - A_R) kT`, are
/// within `0.5 kT²` of those.
///
/// The plane is the closed-form saddle plane: `A‡` and `S` change at first
/// order with its normal, by about 1.1 and 1.3 per radian here, so a
/// dimer normal 7e-4 rad off moves them by 1e-3.
#[test]
fn skewed_well_first_order_correction() {
    let normal = unit(saddle_angle());
    let plane = Hyperplane::new(array![0.0, 0.0], array![normal[0], normal[1]]).unwrap();
    let config = AnharmonicConfig::default().with_probes(4000).with_seed(2);
    let reactant = array![REACTANT[0], REACTANT[1]];
    let result = anharmonic_correction(&SkewedWell, reactant.view(), &plane, &config).unwrap();
    let (unstable, _) = saddle_eigenvalues();
    assert!((result.curvature + unstable).abs() < 1e-5);
    for (name, value, error, exact) in [
        ("A‡", result.saddle, result.saddle_error, 0.27552572),
        ("A_R", result.reactant, result.reactant_error, 0.09307962),
        ("S", result.slope, result.slope_error, 2.05852212),
        (
            "c₁",
            result.coefficient,
            result.coefficient_error,
            -0.03181767,
        ),
    ] {
        assert!(
            (value - exact).abs() < 4.0 * error + 1.0e-6,
            "{name} {value} ± {error} against {exact}"
        );
        assert!(error < 0.01, "{name} error {error}");
    }

    let kt = 0.02;
    let (offset, theta, _) = optimal_line(kt);
    assert!((offset - 0.008252).abs() < 2e-6, "offset {offset}");
    let harmonic = log_harmonic_rate(kt);
    let optimal = log_rate(kt, offset, theta) - harmonic;
    let fixed = log_rate(kt, 0.0, saddle_angle()) - harmonic;
    assert!((optimal + 0.0006827).abs() < 2e-6, "ln k/k_HTST {optimal}");
    assert!((fixed - 0.0037572).abs() < 2e-6, "ln k_fix/k_HTST {fixed}");
    let bound = 0.5 * kt * kt;
    let (value, error) = result.log_ratio(kt);
    assert!((value - optimal).abs() < bound + 4.0 * error);
    let (value, error) = result.fixed_log_ratio(kt);
    assert!((value - fixed).abs() < bound + 4.0 * error);
    let spread = kt * result.slope_error / (2.0 * result.curvature);
    assert!((result.offset(kt) - offset).abs() < bound + 4.0 * spread);
}

/// At `kT = 0.4` the optimizer from the dimer's plane reaches the
/// quadrature maximum of `A` within four errors and resolves the turn
/// from the saddle plane, and the sampled rate matches the quadrature
/// rate of that line, `ln(k/k_HTST) = -0.0310`. The virial on the final
/// plane is `kT`, one free direction. The work along the optimizer's own
/// nine accepted steps, trapezoid on each, is 0.06295 ± 0.00392 by
/// integration and 0.05682 ± 0.00753 by BAR, against 0.06199 by
/// quadrature of its first and last lines.
#[test]
fn skewed_well_optimum_and_rate_at_high_temperature() {
    let kt = 0.4;
    let (offset, theta, curvature) = optimal_line(kt);
    assert!((offset - 0.13736).abs() < 2e-4, "offset {offset}");
    assert!(
        ((theta - saddle_angle()).to_degrees() + 7.359).abs() < 0.01,
        "turn {}",
        (theta - saddle_angle()).to_degrees()
    );
    let exact = log_rate(kt, offset, theta);
    assert!((exact - log_harmonic_rate(kt) + 0.0310).abs() < 2e-4);

    let (seed, _) = dimer_plane();
    let config = PlaneOptimizerConfig::default()
        .with_sampler(sampler(kt, 0.15).with_records(2000))
        .with_max_iterations(100);
    let optimum = PlaneOptimizer::new(&seed, config)
        .unwrap()
        .run(&SkewedWell)
        .unwrap();
    let last = optimum.trace.last().unwrap();
    let found = angle_of(&optimum.plane);
    let shift = optimum.plane.normal().dot(&optimum.plane.point());
    let angle_error = last.rotation_error / kt / curvature[1][1].abs();
    let offset_error = last.normal_force_error / kt / curvature[0][0].abs();
    assert!(
        (found - theta).abs() < 4.0 * angle_error,
        "{}° ± {}° against {}°",
        found.to_degrees(),
        angle_error.to_degrees(),
        theta.to_degrees()
    );
    assert!(
        (shift - offset).abs() < 4.0 * offset_error,
        "{shift} ± {offset_error} against {offset}"
    );
    assert!((found - saddle_angle()).abs() > 4.0 * angle_error);
    let sample = &optimum.sample;
    assert_eq!(sample.free_dimensions, 1);
    assert!(
        (sample.virial - kt).abs() < 4.0 * sample.virial_error,
        "virial {} ± {}",
        sample.virial,
        sample.virial_error
    );

    let trace = PlanePath::trace(&optimum.trace).unwrap();
    let climb_config = ReversibleWorkConfig::default()
        .with_sampler(sampler(kt, 0.15).with_records(2000))
        .with_nodes(2)
        .with_simpson(false);
    let mut climb_sampler = AnalyticSurfaceSampler::new(&SkewedWell, climb_config.sampler.clone());
    let climb = reversible_work_along(&mut climb_sampler, &trace, &climb_config).unwrap();
    let (from, to) = (trace.start().unwrap(), trace.end().unwrap());
    let exact_climb = free_energy([to.point()[0], to.point()[1]], angle_of(&to), kt)
        - free_energy([from.point()[0], from.point()[1]], angle_of(&from), kt);
    for (name, value, error) in [
        ("integration", climb.integrated_work, climb.integrated_error),
        ("BAR", climb.bar_work.unwrap(), climb.bar_error.unwrap()),
    ] {
        assert!(
            (value - exact_climb).abs() < 4.0 * error,
            "{name} along the trace {value} ± {error} against {exact_climb}"
        );
    }

    let optimal = Hyperplane::new(
        array![shift * found.cos(), shift * found.sin()],
        array![found.cos(), found.sin()],
    )
    .unwrap();
    let reactant = array![REACTANT[0], REACTANT[1]];
    let start = Hyperplane::new(reactant.clone(), array![1.0, 1.0]).unwrap();
    let density = reactant_density(
        &SkewedWell,
        &start,
        reactant.view(),
        std::slice::from_ref(&optimal),
        &sampler(kt, 0.1).with_records(16_000),
    )
    .unwrap();
    let config = ReversibleWorkConfig::default()
        .with_sampler(sampler(kt, 0.1).with_records(2000))
        .with_nodes(9);
    let mut work_sampler = AnalyticSurfaceSampler::new(&SkewedWell, config.sampler.clone())
        .with_start(reactant.clone());
    let work = reversible_work(&mut work_sampler, &[start, optimal], &config).unwrap();
    let exact_work = free_energy([shift * found.cos(), shift * found.sin()], found, kt)
        - free_energy(REACTANT, std::f64::consts::FRAC_PI_4, kt);
    let (bar, bar_error) = (work.bar_work.unwrap(), work.bar_error.unwrap());
    assert!(
        (bar - exact_work).abs() < 4.0 * bar_error,
        "BAR {bar} ± {bar_error} against {exact_work}"
    );
    assert!(
        (work.work - exact_work).abs() < 4.0 * work.error,
        "TI {} ± {} against {exact_work}",
        work.work,
        work.error
    );
    let rate = ohtst_rate(&density, &work).unwrap();
    let at_found = log_rate(kt, shift, found);
    assert!(
        (rate.log_rate - at_found).abs() < 4.0 * rate.log_rate_error,
        "ln k {} ± {} against {at_found}",
        rate.log_rate,
        rate.log_rate_error
    );
    assert!(rate.log_rate_error < 0.08);
}

/// Parallel planes along `(1, 1)` around the reactant at `kT = 0.1` give
/// `Z_H(0) / Z_R` within errors of quadrature, 1.396348 with `Z_R` taken
/// behind the parallel line through the saddle, `ξ = √2`. The side behind
/// ends on the cap; the side ahead turns over just past that line, where
/// the plane free energy peaks about 9 `kT` up, so the two regions differ
/// by a part in 10⁵. With 4000 records a plane, the spline gave
/// 1.3966 ± 0.0017, and 1.3900 ± 0.0080 without the control variates,
/// which cut the error of the mean force on the plane through the minimum
/// sevenfold; the quintic fit it replaces gave 1.3871 ± 0.0126 at a
/// reduced chi-square of 5.66, from the shoulder before the barrier.
#[test]
fn skewed_well_basin_density_against_quadrature() {
    let kt = 0.1;
    let normal = [1.0, 1.0];
    let exact = line_integral(SkewedWell::energy, REACTANT, normal, kt, 0.0, EXTENT)
        / half_plane_integral(SkewedWell::energy, [0.0, 0.0], normal, kt, 0.0, EXTENT);
    assert!((exact - 1.396348).abs() < 1e-6, "{exact}");
    let config = BasinConfig::default().with_sampler(sampler(kt, 0.1).with_records(4000));
    let reactant = array![REACTANT[0], REACTANT[1]];
    let result = basin_density(
        &SkewedWell,
        reactant.view(),
        array![normal[0], normal[1]].view(),
        &config,
    )
    .unwrap();
    assert_eq!(result.ends, [BasinEnd::Cap, BasinEnd::Turn]);
    let summit = result.window[1];
    assert!(summit > 2.0f64.sqrt() && summit < 1.6, "{summit}");
    assert!(
        (result.density - exact).abs()
            < 4.0 * result.density_error + result.quadrature_error * exact,
        "{} ± {} against {exact}",
        result.density,
        result.density_error
    );
    assert!(result.density_error < 0.003 * exact);
    assert!(result.quadrature_error < 0.5 * result.density_error / result.density);
    let centre = result.offsets.iter().position(|xi| *xi == 0.0).unwrap();
    let raw = result.uncontrolled_errors.as_ref().unwrap();
    let reduction = raw[centre] / result.slope_errors[centre];
    assert!(reduction > 4.0, "{reduction}");
}

/// `κ k_TST` does not depend on the dividing line. The barrier is 10 kT
/// from both wells at `kT = 0.1`, so with unit friction `κ(t)` is flat
/// from `t = 6` without commitment. Shots from exact draws on the optimal
/// line and on the line turned by 0.6 rad about its summit, whose
/// quadrature `ln k_TST` are -11.78081 and -10.32538, give `κ` = 0.71558 ±
/// 0.00655 and 0.16309 ± 0.00540 at the first crossing, so
/// `ln κ k_TST` = -12.11548 and -12.13886. End states from quenches into
/// the two minima give the same `κ` to the last digit.
#[test]
fn skewed_well_transmission_times_tst_rate_is_line_independent() {
    let kt = 0.1;
    let (offset, theta, _) = optimal_line(kt);
    let summit = off_saddle(offset, theta);
    let shots = 4096;
    let basins = Basins::new(vec![array![-1.0, -1.0]], vec![array![1.0, 1.0]]);
    let mut balance = Vec::new();
    for angle in [theta, theta + 0.6] {
        let log_tst = log_line_rate(kt, summit, angle);
        let starts = line_draws(
            SkewedWell::energy,
            summit,
            unit(angle),
            kt,
            EXTENT,
            shots,
            13,
        );
        let plane = Hyperplane::new(
            array![summit[0], summit[1]],
            array![angle.cos(), angle.sin()],
        )
        .unwrap();
        let run = |end_state: EndState| {
            let config = TransmissionConfig::default()
                .with_thermal(kt)
                .with_friction(1.0)
                .with_time_step(0.05)
                .with_steps(400)
                .with_check_every(40)
                .with_shots(shots)
                .with_blocks(32)
                .with_end_state(end_state);
            transmission_coefficient(
                &mut AnalyticSurfaceDynamics::new(&SkewedWell),
                &plane,
                &starts,
                &config,
            )
            .unwrap()
        };
        let side = run(EndState::Side);
        let quench = run(basins.clone().into());
        assert!(
            (quench.kappa - side.kappa).abs() < 4.0 * quench.error,
            "quench {} ± {} against side {} ± {}",
            quench.kappa,
            quench.error,
            side.kappa,
            side.error
        );
        assert_eq!(quench.unassigned, 0);
        balance.push((side.kappa.ln() + log_tst, side.error / side.kappa));
    }
    let (a, b) = (balance[0], balance[1]);
    assert!(
        (a.0 - b.0).abs() < 4.0 * a.1.hypot(b.1),
        "ln κ k_TST {} ± {} against {} ± {}",
        a.0,
        a.1,
        b.0,
        b.1
    );
}

/// The work from the line through the reactant minimum, normal to its
/// soft mode, to the saddle line along the PAFI planes of the
/// steepest-descent path at `kT = 0.4`, against quadrature of the two
/// lines, 0.87272. Over eight spline pieces integration gave
/// 0.87431 ± 0.00679 and MBAR across the 17 nodes with Gaussian maps
/// 0.87673 ± 0.00545; BAR gave 0.85134 ± 0.01730 with Gaussian maps and
/// 0.86340 ± 0.01547 with rigid ones. The planes are close enough that
/// the rigid map already holds the one free direction: neighbours overlap
/// at 0.482 or more with it and 0.469 or more with the Gaussian map.
#[test]
fn skewed_well_work_along_the_minimum_energy_path() {
    let kt = 0.4;
    let (saddle_plane, _) = dimer_plane();
    let (mode, pivot) = (saddle_plane.normal(), saddle_plane.point());
    let mut points = descent_path(
        SkewedWell::energy,
        [pivot[0], pivot[1]],
        [-mode[0], -mode[1]],
        0.2,
    );
    points.reverse();
    let reactant = points[0];
    assert!((reactant[0] - REACTANT[0]).hypot(reactant[1] - REACTANT[1]) < 1e-9);
    let (_, _, soft) = eigen2(hessian(SkewedWell::energy, reactant));
    let along = [points[1][0] - reactant[0], points[1][1] - reactant[1]];
    let sign = (soft[0] * along[0] + soft[1] * along[1]).signum();
    let start_normal = array![sign * soft[0], sign * soft[1]];
    let points: Vec<Array1<f64>> = points.iter().map(|p| array![p[0], p[1]]).collect();
    let path = PlanePath::curve(&points, Some(start_normal.view()), Some(mode)).unwrap();
    let (first, last) = (path.start().unwrap(), path.end().unwrap());
    let exact = free_energy([pivot[0], pivot[1]], angle_of(&last), kt)
        - free_energy(REACTANT, angle_of(&first), kt);
    let config = ReversibleWorkConfig::default()
        .with_sampler(sampler(kt, 0.15).with_records(2000))
        .with_nodes(3);
    let mut engine = AnalyticSurfaceSampler::new(&SkewedWell, config.sampler.clone());
    let rigid = reversible_work_along(&mut engine, &path, &config).unwrap();
    let config = config
        .with_transport(GaussianTransport::default().into())
        .with_mbar(true)
        .with_estimate(WorkEstimate::Mbar);
    let mut engine = AnalyticSurfaceSampler::new(&SkewedWell, config.sampler.clone());
    let work = reversible_work_along(&mut engine, &path, &config).unwrap();
    let estimates = [
        ("integration", work.integrated_work, work.integrated_error),
        ("BAR", work.bar_work.unwrap(), work.bar_error.unwrap()),
        ("MBAR", work.mbar_work.unwrap(), work.mbar_error.unwrap()),
        (
            "rigid BAR",
            rigid.bar_work.unwrap(),
            rigid.bar_error.unwrap(),
        ),
    ];
    for (name, value, error) in estimates {
        assert!(
            (value - exact).abs() < 4.0 * error,
            "{name} {value} ± {error} against {exact}"
        );
    }
    assert_eq!(work.work, work.mbar_work.unwrap());
    for result in [&work, &rigid] {
        assert!(result.warnings.is_empty(), "{:?}", result.warnings);
        assert!(
            result
                .nodes
                .iter()
                .skip(1)
                .all(|n| n.overlap.unwrap() > 0.4)
        );
    }
}
