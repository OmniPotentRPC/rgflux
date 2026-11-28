//! Exact-reference validation of PAFI and OH-TST on surfaces where
//! quadrature and Fokker-Planck references are computable.

use ndarray::{Array1, array};
use rgflux::ohtst::{
    AnalyticSurfaceDynamics, AnalyticSurfaceSampler, Counting, EndState, Hyperplane,
    LangevinConfig, PlaneOptimizer, PlaneOptimizerConfig, ReversibleWorkConfig, TransmissionConfig,
    ohtst_rate, reactant_density, reversible_work, transmission_coefficient,
};
use rgsaddle::{ArcPath, PafiConfig, pafi_profile};

include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/common/ohtst_pafi_exact_refs.rs"
));

use ohtst_pafi_exact_refs::{
    CURVED_BEND, CURVED_TILT, CurvedWell2D, DoubleWell1D, MullerBrown2D, basin_line_free_energy,
    descend, descent_path, eigen2, harmonic_log_rate_2d, line_free_energy, line_log_tst_rate,
    log_tst_rate_1d, optimize_line, overdamped_mfpt_rate_1d, profile_gradient, stationary, unit,
};

fn argmax(values: &[f64]) -> usize {
    let mut best = 0usize;
    for i in 1..values.len() {
        if values[i] > values[best] {
            best = i;
        }
    }
    best
}

fn pafi_config(kt: f64, samples: usize, seed: u64) -> PafiConfig {
    PafiConfig::default()
        .with_thermal(kt)
        .with_mobility(1.0)
        .with_time_step(0.004)
        .with_thermalize(400)
        .with_stride(6)
        .with_samples(samples)
        .with_blocks(8)
        .with_seed(seed)
        .with_fix_com(false)
        .with_retain(16)
        .with_target_error(None)
        .with_max_rounds(1)
}

fn sampler(kt: f64, records: usize, step: f64, seed: u64) -> LangevinConfig {
    LangevinConfig::default()
        .with_thermal(kt)
        .with_time_step(step)
        .with_friction(1.0)
        .with_burn_in(300)
        .with_stride(5)
        .with_records(records)
        .with_seed(seed)
}

/// Langevin sampler for Müller-Brown. The stiffest curvature is 4068
/// (`ω = 64`), so the `h = 0.05` of the double well is past the BAOAB
/// stability bound `2/ω` and the walk diverges. This step is `ω h = 0.26`,
/// and the friction is raised to decorrelate the soft mode (`λ = 410`) in
/// fewer steps.
fn mb_sampler(kt: f64, records: usize, seed: u64) -> LangevinConfig {
    sampler(kt, records, MB_LANGEVIN_STEP, seed)
        .with_friction(MB_FRICTION)
        .with_burn_in(1_000)
        .with_stride(25)
}

fn line_from_path(path: &ArcPath, arc: f64) -> ([f64; 2], f64) {
    let geometry = path.geometry(arc).expect("path geometry");
    let point = [geometry.positions[0], geometry.positions[1]];
    let theta = geometry.tangent[1].atan2(geometry.tangent[0]);
    (point, theta)
}

fn orient_line(offset: f64, theta: f64, toward: [f64; 2]) -> (f64, f64) {
    let normal = unit(theta);
    if normal[0] * toward[0] + normal[1] * toward[1] >= 0.0 {
        (offset, theta)
    } else {
        (-(offset), theta + std::f64::consts::PI)
    }
}

fn exact_profile_on_path(
    path: &ArcPath,
    arcs: &[f64],
    kt: f64,
    energy: fn(f64, f64) -> (f64, f64, f64),
    shift: f64,
    extent: f64,
    in_basin: bool,
) -> Vec<f64> {
    let line_free = if in_basin {
        basin_line_free_energy
    } else {
        line_free_energy
    };
    let mut free: Vec<f64> = arcs
        .iter()
        .map(|arc| {
            let (point, theta) = line_from_path(path, *arc);
            line_free(energy, point, unit(theta), kt, shift, extent)
        })
        .collect();
    let base = free[0];
    for value in &mut free {
        *value -= base;
    }
    free
}

/// `dF/ds` of the line free energy at each arc by a central difference
/// over `±delta`, one-sided where that leaves the path. The spacing of
/// the arcs is too wide for a difference of the profile itself.
fn exact_gradient_on_path(
    path: &ArcPath,
    arcs: &[f64],
    kt: f64,
    energy: fn(f64, f64) -> (f64, f64, f64),
    shift: f64,
    extent: f64,
) -> Vec<f64> {
    let delta = 1.0e-4;
    let (first, last) = path.bounds();
    let free = |arc: f64| {
        let (point, theta) = line_from_path(path, arc);
        basin_line_free_energy(energy, point, unit(theta), kt, shift, extent)
    };
    arcs.iter()
        .map(|arc| {
            let upper = (arc + delta).min(last);
            let lower = (arc - delta).max(first);
            (free(upper) - free(lower)) / (upper - lower)
        })
        .collect()
}

fn assert_profile_matches_exact(
    profile_arc: &[f64],
    gradient: &[f64],
    gradient_error: &[f64],
    barrier: f64,
    barrier_error: f64,
    exact_free: &[f64],
    exact_gradient: &[f64],
) {
    for i in 1..gradient.len() - 1 {
        let span = profile_arc[i + 1] - profile_arc[i - 1];
        let curvature = (exact_gradient[i + 1] - exact_gradient[i - 1]).abs() / span;
        let quadrature = 0.5 * curvature * span;
        let miss = (gradient[i] - exact_gradient[i]).abs();
        assert!(
            miss < 4.0 * gradient_error[i] + quadrature,
            "gradient[{i}] {:+.6e} ± {:.3e} against {:+.6e}",
            gradient[i],
            gradient_error[i],
            exact_gradient[i]
        );
    }
    let dividing = argmax(exact_free);
    let exact_barrier = exact_free[dividing];
    let max_step = profile_arc
        .windows(2)
        .map(|window| window[1] - window[0])
        .fold(0.0, f64::max);
    let max_grad = exact_gradient.iter().map(|g| g.abs()).fold(0.0, f64::max);
    let discretization = max_step * max_grad;
    let miss = (barrier - exact_barrier).abs();
    assert!(
        miss < 4.0 * barrier_error + discretization,
        "barrier {barrier:.6e} ± {barrier_error:.3e} against {exact_barrier:.6e}"
    );
}

#[test]
fn double_well_exact_profile_quick_temperatures() {
    let points: Vec<Array1<f64>> = (0..21)
        .map(|i| array![-1.0 + 2.0 * i as f64 / 20.0])
        .collect();
    let path = ArcPath::from_positions(&points).expect("double-well path");
    let arcs = path.arcs();

    for (index, kt) in [0.04, 0.07].into_iter().enumerate() {
        let config = pafi_config(kt, 120, 100 + index as u64)
            .with_thermalize(120)
            .with_stride(4);
        let (profile, _) =
            pafi_profile(&DoubleWell1D, &path, &arcs, &config).expect("PAFI profile");
        let exact_free: Vec<f64> = arcs
            .iter()
            .map(|arc| {
                let x = path.geometry(*arc).expect("geometry").positions[0];
                DoubleWell1D::energy(x).0 - DoubleWell1D::energy(-1.0).0
            })
            .collect();
        let exact_gradient: Vec<f64> = arcs
            .iter()
            .map(|arc| {
                let x = path.geometry(*arc).expect("geometry").positions[0];
                DoubleWell1D::energy(x).1
            })
            .collect();
        assert_profile_matches_exact(
            &profile.arc,
            &profile.gradient,
            &profile.gradient_error,
            profile.barrier,
            profile.barrier_error,
            &exact_free,
            &exact_gradient,
        );
    }
}

#[test]
#[ignore]
fn double_well_exact_profile_and_rate() {
    let points: Vec<Array1<f64>> = (0..25)
        .map(|i| array![-1.0 + 2.0 * i as f64 / 24.0])
        .collect();
    let path = ArcPath::from_positions(&points).expect("double-well path");
    let arcs = path.arcs();

    let kt = 0.06;
    let config = pafi_config(kt, 220, 11);
    let (profile, _) = pafi_profile(&DoubleWell1D, &path, &arcs, &config).expect("PAFI profile");
    let exact_free: Vec<f64> = arcs
        .iter()
        .map(|arc| {
            let x = path.geometry(*arc).expect("geometry").positions[0];
            DoubleWell1D::energy(x).0 - DoubleWell1D::energy(-1.0).0
        })
        .collect();
    let exact_gradient: Vec<f64> = arcs
        .iter()
        .map(|arc| {
            let x = path.geometry(*arc).expect("geometry").positions[0];
            DoubleWell1D::energy(x).1
        })
        .collect();
    assert_profile_matches_exact(
        &profile.arc,
        &profile.gradient,
        &profile.gradient_error,
        profile.barrier,
        profile.barrier_error,
        &exact_free,
        &exact_gradient,
    );

    let reactant = Hyperplane::new(array![-1.0], array![1.0]).expect("reactant plane");
    let dividing = Hyperplane::new(array![0.0], array![1.0]).expect("dividing plane");
    let density = reactant_density(
        &DoubleWell1D,
        &reactant,
        array![-1.0].view(),
        std::slice::from_ref(&dividing),
        &sampler(kt, 1_800, 0.02, 21),
    )
    .expect("reactant density");
    let work_config = ReversibleWorkConfig::default()
        .with_sampler(sampler(kt, 500, 0.02, 32))
        .with_nodes(7);
    let mut work_sampler = AnalyticSurfaceSampler::new(&DoubleWell1D, work_config.sampler.clone())
        .with_start(array![-1.0]);
    let work = reversible_work(&mut work_sampler, &[reactant, dividing], &work_config)
        .expect("reversible work");
    let rate = ohtst_rate(&density, &work).expect("OH-TST rate");
    let exact_log_rate = log_tst_rate_1d(|x| DoubleWell1D::energy(x).0, kt, 0.0, -3.0);
    assert!(
        (rate.log_rate - exact_log_rate).abs() < 4.0 * rate.log_rate_error,
        "ln k {} ± {} against {} at kT={kt}",
        rate.log_rate,
        rate.log_rate_error,
        exact_log_rate
    );
}

#[test]
fn curved_well_chord_is_wrong_but_optimal_plane_recovers_the_rate() {
    let kt = 0.08;
    let reactant = [-1.0, CurvedWell2D::valley(-1.0)];
    let product = [1.0, CurvedWell2D::valley(1.0)];
    let saddle = [0.0, 0.0];
    let saddle_energy = CurvedWell2D::energy(saddle[0], saddle[1]).0;

    let chord_points: Vec<Array1<f64>> = (0..41)
        .map(|i| {
            let t = i as f64 / 40.0;
            array![
                reactant[0] + t * (product[0] - reactant[0]),
                reactant[1] + t * (product[1] - reactant[1])
            ]
        })
        .collect();
    let chord = ArcPath::from_positions(&chord_points).expect("chord path");
    let chord_arcs = chord.arcs();
    let exact_chord_free = exact_profile_on_path(
        &chord,
        &chord_arcs,
        kt,
        CurvedWell2D::energy,
        saddle_energy,
        8.0,
        false,
    );
    let exact_chord_gradient = profile_gradient(&chord_arcs, &exact_chord_free);
    let (chord_profile, _) = pafi_profile(
        &CurvedWell2D,
        &chord,
        &chord_arcs,
        &pafi_config(kt, 1_200, 77),
    )
    .expect("chord PAFI");
    assert_profile_matches_exact(
        &chord_profile.arc,
        &chord_profile.gradient,
        &chord_profile.gradient_error,
        chord_profile.barrier,
        chord_profile.barrier_error,
        &exact_chord_free,
        &exact_chord_gradient,
    );

    let (_, _, unstable_mode) = eigen2(CurvedWell2D::hessian(0.0, 0.0));
    let unstable_theta = unstable_mode[1].atan2(unstable_mode[0]);
    let (offset_raw, theta_raw, curvature) = optimize_line(
        CurvedWell2D::energy,
        saddle,
        unstable_theta,
        kt,
        saddle_energy,
        8.0,
    );
    let guide = [product[0] - reactant[0], product[1] - reactant[1]];
    let (offset, theta) = orient_line(offset_raw, theta_raw, guide);
    let exact_point = [
        saddle[0] + offset * theta.cos(),
        saddle[1] + offset * theta.sin(),
    ];
    let exact_log = line_log_tst_rate(
        CurvedWell2D::energy,
        exact_point,
        theta,
        kt,
        saddle_energy,
        8.0,
    );

    let seed = Hyperplane::new(array![0.0, 0.0], array![unstable_mode[0], unstable_mode[1]])
        .expect("seed plane")
        .oriented_toward(array![1.0, 1.0].view());
    let optimum = PlaneOptimizer::new(
        &seed,
        PlaneOptimizerConfig::default()
            .with_sampler(sampler(kt, 1_200, 0.1, 91))
            .with_max_iterations(80),
    )
    .expect("optimizer")
    .run(&CurvedWell2D)
    .expect("run optimizer");
    let last = optimum.trace.last().expect("trace");
    let (found_offset, found_theta) = {
        let normal = optimum.plane.normal();
        let point = optimum.plane.point();
        let theta = normal[1].atan2(normal[0]);
        let offset = normal.dot(&(&point - &array![saddle[0], saddle[1]]));
        orient_line(offset, theta, guide)
    };
    let angle_error = last.rotation_error / kt / curvature[1][1].abs();
    let offset_error = last.normal_force_error / kt / curvature[0][0].abs();
    assert!(
        (found_theta - theta).abs() < 4.0 * angle_error,
        "theta {found_theta} ± {angle_error} against {theta}"
    );
    assert!(
        (found_offset - offset).abs() < 4.0 * offset_error,
        "offset {found_offset} ± {offset_error} against {offset}"
    );

    let start_slope = CURVED_TILT + 3.0 * CURVED_BEND;
    let start = Hyperplane::new(array![reactant[0], reactant[1]], array![1.0, start_slope])
        .expect("reactant plane");
    let density = reactant_density(
        &CurvedWell2D,
        &start,
        array![reactant[0], reactant[1]].view(),
        std::slice::from_ref(&optimum.plane),
        &sampler(kt, 10_000, 0.08, 19),
    )
    .expect("density");
    let work_config = ReversibleWorkConfig::default()
        .with_sampler(sampler(kt, 1_600, 0.08, 29))
        .with_nodes(11);
    let mut work_sampler = AnalyticSurfaceSampler::new(&CurvedWell2D, work_config.sampler.clone())
        .with_start(array![reactant[0], reactant[1]]);
    let work = reversible_work(
        &mut work_sampler,
        &[start.clone(), optimum.plane.clone()],
        &work_config,
    )
    .expect("work");
    let rate = ohtst_rate(&density, &work).expect("OH-TST rate");
    assert!(
        (rate.log_rate - exact_log).abs() < 4.0 * rate.log_rate_error,
        "ln k {} ± {} against {}",
        rate.log_rate,
        rate.log_rate_error,
        exact_log
    );

    let chord_peak = argmax(&exact_chord_free);
    let (chord_point, chord_theta) = line_from_path(&chord, chord_arcs[chord_peak]);
    let chord_log = line_log_tst_rate(
        CurvedWell2D::energy,
        chord_point,
        chord_theta,
        kt,
        saddle_energy,
        8.0,
    );
    let chord_bias = (chord_log - exact_log).abs();
    let sampled_miss = (rate.log_rate - exact_log).abs();
    assert!(
        chord_bias > 4.0 * rate.log_rate_error + sampled_miss,
        "chord ln k {chord_log} too close to optimal {exact_log}"
    );
}

#[test]
fn curved_well_temperature_scaling_tends_to_harmonic_htst() {
    let reactant = [-1.0, CurvedWell2D::valley(-1.0)];
    let saddle = [0.0, 0.0];
    let barrier = CurvedWell2D::energy(saddle[0], saddle[1]).0
        - CurvedWell2D::energy(reactant[0], reactant[1]).0;
    let reactant_hessian = CurvedWell2D::hessian(reactant[0], reactant[1]);
    let saddle_hessian = CurvedWell2D::hessian(saddle[0], saddle[1]);
    let (_, _, unstable_mode) = eigen2(saddle_hessian);
    let unstable_theta = unstable_mode[1].atan2(unstable_mode[0]);
    let mut previous = f64::INFINITY;
    for kt in [0.12, 0.08, 0.05] {
        let (offset_raw, theta_raw, _) = optimize_line(
            CurvedWell2D::energy,
            saddle,
            unstable_theta,
            kt,
            CurvedWell2D::energy(saddle[0], saddle[1]).0,
            8.0,
        );
        let guide = [2.0, 2.0 * (CURVED_TILT + CURVED_BEND)];
        let (offset, theta) = orient_line(offset_raw, theta_raw, guide);
        let point = [
            saddle[0] + offset * theta.cos(),
            saddle[1] + offset * theta.sin(),
        ];
        let exact_log = line_log_tst_rate(
            CurvedWell2D::energy,
            point,
            theta,
            kt,
            CurvedWell2D::energy(saddle[0], saddle[1]).0,
            8.0,
        );
        let harmonic = harmonic_log_rate_2d(reactant_hessian, saddle_hessian, barrier, kt);
        let gap = (exact_log - harmonic).abs();
        assert!(
            gap < previous,
            "harmonic gap did not shrink: {gap} after {previous}"
        );
        previous = gap;
    }
}

/// High-cost Müller-Brown reference checks:
/// run in release with `--ignored`.
#[test]
#[ignore]
fn double_well_overdamped_transmission_matches_mfpt_reference() {
    let kt = 0.05;
    let dividing = Hyperplane::new(array![0.0], array![1.0]).expect("dividing plane");
    let friction = 14.0;
    let mobility = 1.0 / friction;
    let exact_log_tst = log_tst_rate_1d(|x| DoubleWell1D::energy(x).0, kt, 0.0, -3.0);
    let exact_tst = exact_log_tst.exp();
    let exact_mfpt =
        overdamped_mfpt_rate_1d(|x| DoubleWell1D::energy(x).0, kt, mobility, -3.0, -1.0, 0.0);
    // `exact_mfpt` is the rate of first reaching the dividing plane. A
    // trajectory on the plane returns to the reactant as often as it
    // commits to the product, so the reactive rate is half of it.
    let exact_kappa = 0.5 * exact_mfpt / exact_tst;
    let starts = vec![array![0.0]; 1_024];
    let transmission = transmission_coefficient(
        &mut AnalyticSurfaceDynamics::new(&DoubleWell1D),
        &dividing,
        &starts,
        &TransmissionConfig::default()
            .with_thermal(kt)
            .with_friction(friction)
            .with_time_step(0.01)
            .with_steps(1_200)
            .with_check_every(40)
            .with_shots(starts.len())
            .with_blocks(32)
            .with_counting(Counting::FirstCrossing)
            .with_end_state(EndState::Side),
    )
    .expect("transmission");
    let barrier_frequency = (-DoubleWell1D::hessian(0.0)).sqrt();
    let overdamped_remainder = barrier_frequency / friction;
    assert!(
        (transmission.kappa - exact_kappa).abs()
            < 4.0 * transmission.error + overdamped_remainder * exact_kappa,
        "kappa {} ± {} against overdamped {}",
        transmission.kappa,
        transmission.error,
        exact_kappa
    );
}

/// Overdamped time step for the Müller-Brown PAFI planes. The scheme is
/// stable for `μ dt λ < 2` with `λ` the stiffest curvature in the plane,
/// 4068 at the reactant, so the `dt` of 0.004 that suits the double well
/// (`λ = 8`) throws the walk out of the basin. This one gives
/// `μ dt λ = 0.81` there.
const MB_TIME_STEP: f64 = 2.0e-4;
const MB_LANGEVIN_STEP: f64 = 4.0e-3;
const MB_FRICTION: f64 = 10.0;

/// High-cost Müller-Brown reference checks:
/// run in release with `--ignored`.
#[test]
#[ignore]
fn mueller_brown_exact_profile_and_rate() {
    let saddle = stationary(MullerBrown2D::energy, [-0.82, 0.62]);
    let reactant = descend(MullerBrown2D::energy, [-0.56, 1.44]);
    let saddle_energy = MullerBrown2D::energy(saddle[0], saddle[1]).0;
    let (_, _, unstable_mode) = eigen2(MullerBrown2D::hessian(saddle));
    let mut points = descent_path(
        MullerBrown2D::energy,
        saddle,
        [-unstable_mode[0], -unstable_mode[1]],
        0.08,
    );
    points.reverse();
    if points
        .first()
        .map(|p| ((p[0] - reactant[0]).powi(2) + (p[1] - reactant[1]).powi(2)).sqrt() > 0.2)
        .unwrap_or(true)
    {
        points.insert(0, reactant);
    }
    let path_points: Vec<Array1<f64>> = points.iter().map(|p| array![p[0], p[1]]).collect();
    let path = ArcPath::from_positions(&path_points).expect("MB path");
    let arcs = path.arcs();

    for (seed, kt) in [(7u64, 6.0), (17u64, 9.0)] {
        let exact_free = exact_profile_on_path(
            &path,
            &arcs,
            kt,
            MullerBrown2D::energy,
            saddle_energy,
            4.0,
            true,
        );
        let exact_gradient =
            exact_gradient_on_path(&path, &arcs, kt, MullerBrown2D::energy, saddle_energy, 4.0);
        let config = pafi_config(kt, 20_000, seed).with_time_step(MB_TIME_STEP);
        let (profile, _) = pafi_profile(&MullerBrown2D, &path, &arcs, &config).expect("MB PAFI");
        assert_profile_matches_exact(
            &profile.arc,
            &profile.gradient,
            &profile.gradient_error,
            profile.barrier,
            profile.barrier_error,
            &exact_free,
            &exact_gradient,
        );

        let start = Hyperplane::new(
            array![reactant[0], reactant[1]],
            array![points[1][0] - reactant[0], points[1][1] - reactant[1]],
        )
        .expect("MB reactant plane");
        let (offset_raw, theta_raw, _) = optimize_line(
            MullerBrown2D::energy,
            saddle,
            unstable_mode[1].atan2(unstable_mode[0]),
            kt,
            saddle_energy,
            4.0,
        );
        let (offset, theta) = orient_line(offset_raw, theta_raw, [1.0, 0.0]);
        let exact_point = [
            saddle[0] + offset * theta.cos(),
            saddle[1] + offset * theta.sin(),
        ];
        let exact_log = line_log_tst_rate(
            MullerBrown2D::energy,
            exact_point,
            theta,
            kt,
            saddle_energy,
            4.0,
        );
        let seed_plane = Hyperplane::new(
            array![saddle[0], saddle[1]],
            array![unstable_mode[0], unstable_mode[1]],
        )
        .expect("MB seed")
        .oriented_toward(array![1.0, 1.0].view());
        let optimum = PlaneOptimizer::new(
            &seed_plane,
            PlaneOptimizerConfig::default()
                .with_sampler(mb_sampler(kt, 2_000, seed + 1_000))
                .with_max_iterations(120),
        )
        .expect("MB optimizer")
        .run(&MullerBrown2D)
        .expect("MB run");
        let density = reactant_density(
            &MullerBrown2D,
            &start,
            array![reactant[0], reactant[1]].view(),
            std::slice::from_ref(&optimum.plane),
            &mb_sampler(kt, 16_000, seed + 2_000),
        )
        .expect("MB density");
        let work_config = ReversibleWorkConfig::default()
            .with_sampler(mb_sampler(kt, 2_200, seed + 3_000))
            .with_nodes(11);
        let mut work_sampler =
            AnalyticSurfaceSampler::new(&MullerBrown2D, work_config.sampler.clone())
                .with_start(array![reactant[0], reactant[1]]);
        let work = reversible_work(
            &mut work_sampler,
            &[start.clone(), optimum.plane.clone()],
            &work_config,
        )
        .expect("MB work");
        let rate = ohtst_rate(&density, &work).expect("MB rate");
        assert!(
            (rate.log_rate - exact_log).abs() < 4.0 * rate.log_rate_error,
            "MB ln k {} ± {} against {}",
            rate.log_rate,
            rate.log_rate_error,
            exact_log
        );
    }
}
