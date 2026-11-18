//! Analytical, statistical, engine-coupled Kramers, and property tests for
//! `rgflux::kappa::bennett_chandler`.

use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2, ArrayView1};
use proptest::prelude::*;
use rgflux::FluxError;
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::engine::{DynamicsEngine, HyperplaneConstraint, ThermostatConfig};
use rgflux::kappa::bennett_chandler::{
    BennettChandlerConfig, ReactiveTrajectory, ShootingConfig, Side, estimate_bennett_chandler,
    shoot_bennett_chandler,
};
use rgflux::kappa::kramers_transmission_coefficient;
use rgflux::stats::StatsConfig;

/// Algebraic rounding tolerance `32 * f64::EPSILON` (`2^-47`) for closed-form checks.
const ALGEBRAIC_TOL: f64 = 32.0 * f64::EPSILON;

/// Canonical statistical confidence multiplier (`3 sigma`) for the 2D double-well Kramers test.
const KRAMERS_SIGMA_BOUND: f64 = 3.0;

#[test]
fn typed_errors_for_empty_no_positive_flux_and_non_stationary_plateau() {
    let cfg = BennettChandlerConfig::default();

    // 1. Empty trajectory list.
    assert_eq!(
        estimate_bennett_chandler(&[], &cfg),
        Err(FluxError::EmptyInput(
            "no reactive-flux trajectories supplied"
        ))
    );

    // 1b. Empty side history inside ReactiveTrajectory::new.
    assert_eq!(
        ReactiveTrajectory::new(1.0, vec![]),
        Err(FluxError::EmptyInput(
            "ReactiveTrajectory::sides must not be empty"
        ))
    );

    // 2. No positive-flux trajectories (all v_n(0) <= 0).
    let non_positive = vec![
        ReactiveTrajectory::new(0.0, vec![Side::Reactant; 16]).expect("valid"),
        ReactiveTrajectory::new(-0.8, vec![Side::Reactant; 16]).expect("valid"),
        ReactiveTrajectory::new(-2.1, vec![Side::Product; 16]).expect("valid"),
    ];
    assert_eq!(
        estimate_bennett_chandler(&non_positive, &cfg),
        Err(FluxError::NoPositiveFlux)
    );

    // 3. Non-stationary plateau: trajectories recross progressively throughout
    // the plateau window [12, 24), producing a monotonic drop in kappa(t).
    let n_steps = 24_usize;
    let mut decaying_ensemble = Vec::new();
    for idx in 0..24_usize {
        let cross_step = 12 + (idx % 12);
        let sides: Vec<Side> = (0..n_steps)
            .map(|t| {
                if t < cross_step {
                    Side::Product
                } else {
                    Side::Reactant
                }
            })
            .collect();
        decaying_ensemble.push(ReactiveTrajectory::new(1.0, sides).expect("valid"));
    }
    let err = estimate_bennett_chandler(&decaying_ensemble, &cfg)
        .expect_err("decaying window must fail stationarity");
    match err {
        FluxError::NonStationaryPlateau {
            z_score,
            max_z_score,
            drift_slope,
        } => {
            assert!(z_score > max_z_score);
            assert!(drift_slope < 0.0);
        }
        other => panic!("expected NonStationaryPlateau, got {other:?}"),
    }
}

#[test]
fn no_recrossing_gives_exact_unit_kappa() {
    let n_steps = 20_usize;
    let trajectories = vec![
        ReactiveTrajectory::new(0.5, vec![Side::Product; n_steps]).expect("t0"),
        ReactiveTrajectory::new(-1.3, vec![Side::Reactant; n_steps]).expect("t1"),
        ReactiveTrajectory::new(2.2, vec![Side::Product; n_steps]).expect("t2"),
        ReactiveTrajectory::new(-0.4, vec![Side::Reactant; n_steps]).expect("t3"),
        ReactiveTrajectory::new(1.1, vec![Side::Product; n_steps]).expect("t4"),
        ReactiveTrajectory::new(0.0, vec![Side::Reactant; n_steps]).expect("t5"),
    ];

    let est = estimate_bennett_chandler(&trajectories, &BennettChandlerConfig::default())
        .expect("stationary unit plateau");
    assert_eq!(est.positive_flux_count, 3);
    for &kt in &est.kappa_t {
        assert_abs_diff_eq!(kt, 1.0, epsilon = ALGEBRAIC_TOL);
    }
    assert_abs_diff_eq!(est.plateau, 1.0, epsilon = ALGEBRAIC_TOL);
    assert_eq!(est.standard_error, 0.0);
    assert!(est.stationarity.is_stationary);
}

#[test]
fn synthetic_recrossing_set_reproduces_exact_analytic_kappa_and_unclamped_sign() {
    // Construct an ensemble with positive initial flux sum D = 1 + 2 + 3 + 4 = 10:
    // - v0 = +1.0: stays Product (contributes +1.0 at plateau)
    // - v0 = +2.0: starts Product for t < 4, recrosses to Reactant at t >= 4 (contributes 0.0)
    // - v0 = +3.0: stays Product (contributes +3.0 at plateau)
    // - v0 = +4.0: starts Product for t < 6, recrosses to Reactant at t >= 6, then returns to Product at t >= 7 (contributes +4.0)
    // - v0 = -2.0: starts Reactant for t < 5, crosses into Product at t >= 5 (contributes -2.0 at plateau)
    //
    // Analytic kappa(t):
    // - t in 0..4: (1 + 2 + 3 + 4 + 0) / 10 = 1.0
    // - t == 4:    (1 + 0 + 3 + 4 + 0) / 10 = 0.8
    // - t == 5:    (1 + 0 + 3 + 4 - 2) / 10 = 0.6
    // - t == 6:    (1 + 0 + 3 + 0 - 2) / 10 = 0.2
    // - t in 7..20 (including plateau [10, 20)): (1 + 0 + 3 + 4 - 2) / 10 = 6 / 10 = 0.6.
    let n_steps = 20_usize;
    let step_sides =
        |pred: &dyn Fn(usize) -> Side| -> Vec<Side> { (0..n_steps).map(pred).collect() };

    let trajectories = vec![
        ReactiveTrajectory::new(1.0, vec![Side::Product; n_steps]).expect("t0"),
        ReactiveTrajectory::new(
            2.0,
            step_sides(&|t| if t < 4 { Side::Product } else { Side::Reactant }),
        )
        .expect("t1"),
        ReactiveTrajectory::new(3.0, vec![Side::Product; n_steps]).expect("t2"),
        ReactiveTrajectory::new(
            4.0,
            step_sides(&|t| {
                if t == 6 {
                    Side::Reactant
                } else {
                    Side::Product
                }
            }),
        )
        .expect("t3"),
        ReactiveTrajectory::new(
            -2.0,
            step_sides(&|t| if t < 5 { Side::Reactant } else { Side::Product }),
        )
        .expect("t4"),
    ];

    let cfg =
        BennettChandlerConfig::default().with_stats(StatsConfig::default().with_batch_size(1));
    let est = estimate_bennett_chandler(&trajectories, &cfg).expect("synthetic recrossing");

    assert_eq!(est.positive_flux_count, 4);
    assert_eq!(est.plateau_window, (10, 20));
    assert_abs_diff_eq!(est.kappa_t[0], 1.0, epsilon = ALGEBRAIC_TOL);
    assert_abs_diff_eq!(est.kappa_t[4], 0.8, epsilon = ALGEBRAIC_TOL);
    assert_abs_diff_eq!(est.kappa_t[5], 0.6, epsilon = ALGEBRAIC_TOL);
    assert_abs_diff_eq!(est.kappa_t[6], 0.2, epsilon = ALGEBRAIC_TOL);
    assert_abs_diff_eq!(est.plateau, 0.6, epsilon = ALGEBRAIC_TOL);
    assert!(est.standard_error > 0.0 && est.standard_error.is_finite());

    // Also verify negative plateau values are NOT clamped into [0, 1].
    let unclamped = vec![
        ReactiveTrajectory::new(1.0, vec![Side::Reactant; 16]).expect("pos recrossed"),
        ReactiveTrajectory::new(-1.5, vec![Side::Product; 16]).expect("neg crossed"),
    ];
    let neg_est = estimate_bennett_chandler(&unclamped, &BennettChandlerConfig::default())
        .expect("negative plateau still stationary");
    assert_abs_diff_eq!(neg_est.plateau, -1.5, epsilon = ALGEBRAIC_TOL);
}

#[test]
fn inverted_barrier_small_system_has_unit_transmission() {
    let speeds = [
        0.2_f64, 0.6, 0.9, 1.4, 2.1, 3.0, 0.35, 1.8, 2.6, 0.75, 1.1, 1.9, 2.8, 0.5, 1.55, 2.25,
    ];
    let trajectories = speeds
        .iter()
        .copied()
        .map(|v| ReactiveTrajectory::new(v, vec![Side::Product; 24]).expect("trajectory"))
        .collect::<Vec<_>>();
    let cfg = BennettChandlerConfig::default()
        .with_plateau_start_fraction(0.25)
        .with_plateau_end_fraction(1.0)
        .with_stats(StatsConfig::default().with_batch_size(1));
    let est = estimate_bennett_chandler(&trajectories, &cfg).expect("estimate");
    let expected = 1.0;
    assert_abs_diff_eq!(est.kappa_t[23], expected, epsilon = ALGEBRAIC_TOL);
    assert_abs_diff_eq!(est.plateau, 1.0, epsilon = ALGEBRAIC_TOL);
    assert_eq!(est.standard_error, 0.0);
}

#[test]
fn harmonic_recrossing_small_system_matches_closed_form_plateau() {
    let n_steps = 24_usize;
    let speeds = [
        0.25_f64, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 0.33, 0.67, 1.33,
        2.67,
    ];
    let trajectories = speeds
        .iter()
        .copied()
        .map(|v| {
            let mut sides = vec![Side::Product; n_steps];
            for side in sides.iter_mut().skip(12) {
                *side = Side::Reactant;
            }
            ReactiveTrajectory::new(v, sides).expect("trajectory")
        })
        .collect::<Vec<_>>();
    let cfg = BennettChandlerConfig::default()
        .with_plateau_start_fraction(0.5)
        .with_plateau_end_fraction(1.0)
        .with_stats(StatsConfig::default().with_batch_size(1));
    let est = estimate_bennett_chandler(&trajectories, &cfg).expect("estimate");
    let expected = 0.0;
    assert_abs_diff_eq!(est.kappa_t[0], 1.0, epsilon = ALGEBRAIC_TOL);
    assert_abs_diff_eq!(est.kappa_t[23], expected, epsilon = ALGEBRAIC_TOL);
    assert_abs_diff_eq!(est.plateau, expected, epsilon = ALGEBRAIC_TOL);
}

#[test]
fn double_well_2d_native_shooting_matches_high_friction_kramers_within_se() {
    // 2D double well: V(x, y) = V0 (x^2 - 1)^2 + (1/2) k_y y^2.
    // At the saddle (0, 0) with dividing surface x = 0 (unit normal n = (1, 0)):
    // V_xx(0, y) = -4 * V0, so for unit mass m = 1, omega_b = sqrt(4 * V0) = 2.
    // At high friction gamma = 6.0 (alpha = gamma / (2 * omega_b) = 1.5),
    // Kramers' exact transmission coefficient is sqrt(1 + 1.5^2) - 1.5 = sqrt(3.25) - 1.5 ~ 0.3027756.
    struct SeparableDoubleWell2D {
        barrier: f64,
        transverse_k: f64,
    }

    impl rgsaddle::PointSurface for SeparableDoubleWell2D {
        fn eval(&self, r: ArrayView1<f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
            let (x, y) = (r[0], r[1]);
            let dx = x * x - 1.0;
            let energy = self.barrier * dx * dx + 0.5 * self.transverse_k * y * y;
            let gx = 4.0 * self.barrier * x * dx;
            let gy = self.transverse_k * y;
            Ok((energy, Array1::from_vec(vec![gx, gy])))
        }
    }

    let barrier = 1.0_f64;
    let transverse_k = 2.0_f64;
    let temperature = 0.08_f64;
    let gamma = 6.0_f64;
    let omega_b = (4.0 * barrier).sqrt();
    let kramers_exact = kramers_transmission_coefficient(gamma, omega_b).expect("kramers");

    let dividing_surface =
        HyperplaneConstraint::new(Array1::from_vec(vec![1.0, 0.0]), 0.0).expect("x = 0 plane");
    let native_cfg = NativeConfig::default()
        .with_masses(Array1::from_vec(vec![1.0, 1.0]))
        .with_thermostat(
            ThermostatConfig::default()
                .with_gamma(gamma)
                .with_temperature(temperature),
        )
        .with_constraint(dividing_surface.clone())
        .with_seed(20260518);

    let mut engine = NativeEngine::new(
        SeparableDoubleWell2D {
            barrier,
            transverse_k,
        },
        native_cfg,
    )
    .expect("native engine");

    // 1. Thermalize on the dividing hyperplane x = 0 via constrained BAOAB+RATTLE
    // and collect dividing-surface configurations.
    engine.shoot_velocities(20260518).expect("initial velocity");
    let sample_dt = 0.02;
    for _ in 0..400 {
        engine.step(sample_dt).expect("burn-in on dividing plane");
    }
    let n_configs = 600_usize;
    let mut surface_configs = Array2::zeros((n_configs, 2));
    for row_idx in 0..n_configs {
        for _ in 0..10 {
            engine.step(sample_dt).expect("plane stride");
        }
        let pos = engine.positions();
        surface_configs[(row_idx, 0)] = pos[0];
        surface_configs[(row_idx, 1)] = pos[1];
    }

    // 2. Shoot antithetic reactive-flux trajectories through `shoot_bennett_chandler`
    // (total flight time 300 * 0.05 = 15.0 >> 1 / lambda_+ ~ 1.65; plateau window [7.5, 15.0]).
    let shoot_cfg = ShootingConfig::default()
        .with_steps(300)
        .with_time_step(0.05)
        .with_antithetic(true)
        .with_seed(918273);

    let est = shoot_bennett_chandler(
        &mut engine,
        &dividing_surface,
        surface_configs.view(),
        &shoot_cfg,
    )
    .expect("kramers reactive-flux plateau");

    // Confirm `shoot_bennett_chandler` restored the engine's hyperplane constraint.
    assert!(engine.constraint().is_some());
    assert_eq!(est.positive_flux_count, n_configs);
    assert!(est.standard_error > 0.0 && est.standard_error < 0.05);
    assert!(
        (est.plateau - kramers_exact).abs() <= KRAMERS_SIGMA_BOUND * est.standard_error,
        "estimated kappa = {:.5} +/- {:.5} differs from Kramers {:.5} by {:.2} SE",
        est.plateau,
        est.standard_error,
        kramers_exact,
        (est.plateau - kramers_exact).abs() / est.standard_error
    );
}

proptest! {
    #[test]
    fn positive_flux_ensembles_satisfy_bounds_scale_invariance_and_complementarity(
        raw_vels in prop::collection::vec(0.05_f64..10.0_f64, 8..40),
        committed_flags in prop::collection::vec(any::<bool>(), 8..40),
        scale in 0.1_f64..25.0_f64,
    ) {
        let count = raw_vels.len().min(committed_flags.len());
        let n_steps = 16_usize;
        let mut trajs = Vec::with_capacity(count);
        let mut scaled_trajs = Vec::with_capacity(count);
        let mut flipped_trajs = Vec::with_capacity(count);

        for i in 0..count {
            let v = raw_vels[i];
            let end_side = if committed_flags[i] { Side::Product } else { Side::Reactant };
            let comp_side = if committed_flags[i] { Side::Reactant } else { Side::Product };

            let mut sides = vec![Side::Product; n_steps];
            let mut comp_sides = vec![Side::Product; n_steps];
            for t in 4..n_steps {
                sides[t] = end_side;
                comp_sides[t] = comp_side;
            }
            trajs.push(ReactiveTrajectory::new(v, sides.clone()).expect("traj"));
            scaled_trajs.push(ReactiveTrajectory::new(scale * v, sides).expect("scaled"));
            flipped_trajs.push(ReactiveTrajectory::new(v, comp_sides).expect("flipped"));
        }

        let cfg = BennettChandlerConfig::default();
        let est = estimate_bennett_chandler(&trajs, &cfg).expect("est");
        let est_scaled = estimate_bennett_chandler(&scaled_trajs, &cfg).expect("est_scaled");
        let est_flipped = estimate_bennett_chandler(&flipped_trajs, &cfg).expect("est_flipped");

        // 1. Positive-flux-only ensembles stay in [0, 1].
        prop_assert!(est.plateau >= -ALGEBRAIC_TOL && est.plateau <= 1.0 + ALGEBRAIC_TOL);

        // 2. Scaling all initial velocities by `scale > 0` leaves kappa(t), plateau, and SE invariant.
        prop_assert!((est.plateau - est_scaled.plateau).abs() <= ALGEBRAIC_TOL);
        prop_assert!((est.standard_error - est_scaled.standard_error).abs() <= ALGEBRAIC_TOL);
        for (a, b) in est.kappa_t.iter().zip(&est_scaled.kappa_t) {
            prop_assert!((a - b).abs() <= ALGEBRAIC_TOL);
        }

        // 3. Complementing product/reactant commitment maps plateau -> 1 - plateau.
        prop_assert!((est.plateau + est_flipped.plateau - 1.0).abs() <= ALGEBRAIC_TOL);

        // 4. Reversing trajectory order preserves kappa(t) and plateau.
        let mut reversed = trajs;
        reversed.reverse();
        let est_rev = estimate_bennett_chandler(&reversed, &cfg).expect("est_rev");
        prop_assert!((est.plateau - est_rev.plateau).abs() <= ALGEBRAIC_TOL);
    }
}
