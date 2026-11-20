//! Analytical, statistical, engine-coupled Kramers, and property tests for
//! `rgflux::kappa::bennett_chandler`.

use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2, ArrayView1};
use proptest::prelude::*;
use rgflux::FluxError;
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::engine::{
    DynamicsEngine, HyperplaneConstraint, StepReport, SurfaceVelocityPolicy, ThermostatConfig,
};
use rgflux::kappa::bennett_chandler::{
    BennettChandlerConfig, DEFAULT_COMMIT_CHECKPOINTS, KappaConfig, ReactiveTrajectory,
    ShootingConfig, Side, estimate_bennett_chandler, shoot_bennett_chandler,
    shoot_reactive_trajectories,
};
use rgflux::kappa::kramers_transmission_coefficient;
use rgflux::stats::StatsConfig;

/// Algebraic rounding tolerance `32 * f64::EPSILON` (`2^-47`) for closed-form checks.
/// Kappa configuration with early basin commitment switched on.
fn early(commit_checkpoints: usize) -> KappaConfig {
    KappaConfig::default()
        .with_early_commitment(true)
        .with_commit_checkpoints(commit_checkpoints)
}

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
    // At gamma = 6.0 and dt = 0.05, the barrier lingering time 1 / lambda_+ ~ 1.65 is 33 steps,
    // so both the default unconstrained shooting and an opt-in commitment window that
    // outlasts the barrier lingering time (commit_checkpoints = 64, t = 3.2) reproduce Kramers.
    for commit in [None, Some(64_usize)] {
        let mut shoot_cfg = ShootingConfig::default()
            .with_steps(300)
            .with_time_step(0.05)
            .with_antithetic(true)
            .with_seed(918273);
        if let Some(window) = commit {
            shoot_cfg = shoot_cfg.with_kappa(early(window));
        }

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
            "commit = {commit:?}: estimated kappa = {:.5} +/- {:.5} differs from Kramers {:.5} by {:.2} SE",
            est.plateau,
            est.standard_error,
            kramers_exact,
            (est.plateau - kramers_exact).abs() / est.standard_error
        );
    }
}

#[test]
fn double_well_1d_late_return_stays_at_committed_kappa_when_opted_in() {
    // 1D asymmetric double well with saddle at x = 0 (V(0) = 0, V'(0) = 0):
    // - Product basin (x >= 0): V(x) = 16 x^4 - 8 x^2 (minimum at x = 0.5, outer wall at x = 1 / sqrt(2)).
    //   A forward shot (v_x(0) > 0) enters x > 0 and stays on Side::Product for >= 35 steps,
    //   then reflects off the outer wall and returns across x = 0 into x < 0.
    // - Reactant basin (x < 0): V(x) = 0.0001 x^4 - 0.01 x^2 (wide basin, minimum at x = -5 * sqrt(2)).
    //   Once in x < 0, the trajectory remains on Side::Reactant across the entire plateau window [140, 280).
    struct AsymmetricDoubleWell1D;

    impl rgsaddle::PointSurface for AsymmetricDoubleWell1D {
        fn eval(&self, r: ArrayView1<f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
            let x = r[0];
            let (c4, c2) = if x >= 0.0 {
                (16.0, 8.0)
            } else {
                (0.0001, 0.01)
            };
            let energy = c4 * x.powi(4) - c2 * x * x;
            let gx = 4.0 * c4 * x.powi(3) - 2.0 * c2 * x;
            Ok((energy, Array1::from_vec(vec![gx])))
        }
    }

    assert_eq!(
        KappaConfig::default().commit_checkpoints,
        DEFAULT_COMMIT_CHECKPOINTS
    );
    assert_eq!(
        ShootingConfig::default().kappa.commit_checkpoints,
        DEFAULT_COMMIT_CHECKPOINTS
    );

    let plane = HyperplaneConstraint::new(Array1::from_vec(vec![1.0]), 0.0).expect("x = 0");
    let mut engine = NativeEngine::new(
        AsymmetricDoubleWell1D,
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0]))
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(0.0)
                    .with_temperature(0.08),
            )
            .with_seed(23),
    )
    .expect("1d engine");
    let starts = Array2::zeros((8, 1));
    let base_cfg = ShootingConfig::default()
        .with_steps(280)
        .with_time_step(0.02)
        .with_antithetic(true)
        .with_seed(23);

    for cfg in [
        base_cfg.clone().with_kappa(early(1)),
        base_cfg.clone().with_kappa(early(8)),
        base_cfg
            .clone()
            .with_kappa(early(DEFAULT_COMMIT_CHECKPOINTS)),
        base_cfg.clone().with_kappa(early(16)),
    ] {
        let est = shoot_bennett_chandler(&mut engine, &plane, starts.view(), &cfg)
            .expect("1d committed estimate");
        assert_abs_diff_eq!(est.plateau, 1.0, epsilon = ALGEBRAIC_TOL);
        assert_eq!(est.standard_error, 0.0);
        for &kt in &est.kappa_t {
            assert_abs_diff_eq!(kt, 1.0, epsilon = ALGEBRAIC_TOL);
        }
    }

    // By default, and with the window set but early commitment off, every half runs to
    // step 280 and records the late return. So does an opted-in window of 0.
    for cfg in [
        base_cfg.clone(),
        base_cfg.clone().with_commit_checkpoints(1),
        base_cfg.clone().with_kappa(early(0)),
    ] {
        let uncommitted = shoot_bennett_chandler(&mut engine, &plane, starts.view(), &cfg)
            .expect("1d uncommitted estimate");
        assert_abs_diff_eq!(uncommitted.kappa_t[0], 1.0, epsilon = ALGEBRAIC_TOL);
        assert_abs_diff_eq!(uncommitted.plateau, 0.0, epsilon = ALGEBRAIC_TOL);
    }
}

#[test]
fn double_well_2d_late_return_stays_at_committed_kappa_when_opted_in() {
    struct AsymmetricDoubleWell2D;

    impl rgsaddle::PointSurface for AsymmetricDoubleWell2D {
        fn eval(&self, r: ArrayView1<f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
            let (x, y) = (r[0], r[1]);
            let (c4, c2) = if x >= 0.0 {
                (16.0, 8.0)
            } else {
                (0.0001, 0.01)
            };
            let energy = c4 * x.powi(4) - c2 * x * x + y * y;
            let gx = 4.0 * c4 * x.powi(3) - 2.0 * c2 * x;
            let gy = 2.0 * y;
            Ok((energy, Array1::from_vec(vec![gx, gy])))
        }
    }

    assert_eq!(
        KappaConfig::default().commit_checkpoints,
        DEFAULT_COMMIT_CHECKPOINTS
    );

    let plane =
        HyperplaneConstraint::new(Array1::from_vec(vec![1.0, 0.0]), 0.0).expect("x = 0 plane");
    let mut engine = NativeEngine::new(
        AsymmetricDoubleWell2D,
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0, 1.0]))
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(0.0)
                    .with_temperature(0.08),
            )
            .with_seed(23),
    )
    .expect("2d engine");
    let starts = Array2::zeros((8, 2));
    let base_cfg = ShootingConfig::default()
        .with_steps(280)
        .with_time_step(0.02)
        .with_antithetic(true)
        .with_seed(23);

    for cfg in [
        base_cfg.clone().with_kappa(early(12)),
        base_cfg.clone().with_kappa(early(24)),
    ] {
        let est = shoot_bennett_chandler(&mut engine, &plane, starts.view(), &cfg)
            .expect("2d committed estimate");
        assert_abs_diff_eq!(est.plateau, 1.0, epsilon = ALGEBRAIC_TOL);
        assert_eq!(est.standard_error, 0.0);
        for &kt in &est.kappa_t {
            assert_abs_diff_eq!(kt, 1.0, epsilon = ALGEBRAIC_TOL);
        }
    }

    for cfg in [
        base_cfg.clone(),
        base_cfg.clone().with_commit_checkpoints(12),
        base_cfg.clone().with_kappa(early(0)),
    ] {
        let uncommitted = shoot_bennett_chandler(&mut engine, &plane, starts.view(), &cfg)
            .expect("2d uncommitted estimate");
        assert_abs_diff_eq!(uncommitted.kappa_t[0], 1.0, epsilon = ALGEBRAIC_TOL);
        assert_abs_diff_eq!(uncommitted.plateau, 0.0, epsilon = ALGEBRAIC_TOL);
    }
}

struct ScriptedCommitEngine {
    positions: Array1<f64>,
    velocities: Array1<f64>,
    masses: Array1<f64>,
    thermostat: ThermostatConfig,
    constraint: Option<HyperplaneConstraint>,
    forward_trace: Vec<f64>,
    backward_trace: Vec<f64>,
    step_index: usize,
    total_steps: usize,
}

impl ScriptedCommitEngine {
    fn new(forward_trace: Vec<f64>, backward_trace: Vec<f64>) -> Self {
        Self {
            positions: Array1::zeros(1),
            velocities: Array1::zeros(1),
            masses: Array1::from_vec(vec![1.0]),
            thermostat: ThermostatConfig::default(),
            constraint: None,
            forward_trace,
            backward_trace,
            step_index: 0,
            total_steps: 0,
        }
    }
}

impl DynamicsEngine for ScriptedCommitEngine {
    fn dof(&self) -> usize {
        1
    }

    fn masses(&self) -> ArrayView1<'_, f64> {
        self.masses.view()
    }

    fn positions(&self) -> ArrayView1<'_, f64> {
        self.positions.view()
    }

    fn set_positions(&mut self, positions: ArrayView1<'_, f64>) -> Result<(), FluxError> {
        self.positions.assign(&positions);
        self.step_index = 0;
        Ok(())
    }

    fn velocities(&self) -> ArrayView1<'_, f64> {
        self.velocities.view()
    }

    fn set_velocities(&mut self, velocities: ArrayView1<'_, f64>) -> Result<(), FluxError> {
        self.velocities.assign(&velocities);
        self.step_index = 0;
        Ok(())
    }

    fn cell(&self) -> Option<&[f64; 9]> {
        None
    }

    fn set_cell(&mut self, _cell: Option<[f64; 9]>) -> Result<(), FluxError> {
        Ok(())
    }

    fn thermostat(&self) -> &ThermostatConfig {
        &self.thermostat
    }

    fn set_thermostat(&mut self, thermostat: ThermostatConfig) -> Result<(), FluxError> {
        self.thermostat = thermostat;
        Ok(())
    }

    fn constraint(&self) -> Option<&HyperplaneConstraint> {
        self.constraint.as_ref()
    }

    fn set_constraint(
        &mut self,
        constraint: Option<HyperplaneConstraint>,
    ) -> Result<(), FluxError> {
        self.constraint = constraint;
        Ok(())
    }

    fn energy_and_forces(&mut self) -> Result<(f64, Array1<f64>), FluxError> {
        Ok((0.0, Array1::zeros(1)))
    }

    fn step(&mut self, _dt: f64) -> Result<StepReport, FluxError> {
        let trace = if self.velocities[0] >= 0.0 {
            &self.forward_trace
        } else {
            &self.backward_trace
        };
        self.positions[0] = trace[self.step_index];
        self.step_index += 1;
        self.total_steps += 1;
        Ok(StepReport::new(
            0.0,
            0.5 * self.velocities[0] * self.velocities[0],
            None,
        ))
    }

    fn reseed(&mut self, _seed: u64) -> Result<(), FluxError> {
        Ok(())
    }

    fn shoot_velocities(&mut self, _seed: u64) -> Result<(), FluxError> {
        self.velocities[0] = 1.0;
        self.step_index = 0;
        Ok(())
    }

    fn reflect_velocity(&mut self, _normal: ArrayView1<'_, f64>) -> Result<f64, FluxError> {
        self.velocities[0] = -self.velocities[0];
        Ok(self.velocities[0])
    }

    fn apply_dividing_surface(
        &mut self,
        _normal: ArrayView1<'_, f64>,
        _policy: SurfaceVelocityPolicy,
    ) -> Result<f64, FluxError> {
        Ok(self.velocities[0])
    }
}

#[test]
fn shooter_half_commitment_stops_early_and_resets_on_surface_or_transient_flip() {
    let plane = HyperplaneConstraint::new(Array1::from_vec(vec![1.0]), 0.0).expect("x = 0");
    let starts = Array2::zeros((1, 1));

    // Forward half: 2 steps Product (< 3), 1 step on dividing plane (s == 0.0 resets run),
    // 2 steps Product (< 3), then 3 consecutive steps Reactant (commits at index 7, step 8),
    // followed by a late return to Product that must never be stepped or recorded.
    let forward_trace = vec![
        0.2, 0.3, 0.0, 0.4, 0.5, -0.2, -0.3, -0.4, 0.9, 0.9, 0.9, 0.9,
    ];
    // Backward half: 3 consecutive steps Product (commits at index 2, step 3),
    // followed by a late return to Reactant that must never be stepped or recorded.
    let backward_trace = vec![
        0.1, 0.2, 0.3, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8,
    ];

    let mut engine = ScriptedCommitEngine::new(forward_trace, backward_trace);
    let cfg = ShootingConfig::default()
        .with_steps(12)
        .with_time_step(0.01)
        .with_antithetic(true)
        .with_kappa(early(3));

    let trajs =
        shoot_reactive_trajectories(&mut engine, &plane, starts.view(), &cfg).expect("scripted");
    assert_eq!(trajs.len(), 2);
    // Forward half stopped after 8 steps; backward half stopped after 3 steps (8 + 3 = 11 < 24).
    assert_eq!(engine.total_steps, 11);
    assert_eq!(
        trajs[0].sides,
        vec![
            Side::Product,
            Side::Product,
            Side::Reactant, // s == 0.0 has indicator 0 (Reactant) and committed_from_signed_distance == None
            Side::Product,
            Side::Product,
            Side::Reactant,
            Side::Reactant,
            Side::Reactant,
            Side::Reactant,
            Side::Reactant,
            Side::Reactant,
            Side::Reactant,
        ]
    );
    assert_eq!(trajs[1].sides, vec![Side::Product; 12]);
}

#[test]
fn validate_commit_checkpoints_and_config_setters() {
    let defaults = KappaConfig::default();
    assert!(!defaults.early_commitment);
    assert_eq!(defaults.commit_checkpoints, DEFAULT_COMMIT_CHECKPOINTS);
    assert_eq!(ShootingConfig::default().kappa, defaults);

    let kcfg = KappaConfig::default()
        .with_early_commitment(true)
        .with_commit_checkpoints(7)
        .with_quench_force_tol(1e-4)
        .with_quench_max_steps(500);
    assert!(kcfg.early_commitment);
    assert_eq!(kcfg.commit_checkpoints, 7);
    assert_eq!(kcfg.quench_force_tol, 1e-4);
    assert_eq!(kcfg.quench_max_steps, 500);

    let scfg = ShootingConfig::default()
        .with_steps(16)
        .with_kappa(kcfg.clone());
    assert_eq!(scfg.kappa, kcfg);
    let scfg2 = scfg.with_commit_checkpoints(9);
    assert_eq!(scfg2.kappa.commit_checkpoints, 9);

    assert_eq!(
        Side::committed_from_signed_distance(-1e-12),
        Some(Side::Reactant)
    );
    assert_eq!(Side::committed_from_signed_distance(0.0), None);
    assert_eq!(Side::committed_from_signed_distance(-0.0), None);
    assert_eq!(
        Side::committed_from_signed_distance(1e-12),
        Some(Side::Product)
    );

    let plane = HyperplaneConstraint::new(Array1::from_vec(vec![1.0]), 0.0).expect("x = 0");
    let starts = Array2::zeros((1, 1));
    let mut engine = ScriptedCommitEngine::new(vec![0.5; 16], vec![-0.5; 16]);
    engine
        .set_constraint(Some(plane.clone()))
        .expect("set constraint");

    for bad_commit in [9_usize, 17, usize::MAX] {
        let bad_cfg = ShootingConfig::default()
            .with_steps(8)
            .with_kappa(early(bad_commit));
        assert_eq!(
            shoot_reactive_trajectories(&mut engine, &plane, starts.view(), &bad_cfg),
            Err(FluxError::InvalidConfig(
                "KappaConfig::commit_checkpoints must not exceed ShootingConfig::steps"
            ))
        );
        assert_eq!(
            shoot_bennett_chandler(&mut engine, &plane, starts.view(), &bad_cfg),
            Err(FluxError::InvalidConfig(
                "KappaConfig::commit_checkpoints must not exceed ShootingConfig::steps"
            ))
        );
        // Validation rejects before touching the engine's constraint or stepping.
        assert_eq!(engine.total_steps, 0);
        assert!(engine.constraint().is_some());

        // The same window is ignored, and not validated, while early commitment is off.
        let off_cfg = ShootingConfig::default()
            .with_steps(8)
            .with_commit_checkpoints(bad_commit);
        let trajs = shoot_reactive_trajectories(&mut engine, &plane, starts.view(), &off_cfg)
            .expect("window ignored while early commitment is off");
        assert_eq!(trajs.len(), 2);
        assert_eq!(trajs[0].sides.len(), 8);
        engine.total_steps = 0;
    }

    // Boundary values 0, 1, and steps are accepted.
    for valid_commit in [0_usize, 1, 8] {
        let ok_cfg = ShootingConfig::default()
            .with_steps(8)
            .with_kappa(early(valid_commit));
        let trajs = shoot_reactive_trajectories(&mut engine, &plane, starts.view(), &ok_cfg)
            .expect("valid commit_checkpoints");
        assert_eq!(trajs.len(), 2);
        assert_eq!(trajs[0].sides.len(), 8);
        assert!(engine.constraint().is_some());
    }
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

    #[test]
    fn default_shooting_reproduces_fixed_step_shooting_bit_for_bit(
        steps in 8_usize..36,
        time_step in 0.005_f64..0.05_f64,
        gamma in 0.0_f64..4.0_f64,
        temperature in 0.05_f64..0.5_f64,
        antithetic in any::<bool>(),
        seed in any::<u64>(),
        window in 0_usize..64,
        mode in 0_u8..3,
        y_offsets in prop::collection::vec(-1.0_f64..1.0_f64, 4..8),
    ) {
        struct DoubleWell2D;

        impl rgsaddle::PointSurface for DoubleWell2D {
            fn eval(&self, r: ArrayView1<f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
                let (x, y) = (r[0], r[1]);
                let dx = x * x - 1.0;
                Ok((dx * dx + y * y, Array1::from_vec(vec![4.0 * x * dx, 2.0 * y])))
            }
        }

        let plane = HyperplaneConstraint::new(Array1::from_vec(vec![1.0, 0.0]), 0.0).expect("plane");
        let native_cfg = NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0, 1.0]))
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(gamma)
                    .with_temperature(temperature),
            )
            .with_constraint(plane.clone())
            .with_seed(seed);

        let mut engine_new = NativeEngine::new(DoubleWell2D, native_cfg.clone()).expect("engine_new");
        let mut engine_old = NativeEngine::new(DoubleWell2D, native_cfg).expect("engine_old");

        let mut initial_positions = Array2::zeros((y_offsets.len(), 2));
        for (idx, &y) in y_offsets.iter().enumerate() {
            initial_positions[(idx, 0)] = 0.0;
            initial_positions[(idx, 1)] = y;
        }

        let estimator_cfg = BennettChandlerConfig::default()
            .with_stats(StatsConfig::default().with_max_z_score(f64::INFINITY));
        let shoot_cfg = ShootingConfig::default()
            .with_steps(steps)
            .with_time_step(time_step)
            .with_antithetic(antithetic)
            .with_seed(seed)
            .with_estimator(estimator_cfg.clone());
        // Mode 0 is the plain default, mode 1 sets a window without opting in, and
        // mode 2 opts in with a window of 0. None of them may change a single bit.
        let shoot_cfg = match mode {
            0 => shoot_cfg,
            1 => shoot_cfg.with_commit_checkpoints(window),
            _ => shoot_cfg.with_kappa(early(0)),
        };

        let trajs_new = shoot_reactive_trajectories(
            &mut engine_new,
            &plane,
            initial_positions.view(),
            &shoot_cfg,
        )
        .expect("trajs_new");

        // Reference fixed-step loop matching the pre-commit_checkpoints implementation.
        let saved_constraint = engine_old.constraint().cloned();
        engine_old.set_constraint(None).expect("clear constraint");
        let mut trajs_old = Vec::new();
        for (idx, x0) in initial_positions.rows().into_iter().enumerate() {
            let shot_seed = seed.wrapping_add((idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            engine_old.set_positions(x0).expect("set_positions");
            engine_old.shoot_velocities(shot_seed).expect("shoot_velocities");
            let v_plus = engine_old.velocities().to_owned();
            let v0_normal = plane.normal.dot(&v_plus);
            let mut sides_plus = Vec::with_capacity(steps);
            for _ in 0..steps {
                engine_old.step(time_step).expect("step");
                let s = plane.signed_distance(engine_old.positions()).expect("signed_distance");
                sides_plus.push(Side::from_signed_distance(s));
            }
            trajs_old.push(ReactiveTrajectory::new(v0_normal, sides_plus).expect("traj_plus"));

            if antithetic {
                let v_minus = v_plus.mapv(|c| -c);
                engine_old.set_positions(x0).expect("set_positions");
                engine_old
                    .shoot_velocities(shot_seed ^ 0xBF58_476D_1CE4_E5B9)
                    .expect("shoot_velocities");
                engine_old.set_velocities(v_minus.view()).expect("set_velocities");
                let mut sides_minus = Vec::with_capacity(steps);
                for _ in 0..steps {
                    engine_old.step(time_step).expect("step");
                    let s = plane.signed_distance(engine_old.positions()).expect("signed_distance");
                    sides_minus.push(Side::from_signed_distance(s));
                }
                trajs_old.push(ReactiveTrajectory::new(-v0_normal, sides_minus).expect("traj_minus"));
            }
        }
        engine_old.set_constraint(saved_constraint).expect("restore constraint");

        prop_assert_eq!(trajs_new.len(), trajs_old.len());
        for (tn, to) in trajs_new.iter().zip(&trajs_old) {
            prop_assert_eq!(
                tn.initial_normal_velocity.to_bits(),
                to.initial_normal_velocity.to_bits()
            );
            prop_assert_eq!(&tn.sides, &to.sides);
        }
        for (pz, po) in engine_new.positions().iter().zip(engine_old.positions().iter()) {
            prop_assert_eq!(pz.to_bits(), po.to_bits());
        }
        for (vz, vo) in engine_new.velocities().iter().zip(engine_old.velocities().iter()) {
            prop_assert_eq!(vz.to_bits(), vo.to_bits());
        }

        let est_new = estimate_bennett_chandler(&trajs_new, &estimator_cfg);
        let est_old = estimate_bennett_chandler(&trajs_old, &estimator_cfg);
        match (est_new, est_old) {
            (Ok(ez), Ok(eo)) => {
                prop_assert_eq!(ez.kappa_t.len(), eo.kappa_t.len());
                for (kz, ko) in ez.kappa_t.iter().zip(&eo.kappa_t) {
                    prop_assert_eq!(kz.to_bits(), ko.to_bits());
                }
                prop_assert_eq!(ez.plateau.to_bits(), eo.plateau.to_bits());
                prop_assert_eq!(ez.standard_error.to_bits(), eo.standard_error.to_bits());
                prop_assert_eq!(ez.n_eff.to_bits(), eo.n_eff.to_bits());
                prop_assert_eq!(ez.plateau_window, eo.plateau_window);
                prop_assert_eq!(ez.positive_flux_count, eo.positive_flux_count);
                prop_assert_eq!(
                    ez.stationarity.z_score.to_bits(),
                    eo.stationarity.z_score.to_bits()
                );
                prop_assert_eq!(
                    ez.stationarity.drift_slope.to_bits(),
                    eo.stationarity.drift_slope.to_bits()
                );
                prop_assert_eq!(
                    ez.stationarity.first_mean.to_bits(),
                    eo.stationarity.first_mean.to_bits()
                );
                prop_assert_eq!(
                    ez.stationarity.second_mean.to_bits(),
                    eo.stationarity.second_mean.to_bits()
                );
            }
            (Err(ez), Err(eo)) => prop_assert_eq!(ez, eo),
            (z, o) => prop_assert!(false, "mismatched estimate results: {z:?} vs {o:?}"),
        }
    }
}
