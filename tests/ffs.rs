//! Direct Forward Flux Sampling tests.

use std::cell::Cell;

use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rgflux::FluxError;
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::engine::{DynamicsEngine, HyperplaneConstraint, StepReport, ThermostatConfig};
use rgflux::ffs::{FfsConfig, forward_flux_rate};

struct QuarticDoubleWell1D;

impl rgsaddle::PointSurface for QuarticDoubleWell1D {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
        let dx = x[0] * x[0] - 1.0;
        let energy = dx * dx;
        let grad = Array1::from_vec(vec![4.0 * x[0] * dx]);
        Ok((energy, grad))
    }
}

fn make_engine(seed: u64) -> NativeEngine<QuarticDoubleWell1D> {
    let cfg = NativeConfig::default()
        .with_masses(Array1::from_vec(vec![1.0]))
        .with_thermostat(
            ThermostatConfig::default()
                .with_gamma(2.0)
                .with_temperature(0.9),
        )
        .with_seed(seed);
    let mut engine = NativeEngine::new(QuarticDoubleWell1D, cfg).expect("engine");
    engine
        .set_positions(Array1::from_vec(vec![-1.0]).view())
        .expect("start in basin A");
    engine.shoot_velocities(seed).expect("shoot");
    engine
}

fn find_seed_with_flux_seed_parity(want_odd: bool) -> u64 {
    for seed in 1_u64..=4_096 {
        let mut rng = StdRng::seed_from_u64(seed);
        let flux_seed = rng.random::<u64>();
        if (flux_seed & 1 == 1) == want_odd {
            return seed;
        }
    }
    panic!("failed to find matching seed parity");
}

#[test]
fn forward_flux_sampling_native_accuracy_is_within_reported_uncertainty() {
    let mut engine = make_engine(11);
    let cfg = FfsConfig::default()
        .with_interfaces(vec![-0.8, -0.4, -0.1, 0.2])
        .with_points_per_interface(32)
        .with_trials_per_interface(128)
        .with_max_trial_steps(3_000)
        .with_max_flux_steps(800_000)
        .with_time_step(0.01)
        .with_seed(11);
    let estimate = forward_flux_rate(&mut engine, |x| x[0], &cfg).expect("ffs estimate");

    let mut engine_ref = make_engine(12_345);
    let cfg = FfsConfig::default()
        .with_interfaces(vec![-0.8, -0.4, -0.1, 0.2])
        .with_points_per_interface(128)
        .with_trials_per_interface(512)
        .with_max_trial_steps(6_000)
        .with_max_flux_steps(2_000_000)
        .with_time_step(0.01)
        .with_seed(12_345);
    let reference = forward_flux_rate(&mut engine_ref, |x| x[0], &cfg).expect("reference");

    let diff = (estimate.rate - reference.rate).abs();
    let combined_se = estimate
        .standard_error
        .hypot(reference.standard_error)
        .max(f64::EPSILON);
    assert!(
        diff <= 3.0 * combined_se,
        "rate difference {:.6e} exceeds 3 sigma {:.6e}",
        diff,
        3.0 * combined_se
    );

    assert!(estimate.initial_flux > 0.0);
    assert!(estimate.rate > 0.0);
    assert!(estimate.standard_error.is_finite());
    assert_eq!(estimate.interface_probabilities.len(), 3);
    assert!(estimate.interface_probabilities.iter().all(|p| *p > 0.0));
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScriptMode {
    Flux,
    Trial,
}

#[derive(Clone, Debug)]
struct ScriptedFfsEngine {
    masses: Array1<f64>,
    positions: Array1<f64>,
    velocities: Array1<f64>,
    thermostat: ThermostatConfig,
    constraint: Option<HyperplaneConstraint>,
    mode: ScriptMode,
    flux_seed: u64,
    flux_seeded: bool,
    flux_step: usize,
    trial_token: u64,
    trial_seed: u64,
    trial_step: usize,
    trial_origin_lambda: f64,
    initial_position: f64,
    fail_trial_after_steps: Option<usize>,
    fail_restore_position: bool,
    force_trial_success: bool,
    flux_script: Option<Vec<f64>>,
    trial_script: Option<TrialScript>,
    velocity_reads: Cell<usize>,
}

type TrialScript = fn(f64, usize, u64) -> f64;

impl ScriptedFfsEngine {
    fn new() -> Self {
        Self {
            masses: Array1::from_vec(vec![1.0]),
            positions: Array1::from_vec(vec![-1.2]),
            velocities: Array1::from_vec(vec![0.35]),
            thermostat: ThermostatConfig::default(),
            constraint: None,
            mode: ScriptMode::Flux,
            flux_seed: 0,
            flux_seeded: false,
            flux_step: 0,
            trial_token: 0,
            trial_seed: 0,
            trial_step: 0,
            trial_origin_lambda: -1.2,
            initial_position: -1.2,
            fail_trial_after_steps: None,
            fail_restore_position: false,
            force_trial_success: false,
            flux_script: None,
            trial_script: None,
            velocity_reads: Cell::new(0),
        }
    }

    fn with_start(mut self, lambda: f64) -> Self {
        self.positions[0] = lambda;
        self.initial_position = lambda;
        self.trial_origin_lambda = lambda;
        self
    }

    fn with_flux_script(mut self, script: &[f64]) -> Self {
        self.flux_script = Some(script.to_vec());
        self
    }

    fn with_trial_script(mut self, script: TrialScript) -> Self {
        self.trial_script = Some(script);
        self
    }

    fn with_trial_failure(mut self, step: usize) -> Self {
        self.fail_trial_after_steps = Some(step);
        self
    }

    fn with_restore_position_failure(mut self) -> Self {
        self.fail_restore_position = true;
        self
    }

    fn with_force_trial_success(mut self) -> Self {
        self.force_trial_success = true;
        self
    }
}

impl DynamicsEngine for ScriptedFfsEngine {
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
        if self.fail_restore_position
            && (positions[0] - self.initial_position).abs() <= f64::EPSILON
        {
            return Err(FluxError::Surface("scripted restore failure".to_string()));
        }
        self.positions.assign(&positions);
        self.trial_origin_lambda = self.positions[0];
        self.trial_step = 0;
        if self.mode == ScriptMode::Flux
            && (self.positions[0] - self.initial_position).abs() <= f64::EPSILON
        {
            self.flux_step = 0;
        }
        Ok(())
    }

    fn velocities(&self) -> ArrayView1<'_, f64> {
        self.velocity_reads.set(self.velocity_reads.get() + 1);
        self.velocities.view()
    }

    fn set_velocities(&mut self, velocities: ArrayView1<'_, f64>) -> Result<(), FluxError> {
        self.velocities.assign(&velocities);
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
        Ok((0.0, Array1::from_vec(vec![0.0])))
    }

    fn step(&mut self, _dt: f64) -> Result<StepReport, FluxError> {
        self.velocities[0] += 0.125;
        match self.mode {
            ScriptMode::Flux => {
                let lambda = if let Some(script) = &self.flux_script {
                    script[self.flux_step % script.len()]
                } else if self.flux_seed & 1 == 0 {
                    // Excursion with multiple steps above lambda_a:
                    // step 0 crosses lambda_0, step 1 hits product, step 2 stays above.
                    match self.flux_step % 4 {
                        0 => 0.2,
                        1 | 2 => 1.2,
                        _ => -1.2,
                    }
                } else {
                    // Longer excursion before hitting product and reset.
                    match self.flux_step % 5 {
                        0 => 0.2,
                        1 => 0.6,
                        2 | 3 => 1.2,
                        _ => -1.2,
                    }
                };
                self.positions[0] = lambda;
                self.flux_step += 1;
            }
            ScriptMode::Trial => {
                self.trial_step += 1;
                if let Some(limit) = self.fail_trial_after_steps
                    && self.trial_step >= limit
                {
                    return Err(FluxError::Surface("scripted trial failure".to_string()));
                }
                if let Some(script) = self.trial_script {
                    self.positions[0] =
                        script(self.trial_origin_lambda, self.trial_step, self.trial_token);
                    return Ok(StepReport::new(0.0, 0.0, None));
                }
                let in_stage_zero = self.trial_origin_lambda < 0.4;
                let token_mod = self.trial_token % 4;
                let success_step = if self.force_trial_success {
                    true
                } else if self.trial_seed == 0 {
                    !in_stage_zero
                } else if in_stage_zero {
                    token_mod != 0
                } else {
                    token_mod == 0
                };
                self.positions[0] = if success_step {
                    if in_stage_zero { 0.5 } else { 2.5 }
                } else if self.trial_step == 1 {
                    -0.5
                } else {
                    -1.5
                };
            }
        }
        Ok(StepReport::new(0.0, 0.0, None))
    }

    fn reseed(&mut self, seed: u64) -> Result<(), FluxError> {
        if !self.flux_seeded {
            self.flux_seeded = true;
            self.flux_seed = seed;
            self.flux_step = 0;
            self.mode = ScriptMode::Flux;
        } else {
            self.trial_token = self.trial_token.wrapping_add(1);
            self.trial_seed = seed;
            self.trial_step = 0;
            self.mode = ScriptMode::Trial;
        }
        Ok(())
    }

    fn shoot_velocities(&mut self, _seed: u64) -> Result<(), FluxError> {
        Ok(())
    }

    fn reflect_velocity(&mut self, _normal: ArrayView1<'_, f64>) -> Result<f64, FluxError> {
        Ok(0.0)
    }

    fn apply_dividing_surface(
        &mut self,
        _normal: ArrayView1<'_, f64>,
        _policy: rgflux::engine::SurfaceVelocityPolicy,
    ) -> Result<f64, FluxError> {
        Ok(0.0)
    }
}

#[test]
fn scripted_engine_exact_rate_components_match_hand_values() {
    let seed = find_seed_with_flux_seed_parity(false);
    let mut engine = ScriptedFfsEngine::new();
    let cfg = FfsConfig::default()
        .with_interfaces(vec![-1.0, 0.0, 1.0])
        .with_points_per_interface(4)
        .with_trials_per_interface(8)
        .with_max_trial_steps(3)
        .with_max_flux_steps(64)
        .with_time_step(1.0)
        .with_seed(seed)
        .with_stats(rgflux::stats::StatsConfig::default().with_batch_size(1));
    let estimate = forward_flux_rate(&mut engine, |x| x[0], &cfg).expect("scripted estimate");

    let mut flux_rng = StdRng::seed_from_u64(cfg.seed);
    let first_flux_seed = flux_rng.random::<u64>();
    let intervals = if first_flux_seed & 1 == 0 {
        [1.0, 2.0, 2.0, 2.0]
    } else {
        [1.0, 3.0, 3.0, 3.0]
    };
    let total = intervals.iter().sum::<f64>();
    let expected_flux = 4.0 / total;
    let mean = total / 4.0;
    let variance = intervals
        .iter()
        .map(|d| {
            let centered = *d - mean;
            centered * centered
        })
        .sum::<f64>()
        / 3.0;
    let expected_flux_se = expected_flux * variance.sqrt() / (mean * 2.0);

    let expected_stage0 = 0.75;
    let expected_stage1 = 0.25;
    let expected_rate = expected_flux * expected_stage0 * expected_stage1;

    assert!((estimate.initial_flux - expected_flux).abs() <= f64::EPSILON);
    assert!((estimate.interface_probabilities[0] - expected_stage0).abs() <= f64::EPSILON);
    assert!((estimate.interface_probabilities[1] - expected_stage1).abs() <= f64::EPSILON);
    assert!((estimate.rate - expected_rate).abs() <= f64::EPSILON);

    assert!(estimate.standard_error.is_finite());
    assert!(estimate.standard_error > 0.0);
    let expected_flux_se_hand = if first_flux_seed & 1 == 0 {
        0.081_632_653_061_224_48
    } else {
        0.08
    };
    assert!((expected_flux_se - expected_flux_se_hand).abs() <= f64::EPSILON);
    assert!((engine.positions()[0] - (-1.2)).abs() <= f64::EPSILON);
    assert!((engine.velocities()[0] - 0.35).abs() <= f64::EPSILON);
}

#[test]
fn scripted_engine_seed_changes_flux_phase() {
    let seed_even = find_seed_with_flux_seed_parity(false);
    let seed_odd = find_seed_with_flux_seed_parity(true);

    let cfg_a = FfsConfig::default()
        .with_interfaces(vec![-1.0, 0.0, 1.0])
        .with_points_per_interface(4)
        .with_trials_per_interface(8)
        .with_max_trial_steps(3)
        .with_max_flux_steps(64)
        .with_time_step(1.0)
        .with_seed(seed_even)
        .with_stats(rgflux::stats::StatsConfig::default().with_batch_size(1));
    let cfg_b = cfg_a.clone().with_seed(seed_odd);
    let mut a = ScriptedFfsEngine::new();
    let mut b = ScriptedFfsEngine::new();
    let ea = forward_flux_rate(&mut a, |x| x[0], &cfg_a).expect("seed a");
    let eb = forward_flux_rate(&mut b, |x| x[0], &cfg_b).expect("seed b");
    assert_ne!(ea.initial_flux.to_bits(), eb.initial_flux.to_bits());
}

#[test]
fn forward_flux_sampling_is_reproducible_for_identical_seed() {
    let cfg = FfsConfig::default()
        .with_interfaces(vec![-1.0, 0.0, 1.0])
        .with_points_per_interface(4)
        .with_trials_per_interface(8)
        .with_max_trial_steps(3)
        .with_max_flux_steps(64)
        .with_time_step(1.0)
        .with_seed(99)
        .with_stats(rgflux::stats::StatsConfig::default().with_batch_size(1));
    let mut a = ScriptedFfsEngine::new();
    let mut b = ScriptedFfsEngine::new();

    let ea = forward_flux_rate(&mut a, |x| x[0], &cfg).expect("estimate a");
    let eb = forward_flux_rate(&mut b, |x| x[0], &cfg).expect("estimate b");
    assert_eq!(ea.initial_flux.to_bits(), eb.initial_flux.to_bits());
    assert_eq!(ea.rate.to_bits(), eb.rate.to_bits());
    assert_eq!(ea.standard_error.to_bits(), eb.standard_error.to_bits());
    assert_eq!(
        ea.interface_probabilities
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        eb.interface_probabilities
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn invalid_ffs_configuration_is_rejected() {
    let mut engine = ScriptedFfsEngine::new();
    let cfg = FfsConfig::default().with_interfaces(vec![0.1, 0.1]);
    let err = forward_flux_rate(&mut engine, |x| x[0], &cfg).expect_err("invalid interfaces");
    assert!(format!("{err}").contains("interfaces"));
}

#[test]
fn burn_in_skips_initial_crossings() {
    let seed = find_seed_with_flux_seed_parity(false);
    let base = FfsConfig::default()
        .with_interfaces(vec![-1.0, 0.0, 1.0])
        .with_points_per_interface(2)
        .with_trials_per_interface(8)
        .with_max_trial_steps(3)
        .with_max_flux_steps(3)
        .with_time_step(1.0)
        .with_seed(seed)
        .with_stats(rgflux::stats::StatsConfig::default().with_batch_size(1));
    let with_burn = base.clone().with_burn_in_steps(2);
    let mut engine = ScriptedFfsEngine::new();
    let no_burn = forward_flux_rate(&mut engine, |x| x[0], &base).expect("no burn");

    let mut engine_burn = ScriptedFfsEngine::new();
    let err = forward_flux_rate(&mut engine_burn, |x| x[0], &with_burn).expect_err("burned");
    assert!(format!("{err}").contains("flux budget exceeded"));

    let with_burn_ok = base.clone().with_burn_in_steps(1).with_max_flux_steps(8);
    let mut engine_burn_ok = ScriptedFfsEngine::new();
    let burned = forward_flux_rate(&mut engine_burn_ok, |x| x[0], &with_burn_ok).expect("burn ok");
    assert_ne!(
        no_burn.initial_flux.to_bits(),
        burned.initial_flux.to_bits()
    );
}

#[test]
fn restore_runs_after_error_and_after_success() {
    let cfg = FfsConfig::default()
        .with_interfaces(vec![-1.0, 0.0, 1.0])
        .with_points_per_interface(4)
        .with_trials_per_interface(8)
        .with_max_trial_steps(3)
        .with_max_flux_steps(128)
        .with_time_step(1.0)
        .with_seed(42)
        .with_stats(rgflux::stats::StatsConfig::default().with_batch_size(1));

    let mut ok_engine = ScriptedFfsEngine::new();
    let _ = forward_flux_rate(&mut ok_engine, |x| x[0], &cfg).expect("success");
    assert!((ok_engine.positions()[0] - (-1.2)).abs() <= f64::EPSILON);
    assert!((ok_engine.velocities()[0] - 0.35).abs() <= f64::EPSILON);

    let mut err_engine = ScriptedFfsEngine::new().with_trial_failure(1);
    let err = forward_flux_rate(&mut err_engine, |x| x[0], &cfg).expect_err("error");
    assert!(format!("{err}").contains("scripted trial failure"));
    assert!((err_engine.positions()[0] - (-1.2)).abs() <= f64::EPSILON);
    assert!((err_engine.velocities()[0] - 0.35).abs() <= f64::EPSILON);
}

#[test]
fn direct_to_product_crossing_is_counted_once_per_excursion() {
    let seed = find_seed_with_flux_seed_parity(false);
    let mut engine = ScriptedFfsEngine::new();
    let cfg = FfsConfig::default()
        .with_interfaces(vec![-1.0, 0.0, 1.0])
        .with_points_per_interface(4)
        .with_trials_per_interface(8)
        .with_max_trial_steps(3)
        .with_max_flux_steps(64)
        .with_time_step(1.0)
        .with_seed(seed)
        .with_stats(rgflux::stats::StatsConfig::default().with_batch_size(1));
    let estimate = forward_flux_rate(&mut engine, |x| x[0], &cfg).expect("estimate");
    assert!((estimate.initial_flux - 0.5714285714285714).abs() <= f64::EPSILON);
}

#[test]
fn rearm_after_basin_return_is_required_without_product_resets() {
    let mut engine = ScriptedFfsEngine::new();
    let cfg = FfsConfig::default()
        .with_interfaces(vec![-1.0, 0.0, 2.0])
        .with_points_per_interface(3)
        .with_trials_per_interface(8)
        .with_max_trial_steps(3)
        .with_max_flux_steps(32)
        .with_time_step(1.0)
        .with_seed(find_seed_with_flux_seed_parity(false))
        .with_stats(rgflux::stats::StatsConfig::default().with_batch_size(1));
    let estimate = forward_flux_rate(&mut engine, |x| x[0], &cfg).expect("estimate");
    assert!(estimate.initial_flux > 0.0);
}

#[test]
fn restore_error_wins_when_run_succeeds() {
    let mut engine = ScriptedFfsEngine::new().with_restore_position_failure();
    let cfg = FfsConfig::default()
        .with_interfaces(vec![-1.0, 0.0, 2.0])
        .with_points_per_interface(3)
        .with_trials_per_interface(8)
        .with_max_trial_steps(3)
        .with_max_flux_steps(64)
        .with_time_step(1.0)
        .with_seed(7)
        .with_stats(rgflux::stats::StatsConfig::default().with_batch_size(1));
    let err = forward_flux_rate(&mut engine, |x| x[0], &cfg).expect_err("restore failure");
    assert_eq!(
        err,
        FluxError::Surface("scripted restore failure".to_string())
    );
}

#[test]
fn run_error_wins_when_both_run_and_restore_fail() {
    let mut engine = ScriptedFfsEngine::new()
        .with_trial_failure(1)
        .with_restore_position_failure();
    let cfg = FfsConfig::default()
        .with_interfaces(vec![-1.0, 0.0, 2.0])
        .with_points_per_interface(3)
        .with_trials_per_interface(8)
        .with_max_trial_steps(3)
        .with_max_flux_steps(64)
        .with_time_step(1.0)
        .with_seed(7)
        .with_stats(rgflux::stats::StatsConfig::default().with_batch_size(1));
    let err = forward_flux_rate(&mut engine, |x| x[0], &cfg).expect_err("dual failure");
    assert_eq!(
        err,
        FluxError::Surface("scripted trial failure".to_string())
    );
}

#[test]
fn exact_boundary_values_follow_crossing_rules() {
    let mut engine = ScriptedFfsEngine::new();
    let cfg = FfsConfig::default()
        .with_interfaces(vec![0.2, 0.5, 2.5])
        .with_points_per_interface(4)
        .with_trials_per_interface(8)
        .with_max_trial_steps(3)
        .with_max_flux_steps(64)
        .with_time_step(1.0)
        .with_seed(find_seed_with_flux_seed_parity(false))
        .with_stats(rgflux::stats::StatsConfig::default().with_batch_size(1));
    let estimate = forward_flux_rate(&mut engine, |x| x[0], &cfg).expect("boundary estimate");
    assert!((estimate.interface_probabilities[0] - 0.25).abs() <= f64::EPSILON);
    assert!((estimate.interface_probabilities[1] - 0.25).abs() <= f64::EPSILON);
}

#[test]
fn rate_standard_error_matches_hand_propagation_when_stage_se_is_zero() {
    let seed = find_seed_with_flux_seed_parity(false);
    let mut engine = ScriptedFfsEngine::new().with_force_trial_success();
    let cfg = FfsConfig::default()
        .with_interfaces(vec![-1.0, 0.0, 1.0])
        .with_points_per_interface(4)
        .with_trials_per_interface(8)
        .with_max_trial_steps(3)
        .with_max_flux_steps(64)
        .with_time_step(1.0)
        .with_seed(seed)
        .with_stats(rgflux::stats::StatsConfig::default().with_batch_size(1));
    let estimate = forward_flux_rate(&mut engine, |x| x[0], &cfg).expect("se propagation");

    let mut flux_rng = StdRng::seed_from_u64(cfg.seed);
    let first_flux_seed = flux_rng.random::<u64>();
    let intervals = if first_flux_seed & 1 == 0 {
        [1.0, 2.0, 2.0, 2.0]
    } else {
        [1.0, 3.0, 3.0, 3.0]
    };
    let total = intervals.iter().sum::<f64>();
    let expected_flux = 4.0 / total;
    let mean = total / 4.0;
    let variance = intervals
        .iter()
        .map(|d| {
            let centered = *d - mean;
            centered * centered
        })
        .sum::<f64>()
        / 3.0;
    let expected_flux_se = expected_flux * variance.sqrt() / (mean * 2.0);
    assert!((estimate.rate - expected_flux).abs() <= f64::EPSILON);
    assert!((estimate.standard_error - expected_flux_se).abs() <= f64::EPSILON);
}

#[test]
fn basin_start_must_be_inside_a() {
    let mut engine = ScriptedFfsEngine::new();
    engine
        .set_positions(Array1::from_vec(vec![0.1]).view())
        .expect("set position");
    let cfg = FfsConfig::default().with_interfaces(vec![-0.5, 0.0, 0.5]);
    let err = forward_flux_rate(&mut engine, |x| x[0], &cfg).expect_err("basin start");
    assert!(matches!(err, FluxError::InvalidConfig(msg) if msg.contains("begin in basin A")));
}

fn advance_by_one(origin: f64, _step: usize, _token: u64) -> f64 {
    origin + 1.0
}

fn scripted_cfg(interfaces: Vec<f64>, max_flux_steps: usize) -> FfsConfig {
    FfsConfig::default()
        .with_interfaces(interfaces)
        .with_points_per_interface(4)
        .with_trials_per_interface(8)
        .with_max_trial_steps(3)
        .with_max_flux_steps(max_flux_steps)
        .with_time_step(1.0)
        .with_seed(5)
        .with_stats(rgflux::stats::StatsConfig::default().with_batch_size(1))
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 4.0 * f64::EPSILON,
        "expected {expected:.17e}, got {actual:.17e}"
    );
}

/// Four crossings with intervals `[1, 2, 2, 2]` give `Phi_{A,0} = 4 / 7`.
const FLUX_FOUR_OVER_SEVEN: f64 = 4.0 / 7.0;

#[test]
fn landing_exactly_on_lambda_a_while_disarmed_rearms_the_crossing() {
    // 0.5 crosses lambda_0 and disarms, 0.0 sits exactly on lambda_0 and must
    // rearm, the second 0.5 is then a second crossing, 2.5 reaches the product.
    let script = [0.5, 0.0, 0.5, 2.5];
    let cfg = scripted_cfg(vec![0.0, 1.0, 2.0], 7);

    let mut engine = ScriptedFfsEngine::new()
        .with_flux_script(&script)
        .with_trial_script(advance_by_one);
    let estimate = forward_flux_rate(&mut engine, |x| x[0], &cfg).expect("seven steps suffice");
    assert_close(estimate.initial_flux, FLUX_FOUR_OVER_SEVEN);
    assert_close(estimate.rate, FLUX_FOUR_OVER_SEVEN);
    assert_eq!(estimate.interface_probabilities, vec![1.0, 1.0]);

    // Crossings land on steps 1, 3, 5 and 7.
    let mut short = ScriptedFfsEngine::new()
        .with_flux_script(&script)
        .with_trial_script(advance_by_one);
    let err = forward_flux_rate(&mut short, |x| x[0], &cfg.clone().with_max_flux_steps(6))
        .expect_err("six steps hold only three crossings");
    assert!(format!("{err}").contains("flux budget exceeded"));
}

#[test]
fn flux_sample_exactly_at_product_level_resets_to_the_initial_state() {
    // 2.0 equals the product level: it must reset the engine so the script
    // restarts at 0.5 and every other step is a crossing.
    let script = [0.5, 2.0, 0.5, -1.0];
    let cfg = scripted_cfg(vec![0.0, 1.0, 2.0], 7);

    let mut engine = ScriptedFfsEngine::new()
        .with_flux_script(&script)
        .with_trial_script(advance_by_one);
    let estimate = forward_flux_rate(&mut engine, |x| x[0], &cfg).expect("seven steps suffice");
    assert_close(estimate.initial_flux, FLUX_FOUR_OVER_SEVEN);
    assert_close(estimate.rate, FLUX_FOUR_OVER_SEVEN);

    let mut short = ScriptedFfsEngine::new()
        .with_flux_script(&script)
        .with_trial_script(advance_by_one);
    let err = forward_flux_rate(&mut short, |x| x[0], &cfg.clone().with_max_flux_steps(6))
        .expect_err("six steps hold only three crossings");
    assert!(format!("{err}").contains("flux budget exceeded"));
}

#[test]
fn start_exactly_on_lambda_0_is_inside_basin_a() {
    let cfg = scripted_cfg(vec![0.0, 1.0, 2.0], 7);
    let mut engine = ScriptedFfsEngine::new()
        .with_start(0.0)
        .with_flux_script(&[0.5, 2.0])
        .with_trial_script(advance_by_one);
    let estimate = forward_flux_rate(&mut engine, |x| x[0], &cfg).expect("start on lambda_0");
    assert_close(estimate.initial_flux, FLUX_FOUR_OVER_SEVEN);
    assert!(engine.positions()[0].abs() <= f64::EPSILON);
}

fn fail_even_tokens_on_lambda_a(origin: f64, step: usize, token: u64) -> f64 {
    if token & 1 == 0 && step == 1 {
        0.0
    } else {
        origin + 1.0
    }
}

#[test]
fn trial_landing_exactly_on_lambda_a_is_a_failure() {
    // Even-token trials sit exactly on lambda_0 = 0.0 at step 1 and would reach
    // the next interface at step 2 if they were allowed to continue.
    let cfg = scripted_cfg(vec![0.0, 1.0, 2.0], 7);
    let mut engine = ScriptedFfsEngine::new()
        .with_flux_script(&[0.5, 2.0])
        .with_trial_script(fail_even_tokens_on_lambda_a);
    let estimate = forward_flux_rate(&mut engine, |x| x[0], &cfg).expect("estimate");
    assert_eq!(estimate.interface_probabilities, vec![0.5, 0.5]);
    assert_close(estimate.rate, FLUX_FOUR_OVER_SEVEN * 0.25);
}

#[test]
fn burn_in_discards_exactly_the_configured_number_of_steps() {
    // The script alternates a crossing (odd steps) with a product reset, so
    // counting starts at step `burn_in + 1` and the fourth crossing lands on
    // step `first_crossing + 6`.
    let cases = [
        (0_usize, 7_usize, 4.0 / 7.0),
        (1, 9, 4.0 / 8.0),
        (2, 9, 4.0 / 7.0),
        (3, 11, 4.0 / 8.0),
        (4, 11, 4.0 / 7.0),
    ];
    for (burn_in, steps, flux) in cases {
        let cfg = scripted_cfg(vec![0.0, 1.0, 2.0], steps).with_burn_in_steps(burn_in);
        let mut engine = ScriptedFfsEngine::new()
            .with_flux_script(&[0.5, 2.0])
            .with_trial_script(advance_by_one);
        let estimate = forward_flux_rate(&mut engine, |x| x[0], &cfg)
            .unwrap_or_else(|e| panic!("burn_in {burn_in} needs {steps} steps: {e}"));
        assert_close(estimate.initial_flux, flux);

        let mut short = ScriptedFfsEngine::new()
            .with_flux_script(&[0.5, 2.0])
            .with_trial_script(advance_by_one);
        let err = forward_flux_rate(&mut short, |x| x[0], &cfg.with_max_flux_steps(steps - 1))
            .expect_err("one step short");
        assert!(format!("{err}").contains("flux budget exceeded"));
    }
}

/// Stage 0: only the 0.5 starting points advance. Stage 1: odd tokens advance.
fn stage_se_script(origin: f64, _step: usize, token: u64) -> f64 {
    if origin < 1.0 {
        if origin < 0.6 { origin + 1.0 } else { -1.0 }
    } else if token % 2 == 1 {
        origin + 1.0
    } else {
        -1.0
    }
}

#[test]
fn rate_standard_error_propagates_flux_and_every_stage_error() {
    let cfg = scripted_cfg(vec![0.0, 1.0, 2.0], 7).with_seed(3);
    let trials = cfg.trials_per_interface;
    // Crossings alternate 0.5 / 0.7 and never reach the product level.
    let mut engine = ScriptedFfsEngine::new()
        .with_flux_script(&[0.5, -1.0, 0.7, -1.0])
        .with_trial_script(stage_se_script);
    let estimate = forward_flux_rate(&mut engine, |x| x[0], &cfg).expect("estimate");

    // Replay the documented draw order: flux reseed, then (index, reseed) per trial.
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let _flux_seed: u64 = rng.random();
    let mut stage0 = [(0_u64, 0_u64); 4];
    for _ in 0..trials {
        let index = rng.random_range(0..4_usize);
        let _trial_seed: u64 = rng.random();
        stage0[index].0 += 1;
        stage0[index].1 += u64::from(index % 2 == 0);
    }
    let reached = stage0.iter().map(|c| c.1).sum::<u64>() as usize;
    assert!(reached >= 2, "scenario needs two stage-1 start points");
    let mut stage1 = vec![(0_u64, 0_u64); reached];
    for trial in 0..trials {
        let index = rng.random_range(0..reached);
        let _trial_seed: u64 = rng.random();
        stage1[index].0 += 1;
        stage1[index].1 += (trials as u64 + trial as u64 + 1) % 2;
    }

    let stage = |clusters: &[(u64, u64)]| {
        let k = clusters.iter().map(|c| c.0).sum::<u64>() as f64;
        let p = clusters.iter().map(|c| c.1).sum::<u64>() as f64 / k;
        let m = clusters.len() as f64;
        let spread = clusters
            .iter()
            .map(|c| (c.1 as f64 - p * c.0 as f64).powi(2))
            .sum::<f64>();
        (p, (m / (m - 1.0) * spread).sqrt() / k)
    };
    let (p0, se0) = stage(&stage0);
    let (p1, se1) = stage(&stage1);
    assert!(
        se0 > 0.0 && se1 > 0.0,
        "scenario needs nonzero stage errors"
    );
    assert_close(estimate.interface_probabilities[0], p0);
    assert_close(estimate.interface_probabilities[1], p1);

    // Flux intervals [1, 2, 2, 2]: mean 7/4, sample sd 1/2, se of the mean 1/4.
    let flux = 4.0 / 7.0;
    let flux_se = 0.25 / (1.75 * 1.75);
    let rate = flux * p0 * p1;
    let relative_variance = (flux_se / flux).powi(2) + (se0 / p0).powi(2) + (se1 / p1).powi(2);
    assert_close(estimate.initial_flux, flux);
    assert_close(estimate.rate, rate);
    assert_close(estimate.standard_error, rate * relative_variance.sqrt());
    // Each omitted or substituted term would shift the result by far more.
    assert!(estimate.standard_error > rate * (flux_se / flux).hypot(se0 / p0) * 1.01);
}

fn two_interface_engine() -> ScriptedFfsEngine {
    ScriptedFfsEngine::new()
        .with_flux_script(&[0.5, 1.5])
        .with_trial_script(advance_by_one)
}

fn config_error(cfg: &FfsConfig) -> String {
    let mut engine = two_interface_engine();
    let err = forward_flux_rate(&mut engine, |x| x[0], cfg).expect_err("config must be rejected");
    format!("{err}")
}

#[test]
fn interface_count_boundary_is_two_levels() {
    let base = scripted_cfg(vec![0.0, 1.0], 7);
    let mut engine = two_interface_engine();
    let estimate = forward_flux_rate(&mut engine, |x| x[0], &base).expect("two levels are valid");
    assert_eq!(estimate.interface_probabilities, vec![1.0]);
    assert_close(estimate.rate, FLUX_FOUR_OVER_SEVEN);

    for levels in [vec![], vec![0.0]] {
        let msg = config_error(&base.clone().with_interfaces(levels));
        assert!(msg.contains("at least two levels"), "{msg}");
    }
}

#[test]
fn trials_per_interface_boundary_is_one() {
    let base = scripted_cfg(vec![0.0, 1.0], 7);
    let mut engine = two_interface_engine();
    let estimate = forward_flux_rate(
        &mut engine,
        |x| x[0],
        &base.clone().with_trials_per_interface(1),
    )
    .expect("a single trial is valid");
    assert_eq!(estimate.interface_probabilities, vec![1.0]);

    let msg = config_error(&base.with_trials_per_interface(0));
    assert!(
        msg.contains("trials_per_interface must be positive"),
        "{msg}"
    );
}

#[test]
fn step_budgets_must_each_be_positive() {
    let base = scripted_cfg(vec![0.0, 1.0], 7);
    let msg = config_error(&base.clone().with_max_trial_steps(0));
    assert!(msg.contains("step budgets must be positive"), "{msg}");
    let msg = config_error(&base.clone().with_max_flux_steps(0));
    assert!(msg.contains("step budgets must be positive"), "{msg}");

    let mut engine = two_interface_engine();
    let estimate = forward_flux_rate(&mut engine, |x| x[0], &base.clone().with_max_trial_steps(1))
        .expect("one trial step is a valid budget");
    assert_eq!(estimate.interface_probabilities, vec![1.0]);

    let msg = config_error(&base.with_max_flux_steps(1));
    assert!(msg.contains("flux budget exceeded"), "{msg}");
}

#[test]
fn time_step_must_be_finite_and_positive() {
    let base = scripted_cfg(vec![0.0, 1.0], 7);
    for dt in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let msg = config_error(&base.clone().with_time_step(dt));
        assert!(
            msg.contains("time_step must be finite and positive"),
            "{dt}: {msg}"
        );
    }
}

fn slow_commitment(origin: f64, step: usize, _token: u64) -> f64 {
    if step >= 2 {
        origin + 1.0
    } else {
        origin + 0.25
    }
}

#[test]
fn trial_budget_is_exhausted_exactly_when_commitment_takes_longer() {
    let cfg = scripted_cfg(vec![0.0, 1.0, 2.0], 7);
    let run = |max_trial_steps: usize| {
        let mut engine = ScriptedFfsEngine::new()
            .with_flux_script(&[0.5, 2.0])
            .with_trial_script(slow_commitment);
        forward_flux_rate(
            &mut engine,
            |x| x[0],
            &cfg.clone().with_max_trial_steps(max_trial_steps),
        )
    };
    let estimate = run(2).expect("commitment arrives on step 2");
    assert_eq!(estimate.interface_probabilities, vec![1.0, 1.0]);
    let err = run(1).expect_err("commitment arrives after step 1");
    assert!(format!("{err}").contains("trial budget exceeded"));
}

#[test]
fn last_stage_does_not_capture_reached_configurations() {
    let cfg = scripted_cfg(vec![0.0, 1.0, 2.0], 7);
    let mut engine = ScriptedFfsEngine::new()
        .with_flux_script(&[0.5, 2.0])
        .with_trial_script(advance_by_one);
    let _ = forward_flux_rate(&mut engine, |x| x[0], &cfg).expect("estimate");
    // One initial capture, four flux crossings, and eight stage-0 successes;
    // the final stage only counts successes.
    assert_eq!(engine.velocity_reads.get(), 1 + 4 + 8);
}
