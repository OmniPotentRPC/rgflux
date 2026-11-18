//! Direct Forward Flux Sampling tests.

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
    trial_step: usize,
    trial_origin_lambda: f64,
}

impl ScriptedFfsEngine {
    fn new() -> Self {
        Self {
            masses: Array1::from_vec(vec![1.0]),
            positions: Array1::from_vec(vec![-1.2]),
            velocities: Array1::from_vec(vec![0.0]),
            thermostat: ThermostatConfig::default(),
            constraint: None,
            mode: ScriptMode::Flux,
            flux_seed: 0,
            flux_seeded: false,
            flux_step: 0,
            trial_token: 0,
            trial_step: 0,
            trial_origin_lambda: -1.2,
        }
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
        self.positions.assign(&positions);
        self.trial_origin_lambda = self.positions[0];
        self.trial_step = 0;
        Ok(())
    }

    fn velocities(&self) -> ArrayView1<'_, f64> {
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
        match self.mode {
            ScriptMode::Flux => {
                let lambda = if self.flux_seed & 1 == 0 {
                    // Intervals: 1, 2, 2, 2, ...
                    if self.flux_step.is_multiple_of(2) {
                        0.2
                    } else {
                        -1.2
                    }
                } else {
                    // Intervals: 1, 3, 3, 3, ...
                    let rem = self.flux_step % 3;
                    if rem == 0 { 0.2 } else { -1.2 }
                };
                self.positions[0] = lambda;
                self.flux_step += 1;
            }
            ScriptMode::Trial => {
                self.trial_step += 1;
                let in_stage_zero = self.trial_origin_lambda < 0.4;
                let token_mod = self.trial_token % 4;
                let success_step = if in_stage_zero {
                    token_mod != 0
                } else {
                    token_mod == 0
                };
                self.positions[0] = if success_step {
                    if in_stage_zero { 0.5 } else { 1.5 }
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
            self.trial_step = 0;
            self.mode = ScriptMode::Trial;
            let _ = seed;
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
    let mut engine = ScriptedFfsEngine::new();
    let cfg = FfsConfig::default()
        .with_interfaces(vec![-1.0, 0.0, 1.0])
        .with_points_per_interface(4)
        .with_trials_per_interface(8)
        .with_max_trial_steps(3)
        .with_max_flux_steps(64)
        .with_time_step(1.0)
        .with_seed(123);
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
    assert!(expected_flux_se.is_finite());
}

#[test]
fn scripted_engine_seed_changes_flux_phase() {
    let mut even_seed = None;
    let mut odd_seed = None;
    for seed in 1_u64..=512 {
        let mut rng = StdRng::seed_from_u64(seed);
        let flux_seed = rng.random::<u64>();
        if flux_seed & 1 == 0 && even_seed.is_none() {
            even_seed = Some(seed);
        } else if flux_seed & 1 == 1 && odd_seed.is_none() {
            odd_seed = Some(seed);
        }
        if even_seed.is_some() && odd_seed.is_some() {
            break;
        }
    }
    let seed_even = even_seed.expect("at least one even flux seed");
    let seed_odd = odd_seed.expect("at least one odd flux seed");

    let cfg_a = FfsConfig::default()
        .with_interfaces(vec![-1.0, 0.0, 1.0])
        .with_points_per_interface(4)
        .with_trials_per_interface(8)
        .with_max_trial_steps(3)
        .with_max_flux_steps(64)
        .with_time_step(1.0)
        .with_seed(seed_even);
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
        .with_seed(99);
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
