//! Multithermal OPES on a 2D double well driven by the native engine.
//!
//! The well `U = (x^2 - 1)^2 + y^2 / 2` separates, so every reference value
//! is a one-dimensional quadrature in `x` times a Gaussian in `y`.

use ndarray::{Array1, ArrayView1};
use rgflux::FluxError;
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::engine::{DynamicsEngine, ThermostatConfig};
use rgflux::opes::{
    MultithermalReweighting, OpesMultithermal, OpesMultithermalConfig, OpesMultithermalSurface,
    OpesRunConfig, OpesTrajectory, SharedOpesMultithermal, run_opes_multithermal,
};
use rgflux::stats::StatsConfig;

const T0: f64 = 0.3;
const T_MAX: f64 = 1.2;

struct DoubleWell2D;

impl rgsaddle::PointSurface for DoubleWell2D {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
        let a = x[0] * x[0] - 1.0;
        Ok((
            a * a + 0.5 * x[1] * x[1],
            Array1::from_vec(vec![4.0 * x[0] * a, x[1]]),
        ))
    }
}

/// Simpson moments `(Z, <x^2>, <U_x>)` of `exp(-beta (x^2 - 1)^2)` on `[-4, 4]`.
fn x_moments(beta: f64) -> (f64, f64, f64) {
    let n = 40_000;
    let h = 8.0 / n as f64;
    let (mut z, mut x2, mut ux) = (0.0, 0.0, 0.0);
    for i in 0..=n {
        let x = -4.0 + i as f64 * h;
        let u = (x * x - 1.0).powi(2);
        let w = if i == 0 || i == n {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        let p = w * (-beta * u).exp();
        z += p;
        x2 += p * x * x;
        ux += p * u;
    }
    (z * h / 3.0, x2 / z, ux / z)
}

fn exact_delta_f(temperature: f64) -> f64 {
    let (beta, beta0) = (1.0 / temperature, 1.0 / T0);
    let ln_z = |b: f64| x_moments(b).0.ln() + 0.5 * (2.0 * std::f64::consts::PI / b).ln();
    -T0 * (ln_z(beta) - ln_z(beta0))
}

fn exact_mean_energy(temperature: f64) -> f64 {
    x_moments(1.0 / temperature).2 + 0.5 * temperature
}

fn run(seed: u64, steps: usize, config: OpesMultithermalConfig) -> OpesTrajectory {
    let thermostat = ThermostatConfig::default()
        .with_temperature(T0)
        .with_gamma(1.0);
    let shared =
        SharedOpesMultithermal::new(OpesMultithermal::new(config, &thermostat).expect("bias"));
    let surface = OpesMultithermalSurface::new(DoubleWell2D, shared.clone());
    let native = NativeConfig::default()
        .with_masses(Array1::from_vec(vec![1.0, 1.0]))
        .with_thermostat(thermostat)
        .with_seed(seed);
    let mut engine = NativeEngine::new(surface, native).expect("engine");
    engine
        .set_positions(Array1::from_vec(vec![-1.0, 0.0]).view())
        .expect("start");
    engine.shoot_velocities(seed).expect("velocities");
    let run = OpesRunConfig::default()
        .with_steps(steps)
        .with_time_step(0.01)
        .with_frame_stride(50)
        .with_record_positions(true);
    run_opes_multithermal(&mut engine, &shared, &run).expect("run")
}

fn ladder() -> OpesMultithermalConfig {
    OpesMultithermalConfig::range(T0, T_MAX)
        .with_temp_steps(6)
        .with_pace(50)
        .with_observation_steps(100)
}

#[test]
fn delta_f_converges_to_the_quadrature_reference() {
    let trajectory = run(7, 2_000_000, ladder());
    let temps = trajectory.temperatures();
    let mut worst = 0.0_f64;
    for (t, df) in temps.iter().zip(trajectory.delta_f()) {
        let exact = exact_delta_f(*t);
        let error = (df - exact).abs() / T0;
        eprintln!("T = {t:.6}  OPES {df:+.6}  exact {exact:+.6}  |error|/kT0 = {error:.2e}");
        worst = worst.max(error);
    }
    assert!(worst < 0.03, "largest Delta F error {worst} kT0");
    let last = trajectory.delta_f_history.last().expect("history");
    assert_eq!(last.delta_f, trajectory.delta_f());
}

#[test]
fn reweighted_energies_and_frame_weights_match_the_reference() {
    let trajectory = run(11, 2_000_000, ladder());
    let n_frames = trajectory.potential_energies.len();
    assert_eq!(trajectory.positions.len(), n_frames);
    let burn_in = n_frames / 10;
    let rw = MultithermalReweighting::from_bias(
        &trajectory.bias,
        trajectory.potential_energies[burn_in..].to_vec(),
        trajectory.biases[burn_in..].to_vec(),
    )
    .expect("reweighting");
    let x2: Vec<f64> = trajectory.positions[burn_in..]
        .iter()
        .map(|p| p[0] * p[0])
        .collect();
    let stats = StatsConfig::default();
    for target in [T0, 0.45, 0.6, 0.8, 1.0, T_MAX] {
        let weights: Vec<f64> = rw.weights(target).expect("weights");
        assert_eq!(weights.len(), rw.len());
        assert!((weights.iter().sum::<f64>() - 1.0).abs() <= 1e-12);
        let energy = rw
            .average(target, &trajectory.potential_energies[burn_in..], &stats)
            .expect("energy");
        let exact_u = exact_mean_energy(target);
        let exact_x2 = x_moments(1.0 / target).1;
        let mean_x2 = weighted_mean(&weights, &x2);
        eprintln!(
            "T = {target:.3}  <U> {:.5} +- {:.5} (exact {exact_u:.5})  <x^2> {mean_x2:.5} (exact {exact_x2:.5})  Kish {:.0}",
            energy.mean, energy.standard_error, energy.effective_sample_size
        );
        assert!(
            (energy.mean - exact_u).abs() <= 4.0 * energy.standard_error + 1e-3,
            "<U> at T = {target}"
        );
        assert!((mean_x2 - exact_x2).abs() <= 0.02, "<x^2> at T = {target}");
        let df = rw.delta_f(target).expect("frame Delta F");
        assert!(
            (df - exact_delta_f(target)).abs() <= 0.03 * T0,
            "frame Delta F at T = {target}"
        );
    }
}

/// Downstream covariance and clustering code takes the weights as a slice.
fn weighted_mean(weights: &[f64], values: &[f64]) -> f64 {
    weights.iter().zip(values).map(|(w, v)| w * v).sum()
}

#[test]
fn frame_weights_align_with_stored_frames_for_every_target_in_range() {
    let trajectory = run(3, 40_000, ladder());
    let frames = trajectory.steps.len();
    assert_eq!(frames, 40_000 / 50 + 1);
    for target in [T0, 0.5, T_MAX] {
        let weights = trajectory.weights(target).expect("weights");
        assert_eq!(weights.len(), frames);
        assert!(weights.iter().all(|w| w.is_finite() && *w >= 0.0));
    }
    assert!(matches!(
        trajectory.weights(0.9 * T0),
        Err(FluxError::InvalidConfig(_))
    ));
    assert!(matches!(
        trajectory.weights(1.1 * T_MAX),
        Err(FluxError::InvalidConfig(_))
    ));
}

#[test]
fn biased_energy_and_force_stay_consistent_along_the_run() {
    let trajectory = run(5, 20_000, ladder());
    for ((u, v), s) in trajectory
        .potential_energies
        .iter()
        .zip(&trajectory.biases)
        .zip(&trajectory.force_scales)
    {
        let eval = trajectory.bias.bias(*u).expect("bias");
        assert!(v.is_finite() && s.is_finite());
        assert!(eval.force_scale() > 0.0);
    }
    let beta0 = 1.0 / T0;
    let beta_max = 1.0 / T_MAX;
    for &s in &trajectory.force_scales {
        assert!(s <= 1.0 + 1e-12 && s >= beta_max / beta0 - 1e-12);
    }
}

#[test]
fn restart_from_a_state_continues_the_same_bias() {
    let thermostat = ThermostatConfig::default()
        .with_temperature(T0)
        .with_gamma(1.0);
    let first = run(9, 30_000, ladder());
    let mut state = Vec::new();
    first.bias.write_state(&mut state, 300.0).expect("write");
    let restored =
        OpesMultithermal::from_state(ladder(), &thermostat, &state[..]).expect("restore");
    assert_eq!(restored.delta_f(), first.bias.delta_f());
    assert_eq!(restored.counter(), first.bias.counter());
    for u in [0.0, 0.4, 1.3] {
        assert_eq!(
            restored.bias(u).expect("restored"),
            first.bias.bias(u).expect("original")
        );
    }
}

fn engine_over(
    shared: &SharedOpesMultithermal,
    thermostat: ThermostatConfig,
    seed: u64,
) -> NativeEngine<OpesMultithermalSurface<DoubleWell2D>> {
    let surface = OpesMultithermalSurface::new(DoubleWell2D, shared.clone());
    let native = NativeConfig::default()
        .with_masses(Array1::from_vec(vec![1.0, 1.0]))
        .with_thermostat(thermostat)
        .with_seed(seed);
    let mut engine = NativeEngine::new(surface, native).expect("engine");
    engine
        .set_positions(Array1::from_vec(vec![-1.0, 0.0]).view())
        .expect("start");
    engine
}

#[test]
fn restart_continues_at_the_stored_step_and_rejects_another_start() {
    let thermostat = ThermostatConfig::default()
        .with_temperature(T0)
        .with_gamma(1.0);
    let first = run(9, 30_025, ladder());
    assert_eq!(first.bias.last_step(), Some(30_025));
    let mut state = Vec::new();
    first.bias.write_state(&mut state, 300.25).expect("write");
    let restored =
        OpesMultithermal::from_state(ladder(), &thermostat, &state[..]).expect("restore");
    assert_eq!(restored.last_step(), Some(30_025));

    let shared = SharedOpesMultithermal::new(restored.clone());
    let mut engine = engine_over(&shared, thermostat, 4);
    let shifted = OpesRunConfig::default().with_steps(100).with_start_step(0);
    assert_eq!(
        run_opes_multithermal(&mut engine, &shared, &shifted).err(),
        Some(FluxError::InvalidConfig(
            "OpesRunConfig::start_step differs from the last MD step of the OPES bias"
        ))
    );
    let resumed = run_opes_multithermal(
        &mut engine,
        &shared,
        &OpesRunConfig::default().with_steps(100),
    )
    .expect("resume");
    assert_eq!(resumed.steps[0], 30_025);
    assert_eq!(resumed.last_step, 30_125);
    assert_eq!(resumed.bias.counter(), first.bias.counter() + 2);
    let explicit = run_opes_multithermal(
        &mut engine,
        &shared,
        &OpesRunConfig::default()
            .with_steps(100)
            .with_start_step(30_125),
    )
    .expect("explicit start at the stored step");
    assert_eq!(explicit.steps[0], 30_125);
    assert_eq!(explicit.bias.counter(), first.bias.counter() + 4);

    let text = String::from_utf8(state).expect("utf8");
    let plumed: String = text
        .lines()
        .filter(|line| !line.starts_with("# rgflux"))
        .map(|line| format!("{line}\n"))
        .collect();
    assert!(matches!(
        OpesMultithermal::from_state(ladder(), &thermostat, plumed.as_bytes()),
        Err(FluxError::InvalidConfig(_))
    ));
    let from_plumed = OpesMultithermal::from_plumed_state(
        ladder(),
        &thermostat,
        plumed.as_bytes(),
        T0,
        thermostat.boltzmann,
    )
    .expect("restore");
    assert_eq!(from_plumed.last_step(), None);
    let shared = SharedOpesMultithermal::new(from_plumed);
    let mut engine = engine_over(&shared, thermostat, 4);
    assert_eq!(
        run_opes_multithermal(
            &mut engine,
            &shared,
            &OpesRunConfig::default().with_steps(100)
        )
        .err(),
        Some(FluxError::InvalidConfig(
            "OPES state has no MD step; set OpesRunConfig::start_step"
        ))
    );
    let resumed = run_opes_multithermal(
        &mut engine,
        &shared,
        &OpesRunConfig::default()
            .with_steps(100)
            .with_start_step(30_025),
    )
    .expect("explicit start from a PLUMED state");
    assert_eq!(resumed.steps[0], 30_025);
}

#[test]
fn restart_refuses_a_changed_temperature_or_boltzmann_constant() {
    let thermostat = ThermostatConfig::default()
        .with_temperature(T0)
        .with_gamma(1.0);
    let first = run(9, 30_000, ladder());
    let mut state = Vec::new();
    first.bias.write_state(&mut state, 300.0).expect("write");
    let restore = |config: OpesMultithermalConfig, thermostat: &ThermostatConfig| {
        OpesMultithermal::from_state(config, thermostat, &state[..]).err()
    };
    assert_eq!(restore(ladder(), &thermostat), None);
    assert_eq!(
        restore(ladder(), &thermostat.with_temperature(0.35)),
        Some(FluxError::InvalidConfig(
            "OPES state was written at a different simulation temperature T0"
        ))
    );
    assert_eq!(
        restore(ladder(), &thermostat.with_boltzmann(2.0)),
        Some(FluxError::InvalidConfig(
            "OPES state was written with a different Boltzmann constant k_B"
        ))
    );
    assert_eq!(
        restore(ladder().with_temp_max(1.200_000_1), &thermostat),
        Some(FluxError::InvalidConfig(
            "OPES state was written for a different temperature range"
        ))
    );

    let text = String::from_utf8(state.clone()).expect("utf8");
    let plumed: String = text
        .lines()
        .filter(|line| !line.starts_with("# rgflux"))
        .map(|line| format!("{line}\n"))
        .collect();
    let warmer = thermostat.with_temperature(0.35);
    assert_eq!(
        OpesMultithermal::from_state(ladder(), &warmer, plumed.as_bytes()).err(),
        Some(FluxError::InvalidConfig(
            "OPES state has no rgflux T0 and k_B entries; read a PLUMED state with from_plumed_state"
        ))
    );
    assert_eq!(
        OpesMultithermal::from_plumed_state(ladder(), &warmer, plumed.as_bytes(), T0, 1.0).err(),
        Some(FluxError::InvalidConfig(
            "OPES state was written at a different simulation temperature T0"
        ))
    );
    assert_eq!(
        OpesMultithermal::from_plumed_state(
            ladder(),
            &thermostat.with_boltzmann(2.0),
            plumed.as_bytes(),
            T0,
            1.0
        )
        .err(),
        Some(FluxError::InvalidConfig(
            "OPES state was written with a different Boltzmann constant k_B"
        ))
    );
}

#[test]
fn run_refuses_a_thermostat_away_from_the_bias_temperature() {
    let bias_thermostat = ThermostatConfig::default().with_temperature(T0);
    let shared = SharedOpesMultithermal::new(
        OpesMultithermal::new(ladder(), &bias_thermostat).expect("bias"),
    );
    let surface = OpesMultithermalSurface::new(DoubleWell2D, shared.clone());
    let native = NativeConfig::default()
        .with_masses(Array1::from_vec(vec![1.0, 1.0]))
        .with_thermostat(ThermostatConfig::default().with_temperature(2.0 * T0));
    let mut engine = NativeEngine::new(surface, native).expect("engine");
    let err = run_opes_multithermal(&mut engine, &shared, &OpesRunConfig::default())
        .expect_err("temperature mismatch");
    assert!(matches!(err, FluxError::InvalidConfig(_)));
}

#[test]
fn run_refuses_an_engine_that_bypasses_the_surface() {
    let thermostat = ThermostatConfig::default().with_temperature(T0);
    let shared =
        SharedOpesMultithermal::new(OpesMultithermal::new(ladder(), &thermostat).expect("bias"));
    let native = NativeConfig::default()
        .with_masses(Array1::from_vec(vec![1.0, 1.0]))
        .with_thermostat(thermostat);
    let mut engine = NativeEngine::new(DoubleWell2D, native).expect("engine");
    let err = run_opes_multithermal(&mut engine, &shared, &OpesRunConfig::default())
        .expect_err("unbiased engine");
    assert!(matches!(err, FluxError::InvalidConfig(_)));
}
