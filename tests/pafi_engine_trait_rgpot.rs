#![cfg(feature = "rgpot")]

mod pafi_support;

use std::os::raw::c_void;

use ndarray::{Array1, Array2, ArrayView1};
use pafi_support::{assert_below, assert_coverage, block_stats, seed, widening_barrier_se_cap_at};
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::engine::rgpot::{RgpotConfig, RgpotSurface};
use rgflux::engine::{DynamicsEngine, ThermostatConfig};
use rgflux::pafi::{
    FreeEnergyProfile, HermiteIntegral, MeanForceIntegral, PafiConfig, pafi_profile,
    pafi_profile_dyn,
};
use rgflux::stats::StatsConfig;
use rgpot_core::potential::PotentialImpl;
use rgpot_core::status::rgpot_status_t;
use rgpot_core::tensor::rgpot_tensor_cpu_f64_2d;
use rgpot_core::types::{rgpot_force_input_t, rgpot_force_out_t};

struct ThreeDWell;

impl rgsaddle::PointSurface for ThreeDWell {
    fn eval(&self, r: ArrayView1<'_, f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
        let x = r[0];
        let y = r[1];
        let z = r[2];
        let well = (1.0 - x * x).powi(2);
        let ky = 1.5 + 0.5 * x;
        let kz = 2.0_f64;
        let energy = well + 0.5 * ky * y * y + 0.5 * kz * z * z;
        let gx = -4.0 * x * (1.0 - x * x) + 0.25 * y * y;
        let gy = ky * y;
        let gz = kz * z;
        Ok((energy, Array1::from_vec(vec![gx, gy, gz])))
    }
}

struct CallbackState {
    forces: Vec<f64>,
}

unsafe extern "C" fn drop_state(ptr: *mut c_void) {
    if !ptr.is_null() {
        let _ = unsafe { Box::from_raw(ptr as *mut CallbackState) };
    }
}

unsafe extern "C" fn three_d_well_callback(
    user_data: *mut c_void,
    input: *const rgpot_force_input_t,
    output: *mut rgpot_force_out_t,
) -> rgpot_status_t {
    if user_data.is_null() || input.is_null() || output.is_null() {
        return rgpot_status_t::RGPOT_INVALID_PARAMETER;
    }
    let state = unsafe { &mut *(user_data as *mut CallbackState) };
    let inp = unsafe { &*input };
    let out = unsafe { &mut *output };
    let Some(n_atoms) = (unsafe { inp.n_atoms() }) else {
        return rgpot_status_t::RGPOT_INVALID_PARAMETER;
    };
    if n_atoms != 1 {
        return rgpot_status_t::RGPOT_INVALID_PARAMETER;
    }
    let dl = unsafe { &(*inp.positions).dl_tensor };
    let pos = unsafe { std::slice::from_raw_parts(dl.data as *const f64, 3) };
    let x = pos[0];
    let y = pos[1];
    let z = pos[2];

    let well = (1.0 - x * x).powi(2);
    let ky = 1.5 + 0.5 * x;
    let kz = 2.0_f64;
    out.energy = well + 0.5 * ky * y * y + 0.5 * kz * z * z;
    out.variance = 0.0;
    state.forces[0] = -(-4.0 * x * (1.0 - x * x) + 0.25 * y * y);
    state.forces[1] = -(ky * y);
    state.forces[2] = -(kz * z);
    out.forces = unsafe { rgpot_tensor_cpu_f64_2d(state.forces.as_mut_ptr(), 1, 3) };
    rgpot_status_t::RGPOT_SUCCESS
}

fn path_x_only(start: f64, end: f64, planes: usize) -> Array2<f64> {
    let mut path = Array2::<f64>::zeros((planes, 3));
    for i in 0..planes {
        let t = (i as f64) / ((planes - 1) as f64);
        path[[i, 0]] = start + (end - start) * t;
    }
    path
}

fn pafi_config(seed: u64) -> PafiConfig {
    PafiConfig::default()
        .with_planes(61)
        .with_burn_in_steps(20_000)
        .with_sample_steps(64)
        .with_time_step(0.001)
        .with_stats(StatsConfig::default().with_min_batches(2))
        .with_seed(seed)
}

#[test]
fn pafi_trait_object_matches_generic_on_native_engine() {
    let path = path_x_only(-1.0, 0.0, 61);
    let config = pafi_config(77);
    let mut engine = NativeEngine::new(
        ThreeDWell,
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0, 4.0, 2.0]))
            .with_seed(77)
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(1.3)
                    .with_temperature(0.02)
                    .with_boltzmann(1.0),
            ),
    )
    .expect("native engine");
    engine.shoot_velocities(77).expect("initial velocities");
    let generic = pafi_profile(&mut engine, path.view(), &config).expect("generic profile");
    let dyn_engine: &mut dyn DynamicsEngine = &mut engine;
    let dynamic = pafi_profile_dyn(dyn_engine, path.view(), &config).expect("dynamic profile");
    assert_eq!(generic.dividing_index, dynamic.dividing_index);
    assert!((generic.barrier - dynamic.barrier).abs() <= 1.0e-12);
    assert!(
        (generic.normalized_rate.expect("generic rate")
            - dynamic.normalized_rate.expect("dynamic rate"))
        .abs()
            <= 1.0e-12
    );
}

#[test]
fn pafi_runs_unchanged_on_native_and_rgpot_backed_engines() {
    let path = path_x_only(-1.0, 0.0, 61);
    let config = pafi_config(9);

    let mut native = NativeEngine::new(
        ThreeDWell,
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0, 4.0, 2.0]))
            .with_seed(9)
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(1.3)
                    .with_temperature(0.02)
                    .with_boltzmann(1.0),
            ),
    )
    .expect("native engine");
    native
        .shoot_velocities(9)
        .expect("native initial velocities");
    let native_profile = pafi_profile_dyn(&mut native, path.view(), &config).expect("native pafi");

    let state = Box::new(CallbackState {
        forces: vec![0.0; 3],
    });
    let potential = PotentialImpl::new(
        three_d_well_callback,
        Box::into_raw(state).cast::<c_void>(),
        Some(drop_state),
    );
    let rgpot_surface = RgpotSurface::new(
        potential,
        RgpotConfig::default().with_atomic_numbers(vec![29]),
    )
    .expect("rgpot surface");
    let mut rgpot_engine = NativeEngine::new(
        rgpot_surface,
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0, 4.0, 2.0]))
            .with_seed(9)
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(1.3)
                    .with_temperature(0.02)
                    .with_boltzmann(1.0),
            ),
    )
    .expect("rgpot-backed native engine");
    rgpot_engine
        .shoot_velocities(9)
        .expect("rgpot initial velocities");
    let rgpot_profile =
        pafi_profile_dyn(&mut rgpot_engine, path.view(), &config).expect("rgpot pafi");

    let barrier_sigma =
        (native_profile.barrier_error.powi(2) + rgpot_profile.barrier_error.powi(2)).sqrt();
    let z_barrier = (native_profile.barrier - rgpot_profile.barrier).abs() / barrier_sigma;
    assert!(
        z_barrier <= 2.0,
        "barrier native {:.8} ± {:.8}, rgpot {:.8} ± {:.8}, z={:.4}",
        native_profile.barrier,
        native_profile.barrier_error,
        rgpot_profile.barrier,
        rgpot_profile.barrier_error,
        z_barrier
    );

    let native_rate = native_profile
        .normalized_rate
        .expect("native normalized rate");
    let native_rate_error = native_profile
        .normalized_rate_error
        .expect("native normalized rate error");
    let rgpot_rate = rgpot_profile
        .normalized_rate
        .expect("rgpot normalized rate");
    let rgpot_rate_error = rgpot_profile
        .normalized_rate_error
        .expect("rgpot normalized rate error");
    let rate_sigma = (native_rate_error.powi(2) + rgpot_rate_error.powi(2)).sqrt();
    let z_rate = (native_rate - rgpot_rate).abs() / rate_sigma;
    assert!(
        z_rate <= 2.0,
        "rate native {native_rate:.8e} ± {native_rate_error:.8e}, rgpot {rgpot_rate:.8e} ± {rgpot_rate_error:.8e}, z={z_rate:.4}"
    );
}

const COVERAGE_PLANES: usize = 21;
const COVERAGE_TEMPERATURE: f64 = 0.02;
const COVERAGE_SAMPLES: usize = 16_000;
const COVERAGE_MASS_Y: f64 = 4.0;
const COVERAGE_GAMMA: f64 = 1.3;

fn assert_close(lhs: f64, rhs: f64) {
    let tolerance = 1024.0 * f64::EPSILON * lhs.abs().max(rhs.abs()).max(1.0);
    assert!(
        (lhs - rhs).abs() <= tolerance,
        "parity violated: {lhs:.16e} vs {rhs:.16e}"
    );
}

fn assert_option_close(lhs: Option<f64>, rhs: Option<f64>) {
    match (lhs, rhs) {
        (Some(left), Some(right)) => assert_close(left, right),
        (None, None) => {}
        _ => panic!("option parity violated: {lhs:?} vs {rhs:?}"),
    }
}

fn assert_profiles_match(lhs: &FreeEnergyProfile, rhs: &FreeEnergyProfile) {
    assert_eq!(lhs.dividing_index, rhs.dividing_index);
    assert_close(lhs.dividing_arc, rhs.dividing_arc);
    assert_close(lhs.barrier, rhs.barrier);
    assert_close(lhs.barrier_error, rhs.barrier_error);
    assert_close(lhs.reactant_frequency, rhs.reactant_frequency);
    assert_option_close(lhs.reactant_partition, rhs.reactant_partition);
    assert_close(lhs.reactant_log_partition, rhs.reactant_log_partition);
    assert_option_close(lhs.normalized_rate, rhs.normalized_rate);
    assert_close(lhs.normalized_log_rate, rhs.normalized_log_rate);
    assert_option_close(lhs.normalized_rate_error, rhs.normalized_rate_error);
    assert_close(lhs.quadrature_residual, rhs.quadrature_residual);
    for (left, right) in lhs.arc_lengths.iter().zip(&rhs.arc_lengths) {
        assert_close(*left, *right);
    }
    for (left, right) in lhs.free_energy.iter().zip(&rhs.free_energy) {
        assert_close(*left, *right);
    }
    for (left, right) in lhs.standard_error.iter().zip(&rhs.standard_error) {
        assert_close(*left, *right);
    }
}

fn hermite_barrier(planes: usize, thermal: f64) -> f64 {
    let denominator = (planes - 1) as f64;
    let arcs: Vec<f64> = (0..planes).map(|i| (i as f64) / denominator).collect();
    let gradients: Vec<f64> = arcs
        .iter()
        .map(|arc| {
            let x = -1.0 + *arc;
            -4.0 * x * (1.0 - x * x) + 0.25 * thermal / (1.5 + 0.5 * x)
        })
        .collect();
    HermiteIntegral
        .cumulative(&arcs, &gradients)
        .expect("rgpot hermite barrier")
        .values[planes - 1]
}

fn coverage_engine(seed_value: u64) -> NativeEngine<ThreeDWell> {
    let mut engine = NativeEngine::new(
        ThreeDWell,
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0, COVERAGE_MASS_Y, 2.0]))
            .with_seed(seed_value)
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(COVERAGE_GAMMA)
                    .with_temperature(COVERAGE_TEMPERATURE)
                    .with_boltzmann(1.0),
            ),
    )
    .expect("native engine");
    engine
        .shoot_velocities(seed_value)
        .expect("initial velocities");
    engine
}

fn coverage_rgpot_engine(seed_value: u64) -> NativeEngine<RgpotSurface> {
    let state = Box::new(CallbackState {
        forces: vec![0.0; 3],
    });
    let potential = PotentialImpl::new(
        three_d_well_callback,
        Box::into_raw(state).cast::<c_void>(),
        Some(drop_state),
    );
    let surface = RgpotSurface::new(
        potential,
        RgpotConfig::default().with_atomic_numbers(vec![29]),
    )
    .expect("rgpot surface");
    let mut engine = NativeEngine::new(
        surface,
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0, COVERAGE_MASS_Y, 2.0]))
            .with_seed(seed_value)
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(COVERAGE_GAMMA)
                    .with_temperature(COVERAGE_TEMPERATURE)
                    .with_boltzmann(1.0),
            ),
    )
    .expect("rgpot-backed native engine");
    engine
        .shoot_velocities(seed_value)
        .expect("rgpot initial velocities");
    engine
}

/// The base engine tests keep the sixty-one plane burn-in. The same potential
/// and the same seeds must also agree on the whole profile to rounding, on a
/// short run and on the sampled profile used for the Hermite coverage check.
#[test]
fn pafi_trait_object_matches_generic_to_rounding() {
    let path = path_x_only(-1.0, 0.0, COVERAGE_PLANES);
    let seed_value = seed(77);
    let config = PafiConfig::default()
        .with_planes(COVERAGE_PLANES)
        .with_burn_in_steps(100)
        .with_sample_steps(300)
        .with_time_step(0.05)
        .with_stats(StatsConfig::default().with_min_batches(2))
        .with_seed(seed_value);
    let mut engine = coverage_engine(seed_value);
    let generic = pafi_profile(&mut engine, path.view(), &config).expect("generic profile");
    let dyn_engine: &mut dyn DynamicsEngine = &mut engine;
    let dynamic = pafi_profile_dyn(dyn_engine, path.view(), &config).expect("dynamic profile");
    assert_profiles_match(&generic, &dynamic);
}

#[test]
fn pafi_native_and_rgpot_profiles_agree_to_rounding() {
    let path = path_x_only(-1.0, 0.0, COVERAGE_PLANES);
    let seed_value = seed(9);
    let config = PafiConfig::default()
        .with_planes(COVERAGE_PLANES)
        .with_burn_in_steps(100)
        .with_sample_steps(300)
        .with_time_step(0.05)
        .with_stats(StatsConfig::default().with_min_batches(2))
        .with_seed(seed_value);
    let mut native = coverage_engine(seed_value);
    let native_profile = pafi_profile(&mut native, path.view(), &config).expect("native profile");
    let mut rgpot = coverage_rgpot_engine(seed_value);
    let rgpot_profile = pafi_profile(&mut rgpot, path.view(), &config).expect("rgpot profile");
    assert_profiles_match(&native_profile, &rgpot_profile);
}

#[test]
fn pafi_rgpot_barrier_covers_hermite_expectation() {
    let path = path_x_only(-1.0, 0.0, COVERAGE_PLANES);
    let seed_value = seed(41);
    let config = PafiConfig::default()
        .with_planes(COVERAGE_PLANES)
        .with_burn_in_steps(200)
        .with_sample_steps(COVERAGE_SAMPLES)
        .with_time_step(0.05)
        .with_stats(block_stats(COVERAGE_SAMPLES))
        .with_seed(seed_value);
    let mut rgpot = coverage_rgpot_engine(seed_value);
    let profile = pafi_profile(&mut rgpot, path.view(), &config).expect("rgpot profile");
    let last = profile.free_energy.len() - 1;
    let barrier = profile.free_energy[last];
    let se = profile.standard_error[last];
    let expected = hermite_barrier(COVERAGE_PLANES, COVERAGE_TEMPERATURE);
    assert_below(
        "rgpot barrier SE",
        se,
        widening_barrier_se_cap_at(
            COVERAGE_SAMPLES,
            COVERAGE_MASS_Y,
            COVERAGE_GAMMA,
            COVERAGE_TEMPERATURE,
        ),
    );
    assert_coverage("rgpot barrier", barrier, se, expected);
}
