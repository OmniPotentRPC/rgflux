#![cfg(feature = "rgpot")]
//! PAFI parity and analytic checks across native and `rgpot` engines.
//!
//! The parity checks are deterministic. The analytic barrier check is
//! statistical and uses the budget derived in `tests/pafi_support/mod.rs`.
//! Reproducible with
//! `cargo test --features rgpot --test pafi_engine_trait_rgpot -- --nocapture`;
//! set `PAFI_SEED_OFFSET` to draw an independent suite.

mod pafi_support;

use std::os::raw::c_void;

use ndarray::{Array1, Array2, ArrayView1};
use pafi_support::{assert_below, assert_coverage, block_stats, se_cap, seed};
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::engine::rgpot::{RgpotConfig, RgpotSurface};
use rgflux::engine::{DynamicsEngine, ThermostatConfig};
use rgflux::pafi::{FreeEnergyProfile, PafiConfig, pafi_profile, pafi_profile_dyn};
use rgflux::stats::StatsConfig;
use rgpot_core::potential::PotentialImpl;
use rgpot_core::status::rgpot_status_t;
use rgpot_core::tensor::rgpot_tensor_cpu_f64_2d;
use rgpot_core::types::{rgpot_force_input_t, rgpot_force_out_t};

const PLANES: usize = 21;
const TEMPERATURE: f64 = 0.02;
const TIME_STEP: f64 = 0.05;
const BURN_IN: usize = 200;
const SAMPLES: usize = 16_000;
const MASSES: [f64; 3] = [1.0, 4.0, 2.0];

// Reported barrier SE cap: `se_cap(mean, cv)` with the mean and coefficient of
// variation of the SE measured over 4096 seeds of this configuration.
const BARRIER_SE_CAP: f64 = se_cap(9.62e-5, 0.051);

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

fn pafi_config(seed_base: u64, burn_in: usize, samples: usize, stats: StatsConfig) -> PafiConfig {
    PafiConfig::default()
        .with_planes(PLANES)
        .with_burn_in_steps(burn_in)
        .with_sample_steps(samples)
        .with_time_step(TIME_STEP)
        .with_stats(stats)
        .with_seed(seed(seed_base))
}

fn native_engine(seed_base: u64) -> NativeEngine<ThreeDWell> {
    let seed_value = seed(seed_base);
    let mut engine = NativeEngine::new(
        ThreeDWell,
        NativeConfig::default()
            .with_masses(Array1::from_vec(MASSES.to_vec()))
            .with_seed(seed_value)
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(1.3)
                    .with_temperature(TEMPERATURE)
                    .with_boltzmann(1.0),
            ),
    )
    .expect("native engine");
    engine
        .shoot_velocities(seed_value)
        .expect("initial velocities");
    engine
}

fn rgpot_engine(seed_base: u64) -> NativeEngine<RgpotSurface> {
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
    let seed_value = seed(seed_base);
    let mut engine = NativeEngine::new(
        surface,
        NativeConfig::default()
            .with_masses(Array1::from_vec(MASSES.to_vec()))
            .with_seed(seed_value)
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(1.3)
                    .with_temperature(TEMPERATURE)
                    .with_boltzmann(1.0),
            ),
    )
    .expect("rgpot-backed native engine");
    engine
        .shoot_velocities(seed_value)
        .expect("rgpot initial velocities");
    engine
}

/// Expectation of `FreeEnergyProfile::free_energy` at the last plane: the
/// trapezoid sum of the exact mean force on the same planes.
fn expected_barrier(planes: usize, thermal: f64) -> f64 {
    let h = 1.0 / ((planes - 1) as f64);
    let mean_force = |x: f64| -4.0 * x * (1.0 - x * x) + 0.25 * thermal / (1.5 + 0.5 * x);
    let mut sum = 0.0_f64;
    for i in 1..planes {
        let x0 = -1.0 + ((i - 1) as f64) * h;
        let x1 = -1.0 + (i as f64) * h;
        sum += 0.5 * h * (mean_force(x0) + mean_force(x1));
    }
    sum
}

/// Identical arithmetic and seeds give identical profiles up to rounding.
fn assert_close(lhs: f64, rhs: f64) {
    let tolerance = 1024.0 * f64::EPSILON * lhs.abs().max(rhs.abs()).max(1.0);
    assert!(
        (lhs - rhs).abs() <= tolerance,
        "parity violated: {lhs:.16e} vs {rhs:.16e}"
    );
}

fn assert_profiles_match(lhs: &FreeEnergyProfile, rhs: &FreeEnergyProfile) {
    assert_eq!(lhs.dividing_index, rhs.dividing_index);
    assert_close(lhs.barrier, rhs.barrier);
    assert_close(lhs.barrier_error, rhs.barrier_error);
    assert_close(lhs.reactant_partition, rhs.reactant_partition);
    assert_close(lhs.normalized_rate, rhs.normalized_rate);
    assert_close(lhs.normalized_rate_error, rhs.normalized_rate_error);
    for (a, b) in lhs.arc_lengths.iter().zip(&rhs.arc_lengths) {
        assert_close(*a, *b);
    }
    for (a, b) in lhs.free_energy.iter().zip(&rhs.free_energy) {
        assert_close(*a, *b);
    }
    for (a, b) in lhs.standard_error.iter().zip(&rhs.standard_error) {
        assert_close(*a, *b);
    }
}

#[test]
fn pafi_trait_object_matches_generic_on_native_engine() {
    let path = path_x_only(-1.0, 0.0, PLANES);
    let config = pafi_config(77, 100, 300, StatsConfig::default().with_min_batches(2));
    let mut engine = native_engine(77);
    let generic = pafi_profile(&mut engine, path.view(), &config).expect("generic profile");
    let dyn_engine: &mut dyn DynamicsEngine = &mut engine;
    let dynamic = pafi_profile_dyn(dyn_engine, path.view(), &config).expect("dynamic profile");
    assert_profiles_match(&generic, &dynamic);
}

#[test]
fn pafi_runs_unchanged_on_native_and_rgpot_backed_engines() {
    let path = path_x_only(-1.0, 0.0, PLANES);
    let config = pafi_config(9, BURN_IN, SAMPLES, block_stats(SAMPLES));

    let mut native = native_engine(9);
    let native_profile = pafi_profile_dyn(&mut native, path.view(), &config).expect("native pafi");
    let mut rgpot = rgpot_engine(9);
    let rgpot_profile = pafi_profile_dyn(&mut rgpot, path.view(), &config).expect("rgpot pafi");
    assert_profiles_match(&native_profile, &rgpot_profile);

    let last = PLANES - 1;
    let barrier = rgpot_profile.free_energy[last];
    let se = rgpot_profile.standard_error[last];
    assert_below("rgpot barrier SE", se, BARRIER_SE_CAP);
    assert_coverage(
        "rgpot barrier",
        barrier,
        se,
        expected_barrier(PLANES, TEMPERATURE),
    );
}
