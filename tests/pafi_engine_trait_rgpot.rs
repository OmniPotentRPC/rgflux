#![cfg(feature = "rgpot")]
//! Deterministic PAFI parity and analytic checks across native and `rgpot` engines.
//!
//! Reproducible with `cargo test --all-features --test pafi_engine_trait_rgpot -- --nocapture`.

use std::os::raw::c_void;

use ndarray::{Array1, Array2, ArrayView1};
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
const PARITY_TOL: f64 = 1.0e-12;
const MAX_BARRIER_SE: f64 = 6.0e-4;

/// Number of statistical `z` comparisons across the PAFI test suite
/// (`6` in `tests/pafi_native.rs` and `2` in `tests/pafi_engine_trait_rgpot.rs`).
const PAFI_SUITE_COMPARISONS: usize = 8;

/// Two-sided standard normal tail `2 * Phi(-4.4) = erfc(4.4 / sqrt(2)) = 1.082509e-5`.
const TWO_SIDED_NORMAL_TAIL_AT_4_4: f64 = 1.082_509e-5;

/// Two-sided normal threshold `z = 4.4` (`2 * Phi(-4.4) = 1.083e-5`).
/// Across `m = 8` comparisons (`PAFI_SUITE_COMPARISONS`), the Bonferroni union
/// bound gives a family-wise suite false-fail rate of `8 * 1.083e-5 = 8.66e-5 < 1e-4`.
const SUITE_Z_MAX: f64 = 4.4;

const _: () = assert!((PAFI_SUITE_COMPARISONS as f64) * TWO_SIDED_NORMAL_TAIL_AT_4_4 < 1.0e-4);

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
        .with_planes(PLANES)
        .with_burn_in_steps(100)
        .with_sample_steps(300)
        .with_time_step(0.05)
        .with_stats(StatsConfig::default().with_min_batches(2))
        .with_seed(seed)
}

fn trapezoid_three_d_well_barrier(planes: usize, thermal: f64) -> f64 {
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

fn assert_profiles_match(lhs: &FreeEnergyProfile, rhs: &FreeEnergyProfile) {
    assert_eq!(lhs.dividing_index, rhs.dividing_index);
    assert!((lhs.barrier - rhs.barrier).abs() <= PARITY_TOL);
    assert!((lhs.barrier_error - rhs.barrier_error).abs() <= PARITY_TOL);
    assert!((lhs.reactant_partition - rhs.reactant_partition).abs() <= PARITY_TOL);
    assert!((lhs.normalized_rate - rhs.normalized_rate).abs() <= PARITY_TOL);
    assert!((lhs.normalized_rate_error - rhs.normalized_rate_error).abs() <= PARITY_TOL);
    for (a, b) in lhs.arc_lengths.iter().zip(&rhs.arc_lengths) {
        assert!((a - b).abs() <= PARITY_TOL);
    }
    for (a, b) in lhs.free_energy.iter().zip(&rhs.free_energy) {
        assert!((a - b).abs() <= PARITY_TOL);
    }
    for (a, b) in lhs.standard_error.iter().zip(&rhs.standard_error) {
        assert!((a - b).abs() <= PARITY_TOL);
    }
}

#[test]
fn pafi_trait_object_matches_generic_on_native_engine() {
    let path = path_x_only(-1.0, 0.0, PLANES);
    let config = pafi_config(77);
    let mut engine = NativeEngine::new(
        ThreeDWell,
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0, 4.0, 2.0]))
            .with_seed(77)
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(1.3)
                    .with_temperature(TEMPERATURE)
                    .with_boltzmann(1.0),
            ),
    )
    .expect("native engine");
    engine.shoot_velocities(77).expect("initial velocities");
    let generic = pafi_profile(&mut engine, path.view(), &config).expect("generic profile");
    let dyn_engine: &mut dyn DynamicsEngine = &mut engine;
    let dynamic = pafi_profile_dyn(dyn_engine, path.view(), &config).expect("dynamic profile");
    assert_profiles_match(&generic, &dynamic);

    let expected_barrier = trapezoid_three_d_well_barrier(PLANES, TEMPERATURE);
    let z = (dynamic.barrier - expected_barrier).abs() / dynamic.barrier_error;
    assert!(
        dynamic.barrier_error > 0.0 && dynamic.barrier_error <= MAX_BARRIER_SE,
        "barrier SE {:.8} exceeds {:.8}",
        dynamic.barrier_error,
        MAX_BARRIER_SE
    );
    assert!(
        z <= SUITE_Z_MAX,
        "barrier {:.8} ± {:.8} versus trapezoid analytic {:.8} gives z={:.4} > {:.1}",
        dynamic.barrier,
        dynamic.barrier_error,
        expected_barrier,
        z,
        SUITE_Z_MAX
    );
}

#[test]
fn pafi_runs_unchanged_on_native_and_rgpot_backed_engines() {
    let path = path_x_only(-1.0, 0.0, PLANES);
    let config = pafi_config(9);

    let mut native = NativeEngine::new(
        ThreeDWell,
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0, 4.0, 2.0]))
            .with_seed(9)
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(1.3)
                    .with_temperature(TEMPERATURE)
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
                    .with_temperature(TEMPERATURE)
                    .with_boltzmann(1.0),
            ),
    )
    .expect("rgpot-backed native engine");
    rgpot_engine
        .shoot_velocities(9)
        .expect("rgpot initial velocities");
    let rgpot_profile =
        pafi_profile_dyn(&mut rgpot_engine, path.view(), &config).expect("rgpot pafi");

    assert_profiles_match(&native_profile, &rgpot_profile);

    let expected_barrier = trapezoid_three_d_well_barrier(PLANES, TEMPERATURE);
    let z_barrier = (rgpot_profile.barrier - expected_barrier).abs() / rgpot_profile.barrier_error;
    assert!(
        rgpot_profile.barrier_error > 0.0 && rgpot_profile.barrier_error <= MAX_BARRIER_SE,
        "rgpot barrier SE {:.8} exceeds {:.8}",
        rgpot_profile.barrier_error,
        MAX_BARRIER_SE
    );
    assert!(
        z_barrier <= SUITE_Z_MAX,
        "barrier rgpot {:.8} ± {:.8} versus trapezoid analytic {:.8} gives z={:.4} > {:.1}",
        rgpot_profile.barrier,
        rgpot_profile.barrier_error,
        expected_barrier,
        z_barrier,
        SUITE_Z_MAX
    );
}
