//! Projected reactant frequency `omega_0^2 = 1 / (n^T H^-1 n)` on a coupled
//! quadratic well evaluated through the `rgpot` callback ABI.
//!
//! With a quadratic potential `V = x^T A x / 2` and a straight path along `d`
//! from the minimum, `n = M^(1/2) d / |M^(1/2) d|` in mass-weighted
//! coordinates and `H = M^(-1/2) A M^(-1/2)`, so the closed form is
//! `omega_0^2 = |M^(1/2) d|^2 / (d^T M A^-1 M d)`.

#![cfg(feature = "rgpot")]

use std::f64::consts::PI;
use std::os::raw::c_void;

use ndarray::{Array1, Array2};
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::engine::rgpot::{RgpotConfig, RgpotSurface};
use rgflux::engine::{DynamicsEngine, ThermostatConfig};
use rgflux::pafi::{PafiConfig, pafi_profile};
use rgflux::stats::StatsConfig;
use rgpot_core::potential::PotentialImpl;
use rgpot_core::status::rgpot_status_t;
use rgpot_core::tensor::rgpot_tensor_cpu_f64_2d;
use rgpot_core::types::{rgpot_force_input_t, rgpot_force_out_t};

const STIFFNESS: [[f64; 3]; 3] = [[3.0, 0.8, -0.4], [0.8, 2.0, 0.5], [-0.4, 0.5, 1.5]];
const MASSES: [f64; 3] = [1.0, 2.5, 4.0];
const DIRECTION: [f64; 3] = [1.0, 0.6, -0.8];

struct CallbackState {
    forces: Vec<f64>,
}

unsafe extern "C" fn drop_state(ptr: *mut c_void) {
    if !ptr.is_null() {
        let _ = unsafe { Box::from_raw(ptr as *mut CallbackState) };
    }
}

unsafe extern "C" fn coupled_quadratic(
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
    if unsafe { inp.n_atoms() } != Some(1) {
        return rgpot_status_t::RGPOT_INVALID_PARAMETER;
    }
    let dl = unsafe { &(*inp.positions).dl_tensor };
    let x = unsafe { std::slice::from_raw_parts(dl.data as *const f64, 3) };
    let mut energy = 0.0;
    for ((row, coord), force) in STIFFNESS.iter().zip(x).zip(state.forces.iter_mut()) {
        let gradient: f64 = row.iter().zip(x).map(|(stiff, value)| stiff * value).sum();
        energy += 0.5 * coord * gradient;
        *force = -gradient;
    }
    out.energy = energy;
    out.variance = 0.0;
    out.forces = unsafe { rgpot_tensor_cpu_f64_2d(state.forces.as_mut_ptr(), 1, 3) };
    rgpot_status_t::RGPOT_SUCCESS
}

fn solve3(a: [[f64; 3]; 3], b: [f64; 3]) -> [f64; 3] {
    let mut m = [[0.0_f64; 4]; 3];
    for i in 0..3 {
        m[i][..3].copy_from_slice(&a[i]);
        m[i][3] = b[i];
    }
    for col in 0..3 {
        let pivot = (col..3)
            .max_by(|&p, &q| m[p][col].abs().total_cmp(&m[q][col].abs()))
            .expect("pivot");
        m.swap(col, pivot);
        for row in col + 1..3 {
            let factor = m[row][col] / m[col][col];
            for k in col..4 {
                m[row][k] -= factor * m[col][k];
            }
        }
    }
    let mut x = [0.0_f64; 3];
    for row in (0..3).rev() {
        let tail: f64 = m[row][row + 1..3]
            .iter()
            .zip(x[row + 1..].iter())
            .map(|(coeff, unknown)| coeff * unknown)
            .sum();
        x[row] = (m[row][3] - tail) / m[row][row];
    }
    x
}

#[test]
fn reactant_frequency_is_the_mass_weighted_hessian_projection() {
    let momentum_direction: [f64; 3] = std::array::from_fn(|i| MASSES[i] * DIRECTION[i]);
    let solved = solve3(STIFFNESS, momentum_direction);
    let denominator: f64 = (0..3).map(|i| momentum_direction[i] * solved[i]).sum();
    let numerator: f64 = (0..3)
        .map(|i| MASSES[i] * DIRECTION[i] * DIRECTION[i])
        .sum();
    let expected = (numerator / denominator).sqrt();

    let kt = 0.05;
    let planes = 5usize;
    let mut path = Array2::<f64>::zeros((planes, 3));
    for i in 0..planes {
        for j in 0..3 {
            path[[i, j]] = 0.4 * DIRECTION[j] * i as f64 / (planes - 1) as f64;
        }
    }
    let state = Box::new(CallbackState {
        forces: vec![0.0; 3],
    });
    let potential = PotentialImpl::new(
        coupled_quadratic,
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
            .with_masses(Array1::from_vec(MASSES.to_vec()))
            .with_seed(3)
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(2.0)
                    .with_temperature(kt)
                    .with_boltzmann(1.0),
            ),
    )
    .expect("native engine");
    engine.shoot_velocities(3).expect("velocities");

    let config = PafiConfig::default()
        .with_planes(planes)
        .with_burn_in_steps(200)
        .with_sample_steps(400)
        .with_time_step(0.02)
        .with_stats(StatsConfig::default().with_min_batches(4))
        .with_seed(3);
    let profile = pafi_profile(&mut engine, path.view(), &config).expect("profile");

    assert!(
        (profile.reactant_frequency - expected).abs() <= 1.0e-6 * expected,
        "omega_0 {:.10} versus closed form {:.10}",
        profile.reactant_frequency,
        expected
    );
    let reactant_partition = profile
        .reactant_partition
        .expect("representable reactant partition");
    let flux = (kt / (2.0 * PI)).sqrt() * (-profile.barrier / kt).exp() / reactant_partition;
    let normalized_rate = profile
        .normalized_rate
        .expect("representable normalized rate");
    assert!((normalized_rate - flux).abs() <= 1.0e-12 * flux);
    assert!((profile.normalized_log_rate - flux.ln()).abs() <= 1.0e-9 * flux.ln().abs().max(1.0));
}
