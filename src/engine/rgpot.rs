//! Static potential energy surface wrapper over `rgpot_core`'s Rust bindings.
//!
//! `rgpot` evaluates energies and Cartesian forces across DLPack tensors via
//! [`rgpot_core::potential::PotentialImpl`]. [`RgpotSurface`] wraps that handle
//! as a [`crate::engine::ForceSurface`] so `engine::native` can integrate
//! trajectories directly over any `rgpot` potential.

use std::sync::Mutex;

use ndarray::{Array1, ArrayView1};
use rgpot_core::potential::PotentialImpl;
use rgpot_core::status::{rgpot_last_error, rgpot_status_t};
use rgpot_core::tensor::{
    rgpot_tensor_cpu_f64_2d, rgpot_tensor_cpu_f64_matrix3, rgpot_tensor_cpu_i32_1d,
    rgpot_tensor_free,
};
use rgpot_core::types::{rgpot_force_input_t, rgpot_force_out_t};

use crate::engine::{ForceSurface, validate_finite_slice};
use crate::error::FluxError;

/// Default cubic box edge (Angstrom) used when no periodic cell is supplied.
pub const DEFAULT_RGPOT_BOX_EDGE: f64 = 100.0;

/// Configuration for [`RgpotSurface`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct RgpotConfig {
    /// Atomic numbers `Z_i` (`n_atoms` entries) passed to `rgpot_force_input_t`.
    pub atomic_numbers: Vec<i32>,
    /// Fallback row-major 3x3 box matrix passed when the caller supplies `cell = None`.
    pub default_cell: [f64; 9],
}

impl Default for RgpotConfig {
    fn default() -> Self {
        Self {
            atomic_numbers: vec![1],
            default_cell: [
                DEFAULT_RGPOT_BOX_EDGE,
                0.0,
                0.0,
                0.0,
                DEFAULT_RGPOT_BOX_EDGE,
                0.0,
                0.0,
                0.0,
                DEFAULT_RGPOT_BOX_EDGE,
            ],
        }
    }
}

with_setters!(RgpotConfig {
    with_atomic_numbers => atomic_numbers: val Vec<i32>,
    with_default_cell => default_cell: val [f64; 9],
});

/// Thread-safe static potential energy and force surface backed by [`PotentialImpl`].
pub struct RgpotSurface {
    potential: Mutex<PotentialImpl>,
    atomic_numbers: Vec<i32>,
    default_cell: [f64; 9],
}

// SAFETY: `PotentialImpl` is `Send` and every call is serialized by `Mutex`.
unsafe impl Sync for RgpotSurface {}

impl RgpotSurface {
    /// Construct a static PES adapter from an `rgpot_core` [`PotentialImpl`] and [`RgpotConfig`].
    pub fn new(potential: PotentialImpl, config: RgpotConfig) -> Result<Self, FluxError> {
        if config.atomic_numbers.is_empty() {
            return Err(FluxError::EmptyInput(
                "RgpotConfig::atomic_numbers must contain at least one atom",
            ));
        }
        for &entry in &config.default_cell {
            if !entry.is_finite() {
                return Err(FluxError::NonFinite("RgpotConfig::default_cell"));
            }
        }
        Ok(Self {
            potential: Mutex::new(potential),
            atomic_numbers: config.atomic_numbers,
            default_cell: config.default_cell,
        })
    }

    /// Number of atoms configured on this surface.
    pub fn n_atoms(&self) -> usize {
        self.atomic_numbers.len()
    }
}

impl ForceSurface for RgpotSurface {
    fn energy_and_forces(
        &self,
        positions: ArrayView1<'_, f64>,
        cell: Option<&[f64; 9]>,
    ) -> Result<(f64, Array1<f64>), FluxError> {
        let n_atoms = self.atomic_numbers.len();
        let expected_dof = 3 * n_atoms;
        if positions.len() != expected_dof {
            return Err(FluxError::Shape {
                expected: expected_dof,
                got: positions.len(),
                context: "RgpotSurface positions",
            });
        }
        validate_finite_slice(positions, "RgpotSurface positions")?;
        let n_atoms_i64 = i64::try_from(n_atoms)
            .map_err(|_| FluxError::InvalidConfig("RgpotSurface atom count overflows i64"))?;

        let mut box_buf = match cell {
            Some(c) => {
                for &entry in c {
                    if !entry.is_finite() {
                        return Err(FluxError::NonFinite("RgpotSurface cell"));
                    }
                }
                *c
            }
            None => self.default_cell,
        };
        let mut pos_buf: Vec<f64> = positions.to_vec();
        let mut atm_buf: Vec<i32> = self.atomic_numbers.clone();

        let guard = self
            .potential
            .lock()
            .map_err(|_| FluxError::Surface("RgpotSurface mutex poisoned".to_string()))?;

        // SAFETY: `pos_buf`, `atm_buf`, and `box_buf` outlive `input` and are freed
        // immediately after `guard.calculate` returns.
        let (energy, forces_res) = unsafe {
            let pos_t = rgpot_tensor_cpu_f64_2d(pos_buf.as_mut_ptr(), n_atoms_i64, 3);
            let atm_t = rgpot_tensor_cpu_i32_1d(atm_buf.as_mut_ptr(), n_atoms_i64);
            let box_t = rgpot_tensor_cpu_f64_matrix3(box_buf.as_mut_ptr());
            let input = rgpot_force_input_t {
                positions: pos_t,
                atomic_numbers: atm_t,
                box_matrix: box_t,
            };
            let mut output = rgpot_force_out_t {
                forces: std::ptr::null_mut(),
                energy: 0.0,
                variance: 0.0,
            };

            let status = guard.calculate(&input, &mut output);
            rgpot_tensor_free(pos_t);
            rgpot_tensor_free(atm_t);
            rgpot_tensor_free(box_t);

            let extracted = if status == rgpot_status_t::RGPOT_SUCCESS {
                if output.forces.is_null() {
                    Err(FluxError::Surface(
                        "rgpot callback returned null forces tensor".to_string(),
                    ))
                } else {
                    let dl = &(*output.forces).dl_tensor;
                    if dl.data.is_null() {
                        rgpot_tensor_free(output.forces);
                        Err(FluxError::Surface(
                            "rgpot forces tensor has null data pointer".to_string(),
                        ))
                    } else {
                        let slice = std::slice::from_raw_parts(dl.data as *const f64, expected_dof);
                        let vec = slice.to_vec();
                        rgpot_tensor_free(output.forces);
                        Ok(Array1::from_vec(vec))
                    }
                }
            } else {
                if !output.forces.is_null() {
                    rgpot_tensor_free(output.forces);
                }
                let msg_ptr = rgpot_last_error();
                let detail = if msg_ptr.is_null() {
                    format!("rgpot status {status:?}")
                } else {
                    std::ffi::CStr::from_ptr(msg_ptr)
                        .to_string_lossy()
                        .into_owned()
                };
                Err(FluxError::Surface(detail))
            };
            (output.energy, extracted)
        };

        let forces = forces_res?;
        if !energy.is_finite() {
            return Err(FluxError::NonFinite("RgpotSurface energy"));
        }
        validate_finite_slice(forces.view(), "RgpotSurface forces")?;
        Ok((energy, forces))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::raw::c_void;

    unsafe extern "C" fn harmonic_callback(
        user_data: *mut c_void,
        input: *const rgpot_force_input_t,
        output: *mut rgpot_force_out_t,
    ) -> rgpot_status_t {
        let inp = unsafe { &*input };
        let Some(n_atoms) = (unsafe { inp.n_atoms() }) else {
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        };
        let Ok(n_atoms_i64) = i64::try_from(n_atoms) else {
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        };
        let n3 = 3 * n_atoms;
        let pos_dl = unsafe { &(*inp.positions).dl_tensor };
        let pos = unsafe { std::slice::from_raw_parts(pos_dl.data as *const f64, n3) };
        let scratch = unsafe { std::slice::from_raw_parts_mut(user_data as *mut f64, n3) };

        let mut energy = 0.0_f64;
        for i in 0..n3 {
            energy += 0.5 * pos[i] * pos[i];
            scratch[i] = -pos[i];
        }
        let tensor = unsafe { rgpot_tensor_cpu_f64_2d(scratch.as_mut_ptr(), n_atoms_i64, 3) };
        let out = unsafe { &mut *output };
        out.energy = energy;
        out.variance = 0.0;
        out.forces = tensor;
        rgpot_status_t::RGPOT_SUCCESS
    }

    #[test]
    fn rgpot_surface_evaluates_energy_and_forces() {
        let mut force_scratch = [0.0_f64; 3];
        let pot = PotentialImpl::new(
            harmonic_callback,
            force_scratch.as_mut_ptr().cast::<c_void>(),
            None,
        );
        let cfg = RgpotConfig::default().with_atomic_numbers(vec![29]);
        let surf = RgpotSurface::new(pot, cfg).expect("valid rgpot surface");
        let x = Array1::from_vec(vec![1.0, -2.0, 0.5]);
        let (e, f) = surf.energy_and_forces(x.view(), None).expect("rgpot eval");
        assert_eq!(e, 2.625);
        assert_eq!(f, Array1::from_vec(vec![-1.0, 2.0, -0.5]));
    }
}
