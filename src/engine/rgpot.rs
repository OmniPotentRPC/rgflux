//! Static potential energy surface wrapper over `rgpot_core`'s Rust bindings.
//!
//! `rgpot` evaluates energies and Cartesian forces across DLPack tensors via
//! [`rgpot_core::potential::PotentialImpl`]. [`RgpotSurface`] wraps that handle
//! as a [`crate::engine::ForceSurface`] so `engine::native` can integrate
//! trajectories directly over any `rgpot` potential.
//!
//! ## Units
//!
//! Positions and box vectors passed to `rgpot_force_input_t` are in Angstrom,
//! energies in eV, and Cartesian forces in eV/Angstrom.
//!
//! ## References
//!
//! - Goswami, R. (2025). *Efficient Exploration of Chemical Kinetics*.
//!   arXiv:2510.21368. <https://doi.org/10.48550/arXiv.2510.21368>
//! - Goswami, R., et al. (2026). *TheochemUI/eOn: v2.9.0.1*. Zenodo.
//!   <https://doi.org/10.5281/zenodo.18529496>

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::Path;
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

unsafe extern "C" {
    fn rgflux_eam_table_open_file(
        path: *const c_char,
        format: c_int,
        periodic: c_int,
        skin: f64,
        err_buf: *mut c_char,
        err_cap: usize,
    ) -> *mut c_void;

    fn rgflux_eam_table_open_memory(
        z: c_int,
        nrho: c_int,
        drho: f64,
        nr: c_int,
        dr: f64,
        cutoff: f64,
        frho: *const f64,
        rhor: *const f64,
        z2r: *const f64,
        periodic: c_int,
        skin: f64,
        err_buf: *mut c_char,
        err_cap: usize,
    ) -> *mut c_void;

    fn rgflux_eam_table_close(handle: *mut c_void);

    fn rgflux_eam_table_eval(
        handle: *const c_void,
        n_atoms: usize,
        pos: *const f64,
        atmnrs: *const c_int,
        box9: *const f64,
        energy: *mut f64,
        forces: *mut f64,
        err_buf: *mut c_char,
        err_cap: usize,
    ) -> c_int;
}

/// DYNAMO file format for [`RgpotSurface::from_eam_table_file`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum EamTableFormat {
    /// Single-element `funcfl` (`pair_style eam`).
    Funcfl,
    /// Multi-element alloy `setfl` (`pair_style eam/alloy`).
    #[default]
    Setfl,
    /// Finnis-Sinclair `eam/fs` (`pair_style eam/fs`).
    FinnisSinclair,
}

impl EamTableFormat {
    const fn code(self) -> c_int {
        match self {
            Self::Funcfl => 0,
            Self::Setfl => 1,
            Self::FinnisSinclair => 2,
        }
    }
}

struct EamCallbackState {
    handle: *mut c_void,
    forces_scratch: Vec<f64>,
}

impl Drop for EamCallbackState {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { rgflux_eam_table_close(self.handle) };
        }
    }
}

unsafe extern "C" fn drop_eam_state(user_data: *mut c_void) {
    if !user_data.is_null() {
        let _ = unsafe { Box::from_raw(user_data.cast::<EamCallbackState>()) };
    }
}

unsafe extern "C" fn eval_eam_callback(
    user_data: *mut c_void,
    input: *const rgpot_force_input_t,
    output: *mut rgpot_force_out_t,
) -> rgpot_status_t {
    if user_data.is_null() || input.is_null() || output.is_null() {
        return rgpot_status_t::RGPOT_INVALID_PARAMETER;
    }
    let state = unsafe { &mut *user_data.cast::<EamCallbackState>() };
    let inp = unsafe { &*input };
    let Some(n_atoms) = (unsafe { inp.n_atoms() }) else {
        return rgpot_status_t::RGPOT_INVALID_PARAMETER;
    };
    let Ok(n_atoms_i64) = i64::try_from(n_atoms) else {
        return rgpot_status_t::RGPOT_INVALID_PARAMETER;
    };
    if inp.positions.is_null() || inp.atomic_numbers.is_null() || inp.box_matrix.is_null() {
        return rgpot_status_t::RGPOT_INVALID_PARAMETER;
    }
    let pos_ptr = unsafe { (*inp.positions).dl_tensor.data } as *const f64;
    let atm_ptr = unsafe { (*inp.atomic_numbers).dl_tensor.data } as *const c_int;
    let box_ptr = unsafe { (*inp.box_matrix).dl_tensor.data } as *const f64;
    if pos_ptr.is_null() || atm_ptr.is_null() || box_ptr.is_null() {
        return rgpot_status_t::RGPOT_INVALID_PARAMETER;
    }
    let dof = 3 * n_atoms;
    if state.forces_scratch.len() != dof {
        state.forces_scratch.resize(dof, 0.0);
    }
    let mut energy = 0.0_f64;
    let mut err_buf = [0 as c_char; 256];
    let rc = unsafe {
        rgflux_eam_table_eval(
            state.handle,
            n_atoms,
            pos_ptr,
            atm_ptr,
            box_ptr,
            &mut energy,
            state.forces_scratch.as_mut_ptr(),
            err_buf.as_mut_ptr(),
            err_buf.len(),
        )
    };
    if rc != 0 {
        return rgpot_status_t::RGPOT_INTERNAL_ERROR;
    }
    let out = unsafe { &mut *output };
    out.energy = energy;
    out.variance = 0.0;
    out.forces =
        unsafe { rgpot_tensor_cpu_f64_2d(state.forces_scratch.as_mut_ptr(), n_atoms_i64, 3) };
    rgpot_status_t::RGPOT_SUCCESS
}

/// Default cubic box edge (Angstrom) used when no periodic cell is supplied.
pub const DEFAULT_RGPOT_BOX_EDGE: f64 = 100.0;

/// Configuration for [`RgpotSurface`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct RgpotConfig {
    /// Atomic numbers `Z_i` (`n_atoms` entries) passed to `rgpot_force_input_t`.
    ///
    /// Default: `vec![1]`.
    pub atomic_numbers: Vec<i32>,
    /// Fallback row-major 3x3 box matrix passed when the caller supplies `cell = None`.
    ///
    /// Default: cubic box with edge [`DEFAULT_RGPOT_BOX_EDGE`] (`100.0` Angstrom).
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

    /// Construct an in-process `rgpot::EAMTablePot` surface from a DYNAMO table file.
    pub fn from_eam_table_file(
        path: &Path,
        format: EamTableFormat,
        periodic: bool,
        skin: f64,
        config: RgpotConfig,
    ) -> Result<Self, FluxError> {
        let path_str = path.to_str().ok_or(FluxError::InvalidConfig(
            "EAM table path is not valid UTF-8",
        ))?;
        let c_path = CString::new(path_str)
            .map_err(|_| FluxError::InvalidConfig("EAM table path contains interior NUL"))?;
        let mut err_buf = [0 as c_char; 256];
        let handle = unsafe {
            rgflux_eam_table_open_file(
                c_path.as_ptr(),
                format.code(),
                i32::from(periodic),
                skin,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if handle.is_null() {
            let msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            return Err(FluxError::Surface(msg));
        }
        let state = Box::new(EamCallbackState {
            handle,
            forces_scratch: vec![0.0; 3 * config.atomic_numbers.len()],
        });
        let potential = PotentialImpl::new(
            eval_eam_callback,
            Box::into_raw(state).cast::<c_void>(),
            Some(drop_eam_state),
        );
        Self::new(potential, config)
    }

    /// Construct an in-process `rgpot::EAMTablePot` surface from tabulated arrays in memory.
    #[allow(clippy::too_many_arguments)]
    pub fn from_eam_table_memory(
        atomic_number: i32,
        drho: f64,
        dr: f64,
        cutoff: f64,
        frho: &[f64],
        rhor: &[f64],
        z2r: &[f64],
        periodic: bool,
        skin: f64,
        config: RgpotConfig,
    ) -> Result<Self, FluxError> {
        if frho.len() < 5 || rhor.len() < 5 || z2r.len() != rhor.len() {
            return Err(FluxError::InvalidConfig(
                "EAM table arrays require at least 5 knots and matching rhor/z2r lengths",
            ));
        }
        let nrho = c_int::try_from(frho.len())
            .map_err(|_| FluxError::InvalidConfig("EAM nrho overflows c_int"))?;
        let nr = c_int::try_from(rhor.len())
            .map_err(|_| FluxError::InvalidConfig("EAM nr overflows c_int"))?;
        let mut err_buf = [0 as c_char; 256];
        let handle = unsafe {
            rgflux_eam_table_open_memory(
                atomic_number,
                nrho,
                drho,
                nr,
                dr,
                cutoff,
                frho.as_ptr(),
                rhor.as_ptr(),
                z2r.as_ptr(),
                i32::from(periodic),
                skin,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if handle.is_null() {
            let msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            return Err(FluxError::Surface(msg));
        }
        let state = Box::new(EamCallbackState {
            handle,
            forces_scratch: vec![0.0; 3 * config.atomic_numbers.len()],
        });
        let potential = PotentialImpl::new(
            eval_eam_callback,
            Box::into_raw(state).cast::<c_void>(),
            Some(drop_eam_state),
        );
        Self::new(potential, config)
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

    #[test]
    fn eam_table_memory_surface_evaluates_pair_interaction() {
        let frho = vec![0.0_f64; 16];
        let rhor = vec![0.0_f64; 16];
        // Linearly decaying r * phi(r) inside cutoff = 4.0.
        let dr = 4.0 / 15.0;
        let z2r: Vec<f64> = (0..16)
            .map(|i| {
                let r = (i as f64) * dr;
                (4.0 - r).max(0.0)
            })
            .collect();
        let cfg = RgpotConfig::default()
            .with_atomic_numbers(vec![29, 29])
            .with_default_cell([20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0]);
        let surf = RgpotSurface::from_eam_table_memory(
            29, 0.1, dr, 4.0, &frho, &rhor, &z2r, true, 0.5, cfg,
        )
        .expect("in-memory EAMTablePot");
        let pos = Array1::from_vec(vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
        let (energy, forces) = surf
            .energy_and_forces(pos.view(), None)
            .expect("eval eam pair");
        assert!(energy.is_finite() && energy > 0.0);
        assert!(forces[0] < 0.0 && (forces[0] + forces[3]).abs() <= 1.0e-12);
    }
}
