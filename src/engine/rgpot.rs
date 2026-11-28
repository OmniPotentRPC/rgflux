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
//! [`RgpotSurface::from_eam_table_file`] and [`RgpotSurface::from_eam_table_memory`]
//! construct rgpot's `EAMTablePot` in process. The `rgpot` feature compiles
//! that class from the rgpot C++ core. Set `RGPOT_DIR` to the rgpot repository
//! root or its `CppCore` directory, or run `ci/fetch_rgpot_cpp.sh`, which
//! fetches the pinned core. When `RGPOT_DIR` is unset, the build uses that
//! fetched tree when it is already at the pin, and otherwise reuses an
//! existing cargo git checkout of the pin. `RgpotSurface::new` still accepts
//! any `PotentialImpl` handle.
//!
//! ## References
//!
//! - Goswami, R. (2025). *Efficient Exploration of Chemical Kinetics*.
//!   arXiv:2510.21368. <https://doi.org/10.48550/arXiv.2510.21368>
//! - Goswami, R., et al. (2026). *TheochemUI/eOn: v2.9.0.1*. Zenodo.
//!   <https://doi.org/10.5281/zenodo.18529496>

use std::cell::RefCell;
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

thread_local! {
    static EAM_EVAL_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

fn eam_step_ok(step: f64) -> bool {
    step.is_finite() && step > 0.0
}

/// `place` casts `x * (1/step) + 1` to `int` before it clamps the bin.
fn place_cast_defined(x: f64, step: f64) -> bool {
    if !x.is_finite() || !eam_step_ok(step) {
        return false;
    }
    let rdx = 1.0 / step;
    if !rdx.is_finite() {
        return false;
    }
    let q = x * rdx + 1.0;
    q.is_finite() && q < 2147483648.0 && q >= i32::MIN as f64
}

fn knots_are_finite(values: &[f64]) -> bool {
    values.iter().all(|value| value.is_finite())
}

fn reject_eam_memory_grid(
    frho: &[f64],
    drho: f64,
    rhor: &[f64],
    dr: f64,
    z2r: &[f64],
    cutoff: f64,
) -> Result<(), FluxError> {
    let nrho = frho.len();
    let nr = rhor.len();
    if nrho < 5 || nr < 5 {
        return Err(FluxError::InvalidConfig(
            "EAM table arrays require at least 5 knots",
        ));
    }
    if !eam_step_ok(dr) || !eam_step_ok(drho) || !eam_step_ok(cutoff) {
        return Err(FluxError::InvalidConfig(
            "EAM dr, drho and cutoff must be finite and positive",
        ));
    }
    if !knots_are_finite(frho) || !knots_are_finite(rhor) || !knots_are_finite(z2r) {
        return Err(FluxError::InvalidConfig("EAM table knots must be finite"));
    }
    let rhomax = (nrho - 1) as f64 * drho;
    let r_span = cutoff.max((nr - 1) as f64 * dr);
    let density_ok = place_cast_defined(rhomax, drho)
        && place_cast_defined(1.0, drho)
        && place_cast_defined(-1.0, drho)
        && rhor.iter().all(|knot| {
            let magnitude = knot.abs();
            place_cast_defined(magnitude, drho) && place_cast_defined(-magnitude, drho)
        });
    if !rhomax.is_finite() || !r_span.is_finite() || !place_cast_defined(r_span, dr) || !density_ok
    {
        return Err(FluxError::InvalidConfig(
            "EAM grid span does not fit in an int bin index",
        ));
    }
    Ok(())
}

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

#[cfg(test)]
thread_local! {
    static EAM_CLOSE_COUNT: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    static EAM_SCRATCH_PANIC: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn close_eam_handle(handle: *mut c_void) {
    if handle.is_null() {
        return;
    }
    #[cfg(test)]
    EAM_CLOSE_COUNT.with(|count| count.set(count.get().saturating_add(1)));
    unsafe { rgflux_eam_table_close(handle) };
}

impl Drop for EamCallbackState {
    fn drop(&mut self) {
        close_eam_handle(self.handle);
        self.handle = std::ptr::null_mut();
    }
}

/// Closes an open table if scratch allocation fails before the callback owns it.
struct EamOpenGuard {
    handle: *mut c_void,
}

impl Drop for EamOpenGuard {
    fn drop(&mut self) {
        close_eam_handle(self.handle);
        self.handle = std::ptr::null_mut();
    }
}

fn eam_callback_state(mut opened: EamOpenGuard, n_atoms: usize) -> Box<EamCallbackState> {
    #[cfg(test)]
    if EAM_SCRATCH_PANIC.with(|flag| flag.get()) {
        panic!("eam force scratch allocation failed");
    }
    let forces_scratch = vec![0.0; 3 * n_atoms];
    let mut state = Box::new(EamCallbackState {
        handle: std::ptr::null_mut(),
        forces_scratch,
    });
    let handle = opened.handle;
    opened.handle = std::ptr::null_mut();
    state.handle = handle;
    state
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
        let msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        EAM_EVAL_ERROR.with(|slot| *slot.borrow_mut() = msg);
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

    /// Construct an in-process `rgpot::EAMTablePot` from a DYNAMO table file.
    ///
    /// `format` selects the funcfl, setfl, or Finnis-Sinclair layout.
    /// `periodic` and `skin` are forwarded to `EAMTablePot`. The energy and
    /// forces are that class's `forceImpl` result, in eV and eV/Angstrom.
    ///
    /// # Errors
    ///
    /// Returns [`FluxError::InvalidConfig`] when `path` is not UTF-8 or
    /// contains an interior NUL, and [`FluxError::Surface`] when the file
    /// cannot be read, a knot is not finite, `dr`, `drho`, or `cutoff` is
    /// not finite and positive, funcfl resampling leaves fewer than five
    /// knots, a knot count whose spline index does not fit in `int`, or
    /// `EAMTablePot` rejects the table. `place` forms its bin as
    /// `x * (1/step) + 1` and casts that to `int` before clamping. The bridge
    /// refuses a distance out to the cutoff, a tabulated density, or a unit
    /// density whose cast does not fit, before `EAMTablePot` is constructed.
    /// Evaluation counts neighbors inside the cutoff and returns
    /// [`FluxError::Surface`] when a summed density's place index does not
    /// fit, instead of a wrong finite energy.
    ///
    /// # Examples
    ///
    /// ```
    /// use rgflux::engine::rgpot::{EamTableFormat, RgpotConfig, RgpotSurface};
    /// use std::path::Path;
    ///
    /// let missing = RgpotSurface::from_eam_table_file(
    ///     Path::new("missing-eam-table.setfl"),
    ///     EamTableFormat::Setfl,
    ///     true,
    ///     0.5,
    ///     RgpotConfig::default().with_atomic_numbers(vec![29]),
    /// );
    /// assert!(missing.is_err());
    /// ```
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
        let state = eam_callback_state(EamOpenGuard { handle }, config.atomic_numbers.len());
        let potential = PotentialImpl::new(
            eval_eam_callback,
            Box::into_raw(state).cast::<c_void>(),
            Some(drop_eam_state),
        );
        Self::new(potential, config)
    }

    /// Construct an in-process `rgpot::EAMTablePot` from one-element tabulated arrays.
    ///
    /// `frho`, `rhor`, and `z2r` are the embedding, density, and `r phi(r)`
    /// knots `EAMTablePot` splines. At least five knots are required, and
    /// `rhor` and `z2r` must have the same length. `periodic` and `skin` are
    /// forwarded unchanged. The energy and forces are that class's
    /// `forceImpl` result.
    ///
    /// # Errors
    ///
    /// Returns [`FluxError::InvalidConfig`] when an array is shorter than
    /// five knots, a knot is not finite, `z2r` does not match `rhor`, or
    /// `dr`, `drho`, or `cutoff` is not finite and positive. `place` forms
    /// its bin as `x * (1/step) + 1` and casts that to `int` before clamping.
    /// The bridge refuses a distance out to the cutoff, a tabulated density,
    /// or a unit density whose cast does not fit, before `EAMTablePot` is
    /// constructed. [`FluxError::Surface`] reports a table the bridge or
    /// `EAMTablePot` rejects, including a knot count whose spline index does
    /// not fit in `int`, and an evaluation whose summed density, counted from
    /// neighbors inside the cutoff, does not fit in that cast.
    ///
    /// # Examples
    ///
    /// A linear `r phi(r) = max(4 - r, 0)` inside a cutoff of 4 Angstrom has
    /// pair energy 1 eV at separation 2, and the force on the atom at the
    /// origin is -1 eV/Angstrom along x.
    ///
    /// ```
    /// use ndarray::Array1;
    /// use rgflux::engine::ForceSurface;
    /// use rgflux::engine::rgpot::{RgpotConfig, RgpotSurface};
    ///
    /// let dr = 4.0 / 15.0;
    /// let zeros = vec![0.0; 16];
    /// let z2r: Vec<f64> = (0..16).map(|i| (4.0 - i as f64 * dr).max(0.0)).collect();
    /// let cfg = RgpotConfig::default()
    ///     .with_atomic_numbers(vec![29, 29])
    ///     .with_default_cell([20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0]);
    /// let surf = RgpotSurface::from_eam_table_memory(
    ///     29, 0.1, dr, 4.0, &zeros, &zeros, &z2r, true, 0.5, cfg,
    /// )?;
    /// let pos = Array1::from_vec(vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
    /// let (energy, forces) = surf.energy_and_forces(pos.view(), None)?;
    /// assert_eq!(energy, 1.0);
    /// assert_eq!(forces[0], -1.0);
    /// # Ok::<(), rgflux::FluxError>(())
    /// ```
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
        if z2r.len() != rhor.len() {
            return Err(FluxError::InvalidConfig(
                "EAM rhor and z2r lengths must match",
            ));
        }
        reject_eam_memory_grid(frho, drho, rhor, dr, z2r, cutoff)?;
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
        let state = eam_callback_state(EamOpenGuard { handle }, config.atomic_numbers.len());
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
        EAM_EVAL_ERROR.with(|slot| slot.borrow_mut().clear());
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
                let eam_detail = EAM_EVAL_ERROR.with(|slot| slot.borrow().clone());
                let mut detail = if !eam_detail.is_empty() {
                    eam_detail
                } else {
                    let msg_ptr = rgpot_last_error();
                    if msg_ptr.is_null() {
                        String::new()
                    } else {
                        std::ffi::CStr::from_ptr(msg_ptr)
                            .to_string_lossy()
                            .into_owned()
                    }
                };
                if detail.is_empty() {
                    detail = format!("rgpot status {status:?}");
                }
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

    fn pair_cell() -> [f64; 9] {
        [20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0]
    }

    fn linear_pair_knots() -> (f64, f64, f64, Vec<f64>, Vec<f64>, Vec<f64>) {
        let dr = 4.0 / 15.0;
        let zeros = vec![0.0_f64; 16];
        let z2r: Vec<f64> = (0..16).map(|i| (4.0 - i as f64 * dr).max(0.0)).collect();
        (0.1, dr, 4.0, zeros.clone(), zeros, z2r)
    }

    fn copper_pair_config() -> RgpotConfig {
        RgpotConfig::default()
            .with_atomic_numbers(vec![29, 29])
            .with_default_cell(pair_cell())
    }

    fn write_row(out: &mut String, values: &[f64]) {
        for (i, value) in values.iter().enumerate() {
            if i > 0 {
                out.push(' ');
            }
            out.push_str(&format!("{value:?}"));
        }
        out.push('\n');
    }

    #[test]
    fn eam_table_memory_surface_evaluates_pair_interaction() {
        let (drho, dr, cutoff, frho, rhor, z2r) = linear_pair_knots();
        let surf = RgpotSurface::from_eam_table_memory(
            29,
            drho,
            dr,
            cutoff,
            &frho,
            &rhor,
            &z2r,
            true,
            0.5,
            copper_pair_config(),
        )
        .expect("in-memory EAMTablePot");
        assert_eq!(surf.n_atoms(), 2);
        let pos = Array1::from_vec(vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
        let (energy, forces) = surf
            .energy_and_forces(pos.view(), None)
            .expect("eval eam pair");
        assert_eq!(energy, 1.0);
        assert_eq!(
            forces,
            Array1::from_vec(vec![-1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        );
        let (again, _) = surf
            .energy_and_forces(pos.view(), None)
            .expect("second eval reuses the pair list");
        assert_eq!(again, energy);
        let far = Array1::from_vec(vec![0.0, 0.0, 0.0, 10.0, 0.0, 0.0]);
        let (far_energy, far_forces) = surf
            .energy_and_forces(far.view(), None)
            .expect("beyond cutoff");
        assert_eq!(far_energy, 0.0);
        assert_eq!(far_forces, Array1::zeros(6));
    }

    #[test]
    fn eam_table_file_matches_memory_and_rejects_a_missing_file() {
        let (drho, dr, cutoff, frho, rhor, z2r) = linear_pair_knots();
        let mut text = String::from("linear pair\nsetfl\nrgpot EAMTablePot\n1\n");
        text.push_str(&format!(
            "{} {drho:?} {} {dr:?} {cutoff:?}\n29 63.546 Cu\n",
            frho.len(),
            rhor.len()
        ));
        write_row(&mut text, &frho);
        write_row(&mut text, &rhor);
        write_row(&mut text, &z2r);
        let path =
            std::env::temp_dir().join(format!("rgflux-eam-pair-{}.setfl", std::process::id()));
        std::fs::write(&path, text).expect("write setfl");
        let from_file = RgpotSurface::from_eam_table_file(
            &path,
            EamTableFormat::Setfl,
            true,
            0.5,
            copper_pair_config(),
        )
        .expect("setfl EAMTablePot");
        let _ = std::fs::remove_file(&path);
        let from_memory = RgpotSurface::from_eam_table_memory(
            29,
            drho,
            dr,
            cutoff,
            &frho,
            &rhor,
            &z2r,
            true,
            0.5,
            copper_pair_config(),
        )
        .expect("memory EAMTablePot");
        let pos = Array1::from_vec(vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
        let (file_energy, file_forces) = from_file
            .energy_and_forces(pos.view(), None)
            .expect("eval setfl");
        let (memory_energy, memory_forces) = from_memory
            .energy_and_forces(pos.view(), None)
            .expect("eval memory");
        assert_eq!(file_energy, memory_energy);
        assert_eq!(file_forces, memory_forces);
        assert_eq!(file_energy, 1.0);

        let missing = RgpotSurface::from_eam_table_file(
            Path::new("missing-eam-table.setfl"),
            EamTableFormat::Setfl,
            true,
            0.5,
            copper_pair_config(),
        );
        assert!(missing.is_err());
    }

    #[test]
    fn eam_table_memory_rejects_bad_tables_and_unmapped_atoms() {
        let (drho, dr, cutoff, frho, rhor, z2r) = linear_pair_knots();
        let short = RgpotSurface::from_eam_table_memory(
            29,
            drho,
            dr,
            cutoff,
            &frho[..4],
            &rhor,
            &z2r,
            true,
            0.5,
            copper_pair_config(),
        );
        assert!(matches!(short, Err(FluxError::InvalidConfig(_))));
        let negative_skin = RgpotSurface::from_eam_table_memory(
            29,
            drho,
            dr,
            cutoff,
            &frho,
            &rhor,
            &z2r,
            true,
            -0.5,
            copper_pair_config(),
        );
        assert!(negative_skin.is_err());
        let surf = RgpotSurface::from_eam_table_memory(
            29,
            drho,
            dr,
            cutoff,
            &frho,
            &rhor,
            &z2r,
            true,
            0.5,
            RgpotConfig::default()
                .with_atomic_numbers(vec![8, 8])
                .with_default_cell(pair_cell()),
        )
        .expect("table of copper");
        let pos = Array1::from_vec(vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
        let err = surf.energy_and_forces(pos.view(), None);
        match err {
            Err(FluxError::Surface(msg)) => {
                assert!(
                    msg.contains("atomic number"),
                    "unmapped atom error lost the C++ text: {msg}"
                );
            }
            other => panic!("expected Surface with the C++ text, got {other:?}"),
        }
    }

    #[test]
    fn eam_table_memory_rejects_nonpositive_steps_and_short_spans() {
        let (drho, dr, cutoff, frho, rhor, z2r) = linear_pair_knots();
        let open = |drho: f64, dr: f64, cutoff: f64| {
            RgpotSurface::from_eam_table_memory(
                29,
                drho,
                dr,
                cutoff,
                &frho,
                &rhor,
                &z2r,
                true,
                0.5,
                copper_pair_config(),
            )
        };
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -0.1] {
            assert!(
                matches!(open(bad, dr, cutoff), Err(FluxError::InvalidConfig(_))),
                "drho {bad}"
            );
            assert!(
                matches!(open(drho, bad, cutoff), Err(FluxError::InvalidConfig(_))),
                "dr {bad}"
            );
            assert!(
                matches!(open(drho, dr, bad), Err(FluxError::InvalidConfig(_))),
                "cutoff {bad}"
            );
        }
        assert!(matches!(
            open(drho, -(4.0 / 15.0), cutoff),
            Err(FluxError::InvalidConfig(_))
        ));
        assert!(matches!(
            open(drho, 1.0e-300, cutoff),
            Err(FluxError::InvalidConfig(_))
        ));
        let mut short_z = z2r.clone();
        short_z.pop();
        let mismatch = RgpotSurface::from_eam_table_memory(
            29,
            drho,
            dr,
            cutoff,
            &frho,
            &rhor,
            &short_z,
            true,
            0.5,
            copper_pair_config(),
        );
        assert!(matches!(mismatch, Err(FluxError::InvalidConfig(_))));
        let short_rho = RgpotSurface::from_eam_table_memory(
            29,
            drho,
            dr,
            cutoff,
            &frho,
            &rhor[..4],
            &z2r[..4],
            true,
            0.5,
            copper_pair_config(),
        );
        assert!(matches!(short_rho, Err(FluxError::InvalidConfig(_))));

        let mut err_buf = [0 as c_char; 256];
        let handle = unsafe {
            rgflux_eam_table_open_memory(
                29,
                2,
                0.1,
                2,
                0.25,
                4.0,
                frho.as_ptr(),
                rhor.as_ptr(),
                z2r.as_ptr(),
                1,
                0.5,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        assert!(handle.is_null());
        let msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }.to_string_lossy();
        assert!(
            msg.contains("5 knots"),
            "C open_memory accepted a 2-knot table: {msg}"
        );

        let mut probe_c = |drho: f64, dr: f64, cutoff: f64| {
            err_buf.fill(0);
            let handle = unsafe {
                rgflux_eam_table_open_memory(
                    29,
                    i32::try_from(frho.len()).expect("nrho"),
                    drho,
                    i32::try_from(rhor.len()).expect("nr"),
                    dr,
                    cutoff,
                    frho.as_ptr(),
                    rhor.as_ptr(),
                    z2r.as_ptr(),
                    1,
                    0.5,
                    err_buf.as_mut_ptr(),
                    err_buf.len(),
                )
            };
            if !handle.is_null() {
                unsafe { rgflux_eam_table_close(handle) };
            }
            assert!(
                handle.is_null(),
                "C open_memory accepted drho={drho} dr={dr} cutoff={cutoff}"
            );
            let msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }.to_string_lossy();
            assert!(
                msg.contains("finite and positive"),
                "C open_memory dropped the grid error: {msg}"
            );
        };
        probe_c(0.1, -0.25, 4.0);
        probe_c(-0.1, 4.0 / 15.0, 4.0);
        probe_c(0.1, 4.0 / 15.0, 0.0);
        probe_c(0.0, 4.0 / 15.0, 4.0);
        probe_c(0.1, 0.0, 4.0);
        probe_c(f64::NEG_INFINITY, 4.0 / 15.0, 4.0);
        probe_c(0.1, 4.0 / 15.0, -4.0);
        probe_c(f64::NAN, 4.0 / 15.0, 4.0);
        probe_c(0.1, f64::INFINITY, 4.0);
        assert!(matches!(
            open(1.0e-300, dr, cutoff),
            Err(FluxError::InvalidConfig(_))
        ));
    }

    #[test]
    fn eam_table_funcfl_reader_evaluates_a_pair() {
        let mut text = String::from("funcfl probe\n29 63.546\n16 0.1 16 0.25 4.0\n");
        write_row(&mut text, &[0.0; 16]);
        write_row(&mut text, &[0.1; 16]);
        write_row(&mut text, &[0.0; 16]);
        let path =
            std::env::temp_dir().join(format!("rgflux-eam-funcfl-{}.eam", std::process::id()));
        std::fs::write(&path, text).expect("write funcfl");
        let surf = RgpotSurface::from_eam_table_file(
            &path,
            EamTableFormat::Funcfl,
            true,
            0.0,
            copper_pair_config(),
        )
        .expect("funcfl EAMTablePot");
        let _ = std::fs::remove_file(&path);
        let pos = Array1::from_vec(vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
        let (energy, forces) = surf
            .energy_and_forces(pos.view(), None)
            .expect("eval funcfl");
        let expected = 27.2 * 0.529 * 0.1 * 0.1 / 2.0;
        assert_eq!(energy, expected);
        assert!((energy - 0.071944).abs() <= 1.0e-12);
        assert!((forces[0] + forces[3]).abs() <= 1.0e-12);
        assert!(forces[0] != 0.0);
    }

    #[test]
    fn eam_table_setfl_nonzero_embedding_matches_memory() {
        let (drho, dr, cutoff, _frho, _rhor, z2r) = linear_pair_knots();
        let frho: Vec<f64> = (0..16).map(|i| 0.05 * i as f64 * drho).collect();
        let density = vec![0.2; 16];
        let mut text = String::from("embedding pair\nsetfl\nrgpot EAMTablePot\n1\n");
        text.push_str(&format!(
            "{} {drho:?} {} {dr:?} {cutoff:?}\n29 63.546 Cu\n",
            frho.len(),
            density.len()
        ));
        write_row(&mut text, &frho);
        write_row(&mut text, &density);
        write_row(&mut text, &z2r);
        let path =
            std::env::temp_dir().join(format!("rgflux-eam-embed-{}.setfl", std::process::id()));
        std::fs::write(&path, text).expect("write setfl");
        let from_file = RgpotSurface::from_eam_table_file(
            &path,
            EamTableFormat::Setfl,
            true,
            0.5,
            copper_pair_config(),
        )
        .expect("setfl with embedding");
        let _ = std::fs::remove_file(&path);
        let from_memory = RgpotSurface::from_eam_table_memory(
            29,
            drho,
            dr,
            cutoff,
            &frho,
            &density,
            &z2r,
            true,
            0.5,
            copper_pair_config(),
        )
        .expect("memory with embedding");
        let pos = Array1::from_vec(vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
        let (file_energy, file_forces) = from_file
            .energy_and_forces(pos.view(), None)
            .expect("eval setfl embedding");
        let (memory_energy, memory_forces) = from_memory
            .energy_and_forces(pos.view(), None)
            .expect("eval memory embedding");
        assert_eq!(file_energy, memory_energy);
        assert_eq!(file_forces, memory_forces);
        assert_eq!(file_energy, 1.02);
    }

    #[test]
    fn eam_table_setfl_tiny_spacing_is_refused() {
        let (drho, _dr, cutoff, frho, rhor, z2r) = linear_pair_knots();
        let mut text = String::from("tiny dr\nsetfl\nrgpot EAMTablePot\n1\n");
        text.push_str(&format!(
            "{} {drho:?} {} {:?} {cutoff:?}\n29 63.546 Cu\n",
            frho.len(),
            rhor.len(),
            1.0e-300_f64
        ));
        write_row(&mut text, &frho);
        write_row(&mut text, &rhor);
        write_row(&mut text, &z2r);
        let path =
            std::env::temp_dir().join(format!("rgflux-eam-tiny-{}.setfl", std::process::id()));
        std::fs::write(&path, text).expect("write tiny setfl");
        let opened = RgpotSurface::from_eam_table_file(
            &path,
            EamTableFormat::Setfl,
            true,
            0.5,
            copper_pair_config(),
        );
        let _ = std::fs::remove_file(&path);
        match opened {
            Err(FluxError::Surface(msg)) => {
                assert!(
                    msg.contains("bin index"),
                    "tiny dr was not refused at construction: {msg}"
                );
            }
            Err(other) => panic!("expected Surface, got {other:?}"),
            Ok(_) => panic!("tiny grid opened"),
        }
    }

    #[test]
    fn eam_table_finnis_sinclair_reader_evaluates_two_elements() {
        let mut text = String::from("c\nc\nc\n2 A B\n11 1.0 11 0.5 4.0\n");
        let row = |values: &[f64]| {
            let mut line = String::new();
            write_row(&mut line, values);
            line
        };
        text.push_str("13 1.0 1.0 fcc\n");
        text.push_str(&row(&(0..11).map(|m| -(m as f64)).collect::<Vec<_>>()));
        text.push_str(&row(&[1.0; 11]));
        text.push_str(&row(&[3.0; 11]));
        text.push_str("28 1.0 1.0 fcc\n");
        text.push_str(&row(&(0..11).map(|m| -2.0 * m as f64).collect::<Vec<_>>()));
        text.push_str(&row(&[5.0; 11]));
        text.push_str(&row(&[7.0; 11]));
        text.push_str(&row(&[0.0; 11]));
        text.push_str(&row(&[0.0; 11]));
        text.push_str(&row(&[0.0; 11]));
        let path =
            std::env::temp_dir().join(format!("rgflux-eam-fs-{}.eam.fs", std::process::id()));
        std::fs::write(&path, text).expect("write finnis-sinclair");
        let eval = |atomic_numbers: Vec<i32>, expected: f64| {
            let cfg = RgpotConfig::default()
                .with_atomic_numbers(atomic_numbers)
                .with_default_cell(pair_cell());
            let surf = RgpotSurface::from_eam_table_file(
                &path,
                EamTableFormat::FinnisSinclair,
                false,
                0.0,
                cfg,
            )
            .expect("finnis-sinclair table");
            let pos = Array1::from_vec(vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
            let (energy, _) = surf
                .energy_and_forces(pos.view(), None)
                .expect("eval finnis-sinclair");
            assert_eq!(energy, expected);
        };
        eval(vec![13, 13], -2.0);
        eval(vec![13, 28], -11.0);
        eval(vec![28, 28], -28.0);
        let _ = std::fs::remove_file(&path);
    }

    fn assert_refused(result: Result<(f64, Array1<f64>), FluxError>) {
        if let Ok((energy, _)) = result {
            panic!("returned a finite energy {energy}");
        }
    }

    fn assert_surface_err(result: Result<RgpotSurface, FluxError>, what: &str) -> String {
        match result {
            Err(FluxError::Surface(msg)) => msg,
            Err(other) => panic!("{what}: {other}"),
            Ok(_) => panic!("{what} opened"),
        }
    }

    fn memory_pair(
        drho: f64,
        dr: f64,
        cutoff: f64,
        frho: &[f64],
        rhor: &[f64],
        z2r: &[f64],
        atomic_numbers: Vec<i32>,
    ) -> Result<RgpotSurface, FluxError> {
        let cfg = RgpotConfig::default()
            .with_atomic_numbers(atomic_numbers)
            .with_default_cell(pair_cell());
        RgpotSurface::from_eam_table_memory(29, drho, dr, cutoff, frho, rhor, z2r, true, 0.5, cfg)
    }

    #[test]
    fn eam_table_refuses_place_cast_overflow() {
        let dr = 4.0 / 15.0;
        let cutoff = 4.0;
        let mut frho = vec![0.0; 8];
        frho[7] = 100.0;
        let rhor = vec![1.0; 8];
        let z2r = vec![0.0; 8];
        let numbers = vec![29, 29];
        let overflow = memory_pair(4.0e-10, dr, cutoff, &frho, &rhor, &z2r, numbers.clone());
        match overflow {
            Err(FluxError::InvalidConfig(msg)) => assert!(msg.contains("bin index"), "{msg}"),
            Err(other) => panic!("drho 4e-10: {other}"),
            Ok(_) => panic!("drho 4e-10 opened"),
        }
        let radial = memory_pair(0.1, 1.0e-9, cutoff, &frho, &rhor, &z2r, numbers.clone());
        match radial {
            Err(FluxError::InvalidConfig(msg)) => assert!(msg.contains("bin index"), "{msg}"),
            Err(other) => panic!("dr 1e-9: {other}"),
            Ok(_) => panic!("dr 1e-9 opened"),
        }
        let tiny_drho = memory_pair(1.0e-300, dr, cutoff, &frho, &rhor, &z2r, numbers.clone());
        assert!(
            matches!(tiny_drho, Err(FluxError::InvalidConfig(_))),
            "drho 1e-300 opened"
        );
        let subnormal = memory_pair(0.1, 1.0e-320, 1.0e-320, &frho, &rhor, &z2r, numbers.clone());
        assert!(
            matches!(subnormal, Err(FluxError::InvalidConfig(_))),
            "subnormal dr opened"
        );

        let mut err_buf = [0 as c_char; 256];
        let mut probe = |drho: f64, dr: f64, cutoff: f64, needle: &str| {
            err_buf.fill(0);
            let handle = unsafe {
                rgflux_eam_table_open_memory(
                    29,
                    8,
                    drho,
                    8,
                    dr,
                    cutoff,
                    frho.as_ptr(),
                    rhor.as_ptr(),
                    z2r.as_ptr(),
                    1,
                    0.5,
                    err_buf.as_mut_ptr(),
                    err_buf.len(),
                )
            };
            if !handle.is_null() {
                unsafe { rgflux_eam_table_close(handle) };
            }
            let msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }.to_string_lossy();
            assert!(
                handle.is_null(),
                "C open_memory accepted {drho} {dr} {cutoff}: {msg}"
            );
            assert!(msg.contains(needle), "C open_memory message {msg}");
        };
        probe(4.0e-10, dr, cutoff, "bin index");
        probe(0.1, 1.0e-9, cutoff, "bin index");
        probe(1.0e-300, dr, cutoff, "bin index");
        probe(0.1, 1.0e-320, 1.0e-320, "bin index");

        let steep = memory_pair(5.0e-10, dr, cutoff, &frho, &rhor, &z2r, numbers.clone())
            .expect("unit density still fits at drho 5e-10");
        let pos = Array1::from_vec(vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
        let (energy, _) = steep
            .energy_and_forces(pos.view(), None)
            .expect("linear extension above rhomax");
        let rhomax = 7.0 * 5.0e-10;
        let one = 100.0 + (100.0 / 5.0e-10) * (1.0 - rhomax);
        let expected = 2.0 * one;
        assert!(
            energy.is_finite() && (energy - expected).abs() <= 1.0e-6 * expected.abs(),
            "energy {energy} expected {expected}"
        );

        let mut wide = vec![0.0; 16];
        for value in &mut wide[8..] {
            *value = 100.0;
        }
        let dense = vec![10.0; 16];
        let pair = vec![0.0; 16];
        let summed = memory_pair(4.0e-9, dr, cutoff, &wide, &dense, &pair, numbers.clone());
        match summed {
            Err(_) => {}
            Ok(surf) => assert_refused(surf.energy_and_forces(pos.view(), None)),
        }
        let safe = memory_pair(5.0e-9, dr, cutoff, &wide, &dense, &pair, numbers)
            .expect("density 10 fits at drho 5e-9");
        let (energy, _) = safe
            .energy_and_forces(pos.view(), None)
            .expect("flat embedding above rhomax");
        assert_eq!(energy, 200.0);

        let ones = vec![1.0; 8];
        let zeros = vec![0.0; 8];
        let three = RgpotConfig::default()
            .with_atomic_numbers(vec![29, 29, 29])
            .with_default_cell(pair_cell());
        let crowded = RgpotSurface::from_eam_table_memory(
            29, 8.0e-10, dr, cutoff, &zeros, &ones, &zeros, true, 0.5, three,
        )
        .expect("one neighbor fits");
        let trio = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
        let crowded_eval = crowded.energy_and_forces(trio.view(), None);
        match crowded_eval {
            Err(FluxError::Surface(msg)) => {
                assert!(msg.contains("place index"), "{msg}");
            }
            other => panic!("summed density overflow returned {other:?}"),
        }
    }

    #[test]
    fn eam_table_refuses_nonfinite_knots() {
        let (drho, dr, cutoff, mut frho, rhor, z2r) = linear_pair_knots();
        frho[0] = f64::NAN;
        let nan_embed = memory_pair(drho, dr, cutoff, &frho, &rhor, &z2r, vec![29, 29]);
        assert!(matches!(nan_embed, Err(FluxError::InvalidConfig(_))));
        frho[0] = 0.0;
        let mut density = rhor.clone();
        density[1] = f64::INFINITY;
        let inf_density = memory_pair(drho, dr, cutoff, &frho, &density, &z2r, vec![29, 29]);
        assert!(matches!(inf_density, Err(FluxError::InvalidConfig(_))));
        let mut pair = z2r.clone();
        pair[2] = f64::NEG_INFINITY;
        let inf_pair = memory_pair(drho, dr, cutoff, &frho, &rhor, &pair, vec![29, 29]);
        assert!(matches!(inf_pair, Err(FluxError::InvalidConfig(_))));

        let mut err_buf = [0 as c_char; 256];
        let handle = unsafe {
            rgflux_eam_table_open_memory(
                29,
                i32::try_from(frho.len()).unwrap(),
                drho,
                i32::try_from(rhor.len()).unwrap(),
                dr,
                cutoff,
                frho.as_ptr(),
                density.as_ptr(),
                z2r.as_ptr(),
                1,
                0.5,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        assert!(handle.is_null());
        let msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }.to_string_lossy();
        assert!(msg.contains("finite"), "{msg}");
    }

    #[test]
    fn eam_eval_prefers_cpp_text_over_stale_last_error() {
        let _ =
            unsafe { rgpot_core::tensor::rgpot_tensor_owned_cpu_f64_2d(std::ptr::null(), 1, 3) };
        let (drho, dr, cutoff, frho, rhor, z2r) = linear_pair_knots();
        let surf = memory_pair(drho, dr, cutoff, &frho, &rhor, &z2r, vec![8, 8]).expect("copper");
        let pos = Array1::from_vec(vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
        match surf.energy_and_forces(pos.view(), None) {
            Err(FluxError::Surface(msg)) => {
                assert!(
                    msg.contains("atomic number"),
                    "stale last_error hid the C++ text: {msg}"
                );
                assert!(
                    !msg.contains("data is NULL"),
                    "stale last_error replaced the C++ text: {msg}"
                );
            }
            other => panic!("expected the atomic-number Surface, got {other:?}"),
        }
    }

    fn write_table(name: &str, text: &str) -> std::path::PathBuf {
        let path =
            std::env::temp_dir().join(format!("rgflux-eam-{name}-{}.set", std::process::id()));
        std::fs::write(&path, text).expect("write table");
        path
    }

    fn open_table(
        path: &std::path::Path,
        format: EamTableFormat,
    ) -> Result<RgpotSurface, FluxError> {
        RgpotSurface::from_eam_table_file(path, format, true, 0.0, copper_pair_config())
    }

    #[test]
    fn eam_table_file_rejects_short_nan_and_funcfl_headers() {
        let short_knots = "\
c
c
c
1
4 0.1 4 0.25 4.0
29 63.546 Cu
";
        let path = write_table("short-knots", short_knots);
        let opened = open_table(&path, EamTableFormat::Setfl);
        let _ = std::fs::remove_file(&path);
        let msg = assert_surface_err(opened, "short knot count");
        assert!(msg.contains("5 knots"), "{msg}");

        let short_row = "\
c
c
c
1
8 0.1 8 0.25 4.0
29 63.546 Cu
1 2 3
";
        let path = write_table("short-row", short_row);
        let opened = open_table(&path, EamTableFormat::Setfl);
        let _ = std::fs::remove_file(&path);
        let msg = assert_surface_err(opened, "short row");
        assert!(msg.contains("short"), "{msg}");

        let mut nan_row = String::from(
            "\
c
c
c
1
8 0.1 8 0.25 4.0
29 63.546 Cu
",
        );
        nan_row.push_str("nan 0 0 0 0 0 0 0\n");
        nan_row.push_str("0 0 0 0 0 0 0 0\n");
        nan_row.push_str("0 0 0 0 0 0 0 0\n");
        let path = write_table("nan", &nan_row);
        let opened = open_table(&path, EamTableFormat::Setfl);
        let _ = std::fs::remove_file(&path);
        let msg = assert_surface_err(opened, "nan knots");
        assert!(msg.contains("finite"), "{msg}");

        let mut funcfl = String::from("five knots\n29 63.546\n5 0.1 5 1.0 4.0\n");
        funcfl.push_str("0 0 0 0 0\n0 0 0 0 0\n0 0 0 0 0\n");
        let path = write_table("funcfl-five", &funcfl);
        let opened = open_table(&path, EamTableFormat::Funcfl);
        let _ = std::fs::remove_file(&path);
        let msg = assert_surface_err(opened, "5-knot funcfl");
        assert!(msg.contains("resampl"), "{msg}");

        let mut huge = String::from("huge drho\n29 63.546\n5 1e308 5 1.0 4.0\n");
        huge.push_str("0 0 0 0 0\n0 0 0 0 0\n0 0 0 0 0\n");
        let path = write_table("funcfl-huge", &huge);
        let opened = open_table(&path, EamTableFormat::Funcfl);
        let _ = std::fs::remove_file(&path);
        let msg = assert_surface_err(opened, "infinite drho");
        assert!(!msg.contains("5 knots"), "{msg}");
        assert!(
            msg.contains("not finite") || msg.contains("bin index"),
            "{msg}"
        );

        let mut blank_header = String::from("huge drho\n\n29 63.546\n5 1e308 5 1.0 4.0\n");
        blank_header.push_str("0 0 0 0 0\n0 0 0 0 0\n0 0 0 0 0\n");
        let path = write_table("funcfl-blank-header", &blank_header);
        let opened = open_table(&path, EamTableFormat::Funcfl);
        let _ = std::fs::remove_file(&path);
        let msg = assert_surface_err(opened, "blank funcfl header");
        assert!(!msg.contains("5 knots"), "{msg}");
        assert!(msg.contains("span"), "{msg}");

        let mut infinite_dr = String::from("huge dr\n29 63.546\n5 0.1 5 inf 4.0\n");
        infinite_dr.push_str("0 0 0 0 0\n0 0 0 0 0\n0 0 0 0 0\n");
        let path = write_table("funcfl-inf-dr", &infinite_dr);
        let opened = open_table(&path, EamTableFormat::Funcfl);
        let _ = std::fs::remove_file(&path);
        let msg = assert_surface_err(opened, "infinite dr");
        assert_eq!(msg, "EAM table header could not be read", "{msg}");
    }

    #[test]
    fn eam_table_setfl_comment_may_say_nan_and_inf() {
        let mut text = String::from("comment with nan in it\nsecond line has inf\nthird\n1\n");
        text.push_str("8 0.1 8 0.25 4.0\n29 63.546 Cu\n");
        text.push_str("0 0 0 0 0 0 0 0\n");
        text.push_str("0 0 0 0 0 0 0 0\n");
        text.push_str("0 0 0 0 0 0 0 0\n");
        let path = write_table("comment-nan", &text);
        let opened = open_table(&path, EamTableFormat::Setfl);
        let _ = std::fs::remove_file(&path);
        match opened {
            Ok(surf) => {
                let pos = Array1::from_vec(vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
                let (energy, _) = surf
                    .energy_and_forces(pos.view(), None)
                    .expect("comment nan still evaluates");
                assert_eq!(energy, 0.0);
            }
            Err(FluxError::Surface(msg)) => {
                panic!("comment nan was scanned as a knot: {msg}");
            }
            Err(other) => panic!("comment nan: {other}"),
        }
    }

    #[test]
    fn eam_table_comment_line_infinity_model_is_not_a_knot() {
        let mut text = String::from("infinity model\nsecond line has nan\nthird\n1\n");
        text.push_str("8 0.1 8 0.25 4.0\n29 63.546 Cu\n");
        text.push_str("0 0 0 0 0 0 0 0\n");
        text.push_str("0 0 0 0 0 0 0 0\n");
        text.push_str("0 0 0 0 0 0 0 0\n");
        let path = write_table("comment-infinity-model", &text);
        let opened = open_table(&path, EamTableFormat::Setfl);
        let _ = std::fs::remove_file(&path);
        match opened {
            Ok(surf) => {
                let pos = Array1::from_vec(vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
                let (energy, _) = surf
                    .energy_and_forces(pos.view(), None)
                    .expect("infinity model comment still evaluates");
                assert_eq!(energy, 0.0);
            }
            Err(FluxError::Surface(msg)) => {
                panic!("infinity model was scanned as a knot: {msg}");
            }
            Err(other) => panic!("infinity model: {other}"),
        }

        let mut funcfl = String::from("infinity model\n29 63.546\n8 0.1 8 0.25 4.0\n");
        funcfl.push_str("0 0 0 0 0 0 0 0\n");
        funcfl.push_str("0 0 0 0 0 0 0 0\n");
        funcfl.push_str("0 0 0 0 0 0 0 0\n");
        let path = write_table("funcfl-infinity-model", &funcfl);
        let opened = open_table(&path, EamTableFormat::Funcfl);
        let _ = std::fs::remove_file(&path);
        match opened {
            Ok(surf) => {
                let pos = Array1::from_vec(vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
                let (energy, _) = surf
                    .energy_and_forces(pos.view(), None)
                    .expect("funcfl infinity model comment still evaluates");
                assert_eq!(energy, 0.0);
            }
            Err(FluxError::Surface(msg)) => {
                panic!("funcfl infinity model was scanned as a knot: {msg}");
            }
            Err(other) => panic!("funcfl infinity model: {other}"),
        }
    }

    #[test]
    fn eam_table_element_line_discards_an_unused_nan() {
        let mut text = String::from("c\nc\nc\n1\n");
        text.push_str("8 0.1 8 0.25 4.0\n29 63.546 Cu nan\n");
        text.push_str("0 0 0 0 0 0 0 0\n");
        text.push_str("0 0 0 0 0 0 0 0\n");
        text.push_str("0 0 0 0 0 0 0 0\n");
        let path = write_table("element-nan", &text);
        let opened = open_table(&path, EamTableFormat::Setfl);
        let _ = std::fs::remove_file(&path);
        match opened {
            Ok(surf) => {
                let pos = Array1::from_vec(vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
                let (energy, _) = surf
                    .energy_and_forces(pos.view(), None)
                    .expect("discarded element word still evaluates");
                assert_eq!(energy, 0.0);
            }
            Err(FluxError::Surface(msg)) => {
                panic!("unused element word was refused: {msg}");
            }
            Err(other) => panic!("unused element word: {other}"),
        }

        let mut used = String::from("c\nc\nc\n1\n");
        used.push_str("8 0.1 8 0.25 4.0\n29 63.546 Cu\n");
        used.push_str("0 0 0 0 0 0 0 nan\n");
        used.push_str("0 0 0 0 0 0 0 0\n");
        used.push_str("0 0 0 0 0 0 0 0\n");
        let path = write_table("used-nan", &used);
        let opened = open_table(&path, EamTableFormat::Setfl);
        let _ = std::fs::remove_file(&path);
        let msg = assert_surface_err(opened, "nan knot");
        assert!(msg.contains("finite"), "{msg}");
    }

    #[test]
    fn eam_table_header_refuses_inf_nan_and_hex_floats() {
        let expect = |name: &str, text: String, format: EamTableFormat| {
            let path = write_table(name, &text);
            let opened = open_table(&path, format);
            let _ = std::fs::remove_file(&path);
            let msg = assert_surface_err(opened, name);
            assert_eq!(msg, "EAM table header could not be read", "{name}: {msg}");
        };
        for token in ["inf", "nan", "infinity", "0x1p+3"] {
            expect(
                &format!("funcfl-{token}"),
                format!("comment\n29 63.546\n5 {token} 5 1.0 4.0\n"),
                EamTableFormat::Funcfl,
            );
            expect(
                &format!("setfl-{token}"),
                format!("c\nc\nc\n1\n5 {token} 5 1.0 4.0\n"),
                EamTableFormat::Setfl,
            );
        }
    }

    #[test]
    fn eam_table_blank_line_and_cutoff_suffix_keep_the_span_guard() {
        let zeros = "0 0 0 0 0\n0 0 0 0 0\n0 0 0 0 0\n";
        let expect_span = |name: &str, text: String, format: EamTableFormat| {
            let path = write_table(name, &text);
            let opened = open_table(&path, format);
            let _ = std::fs::remove_file(&path);
            let msg = assert_surface_err(opened, name);
            assert_eq!(msg, "EAM grid span is not finite", "{name}: {msg}");
        };

        let mut blank = String::from("huge drho\n\n29 63.546\n5 1e308 5 1.0 4.0\n");
        blank.push_str(zeros);
        expect_span("funcfl-blank-between", blank, EamTableFormat::Funcfl);

        let mut split = String::from("huge drho\n\n29\n63.546\n5 1e308 5 1.0 4.0\n");
        split.push_str(zeros);
        expect_span("funcfl-blank-split-mass", split, EamTableFormat::Funcfl);

        let mut junk = String::from("huge drho\n29 63.546\n5 1e308 5 1.0 4.0x\n");
        junk.push_str(zeros);
        expect_span("funcfl-cutoff-suffix", junk, EamTableFormat::Funcfl);

        let setfl = String::from("c\nc\nc\n1\n\n5 1e308 5 1.0 4.0\n29 63.546 Cu\n");
        expect_span("setfl-blank-before-grid", setfl, EamTableFormat::Setfl);
    }

    #[test]
    fn eam_table_int_min_header_is_checked_before_the_reader() {
        let expect = |name: &str, text: String, message: &str| {
            let path = write_table(name, &text);
            let opened = open_table(&path, EamTableFormat::Funcfl);
            let _ = std::fs::remove_file(&path);
            let msg = assert_surface_err(opened, name);
            assert_eq!(msg, message, "{name}: {msg}");
        };

        let nrho = 306783379;
        let spline = "EAM spline index does not fit in int";
        expect(
            "int-min-spline",
            format!("comment\n{} 63.546\n{nrho} 0.1 5 1.0 4.0\n", i32::MIN),
            spline,
        );
        expect(
            "z29-spline",
            format!("comment\n29 63.546\n{nrho} 0.1 5 1.0 4.0\n"),
            spline,
        );
        expect(
            "int-min-span",
            format!("comment\n{} 63.546\n5 1e308 5 1.0 4.0\n", i32::MIN),
            "EAM grid span is not finite",
        );
    }

    #[test]
    fn eam_table_place_bound_is_the_cubic_maximum() {
        let dr = 0.5;
        let cutoff = 4.0;
        let zeros = vec![0.0; 8];
        let mut spike = vec![0.0; 8];
        spike[3] = 1.0e6;
        let open = |drho: f64, density: &[f64], numbers: Vec<i32>| {
            let cfg = RgpotConfig::default()
                .with_atomic_numbers(numbers)
                .with_default_cell(pair_cell());
            RgpotSurface::from_eam_table_memory(
                29, drho, dr, cutoff, &zeros, density, &zeros, false, 0.0, cfg,
            )
        };

        let two = open(1.0e-3, &spike, vec![29, 29]).expect("fitting spike opens");
        let near = Array1::from_vec(vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
        let (energy, _) = two
            .energy_and_forces(near.view(), None)
            .expect("two atoms at 2 Angstrom evaluate");
        assert_eq!(energy, 0.0);

        let crowded = open(1.0e-3, &spike, vec![29, 29, 29, 29]).expect("crowded spike opens");
        let peak = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.5,
        ]);
        match crowded.energy_and_forces(peak.view(), None) {
            Err(FluxError::Surface(msg)) => assert!(msg.contains("place index"), "{msg}"),
            other => panic!("three neighbors at the spike returned {other:?}"),
        }

        let ones = vec![1.0; 8];
        let far = open(8.0e-10, &ones, vec![29, 29, 29]).expect("distant flat density opens");
        let distant = Array1::from_vec(vec![0.0, 0.0, 0.0, 50.0, 0.0, 0.0, 80.0, 0.0, 0.0]);
        let (energy, _) = far
            .energy_and_forces(distant.view(), None)
            .expect("atoms at 0, 50 and 80 have no neighbor inside the cutoff");
        assert_eq!(energy, 0.0);

        let mut plateau = vec![0.0; 8];
        plateau[4] = 1.0e154;
        plateau[5] = 1.0e154;
        plateau[6] = 1.0e154;
        plateau[7] = 1.0e154;
        let mut embedding = vec![0.0; 8];
        embedding[7] = 1.0;
        let cfg = RgpotConfig::default()
            .with_atomic_numbers(vec![29, 29, 29])
            .with_default_cell(pair_cell());
        let overflow = RgpotSurface::from_eam_table_memory(
            29, 1.0e145, 0.5, 3.0, &embedding, &plateau, &zeros, false, 0.0, cfg,
        )
        .expect("overflowing discriminant opens");
        let centre = Array1::from_vec(vec![-2.17895, 0.0, 0.0, 0.0, 0.0, 0.0, 2.17895, 0.0, 0.0]);
        match overflow.energy_and_forces(centre.view(), None) {
            Err(FluxError::Surface(msg)) => assert!(msg.contains("place index"), "{msg}"),
            other => panic!("discriminant overflow returned {other:?}"),
        }

        // Negated eight-knot row. b*b overflows on the interior cubic, the
        // knots still fit, and one neighbor makes the place cast miss int.
        let raw = [
            1.6848978999193908e199,
            -4.629903741144443e199,
            -9.67137533819585e199,
            6.13767193282331e199,
            -2.456458098164722e199,
            8.918230826063796e199,
            4.069341781847155e199,
            -3.879906489180879e199,
        ];
        let density: Vec<f64> = raw.iter().map(|value| -value).collect();
        let mut tall = vec![0.0; 8];
        tall[7] = 1000.0;
        let cfg = RgpotConfig::default()
            .with_atomic_numbers(vec![29, 29])
            .with_default_cell(pair_cell());
        let squared = RgpotSurface::from_eam_table_memory(
            29,
            4.50358510679778e190,
            0.5,
            4.0,
            &tall,
            &density,
            &zeros,
            false,
            0.0,
            cfg,
        )
        .expect("squared overflow opens");
        let close = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.911875, 0.0, 0.0]);
        match squared.energy_and_forces(close.view(), None) {
            Err(FluxError::Surface(msg)) => assert!(msg.contains("place index"), "{msg}"),
            other => panic!("squared overflow returned {other:?}"),
        }
    }

    #[test]
    fn eam_table_sign_flipped_density_refuses_an_overflowing_place() {
        // Negated eight-knot row. The interior cubic exceeds every knot, and
        // the derivative discriminant is not finite. One neighbor at this
        // distance makes the place cast miss int; the endpoint magnitude fits.
        let raw = [
            -7.65731413582964e179,
            -5.590789262643429e179,
            5.891659434211519e179,
            -3.349277015606891e179,
            6.318261930673191e179,
            -7.987849595678076e179,
            -7.072830221753923e179,
            3.953412803824776e179,
        ];
        let rhor: Vec<f64> = raw.iter().map(|value| -value).collect();
        let mut embedding = vec![0.0; 8];
        embedding[7] = 100.0;
        let zeros = vec![0.0; 8];
        let knot_max = rhor.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
        let cfg = RgpotConfig::default()
            .with_atomic_numbers(vec![29, 29])
            .with_default_cell(pair_cell());
        let opened = RgpotSurface::from_eam_table_memory(
            29,
            knot_max / 1.9e9,
            1.0,
            8.0,
            &embedding,
            &rhor,
            &zeros,
            false,
            0.0,
            cfg,
        )
        .expect("sign-flipped density opens");
        let pos = Array1::from_vec(vec![0.0, 0.0, 0.0, 5.3713, 0.0, 0.0]);
        match opened.energy_and_forces(pos.view(), None) {
            Err(FluxError::Surface(msg)) => assert!(msg.contains("place index"), "{msg}"),
            other => panic!("sign-flipped density returned {other:?}"),
        }
    }

    #[test]
    fn eam_table_step_density_refuses_the_interior_place() {
        let drho = 1.0e150;
        let step = 0.97 * f64::from(i32::MAX) * drho;
        let mut density = vec![0.0; 8];
        density[4] = step;
        density[5] = step;
        density[6] = step;
        density[7] = step;
        let mut embedding = vec![0.0; 8];
        embedding[7] = 100.0;
        let pair = vec![1.0; 8];
        let cfg = RgpotConfig::default()
            .with_atomic_numbers(vec![29, 29])
            .with_default_cell(pair_cell());
        let opened = RgpotSurface::from_eam_table_memory(
            29, drho, 1.0, 6.0, &embedding, &density, &pair, false, 0.0, cfg,
        )
        .expect("step density opens");
        let pos = Array1::from_vec(vec![0.0, 0.0, 0.0, 4.357925, 0.0, 0.0]);
        match opened.energy_and_forces(pos.view(), None) {
            Err(FluxError::Surface(msg)) => assert!(msg.contains("place index"), "{msg}"),
            other => panic!("step density returned {other:?}"),
        }
    }

    #[test]
    fn eam_table_cancelled_critical_sample_is_refused() {
        let density = [
            101222.78926303517,
            -1.1930726680185628e-16,
            -3340.9470975651357,
            202986064.250324,
            202986064.250324,
            13302505.579372644,
            93140925.68529058,
            -28091.831237726492,
        ];
        let mut embedding = vec![0.0; density.len()];
        embedding[1] = 100.0;
        let pair = vec![0.0; density.len()];
        let cfg = RgpotConfig::default()
            .with_atomic_numbers(vec![29, 29])
            .with_default_cell(pair_cell());
        let opened = RgpotSurface::from_eam_table_memory(
            29,
            0.11011303154958862,
            1.0,
            6.0,
            &embedding,
            &density,
            &pair,
            false,
            0.0,
            cfg,
        )
        .expect("cancelled critical sample opens");
        let pos = Array1::from_vec(vec![0.0, 0.0, 0.0, 3.5, 0.0, 0.0]);
        match opened.energy_and_forces(pos.view(), None) {
            Err(FluxError::Surface(msg)) => assert!(msg.contains("place index"), "{msg}"),
            other => panic!("cancelled critical sample returned {other:?}"),
        }
    }

    #[test]
    fn eam_table_large_knot_row_refuses_the_bernstein_bound() {
        let density = [
            -8.305933995245273e199,
            3.1555802287374668e199,
            3.596781745376405e199,
            -1.243824924521969e199,
            7.803656485673778e199,
            7.290979594738755e199,
            -5.277109112004301e199,
            6.509758290242633e199,
        ];
        let zeros = vec![0.0; 8];
        let cfg = RgpotConfig::default()
            .with_atomic_numbers(vec![29, 29])
            .with_default_cell(pair_cell());
        let opened = RgpotSurface::from_eam_table_memory(
            29,
            3.8677519201211666e190,
            0.5,
            4.0,
            &zeros,
            &density,
            &zeros,
            false,
            0.0,
            cfg,
        )
        .expect("large knot row opens");
        let pos = Array1::from_vec(vec![0.0, 0.0, 0.0, 2.248875, 0.0, 0.0]);
        match opened.energy_and_forces(pos.view(), None) {
            Err(FluxError::Surface(msg)) => assert!(msg.contains("place index"), "{msg}"),
            other => panic!("large knot row returned {other:?}"),
        }
    }

    #[test]
    fn eam_table_place_uses_neighbors_inside_cutoff() {
        let dr = 4.0 / 15.0;
        let cutoff = 4.0;
        let zeros = vec![0.0; 8];
        let ones = vec![1.0; 8];
        let open = |drho: f64| {
            let cfg = RgpotConfig::default()
                .with_atomic_numbers(vec![29, 29, 29])
                .with_default_cell(pair_cell());
            RgpotSurface::from_eam_table_memory(
                29, drho, dr, cutoff, &zeros, &ones, &zeros, false, 0.0, cfg,
            )
        };
        let far = open(8.0e-10).expect("distant atoms open");
        let distant = Array1::from_vec(vec![0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 200.0, 0.0, 0.0]);
        let (energy, _) = far
            .energy_and_forces(distant.view(), None)
            .expect("atoms outside the cutoff have zero density");
        assert_eq!(energy, 0.0);

        let close = open(8.0e-10).expect("close atoms open");
        let trio = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
        match close.energy_and_forces(trio.view(), None) {
            Err(FluxError::Surface(msg)) => assert!(msg.contains("place index"), "{msg}"),
            other => panic!("atoms at 0, 1 and 2 returned {other:?}"),
        }
    }

    #[test]
    fn eam_table_unit_spike_fits_at_drho_5e_10() {
        let dr = 4.0 / 15.0;
        let mut density = vec![0.0; 8];
        density[4] = 1.0;
        let zeros = vec![0.0; 8];
        let surf = memory_pair(5.0e-10, dr, 4.0, &zeros, &density, &zeros, vec![29, 29])
            .expect("unit spike fits");
        let pos = Array1::from_vec(vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
        let (energy, _) = surf
            .energy_and_forces(pos.view(), None)
            .expect("unit spike evaluates");
        assert_eq!(energy, 0.0);
    }

    #[test]
    fn eam_table_refuses_spline_index_overflow() {
        let one = [0.0_f64];
        let nrho = (i32::MAX - 6) / 7 + 2;
        let mut err_buf = [0 as c_char; 256];
        let handle = unsafe {
            rgflux_eam_table_open_memory(
                29,
                nrho,
                0.1,
                8,
                0.25,
                4.0,
                one.as_ptr(),
                one.as_ptr(),
                one.as_ptr(),
                0,
                0.0,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            )
        };
        if !handle.is_null() {
            unsafe { rgflux_eam_table_close(handle) };
        }
        let msg = unsafe { CStr::from_ptr(err_buf.as_ptr()) }.to_string_lossy();
        assert!(handle.is_null(), "spline index opened: {msg}");
        assert!(msg.contains("spline index"), "{msg}");
    }

    #[test]
    fn eam_open_guard_closes_handle_when_scratch_panics() {
        struct ClearScratchPanic;
        impl Drop for ClearScratchPanic {
            fn drop(&mut self) {
                EAM_SCRATCH_PANIC.with(|flag| flag.set(false));
            }
        }
        let _clear = ClearScratchPanic;
        let (drho, dr, cutoff, frho, rhor, z2r) = linear_pair_knots();
        let before = EAM_CLOSE_COUNT.with(|count| count.get());
        EAM_SCRATCH_PANIC.with(|flag| flag.set(true));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = memory_pair(drho, dr, cutoff, &frho, &rhor, &z2r, vec![29, 29]);
        }));
        let after = EAM_CLOSE_COUNT.with(|count| count.get());
        assert!(result.is_err(), "scratch allocation did not panic");
        assert_eq!(after, before + 1, "open handle was not closed");
    }
}
