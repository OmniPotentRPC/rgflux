//! Direct lammpc-backed [`DynamicsEngine`] implementation.
//!
//! This backend links `liblammpc` and drives its persistent `lammpc_session_*`
//! C ABI with `ForceInput` and `PotentialConfig` Cap'n Proto messages from
//! `rgpot_core`. It supports constrained white-noise Langevin or zero-friction
//! projected velocity-Verlet dynamics and unconstrained NVE. Colored-noise
//! thermostats and unconstrained Langevin are unavailable in the current C
//! ABI and are rejected. [`DynamicsEngine::positions`] returns unwrapped
//! Cartesian coordinates. LAMMPS reports wrapped `x`; the integer image count
//! implied by the periodic cell is applied before those coordinates are
//! stored, matching the engine contract.
//!
//! ## Units and linking
//!
//! All quantities follow LAMMPS `metal` units: positions and box vectors in
//! Angstrom, velocities in Angstrom/ps, forces in eV/Angstrom, potential and
//! kinetic energies and `k_B T` in eV, masses in amu (scaled by
//! [`MASS_VELOCITY_TO_EV`] in kinetic energy and Maxwell-Boltzmann draws),
//! timesteps in ps, and Langevin friction `gamma` in `1/ps`.
//!
//! At build time, `build.rs` resolves `liblammpc` from `LAMMPC_LIB_DIR`,
//! `pkg-config` (`lammpc`), or `LAMMPC_DIR` (`lib/`, `lib64/`, or
//! `lib/*-linux-gnu/`).
//!
//! ## References
//!
//! - Goswami, R. (2025). *Efficient Exploration of Chemical Kinetics*.
//!   arXiv:2510.21368. <https://doi.org/10.48550/arXiv.2510.21368>
//! - Goswami, R., et al. (2026). *TheochemUI/eOn: v2.9.0.1*. Zenodo.
//!   <https://doi.org/10.5281/zenodo.18529496>
//! - Leimkuhler, B., and Matthews, C. (2016). *Efficient molecular dynamics
//!   using geodesic integration and solvent-solute splitting*. Proceedings of
//!   the Royal Society A, 472, 20160138. <https://doi.org/10.1098/rspa.2016.0138>
//! - Sprik, M., and Ciccotti, G. (1998). *Free energy from constrained
//!   molecular dynamics*. The Journal of Chemical Physics, 109, 7737-7744.
//!   <https://doi.org/10.1063/1.477419>

use std::ffi::{CStr, c_char, c_int, c_void};
use std::ptr::NonNull;

use capnp::message::Builder;
use capnp::serialize;
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::engine::{
    DynamicsEngine, HyperplaneConstraint, StepReport, SurfaceVelocityPolicy, ThermostatConfig,
    constraint_lagrange_multiplier, rattle_velocities_in_place, reflect_velocity_in_place,
    sample_maxwell_boltzmann_with_rng, shake_positions_in_place, validate_finite_slice,
    validate_masses, validate_unit_normal,
};
use crate::error::FluxError;

const METAL_LENGTH_UNIT: &str = "angstrom";
const METAL_ENERGY_UNIT: &str = "eV";
const MIN_POSITIVE_THERMAL_ENERGY_EV: f64 = f64::MIN_POSITIVE;
/// Relative tolerance for deciding whether a constrained-MD timestep changed.
const TIMESTEP_RELATIVE_TOL: f64 = 8.0 * f64::EPSILON;
/// LAMMPS metal-unit factor `mvv2e`: `amu * (Å/ps)^2` equals this many eV.
pub const MASS_VELOCITY_TO_EV: f64 = 0.000_103_642_696_562_621_75;
const LAMMPC_ABI_VERSION: c_int = 0;
const CAPNP_WORD_BYTES: usize = std::mem::size_of::<u64>();

#[repr(C)]
struct LammpcSession {
    _private: [u8; 0],
}

#[repr(C)]
struct LammpcResult {
    ok: c_int,
    energy_h: f64,
    message: [c_char; 512],
}

#[repr(C)]
#[derive(Clone, Copy)]
enum LammpcPlaneCoordinates {
    Cartesian = 1,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct LammpcMdConfig {
    time_step_ps: f64,
    thermal_energy_ev: f64,
    friction_per_ps: f64,
    seed: u64,
    coordinates: LammpcPlaneCoordinates,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
#[allow(dead_code)]
struct LammpcMdObservation {
    constraint_force: f64,
    force_evaluations: usize,
    steps: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
#[allow(dead_code)]
struct LammpcNveObservation {
    potential_energy_ev: f64,
    kinetic_energy_ev: f64,
}

unsafe extern "C" {
    fn lammpc_session_create_from_config(
        config_capnp: *const c_void,
        config_capnp_size_bytes: usize,
    ) -> *mut LammpcSession;
    fn lammpc_session_destroy(session: *mut LammpcSession);
    fn lammpc_abi_version() -> c_int;
    fn lammpc_session_calculate_result(
        session: *mut LammpcSession,
        force_input_capnp: *const c_void,
        force_input_capnp_size_bytes: usize,
        potential_result_capnp: *mut c_void,
        potential_result_capnp_capacity_bytes: usize,
        potential_result_capnp_size_bytes: *mut usize,
    ) -> LammpcResult;
    fn lammpc_potential_result_size_for_force_input(
        force_input_capnp: *const c_void,
        force_input_capnp_size_bytes: usize,
    ) -> usize;
    fn lammpc_session_md_initialize(
        session: *mut LammpcSession,
        force_input_capnp: *const c_void,
        force_input_capnp_size_bytes: usize,
        point: *const f64,
        normal: *const f64,
        coordinate_count: usize,
        config: *const LammpcMdConfig,
    ) -> c_int;
    fn lammpc_session_md_step(session: *mut LammpcSession, step_count: usize) -> c_int;
    fn lammpc_session_md_set_velocities(
        session: *mut LammpcSession,
        velocities_ang_ps: *const f64,
        coordinate_count: usize,
        seed: u64,
    ) -> c_int;
    fn lammpc_session_md_state(
        session: *mut LammpcSession,
        positions_ang: *mut f64,
        velocities_ang_ps: *mut f64,
        projected_forces_ev_ang: *mut f64,
        removed_velocities_ang_ps: *mut f64,
        coordinate_count: usize,
        observation: *mut LammpcMdObservation,
    ) -> c_int;
    fn lammpc_session_nve_set_positions(
        session: *mut LammpcSession,
        positions_ang: *const f64,
        coordinate_count: usize,
        cell_ang: *const f64,
        cell_len: usize,
    ) -> c_int;
    fn lammpc_session_nve_set_velocities(
        session: *mut LammpcSession,
        velocities_ang_ps: *const f64,
        coordinate_count: usize,
    ) -> c_int;
    fn lammpc_session_nve_step(
        session: *mut LammpcSession,
        step_count: usize,
        time_step_ps: f64,
    ) -> c_int;
    fn lammpc_session_nve_state(
        session: *mut LammpcSession,
        positions_ang: *mut f64,
        velocities_ang_ps: *mut f64,
        coordinate_count: usize,
        observation: *mut LammpcNveObservation,
    ) -> c_int;
    fn lammpc_last_error() -> *const c_char;
    fn lammpc_capabilities_result(
        capabilities_capnp: *mut c_void,
        capabilities_capnp_capacity_bytes: usize,
        capabilities_capnp_size_bytes: *mut usize,
    ) -> c_int;
}

/// Public configuration for [`LammpcEngine`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct LammpcConfig {
    /// Atomic numbers in live atom order (`n_atoms` entries).
    ///
    /// Default: `vec![18, 18]`.
    pub atomic_numbers: Vec<i32>,
    /// LAMMPS type map (`type i+1 -> Z`) used to build `LammpsParams`.
    ///
    /// Default: `vec![18]`.
    pub type_to_atomic_number: Vec<i32>,
    /// Per-type masses in amu (`type_to_atomic_number` order).
    ///
    /// Default: `vec![39.948]`.
    pub masses_by_type: Vec<f64>,
    /// `pair_style` arguments. Non-ASCII text, NULs, newlines, carriage
    /// returns, and command separators are rejected.
    ///
    /// Default: `"lj/cut 8.5"`.
    pub pair_style: String,
    /// `pair_coeff` lines, one entry per command. Non-ASCII text, NULs,
    /// newlines, carriage returns, and command separators are rejected.
    ///
    /// Default: `vec!["1 1 0.010323 3.405"]`.
    pub pair_coeffs: Vec<String>,
    /// Optional setup commands without embedded command separators.
    ///
    /// Empty lines and lines made only of spaces and tabs are omitted. A
    /// line of only a newline, carriage return, vertical tab, or form feed
    /// is rejected. Every other line must be ASCII. Leading spaces and
    /// tabs are kept, and the first word must be `neighbor`,
    /// `neigh_modify`, `comm_modify` or `pair_modify`. Lines that mention
    /// `if`, `partition`, `shell`, `jump`, `python`, `label`
    /// or `next` as a word are rejected. `include` is rejected except as
    /// the `neigh_modify` keyword followed by a group id. ASCII controls
    /// other than tab, quotes, `$`, `&` and `#` are rejected. Non-ASCII
    /// input is rejected first.
    ///
    /// Default: empty `Vec`.
    pub extra_setup: Vec<String>,
    /// LAMMPS boundary arguments. Non-ASCII text, NULs, newlines, carriage
    /// returns, and command separators are rejected.
    ///
    /// Default: `"p p p"`.
    pub boundary: String,
    /// Optional LAMMPS suffix (`omp`, `gpu`, `kk`). Non-ASCII text, NULs,
    /// newlines, carriage returns, and command separators are rejected.
    ///
    /// Default: empty `String`.
    pub suffix: String,
    /// Thermostat controls in metal units (`gamma` in `1/ps`, `k_B T` in `eV`).
    ///
    /// Only [`crate::engine::gle::ColoredNoise::White`] is supported.
    /// Colored-noise configurations return [`FluxError::InvalidConfig`].
    ///
    /// Default: [`ThermostatConfig::default()`].
    pub thermostat: ThermostatConfig,
    /// Optional hyperplane constraint.
    ///
    /// Default: `None`.
    pub constraint: Option<HyperplaneConstraint>,
    /// Optional row-major 3x3 cell in Angstrom.
    ///
    /// Default: `Some([30.0, 0.0, 0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 30.0])`.
    pub cell: Option<[f64; 9]>,
    /// Initial deterministic seed consumed by `lammpc_session_md_initialize`.
    ///
    /// Default: `1`.
    pub seed: u64,
}

impl Default for LammpcConfig {
    fn default() -> Self {
        Self {
            atomic_numbers: vec![18, 18],
            type_to_atomic_number: vec![18],
            masses_by_type: vec![39.948],
            pair_style: "lj/cut 8.5".to_string(),
            pair_coeffs: vec!["1 1 0.010323 3.405".to_string()],
            extra_setup: Vec::new(),
            boundary: "p p p".to_string(),
            suffix: String::new(),
            thermostat: ThermostatConfig::default(),
            constraint: None,
            cell: Some([30.0, 0.0, 0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 30.0]),
            seed: 1,
        }
    }
}

with_setters!(LammpcConfig {
    with_atomic_numbers => atomic_numbers: val Vec<i32>,
    with_type_to_atomic_number => type_to_atomic_number: val Vec<i32>,
    with_masses_by_type => masses_by_type: val Vec<f64>,
    with_pair_style => pair_style: val String,
    with_pair_coeffs => pair_coeffs: val Vec<String>,
    with_extra_setup => extra_setup: val Vec<String>,
    with_boundary => boundary: val String,
    with_suffix => suffix: val String,
    with_thermostat => thermostat: val ThermostatConfig,
    with_constraint => constraint: opt HyperplaneConstraint,
    with_cell => cell: opt [f64; 9],
    with_seed => seed: val u64,
});

/// `DynamicsEngine` implemented over `lammpc_session_*`.
pub struct LammpcEngine {
    session: NonNull<LammpcSession>,
    config_bytes: Vec<u8>,
    positions: Array1<f64>,
    velocities: Array1<f64>,
    masses: Array1<f64>,
    atomic_numbers: Vec<i32>,
    thermostat: ThermostatConfig,
    constraint: Option<HyperplaneConstraint>,
    cell: Option<[f64; 9]>,
    seed: u64,
    boundary: String,
    md_initialized: bool,
    md_timestep_ps: Option<f64>,
    nve_instance_ready: bool,
}

impl LammpcEngine {
    /// Create a new lammpc-backed engine session.
    pub fn new(config: LammpcConfig) -> Result<Self, FluxError> {
        if config.atomic_numbers.is_empty() {
            return Err(FluxError::EmptyInput("LammpcConfig::atomic_numbers"));
        }
        if config.type_to_atomic_number.is_empty() {
            return Err(FluxError::EmptyInput("LammpcConfig::type_to_atomic_number"));
        }
        if config.type_to_atomic_number.len() != config.masses_by_type.len() {
            return Err(FluxError::Shape {
                expected: config.type_to_atomic_number.len(),
                got: config.masses_by_type.len(),
                context: "LammpcConfig::masses_by_type",
            });
        }
        if config.pair_style.trim().is_empty() {
            return Err(FluxError::InvalidConfig(
                "LammpcConfig::pair_style must be non-empty",
            ));
        }
        validate_lammps_command_text(&config.pair_style, "LammpcConfig::pair_style")?;
        for coeff in &config.pair_coeffs {
            validate_lammps_command_text(coeff, "LammpcConfig::pair_coeffs")?;
        }
        validate_lammps_command_text(&config.boundary, "LammpcConfig::boundary")?;
        validate_lammps_command_text(&config.suffix, "LammpcConfig::suffix")?;
        for command in &config.extra_setup {
            validate_lammps_command_text(command, "LammpcConfig::extra_setup")?;
            validate_extra_setup_directive(command)?;
        }
        let _ = validate_lammpc_thermostat(&config.thermostat)?;

        let dof = config.atomic_numbers.len() * 3;
        let masses = expand_coordinate_masses(&config.atomic_numbers, &config)?;
        validate_masses(masses.view())?;
        if let Some(plane) = &config.constraint {
            plane.validate()?;
            if plane.normal.len() != dof {
                return Err(FluxError::Shape {
                    expected: dof,
                    got: plane.normal.len(),
                    context: "LammpcConfig::constraint.normal",
                });
            }
        }
        if let Some(cell) = &config.cell {
            for entry in cell {
                if !entry.is_finite() {
                    return Err(FluxError::NonFinite("LammpcConfig::cell"));
                }
            }
        }
        // SAFETY: the ABI query has no inputs and does not create a session.
        let actual_abi = unsafe { lammpc_abi_version() };
        if actual_abi != LAMMPC_ABI_VERSION {
            return Err(FluxError::Lammpc(format!(
                "lammpc ABI version mismatch: expected {LAMMPC_ABI_VERSION}, got {actual_abi}"
            )));
        }
        handshake_capabilities()?;
        let cfg_bytes = encode_lammps_config(&config)?;
        // SAFETY: We pass a stable byte slice and validate non-null result.
        let raw = unsafe {
            lammpc_session_create_from_config(cfg_bytes.as_ptr().cast(), cfg_bytes.len())
        };
        let session = NonNull::new(raw).ok_or_else(|| FluxError::Lammpc(last_error_message()))?;

        let mut engine = Self {
            session,
            config_bytes: cfg_bytes,
            positions: Array1::zeros(dof),
            velocities: Array1::zeros(dof),
            masses,
            atomic_numbers: config.atomic_numbers,
            thermostat: config.thermostat,
            constraint: config.constraint,
            cell: config.cell,
            seed: config.seed,
            boundary: config.boundary,
            md_initialized: false,
            md_timestep_ps: None,
            nve_instance_ready: false,
        };
        engine.project_state_onto_constraint()?;
        Ok(engine)
    }

    fn recreate_session_after_error(&mut self) -> Result<(), FluxError> {
        // SAFETY: We pass a stable encoded config buffer and validate non-null output.
        let raw = unsafe {
            lammpc_session_create_from_config(
                self.config_bytes.as_ptr().cast(),
                self.config_bytes.len(),
            )
        };
        let fresh = NonNull::new(raw).ok_or_else(|| FluxError::Lammpc(last_error_message()))?;
        // SAFETY: `self.session` was created by lammpc and is owned by this engine.
        unsafe { lammpc_session_destroy(self.session.as_ptr()) };
        self.session = fresh;
        self.md_initialized = false;
        self.md_timestep_ps = None;
        self.nve_instance_ready = false;
        Ok(())
    }

    fn project_state_onto_constraint(&mut self) -> Result<(), FluxError> {
        let (positions, velocities) = self.projected_state(
            self.positions.clone(),
            self.velocities.clone(),
            self.constraint.as_ref(),
        )?;
        self.positions = positions;
        self.velocities = velocities;
        if self.constraint.is_some() {
            self.md_initialized = false;
            self.md_timestep_ps = None;
        }
        Ok(())
    }

    fn projected_state(
        &self,
        mut positions: Array1<f64>,
        mut velocities: Array1<f64>,
        constraint: Option<&HyperplaneConstraint>,
    ) -> Result<(Array1<f64>, Array1<f64>), FluxError> {
        if let Some(plane) = constraint {
            shake_positions_in_place(&mut positions, self.masses.view(), plane)?;
            let scale = 1.0 + plane.offset.abs() + rgmin::vecops::nrm2(positions.view());
            let residual = rgmin::vecops::dot(plane.normal.view(), positions.view()) - plane.offset;
            let relative_residual = residual.abs() / scale;
            if !relative_residual.is_finite() {
                return Err(FluxError::ConstraintViolation {
                    residual: relative_residual,
                    tolerance: crate::engine::HYPERPLANE_PROJECTION_TOL,
                });
            }
            rattle_velocities_in_place(&mut velocities, self.masses.view(), plane);
        }
        Ok((positions, velocities))
    }

    fn force_input_bytes(&self) -> Result<Vec<u8>, FluxError> {
        let request = rgpot_core::profile::ProfileRequest {
            positions: self.positions.as_slice().ok_or_else(|| {
                FluxError::Lammpc("positions storage is not contiguous".to_string())
            })?,
            atomic_numbers: &self.atomic_numbers,
            box_matrix: self.cell.as_ref(),
            length_unit: METAL_LENGTH_UNIT,
            energy_unit: METAL_ENERGY_UNIT,
        };
        rgpot_core::profile::encode_force_input(&request)
            .map_err(|err| FluxError::Lammpc(format!("encode ForceInput failed: {err}")))
    }

    fn call_calculate(&mut self) -> Result<(f64, Array1<f64>), FluxError> {
        let input = self.force_input_bytes()?;
        // SAFETY: size query takes the `ForceInput` bytes and a null output, as in
        // `rgpot_core::profile::ProfileSession::evaluate`.
        let required = unsafe {
            lammpc_potential_result_size_for_force_input(input.as_ptr().cast(), input.len())
        };
        if required == 0 || !required.is_multiple_of(CAPNP_WORD_BYTES) {
            return Err(FluxError::Lammpc(last_error_message()));
        }
        let mut output = vec![0_u64; required / CAPNP_WORD_BYTES];
        let capacity = output.len() * CAPNP_WORD_BYTES;
        let mut written = 0usize;
        // SAFETY: `session` is a live handle, the word buffer outlives the call, and
        // `capacity` is the byte length of that buffer.
        let status = unsafe {
            lammpc_session_calculate_result(
                self.session.as_ptr(),
                input.as_ptr().cast(),
                input.len(),
                output.as_mut_ptr().cast(),
                capacity,
                &mut written,
            )
        };
        if status.ok == 0 {
            let message = result_message(&status);
            return Err(FluxError::Lammpc(message));
        }
        if written == 0 || written > capacity || !written.is_multiple_of(CAPNP_WORD_BYTES) {
            return Err(FluxError::Lammpc(format!(
                "lammpc_session_calculate_result wrote invalid size {written} for capacity {capacity}"
            )));
        }
        // SAFETY: `written` bytes sit at the front of the word buffer just filled.
        let encoded = unsafe { std::slice::from_raw_parts(output.as_ptr().cast::<u8>(), written) };
        let decoded = rgpot_core::profile::decode_potential_result(encoded, self.dof())
            .map_err(|err| FluxError::Lammpc(format!("decode PotentialResult failed: {err}")))?;
        if !decoded.energy.is_finite() {
            return Err(FluxError::NonFinite("LammpcEngine potential energy"));
        }
        let forces = Array1::from_vec(decoded.forces);
        validate_finite_slice(forces.view(), "LammpcEngine forces")?;
        self.nve_instance_ready = true;
        Ok((decoded.energy, forces))
    }

    fn ensure_constrained_md_initialized(&mut self, dt: f64) -> Result<(), FluxError> {
        if self.md_initialized
            && self
                .md_timestep_ps
                .is_some_and(|stored| timesteps_match(stored, dt))
        {
            return Ok(());
        }
        let plane = self
            .constraint
            .as_ref()
            .ok_or(FluxError::BackendUnavailable(
                "lammpc constrained mode requires a plane",
            ))?;
        let thermal_energy = self.thermostat.thermal_energy()?;
        let md_thermal = if thermal_energy > MIN_POSITIVE_THERMAL_ENERGY_EV {
            thermal_energy
        } else {
            MIN_POSITIVE_THERMAL_ENERGY_EV
        };
        let friction = if self.thermostat.gamma > 0.0 {
            self.thermostat.gamma
        } else {
            // A positive floor keeps the OU factor at 1. Exact zero and this
            // floor produce identical step hashes; the floor also satisfies
            // builds that reject a zero friction control.
            MIN_POSITIVE_THERMAL_ENERGY_EV
        };
        let cfg = LammpcMdConfig {
            time_step_ps: dt,
            thermal_energy_ev: md_thermal,
            friction_per_ps: friction,
            seed: self.seed,
            coordinates: LammpcPlaneCoordinates::Cartesian,
        };
        let input = self.force_input_bytes()?;
        // The plane is `n · x = offset`. A point on it is `offset * n`, which
        // does not move when the current coordinates are shifted inside the plane.
        let point: Vec<f64> = plane
            .normal
            .iter()
            .map(|component| component * plane.offset)
            .collect();
        // SAFETY: All pointers target owned contiguous buffers with `dof` entries.
        let rc = unsafe {
            lammpc_session_md_initialize(
                self.session.as_ptr(),
                input.as_ptr().cast(),
                input.len(),
                point.as_ptr(),
                plane.normal.as_ptr(),
                self.dof(),
                &cfg as *const LammpcMdConfig,
            )
        };
        if rc != 0 {
            return Err(FluxError::Lammpc(last_error_message()));
        }
        // Initialization draws a fresh Maxwell sample. Put the caller velocities
        // back before any step, including the first step of a new session and
        // every later re-initialization.
        self.install_caller_velocities()?;
        self.md_initialized = true;
        self.md_timestep_ps = Some(dt);
        Ok(())
    }

    fn constrained_langevin_step(&mut self, dt: f64) -> Result<StepReport, FluxError> {
        if self.constraint.is_none() {
            return Err(FluxError::BackendUnavailable(
                "lammpc has no unconstrained Langevin ABI entry point",
            ));
        }
        self.ensure_constrained_md_initialized(dt)?;
        // SAFETY: live session and step_count=1 are valid.
        let rc_step = unsafe { lammpc_session_md_step(self.session.as_ptr(), 1) };
        if rc_step != 0 {
            self.md_initialized = false;
            self.md_timestep_ps = None;
            return Err(FluxError::Lammpc(last_error_message()));
        }

        // SAFETY: positions and velocities are sized to `dof`. Force and
        // observation outputs are optional and unused: the observation's
        // `constraint_force` is `n · F`, not the mass-weighted multiplier.
        let rc_state = unsafe {
            lammpc_session_md_state(
                self.session.as_ptr(),
                self.positions.as_mut_ptr(),
                self.velocities.as_mut_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                self.dof(),
                std::ptr::null_mut(),
            )
        };
        if rc_state != 0 {
            return Err(FluxError::Lammpc(last_error_message()));
        }
        validate_finite_slice(self.positions.view(), "LammpcEngine positions")?;
        validate_finite_slice(self.velocities.view(), "LammpcEngine velocities")?;
        // The MD state exposes the projected force and `n · F`, but ABI 0
        // exposes no constrained endpoint energy. The engine report requires
        // that energy, and the only energy entry point evaluates both energy
        // and physical force. Use that force for the mass-weighted multiplier.
        let (potential_energy, forces) = self.call_calculate()?;
        let plane = self
            .constraint
            .as_ref()
            .ok_or(FluxError::BackendUnavailable(
                "lammpc constrained mode requires a plane",
            ))?;
        let lambda = constraint_lagrange_multiplier(forces.view(), self.masses.view(), plane);
        if !lambda.is_finite() {
            return Err(FluxError::NonFinite("LammpcEngine constraint force"));
        }
        let kinetic_energy = self.kinetic_energy();
        if !kinetic_energy.is_finite() {
            return Err(FluxError::NonFinite("LammpcEngine kinetic energy"));
        }
        Ok(StepReport::new(
            potential_energy,
            kinetic_energy,
            Some(lambda),
        ))
    }

    fn install_caller_velocities(&mut self) -> Result<(), FluxError> {
        // SAFETY: the session was initialized above and the velocity slice has `dof` entries.
        let rc_set = unsafe {
            lammpc_session_md_set_velocities(
                self.session.as_ptr(),
                self.velocities.as_ptr(),
                self.dof(),
                self.seed,
            )
        };
        if rc_set != 0 {
            return Err(FluxError::Lammpc(last_error_message()));
        }
        Ok(())
    }

    fn ensure_nve_instance(&mut self) -> Result<(), FluxError> {
        if self.nve_instance_ready {
            return Ok(());
        }
        // The NVE entry points need a live geometry. One evaluation creates it;
        // later steps scatter coordinates themselves and must not evaluate again.
        let _ = self.call_calculate()?;
        Ok(())
    }

    fn unconstrained_nve_step(&mut self, dt: f64) -> Result<StepReport, FluxError> {
        if self.constraint.is_some() {
            return Err(FluxError::BackendUnavailable(
                "lammpc constrained NVE is unavailable through current C ABI",
            ));
        }
        self.ensure_nve_instance()?;
        let (cell_ptr, cell_len) = if let Some(cell) = self.cell.as_ref() {
            (cell.as_ptr(), 9usize)
        } else {
            (std::ptr::null(), 0usize)
        };
        // SAFETY: session handle and state slices are valid.
        let rc_pos = unsafe {
            lammpc_session_nve_set_positions(
                self.session.as_ptr(),
                self.positions.as_ptr(),
                self.dof(),
                cell_ptr,
                cell_len,
            )
        };
        if rc_pos != 0 {
            return Err(FluxError::Lammpc(last_error_message()));
        }
        // SAFETY: session handle and velocity slice are valid.
        let rc_vel = unsafe {
            lammpc_session_nve_set_velocities(
                self.session.as_ptr(),
                self.velocities.as_ptr(),
                self.dof(),
            )
        };
        if rc_vel != 0 {
            return Err(FluxError::Lammpc(last_error_message()));
        }
        // SAFETY: valid step count and finite positive dt checked by caller.
        let rc_step = unsafe { lammpc_session_nve_step(self.session.as_ptr(), 1, dt) };
        if rc_step != 0 {
            return Err(FluxError::Lammpc(last_error_message()));
        }
        let mut wrapped = vec![0.0_f64; self.dof()];
        let mut observation = LammpcNveObservation::default();
        // SAFETY: output buffers have `dof` coordinates.
        let rc_state = unsafe {
            lammpc_session_nve_state(
                self.session.as_ptr(),
                wrapped.as_mut_ptr(),
                self.velocities.as_mut_ptr(),
                self.dof(),
                &mut observation as *mut LammpcNveObservation,
            )
        };
        if rc_state != 0 {
            return Err(FluxError::Lammpc(last_error_message()));
        }
        validate_finite_slice(self.velocities.view(), "LammpcEngine velocities")?;
        self.restore_unwrapped_positions(&wrapped)?;
        if !observation.potential_energy_ev.is_finite() {
            return Err(FluxError::NonFinite("LammpcEngine potential energy"));
        }
        // LAMMPS thermo `ke` drops the center of mass and is zero for one atom.
        // The engine contract is `(1/2) sum m_i v_i^2` in metal energy units.
        let kinetic_energy = self.kinetic_energy();
        if !kinetic_energy.is_finite() {
            return Err(FluxError::NonFinite("LammpcEngine kinetic energy"));
        }
        Ok(StepReport::new(
            observation.potential_energy_ev,
            kinetic_energy,
            None,
        ))
    }

    fn restore_unwrapped_positions(&mut self, wrapped: &[f64]) -> Result<(), FluxError> {
        let mut unwrapped = self.positions.clone();
        if let Some(cell) = self.cell {
            let periodic = periodic_axes(&self.boundary);
            let lengths = [cell[0], cell[4], cell[8]];
            let xy = cell[3];
            let xz = cell[6];
            let yz = cell[7];
            let n_atoms = wrapped.len() / 3;
            for atom in 0..n_atoms {
                let base = atom * 3;
                let previous = [
                    self.positions[base],
                    self.positions[base + 1],
                    self.positions[base + 2],
                ];
                let current = [wrapped[base], wrapped[base + 1], wrapped[base + 2]];
                let restored =
                    unwrap_with_image_flags(previous, current, lengths, xy, xz, yz, periodic);
                unwrapped[base] = restored[0];
                unwrapped[base + 1] = restored[1];
                unwrapped[base + 2] = restored[2];
            }
        } else if let Some(slice) = unwrapped.as_slice_mut() {
            slice.copy_from_slice(wrapped);
        } else {
            return Err(FluxError::Lammpc(
                "positions storage is not contiguous".to_string(),
            ));
        }
        validate_finite_slice(unwrapped.view(), "LammpcEngine positions")?;
        self.positions = unwrapped;
        Ok(())
    }
}

impl Drop for LammpcEngine {
    fn drop(&mut self) {
        // SAFETY: `session` was created by lammpc and is owned by this engine.
        unsafe { lammpc_session_destroy(self.session.as_ptr()) };
    }
}

impl DynamicsEngine for LammpcEngine {
    fn dof(&self) -> usize {
        self.positions.len()
    }

    fn masses(&self) -> ArrayView1<'_, f64> {
        self.masses.view()
    }

    fn positions(&self) -> ArrayView1<'_, f64> {
        self.positions.view()
    }

    fn set_positions(&mut self, positions: ArrayView1<'_, f64>) -> Result<(), FluxError> {
        if positions.len() != self.dof() {
            return Err(FluxError::Shape {
                expected: self.dof(),
                got: positions.len(),
                context: "LammpcEngine positions",
            });
        }
        validate_finite_slice(positions, "LammpcEngine positions")?;
        let (positions, velocities) = self.projected_state(
            positions.to_owned(),
            self.velocities.clone(),
            self.constraint.as_ref(),
        )?;
        self.positions = positions;
        self.velocities = velocities;
        self.md_initialized = false;
        self.md_timestep_ps = None;
        Ok(())
    }

    fn velocities(&self) -> ArrayView1<'_, f64> {
        self.velocities.view()
    }

    fn set_velocities(&mut self, velocities: ArrayView1<'_, f64>) -> Result<(), FluxError> {
        if velocities.len() != self.dof() {
            return Err(FluxError::Shape {
                expected: self.dof(),
                got: velocities.len(),
                context: "LammpcEngine velocities",
            });
        }
        validate_finite_slice(velocities, "LammpcEngine velocities")?;
        self.velocities.assign(&velocities);
        if let Some(plane) = &self.constraint {
            rattle_velocities_in_place(&mut self.velocities, self.masses.view(), plane);
            self.md_initialized = false;
            self.md_timestep_ps = None;
        }
        Ok(())
    }

    fn cell(&self) -> Option<&[f64; 9]> {
        self.cell.as_ref()
    }

    fn set_cell(&mut self, cell: Option<[f64; 9]>) -> Result<(), FluxError> {
        if let Some(matrix) = &cell {
            for value in matrix {
                if !value.is_finite() {
                    return Err(FluxError::NonFinite("LammpcEngine cell"));
                }
            }
        }
        self.cell = cell;
        self.md_initialized = false;
        self.md_timestep_ps = None;
        Ok(())
    }

    fn thermostat(&self) -> &ThermostatConfig {
        &self.thermostat
    }

    fn set_thermostat(&mut self, thermostat: ThermostatConfig) -> Result<(), FluxError> {
        let _ = validate_lammpc_thermostat(&thermostat)?;
        self.thermostat = thermostat;
        self.md_initialized = false;
        self.md_timestep_ps = None;
        Ok(())
    }

    fn constraint(&self) -> Option<&HyperplaneConstraint> {
        self.constraint.as_ref()
    }

    fn set_constraint(
        &mut self,
        constraint: Option<HyperplaneConstraint>,
    ) -> Result<(), FluxError> {
        if let Some(plane) = &constraint {
            plane.validate()?;
            if plane.normal.len() != self.dof() {
                return Err(FluxError::Shape {
                    expected: self.dof(),
                    got: plane.normal.len(),
                    context: "LammpcEngine constraint normal",
                });
            }
        }
        let (positions, velocities) = self.projected_state(
            self.positions.clone(),
            self.velocities.clone(),
            constraint.as_ref(),
        )?;
        self.constraint = constraint;
        self.positions = positions;
        self.velocities = velocities;
        self.md_initialized = false;
        self.md_timestep_ps = None;
        Ok(())
    }

    fn energy_and_forces(&mut self) -> Result<(f64, Array1<f64>), FluxError> {
        self.call_calculate()
    }

    fn kinetic_energy(&self) -> f64 {
        let mut ke = 0.0_f64;
        for i in 0..self.velocities.len() {
            ke += 0.5 * self.masses[i] * self.velocities[i] * self.velocities[i];
        }
        ke * MASS_VELOCITY_TO_EV
    }

    fn step(&mut self, dt: f64) -> Result<StepReport, FluxError> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(FluxError::InvalidConfig(
                "LammpcEngine::step requires a finite positive timestep",
            ));
        }
        let _ = validate_lammpc_thermostat(&self.thermostat)?;
        let positions = self.positions.clone();
        let velocities = self.velocities.clone();
        let result = if self.constraint.is_some() {
            self.constrained_langevin_step(dt)
        } else if self.thermostat.gamma == 0.0 {
            self.unconstrained_nve_step(dt)
        } else {
            Err(FluxError::BackendUnavailable(
                "lammpc has no unconstrained Langevin ABI entry point",
            ))
        };
        if result.is_err() {
            self.positions = positions;
            self.velocities = velocities;
            self.md_initialized = false;
            self.md_timestep_ps = None;
            self.nve_instance_ready = false;
            if let Err(FluxError::Lammpc(original)) = &result
                && let Err(recovery) = self.recreate_session_after_error()
            {
                return Err(FluxError::Lammpc(format!(
                    "{original}; session recovery failed: {recovery}"
                )));
            }
        }
        result
    }

    fn reseed(&mut self, seed: u64) -> Result<(), FluxError> {
        self.seed = seed;
        self.md_initialized = false;
        self.md_timestep_ps = None;
        Ok(())
    }

    fn shoot_velocities(&mut self, seed: u64) -> Result<(), FluxError> {
        let kbt = self.thermostat.thermal_energy()?;
        let scaled_masses = self.masses.mapv(|m| m * MASS_VELOCITY_TO_EV);
        let mut rng = StdRng::seed_from_u64(seed);
        self.velocities = sample_maxwell_boltzmann_with_rng(scaled_masses.view(), kbt, &mut rng);
        if let Some(plane) = &self.constraint {
            rattle_velocities_in_place(&mut self.velocities, self.masses.view(), plane);
        }
        self.reseed(seed)
    }

    fn reflect_velocity(&mut self, normal: ArrayView1<'_, f64>) -> Result<f64, FluxError> {
        let normal_velocity =
            reflect_velocity_in_place(&mut self.velocities, self.masses.view(), normal)?;
        self.md_initialized = false;
        self.md_timestep_ps = None;
        Ok(normal_velocity)
    }

    fn apply_dividing_surface(
        &mut self,
        normal: ArrayView1<'_, f64>,
        policy: SurfaceVelocityPolicy,
    ) -> Result<f64, FluxError> {
        validate_unit_normal(normal, self.dof())?;
        let current_nv = rgmin::vecops::dot(normal, self.velocities.view());
        if current_nv > 0.0 {
            return Ok(current_nv);
        }
        match policy {
            SurfaceVelocityPolicy::Reflect => {
                if current_nv < 0.0 {
                    self.reflect_velocity(normal)
                } else {
                    Err(FluxError::InvalidConfig(
                        "zero normal velocity cannot be oriented positive by reflection",
                    ))
                }
            }
            SurfaceVelocityPolicy::RejectRedraw { seed, max_attempts } => {
                for attempt in 0..max_attempts {
                    self.shoot_velocities(seed.wrapping_add(attempt as u64))?;
                    let nv = rgmin::vecops::dot(normal, self.velocities.view());
                    if nv > 0.0 {
                        return Ok(nv);
                    }
                }
                Err(FluxError::SurfaceRejection { max_attempts })
            }
        }
    }
}

fn validate_lammpc_thermostat(thermostat: &ThermostatConfig) -> Result<f64, FluxError> {
    let kbt = thermostat.thermal_energy()?;
    if thermostat.noise != crate::engine::gle::ColoredNoise::White {
        return Err(FluxError::InvalidConfig(
            "LammpcEngine does not support colored-noise thermostats",
        ));
    }
    Ok(kbt)
}

fn type_index_for_atomic_number(z: i32, type_to_atomic_number: &[i32]) -> Option<usize> {
    type_to_atomic_number
        .iter()
        .position(|candidate| *candidate == z)
}

fn expand_coordinate_masses(
    atomic_numbers: &[i32],
    config: &LammpcConfig,
) -> Result<Array1<f64>, FluxError> {
    let mut out = Vec::with_capacity(atomic_numbers.len() * 3);
    for &z in atomic_numbers {
        let type_index = type_index_for_atomic_number(z, &config.type_to_atomic_number).ok_or(
            FluxError::InvalidConfig("atomic number missing from type_to_atomic_number"),
        )?;
        let mass = config.masses_by_type[type_index];
        if !mass.is_finite() || mass <= 0.0 {
            return Err(FluxError::InvalidConfig(
                "masses_by_type must be finite and strictly positive",
            ));
        }
        out.extend_from_slice(&[mass, mass, mass]);
    }
    Ok(Array1::from_vec(out))
}

fn encode_lammps_config(config: &LammpcConfig) -> Result<Vec<u8>, FluxError> {
    let mut message = Builder::new_default();
    {
        let mut root =
            message.init_root::<rgpot_core::Potentials_capnp::potential_config::Builder<'_>>();
        let mut lammps = root.reborrow().init_lammps();
        lammps.set_units_style("metal");
        lammps.set_atom_style("atomic");
        lammps.set_pair_style(&config.pair_style);
        lammps.set_boundary(&config.boundary);
        lammps.set_suffix(&config.suffix);

        let coeff_len = u32::try_from(config.pair_coeffs.len())
            .map_err(|_| FluxError::InvalidConfig("pair_coeffs length exceeds u32"))?;
        let mut coeffs = lammps.reborrow().init_pair_coeffs(coeff_len);
        for (i, value) in config.pair_coeffs.iter().enumerate() {
            coeffs.set(i as u32, value);
        }

        let types_len = u32::try_from(config.type_to_atomic_number.len())
            .map_err(|_| FluxError::InvalidConfig("type_to_atomic_number length exceeds u32"))?;
        let mut type_map = lammps.reborrow().init_type_to_atomic_number(types_len);
        for (i, value) in config.type_to_atomic_number.iter().copied().enumerate() {
            type_map.set(i as u32, value);
        }

        let masses_len = u32::try_from(config.masses_by_type.len())
            .map_err(|_| FluxError::InvalidConfig("masses_by_type length exceeds u32"))?;
        let mut masses = lammps.reborrow().init_masses(masses_len);
        for (i, value) in config.masses_by_type.iter().copied().enumerate() {
            masses.set(i as u32, value);
        }

        let sent_setup: Vec<&str> = config
            .extra_setup
            .iter()
            .map(String::as_str)
            .filter(|line| !setup_line_is_spaces_and_tabs(line))
            .collect();
        let extra_len = u32::try_from(sent_setup.len())
            .map_err(|_| FluxError::InvalidConfig("extra_setup length exceeds u32"))?;
        let mut extra = lammps.reborrow().init_extra_setup(extra_len);
        for (i, value) in sent_setup.iter().enumerate() {
            extra.set(i as u32, value);
        }
    }
    Ok(serialize::write_message_to_words(&message))
}

fn timesteps_match(stored: f64, requested: f64) -> bool {
    let scale = stored.abs().max(requested.abs());
    (stored - requested).abs() <= TIMESTEP_RELATIVE_TOL * scale
}

fn validate_lammps_command_text(value: &str, field: &'static str) -> Result<(), FluxError> {
    // Reject non-ASCII before a later substitution can rewrite characters.
    if !value.is_ascii()
        || value
            .chars()
            .any(|character| matches!(character, '\0' | '\n' | '\r' | ';'))
    {
        return Err(FluxError::InvalidLammpcCommand { field });
    }
    Ok(())
}

const EXTRA_SETUP_FIELD: &str = "LammpcConfig::extra_setup";

/// The first word of an `extra_setup` line may be one of these. Each
/// one tunes how LAMMPS evaluates the configured pair style on the live
/// system and takes only numbers and keywords: none reads a file,
/// spawns a process, or branches.
///
/// - `neighbor`: skin distance and list style, the usual tuning knob for
///   the engine's short restart-free runs.
/// - `neigh_modify`: rebuild frequency and delay, the companion of
///   `neighbor` when forces are re-evaluated after every position update.
/// - `comm_modify`: ghost cutoff, needed when a pair cutoff plus skin
///   exceeds the default communication cutoff in small periodic cells.
/// - `pair_modify`: `shift`, `mix` and tail correction for the configured
///   `pair_style`, which `pair_style` and `pair_coeff` cannot express.
const ALLOWED_EXTRA_SETUP_DIRECTIVES: [&str; 4] =
    ["neighbor", "neigh_modify", "comm_modify", "pair_modify"];

/// LAMMPS commands that run a process, read another script, branch, loop or
/// run Python. A line mentioning any of them as a word is rejected even when
/// the first token is allowed. `include` is allowed only as the
/// `neigh_modify` keyword immediately followed by a group id.
const FORBIDDEN_EXTRA_SETUP_WORDS: [&str; 8] = [
    "if",
    "partition",
    "shell",
    "include",
    "jump",
    "python",
    "label",
    "next",
];

struct SetupToken<'a> {
    text: &'a str,
    start: usize,
    end: usize,
}

fn whitespace_setup_tokens(command: &str) -> Vec<SetupToken<'_>> {
    let bytes = command.as_bytes();
    let mut tokens = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b' ' || bytes[index] == b'\t' {
            index += 1;
            continue;
        }
        let start = index;
        while index < bytes.len() && bytes[index] != b' ' && bytes[index] != b'\t' {
            index += 1;
        }
        tokens.push(SetupToken {
            text: &command[start..index],
            start,
            end: index,
        });
    }
    tokens
}

fn setup_line_is_spaces_and_tabs(line: &str) -> bool {
    line.bytes().all(|byte| matches!(byte, b' ' | b'\t'))
}

fn next_after_exclude(tokens: &[SetupToken<'_>], index: usize) -> usize {
    let Some(kind) = tokens.get(index + 1).map(|token| token.text) else {
        return tokens.len();
    };
    let consumed = match kind {
        "none" => 2,
        "group" | "type" => 4,
        "molecule/intra" | "molecule/inter" => 3,
        _ => return tokens.len(),
    };
    if index + consumed > tokens.len() {
        tokens.len()
    } else {
        index + consumed
    }
}

fn next_after_collection(tokens: &[SetupToken<'_>], index: usize) -> usize {
    let Some(count) = tokens
        .get(index + 1)
        .and_then(|token| token.text.parse::<usize>().ok())
    else {
        return tokens.len();
    };
    let next = index.saturating_add(2).saturating_add(count);
    if next > tokens.len() {
        tokens.len()
    } else {
        next
    }
}

/// Byte ranges of `include` keywords that are followed by a group id.
///
/// `neigh_modify every include` puts `include` in the value position of
/// `every`, so that word stays forbidden. `neigh_modify include all` and
/// `neigh_modify every 1 include solvent` use `include` as the keyword.
fn neigh_modify_include_keyword_ranges(command: &str) -> Vec<std::ops::Range<usize>> {
    let tokens = whitespace_setup_tokens(command);
    let mut ranges = Vec::new();
    if tokens.first().map(|token| token.text) != Some("neigh_modify") {
        return ranges;
    }
    let mut index = 1;
    while index < tokens.len() {
        let keyword = tokens[index].text;
        if keyword == "include" {
            if index + 1 < tokens.len() {
                ranges.push(tokens[index].start..tokens[index].end);
            }
            index += 2;
            continue;
        }
        if matches!(
            keyword,
            "delay"
                | "every"
                | "check"
                | "once"
                | "cluster"
                | "page"
                | "one"
                | "binsize"
                | "bin/hash"
        ) {
            index += 2;
            continue;
        }
        if keyword == "exclude" {
            index = next_after_exclude(&tokens, index);
            continue;
        }
        if keyword == "collection/type" || keyword == "collection/interval" {
            index = next_after_collection(&tokens, index);
            continue;
        }
        index += 1;
    }
    ranges
}

fn validate_extra_setup_directive(command: &str) -> Result<(), FluxError> {
    // Reject every non-ASCII byte first, before a later substitution can
    // rewrite characters in the line.
    if !command.is_ascii() {
        return Err(FluxError::InvalidLammpcCommand {
            field: EXTRA_SETUP_FIELD,
        });
    }
    // Split on every non-word byte so `shell#touch`, `"shell` and `x,shell`
    // all expose the `shell` word. `include` is exempt only as the
    // `neigh_modify` keyword immediately followed by a group id.
    let exempt_includes = neigh_modify_include_keyword_ranges(command);
    let mut word_start = None;
    for (offset, character) in command.char_indices() {
        let is_word = character.is_ascii_alphanumeric() || character == '_';
        if is_word {
            if word_start.is_none() {
                word_start = Some(offset);
            }
            continue;
        }
        if let Some(start) = word_start.take() {
            reject_forbidden_setup_word(command, start, offset, &exempt_includes)?;
        }
    }
    if let Some(start) = word_start.take() {
        reject_forbidden_setup_word(command, start, command.len(), &exempt_includes)?;
    }
    // No allowed directive needs quoting, variable substitution, line
    // continuation, comments, or control bytes other than tab.
    if command
        .chars()
        .any(|c| matches!(c, '"' | '\'' | '$' | '&' | '#') || (c.is_ascii_control() && c != '\t'))
    {
        return Err(FluxError::InvalidLammpcCommand {
            field: EXTRA_SETUP_FIELD,
        });
    }
    let Some(first_token) = command.split_ascii_whitespace().next() else {
        return Ok(());
    };
    if !ALLOWED_EXTRA_SETUP_DIRECTIVES.contains(&first_token) {
        return Err(FluxError::InvalidLammpcCommand {
            field: EXTRA_SETUP_FIELD,
        });
    }
    Ok(())
}

fn reject_forbidden_setup_word(
    command: &str,
    start: usize,
    end: usize,
    exempt_includes: &[std::ops::Range<usize>],
) -> Result<(), FluxError> {
    let word = &command[start..end];
    if let Some(directive) = FORBIDDEN_EXTRA_SETUP_WORDS
        .iter()
        .copied()
        .find(|forbidden| word.eq_ignore_ascii_case(forbidden))
    {
        if directive == "include"
            && exempt_includes
                .iter()
                .any(|range| range.start == start && range.end == end)
        {
            return Ok(());
        }
        return Err(FluxError::ForbiddenLammpcDirective {
            field: EXTRA_SETUP_FIELD,
            directive,
        });
    }
    Ok(())
}

fn result_message(result: &LammpcResult) -> String {
    let mut bytes = Vec::new();
    for &entry in &result.message {
        let byte = entry as u8;
        if byte == 0 {
            break;
        }
        bytes.push(byte);
    }
    if bytes.is_empty() {
        last_error_message()
    } else {
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

fn handshake_capabilities() -> Result<(), FluxError> {
    let mut required = 0usize;
    // SAFETY: a null buffer with zero capacity is the documented size query.
    let _ = unsafe { lammpc_capabilities_result(std::ptr::null_mut(), 0, &mut required) };
    if required == 0 || !required.is_multiple_of(CAPNP_WORD_BYTES) {
        return Err(FluxError::Lammpc(
            "lammpc_capabilities_result reported an invalid size".to_string(),
        ));
    }
    let mut words = vec![0_u64; required / CAPNP_WORD_BYTES];
    let mut written = 0usize;
    // SAFETY: `words` holds `required` bytes and outlives the call.
    let status =
        unsafe { lammpc_capabilities_result(words.as_mut_ptr().cast(), required, &mut written) };
    if status != 0 || written != required {
        return Err(FluxError::Lammpc(format!(
            "lammpc_capabilities_result failed (status {status}, wrote {written} of {required})"
        )));
    }
    rgpot_core::profile::check_capabilities_bytes(
        "lammpc",
        &words,
        &rgpot_core::compat::Expectation::default(),
    )
    .map_err(|err| FluxError::Lammpc(err.to_string()))?;
    Ok(())
}

fn periodic_axes(boundary: &str) -> [bool; 3] {
    let mut axes = [false; 3];
    for (index, token) in boundary.split_whitespace().take(3).enumerate() {
        let periodic = token.starts_with('p') || token.starts_with('P');
        axes[index] = periodic;
    }
    axes
}

/// Recover unwrapped coordinates from wrapped `x` and the LAMMPS image count.
///
/// The image count on each periodic axis is the integer that places the
/// wrapped coordinate on the continuous branch through `previous`. Tilt
/// factors follow LAMMPS `Domain::unmap` for a restricted triclinic cell.
fn unwrap_with_image_flags(
    previous: [f64; 3],
    wrapped: [f64; 3],
    lengths: [f64; 3],
    xy: f64,
    xz: f64,
    yz: f64,
    periodic: [bool; 3],
) -> [f64; 3] {
    let usable = |axis: usize| periodic[axis] && lengths[axis].is_finite() && lengths[axis] > 0.0;
    let iz = if usable(2) {
        ((previous[2] - wrapped[2]) / lengths[2]).round()
    } else {
        0.0
    };
    let iy = if usable(1) {
        ((previous[1] - wrapped[1] - iz * yz) / lengths[1]).round()
    } else {
        0.0
    };
    let ix = if usable(0) {
        ((previous[0] - wrapped[0] - iy * xy - iz * xz) / lengths[0]).round()
    } else {
        0.0
    };
    [
        wrapped[0] + ix * lengths[0] + iy * xy + iz * xz,
        wrapped[1] + iy * lengths[1] + iz * yz,
        wrapped[2] + iz * lengths[2],
    ]
}

fn last_error_message() -> String {
    // SAFETY: `lammpc_last_error` returns a NUL-terminated thread-local message.
    let pointer = unsafe { lammpc_last_error() };
    if pointer.is_null() {
        return "lammpc returned a null last_error pointer".to_string();
    }
    // SAFETY: pointer comes from lammpc and is valid for immediate read.
    let text = unsafe { CStr::from_ptr(pointer) }
        .to_string_lossy()
        .into_owned();
    if text.is_empty() {
        "lammpc returned an empty error message".to_string()
    } else {
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::gle::ColoredNoise;

    #[test]
    fn zero_friction_white_noise_is_nve_and_not_canonical() {
        let thermostat = ThermostatConfig::default()
            .with_gamma(0.0)
            .with_noise(ColoredNoise::White);

        assert_eq!(validate_lammpc_thermostat(&thermostat), Ok(1.0));
        assert!(thermostat.require_canonical_sampling().is_err());
    }

    #[test]
    fn zero_friction_colored_noise_is_rejected() {
        let thermostat = ThermostatConfig::default()
            .with_gamma(0.0)
            .with_noise(ColoredNoise::SmartSampling);

        assert!(thermostat.require_canonical_sampling().is_ok());
        let error = match LammpcEngine::new(LammpcConfig::default().with_thermostat(thermostat)) {
            Err(error) => error,
            Ok(_) => panic!("unsupported noise was accepted"),
        };
        assert!(matches!(
            error,
            FluxError::InvalidConfig(message) if message.contains("colored-noise")
        ));
    }

    fn forbidden(directive: &'static str) -> FluxError {
        FluxError::ForbiddenLammpcDirective {
            field: EXTRA_SETUP_FIELD,
            directive,
        }
    }

    fn invalid() -> FluxError {
        FluxError::InvalidLammpcCommand {
            field: EXTRA_SETUP_FIELD,
        }
    }

    #[test]
    fn extra_setup_allows_only_listed_directives() {
        for command in [
            "neighbor 2.0 bin",
            "neigh_modify every 1 delay 0 check yes",
            "neigh_modify include all",
            "neigh_modify include setup.in",
            "neigh_modify every 1 delay 0 include solvent",
            "neigh_modify delay 0 include all",
            "neigh_modify exclude none include solvent",
            "neigh_modify\tinclude solvent",
            "",
            "   ",
            "\t",
            " \t ",
            "comm_modify cutoff 12.0",
            "pair_modify shift yes",
        ] {
            assert_eq!(validate_extra_setup_directive(command), Ok(()), "{command}");
        }
        for command in ["units real", "clear", "NEIGHBOR 2.0 bin", "log out.txt"] {
            assert_eq!(
                validate_extra_setup_directive(command),
                Err(invalid()),
                "{command}"
            );
        }
    }

    #[test]
    fn extra_setup_rejects_forbidden_words_anywhere_in_the_line() {
        for (command, directive) in [
            ("shell touch X", "shell"),
            ("SHELL touch X", "shell"),
            ("include setup.in", "include"),
            ("INCLUDE setup.in", "include"),
            ("NEIGH_MODIFY include all", "include"),
            ("neigh_modify every include", "include"),
            ("neigh_modify every include solvent", "include"),
            ("neigh_modify include", "include"),
            ("neighbor 2.0 bin include all", "include"),
            ("pair_modify include all", "include"),
            ("comm_modify include all", "include"),
            ("neigh_modify include shell", "shell"),
            ("if 1 then \"shell touch X\"", "if"),
            ("partition yes 1 shell touch X", "partition"),
            ("neighbor 2.0 bin shell touch X", "shell"),
            ("neighbor 2.0 bin jump setup.in", "jump"),
            ("python f input 1 x", "python"),
            ("label a", "label"),
            ("next i", "next"),
            ("shell#touch X", "shell"),
            ("neighbor 2.0 bin #shell", "shell"),
            ("shell\u{c}touch X", "shell"),
        ] {
            assert_eq!(
                validate_extra_setup_directive(command),
                Err(forbidden(directive)),
                "{command:?}"
            );
        }
    }

    #[test]
    fn extra_setup_rejects_non_ascii_text() {
        for separator in [
            '\u{200b}', '\u{2060}', '\u{2063}', '\u{feff}', '\u{a0}', '\u{e9}',
        ] {
            for command in [
                format!("shell{separator}touch X"),
                format!("{separator}shell touch X"),
                format!("neighbor{separator}2.0 bin"),
                format!("neighbor 2.0 bin{separator}"),
            ] {
                assert_eq!(
                    validate_extra_setup_directive(&command),
                    Err(invalid()),
                    "{command:?}"
                );
            }
        }
    }

    #[test]
    fn extra_setup_rejects_quotes_substitution_continuation_and_comments() {
        for command in [
            "neighbor 2.0 \"bin\"",
            "neighbor 2.0 'bin'",
            "neighbor ${skin} bin",
            "neighbor $s bin",
            "neighbor 2.0 bin &",
            "neighbor 2.0 bin # trailing",
            "neighbor 2.0\u{1}bin",
            "neighbor 2.0\u{7f}bin",
            "\n",
            "\r",
            "\r\n",
            "\u{b}",
            "\u{c}",
            "\u{b} \u{c}",
        ] {
            assert_eq!(
                validate_extra_setup_directive(command),
                Err(invalid()),
                "{command:?}"
            );
        }
    }

    #[test]
    fn blank_extra_setup_lines_are_omitted_from_the_encoded_message() {
        let config = LammpcConfig::default().with_extra_setup(vec![
            String::new(),
            " ".to_string(),
            "\t".to_string(),
            " \t ".to_string(),
            "neighbor 2.0 bin".to_string(),
            "\t\t".to_string(),
        ]);
        let bytes = encode_lammps_config(&config).expect("encode");
        let mut slice = bytes.as_slice();
        let message = serialize::read_message_from_flat_slice(&mut slice, Default::default())
            .expect("decode");
        let root = message
            .get_root::<rgpot_core::Potentials_capnp::potential_config::Reader<'_>>()
            .expect("root");
        let rgpot_core::Potentials_capnp::potential_config::Lammps(params) =
            root.which().expect("schema")
        else {
            panic!("expected the lammps potential");
        };
        let extra = params
            .expect("lammps")
            .get_extra_setup()
            .expect("extra_setup");
        assert_eq!(extra.len(), 1);
        assert_eq!(
            extra.get(0).expect("line").to_str().expect("utf8"),
            "neighbor 2.0 bin"
        );
    }

    #[test]
    fn timestep_matching_scales_with_the_requested_magnitude() {
        let stored = 2.0_f64;
        let adjacent = f64::from_bits(stored.to_bits() + 1);

        assert!((stored - adjacent).abs() > f64::EPSILON);
        assert!(timesteps_match(stored, adjacent));
    }

    #[test]
    fn failed_constrained_step_invalidates_md_and_preserves_rust_state() {
        let plane = HyperplaneConstraint::from_direction(
            Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).view(),
            0.0,
        )
        .expect("plane");
        let mut engine =
            LammpcEngine::new(LammpcConfig::default().with_constraint(plane)).expect("engine");
        let positions = engine.positions.clone();
        let velocities = engine.velocities.clone();
        engine.md_initialized = true;
        engine.md_timestep_ps = Some(0.001);

        let error = engine.step(0.001).expect_err("session was not initialized");

        assert!(matches!(error, FluxError::Lammpc(_)));
        assert_eq!(engine.positions, positions);
        assert_eq!(engine.velocities, velocities);
        assert!(!engine.md_initialized);
        assert_eq!(engine.md_timestep_ps, None);

        engine.positions = Array1::from_vec(vec![0.0, 0.0, 0.0, 4.2, 0.0, 0.0]);
        let report = engine
            .step(0.001)
            .expect("session recovery should reinitialize");
        assert!(report.potential_energy.is_finite());
        assert!(report.kinetic_energy.is_finite());
        assert!(report.constraint_force.is_some());
    }

    #[test]
    fn failed_constrained_step_recreates_the_lammpc_session() {
        let plane = HyperplaneConstraint::from_direction(
            Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).view(),
            0.0,
        )
        .expect("plane");
        let mut engine =
            LammpcEngine::new(LammpcConfig::default().with_constraint(plane)).expect("engine");
        let before = engine.session.as_ptr();
        engine.md_initialized = true;
        engine.md_timestep_ps = Some(0.001);

        let _ = engine.step(0.001).expect_err("session was not initialized");

        let after = engine.session.as_ptr();
        assert_ne!(before, after);
    }

    #[test]
    fn command_text_rejects_separators_and_non_ascii() {
        for value in [
            "lj/cut 8.5; clear",
            "p p p\nclear",
            "omp\rclear",
            "lj/cut 8.5\0x",
            "1 1 0.01\u{2212}1",
        ] {
            assert_eq!(
                validate_lammps_command_text(value, "LammpcConfig::pair_style"),
                Err(FluxError::InvalidLammpcCommand {
                    field: "LammpcConfig::pair_style",
                }),
                "{value:?}"
            );
        }
        assert!(validate_lammps_command_text("lj/cut 8.5", "LammpcConfig::pair_style").is_ok());
    }
}
