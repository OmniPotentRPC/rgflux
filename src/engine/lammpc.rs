//! Direct lammpc-backed [`DynamicsEngine`] implementation.
//!
//! This backend links `liblammpc` and drives its persistent `lammpc_session_*`
//! C ABI with `ForceInput` and `PotentialConfig` Cap'n Proto messages from
//! `rgpot_core`.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::ptr::NonNull;

use capnp::message::Builder;
use capnp::serialize;
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::engine::{
    DynamicsEngine, HyperplaneConstraint, StepReport, SurfaceVelocityPolicy, ThermostatConfig,
    reflect_velocity_in_place, sample_maxwell_boltzmann_with_rng, validate_finite_slice,
    validate_masses, validate_unit_normal,
};
use crate::error::FluxError;

const METAL_LENGTH_UNIT: &str = "angstrom";
const METAL_ENERGY_UNIT: &str = "eV";
const MIN_POSITIVE_THERMAL_ENERGY_EV: f64 = f64::MIN_POSITIVE;
const MASS_VELOCITY_TO_EV: f64 = 0.000_103_642_696_562_621_75;

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
struct LammpcMdObservation {
    constraint_force: f64,
    force_evaluations: usize,
    steps: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
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
}

/// Public configuration for [`LammpcEngine`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct LammpcConfig {
    /// Atomic numbers in live atom order (`n_atoms` entries).
    pub atomic_numbers: Vec<i32>,
    /// LAMMPS type map (`type i+1 -> Z`) used to build `LammpsParams`.
    pub type_to_atomic_number: Vec<i32>,
    /// Per-type masses in amu (`type_to_atomic_number` order).
    pub masses_by_type: Vec<f64>,
    /// `pair_style` argument string.
    pub pair_style: String,
    /// `pair_coeff` lines, one entry per command.
    pub pair_coeffs: Vec<String>,
    /// Optional extra setup lines applied after pair setup.
    pub extra_setup: Vec<String>,
    /// LAMMPS boundary argument string (`"p p p"` by default).
    pub boundary: String,
    /// Optional LAMMPS suffix (`omp`, `gpu`, `kk`).
    pub suffix: String,
    /// Thermostat controls in metal units (`gamma` in `1/ps`, `k_B T` in `eV`).
    pub thermostat: ThermostatConfig,
    /// Optional hyperplane constraint.
    pub constraint: Option<HyperplaneConstraint>,
    /// Optional row-major 3x3 cell in Angstrom.
    pub cell: Option<[f64; 9]>,
    /// Initial deterministic seed consumed by `lammpc_session_md_initialize`.
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
    positions: Array1<f64>,
    velocities: Array1<f64>,
    masses: Array1<f64>,
    atomic_numbers: Vec<i32>,
    thermostat: ThermostatConfig,
    constraint: Option<HyperplaneConstraint>,
    cell: Option<[f64; 9]>,
    seed: u64,
    md_initialized: bool,
    md_timestep_ps: Option<f64>,
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
        let _ = config.thermostat.thermal_energy()?;

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
        let cfg_bytes = encode_lammps_config(&config)?;
        // SAFETY: We pass a stable byte slice and validate non-null result.
        let raw = unsafe {
            lammpc_session_create_from_config(cfg_bytes.as_ptr().cast(), cfg_bytes.len())
        };
        let session = NonNull::new(raw).ok_or_else(|| FluxError::Lammpc(last_error_message()))?;

        let mut engine = Self {
            session,
            positions: Array1::zeros(dof),
            velocities: Array1::zeros(dof),
            masses,
            atomic_numbers: config.atomic_numbers,
            thermostat: config.thermostat,
            constraint: config.constraint,
            cell: config.cell,
            seed: config.seed,
            md_initialized: false,
            md_timestep_ps: None,
        };
        engine.project_state_onto_constraint()?;
        Ok(engine)
    }

    fn project_state_onto_constraint(&mut self) -> Result<(), FluxError> {
        if let Some(plane) = &self.constraint {
            project_positions_to_plane(&mut self.positions, self.masses.view(), plane)?;
            project_velocities_to_plane(&mut self.velocities, self.masses.view(), plane);
            self.md_initialized = false;
            self.md_timestep_ps = None;
        }
        Ok(())
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
        let mut written = 0usize;
        // SAFETY: We pass a valid `ForceInput` byte slice and a null output buffer for the
        // mandatory size-query path described by the ABI.
        let required = unsafe {
            lammpc_potential_result_size_for_force_input(input.as_ptr().cast(), input.len())
        };
        if required == 0 {
            return Err(FluxError::Lammpc(last_error_message()));
        }
        let mut payload = vec![0_u8; required];
        // SAFETY: `session` is a live handle, buffers outlive the call, and sizes match.
        let status = unsafe {
            lammpc_session_calculate_result(
                self.session.as_ptr(),
                input.as_ptr().cast(),
                input.len(),
                payload.as_mut_ptr().cast(),
                payload.len(),
                &mut written as *mut usize,
            )
        };
        if status.ok == 0 {
            let message = result_message(&status);
            return Err(FluxError::Lammpc(message));
        }
        if written == 0 || written > payload.len() {
            return Err(FluxError::Lammpc(
                "lammpc returned an invalid PotentialResult size".to_string(),
            ));
        }
        let decoded = rgpot_core::profile::decode_potential_result(&payload[..written], self.dof())
            .map_err(|err| FluxError::Lammpc(format!("decode PotentialResult failed: {err}")))?;
        Ok((decoded.energy, Array1::from_vec(decoded.forces)))
    }

    fn ensure_constrained_md_initialized(&mut self, dt: f64) -> Result<(), FluxError> {
        if self.md_initialized
            && self
                .md_timestep_ps
                .is_some_and(|stored| (stored - dt).abs() <= f64::EPSILON)
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
        let cfg = LammpcMdConfig {
            time_step_ps: dt,
            thermal_energy_ev: md_thermal,
            friction_per_ps: self.thermostat.gamma.max(MIN_POSITIVE_THERMAL_ENERGY_EV),
            seed: self.seed,
            coordinates: LammpcPlaneCoordinates::Cartesian,
        };
        let input = self.force_input_bytes()?;
        let point = self.positions.to_vec();
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
            return Err(FluxError::Lammpc(last_error_message()));
        }

        let mut forces = vec![0.0_f64; self.dof()];
        let mut removed = vec![0.0_f64; self.dof()];
        let mut observation = LammpcMdObservation::default();
        // SAFETY: all output buffers are sized to `dof`.
        let rc_state = unsafe {
            lammpc_session_md_state(
                self.session.as_ptr(),
                self.positions.as_mut_ptr(),
                self.velocities.as_mut_ptr(),
                forces.as_mut_ptr(),
                removed.as_mut_ptr(),
                self.dof(),
                &mut observation as *mut LammpcMdObservation,
            )
        };
        if rc_state != 0 {
            return Err(FluxError::Lammpc(last_error_message()));
        }
        validate_finite_slice(self.positions.view(), "LammpcEngine positions")?;
        validate_finite_slice(self.velocities.view(), "LammpcEngine velocities")?;
        let (potential_energy, _) = self.call_calculate()?;
        let kinetic_energy = self.kinetic_energy();
        Ok(StepReport::new(
            potential_energy,
            kinetic_energy,
            Some(observation.constraint_force),
        ))
    }

    fn unconstrained_nve_step(&mut self, dt: f64) -> Result<StepReport, FluxError> {
        if self.constraint.is_some() {
            return Err(FluxError::BackendUnavailable(
                "lammpc constrained NVE is unavailable through current C ABI",
            ));
        }
        let _ = self.call_calculate()?;
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
        let mut observation = LammpcNveObservation::default();
        // SAFETY: output buffers have `dof` coordinates.
        let rc_state = unsafe {
            lammpc_session_nve_state(
                self.session.as_ptr(),
                self.positions.as_mut_ptr(),
                self.velocities.as_mut_ptr(),
                self.dof(),
                &mut observation as *mut LammpcNveObservation,
            )
        };
        if rc_state != 0 {
            return Err(FluxError::Lammpc(last_error_message()));
        }
        Ok(StepReport::new(
            observation.potential_energy_ev,
            observation.kinetic_energy_ev,
            None,
        ))
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
        self.positions.assign(&positions);
        self.project_state_onto_constraint()?;
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
            project_velocities_to_plane(&mut self.velocities, self.masses.view(), plane);
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
        let _ = thermostat.thermal_energy()?;
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
        self.constraint = constraint;
        self.project_state_onto_constraint()?;
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
        if self.thermostat.gamma > 0.0 {
            self.constrained_langevin_step(dt)
        } else {
            self.unconstrained_nve_step(dt)
        }
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
            project_velocities_to_plane(&mut self.velocities, self.masses.view(), plane);
        }
        self.reseed(seed)
    }

    fn reflect_velocity(&mut self, normal: ArrayView1<'_, f64>) -> Result<f64, FluxError> {
        reflect_velocity_in_place(&mut self.velocities, self.masses.view(), normal)
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

        let extra_len = u32::try_from(config.extra_setup.len())
            .map_err(|_| FluxError::InvalidConfig("extra_setup length exceeds u32"))?;
        let mut extra = lammps.reborrow().init_extra_setup(extra_len);
        for (i, value) in config.extra_setup.iter().enumerate() {
            extra.set(i as u32, value);
        }
    }
    Ok(serialize::write_message_to_words(&message))
}

fn project_positions_to_plane(
    positions: &mut Array1<f64>,
    masses: ArrayView1<'_, f64>,
    plane: &HyperplaneConstraint,
) -> Result<(), FluxError> {
    let z = crate::engine::inv_mass_metric_norm(plane.normal.view(), masses);
    for _ in 0..2 {
        let sigma = rgmin::vecops::dot(plane.normal.view(), positions.view()) - plane.offset;
        let alpha = sigma / z;
        for i in 0..positions.len() {
            positions[i] -= alpha * plane.normal[i] / masses[i];
        }
    }
    Ok(())
}

fn project_velocities_to_plane(
    velocities: &mut Array1<f64>,
    masses: ArrayView1<'_, f64>,
    plane: &HyperplaneConstraint,
) {
    let z = crate::engine::inv_mass_metric_norm(plane.normal.view(), masses);
    for _ in 0..2 {
        let nv = rgmin::vecops::dot(plane.normal.view(), velocities.view());
        let beta = nv / z;
        for i in 0..velocities.len() {
            velocities[i] -= beta * plane.normal[i] / masses[i];
        }
    }
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
