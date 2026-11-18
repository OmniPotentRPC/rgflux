//! Engine-agnostic MD session backend over `rgpot`'s session C ABI and Rust bindings.
//!
//! External molecular dynamics packages (OpenMM, GROMACS `nblib`, and LAMMPS)
//! share one engine-neutral session interface in `OmniPotentRPC/rgpot`
//! (`CppCore/rgpot/ExternalPot/external_potentials.h` and
//! `rgpot_core::md_session`, on the `feat/md-session-openmm`,
//! `feat/md-session-neighbours`, and `feat/md-session-nblib-engine` branches).
//! [`RgpotSessionEngine`] is the [`DynamicsEngine`]
//! skeleton that selects an engine by [`SessionEngineKind`] (`"openmm"`,
//! `"nblib"`, or `"lammps"`).
//!
//! ## Mapping onto `rgpot` MD session calls
//!
//! All quantities cross `rgpot`'s MD session boundary in `rgpot` units:
//! positions and plane/wall coordinates in Angstrom, velocities in Angstrom/ps,
//! forces in eV/Angstrom, potential and kinetic energies and `kT` in eV,
//! time steps in ps, Langevin friction in `1/ps`, and masses in amu.
//!
//! - **Lifecycle and engine dispatch**:
//!   `rgpot_md_session_create(engine, engine_path, spec, spec_len, options, err, err_len)`,
//!   `rgpot_md_session_from_ops`, and `rgpot_md_session_destroy` in C;
//!   `rgpot_core::md_session::{CAbiMdSession, MdSession, MdSessionOps}`
//!   (`CAbiMdSession::open_engine`, `CAbiMdSession::from_ops`, and
//!   `MdSession::initialize`) in Rust.
//! - **Degrees of freedom and masses**:
//!   `rgpot_md_session_n_atoms` and `rgpot_md_session_get_masses`
//!   (`MdSession::dof` returning `3 * n_atoms`, and `MdSession::masses`
//!   returning per-atom masses in amu expanded across `(x, y, z)`).
//! - **State and periodic box**:
//!   `rgpot_md_session_set_positions`, `rgpot_md_session_set_velocities`,
//!   `rgpot_md_session_ops::set_periodic_box`, and `rgpot_md_session_get_state`
//!   (`MdSession::{set_positions, set_velocities, set_periodic_box, state}`
//!   yielding `MdSessionState { positions, velocities, forces, potential_energy, kinetic_energy }`).
//! - **Projected BAOAB Langevin, wall reflection, seeded runs, and NVE**:
//!   `rgpot_md_session_step_langevin(session, kT, friction, time_step, plane_point, plane_normal, wall_points, wall_normals, n_walls, n3, steps, seed, &wall_hits, err, err_len)`
//!   (`MdSession::step_langevin` with `MdLangevinConfig`) on the `3N`
//!   mass-metric hyperplane, and `rgpot_md_session_step_nve`
//!   (`MdSession::step_nve`) when `gamma == 0` and no hyperplane or reflecting
//!   wall is active.
//! - **Neighbour-list exchange slots**:
//!   `rgpot_md_session_get_neighbors` and `rgpot_md_session_set_neighbors`
//!   (`rgpot_md_session_ops::{get_neighbors, set_neighbors}`).

use std::path::PathBuf;

use ndarray::{Array1, ArrayView1};

use crate::engine::{
    DynamicsEngine, HyperplaneConstraint, StepReport, SurfaceVelocityPolicy, ThermostatConfig,
    draw_maxwell_boltzmann_velocities, reflect_velocity_in_place, validate_finite_slice,
    validate_masses, validate_unit_normal,
};
use crate::error::FluxError;

/// Engine selector passed to `rgpot_md_session_create`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SessionEngineKind {
    /// OpenMM session engine (`"openmm"`).
    #[default]
    OpenMm,
    /// GROMACS `nblib` session engine (`"nblib"`).
    NbLib,
    /// LAMMPS / `lammpc` session engine (`"lammps"`).
    Lammps,
}

impl SessionEngineKind {
    /// Canonical engine identifier expected by `rgpot_md_session_create`.
    pub const fn engine_name(self) -> &'static str {
        match self {
            Self::OpenMm => "openmm",
            Self::NbLib => "nblib",
            Self::Lammps => "lammps",
        }
    }
}

/// Configuration for [`RgpotSessionEngine`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct RgpotSessionConfig {
    /// Backend engine name (`"openmm"`, `"nblib"`, or `"lammps"`).
    pub engine: SessionEngineKind,
    /// Optional explicit plugin shared-library path for `CAbiMdSession::open_engine`.
    pub engine_path: Option<PathBuf>,
    /// Serialized system specification bytes (`spec`, `spec_len`) passed at session creation.
    pub spec: Vec<u8>,
    /// Backend-specific option string passed to `rgpot_md_session_create`.
    pub options: Option<String>,
    /// Initial per-coordinate masses (`3 * n_atoms`, in amu) before session query.
    pub masses: Array1<f64>,
    /// Langevin thermostat configuration (`gamma` in `1/ps`, `k_B * T` in `eV`).
    pub thermostat: ThermostatConfig,
    /// Optional holonomic hyperplane constraint forwarded as `(plane_point, plane_normal)`.
    pub constraint: Option<HyperplaneConstraint>,
    /// Optional row-major 3x3 periodic box vectors in Angstrom.
    pub cell: Option<[f64; 9]>,
    /// Deterministic 64-bit PRNG seed forwarded to `MdLangevinConfig::seed`.
    pub seed: u64,
}

impl Default for RgpotSessionConfig {
    fn default() -> Self {
        Self {
            engine: SessionEngineKind::OpenMm,
            engine_path: None,
            spec: Vec::new(),
            options: None,
            masses: Array1::from_elem(3, 1.0),
            thermostat: ThermostatConfig::default(),
            constraint: None,
            cell: None,
            seed: 1,
        }
    }
}

with_setters!(RgpotSessionConfig {
    with_engine => engine: val SessionEngineKind,
    with_engine_path => engine_path: opt PathBuf,
    with_spec => spec: val Vec<u8>,
    with_options => options: opt String,
    with_masses => masses: val Array1<f64>,
    with_thermostat => thermostat: val ThermostatConfig,
    with_constraint => constraint: opt HyperplaneConstraint,
    with_cell => cell: opt [f64; 9],
    with_seed => seed: val u64,
});

/// Skeleton [`DynamicsEngine`] wrapping `rgpot`'s MD session ABI.
///
/// Until `rgpot_core::md_session` merges onto `rgpot`'s `develop` branch, force
/// evaluation and time integration return [`FluxError::BackendUnavailable`]
/// rather than substituting a fake implementation.
#[derive(Clone, Debug)]
pub struct RgpotSessionEngine {
    config: RgpotSessionConfig,
    positions: Array1<f64>,
    velocities: Array1<f64>,
    rgpot_abi_major: u32,
}

impl RgpotSessionEngine {
    /// Construct an `rgpot` MD session skeleton configured for `config`.
    pub fn new(config: RgpotSessionConfig) -> Result<Self, FluxError> {
        let dof = config.masses.len();
        if dof == 0 || !dof.is_multiple_of(3) {
            return Err(FluxError::InvalidConfig(
                "RgpotSessionConfig::masses length must be a positive multiple of 3",
            ));
        }
        validate_masses(config.masses.view())?;
        let _ = config.thermostat.thermal_energy()?;
        if let Some(plane) = &config.constraint {
            plane.validate()?;
            if plane.normal.len() != dof {
                return Err(FluxError::Shape {
                    expected: dof,
                    got: plane.normal.len(),
                    context: "RgpotSessionConfig constraint normal",
                });
            }
        }
        if let Some(c) = &config.cell {
            for &entry in c {
                if !entry.is_finite() {
                    return Err(FluxError::NonFinite("RgpotSessionConfig::cell"));
                }
            }
        }
        let rgpot_abi_major = rgpot_core::version::rgpot_version_major();
        Ok(Self {
            positions: Array1::zeros(dof),
            velocities: Array1::zeros(dof),
            config,
            rgpot_abi_major,
        })
    }

    /// Selected `rgpot` session backend kind.
    pub const fn engine_kind(&self) -> SessionEngineKind {
        self.config.engine
    }

    /// Linked `rgpot-core` major version.
    pub const fn rgpot_version_major(&self) -> u32 {
        self.rgpot_abi_major
    }

    const fn unavailable_reason(&self) -> &'static str {
        match self.config.engine {
            SessionEngineKind::OpenMm => {
                "rgpot_md_session_create(\"openmm\") awaits rgpot_core::md_session merge"
            }
            SessionEngineKind::NbLib => {
                "rgpot_md_session_create(\"nblib\") awaits rgpot_core::md_session merge"
            }
            SessionEngineKind::Lammps => {
                "rgpot_md_session_create(\"lammps\") awaits rgpot_core::md_session merge"
            }
        }
    }
}

impl DynamicsEngine for RgpotSessionEngine {
    fn dof(&self) -> usize {
        self.positions.len()
    }

    fn masses(&self) -> ArrayView1<'_, f64> {
        self.config.masses.view()
    }

    fn positions(&self) -> ArrayView1<'_, f64> {
        self.positions.view()
    }

    fn set_positions(&mut self, positions: ArrayView1<'_, f64>) -> Result<(), FluxError> {
        if positions.len() != self.dof() {
            return Err(FluxError::Shape {
                expected: self.dof(),
                got: positions.len(),
                context: "RgpotSessionEngine positions",
            });
        }
        validate_finite_slice(positions, "RgpotSessionEngine positions")?;
        self.positions.assign(&positions);
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
                context: "RgpotSessionEngine velocities",
            });
        }
        validate_finite_slice(velocities, "RgpotSessionEngine velocities")?;
        self.velocities.assign(&velocities);
        Ok(())
    }

    fn cell(&self) -> Option<&[f64; 9]> {
        self.config.cell.as_ref()
    }

    fn set_cell(&mut self, cell: Option<[f64; 9]>) -> Result<(), FluxError> {
        if let Some(c) = &cell {
            for &entry in c {
                if !entry.is_finite() {
                    return Err(FluxError::NonFinite("RgpotSessionEngine cell"));
                }
            }
        }
        self.config.cell = cell;
        Ok(())
    }

    fn thermostat(&self) -> &ThermostatConfig {
        &self.config.thermostat
    }

    fn set_thermostat(&mut self, thermostat: ThermostatConfig) -> Result<(), FluxError> {
        let _ = thermostat.thermal_energy()?;
        self.config.thermostat = thermostat;
        Ok(())
    }

    fn constraint(&self) -> Option<&HyperplaneConstraint> {
        self.config.constraint.as_ref()
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
                    context: "RgpotSessionEngine constraint normal",
                });
            }
        }
        self.config.constraint = constraint;
        Ok(())
    }

    fn energy_and_forces(&mut self) -> Result<(f64, Array1<f64>), FluxError> {
        Err(FluxError::BackendUnavailable(self.unavailable_reason()))
    }

    fn step(&mut self, dt: f64) -> Result<StepReport, FluxError> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(FluxError::InvalidConfig(
                "DynamicsEngine::step requires a finite positive timestep",
            ));
        }
        Err(FluxError::BackendUnavailable(self.unavailable_reason()))
    }

    fn reseed(&mut self, seed: u64) -> Result<(), FluxError> {
        let _ = seed;
        Err(FluxError::BackendUnavailable(
            "rgpot session reseed is unsupported until rgpot_core::md_session is wired on develop",
        ))
    }

    fn shoot_velocities(&mut self, seed: u64) -> Result<(), FluxError> {
        let kbt = self.config.thermostat.thermal_energy()?;
        self.velocities = draw_maxwell_boltzmann_velocities(self.masses(), kbt, seed)?;
        Ok(())
    }

    fn reflect_velocity(&mut self, normal: ArrayView1<'_, f64>) -> Result<f64, FluxError> {
        reflect_velocity_in_place(&mut self.velocities, self.config.masses.view(), normal)
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
                    let trial_seed = seed.wrapping_add(attempt as u64);
                    self.shoot_velocities(trial_seed)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_engine_names_and_unavailable_contract() {
        assert_eq!(SessionEngineKind::OpenMm.engine_name(), "openmm");
        assert_eq!(SessionEngineKind::NbLib.engine_name(), "nblib");
        assert_eq!(SessionEngineKind::Lammps.engine_name(), "lammps");

        for kind in [
            SessionEngineKind::OpenMm,
            SessionEngineKind::NbLib,
            SessionEngineKind::Lammps,
        ] {
            let cfg = RgpotSessionConfig::default()
                .with_engine(kind)
                .with_masses(Array1::from_elem(6, 12.0));
            let mut engine = RgpotSessionEngine::new(cfg).expect("valid skeleton");
            assert_eq!(engine.dof(), 6);
            assert!(engine.rgpot_version_major() >= 3);
            assert!(matches!(
                engine.step(0.001),
                Err(FluxError::BackendUnavailable(_))
            ));
            assert!(matches!(
                engine.energy_and_forces(),
                Err(FluxError::BackendUnavailable(_))
            ));
            assert!(matches!(
                engine.reseed(99),
                Err(FluxError::BackendUnavailable(message))
                    if message.contains("reseed is unsupported")
            ));
        }
    }

    #[test]
    #[ignore = "requires rgpot_core::md_session on rgpot develop"]
    fn live_openmm_md_session_roundtrip() {
        let cfg = RgpotSessionConfig::default().with_engine(SessionEngineKind::OpenMm);
        let mut engine = RgpotSessionEngine::new(cfg).expect("skeleton");
        let report = engine.step(0.001).expect("live session step");
        assert!(report.potential_energy.is_finite());
    }
}
