//! Live [`DynamicsEngine`] over an `rgpot` MD session.
//!
//! [`MdSessionEngine`] runs any backend of `rgpot_core::md_session`: the
//! nblib engine on a topology text or on a GROMACS run input file, OpenMM, or
//! a custom [`MdSession`]. The session owns the integrator, its rigid
//! constraints and the random stream, and reports the hyperplane multiplier
//! of every step.
//!
//! Units are Angstrom, ps and eV. The session counts mass in amu, and
//! `1 eV = 9648.5 amu A^2 / ps^2`, so [`MdSessionEngine::masses`] are in
//! `eV ps^2 / A^2`. With them `m v^2` and `k_B T` are both in eV.
//!
//! The noise of a run follows from the seed given to
//! [`DynamicsEngine::reseed`] and from the number of states set since. A new
//! state therefore starts a new stream instead of replaying the last one.
//!
//! A hyperplane normal that does not move rigid groups as a whole couples to
//! their constraints. The mean of the multiplier is then not the free-energy
//! derivative, because the metric determinant varies over the plane.
//! [`MdSessionEngine::check_rigid_translation`] refuses such a normal.

use ndarray::{Array1, ArrayView1};
use rgpot_core::md_session::{
    AMU_A2_PS2_PER_EV, MdLangevinConfig, MdSession, MdSessionError, MdSessionState,
};

use crate::engine::gle::ColoredNoise;
use crate::engine::{
    DynamicsEngine, HyperplaneConstraint, StepReport, SurfaceVelocityPolicy, ThermostatConfig,
    draw_maxwell_boltzmann_velocities, reflect_velocity_in_place, validate_finite_slice,
    validate_unit_normal,
};
use crate::error::FluxError;

fn surface(error: MdSessionError) -> FluxError {
    FluxError::Surface(error.to_string())
}

fn slice(values: ArrayView1<'_, f64>) -> Vec<f64> {
    values.iter().copied().collect()
}

/// An engine that integrates through an `rgpot` MD session.
pub struct MdSessionEngine<S: MdSession> {
    session: S,
    masses: Array1<f64>,
    positions: Array1<f64>,
    velocities: Array1<f64>,
    forces: Array1<f64>,
    potential: f64,
    thermostat: ThermostatConfig,
    constraint: Option<HyperplaneConstraint>,
    plane: Option<(Vec<f64>, Vec<f64>)>,
    cell: Option<[f64; 9]>,
    seed: u64,
    epoch: u64,
    metrics: Vec<f64>,
}

impl<S: MdSession> MdSessionEngine<S> {
    /// Take over `session` at its current state.
    ///
    /// # Errors
    ///
    /// A colored-noise or athermal thermostat, or a failing session call.
    pub fn new(mut session: S, thermostat: ThermostatConfig) -> Result<Self, FluxError> {
        check_thermostat(&thermostat)?;
        session.initialize().map_err(surface)?;
        let state = session.state().map_err(surface)?;
        let masses: Array1<f64> = session
            .masses()
            .map_err(surface)?
            .iter()
            .flat_map(|mass| [mass / AMU_A2_PS2_PER_EV; 3])
            .collect();
        if masses.len() != session.dof() || state.positions.len() != masses.len() {
            return Err(FluxError::Shape {
                expected: session.dof(),
                got: masses.len(),
                context: "MD session masses",
            });
        }
        crate::engine::validate_masses(masses.view())?;
        let cell = session.periodic_box().ok();
        Ok(Self {
            session,
            masses,
            positions: state.positions.into(),
            velocities: state.velocities.into(),
            forces: state.forces.into(),
            potential: state.potential_energy,
            thermostat,
            constraint: None,
            plane: None,
            cell,
            seed: 1,
            epoch: 0,
            metrics: Vec::new(),
        })
    }

    /// Borrow the session.
    pub const fn session(&self) -> &S {
        &self.session
    }

    /// Refuse a normal that deforms or rotates a rigid group.
    ///
    /// `groups` are atom index lists, one per group of atoms joined by rigid
    /// constraints. The normal must be proportional to the atom masses within
    /// each group and axis. A translation of whole groups is orthogonal, in
    /// the inverse-mass metric, to every internal constraint, so the metric
    /// determinant of the plane is constant.
    pub fn check_rigid_translation(
        &self,
        normal: ArrayView1<'_, f64>,
        groups: &[Vec<usize>],
    ) -> Result<(), FluxError> {
        validate_unit_normal(normal, self.masses.len())?;
        for group in groups {
            for axis in 0..3 {
                let ratio = |atom: &usize| normal[3 * atom + axis] / self.masses[3 * atom + axis];
                let Some(first) = group.first().map(ratio) else {
                    continue;
                };
                for atom in group {
                    let scale = first.abs().max(ratio(atom).abs());
                    if (ratio(atom) - first).abs() > 1e-9 * scale {
                        return Err(FluxError::InvalidConfig(
                            "the hyperplane normal does not translate a rigid group as a whole",
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    fn config(&self, dt: f64) -> Result<MdLangevinConfig, FluxError> {
        Ok(MdLangevinConfig::default()
            .with_k_t(self.thermostat.thermal_energy()?)
            .with_friction(self.thermostat.gamma)
            .with_time_step(dt)
            .with_seed(
                self.seed
                    .wrapping_add(self.epoch.wrapping_mul(0x9e37_79b9_7f4a_7c15)),
            )
            .with_record_constraint_force(self.plane.is_some()))
    }

    fn adopt(&mut self, state: MdSessionState) {
        self.positions = state.positions.into();
        self.velocities = state.velocities.into();
        self.forces = state.forces.into();
        self.potential = state.potential_energy;
    }

    /// Advance `steps` steps in one session call and return the multiplier of
    /// each step, empty without a hyperplane.
    pub fn run(&mut self, dt: f64, steps: usize) -> Result<Vec<f64>, FluxError> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(FluxError::InvalidConfig(
                "MdSessionEngine needs a finite positive timestep",
            ));
        }
        let config = self.config(dt)?;
        let plane = self
            .plane
            .as_ref()
            .map(|(point, normal)| (point.as_slice(), normal.as_slice()));
        let outcome = self
            .session
            .run_langevin(&config, plane, &[], &[], steps)
            .map_err(surface)?;
        let state = self.session.state().map_err(surface)?;
        self.adopt(state);
        if self.plane.is_some() && outcome.constraint_force.len() != steps {
            return Err(FluxError::InvalidConfig(
                "the MD session reported no constraint force",
            ));
        }
        if outcome.constraint_force.iter().any(|l| !l.is_finite()) {
            return Err(FluxError::NonFinite("MD session constraint force"));
        }
        self.metrics = outcome.constraint_metric;
        Ok(outcome.constraint_force)
    }

    /// Plane metric of each step of the last [`Self::run`] in 1/amu, empty when
    /// the engine does not report one.
    ///
    /// The free energy along the plane offset is the integral of the mean
    /// multiplier minus `k_B T ln <Z^(-1/2)>`.
    pub fn metrics(&self) -> &[f64] {
        &self.metrics
    }
}

fn check_thermostat(thermostat: &ThermostatConfig) -> Result<(), FluxError> {
    if thermostat.thermal_energy()? <= 0.0 {
        return Err(FluxError::InvalidConfig(
            "MdSessionEngine needs a positive thermal energy",
        ));
    }
    if thermostat.noise != ColoredNoise::White {
        return Err(FluxError::InvalidConfig(
            "MdSessionEngine does not support colored-noise thermostats",
        ));
    }
    Ok(())
}

impl<S: MdSession> DynamicsEngine for MdSessionEngine<S> {
    fn dof(&self) -> usize {
        self.masses.len()
    }

    fn masses(&self) -> ArrayView1<'_, f64> {
        self.masses.view()
    }

    fn positions(&self) -> ArrayView1<'_, f64> {
        self.positions.view()
    }

    fn set_positions(&mut self, positions: ArrayView1<'_, f64>) -> Result<(), FluxError> {
        validate_finite_slice(positions, "MdSessionEngine positions")?;
        self.session
            .set_positions(&slice(positions))
            .map_err(surface)?;
        self.epoch += 1;
        let state = self.session.state().map_err(surface)?;
        self.adopt(state);
        Ok(())
    }

    fn velocities(&self) -> ArrayView1<'_, f64> {
        self.velocities.view()
    }

    fn set_velocities(&mut self, velocities: ArrayView1<'_, f64>) -> Result<(), FluxError> {
        validate_finite_slice(velocities, "MdSessionEngine velocities")?;
        self.session
            .set_velocities(&slice(velocities))
            .map_err(surface)?;
        self.epoch += 1;
        self.velocities.assign(&velocities);
        Ok(())
    }

    fn cell(&self) -> Option<&[f64; 9]> {
        self.cell.as_ref()
    }

    fn set_cell(&mut self, cell: Option<[f64; 9]>) -> Result<(), FluxError> {
        match cell {
            Some(cell) if self.cell != Some(cell) => {
                self.session.set_periodic_box(&cell).map_err(surface)?;
                self.cell = Some(cell);
                let state = self.session.state().map_err(surface)?;
                self.adopt(state);
            }
            None if self.cell.is_some() => {
                return Err(FluxError::InvalidConfig(
                    "a periodic MD session cannot drop its cell",
                ));
            }
            _ => {}
        }
        Ok(())
    }

    fn thermostat(&self) -> &ThermostatConfig {
        &self.thermostat
    }

    fn set_thermostat(&mut self, thermostat: ThermostatConfig) -> Result<(), FluxError> {
        check_thermostat(&thermostat)?;
        self.thermostat = thermostat;
        Ok(())
    }

    fn constraint(&self) -> Option<&HyperplaneConstraint> {
        self.constraint.as_ref()
    }

    fn set_constraint(
        &mut self,
        constraint: Option<HyperplaneConstraint>,
    ) -> Result<(), FluxError> {
        self.plane = match &constraint {
            Some(plane) => {
                plane.validate()?;
                validate_unit_normal(plane.normal.view(), self.dof())?;
                let normal = slice(plane.normal.view());
                let point = normal.iter().map(|n| n * plane.offset).collect();
                Some((point, normal))
            }
            None => None,
        };
        self.constraint = constraint;
        Ok(())
    }

    fn energy_and_forces(&mut self) -> Result<(f64, Array1<f64>), FluxError> {
        Ok((self.potential, self.forces.clone()))
    }

    fn step(&mut self, dt: f64) -> Result<StepReport, FluxError> {
        let lambda = self.run(dt, 1)?.first().copied();
        Ok(
            StepReport::new(self.potential, self.kinetic_energy(), lambda)
                .with_constraint_metric(self.metrics.first().copied()),
        )
    }

    fn reseed(&mut self, seed: u64) -> Result<(), FluxError> {
        self.seed = seed;
        self.epoch = 0;
        Ok(())
    }

    /// Draws unconstrained velocities; the session projects them onto its
    /// rigid constraints and the hyperplane when the next run starts.
    fn shoot_velocities(&mut self, seed: u64) -> Result<(), FluxError> {
        let kbt = self.thermostat.thermal_energy()?;
        let drawn = draw_maxwell_boltzmann_velocities(self.masses(), kbt, seed)?;
        self.set_velocities(drawn.view())?;
        self.reseed(seed)
    }

    fn reflect_velocity(&mut self, normal: ArrayView1<'_, f64>) -> Result<f64, FluxError> {
        let mut velocities = self.velocities.clone();
        let along = reflect_velocity_in_place(&mut velocities, self.masses.view(), normal)?;
        self.set_velocities(velocities.view())?;
        Ok(along)
    }

    fn apply_dividing_surface(
        &mut self,
        normal: ArrayView1<'_, f64>,
        policy: SurfaceVelocityPolicy,
    ) -> Result<f64, FluxError> {
        validate_unit_normal(normal, self.dof())?;
        let along = rgmin::vecops::dot(normal, self.velocities.view());
        if along > 0.0 {
            return Ok(along);
        }
        match policy {
            SurfaceVelocityPolicy::Reflect if along < 0.0 => self.reflect_velocity(normal),
            SurfaceVelocityPolicy::Reflect => Err(FluxError::InvalidConfig(
                "zero normal velocity cannot be oriented positive by reflection",
            )),
            SurfaceVelocityPolicy::RejectRedraw { seed, max_attempts } => {
                for attempt in 0..max_attempts {
                    self.shoot_velocities(seed.wrapping_add(attempt as u64))?;
                    let along = rgmin::vecops::dot(normal, self.velocities.view());
                    if along > 0.0 {
                        return Ok(along);
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
    use approx::assert_relative_eq;
    use rgpot_core::md_session::MdLangevinOutcome;

    /// Records what the engine asks for; the multiplier is a constant.
    struct Recorder {
        x: Vec<f64>,
        v: Vec<f64>,
        seeds: Vec<u64>,
        offsets: Vec<f64>,
    }

    impl MdSession for Recorder {
        fn dof(&self) -> usize {
            6
        }
        fn masses(&self) -> Result<Vec<f64>, MdSessionError> {
            Ok(vec![16.0, 1.0])
        }
        fn set_positions(&mut self, positions: &[f64]) -> Result<(), MdSessionError> {
            self.x = positions.to_vec();
            Ok(())
        }
        fn set_velocities(&mut self, velocities: &[f64]) -> Result<(), MdSessionError> {
            self.v = velocities.to_vec();
            Ok(())
        }
        fn step_langevin(
            &mut self,
            _: &MdLangevinConfig,
            _: &[f64],
            _: &[f64],
            _: &[f64],
            _: &[f64],
            _: usize,
        ) -> Result<usize, MdSessionError> {
            Err(MdSessionError::new("unused"))
        }
        fn step_nve(&mut self, _: f64, _: usize) -> Result<(), MdSessionError> {
            Err(MdSessionError::new("unused"))
        }
        fn initialize(&mut self) -> Result<(), MdSessionError> {
            Ok(())
        }
        fn state(&mut self) -> Result<MdSessionState, MdSessionError> {
            Ok(MdSessionState::default()
                .with_positions(self.x.clone())
                .with_velocities(self.v.clone())
                .with_forces(vec![0.0; 6]))
        }
        fn run_langevin(
            &mut self,
            config: &MdLangevinConfig,
            plane: Option<(&[f64], &[f64])>,
            _: &[f64],
            _: &[f64],
            steps: usize,
        ) -> Result<MdLangevinOutcome, MdSessionError> {
            self.seeds.push(config.seed);
            let mut outcome = MdLangevinOutcome::default();
            if let Some((point, normal)) = plane {
                self.offsets
                    .push(point.iter().zip(normal).map(|(p, n)| p * n).sum());
                outcome.constraint_force = vec![0.25; steps];
                outcome.constraint_metric = vec![0.0625; steps];
            }
            Ok(outcome)
        }
    }

    fn engine() -> MdSessionEngine<Recorder> {
        MdSessionEngine::new(
            Recorder {
                x: vec![0.0; 6],
                v: vec![0.0; 6],
                seeds: vec![],
                offsets: vec![],
            },
            ThermostatConfig::default().with_temperature(0.025),
        )
        .expect("engine")
    }

    #[test]
    fn masses_make_kinetic_and_thermal_energy_both_ev() {
        assert_relative_eq!(engine().masses()[0] * AMU_A2_PS2_PER_EV, 16.0);
    }

    #[test]
    fn a_new_state_starts_a_new_stream_and_a_chunk_continues_one() {
        let mut engine = engine();
        engine.reseed(7).expect("reseed");
        engine.run(0.001, 2).expect("run");
        engine.run(0.001, 2).expect("run");
        engine
            .set_positions(Array1::zeros(6).view())
            .expect("positions");
        engine.run(0.001, 2).expect("run");
        let seeds = &engine.session().seeds;
        assert_eq!(seeds[0], 7);
        assert_eq!(seeds[0], seeds[1]);
        assert_ne!(seeds[1], seeds[2]);
    }

    #[test]
    fn the_plane_reaches_the_session_and_its_multiplier_returns() {
        let mut engine = engine();
        let mut normal = Array1::zeros(6);
        normal[0] = 1.0;
        let plane = HyperplaneConstraint::new(normal, 1.5).expect("plane");
        engine.set_constraint(Some(plane)).expect("constraint");
        let report = engine.step(0.001).expect("step");
        assert_eq!(report.constraint_force, Some(0.25));
        assert_eq!(report.constraint_metric, Some(0.0625));
        assert_relative_eq!(engine.session().offsets[0], 1.5);
    }

    #[test]
    fn a_normal_that_bends_a_rigid_group_is_refused() {
        let engine = engine();
        let norm = 16.0_f64.hypot(1.0);
        let whole = Array1::from(vec![16.0 / norm, 0.0, 0.0, 1.0 / norm, 0.0, 0.0]);
        let single = Array1::from(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let groups = [vec![0, 1]];
        assert!(
            engine
                .check_rigid_translation(whole.view(), &groups)
                .is_ok()
        );
        assert!(
            engine
                .check_rigid_translation(single.view(), &groups)
                .is_err()
        );
    }
}
