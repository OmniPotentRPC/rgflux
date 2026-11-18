//! Pure-Rust BAOAB Langevin reference engine with RATTLE and SHAKE hyperplane constraints.
//!
//! [`NativeEngine`] integrates unconstrained or hyperplane-constrained Langevin
//! dynamics (`gamma > 0`) and velocity-Verlet NVE dynamics (`gamma == 0`) over
//! any [`ForceSurface`] (including [`rgsaddle::PointSurface`],
//! [`crate::engine::EindirSurface`], and `engine::rgpot::RgpotSurface`).
//!
//! When a [`HyperplaneConstraint`] `sigma(x) = n · x - c = 0` is installed,
//! positions and velocities are projected onto the cotangent bundle in the
//! mass metric (`M = diag(m_i)`), and [`StepReport::constraint_force`] reports
//! the holonomic Lagrange multiplier
//! `lambda = -(n^T M^-1 F(x)) / (n^T M^-1 n) = (n^T M^-1 grad V(x)) / (n^T M^-1 n)`
//! whose canonical mean equals `dF/dc` (Sprik and Ciccotti 1998).
//!
//! ## Example
//!
//! ```rust
//! use ndarray::{Array1, ArrayView1};
//! use rgflux::engine::native::{NativeConfig, NativeEngine};
//! use rgflux::engine::{DynamicsEngine, ThermostatConfig};
//!
//! struct Harmonic1D;
//! impl rgsaddle::PointSurface for Harmonic1D {
//!     fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
//!         Ok((0.5 * x[0] * x[0], Array1::from_vec(vec![x[0]])))
//!     }
//! }
//!
//! let cfg = NativeConfig::default()
//!     .with_masses(Array1::from_vec(vec![1.0]))
//!     .with_thermostat(ThermostatConfig::default().with_gamma(1.0).with_temperature(0.5))
//!     .with_seed(7);
//! let mut engine = NativeEngine::new(Harmonic1D, cfg)?;
//! engine.shoot_velocities(7)?;
//! let report = engine.step(0.01)?;
//! assert!(report.potential_energy.is_finite());
//! # Ok::<(), rgflux::FluxError>(())
//! ```
//!
//! ## References
//!
//! - Leimkuhler, B., and Matthews, C. (2012). *Rational Construction of
//!   Stochastic Numerical Methods for Molecular Sampling*. Applied Mathematics
//!   Research eXpress. <https://doi.org/10.1093/amrx/abs010>
//! - Leimkuhler, B., and Matthews, C. (2016). *Efficient molecular dynamics
//!   using geodesic integration and solvent–solute splitting*. Proceedings of
//!   the Royal Society A, 472, 20160138. <https://doi.org/10.1098/rspa.2016.0138>
//! - Ryckaert, J.-P., Ciccotti, G., and Berendsen, H. J. C. (1977). *Numerical
//!   integration of the cartesian equations of motion of a system with
//!   constraints: molecular dynamics of n-alkanes*. Journal of Computational
//!   Physics, 23, 327–341. <https://doi.org/10.1016/0021-9991(77)90098-5>
//! - Andersen, H. C. (1983). *Rattle: A "velocity" version of the shake
//!   algorithm for molecular dynamics calculations*. Journal of Computational
//!   Physics, 52, 24–34. <https://doi.org/10.1016/0021-9991(83)90014-1>
//! - Sprik, M., and Ciccotti, G. (1998). *Free energy from constrained
//!   molecular dynamics*. The Journal of Chemical Physics, 109, 7737–7744.
//!   <https://doi.org/10.1063/1.477419>

use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rgmin::vecops::dot;

use crate::engine::{
    DynamicsEngine, ForceSurface, HyperplaneConstraint, StepReport, SurfaceVelocityPolicy,
    ThermostatConfig, inv_mass_metric_norm, rattle_velocities_in_place, reflect_velocity_in_place,
    sample_maxwell_boltzmann_with_rng, shake_positions_in_place, validate_finite_slice,
    validate_masses, validate_unit_normal,
};
use crate::error::FluxError;

/// Odd 64-bit SplitMix64 Weyl constant used to derive an independent thermal-noise
/// stream from the velocity-shooting seed during [`DynamicsEngine::shoot_velocities`].
const SHOOT_NOISE_STREAM_MIX: u64 = 0x9E37_79B9_7F4A_7C15;

/// Holonomic hyperplane constraint enforcement scheme in [`NativeEngine`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ConstraintProjection {
    /// Geodesic BAOAB: RATTLE tangent velocity projection after velocity kicks and
    /// Ornstein-Uhlenbeck noise, paired with SHAKE position projection after drifts.
    #[default]
    Rattle,
    /// SHAKE coordinate projection after each drift step, updating velocities from
    /// the mass-weighted coordinate correction `(x_proj - x_trial) / (dt / 2)`.
    Shake,
}

/// Configuration for [`NativeEngine`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct NativeConfig {
    /// Per-coordinate masses `m_i > 0` (defines `dof = masses.len()`).
    pub masses: Array1<f64>,
    /// Langevin thermostat parameters (`gamma` and `k_B * T`).
    pub thermostat: ThermostatConfig,
    /// Optional holonomic hyperplane constraint `n · x - offset = 0`.
    pub constraint: Option<HyperplaneConstraint>,
    /// Constraint projection algorithm (`Rattle` by default, or `Shake`).
    pub projection: ConstraintProjection,
    /// Optional row-major 3x3 periodic simulation cell.
    pub cell: Option<[f64; 9]>,
    /// Initial deterministic 64-bit PRNG seed for BAOAB Ornstein-Uhlenbeck noise.
    pub seed: u64,
}

impl Default for NativeConfig {
    fn default() -> Self {
        Self {
            masses: Array1::from_vec(vec![1.0]),
            thermostat: ThermostatConfig::default(),
            constraint: None,
            projection: ConstraintProjection::Rattle,
            cell: None,
            seed: 1,
        }
    }
}

with_setters!(NativeConfig {
    with_masses => masses: val Array1<f64>,
    with_thermostat => thermostat: val ThermostatConfig,
    with_constraint => constraint: opt HyperplaneConstraint,
    with_projection => projection: val ConstraintProjection,
    with_cell => cell: opt [f64; 9],
    with_seed => seed: val u64,
});

/// In-process BAOAB Langevin and velocity-Verlet dynamics engine over a [`ForceSurface`].
pub struct NativeEngine<S> {
    surface: S,
    masses: Array1<f64>,
    positions: Array1<f64>,
    velocities: Array1<f64>,
    thermostat: ThermostatConfig,
    constraint: Option<HyperplaneConstraint>,
    projection: ConstraintProjection,
    cell: Option<[f64; 9]>,
    rng: StdRng,
    cached_eval: Option<(f64, Array1<f64>)>,
}

impl<S: ForceSurface> NativeEngine<S> {
    /// Create a [`NativeEngine`] over `surface` initialized at the origin (projected onto
    /// `config.constraint` if active) with zero initial velocities.
    pub fn new(surface: S, config: NativeConfig) -> Result<Self, FluxError> {
        validate_masses(config.masses.view())?;
        let dof = config.masses.len();
        let _ = config.thermostat.thermal_energy()?;
        if let Some(c) = &config.cell {
            for &entry in c {
                if !entry.is_finite() {
                    return Err(FluxError::NonFinite("NativeConfig::cell"));
                }
            }
        }
        if let Some(plane) = &config.constraint {
            plane.validate()?;
            if plane.normal.len() != dof {
                return Err(FluxError::Shape {
                    expected: dof,
                    got: plane.normal.len(),
                    context: "NativeConfig constraint normal",
                });
            }
        }

        let mut engine = Self {
            surface,
            positions: Array1::zeros(dof),
            velocities: Array1::zeros(dof),
            masses: config.masses,
            thermostat: config.thermostat,
            constraint: config.constraint,
            projection: config.projection,
            cell: config.cell,
            rng: StdRng::seed_from_u64(config.seed),
            cached_eval: None,
        };
        engine.project_state_onto_constraint()?;
        Ok(engine)
    }

    /// Borrow the underlying [`ForceSurface`].
    pub fn surface(&self) -> &S {
        &self.surface
    }

    /// Active holonomic constraint projection mode.
    pub const fn projection(&self) -> ConstraintProjection {
        self.projection
    }

    /// Reset the internal Langevin PRNG stream to `seed`.
    pub fn reseed_rng(&mut self, seed: u64) {
        self.rng = StdRng::seed_from_u64(seed);
    }

    /// Public reseed alias matching [`DynamicsEngine::reseed`].
    pub fn reseed(&mut self, seed: u64) -> Result<(), FluxError> {
        self.reseed_rng(seed);
        Ok(())
    }

    fn constraint_lagrange_multiplier(
        forces: ArrayView1<'_, f64>,
        masses: ArrayView1<'_, f64>,
        plane: &HyperplaneConstraint,
    ) -> f64 {
        let z = inv_mass_metric_norm(plane.normal.view(), masses);
        let mut normal_accel = 0.0_f64;
        for i in 0..forces.len() {
            normal_accel += plane.normal[i] * forces[i] / masses[i];
        }
        -normal_accel / z
    }

    fn project_state_onto_constraint(&mut self) -> Result<(), FluxError> {
        if let Some(plane) = &self.constraint {
            shake_positions_in_place(&mut self.positions, self.masses.view(), plane)?;
            rattle_velocities_in_place(&mut self.velocities, self.masses.view(), plane);
            self.cached_eval = None;
        }
        Ok(())
    }

    fn ensure_eval(&mut self) -> Result<(f64, Array1<f64>), FluxError> {
        if let Some((e, f)) = &self.cached_eval {
            return Ok((*e, f.clone()));
        }
        let eval = self
            .surface
            .energy_and_forces(self.positions.view(), self.cell.as_ref())?;
        self.cached_eval = Some((eval.0, eval.1.clone()));
        Ok(eval)
    }

    fn drift_half_step(&mut self, half_dt: f64) -> Result<(), FluxError> {
        match &self.constraint {
            None => {
                for i in 0..self.positions.len() {
                    self.positions[i] += half_dt * self.velocities[i];
                }
            }
            Some(plane) => match self.projection {
                ConstraintProjection::Rattle => {
                    for i in 0..self.positions.len() {
                        self.positions[i] += half_dt * self.velocities[i];
                    }
                    shake_positions_in_place(&mut self.positions, self.masses.view(), plane)?;
                }
                ConstraintProjection::Shake => {
                    let mut trial = Array1::zeros(self.positions.len());
                    for i in 0..self.positions.len() {
                        trial[i] = self.positions[i] + half_dt * self.velocities[i];
                    }
                    let mut projected = trial.clone();
                    shake_positions_in_place(&mut projected, self.masses.view(), plane)?;
                    for i in 0..self.positions.len() {
                        self.velocities[i] += (projected[i] - trial[i]) / half_dt;
                    }
                    self.positions = projected;
                }
            },
        }
        self.cached_eval = None;
        Ok(())
    }
}

impl<S: ForceSurface> DynamicsEngine for NativeEngine<S> {
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
                context: "NativeEngine positions",
            });
        }
        validate_finite_slice(positions, "NativeEngine positions")?;
        self.positions.assign(&positions);
        self.cached_eval = None;
        if let Some(plane) = &self.constraint {
            shake_positions_in_place(&mut self.positions, self.masses.view(), plane)?;
        }
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
                context: "NativeEngine velocities",
            });
        }
        validate_finite_slice(velocities, "NativeEngine velocities")?;
        self.velocities.assign(&velocities);
        if let Some(plane) = &self.constraint {
            rattle_velocities_in_place(&mut self.velocities, self.masses.view(), plane);
        }
        Ok(())
    }

    fn cell(&self) -> Option<&[f64; 9]> {
        self.cell.as_ref()
    }

    fn set_cell(&mut self, cell: Option<[f64; 9]>) -> Result<(), FluxError> {
        if let Some(c) = &cell {
            for &entry in c {
                if !entry.is_finite() {
                    return Err(FluxError::NonFinite("NativeEngine cell"));
                }
            }
        }
        self.cell = cell;
        self.cached_eval = None;
        Ok(())
    }

    fn thermostat(&self) -> &ThermostatConfig {
        &self.thermostat
    }

    fn set_thermostat(&mut self, thermostat: ThermostatConfig) -> Result<(), FluxError> {
        let _ = thermostat.thermal_energy()?;
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
        if let Some(plane) = &constraint {
            plane.validate()?;
            if plane.normal.len() != self.dof() {
                return Err(FluxError::Shape {
                    expected: self.dof(),
                    got: plane.normal.len(),
                    context: "NativeEngine constraint normal",
                });
            }
        }
        self.constraint = constraint;
        self.project_state_onto_constraint()
    }

    fn energy_and_forces(&mut self) -> Result<(f64, Array1<f64>), FluxError> {
        self.ensure_eval()
    }

    fn step(&mut self, dt: f64) -> Result<StepReport, FluxError> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(FluxError::InvalidConfig(
                "NativeEngine::step requires a finite positive timestep",
            ));
        }
        let kbt = self.thermostat.thermal_energy()?;
        let gamma = self.thermostat.gamma;
        let half_dt = 0.5 * dt;

        if let Some(plane) = &self.constraint {
            shake_positions_in_place(&mut self.positions, self.masses.view(), plane)?;
            if self.projection == ConstraintProjection::Rattle {
                rattle_velocities_in_place(&mut self.velocities, self.masses.view(), plane);
            }
        }

        // B: opening half velocity kick.
        let (_e0, f0) = self.ensure_eval()?;
        for i in 0..self.velocities.len() {
            self.velocities[i] += half_dt * f0[i] / self.masses[i];
        }
        if let Some(plane) = &self.constraint
            && self.projection == ConstraintProjection::Rattle
        {
            rattle_velocities_in_place(&mut self.velocities, self.masses.view(), plane);
        }

        // A: first half drift (+ SHAKE on hyperplane if active).
        self.drift_half_step(half_dt)?;

        // O: exact Ornstein-Uhlenbeck velocity update over full dt.
        if gamma > 0.0 {
            let c1 = (-gamma * dt).exp();
            let one_minus_c1_sq = {
                let raw = -(-2.0 * gamma * dt).exp_m1();
                if raw < 0.0 { 0.0 } else { raw }
            };
            let thermal_kick = sample_maxwell_boltzmann_with_rng(
                self.masses.view(),
                kbt * one_minus_c1_sq,
                &mut self.rng,
            );
            for i in 0..self.velocities.len() {
                self.velocities[i] = c1 * self.velocities[i] + thermal_kick[i];
            }
            if let Some(plane) = &self.constraint
                && self.projection == ConstraintProjection::Rattle
            {
                rattle_velocities_in_place(&mut self.velocities, self.masses.view(), plane);
            }
        }

        // A: second half drift (+ SHAKE on hyperplane if active).
        self.drift_half_step(half_dt)?;

        // Evaluate potential and forces at the new positions, then B: closing half kick.
        let (potential_energy, f1) = self.ensure_eval()?;
        for i in 0..self.velocities.len() {
            self.velocities[i] += half_dt * f1[i] / self.masses[i];
        }

        let constraint_force = if let Some(plane) = &self.constraint {
            let lambda = Self::constraint_lagrange_multiplier(f1.view(), self.masses.view(), plane);
            rattle_velocities_in_place(&mut self.velocities, self.masses.view(), plane);
            Some(lambda)
        } else {
            None
        };

        validate_finite_slice(self.positions.view(), "NativeEngine stepped positions")?;
        validate_finite_slice(self.velocities.view(), "NativeEngine stepped velocities")?;
        let kinetic_energy = self.kinetic_energy();
        Ok(StepReport {
            potential_energy,
            kinetic_energy,
            constraint_force,
        })
    }

    fn reseed(&mut self, seed: u64) -> Result<(), FluxError> {
        NativeEngine::reseed(self, seed)
    }

    fn shoot_velocities(&mut self, seed: u64) -> Result<(), FluxError> {
        let kbt = self.thermostat.thermal_energy()?;
        let mut shoot_rng = StdRng::seed_from_u64(seed);
        self.velocities =
            sample_maxwell_boltzmann_with_rng(self.masses.view(), kbt, &mut shoot_rng);
        if let Some(plane) = &self.constraint {
            rattle_velocities_in_place(&mut self.velocities, self.masses.view(), plane);
        }
        self.rng = StdRng::seed_from_u64(seed ^ SHOOT_NOISE_STREAM_MIX);
        Ok(())
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
        let current_nv = dot(normal, self.velocities.view());
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
                    let nv = dot(normal, self.velocities.view());
                    if nv > 0.0 {
                        return Ok(nv);
                    }
                }
                Err(FluxError::SurfaceRejection { max_attempts })
            }
        }
    }
}
