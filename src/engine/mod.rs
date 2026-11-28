//! Engine-agnostic molecular dynamics contract and static potential adapters.
//!
//! Samplers in `rgflux` drive [`DynamicsEngine`] rather than binding to one
//! simulation package. Static potential energy surfaces enter through
//! [`ForceSurface`], which is implemented for every [`rgsaddle::PointSurface`],
//! for [`eindir_core::DifferentiableObjective`] via [`EindirSurface`], and
//! (behind the `rgpot` feature) for `rgpot` callback potentials. Direct
//! LAMMPS sessions enter through `lammpc` (behind the `lammpc` feature), and
//! external molecular dynamics packages (OpenMM, GROMACS `nblib`, and LAMMPS)
//! also enter through `rgpot_session` (behind the `rgpot-session` feature).
//!
//! ## References
//!
//! - Goswami, R. (2025). *Efficient Exploration of Chemical Kinetics*.
//!   arXiv:2510.21368. <https://doi.org/10.48550/arXiv.2510.21368>
//! - Leimkuhler, B., and Matthews, C. (2012). *Rational Construction of
//!   Stochastic Numerical Methods for Molecular Sampling*. Applied Mathematics
//!   Research eXpress. <https://doi.org/10.1093/amrx/abs010>
//! - Leimkuhler, B., and Matthews, C. (2016). *Efficient molecular dynamics
//!   using geodesic integration and solvent-solute splitting*. Proceedings of
//!   the Royal Society A, 472, 20160138. <https://doi.org/10.1098/rspa.2016.0138>
//! - Ryckaert, J.-P., Ciccotti, G., and Berendsen, H. J. C. (1977). *Numerical
//!   integration of the cartesian equations of motion of a system with
//!   constraints: molecular dynamics of n-alkanes*. Journal of Computational
//!   Physics, 23, 327-341. <https://doi.org/10.1016/0021-9991(77)90098-5>
//! - Andersen, H. C. (1983). *Rattle: A "velocity" version of the shake
//!   algorithm for molecular dynamics calculations*. Journal of Computational
//!   Physics, 52, 24-34. <https://doi.org/10.1016/0021-9991(83)90014-1>
//! - Sprik, M., and Ciccotti, G. (1998). *Free energy from constrained
//!   molecular dynamics*. The Journal of Chemical Physics, 109, 7737-7744.
//!   <https://doi.org/10.1063/1.477419>
//! - Ceriotti, M., Bussi, G., and Parrinello, M. (2010). *Colored-noise
//!   thermostats à la carte*. Journal of Chemical Theory and Computation, 6,
//!   1170-1180. <https://doi.org/10.1021/ct900563s>
//! - Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*
//!   (2nd ed.). SIAM. <https://doi.org/10.1137/1.9780898718027>

pub mod gle;
#[cfg(feature = "lammpc")]
pub mod lammpc;
pub mod native;
#[cfg(feature = "rgpot")]
pub mod rgpot;
#[cfg(feature = "rgpot-session")]
pub mod rgpot_session;

use eindir_core::DifferentiableObjective;
use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rgmin::vecops::{dot, nrm2};

use crate::error::FluxError;

/// Unit-normal deviation tolerance: `64 * f64::EPSILON` (`2^-46`), bounding
/// Euclidean norm rounding in IEEE-754 binary64 (Higham 2002, §3.1).
pub const UNIT_NORMAL_TOL: f64 = 64.0 * f64::EPSILON;

/// Relative holonomic hyperplane residual tolerance: `16 * f64::EPSILON` (`2^-48`),
/// bounding `|n · x - c| / (1 + |c| + ||x||_2)` after two-pass linear projection.
pub const HYPERPLANE_PROJECTION_TOL: f64 = 16.0 * f64::EPSILON;

/// Default maximum redraw attempts when rejecting non-positive normal velocities.
///
/// For an unbiased Maxwell-Boltzmann draw where `P(n · v > 0) = 1/2`, the chance
/// of `4096` consecutive rejections is `2^-4096`, far below `f64::MIN_POSITIVE`.
pub const DEFAULT_REJECTION_MAX_ATTEMPTS: usize = 4_096;

/// Static potential energy and force surface consumed by in-process integrators.
///
/// Modeled on [`rgsaddle::PointSurface`], returning physical forces `F = -grad V(x)`
/// alongside the scalar potential energy `V(x)`.
pub trait ForceSurface: Send + Sync {
    /// Evaluate potential energy `V(x)` and Cartesian forces `F(x) = -grad V(x)`.
    fn energy_and_forces(
        &self,
        positions: ArrayView1<'_, f64>,
        cell: Option<&[f64; 9]>,
    ) -> Result<(f64, Array1<f64>), FluxError>;
}

impl<S> ForceSurface for S
where
    S: rgsaddle::PointSurface + Send + Sync + ?Sized,
{
    fn energy_and_forces(
        &self,
        positions: ArrayView1<'_, f64>,
        cell: Option<&[f64; 9]>,
    ) -> Result<(f64, Array1<f64>), FluxError> {
        validate_finite_slice(positions, "surface positions")?;
        let (energy, gradient) = match cell {
            Some(c) => {
                for &entry in c {
                    if !entry.is_finite() {
                        return Err(FluxError::NonFinite("periodic cell"));
                    }
                }
                self.eval_in_cell(positions, c)?
            }
            None => self.eval(positions)?,
        };
        if !energy.is_finite() {
            return Err(FluxError::NonFinite("surface energy"));
        }
        if gradient.len() != positions.len() {
            return Err(FluxError::Shape {
                expected: positions.len(),
                got: gradient.len(),
                context: "surface gradient",
            });
        }
        validate_finite_slice(gradient.view(), "surface gradient")?;
        Ok((energy, gradient.mapv(|g| -g)))
    }
}

/// Adapter turning an [`eindir_core::DifferentiableObjective`] (as used by [`rgmin`])
/// into a [`ForceSurface`].
#[derive(Clone, Debug)]
pub struct EindirSurface<O> {
    objective: O,
}

impl<O> EindirSurface<O>
where
    O: DifferentiableObjective<f64> + Send + Sync,
{
    /// Wrap a differentiable objective whose gradient is `+grad V(x)`.
    pub fn new(objective: O) -> Self {
        Self { objective }
    }

    /// Borrow the underlying objective.
    pub fn inner(&self) -> &O {
        &self.objective
    }
}

impl<O> ForceSurface for EindirSurface<O>
where
    O: DifferentiableObjective<f64> + Send + Sync,
{
    fn energy_and_forces(
        &self,
        positions: ArrayView1<'_, f64>,
        _cell: Option<&[f64; 9]>,
    ) -> Result<(f64, Array1<f64>), FluxError> {
        let expected_dim = eindir_core::Objective::dim(&self.objective);
        if positions.len() != expected_dim {
            return Err(FluxError::Shape {
                expected: expected_dim,
                got: positions.len(),
                context: "EindirSurface positions",
            });
        }
        validate_finite_slice(positions, "EindirSurface positions")?;
        let (energy, gradient) = self.objective.value_and_gradient(positions);
        if !energy.is_finite() {
            return Err(FluxError::NonFinite("EindirSurface energy"));
        }
        if gradient.len() != positions.len() {
            return Err(FluxError::Shape {
                expected: positions.len(),
                got: gradient.len(),
                context: "EindirSurface gradient",
            });
        }
        validate_finite_slice(gradient.view(), "EindirSurface gradient")?;
        Ok((energy, gradient.mapv(|g| -g)))
    }
}

/// Langevin thermostat parameters (`gamma` and thermal energy `boltzmann * temperature`).
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct ThermostatConfig {
    /// Langevin friction coefficient `gamma >= 0` in inverse time units
    /// (`0` with white noise gives NVE).
    ///
    /// With white noise, samplers that average a constrained force as a
    /// canonical expectation, [`crate::hyperplane::sample_hyperplane`] and
    /// [`crate::pafi::pafi_profile`], require `gamma > 0` and return
    /// [`FluxError::InvalidConfig`] otherwise.
    /// A colored-noise model may provide canonical sampling independently of
    /// `gamma`, but each engine must implement that model or reject it. Within
    /// a supported canonical regime the sampled mean does not depend on
    /// `gamma`; only the statistical error and the burn-in needed do.
    ///
    /// Default: `1.0`.
    pub gamma: f64,
    /// Absolute temperature `T >= 0`.
    ///
    /// Default: `1.0`.
    pub temperature: f64,
    /// Boltzmann constant `k_B > 0` in the engine's energy-per-temperature unit (`1` for reduced units).
    ///
    /// Default: `1.0`.
    pub boltzmann: f64,
    /// White-noise Langevin, or the published smart-sampling GLE.
    ///
    /// The GLE step ignores `gamma`. On a holonomic constraint it thermostats
    /// only the tangent momenta. Default: [`gle::ColoredNoise::White`].
    pub noise: gle::ColoredNoise,
}

impl Default for ThermostatConfig {
    fn default() -> Self {
        Self {
            gamma: 1.0,
            temperature: 1.0,
            boltzmann: 1.0,
            noise: gle::ColoredNoise::White,
        }
    }
}

with_setters!(ThermostatConfig {
    with_gamma => gamma: val f64,
    with_temperature => temperature: val f64,
    with_boltzmann => boltzmann: val f64,
    with_noise => noise: val gle::ColoredNoise,
});

impl ThermostatConfig {
    /// Validate parameters and return `k_B * T`.
    pub fn thermal_energy(&self) -> Result<f64, FluxError> {
        if !self.gamma.is_finite() || !self.temperature.is_finite() || !self.boltzmann.is_finite() {
            return Err(FluxError::NonFinite("ThermostatConfig"));
        }
        if self.gamma < 0.0 {
            return Err(FluxError::InvalidConfig(
                "ThermostatConfig::gamma must be non-negative",
            ));
        }
        if self.temperature < 0.0 {
            return Err(FluxError::InvalidConfig(
                "ThermostatConfig::temperature must be non-negative",
            ));
        }
        if self.boltzmann <= 0.0 {
            return Err(FluxError::InvalidConfig(
                "ThermostatConfig::boltzmann must be positive",
            ));
        }
        let kbt = self.boltzmann * self.temperature;
        if !kbt.is_finite() {
            return Err(FluxError::NonFinite("thermal energy k_B * T"));
        }
        Ok(kbt)
    }

    /// Validate parameters and require friction `gamma > 0` for white noise.
    ///
    /// Equilibrium samplers average over constrained dynamics and read the
    /// mean as a canonical expectation. With white noise and `gamma == 0` the
    /// engine is NVE, which samples an energy shell started from the anchor
    /// and not the canonical ensemble, so those samplers refuse it. The GLE
    /// noise ignores `gamma` and is always thermostatted.
    pub(crate) fn require_canonical_sampling(&self) -> Result<f64, FluxError> {
        let kbt = self.thermal_energy()?;
        if self.gamma <= 0.0 && self.noise == gle::ColoredNoise::White {
            return Err(FluxError::InvalidConfig(
                "canonical sampling requires ThermostatConfig::gamma > 0; gamma == 0 is NVE dynamics",
            ));
        }
        Ok(kbt)
    }
}

/// Holonomic hyperplane constraint `sigma(x) = n · x - offset = 0` with unit normal `n`.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct HyperplaneConstraint {
    /// Unit normal vector `n` (`||n||_2 = 1`) oriented from reactant toward product.
    ///
    /// Default: `Array1::from_vec(vec![1.0])`.
    pub normal: Array1<f64>,
    /// Scalar plane offset `c` in `n · x - c = 0`.
    ///
    /// Default: `0.0`.
    pub offset: f64,
}

impl Default for HyperplaneConstraint {
    fn default() -> Self {
        Self {
            normal: Array1::from_vec(vec![1.0]),
            offset: 0.0,
        }
    }
}

with_setters!(HyperplaneConstraint {
    with_normal => normal: val Array1<f64>,
    with_offset => offset: val f64,
});

impl HyperplaneConstraint {
    /// Construct a hyperplane from an already unit-normalized `normal` and `offset`.
    pub fn new(normal: Array1<f64>, offset: f64) -> Result<Self, FluxError> {
        let plane = Self { normal, offset };
        plane.validate()?;
        Ok(plane)
    }

    /// Normalize a non-zero `direction` vector to unit Euclidean length and build the plane.
    pub fn from_direction(direction: ArrayView1<'_, f64>, offset: f64) -> Result<Self, FluxError> {
        if direction.is_empty() {
            return Err(FluxError::EmptyInput("hyperplane direction is empty"));
        }
        validate_finite_slice(direction, "hyperplane direction")?;
        if !offset.is_finite() {
            return Err(FluxError::NonFinite("hyperplane offset"));
        }
        let norm = nrm2(direction);
        if norm <= f64::MIN_POSITIVE.sqrt() {
            return Err(FluxError::InvalidConfig(
                "hyperplane direction has zero or underflowed norm",
            ));
        }
        let normal = direction.mapv(|c| c / norm);
        Self::new(normal, offset)
    }

    /// Validate finiteness and unit normal length (`| ||n||_2 - 1 | <= UNIT_NORMAL_TOL`).
    pub fn validate(&self) -> Result<(), FluxError> {
        if self.normal.is_empty() {
            return Err(FluxError::EmptyInput("hyperplane normal is empty"));
        }
        validate_finite_slice(self.normal.view(), "hyperplane normal")?;
        if !self.offset.is_finite() {
            return Err(FluxError::NonFinite("hyperplane offset"));
        }
        let norm = nrm2(self.normal.view());
        if (norm - 1.0).abs() > UNIT_NORMAL_TOL {
            return Err(FluxError::InvalidConfig(
                "hyperplane normal must have unit Euclidean norm",
            ));
        }
        Ok(())
    }

    /// Signed reaction coordinate `s(x) = n · x - offset`.
    pub fn signed_distance(&self, positions: ArrayView1<'_, f64>) -> Result<f64, FluxError> {
        if positions.len() != self.normal.len() {
            return Err(FluxError::Shape {
                expected: self.normal.len(),
                got: positions.len(),
                context: "hyperplane signed_distance",
            });
        }
        Ok(dot(self.normal.view(), positions) - self.offset)
    }
}

/// Policy for enforcing positive dividing-surface flux on velocities.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum SurfaceVelocityPolicy {
    /// Reflect a negative normal velocity across the dividing surface by a
    /// mass-weighted Householder reflection preserving kinetic energy.
    #[default]
    Reflect,
    /// Reject draws with `n · v <= 0` and redraw Maxwell-Boltzmann velocities
    /// from a seeded stream up to `max_attempts`.
    RejectRedraw {
        /// Deterministic RNG seed for redraw trials.
        seed: u64,
        /// Maximum redraw attempts before returning [`FluxError::SurfaceRejection`].
        max_attempts: usize,
    },
}

/// State diagnostics reported at the end of one [`DynamicsEngine::step`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct StepReport {
    /// Potential energy `V(x)` at the end-of-step positions.
    pub potential_energy: f64,
    /// Kinetic energy `(1/2) sum_i m_i v_i^2` at the end-of-step velocities.
    pub kinetic_energy: f64,
    /// Holonomic hyperplane Lagrange multiplier `lambda` (`<lambda> = dF/dc`) when constrained.
    pub constraint_force: Option<f64>,
}

impl StepReport {
    /// Build one end-of-step diagnostic report.
    #[must_use]
    pub const fn new(
        potential_energy: f64,
        kinetic_energy: f64,
        constraint_force: Option<f64>,
    ) -> Self {
        Self {
            potential_energy,
            kinetic_energy,
            constraint_force,
        }
    }
}

/// Object-safe molecular dynamics engine contract shared by all samplers.
///
/// Every sampler (`kappa::bennett_chandler`, `pafi`, `ohtst`, `ffs`,
/// `hyperplane`) accepts `&mut E` for `E: DynamicsEngine + ?Sized`, supporting
/// both monomorphized generics and `&mut dyn DynamicsEngine` dispatch.
pub trait DynamicsEngine {
    /// Number of scalar degrees of freedom.
    fn dof(&self) -> usize;

    /// Per-coordinate masses `m_i > 0`.
    fn masses(&self) -> ArrayView1<'_, f64>;

    /// Current unwrapped Cartesian positions.
    fn positions(&self) -> ArrayView1<'_, f64>;

    /// Overwrite positions (re-projecting onto the active hyperplane constraint if set).
    fn set_positions(&mut self, positions: ArrayView1<'_, f64>) -> Result<(), FluxError>;

    /// Current Cartesian velocities.
    fn velocities(&self) -> ArrayView1<'_, f64>;

    /// Overwrite velocities (re-projecting onto the active constraint tangent space if set).
    fn set_velocities(&mut self, velocities: ArrayView1<'_, f64>) -> Result<(), FluxError>;

    /// Row-major 3x3 periodic simulation cell, if periodic boundaries are active.
    fn cell(&self) -> Option<&[f64; 9]>;

    /// Set or clear the row-major 3x3 periodic simulation cell.
    fn set_cell(&mut self, cell: Option<[f64; 9]>) -> Result<(), FluxError>;

    /// Active Langevin thermostat configuration.
    fn thermostat(&self) -> &ThermostatConfig;

    /// Update the Langevin thermostat configuration.
    fn set_thermostat(&mut self, thermostat: ThermostatConfig) -> Result<(), FluxError>;

    /// Active holonomic hyperplane constraint, if any.
    fn constraint(&self) -> Option<&HyperplaneConstraint>;

    /// Install or clear a holonomic hyperplane constraint.
    fn set_constraint(&mut self, constraint: Option<HyperplaneConstraint>)
    -> Result<(), FluxError>;

    /// Potential energy `V(x)` and physical forces `F(x) = -grad V(x)` at [`Self::positions`].
    fn energy_and_forces(&mut self) -> Result<(f64, Array1<f64>), FluxError>;

    /// Instantaneous kinetic energy `(1/2) sum_i m_i v_i^2`.
    fn kinetic_energy(&self) -> f64 {
        let m = self.masses();
        let v = self.velocities();
        let mut ke = 0.0_f64;
        for i in 0..m.len() {
            ke += 0.5 * m[i] * v[i] * v[i];
        }
        ke
    }

    /// Advance the engine by one integrator step of duration `dt > 0`.
    fn step(&mut self, dt: f64) -> Result<StepReport, FluxError>;

    /// Reset the engine's stochastic stream to `seed`.
    ///
    /// Engines with deterministic dynamics and no stochastic terms may accept the
    /// call as a no-op. Engines whose stochastic backend is unavailable should
    /// return a typed error describing that reseeding is unsupported.
    fn reseed(&mut self, seed: u64) -> Result<(), FluxError>;

    /// Reseed velocities from the Maxwell-Boltzmann distribution at [`Self::thermostat`]
    /// using a deterministic stream initialized from `seed`.
    fn shoot_velocities(&mut self, seed: u64) -> Result<(), FluxError>;

    /// Reflect [`Self::velocities`] across the dividing surface with unit `normal`,
    /// preserving kinetic energy under the mass metric, and return the new `normal · v`.
    fn reflect_velocity(&mut self, normal: ArrayView1<'_, f64>) -> Result<f64, FluxError>;

    /// Enforce positive normal velocity `normal · v > 0` at a dividing surface via `policy`.
    fn apply_dividing_surface(
        &mut self,
        normal: ArrayView1<'_, f64>,
        policy: SurfaceVelocityPolicy,
    ) -> Result<f64, FluxError>;
}

/// Positions, velocities, periodic cell, thermostat, and holonomic constraint.
///
/// [`Self::restore`] writes those five pieces back. It does not rewind the
/// stochastic stream: [`DynamicsEngine::reseed`] replaces that stream, and
/// the trait has no engine-agnostic copy of it to put back.
pub(crate) struct EngineSnapshot {
    constraint: Option<HyperplaneConstraint>,
    thermostat: ThermostatConfig,
    cell: Option<[f64; 9]>,
    positions: Array1<f64>,
    velocities: Array1<f64>,
}

impl EngineSnapshot {
    pub(crate) fn capture<E: DynamicsEngine + ?Sized>(engine: &E) -> Self {
        Self {
            constraint: engine.constraint().cloned(),
            thermostat: *engine.thermostat(),
            cell: engine.cell().copied(),
            positions: engine.positions().to_owned(),
            velocities: engine.velocities().to_owned(),
        }
    }

    pub(crate) fn positions(&self) -> ArrayView1<'_, f64> {
        self.positions.view()
    }

    pub(crate) fn velocities(&self) -> ArrayView1<'_, f64> {
        self.velocities.view()
    }

    fn restore<E: DynamicsEngine + ?Sized>(self, engine: &mut E) -> Result<(), FluxError> {
        engine.set_cell(self.cell)?;
        engine.set_constraint(self.constraint)?;
        engine.set_thermostat(self.thermostat)?;
        engine.set_positions(self.positions.view())?;
        engine.set_velocities(self.velocities.view())?;
        Ok(())
    }

    pub(crate) fn restore_result<T, E: DynamicsEngine + ?Sized>(
        self,
        engine: &mut E,
        result: Result<T, FluxError>,
    ) -> Result<T, FluxError> {
        let restore = self.restore(engine);
        match (result, restore) {
            (Ok(value), Ok(())) => Ok(value),
            (Ok(_), Err(restore_err)) => Err(restore_err),
            (Err(operation_err), _) => Err(operation_err),
        }
    }
}

/// Draw Maxwell-Boltzmann velocities `v_i ~ N(0, k_B * T / m_i)` using `seed`.
pub fn draw_maxwell_boltzmann_velocities(
    masses: ArrayView1<'_, f64>,
    kbt: f64,
    seed: u64,
) -> Result<Array1<f64>, FluxError> {
    if masses.is_empty() {
        return Err(FluxError::EmptyInput("masses slice is empty"));
    }
    validate_masses(masses)?;
    if !kbt.is_finite() {
        return Err(FluxError::NonFinite("thermal energy k_B * T"));
    }
    if kbt < 0.0 {
        return Err(FluxError::InvalidConfig(
            "thermal energy k_B * T must be non-negative",
        ));
    }
    let mut rng = StdRng::seed_from_u64(seed);
    Ok(sample_maxwell_boltzmann_with_rng(masses, kbt, &mut rng))
}

/// Reflect `velocities` across a dividing surface with unit `normal` via the
/// kinetic-energy-preserving mass-weighted Householder reflection, returning `normal · v`.
pub fn reflect_velocity_in_place(
    velocities: &mut Array1<f64>,
    masses: ArrayView1<'_, f64>,
    normal: ArrayView1<'_, f64>,
) -> Result<f64, FluxError> {
    validate_masses(masses)?;
    if velocities.len() != masses.len() {
        return Err(FluxError::Shape {
            expected: masses.len(),
            got: velocities.len(),
            context: "velocities length",
        });
    }
    validate_finite_slice(velocities.view(), "velocities")?;
    validate_unit_normal(normal, masses.len())?;
    let z = inv_mass_metric_norm(normal, masses);
    let nv = dot(normal, velocities.view());
    let scale = 2.0 * nv / z;
    for i in 0..velocities.len() {
        velocities[i] -= scale * normal[i] / masses[i];
    }
    Ok(dot(normal, velocities.view()))
}

pub(crate) fn validate_finite_slice(
    values: ArrayView1<'_, f64>,
    context: &'static str,
) -> Result<(), FluxError> {
    for &v in values {
        if !v.is_finite() {
            return Err(FluxError::NonFinite(context));
        }
    }
    Ok(())
}

pub(crate) fn validate_masses(masses: ArrayView1<'_, f64>) -> Result<(), FluxError> {
    if masses.is_empty() {
        return Err(FluxError::EmptyInput("masses slice is empty"));
    }
    for &m in masses {
        if !m.is_finite() {
            return Err(FluxError::NonFinite("masses"));
        }
        if m <= 0.0 {
            return Err(FluxError::InvalidConfig("masses must be strictly positive"));
        }
    }
    Ok(())
}

pub(crate) fn validate_unit_normal(
    normal: ArrayView1<'_, f64>,
    expected_dof: usize,
) -> Result<(), FluxError> {
    if normal.len() != expected_dof {
        return Err(FluxError::Shape {
            expected: expected_dof,
            got: normal.len(),
            context: "dividing-surface normal",
        });
    }
    validate_finite_slice(normal, "dividing-surface normal")?;
    let norm = nrm2(normal);
    if (norm - 1.0).abs() > UNIT_NORMAL_TOL {
        return Err(FluxError::InvalidConfig(
            "dividing-surface normal must have unit Euclidean norm",
        ));
    }
    Ok(())
}

pub(crate) fn inv_mass_metric_norm(
    normal: ArrayView1<'_, f64>,
    masses: ArrayView1<'_, f64>,
) -> f64 {
    let mut z = 0.0_f64;
    for i in 0..normal.len() {
        z += (normal[i] * normal[i]) / masses[i];
    }
    z
}

/// Holonomic multiplier `lambda = -(n^T M^{-1} F) / (n^T M^{-1} n)`.
///
/// `F` is the physical force. The canonical mean of `lambda` is `dF/dc`
/// (Sprik and Ciccotti 1998). A global rescaling of `masses` cancels, so an
/// engine may pass either bare masses or the metal-unit dynamical masses.
pub(crate) fn constraint_lagrange_multiplier(
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

/// Project positions onto a hyperplane and reject numerically unresolved residuals.
pub(crate) fn shake_positions_in_place(
    positions: &mut Array1<f64>,
    masses: ArrayView1<'_, f64>,
    plane: &HyperplaneConstraint,
) -> Result<(), FluxError> {
    let z = inv_mass_metric_norm(plane.normal.view(), masses);
    // Two passes reduce the residual left by the first floating-point projection.
    for _ in 0..2 {
        let sigma = dot(plane.normal.view(), positions.view()) - plane.offset;
        let alpha = sigma / z;
        for i in 0..positions.len() {
            positions[i] -= alpha * plane.normal[i] / masses[i];
        }
    }
    let scale = 1.0 + plane.offset.abs() + nrm2(positions.view());
    let residual = (dot(plane.normal.view(), positions.view()) - plane.offset).abs() / scale;
    if !residual.is_finite() || residual > HYPERPLANE_PROJECTION_TOL {
        return Err(FluxError::ConstraintViolation {
            residual,
            tolerance: HYPERPLANE_PROJECTION_TOL,
        });
    }
    Ok(())
}

/// Project velocities onto a hyperplane's tangent space in the mass metric.
pub(crate) fn rattle_velocities_in_place(
    velocities: &mut Array1<f64>,
    masses: ArrayView1<'_, f64>,
    plane: &HyperplaneConstraint,
) {
    let z = inv_mass_metric_norm(plane.normal.view(), masses);
    for _ in 0..2 {
        let nv = dot(plane.normal.view(), velocities.view());
        let beta = nv / z;
        for i in 0..velocities.len() {
            velocities[i] -= beta * plane.normal[i] / masses[i];
        }
    }
}

pub(crate) fn box_muller_pair(rng: &mut StdRng) -> (f64, f64) {
    let u1 = 1.0 - rng.random::<f64>();
    let u2 = rng.random::<f64>();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

pub(crate) fn sample_maxwell_boltzmann_with_rng(
    masses: ArrayView1<'_, f64>,
    kbt: f64,
    rng: &mut StdRng,
) -> Array1<f64> {
    let n = masses.len();
    let mut v = Array1::zeros(n);
    if kbt == 0.0 {
        return v;
    }
    let mut i = 0_usize;
    while i < n {
        let (z0, z1) = box_muller_pair(rng);
        v[i] = (kbt / masses[i]).sqrt() * z0;
        if i + 1 < n {
            v[i + 1] = (kbt / masses[i + 1]).sqrt() * z1;
        }
        i += 2;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn _assert_object_safe(_engine: &mut dyn DynamicsEngine, _surface: &dyn ForceSurface) {}

    struct Quadratic1D;

    impl rgsaddle::PointSurface for Quadratic1D {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
            Ok((0.5 * x[0] * x[0], Array1::from_vec(vec![x[0]])))
        }
    }

    #[test]
    fn canonical_sampling_refuses_nve_only_for_white_noise() {
        let nve = ThermostatConfig::default().with_gamma(0.0);
        assert!(nve.require_canonical_sampling().is_err());
        let gle = nve.with_noise(gle::ColoredNoise::SmartSampling);
        assert_eq!(gle.require_canonical_sampling(), Ok(1.0));
    }

    #[test]
    fn point_surface_blanket_negates_gradient_into_force() {
        let surf = Quadratic1D;
        let x = Array1::from_vec(vec![2.5]);
        let (e, f) = ForceSurface::energy_and_forces(&surf, x.view(), None).expect("eval ok");
        assert_eq!(e, 3.125);
        assert_eq!(f[0], -2.5);
    }

    #[test]
    fn eindir_surface_adapts_rgmin_oracle() {
        let oracle = rgmin::Oracle::unbounded(2, |x| {
            let e = x[0] * x[0] + 2.0 * x[1] * x[1];
            let g = Array1::from_vec(vec![2.0 * x[0], 4.0 * x[1]]);
            (e, g)
        });
        let surf = EindirSurface::new(oracle);
        let x = Array1::from_vec(vec![1.5, -0.5]);
        let (e, f) = surf.energy_and_forces(x.view(), None).expect("oracle eval");
        assert_eq!(e, 2.75);
        assert_eq!(f[0], -3.0);
        assert_eq!(f[1], 2.0);
    }

    #[test]
    fn reflection_preserves_mass_weighted_kinetic_energy() {
        let masses = Array1::from_vec(vec![1.2, 2.7]);
        let mut v = Array1::from_vec(vec![-0.8, 0.35]);
        let plane =
            HyperplaneConstraint::from_direction(Array1::from_vec(vec![3.0, 4.0]).view(), 0.0)
                .expect("unit plane");
        let ke_before = 0.5 * (masses[0] * v[0] * v[0] + masses[1] * v[1] * v[1]);
        let nv_before = dot(plane.normal.view(), v.view());
        let nv_after =
            reflect_velocity_in_place(&mut v, masses.view(), plane.normal.view()).expect("reflect");
        let ke_after = 0.5 * (masses[0] * v[0] * v[0] + masses[1] * v[1] * v[1]);
        assert!((nv_after + nv_before).abs() <= UNIT_NORMAL_TOL);
        assert!((ke_after - ke_before).abs() <= UNIT_NORMAL_TOL);
    }
}
