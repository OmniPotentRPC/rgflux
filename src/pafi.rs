//! Projected Average Force Integration (PAFI) along minimum-energy paths.
//!
//! Integrates the curvature-corrected hyperplane mean force
//! `<(grad V · n)(1 - psi)>` across planes orthogonal to a converged band or
//! spline reaction path to obtain finite-temperature free-energy barriers
//! without a predefined collective variable.
//!
//! Note: implementation moves from rgsaddle.
//!
//! ## References
//!
//! - Goswami, R., Gunde, M., and Jónsson, H. (2026). *Enhanced climbing image
//!   nudged elastic band method with Hessian eigenmode alignment*. Frontiers
//!   in Chemistry, 14, 1807063. <https://doi.org/10.3389/fchem.2026.1807063>
//! - Goswami, R. (2026). *Reproducible orchestration of best practices for
//!   reaction path optimization with the nudged elastic band*. MethodsX, 16,
//!   103899. <https://doi.org/10.1016/j.mex.2026.103899>
//! - Swinburne, T. D., and Marinica, M.-C. (2018). *Unsupervised Calculation
//!   of Free Energy Barriers in Large Crystalline Systems*. Physical Review
//!   Letters, 120, 135503. <https://doi.org/10.1103/PhysRevLett.120.135503>

use std::f64::consts::TAU;

use ndarray::{Array1, ArrayView1, ArrayView2};
use rgmin::vecops::{dot, nrm2};

use crate::engine::{
    DynamicsEngine, EngineSnapshot, HyperplaneConstraint, UNIT_NORMAL_TOL, validate_finite_slice,
    validate_masses,
};
use crate::error::FluxError;
use crate::stats::{StatsConfig, batch_means};

/// Configuration for Projected Average Force Integration (PAFI).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct PafiConfig {
    /// Number of hyperplanes discretized along the reaction path (`>= 2`).
    pub planes: usize,
    /// Equilibration steps discarded on each hyperplane.
    pub burn_in_steps: usize,
    /// Production steps added per hyperplane and adaptive round (`>= 2`).
    pub sample_steps: usize,
    /// Integration timestep `dt > 0`.
    pub time_step: f64,
    /// Optional target standard error on the integrated barrier `Delta F`.
    pub target_error: Option<f64>,
    /// Maximum adaptive resampling rounds when `target_error` is set.
    pub max_rounds: usize,
    /// Relative tolerance on the zero-temperature path-force integral vs `Delta E_0`.
    pub zero_temperature_tol: f64,
    /// Deterministic PRNG seed.
    pub seed: u64,
    /// Batch-means uncertainty settings for per-plane forces.
    pub stats: StatsConfig,
}

impl Default for PafiConfig {
    fn default() -> Self {
        Self {
            planes: 9,
            burn_in_steps: 500,
            sample_steps: 4_000,
            time_step: 0.002,
            target_error: None,
            max_rounds: 4,
            zero_temperature_tol: 0.01,
            seed: 1,
            stats: StatsConfig::default(),
        }
    }
}

with_setters!(PafiConfig {
    with_planes => planes: val usize,
    with_burn_in_steps => burn_in_steps: val usize,
    with_sample_steps => sample_steps: val usize,
    with_time_step => time_step: val f64,
    with_target_error => target_error: opt f64,
    with_max_rounds => max_rounds: val usize,
    with_zero_temperature_tol => zero_temperature_tol: val f64,
    with_seed => seed: val u64,
    with_stats => stats: val StatsConfig,
});

/// Integrated free-energy profile along a discretized reaction path.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct FreeEnergyProfile {
    /// Normalized arc-length coordinates `s_k` along the path.
    pub arc_lengths: Vec<f64>,
    /// Integrated free energy `F(s_k) - F(s_0)` at each plane.
    pub free_energy: Vec<f64>,
    /// Propagated block-averaged standard error at each plane.
    pub standard_error: Vec<f64>,
    /// Free-energy barrier `Delta F = max_s F(s)` at dividing plane `s*`.
    pub barrier: f64,
    /// Standard error of [`Self::barrier`].
    pub barrier_error: f64,
    /// Index `s*` where [`Self::barrier`] is attained.
    pub dividing_index: usize,
    /// Reactant-side normalization
    /// `Z_R = 2 * \int_0^{s*} exp(-F(s) / k_B T) ds`.
    pub reactant_partition: f64,
    /// Normalized PAFI rate
    /// `k = sqrt(k_B T / (2 pi)) * exp(-F(s*) / k_B T) / Z_R`.
    pub normalized_rate: f64,
    /// Propagated standard error of [`Self::normalized_rate`].
    pub normalized_rate_error: f64,
}

#[derive(Clone, Debug)]
struct PlaneGeometry {
    anchor: Array1<f64>,
    normal: Array1<f64>,
    dn_ds: Array1<f64>,
    dc_ds: f64,
    tangent: Array1<f64>,
    arc: f64,
}

#[derive(Clone, Debug)]
struct PlaneEstimate {
    gradient: f64,
    standard_error: f64,
}

#[derive(Clone, Debug)]
struct NaturalSpline {
    knots: Vec<f64>,
    values: Vec<Array1<f64>>,
    second: Vec<Array1<f64>>,
}

impl NaturalSpline {
    fn new(
        path_images: ArrayView2<'_, f64>,
        masses: ArrayView1<'_, f64>,
    ) -> Result<Self, FluxError> {
        let rows = path_images.nrows();
        let dof = path_images.ncols();
        let mut knots = Vec::with_capacity(rows);
        knots.push(0.0);
        for i in 1..rows {
            let mut interval_squared = 0.0_f64;
            for j in 0..dof {
                let delta = path_images[[i, j]] - path_images[[i - 1, j]];
                interval_squared += masses[j] * delta * delta;
            }
            let interval = interval_squared.sqrt();
            if !interval.is_finite() || interval <= f64::MIN_POSITIVE {
                return Err(FluxError::InvalidConfig(
                    "PAFI path images must be distinct in mass-weighted arc length",
                ));
            }
            knots.push(knots[i - 1] + interval);
        }

        let values: Vec<Array1<f64>> = path_images
            .rows()
            .into_iter()
            .map(|row| row.to_owned())
            .collect();
        let mut second = vec![Array1::zeros(dof); rows];
        if rows > 2 {
            let interior = rows - 2;
            let mut lower = vec![0.0_f64; interior];
            let mut diagonal = vec![0.0_f64; interior];
            let mut upper = vec![0.0_f64; interior];
            let mut rhs = vec![Array1::zeros(dof); interior];
            for k in 0..interior {
                let i = k + 1;
                let left = knots[i] - knots[i - 1];
                let right = knots[i + 1] - knots[i];
                lower[k] = left;
                diagonal[k] = 2.0 * (left + right);
                upper[k] = right;
                for j in 0..dof {
                    rhs[k][j] = 6.0
                        * ((values[i + 1][j] - values[i][j]) / right
                            - (values[i][j] - values[i - 1][j]) / left);
                }
            }
            for k in 1..interior {
                let factor = lower[k] / diagonal[k - 1];
                diagonal[k] -= factor * upper[k - 1];
                for j in 0..dof {
                    rhs[k][j] -= factor * rhs[k - 1][j];
                }
            }
            for k in (0..interior).rev() {
                let i = k + 1;
                for j in 0..dof {
                    let next = if k + 1 < interior {
                        upper[k] * second[i + 1][j]
                    } else {
                        0.0
                    };
                    second[i][j] = (rhs[k][j] - next) / diagonal[k];
                }
            }
        }
        Ok(Self {
            knots,
            values,
            second,
        })
    }

    fn end(&self) -> f64 {
        self.knots[self.knots.len() - 1]
    }

    fn evaluate(&self, s: f64) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let last = self.knots.len() - 1;
        let mut segment = last - 1;
        for i in 0..last {
            if s <= self.knots[i + 1] {
                segment = i;
                break;
            }
        }
        let left = self.knots[segment];
        let right = self.knots[segment + 1];
        let width = right - left;
        let a = (right - s) / width;
        let b = (s - left) / width;
        let dof = self.values[segment].len();
        let mut position = Array1::zeros(dof);
        let mut tangent = Array1::zeros(dof);
        let mut curvature = Array1::zeros(dof);
        for j in 0..dof {
            let y0 = self.values[segment][j];
            let y1 = self.values[segment + 1][j];
            let m0 = self.second[segment][j];
            let m1 = self.second[segment + 1][j];
            position[j] = a * y0
                + b * y1
                + ((a * a * a - a) * m0 + (b * b * b - b) * m1) * width * width / 6.0;
            tangent[j] = (y1 - y0) / width
                + width * ((-3.0 * a * a + 1.0) * m0 + (3.0 * b * b - 1.0) * m1) / 6.0;
            curvature[j] = a * m0 + b * m1;
        }
        (position, tangent, curvature)
    }
}

fn build_plane_geometries(
    path_images: ArrayView2<'_, f64>,
    masses: ArrayView1<'_, f64>,
    planes: usize,
) -> Result<Vec<PlaneGeometry>, FluxError> {
    if path_images.nrows() < 2 {
        return Err(FluxError::InvalidConfig(
            "PAFI path requires at least two images",
        ));
    }
    if path_images.ncols() == 0 {
        return Err(FluxError::EmptyInput("PAFI path image coordinates"));
    }
    if planes < 2 {
        return Err(FluxError::InvalidConfig(
            "PafiConfig::planes must be at least 2",
        ));
    }
    validate_masses(masses)?;
    if masses.len() != path_images.ncols() {
        return Err(FluxError::Shape {
            expected: path_images.ncols(),
            got: masses.len(),
            context: "PAFI masses versus path coordinates",
        });
    }
    for row in path_images.rows() {
        validate_finite_slice(row, "PAFI path image")?;
    }

    let spline = NaturalSpline::new(path_images, masses)?;
    let total_arc = spline.end();
    if !total_arc.is_finite() || total_arc <= f64::MIN_POSITIVE {
        return Err(FluxError::InvalidConfig(
            "PAFI path total arc length must be positive",
        ));
    }

    let mut geometries = Vec::with_capacity(planes);
    for k in 0..planes {
        let arc = total_arc * (k as f64) / ((planes - 1) as f64);
        let (anchor, tangent, curvature) = spline.evaluate(arc);
        let mut normal_raw = Array1::zeros(path_images.ncols());
        let mut normal_derivative_raw = Array1::zeros(path_images.ncols());
        for j in 0..path_images.ncols() {
            normal_raw[j] = masses[j] * tangent[j];
            normal_derivative_raw[j] = masses[j] * curvature[j];
        }
        let normal_norm = nrm2(normal_raw.view());
        if !normal_norm.is_finite() || normal_norm <= f64::MIN_POSITIVE.sqrt() {
            return Err(FluxError::InvalidConfig("PAFI normal norm underflowed"));
        }
        let normal = normal_raw.mapv(|v| v / normal_norm);
        if (nrm2(normal.view()) - 1.0).abs() > UNIT_NORMAL_TOL {
            return Err(FluxError::InvalidConfig(
                "PAFI unit normal construction failed",
            ));
        }
        let projection = dot(normal.view(), normal_derivative_raw.view());
        let mut dn_ds = Array1::zeros(path_images.ncols());
        for j in 0..path_images.ncols() {
            dn_ds[j] = (normal_derivative_raw[j] - normal[j] * projection) / normal_norm;
        }
        let dc_ds = dot(normal.view(), tangent.view());
        if !dc_ds.is_finite() || dc_ds <= f64::MIN_POSITIVE.sqrt() {
            return Err(FluxError::InvalidConfig(
                "PAFI tangent projection to hyperplane normal is non-positive",
            ));
        }

        geometries.push(PlaneGeometry {
            anchor,
            normal,
            dn_ds,
            dc_ds,
            tangent,
            arc,
        });
    }
    Ok(geometries)
}

fn sample_plane<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    geometry: &PlaneGeometry,
    config: &PafiConfig,
    plane_index: usize,
    round: usize,
) -> Result<Vec<f64>, FluxError> {
    let offset = dot(geometry.normal.view(), geometry.anchor.view());
    let plane = HyperplaneConstraint::new(geometry.normal.clone(), offset)?;
    engine.set_constraint(Some(plane))?;
    engine.set_positions(geometry.anchor.view())?;
    let seed = config
        .seed
        .wrapping_add((plane_index as u64).wrapping_mul(4_099))
        .wrapping_add((round as u64).wrapping_mul(1_000_003));
    engine.shoot_velocities(seed)?;
    for _ in 0..config.burn_in_steps {
        let report = engine.step(config.time_step)?;
        let lambda = report.constraint_force.ok_or(FluxError::InvalidConfig(
            "PAFI requires constraint force reporting on constrained steps",
        ))?;
        if !lambda.is_finite() {
            return Err(FluxError::NonFinite("PAFI burn-in constraint force"));
        }
    }

    let mut values = Vec::with_capacity(config.sample_steps);
    for _ in 0..config.sample_steps {
        let report = engine.step(config.time_step)?;
        let lambda = report.constraint_force.ok_or(FluxError::InvalidConfig(
            "PAFI requires constraint force reporting on constrained steps",
        ))?;
        values.push(projected_force_sample(engine, geometry, lambda)?);
    }
    Ok(values)
}

fn projected_force_sample<E: DynamicsEngine + ?Sized>(
    engine: &E,
    geometry: &PlaneGeometry,
    lambda: f64,
) -> Result<f64, FluxError> {
    let positions = engine.positions();
    let mut jacobian_projection = 0.0_f64;
    for i in 0..engine.dof() {
        jacobian_projection += (positions[i] - geometry.anchor[i]) * geometry.dn_ds[i];
    }
    let value = lambda * (geometry.dc_ds - jacobian_projection);
    if !value.is_finite() {
        return Err(FluxError::NonFinite("PAFI projected force sample"));
    }
    Ok(value)
}

fn estimate_plane(samples: &[f64], config: &PafiConfig) -> Result<PlaneEstimate, FluxError> {
    let estimate = batch_means(samples, &config.stats)?;
    if !estimate.mean.is_finite() {
        return Err(FluxError::NonFinite("PAFI plane force mean"));
    }
    if !estimate.standard_error.is_finite() || estimate.standard_error < 0.0 {
        return Err(FluxError::InvalidConfig(
            "PAFI plane standard error must be finite and non-negative",
        ));
    }
    Ok(PlaneEstimate {
        gradient: estimate.mean,
        standard_error: estimate.standard_error,
    })
}

fn integrate_means(
    arc_lengths: &[f64],
    gradients: &[f64],
    kbt: f64,
) -> Result<(Vec<f64>, f64, f64, usize), FluxError> {
    let n = arc_lengths.len();
    if n < 2 || gradients.len() != n {
        return Err(FluxError::Shape {
            expected: n,
            got: gradients.len(),
            context: "PAFI mean-profile integration vectors",
        });
    }
    let mut free_energy = vec![0.0_f64; n];
    for i in 1..n {
        let ds = arc_lengths[i] - arc_lengths[i - 1];
        if !ds.is_finite() || ds <= 0.0 {
            return Err(FluxError::InvalidConfig(
                "PAFI arc lengths must strictly increase",
            ));
        }
        free_energy[i] = free_energy[i - 1] + 0.5 * (gradients[i - 1] + gradients[i]) * ds;
        if !free_energy[i].is_finite() {
            return Err(FluxError::NonFinite("PAFI integrated free energy"));
        }
    }
    let mut dividing = 0usize;
    for i in 1..n {
        if free_energy[i] > free_energy[dividing] {
            dividing = i;
        }
    }
    let barrier = free_energy[dividing];
    let mut zr_half = 0.0_f64;
    for i in 1..=dividing {
        let ds = arc_lengths[i] - arc_lengths[i - 1];
        let w0 = (-free_energy[i - 1] / kbt).exp();
        let w1 = (-free_energy[i] / kbt).exp();
        zr_half += 0.5 * ds * (w0 + w1);
    }
    let reactant_partition = 2.0 * zr_half;
    if !reactant_partition.is_finite() || reactant_partition <= 0.0 {
        return Err(FluxError::InvalidConfig(
            "PAFI reactant partition must be finite and positive",
        ));
    }
    Ok((free_energy, barrier, reactant_partition, dividing))
}

fn integrate_profile(
    arc_lengths: &[f64],
    gradients: &[f64],
    gradient_errors: &[f64],
    kbt: f64,
) -> Result<FreeEnergyProfile, FluxError> {
    let n = arc_lengths.len();
    if n < 2 || gradients.len() != n || gradient_errors.len() != n {
        return Err(FluxError::Shape {
            expected: n,
            got: gradients.len().max(gradient_errors.len()),
            context: "PAFI profile integration vectors",
        });
    }
    let (free_energy, barrier, reactant_partition, dividing) =
        integrate_means(arc_lengths, gradients, kbt)?;

    let mut integration_weights = vec![vec![0.0_f64; n]; n];
    for i in 1..n {
        let previous = integration_weights[i - 1].clone();
        integration_weights[i] = previous;
        let ds = arc_lengths[i] - arc_lengths[i - 1];
        integration_weights[i][i - 1] += 0.5 * ds;
        integration_weights[i][i] += 0.5 * ds;
    }
    for &sigma in gradient_errors {
        if !sigma.is_finite() || sigma < 0.0 {
            return Err(FluxError::InvalidConfig(
                "PAFI gradient standard errors must be finite and non-negative",
            ));
        }
    }
    let mut standard_error = vec![0.0_f64; n];
    for i in 1..n {
        let mut variance = 0.0_f64;
        for (weight, sigma) in integration_weights[i].iter().zip(gradient_errors) {
            variance += weight * weight * sigma * sigma;
        }
        standard_error[i] = variance.sqrt();
    }

    let mut d_zr_d_f = vec![0.0_f64; n];
    for i in 1..=dividing {
        let ds = arc_lengths[i] - arc_lengths[i - 1];
        let w0 = (-free_energy[i - 1] / kbt).exp();
        let w1 = (-free_energy[i] / kbt).exp();
        d_zr_d_f[i - 1] += -0.5 * ds * w0 / kbt;
        d_zr_d_f[i] += -0.5 * ds * w1 / kbt;
    }

    let barrier_error = standard_error[dividing];
    let normalized_rate = (kbt / TAU).sqrt() * (-barrier / kbt).exp() / reactant_partition;
    if !normalized_rate.is_finite() || normalized_rate <= 0.0 {
        return Err(FluxError::InvalidConfig(
            "PAFI normalized rate must be finite and positive",
        ));
    }
    let mut variance_ln_rate = 0.0_f64;
    for (gradient_index, sigma) in gradient_errors.iter().copied().enumerate() {
        let barrier_sensitivity = integration_weights[dividing][gradient_index];
        let mut partition_sensitivity = 0.0_f64;
        for profile_index in 0..=dividing {
            partition_sensitivity +=
                2.0 * d_zr_d_f[profile_index] * integration_weights[profile_index][gradient_index];
        }
        let log_rate_sensitivity =
            -barrier_sensitivity / kbt - partition_sensitivity / reactant_partition;
        variance_ln_rate += log_rate_sensitivity * log_rate_sensitivity * sigma * sigma;
    }
    let sigma_ln_k = variance_ln_rate.sqrt();
    if !sigma_ln_k.is_finite() || sigma_ln_k < 0.0 {
        return Err(FluxError::InvalidConfig(
            "PAFI normalized-rate logarithmic error must be finite and non-negative",
        ));
    }
    let normalized_rate_error = normalized_rate * sigma_ln_k;
    if !normalized_rate_error.is_finite() || normalized_rate_error < 0.0 {
        return Err(FluxError::InvalidConfig(
            "PAFI normalized-rate standard error must be finite and non-negative",
        ));
    }

    Ok(FreeEnergyProfile {
        arc_lengths: arc_lengths.to_vec(),
        free_energy,
        standard_error,
        barrier,
        barrier_error,
        dividing_index: dividing,
        reactant_partition,
        normalized_rate,
        normalized_rate_error,
    })
}

/// Integrate the PAFI free-energy profile and normalized rate along `path_images`.
pub fn pafi_profile<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    path_images: ArrayView2<'_, f64>,
    config: &PafiConfig,
) -> Result<FreeEnergyProfile, FluxError> {
    if !config.time_step.is_finite() || config.time_step <= 0.0 {
        return Err(FluxError::InvalidConfig(
            "PafiConfig::time_step must be finite and positive",
        ));
    }
    if config.burn_in_steps == 0 {
        return Err(FluxError::InvalidConfig(
            "PafiConfig::burn_in_steps must be at least 1",
        ));
    }
    if config.sample_steps < 2 {
        return Err(FluxError::InvalidConfig(
            "PafiConfig::sample_steps must be at least 2",
        ));
    }
    if config.max_rounds == 0 {
        return Err(FluxError::InvalidConfig(
            "PafiConfig::max_rounds must be at least 1",
        ));
    }
    if !config.zero_temperature_tol.is_finite() || config.zero_temperature_tol < 0.0 {
        return Err(FluxError::InvalidConfig(
            "PafiConfig::zero_temperature_tol must be finite and non-negative",
        ));
    }
    if let Some(target) = config.target_error
        && (!target.is_finite() || target < 0.0)
    {
        return Err(FluxError::InvalidConfig(
            "PafiConfig::target_error must be finite and non-negative",
        ));
    }

    let snapshot = EngineSnapshot::capture(engine);
    let result = (|| {
        let kbt = engine.thermostat().thermal_energy()?;
        if kbt <= 0.0 {
            return Err(FluxError::InvalidConfig(
                "PAFI requires positive thermal energy",
            ));
        }
        let masses = engine.masses().to_owned();
        let geometries = build_plane_geometries(path_images, masses.view(), config.planes)?;
        let arcs: Vec<f64> = geometries.iter().map(|geometry| geometry.arc).collect();

        engine.set_constraint(None)?;
        let mut path_gradients = Vec::with_capacity(geometries.len());
        let mut path_energies = Vec::with_capacity(geometries.len());
        for geometry in &geometries {
            engine.set_positions(geometry.anchor.view())?;
            let (energy, forces) = engine.energy_and_forces()?;
            path_energies.push(energy);
            path_gradients.push(-dot(geometry.tangent.view(), forces.view()));
        }
        let integrated_energy = if geometries.len() >= 3 && geometries.len() % 2 == 1 {
            let spacing = arcs[1] - arcs[0];
            let mut weighted = path_gradients[0] + path_gradients[path_gradients.len() - 1];
            for (index, gradient) in path_gradients[1..path_gradients.len() - 1]
                .iter()
                .copied()
                .enumerate()
            {
                let weight = if index % 2 == 0 { 4.0 } else { 2.0 };
                weighted += weight * gradient;
            }
            spacing * weighted / 3.0
        } else {
            let mut integral = 0.0_f64;
            for i in 1..geometries.len() {
                let ds = arcs[i] - arcs[i - 1];
                integral += 0.5 * ds * (path_gradients[i - 1] + path_gradients[i]);
            }
            integral
        };
        let endpoint_energy = path_energies[path_energies.len() - 1] - path_energies[0];
        let discrepancy = (integrated_energy - endpoint_energy).abs();
        let scale = 1.0_f64.max(endpoint_energy.abs());
        if discrepancy > config.zero_temperature_tol * scale {
            return Err(FluxError::InvalidConfig(
                "PAFI zero-temperature path-force quadrature exceeds tolerance",
            ));
        }

        let mut plane_samples = vec![Vec::new(); geometries.len()];
        let mut final_profile = None;
        for round in 0..config.max_rounds {
            for (plane_index, geometry) in geometries.iter().enumerate() {
                let samples = sample_plane(engine, geometry, config, plane_index, round)?;
                plane_samples[plane_index].extend(samples);
            }
            let estimates: Vec<PlaneEstimate> = plane_samples
                .iter()
                .map(|samples| estimate_plane(samples, config))
                .collect::<Result<_, _>>()?;
            let gradients: Vec<f64> = estimates.iter().map(|estimate| estimate.gradient).collect();
            let errors: Vec<f64> = estimates
                .iter()
                .map(|estimate| estimate.standard_error)
                .collect();
            let profile = integrate_profile(&arcs, &gradients, &errors, kbt)?;
            let target_reached = match config.target_error {
                Some(target) => profile.barrier_error <= target,
                None => true,
            };
            final_profile = Some(profile);
            if target_reached {
                break;
            }
        }
        match final_profile {
            Some(profile) => Ok(profile),
            None => Err(FluxError::InvalidConfig(
                "PAFI completed no sampling rounds",
            )),
        }
    })();
    snapshot.restore_result(engine, result)
}

/// Trait-object entry point for [`pafi_profile`] over [`DynamicsEngine`].
///
/// The algorithm path is identical to [`pafi_profile`], enabling direct
/// dispatch through `&mut dyn DynamicsEngine` for native and `rgpot`-backed
/// engines without code forks.
pub fn pafi_profile_dyn(
    engine: &mut dyn DynamicsEngine,
    path_images: ArrayView2<'_, f64>,
    config: &PafiConfig,
) -> Result<FreeEnergyProfile, FluxError> {
    pafi_profile(engine, path_images, config)
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, Array2, ArrayView1};

    use super::*;
    use crate::engine::native::{NativeConfig, NativeEngine};
    use crate::engine::{DynamicsEngine, ThermostatConfig};

    struct Quartic;

    impl rgsaddle::PointSurface for Quartic {
        fn eval(
            &self,
            x: ArrayView1<'_, f64>,
        ) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
            Ok((x[0].powi(4), Array1::from_vec(vec![4.0 * x[0].powi(3)])))
        }
    }

    #[test]
    fn mass_metric_sets_normal_and_multiplier_conversion() {
        let path = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).expect("shape");
        let masses = Array1::from_vec(vec![1.0, 4.0]);
        let geometry = build_plane_geometries(path.view(), masses.view(), 3).expect("geometry");
        let middle = &geometry[1];
        assert_abs_diff_eq!(
            middle.normal[1] / middle.normal[0],
            4.0,
            epsilon = f64::EPSILON
        );

        let mut engine = NativeEngine::new(
            Quartic,
            NativeConfig::default()
                .with_masses(masses)
                .with_thermostat(ThermostatConfig::default().with_temperature(1.0)),
        )
        .expect("engine");
        engine
            .set_positions(middle.anchor.view())
            .expect("set anchor");
        let projected = projected_force_sample(&engine, middle, 2.0).expect("projected force");
        assert_abs_diff_eq!(projected, 2.0 * middle.dc_ds, epsilon = f64::EPSILON);
        assert!((middle.dc_ds - 1.0).abs() > 0.1);
    }

    #[test]
    fn spline_geometry_has_distinct_endpoint_normals() {
        let path =
            Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.2, 1.4, 1.0]).expect("shape");
        let masses = Array1::from_vec(vec![1.0, 3.0]);
        let geometry = build_plane_geometries(path.view(), masses.view(), 7).expect("geometry");
        let difference = (&geometry[0].normal - &geometry[1].normal)
            .mapv(|value| value * value)
            .sum()
            .sqrt();
        assert!(difference > f64::EPSILON.sqrt());
    }

    #[test]
    fn profile_uses_interior_maximum_and_accepts_exact_errors() {
        let arcs = [0.0, 1.0, 2.0, 3.0];
        let gradients = [2.0, 0.0, -2.0, -2.0];
        let errors = [0.0; 4];
        let profile = integrate_profile(&arcs, &gradients, &errors, 1.0).expect("profile");
        assert_eq!(profile.dividing_index, 1);
        assert_eq!(profile.barrier, 1.0);
        assert_eq!(profile.barrier_error, 0.0);
        assert_eq!(profile.normalized_rate_error, 0.0);
    }

    #[test]
    fn dyn_entrypoint_restores_engine_after_error() {
        let mut engine = NativeEngine::new(
            Quartic,
            NativeConfig::default()
                .with_masses(Array1::from_vec(vec![2.0]))
                .with_thermostat(ThermostatConfig::default().with_temperature(1.0)),
        )
        .expect("engine");
        let original_positions = Array1::from_vec(vec![0.25]);
        let requested_velocities = Array1::from_vec(vec![-0.75]);
        let original_constraint =
            HyperplaneConstraint::from_direction(Array1::from_vec(vec![1.0]).view(), 0.25)
                .expect("constraint");
        engine
            .set_constraint(Some(original_constraint.clone()))
            .expect("constraint");
        engine
            .set_positions(original_positions.view())
            .expect("positions");
        engine
            .set_velocities(requested_velocities.view())
            .expect("velocities");
        let original_velocities = engine.velocities().to_owned();

        let path = Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).expect("shape");
        let config = PafiConfig::default()
            .with_planes(2)
            .with_burn_in_steps(1)
            .with_sample_steps(2)
            .with_zero_temperature_tol(0.0);
        let result = pafi_profile_dyn(&mut engine, path.view(), &config);
        assert_eq!(
            result,
            Err(FluxError::InvalidConfig(
                "PAFI zero-temperature path-force quadrature exceeds tolerance"
            ))
        );
        assert_eq!(engine.positions(), original_positions.view());
        assert_eq!(engine.velocities(), original_velocities.view());
        assert_eq!(engine.constraint(), Some(&original_constraint));
    }
}
