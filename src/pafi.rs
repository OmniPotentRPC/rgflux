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

use crate::engine::{DynamicsEngine, HyperplaneConstraint, validate_finite_slice, validate_masses};
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
    /// Legacy production steps field retained for API compatibility.
    ///
    /// Sampling now uses the fixed protocol requested by the method notes:
    /// pilot `n = 14`, then `ceil(1.1 * n * (se / 0.03)^2)`, capped at `2048`.
    pub sample_steps: usize,
    /// Integration timestep `dt > 0`.
    pub time_step: f64,
    /// Optional target standard error on the integrated barrier `Delta F`.
    pub target_error: Option<f64>,
    /// Maximum adaptive resampling rounds when `target_error` is set.
    pub max_rounds: usize,
    /// Tolerance on the zero-temperature path-force integral vs `Delta E_0`.
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
    /// Free-energy barrier `Delta F = F(s*)` at the final plane `s*`.
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

const PILOT_SAMPLES: usize = 14;
const TARGET_FORCE_SE: f64 = 0.03;
const MAX_PLANE_SAMPLES: usize = 2_048;

#[derive(Clone, Debug)]
struct PlaneGeometry {
    anchor: Array1<f64>,
    normal: Array1<f64>,
    dn_ds: Array1<f64>,
    dc_ds: f64,
    arc: f64,
}

#[derive(Clone, Debug)]
struct PlaneEstimate {
    gradient: f64,
    standard_error: f64,
    samples: Vec<f64>,
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
        return Err(FluxError::InvalidConfig("PafiConfig::planes must be at least 2"));
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

    let mut segment_lengths = Vec::with_capacity(path_images.nrows() - 1);
    let mut cumulative = Vec::with_capacity(path_images.nrows());
    cumulative.push(0.0_f64);
    for i in 1..path_images.nrows() {
        let mut ds2 = 0.0_f64;
        for j in 0..path_images.ncols() {
            let delta = path_images[[i, j]] - path_images[[i - 1, j]];
            ds2 += masses[j] * delta * delta;
        }
        let ds = ds2.sqrt();
        if !ds.is_finite() || ds <= f64::MIN_POSITIVE {
            return Err(FluxError::InvalidConfig(
                "PAFI path images must be distinct in mass-weighted arc length",
            ));
        }
        segment_lengths.push(ds);
        cumulative.push(cumulative[i - 1] + ds);
    }

    let total_arc = *cumulative
        .last()
        .ok_or(FluxError::InvalidConfig("PAFI path arc is empty"))?;
    if !total_arc.is_finite() || total_arc <= f64::MIN_POSITIVE {
        return Err(FluxError::InvalidConfig(
            "PAFI path total arc length must be positive",
        ));
    }

    let mut anchors = Vec::with_capacity(planes);
    let mut normals = Vec::with_capacity(planes);
    let mut arcs = Vec::with_capacity(planes);

    let mut seg = 0usize;
    for k in 0..planes {
        let target_arc = total_arc * (k as f64) / ((planes - 1) as f64);
        while seg + 1 < cumulative.len() - 1 && cumulative[seg + 1] < target_arc {
            seg += 1;
        }
        let s0 = cumulative[seg];
        let s1 = cumulative[seg + 1];
        let ds = s1 - s0;
        let t = if ds <= f64::MIN_POSITIVE {
            0.0
        } else {
            (target_arc - s0) / ds
        };

        let mut anchor = Array1::<f64>::zeros(path_images.ncols());
        let mut tangent = Array1::<f64>::zeros(path_images.ncols());
        let mut normal_raw = Array1::<f64>::zeros(path_images.ncols());
        for j in 0..path_images.ncols() {
            let x0 = path_images[[seg, j]];
            let x1 = path_images[[seg + 1, j]];
            anchor[j] = x0 + t * (x1 - x0);
            tangent[j] = (x1 - x0) / ds;
            normal_raw[j] = masses[j] * tangent[j];
        }
        let normal_norm = nrm2(normal_raw.view());
        if !normal_norm.is_finite() || normal_norm <= f64::MIN_POSITIVE.sqrt() {
            return Err(FluxError::InvalidConfig("PAFI normal norm underflowed"));
        }
        let normal = normal_raw.mapv(|v| v / normal_norm);
        anchors.push(anchor);
        normals.push(normal);
        arcs.push(target_arc);
    }

    let mut geometries = Vec::with_capacity(planes);
    for i in 0..planes {
        let mut dn_ds = Array1::<f64>::zeros(path_images.ncols());
        if i > 0 && i + 1 < planes {
            let ds = arcs[i + 1] - arcs[i - 1];
            for j in 0..path_images.ncols() {
                dn_ds[j] = (normals[i + 1][j] - normals[i - 1][j]) / ds;
            }
        } else if i + 1 < planes {
            let ds = arcs[i + 1] - arcs[i];
            for j in 0..path_images.ncols() {
                dn_ds[j] = (normals[i + 1][j] - normals[i][j]) / ds;
            }
        } else if i > 0 {
            let ds = arcs[i] - arcs[i - 1];
            for j in 0..path_images.ncols() {
                dn_ds[j] = (normals[i][j] - normals[i - 1][j]) / ds;
            }
        }

        let tangent_norm = nrm2(normals[i].view());
        if (tangent_norm - 1.0).abs() > 64.0 * f64::EPSILON {
            return Err(FluxError::InvalidConfig(
                "PAFI unit normal construction failed",
            ));
        }
        let mut tangent = Array1::<f64>::zeros(path_images.ncols());
        if i == planes - 1 {
            for j in 0..path_images.ncols() {
                tangent[j] = (anchors[i][j] - anchors[i - 1][j]) / (arcs[i] - arcs[i - 1]);
            }
        } else {
            for j in 0..path_images.ncols() {
                tangent[j] = (anchors[i + 1][j] - anchors[i][j]) / (arcs[i + 1] - arcs[i]);
            }
        }
        let dc_ds = dot(normals[i].view(), tangent.view());
        if !dc_ds.is_finite() || dc_ds <= f64::MIN_POSITIVE.sqrt() {
            return Err(FluxError::InvalidConfig(
                "PAFI tangent projection to hyperplane normal is non-positive",
            ));
        }

        geometries.push(PlaneGeometry {
            anchor: anchors[i].clone(),
            normal: normals[i].clone(),
            dn_ds,
            dc_ds,
            arc: arcs[i],
        });
    }
    Ok(geometries)
}

fn sample_plane<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    geometry: &PlaneGeometry,
    config: &PafiConfig,
    plane_index: usize,
) -> Result<PlaneEstimate, FluxError> {
    let offset = dot(geometry.normal.view(), geometry.anchor.view());
    let plane = HyperplaneConstraint::new(geometry.normal.clone(), offset)?;
    engine.set_constraint(Some(plane))?;
    engine.set_positions(geometry.anchor.view())?;
    engine.shoot_velocities(config.seed.wrapping_add((plane_index as u64).wrapping_mul(4099)))?;

    let blocks = 2usize;
    let mut values = Vec::with_capacity(blocks * MAX_PLANE_SAMPLES);
    let mut block_means = Vec::with_capacity(blocks);
    for block in 0..blocks {
        engine.set_positions(geometry.anchor.view())?;
        engine.shoot_velocities(
            config
                .seed
                .wrapping_add((plane_index as u64).wrapping_mul(4099))
                .wrapping_add((block as u64).wrapping_mul(1_000_003)),
        )?;
        for _ in 0..config.burn_in_steps {
            let report = engine.step(config.time_step)?;
            let lambda = report.constraint_force.ok_or(FluxError::InvalidConfig(
                "PAFI requires constraint force reporting on constrained steps",
            ))?;
            if !lambda.is_finite() {
                return Err(FluxError::NonFinite("PAFI burn-in constraint force"));
            }
        }

        let mut block_values = Vec::with_capacity(MAX_PLANE_SAMPLES);
        while block_values.len() < PILOT_SAMPLES {
            let report = engine.step(config.time_step)?;
            let lambda = report.constraint_force.ok_or(FluxError::InvalidConfig(
                "PAFI requires constraint force reporting on constrained steps",
            ))?;
            let mut displacement = Array1::<f64>::zeros(engine.dof());
            let positions = engine.positions();
            for i in 0..displacement.len() {
                displacement[i] = positions[i] - geometry.anchor[i];
            }
            let psi = dot(displacement.view(), geometry.dn_ds.view());
            let value = lambda * (1.0 - psi) * geometry.dc_ds;
            if !value.is_finite() {
                return Err(FluxError::NonFinite("PAFI projected force sample"));
            }
            block_values.push(value);
        }

        let pilot = batch_means(&block_values, &config.stats)?;
        if !pilot.standard_error.is_finite() || pilot.standard_error <= 0.0 {
            return Err(FluxError::InvalidConfig(
                "PAFI pilot standard error must be finite and positive",
            ));
        }
        let scaled =
            (1.1 * (PILOT_SAMPLES as f64) * (pilot.standard_error / TARGET_FORCE_SE).powi(2))
                .ceil() as usize;
        let target = scaled.max(PILOT_SAMPLES).min(MAX_PLANE_SAMPLES);
        while block_values.len() < target {
            let report = engine.step(config.time_step)?;
            let lambda = report.constraint_force.ok_or(FluxError::InvalidConfig(
                "PAFI requires constraint force reporting on constrained steps",
            ))?;
            let mut displacement = Array1::<f64>::zeros(engine.dof());
            let positions = engine.positions();
            for i in 0..displacement.len() {
                displacement[i] = positions[i] - geometry.anchor[i];
            }
            let psi = dot(displacement.view(), geometry.dn_ds.view());
            let value = lambda * (1.0 - psi) * geometry.dc_ds;
            if !value.is_finite() {
                return Err(FluxError::NonFinite("PAFI projected force sample"));
            }
            block_values.push(value);
        }

        let block_mean = block_values.iter().copied().sum::<f64>() / (block_values.len() as f64);
        if !block_mean.is_finite() {
            return Err(FluxError::NonFinite("PAFI block mean force"));
        }
        block_means.push(block_mean);
        values.extend(block_values);
    }
    let gradient = block_means.iter().copied().sum::<f64>() / (block_means.len() as f64);
    let mut var = 0.0_f64;
    for &mean in &block_means {
        let d = mean - gradient;
        var += d * d;
    }
    let standard_error = (var / ((block_means.len() - 1) as f64)).sqrt() / (block_means.len() as f64).sqrt();
    if !gradient.is_finite() {
        return Err(FluxError::NonFinite("PAFI plane force mean"));
    }
    if !standard_error.is_finite() || standard_error <= 0.0 {
        return Err(FluxError::InvalidConfig(
            "PAFI plane standard error must be finite and positive",
        ));
    }
    Ok(PlaneEstimate {
        gradient,
        standard_error,
        samples: values,
    })
}

fn integrate_means(arc_lengths: &[f64], gradients: &[f64], kbt: f64) -> Result<(Vec<f64>, f64, f64, usize), FluxError> {
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
    let dividing = n - 1;
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

    let mut standard_error = vec![0.0_f64; n];
    for i in 1..n {
        let mut weights = vec![0.0_f64; n];
        for k in 1..=i {
            let ds = arc_lengths[k] - arc_lengths[k - 1];
            weights[k - 1] += 0.5 * ds;
            weights[k] += 0.5 * ds;
        }
        let mut bound = 0.0_f64;
        for j in 0..=i {
            let sigma = gradient_errors[j];
            if !sigma.is_finite() || sigma <= 0.0 {
                return Err(FluxError::InvalidConfig(
                    "PAFI gradient standard errors must be finite and positive",
                ));
            }
            bound += weights[j].abs() * sigma;
        }
        standard_error[i] = bound;
    }

    let mut d_zr_d_f = vec![0.0_f64; n];
    for i in 1..=dividing {
        let ds = arc_lengths[i] - arc_lengths[i - 1];
        let w0 = (-free_energy[i - 1] / kbt).exp();
        let w1 = (-free_energy[i] / kbt).exp();
        d_zr_d_f[i - 1] += -0.5 * ds * w0 / kbt;
        d_zr_d_f[i] += -0.5 * ds * w1 / kbt;
    }

    let mut variance_zr = 0.0_f64;
    for j in 0..=dividing {
        let sensitivity = 2.0 * d_zr_d_f[j];
        variance_zr += sensitivity * sensitivity * standard_error[j] * standard_error[j];
    }
    let sigma_zr = variance_zr.sqrt();
    if !sigma_zr.is_finite() {
        return Err(FluxError::NonFinite("PAFI reactant partition standard error"));
    }

    let barrier_error = standard_error[dividing];
    let normalized_rate = (kbt / TAU).sqrt() * (-barrier / kbt).exp() / reactant_partition;
    if !normalized_rate.is_finite() || normalized_rate <= 0.0 {
        return Err(FluxError::InvalidConfig(
            "PAFI normalized rate must be finite and positive",
        ));
    }
    let sigma_ln_k = ((barrier_error / kbt).powi(2) + (sigma_zr / reactant_partition).powi(2))
        .sqrt();
    if !sigma_ln_k.is_finite() || sigma_ln_k <= 0.0 {
        return Err(FluxError::InvalidConfig(
            "PAFI normalized-rate logarithmic error must be finite and positive",
        ));
    }
    let normalized_rate_error = normalized_rate * sigma_ln_k;
    if !normalized_rate_error.is_finite() || normalized_rate_error <= 0.0 {
        return Err(FluxError::InvalidConfig(
            "PAFI normalized-rate standard error must be finite and positive",
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

    let kbt = engine.thermostat().thermal_energy()?;
    let masses = engine.masses().to_owned();
    let geometries = build_plane_geometries(path_images, masses.view(), config.planes)?;
    let saved_constraint = engine.constraint().cloned();

    let mut gradients = Vec::with_capacity(geometries.len());
    let mut errors = Vec::with_capacity(geometries.len());
    let mut plane_samples: Vec<Vec<f64>> = Vec::with_capacity(geometries.len());
    for (i, geometry) in geometries.iter().enumerate() {
        let estimate = sample_plane(engine, geometry, config, i)?;
        gradients.push(estimate.gradient);
        errors.push(estimate.standard_error);
        plane_samples.push(estimate.samples);
    }
    engine.set_constraint(saved_constraint)?;

    let arcs: Vec<f64> = geometries.iter().map(|g| g.arc).collect();
    let mut profile = integrate_profile(&arcs, &gradients, &errors, kbt)?;

    let min_samples = plane_samples
        .iter()
        .map(Vec::len)
        .min()
        .ok_or(FluxError::EmptyInput("PAFI plane samples"))?;
    if min_samples < 4 {
        return Err(FluxError::InvalidConfig(
            "PAFI needs at least four samples per plane for block averaging",
        ));
    }
    let mut block_count = 2usize;
    if block_count > min_samples / 2 {
        block_count = min_samples / 2;
    }
    let block_size = min_samples / block_count;
    if block_size < 2 {
        return Err(FluxError::InvalidConfig(
            "PAFI block size must be at least two samples",
        ));
    }

    let mut barrier_blocks = Vec::with_capacity(block_count);
    let mut rate_blocks = Vec::with_capacity(block_count);
    for b in 0..block_count {
        let start = b * block_size;
        let end = if b + 1 == block_count {
            block_count * block_size
        } else {
            (b + 1) * block_size
        };
        let mut block_gradients = Vec::with_capacity(plane_samples.len());
        for samples in &plane_samples {
            let mut acc = 0.0_f64;
            let mut n = 0usize;
            for &value in &samples[start..end] {
                acc += value;
                n += 1;
            }
            block_gradients.push(acc / (n as f64));
        }
        let (_free, barrier, zr, _dividing) = integrate_means(&arcs, &block_gradients, kbt)?;
        let rate = (kbt / TAU).sqrt() * (-barrier / kbt).exp() / zr;
        barrier_blocks.push(barrier);
        rate_blocks.push(rate);
    }

    let block_se = |values: &[f64], context: &'static str| -> Result<f64, FluxError> {
        let n = values.len();
        let mean = values.iter().copied().sum::<f64>() / (n as f64);
        let mut var = 0.0_f64;
        for &v in values {
            if !v.is_finite() {
                return Err(FluxError::NonFinite(context));
            }
            let d = v - mean;
            var += d * d;
        }
        let se = (var / ((n - 1) as f64)).sqrt() / (n as f64).sqrt();
        if !se.is_finite() || se <= 0.0 {
            return Err(FluxError::InvalidConfig(
                "PAFI block-averaged standard error must be finite and positive",
            ));
        }
        Ok(se)
    };
    profile.barrier_error = block_se(&barrier_blocks, "PAFI barrier block estimate")?;
    profile.normalized_rate_error = block_se(&rate_blocks, "PAFI rate block estimate")?;
    Ok(profile)
}
