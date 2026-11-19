//! Projected Average Force Integration (PAFI) along minimum-energy paths.
//!
//! Integrates the curvature-corrected hyperplane mean force
//! `<(grad V · n)(1 - psi)>` across planes orthogonal to a converged band or
//! spline reaction path to obtain finite-temperature free-energy barriers
//! without a predefined collective variable.
//!
//! Note: implementation moves from rgsaddle.
//!
//! ## Barrier and rate
//!
//! The plane means `g(s) = dF/ds` are integrated with a fourth-order Hermite
//! rule, and the dividing plane is the maximum of the interpolated profile.
//! With the first path image at the reactant minimum, `F(s_0) = 0`, and the
//! flux through the dividing plane of a mass-weighted unit normal is
//! `sqrt(k_B T / 2 pi) exp(-F* / k_B T)`. The reactant normalization is the
//! Gaussian integral over the mass-weighted reactant Hessian `H` and the unit
//! path normal `n` there, `Z_R = sqrt(2 pi k_B T) / omega_0` with
//! `omega_0^2 = 1 / (n^T H^-1 n)`, so
//! `k = (omega_0 / 2 pi) exp(-(F* - F(s_0)) / k_B T)`. In the harmonic limit
//! this is the Vineyard rate.
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

/// Plane placement and integration rule for the PAFI mean force.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub enum PafiIntegration {
    /// Evenly spaced planes with fourth-order Hermite integration.
    #[default]
    Hermite,
    /// Gauss--Legendre nodes on the path interval.
    GaussLegendre,
    /// Gauss--Jacobi nodes with the endpoint-linear weight
    /// `(s - s_0) (s_1 - s)`.
    GaussJacobi,
    /// Nested Clenshaw--Curtis planes, including both endpoints.
    ClenshawCurtis,
    /// Noisy Gaussian-process quadrature with active plane placement.
    Bayesian,
}

/// Configuration for Projected Average Force Integration (PAFI).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct PafiConfig {
    /// Number of hyperplanes discretized along the reaction path (`>= 2`).
    pub planes: usize,
    /// Plane placement and integration rule.
    pub integration: PafiIntegration,
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
    /// Tolerance on the zero-temperature path-force integral versus `Delta E`
    /// at every plane, relative to the largest `|E_k - E_0|` along the path.
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
            integration: PafiIntegration::default(),
            burn_in_steps: 500,
            sample_steps: 4_000,
            time_step: 0.002,
            target_error: None,
            max_rounds: 4,
            zero_temperature_tol: 1.0e-3,
            seed: 1,
            stats: StatsConfig::default(),
        }
    }
}

with_setters!(PafiConfig {
    with_planes => planes: val usize,
    with_integration => integration: val PafiIntegration,
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
    /// Mass-weighted arc-length coordinates `s_k` along the path.
    pub arc_lengths: Vec<f64>,
    /// Integrated free energy `F(s_k) - F(s_0)` at each plane.
    pub free_energy: Vec<f64>,
    /// Propagated block-averaged standard error at each plane.
    pub standard_error: Vec<f64>,
    /// Free-energy barrier `Delta F = F(s*) - F(s_0)` at the dividing plane,
    /// the maximum of the interpolated profile.
    pub barrier: f64,
    /// Standard error of [`Self::barrier`].
    pub barrier_error: f64,
    /// Index of the sampled plane with the highest `F`, adjacent to `s*`.
    pub dividing_index: usize,
    /// Arc length `s*` of the dividing plane (not restricted to a sampled plane).
    pub dividing_arc: f64,
    /// Projected reactant frequency `omega_0 = (n^T H^-1 n)^(-1/2)` from the
    /// mass-weighted reactant Hessian `H` and the unit path normal `n` at the
    /// first image.
    pub reactant_frequency: f64,
    /// Gaussian reactant normalization
    /// `Z_R = \int exp(-(F(s) - F(s_0)) / k_B T) ds = sqrt(2 pi k_B T) / omega_0`.
    pub reactant_partition: f64,
    /// Normalized PAFI rate
    /// `k = sqrt(k_B T / (2 pi)) * exp(-Delta F / k_B T) / Z_R
    ///    = (omega_0 / 2 pi) * exp(-Delta F / k_B T)`.
    pub normalized_rate: f64,
    /// Propagated standard error of [`Self::normalized_rate`] (the reactant
    /// Hessian is deterministic, so only `Delta F` contributes).
    pub normalized_rate_error: f64,
    /// Largest `|path-force integral - Delta E|` over the planes at zero
    /// temperature, evaluated with the same quadrature as [`Self::barrier`].
    /// It is the deterministic discretization bias in the profile at this
    /// plane count and should sit well below [`Self::barrier_error`].
    pub quadrature_residual: f64,
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
        if rows == 3 {
            let left = knots[1] - knots[0];
            let right = knots[2] - knots[1];
            for j in 0..dof {
                let curvature = 2.0
                    * ((values[2][j] - values[1][j]) / right
                        - (values[1][j] - values[0][j]) / left)
                    / (left + right);
                for endpoint in &mut second {
                    endpoint[j] = curvature;
                }
            }
        } else if rows > 3 {
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
            let first = knots[1] - knots[0];
            let second_width = knots[2] - knots[1];
            diagonal[0] += first * (first + second_width) / second_width;
            upper[0] -= first * first / second_width;

            let penultimate = knots[rows - 2] - knots[rows - 3];
            let last = knots[rows - 1] - knots[rows - 2];
            lower[interior - 1] -= last * last / penultimate;
            diagonal[interior - 1] += last * (penultimate + last) / penultimate;

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
            for j in 0..dof {
                second[0][j] =
                    ((first + second_width) * second[1][j] - first * second[2][j]) / second_width;
                second[rows - 1][j] = ((penultimate + last) * second[rows - 2][j]
                    - last * second[rows - 3][j])
                    / penultimate;
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

/// Number of grid points in the Lagrange window that estimates the
/// derivative of the mean force at each plane.
const DERIVATIVE_WINDOW: usize = 5;

/// Scan resolution used to bracket the maximum of the interpolated profile
/// inside one plane interval before bisection refines it.
const MAXIMUM_SCAN_POINTS: usize = 128;

/// Fourth-order quadrature of the plane mean forces.
///
/// Each interval uses the cubic Hermite interpolant of `g = dF/ds`, whose
/// derivatives at the planes come from Lagrange differentiation over a window
/// of neighbouring planes. The integral over `[s_k, s_{k+1}]` is
/// `h/2 (g_k + g_{k+1}) + h^2/12 (g'_k - g'_{k+1})`, which is exact for cubic
/// `g` and has `O(h^4)` global error, against `O(h^2)` for the trapezoid rule
/// (`h^2/12 (g'(b) - g'(a))`, a fixed free-energy bias at fixed plane count).
/// Everything is linear in the plane means, so the weight matrices propagate
/// the plane standard errors exactly.
#[derive(Clone, Debug)]
struct Quadrature {
    arcs: Vec<f64>,
    derivative: Vec<Vec<f64>>,
    cumulative: Vec<Vec<f64>>,
}

#[derive(Clone, Debug)]
struct DividingPlane {
    index: usize,
    arc: f64,
    free_energy: f64,
    weights: Vec<f64>,
}

fn lagrange_derivative_weights(nodes: &[f64], at: usize) -> Vec<f64> {
    let mut weights = vec![0.0_f64; nodes.len()];
    for j in 0..nodes.len() {
        if j == at {
            weights[j] = (0..nodes.len())
                .filter(|&m| m != at)
                .map(|m| 1.0 / (nodes[at] - nodes[m]))
                .sum();
        } else {
            let mut numerator = 1.0_f64;
            let mut denominator = 1.0_f64;
            for m in 0..nodes.len() {
                if m != j {
                    denominator *= nodes[j] - nodes[m];
                    if m != at {
                        numerator *= nodes[at] - nodes[m];
                    }
                }
            }
            weights[j] = numerator / denominator;
        }
    }
    weights
}

impl Quadrature {
    fn new(arcs: &[f64]) -> Result<Self, FluxError> {
        let n = arcs.len();
        if n < 2 {
            return Err(FluxError::Shape {
                expected: 2,
                got: n,
                context: "PAFI quadrature arc lengths",
            });
        }
        for pair in arcs.windows(2) {
            let ds = pair[1] - pair[0];
            if !ds.is_finite() || ds <= 0.0 {
                return Err(FluxError::InvalidConfig(
                    "PAFI arc lengths must strictly increase",
                ));
            }
        }
        let window = n.min(DERIVATIVE_WINDOW);
        let mut derivative = vec![vec![0.0_f64; n]; n];
        for (i, row) in derivative.iter_mut().enumerate() {
            let start = i.saturating_sub(window / 2).min(n - window);
            let local = lagrange_derivative_weights(&arcs[start..start + window], i - start);
            for (offset, weight) in local.into_iter().enumerate() {
                row[start + offset] = weight;
            }
        }
        let mut cumulative = vec![vec![0.0_f64; n]; n];
        for k in 0..n - 1 {
            let h = arcs[k + 1] - arcs[k];
            let mut previous = cumulative[k].clone();
            previous[k] += 0.5 * h;
            previous[k + 1] += 0.5 * h;
            for j in 0..n {
                previous[j] += h * h / 12.0 * (derivative[k][j] - derivative[k + 1][j]);
            }
            cumulative[k + 1] = previous;
        }
        Ok(Self {
            arcs: arcs.to_vec(),
            derivative,
            cumulative,
        })
    }

    fn len(&self) -> usize {
        self.arcs.len()
    }

    fn integrate(&self, gradients: &[f64]) -> Vec<f64> {
        self.cumulative
            .iter()
            .map(|row| row.iter().zip(gradients).map(|(w, g)| w * g).sum())
            .collect()
    }

    fn slopes(&self, gradients: &[f64]) -> Vec<f64> {
        self.derivative
            .iter()
            .map(|row| row.iter().zip(gradients).map(|(w, g)| w * g).sum())
            .collect()
    }

    /// Cumulative-integral weights of `F(s_k + u h_k)`, `u` in `[0, 1]`.
    fn interval_weights(&self, k: usize, u: f64) -> Vec<f64> {
        let h = self.arcs[k + 1] - self.arcs[k];
        let u2 = u * u;
        let u3 = u2 * u;
        let u4 = u2 * u2;
        let i00 = u - u3 + 0.5 * u4;
        let i01 = u3 - 0.5 * u4;
        let i10 = 0.5 * u2 - 2.0 * u3 / 3.0 + 0.25 * u4;
        let i11 = 0.25 * u4 - u3 / 3.0;
        let mut weights = self.cumulative[k].clone();
        weights[k] += h * i00;
        weights[k + 1] += h * i01;
        for (j, weight) in weights.iter_mut().enumerate() {
            *weight += h * h * (i10 * self.derivative[k][j] + i11 * self.derivative[k + 1][j]);
        }
        weights
    }

    /// Maximum of the interpolated profile, searched on the two intervals
    /// that touch the highest plane.
    fn dividing_plane(&self, gradients: &[f64], free_energy: &[f64]) -> DividingPlane {
        let n = self.len();
        let mut index = 0usize;
        for i in 1..n {
            if free_energy[i] > free_energy[index] {
                index = i;
            }
        }
        let slopes = self.slopes(gradients);
        let mut best: Option<(usize, f64, f64)> = None;
        let mut candidates = Vec::with_capacity(2);
        if index > 0 {
            candidates.push(index - 1);
        }
        if index + 1 < n {
            candidates.push(index);
        }
        for k in candidates {
            let h = self.arcs[k + 1] - self.arcs[k];
            let profile = |u: f64| {
                let u2 = u * u;
                let u3 = u2 * u;
                let u4 = u2 * u2;
                free_energy[k]
                    + h * (gradients[k] * (u - u3 + 0.5 * u4) + gradients[k + 1] * (u3 - 0.5 * u4))
                    + h * h
                        * (slopes[k] * (0.5 * u2 - 2.0 * u3 / 3.0 + 0.25 * u4)
                            + slopes[k + 1] * (0.25 * u4 - u3 / 3.0))
            };
            let force = |u: f64| {
                let u2 = u * u;
                let u3 = u2 * u;
                gradients[k] * (2.0 * u3 - 3.0 * u2 + 1.0)
                    + gradients[k + 1] * (-2.0 * u3 + 3.0 * u2)
                    + h * (slopes[k] * (u3 - 2.0 * u2 + u) + slopes[k + 1] * (u3 - u2))
            };
            let mut scan_best = 0usize;
            let mut scan_value = profile(0.0);
            for j in 1..=MAXIMUM_SCAN_POINTS {
                let value = profile(j as f64 / MAXIMUM_SCAN_POINTS as f64);
                if value > scan_value {
                    scan_value = value;
                    scan_best = j;
                }
            }
            let mut u = scan_best as f64 / MAXIMUM_SCAN_POINTS as f64;
            if scan_best > 0 && scan_best < MAXIMUM_SCAN_POINTS {
                let mut lo = (scan_best - 1) as f64 / MAXIMUM_SCAN_POINTS as f64;
                let mut hi = (scan_best + 1) as f64 / MAXIMUM_SCAN_POINTS as f64;
                if force(lo) > 0.0 && force(hi) < 0.0 {
                    for _ in 0..80 {
                        let mid = 0.5 * (lo + hi);
                        if force(mid) > 0.0 {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }
                    u = 0.5 * (lo + hi);
                }
            }
            let value = profile(u);
            if best.is_none_or(|(_, _, current)| value > current) {
                best = Some((k, u, value));
            }
        }
        match best {
            Some((k, u, value)) => DividingPlane {
                index,
                arc: self.arcs[k] + u * (self.arcs[k + 1] - self.arcs[k]),
                free_energy: value,
                weights: self.interval_weights(k, u),
            },
            None => DividingPlane {
                index,
                arc: self.arcs[index],
                free_energy: free_energy[index],
                weights: self.cumulative[index].clone(),
            },
        }
    }
}

fn integrate_profile(
    quadrature: &Quadrature,
    gradients: &[f64],
    gradient_errors: &[f64],
    kbt: f64,
    reactant_curvature: f64,
) -> Result<FreeEnergyProfile, FluxError> {
    let n = quadrature.len();
    if gradients.len() != n || gradient_errors.len() != n {
        return Err(FluxError::Shape {
            expected: n,
            got: gradients.len().max(gradient_errors.len()),
            context: "PAFI profile integration vectors",
        });
    }
    for &sigma in gradient_errors {
        if !sigma.is_finite() || sigma < 0.0 {
            return Err(FluxError::InvalidConfig(
                "PAFI gradient standard errors must be finite and non-negative",
            ));
        }
    }
    if !reactant_curvature.is_finite() || reactant_curvature <= 0.0 {
        return Err(FluxError::InvalidConfig(
            "PAFI reactant curvature must be finite and positive",
        ));
    }
    let free_energy = quadrature.integrate(gradients);
    if free_energy.iter().any(|value| !value.is_finite()) {
        return Err(FluxError::NonFinite("PAFI integrated free energy"));
    }
    let dividing = quadrature.dividing_plane(gradients, &free_energy);
    let propagate = |weights: &[f64]| {
        weights
            .iter()
            .zip(gradient_errors)
            .map(|(weight, sigma)| weight * weight * sigma * sigma)
            .sum::<f64>()
            .sqrt()
    };
    let standard_error: Vec<f64> = quadrature
        .cumulative
        .iter()
        .map(|weights| propagate(weights))
        .collect();
    let barrier = dividing.free_energy;
    let barrier_error = propagate(&dividing.weights);

    // With F(0) = 0 and a Gaussian reactant well F ~ (omega_0^2 / 2) s^2,
    // Z_R = sqrt(2 pi kT) / omega_0, so the flux sqrt(kT / 2 pi) e^{-F*/kT} / Z_R
    // is (omega_0 / 2 pi) e^{-F*/kT}.
    let reactant_frequency = reactant_curvature.sqrt();
    let reactant_partition = (TAU * kbt).sqrt() / reactant_frequency;
    let normalized_rate = reactant_frequency / TAU * (-barrier / kbt).exp();
    if !normalized_rate.is_finite() || normalized_rate <= 0.0 {
        return Err(FluxError::InvalidConfig(
            "PAFI normalized rate must be finite and positive",
        ));
    }
    let normalized_rate_error = normalized_rate * barrier_error / kbt;
    if !normalized_rate_error.is_finite() || normalized_rate_error < 0.0 {
        return Err(FluxError::InvalidConfig(
            "PAFI normalized-rate standard error must be finite and non-negative",
        ));
    }

    Ok(FreeEnergyProfile {
        arc_lengths: quadrature.arcs.clone(),
        free_energy,
        standard_error,
        barrier,
        barrier_error,
        dividing_index: dividing.index,
        dividing_arc: dividing.arc,
        reactant_frequency,
        reactant_partition,
        normalized_rate,
        normalized_rate_error,
        quadrature_residual: 0.0,
    })
}

/// Squared projected reactant frequency `omega_0^2 = 1 / (n^T H^-1 n)`.
///
/// `H` is the mass-weighted Hessian at the first path image and `n` the unit
/// mass-weighted path tangent there, so `omega_0^2` is the curvature of the
/// reactant free-energy profile along the first plane normal. The linear
/// system `H u = n` is solved by conjugate gradients with Hessian-vector
/// products from central differences of the engine forces, which keeps the
/// computation generic over [`DynamicsEngine`] and matrix-free.
fn reactant_curvature<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    geometry: &PlaneGeometry,
) -> Result<f64, FluxError> {
    let dof = engine.dof();
    let root_mass: Array1<f64> = engine.masses().mapv(f64::sqrt);
    let origin = geometry.anchor.clone();
    let reach = origin.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let step = f64::EPSILON.cbrt() * (1.0 + reach);

    let mut normal = Array1::zeros(dof);
    for i in 0..dof {
        normal[i] = root_mass[i] * geometry.tangent[i];
    }
    let normal_norm = nrm2(normal.view());
    if !normal_norm.is_finite() || normal_norm <= f64::MIN_POSITIVE.sqrt() {
        return Err(FluxError::InvalidConfig("PAFI reactant normal underflowed"));
    }
    normal.mapv_inplace(|v| v / normal_norm);

    let mut apply = |direction: &Array1<f64>| -> Result<Array1<f64>, FluxError> {
        let mut displacement = Array1::zeros(dof);
        for i in 0..dof {
            displacement[i] = direction[i] / root_mass[i];
        }
        let length = nrm2(displacement.view());
        let scale = step / length;
        let mut gradients = Vec::with_capacity(2);
        for sign in [1.0_f64, -1.0] {
            let probe = &origin + &displacement.mapv(|v| sign * scale * v);
            engine.set_positions(probe.view())?;
            let (_, forces) = engine.energy_and_forces()?;
            gradients.push(forces.mapv(|f| -f));
        }
        // (g+ - g-) / (2 scale) = H d, then the mass-weighted H_y v = M^-1/2 H d.
        let mut product = Array1::zeros(dof);
        for i in 0..dof {
            product[i] = (gradients[0][i] - gradients[1][i]) / (2.0 * scale) / root_mass[i];
        }
        Ok(product)
    };

    let tolerance = f64::EPSILON.sqrt();
    let mut solution = Array1::<f64>::zeros(dof);
    let mut residual = normal.clone();
    let mut search = residual.clone();
    let mut residual_sq = dot(residual.view(), residual.view());
    for _ in 0..(2 * dof).max(8) {
        if residual_sq.sqrt() <= tolerance {
            break;
        }
        let image = apply(&search)?;
        let curvature = dot(search.view(), image.view());
        if !curvature.is_finite() || curvature <= 0.0 {
            return Err(FluxError::InvalidConfig(
                "PAFI reactant Hessian is not positive definite along the path normal",
            ));
        }
        let alpha = residual_sq / curvature;
        solution.scaled_add(alpha, &search);
        residual.scaled_add(-alpha, &image);
        let next_sq = dot(residual.view(), residual.view());
        let beta = next_sq / residual_sq;
        residual_sq = next_sq;
        search = &residual + &search.mapv(|v| beta * v);
    }
    if residual_sq.sqrt() > tolerance {
        return Err(FluxError::InvalidConfig(
            "PAFI reactant Hessian solve did not converge",
        ));
    }
    let inverse_curvature = dot(normal.view(), solution.view());
    if !inverse_curvature.is_finite() || inverse_curvature <= 0.0 {
        return Err(FluxError::InvalidConfig(
            "PAFI reactant curvature along the path normal must be positive",
        ));
    }
    Ok(1.0 / inverse_curvature)
}

/// Integrate the PAFI free-energy profile and normalized rate along `path_images`.
///
/// The first image must be the reactant minimum: the rate normalization uses
/// the reactant Hessian there, evaluated by central differences of
/// [`DynamicsEngine::energy_and_forces`] at displaced positions, so the engine
/// is queried at `O(iterations)` extra geometries before sampling starts. The
/// returned [`FreeEnergyProfile::quadrature_residual`] is the deterministic
/// integration bias at this plane count and should be small against
/// [`FreeEnergyProfile::barrier_error`].
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
        let quadrature = Quadrature::new(&arcs)?;
        let integrated_energy = quadrature.integrate(&path_gradients);
        let energy_scale = path_energies.iter().fold(0.0_f64, |acc, energy| {
            acc.max((energy - path_energies[0]).abs())
        });
        let quadrature_residual = integrated_energy
            .iter()
            .zip(&path_energies)
            .fold(0.0_f64, |acc, (integral, energy)| {
                acc.max((integral - (energy - path_energies[0])).abs())
            });
        if !quadrature_residual.is_finite()
            || quadrature_residual > config.zero_temperature_tol * energy_scale
        {
            return Err(FluxError::InvalidConfig(
                "PAFI zero-temperature path-force quadrature exceeds tolerance",
            ));
        }
        let curvature = reactant_curvature(engine, &geometries[0])?;

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
            let mut profile = integrate_profile(&quadrature, &gradients, &errors, kbt, curvature)?;
            profile.quadrature_residual = quadrature_residual;
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

    fn circular_path_in_mass_coordinates(
        images: usize,
        radius: f64,
        theta_start: f64,
        theta_end: f64,
        masses: &Array1<f64>,
    ) -> Array2<f64> {
        let mut path = Array2::zeros((images, 2));
        for i in 0..images {
            let fraction = (i as f64) / ((images - 1) as f64);
            let theta = theta_start + fraction * (theta_end - theta_start);
            path[[i, 0]] = radius * theta.cos() / masses[0].sqrt();
            path[[i, 1]] = radius * theta.sin() / masses[1].sqrt();
        }
        path
    }

    fn circular_normal_derivative(
        radius: f64,
        theta: f64,
        direction: f64,
        masses: &Array1<f64>,
    ) -> Array1<f64> {
        let tangent_mass =
            Array1::from_vec(vec![direction * -theta.sin(), direction * theta.cos()]);
        let curvature_mass = Array1::from_vec(vec![-theta.cos() / radius, -theta.sin() / radius]);
        let mut raw_normal = Array1::zeros(2);
        let mut raw_derivative = Array1::zeros(2);
        for i in 0..2 {
            raw_normal[i] = masses[i].sqrt() * tangent_mass[i];
            raw_derivative[i] = masses[i].sqrt() * curvature_mass[i];
        }
        let norm = nrm2(raw_normal.view());
        let normal = raw_normal.mapv(|value| value / norm);
        let projection = dot(normal.view(), raw_derivative.view());
        raw_derivative.mapv(|value| value / norm) - normal.mapv(|value| value * projection / norm)
    }

    #[test]
    fn circular_spline_preserves_endpoint_normal_curvature() {
        let images = 33usize;
        let radius = 1.3_f64;
        let theta_start = std::f64::consts::FRAC_PI_2;
        let theta_end = 0.0_f64;
        let masses = Array1::from_vec(vec![1.0, 4.0]);
        let path =
            circular_path_in_mass_coordinates(images, radius, theta_start, theta_end, &masses);
        let geometry =
            build_plane_geometries(path.view(), masses.view(), images).expect("geometry");
        let direction = (theta_end - theta_start).signum();
        let tolerance = 2.0 / (images as f64);
        for (index, theta) in [(0, theta_start), (images - 1, theta_end)] {
            let expected = circular_normal_derivative(radius, theta, direction, &masses);
            let error = (&geometry[index].dn_ds - &expected)
                .mapv(|value| value * value)
                .sum()
                .sqrt();
            assert!(
                error < tolerance,
                "endpoint {index} normal-curvature error {error} exceeds {tolerance}"
            );
        }
    }

    #[test]
    fn profile_uses_interior_maximum_and_accepts_exact_errors() {
        // The four means lie on the cubic g(s) = 2 - 4s/3 - s^2 + s^3/3, which
        // the Hermite quadrature integrates exactly: F(s) = 2s - 2s^2/3 -
        // s^3/3 + s^4/12 peaks at s = 1 with F = 13/12.
        let arcs = [0.0, 1.0, 2.0, 3.0];
        let gradients = [2.0, 0.0, -2.0, -2.0];
        let errors = [0.0; 4];
        let quadrature = Quadrature::new(&arcs).expect("quadrature");
        let profile =
            integrate_profile(&quadrature, &gradients, &errors, 1.0, 4.0).expect("profile");
        assert_eq!(profile.dividing_index, 1);
        assert_abs_diff_eq!(profile.dividing_arc, 1.0, epsilon = 1.0e-12);
        assert_abs_diff_eq!(profile.barrier, 13.0 / 12.0, epsilon = 1.0e-12);
        assert_eq!(profile.barrier_error, 0.0);
        assert_eq!(profile.normalized_rate_error, 0.0);
    }

    #[test]
    fn quadrature_integrates_cubics_exactly_and_beats_the_trapezoid_rule() {
        let arcs: Vec<f64> = (0..6).map(|k| 0.4 * f64::from(k)).collect();
        let quadrature = Quadrature::new(&arcs).expect("quadrature");
        let cubic = |s: f64| 1.0 - 2.0 * s + 0.5 * s * s + 0.75 * s * s * s;
        let integral = |s: f64| s - s * s + s * s * s / 6.0 + 0.1875 * s * s * s * s;
        let gradients: Vec<f64> = arcs.iter().map(|&s| cubic(s)).collect();
        for (value, &s) in quadrature.integrate(&gradients).iter().zip(&arcs) {
            assert_abs_diff_eq!(*value, integral(s), epsilon = 1.0e-12);
        }

        let smooth = |s: f64| (2.0 * s).cos();
        let exact = |s: f64| 0.5 * (2.0 * s).sin();
        let coarse: Vec<f64> = (0..9).map(|k| 0.25 * f64::from(k)).collect();
        let gradients: Vec<f64> = coarse.iter().map(|&s| smooth(s)).collect();
        let quadrature = Quadrature::new(&coarse).expect("quadrature");
        let hermite = quadrature.integrate(&gradients)[8];
        let mut trapezoid = 0.0;
        for k in 1..coarse.len() {
            trapezoid += 0.5 * (coarse[k] - coarse[k - 1]) * (gradients[k - 1] + gradients[k]);
        }
        let target = exact(coarse[8]);
        assert!((hermite - target).abs() < 0.05 * (trapezoid - target).abs());
    }

    #[test]
    fn dividing_plane_between_planes_removes_the_grid_maximum_bias() {
        let arcs: Vec<f64> = (0..8).map(|k| f64::from(k) * 0.5).collect();
        let peak = 1.7_f64;
        let gradients: Vec<f64> = arcs.iter().map(|&s| -(s - peak)).collect();
        let quadrature = Quadrature::new(&arcs).expect("quadrature");
        let profile =
            integrate_profile(&quadrature, &gradients, &[0.0; 8], 1.0, 1.0).expect("profile");
        assert_abs_diff_eq!(profile.dividing_arc, peak, epsilon = 1.0e-10);
        assert_abs_diff_eq!(profile.barrier, peak * peak / 2.0, epsilon = 1.0e-10);
        assert!(profile.free_energy.iter().all(|&f| f < profile.barrier));
    }

    #[test]
    fn normalized_rate_is_the_projected_frequency_formula() {
        let arcs = [0.0, 0.5, 1.0];
        let gradients = [0.0, 0.5, 0.0];
        let quadrature = Quadrature::new(&arcs).expect("quadrature");
        let kbt = 0.3_f64;
        let curvature = 5.0_f64;
        let profile =
            integrate_profile(&quadrature, &gradients, &[0.0; 3], kbt, curvature).expect("profile");
        let expected = curvature.sqrt() / TAU * (-profile.barrier / kbt).exp();
        assert_abs_diff_eq!(
            profile.normalized_rate,
            expected,
            epsilon = 1.0e-15 * expected
        );
        let gaussian = (TAU * kbt / curvature).sqrt();
        assert_abs_diff_eq!(profile.reactant_partition, gaussian, epsilon = 1.0e-15);
        let flux = (kbt / TAU).sqrt() * (-profile.barrier / kbt).exp() / profile.reactant_partition;
        assert_abs_diff_eq!(profile.normalized_rate, flux, epsilon = 1.0e-15 * expected);
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
            .with_planes(3)
            .with_burn_in_steps(1)
            .with_sample_steps(2)
            .with_zero_temperature_tol(1.0e-2);
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
