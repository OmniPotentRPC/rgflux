//! First-order anharmonic correction to harmonic TST, from gradients and
//! Hessian-vector products only.
//!
//! Near a stationary point `x₀` with Hessian `H`, positive on the free
//! subspace, write `x = x₀ + √kT z`. Then
//! `ln Z = ln Z_h + kT A + O(kT²)` with
//!
//! `A = E[-c₄(z) + ½ c₃(z)²]`, `z ~ N(0, H⁻¹)`,
//!
//! where `c₃ = V_ijk z_i z_j z_k / 6` and `c₄ = V_ijkl z_i z_j z_k z_l / 24`
//! are the cubic and quartic terms of the Taylor series along `z`. Wick's
//! theorem turns the expectation into the familiar contractions in the
//! eigenbasis,
//!
//! `A = -⅛ Σ V_iijj/(λ_i λ_j) + (1/12) Σ V_ijk²/(λ_i λ_j λ_k) + ⅛ Σ_k (Σ_i V_iik/λ_i)²/λ_k`,
//!
//! but no tensor is formed here. Each probe `z = H^{-1/2} ξ` comes from Lanczos on the Hessian
//! action, and `c₃`, `c₄` come from gradients along `z`. Both are
//! homogeneous in `ξ`, so `ξ` is drawn on the unit sphere of the free
//! subspace, of dimension `d`, and its radius is integrated exactly:
//! `E|ξ|⁴ = d(d + 2)`, `E|ξ|⁶ = d(d + 2)(d + 4)`. That removes the
//! radial part of the variance, and in one dimension all of it.
//!
//! At the saddle the plane through it with the unstable mode `u` as normal
//! has its own coefficient `A‡`. Moving that plane along `u` by `s` changes
//! its free energy by `½ λ_u s² + ½ kT S s` with
//! `S = Σ_j V_ujj / λ_j = E[V_ujk z_j z_k]`, so the maximum sits at
//! `s* = kT S / (2|λ_u|)` and is higher by `kT² S² / (8|λ_u|)`. Turning the
//! plane enters `ln k` only at `O(kT²)`. The rate of the optimal plane over
//! harmonic TST is therefore
//!
//! `ln(k / k_HTST) = kT (A‡ - A_R - S² / (8|λ_u|)) + O(kT²)`.

use ndarray::{Array1, ArrayView1};

use super::sample::held_directions;
use rgsaddle::projector::Hyperplane;
use rgsaddle::stats::Rng;
use rgsaddle::{PointSurface, SaddleError};

/// Probes and finite differences of [`anharmonic_correction`].
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct AnharmonicConfig {
    /// Gaussian probes at each stationary point.
    pub probes: usize,
    /// Length of the displacement along a probe for its cubic and quartic
    /// terms, in the length unit of the surface.
    pub step: f64,
    /// Displacement for a Hessian action by gradient differences, used
    /// when the surface has no analytic action.
    pub hvp_step: f64,
    /// Most Lanczos steps for one probe.
    pub krylov: usize,
    /// Relative change of a probe between Lanczos checks below which the
    /// iteration stops.
    pub krylov_tolerance: f64,
    /// Seed of the probes.
    pub seed: u64,
}

impl Default for AnharmonicConfig {
    fn default() -> Self {
        Self {
            probes: 2000,
            step: 1.0e-2,
            hvp_step: 1.0e-4,
            krylov: 64,
            krylov_tolerance: 1.0e-8,
            seed: 1,
        }
    }
}

impl AnharmonicConfig {
    /// Set [`Self::probes`].
    #[must_use]
    pub fn with_probes(mut self, probes: usize) -> Self {
        self.probes = probes;
        self
    }

    /// Set [`Self::step`].
    #[must_use]
    pub fn with_step(mut self, step: f64) -> Self {
        self.step = step;
        self
    }

    /// Set [`Self::hvp_step`].
    #[must_use]
    pub fn with_hvp_step(mut self, hvp_step: f64) -> Self {
        self.hvp_step = hvp_step;
        self
    }

    /// Set [`Self::krylov`].
    #[must_use]
    pub fn with_krylov(mut self, krylov: usize) -> Self {
        self.krylov = krylov;
        self
    }

    /// Set [`Self::krylov_tolerance`].
    #[must_use]
    pub fn with_krylov_tolerance(mut self, krylov_tolerance: f64) -> Self {
        self.krylov_tolerance = krylov_tolerance;
        self
    }

    /// Set [`Self::seed`].
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    fn check(&self) -> Result<(), SaddleError> {
        if self.probes < 2 {
            return Err(SaddleError::Invalid(
                "at least two probes are needed".into(),
            ));
        }
        if !(self.step.is_finite() && self.step > 0.0) {
            return Err(SaddleError::Invalid(
                "the probe step must be positive".into(),
            ));
        }
        if !(self.hvp_step.is_finite() && self.hvp_step > 0.0) {
            return Err(SaddleError::Invalid(
                "the Hessian action step must be positive".into(),
            ));
        }
        if self.krylov == 0 {
            return Err(SaddleError::Invalid(
                "at least one Lanczos step is needed".into(),
            ));
        }
        if !(self.krylov_tolerance.is_finite() && self.krylov_tolerance > 0.0) {
            return Err(SaddleError::Invalid(
                "the Lanczos tolerance must be positive".into(),
            ));
        }
        Ok(())
    }
}

/// Coefficients of the first-order correction, each with the standard
/// error of its probe mean.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct AnharmonicCorrection {
    /// `A_R` at the minimum.
    pub reactant: f64,
    /// Standard error of [`Self::reactant`].
    pub reactant_error: f64,
    /// `A‡` of the plane through the saddle normal to the unstable mode.
    pub saddle: f64,
    /// Standard error of [`Self::saddle`].
    pub saddle_error: f64,
    /// `S = Σ_j V_ujj / λ_j`, with `u` the plane normal.
    pub slope: f64,
    /// Standard error of [`Self::slope`].
    pub slope_error: f64,
    /// `|λ_u| = -uᵀ H u` at the saddle.
    pub curvature: f64,
    /// `c₁ = A‡ - A_R - S² / (8|λ_u|)`.
    pub coefficient: f64,
    /// Standard error of [`Self::coefficient`], with the correlation of
    /// `A‡` and `S` on shared probes.
    pub coefficient_error: f64,
    /// Surface evaluations, gradient differences included.
    pub evaluations: usize,
}

impl AnharmonicCorrection {
    /// `ln(k / k_HTST)` of the optimal plane to first order, `c₁ kT`, and
    /// its error.
    pub fn log_ratio(&self, kt: f64) -> (f64, f64) {
        (self.coefficient * kt, self.coefficient_error * kt)
    }

    /// `ln(k / k_HTST)` of the plane fixed at the saddle to first order,
    /// `(A‡ - A_R) kT`, and its error.
    pub fn fixed_log_ratio(&self, kt: f64) -> (f64, f64) {
        let error = self.saddle_error.hypot(self.reactant_error);
        ((self.saddle - self.reactant) * kt, error * kt)
    }

    /// Offset of the optimal plane from the saddle along its normal to
    /// first order, `kT S / (2|λ_u|)`.
    pub fn offset(&self, kt: f64) -> f64 {
        kt * self.slope / (2.0 * self.curvature)
    }
}

/// First-order anharmonic coefficients of the minimum and of the saddle
/// plane, as the module documentation defines them.
///
/// `saddle` is the plane through the saddle with the unstable mode as its
/// normal, for example [`Hyperplane::from_min_mode`]. `A‡` and `S` change
/// at first order with that normal, through the third and fourth
/// derivatives along the plane, so the mode should be converged well
/// beyond what locating the saddle needs. Coordinates are mass-weighted,
/// as for the rest of OH-TST. The surface's excluded modes are left out
/// of both subspaces, and the Hessian must be positive on what remains.
///
/// `minimum` is the reactant minimum in the coordinates of `surface`, the same
/// length as the saddle plane.
///
/// # Errors
///
/// [`SaddleError::Invalid`] when `config` fails its checks (fewer than two probes,
/// a non-positive or non-finite step, or a zero or non-positive Lanczos setting),
/// when the plane normal is not an unstable direction of the saddle, or the
/// Hessian is not positive on the free subspace; [`SaddleError::Shape`] when
/// `minimum` differs in length from the saddle plane; and the errors of `surface`.
pub fn anharmonic_correction<S: PointSurface>(
    surface: &S,
    minimum: ArrayView1<f64>,
    saddle: &Hyperplane,
    config: &AnharmonicConfig,
) -> Result<AnharmonicCorrection, SaddleError> {
    config.check()?;
    if minimum.len() != saddle.dim() {
        return Err(SaddleError::Shape(
            "the minimum differs in length from the saddle plane".into(),
        ));
    }
    let mut rng = Rng::new(config.seed);
    let mut evaluations = 0;
    let held = held_directions(surface, minimum, None, true)?;
    let (reactant, _) = probe(
        surface,
        minimum,
        &held,
        None,
        config,
        &mut rng,
        &mut evaluations,
    )?;
    let point = saddle.point();
    let unstable = saddle.normal().to_owned();
    let held = held_directions(surface, point, Some(unstable.view()), true)?;
    let hu = hessian_action(surface, point, &unstable, config.hvp_step, &mut evaluations)?;
    let curvature = -unstable.dot(&hu);
    if curvature.is_nan() || curvature <= 0.0 {
        return Err(SaddleError::Invalid(
            "the plane normal is not an unstable direction of the saddle".into(),
        ));
    }
    let (plane, slopes) = probe(
        surface,
        point,
        &held,
        Some(&unstable),
        config,
        &mut rng,
        &mut evaluations,
    )?;
    let (reactant, reactant_error) = mean_and_error(&reactant);
    let (saddle_value, saddle_error) = mean_and_error(&plane);
    let (slope, slope_error) = mean_and_error(&slopes);
    let lever = slope / (4.0 * curvature);
    let linear: Vec<f64> = plane
        .iter()
        .zip(&slopes)
        .map(|(a, s)| a - lever * s)
        .collect();
    let (_, plane_error) = mean_and_error(&linear);
    Ok(AnharmonicCorrection {
        reactant,
        reactant_error,
        saddle: saddle_value,
        saddle_error,
        slope,
        slope_error,
        curvature,
        coefficient: saddle_value - reactant - slope * slope / (8.0 * curvature),
        coefficient_error: plane_error.hypot(reactant_error),
        evaluations,
    })
}

/// Per-probe `-c₄ + ½ c₃²` at `x0` on the subspace orthogonal to `held`,
/// and with `unstable` the per-probe `V_ujk z_j z_k`.
fn probe<S: PointSurface>(
    surface: &S,
    x0: ArrayView1<f64>,
    held: &[Array1<f64>],
    unstable: Option<&Array1<f64>>,
    config: &AnharmonicConfig,
    rng: &mut Rng,
    evaluations: &mut usize,
) -> Result<(Vec<f64>, Vec<f64>), SaddleError> {
    let n = x0.len();
    if held.len() >= n {
        return Err(SaddleError::Invalid(
            "no free direction is left at the stationary point".into(),
        ));
    }
    let (_, g0) = evaluate(surface, x0, evaluations)?;
    let project = |v: &mut Array1<f64>| {
        for t in held {
            let c = t.dot(v);
            v.scaled_add(-c, t);
        }
    };
    let d = (n - held.len()) as f64;
    let moments = [d, d * (d + 2.0), d * (d + 2.0) * (d + 4.0)];
    let mut terms = Vec::with_capacity(config.probes);
    let mut slopes = Vec::with_capacity(if unstable.is_some() { config.probes } else { 0 });
    while terms.len() < config.probes {
        let mut xi: Array1<f64> = (0..n).map(|_| rng.normal()).collect();
        project(&mut xi);
        let radius = xi.dot(&xi).sqrt();
        if radius < 1.0e-12 {
            continue;
        }
        xi /= radius;
        let z = inverse_sqrt(
            |v| {
                let mut hv = hessian_action(surface, x0, v, config.hvp_step, evaluations)?;
                project(&mut hv);
                Ok(hv)
            },
            &xi,
            config.krylov.min(n - held.len()),
            config.krylov_tolerance,
        )?;
        let t = config.step / z.dot(&z).sqrt();
        let mut along = [[0.0; 2]; 4];
        for (k, scale) in [t, -t, 2.0 * t, -2.0 * t].into_iter().enumerate() {
            let mut x = x0.to_owned();
            x.scaled_add(scale, &z);
            let (_, g) = evaluate(surface, x.view(), evaluations)?;
            along[k] = [z.dot(&g), unstable.map_or(0.0, |u| u.dot(&g))];
        }
        let centre = [z.dot(&g0), unstable.map_or(0.0, |u| u.dot(&g0))];
        let even = |i: usize, scale: f64| {
            let near = (along[0][i] + along[1][i] - 2.0 * centre[i]) / (t * t);
            let far = (along[2][i] + along[3][i] - 2.0 * centre[i]) / (4.0 * t * t);
            (4.0 * near - far) / 3.0 * scale
        };
        let c3 = even(0, 1.0 / 6.0);
        let odd_near = along[0][0] - along[1][0];
        let odd_far = along[2][0] - along[3][0];
        let c4 = (odd_far - 2.0 * odd_near) / (2.0 * t * t * t) / 24.0;
        terms.push(-moments[1] * c4 + 0.5 * moments[2] * c3 * c3);
        if unstable.is_some() {
            slopes.push(moments[0] * even(1, 1.0));
        }
    }
    Ok((terms, slopes))
}

pub(crate) fn evaluate<S: PointSurface>(
    surface: &S,
    x: ArrayView1<f64>,
    evaluations: &mut usize,
) -> Result<(f64, Array1<f64>), SaddleError> {
    let (energy, gradient) = surface.eval(x)?;
    *evaluations += 1;
    if gradient.len() != x.len() {
        return Err(SaddleError::Shape(
            "gradient length differs from the configuration".into(),
        ));
    }
    if !energy.is_finite() || !gradient.iter().all(|g| g.is_finite()) {
        return Err(SaddleError::NonFinite(
            "energy or gradient near a stationary point",
        ));
    }
    Ok((energy, gradient))
}

/// `H v`, analytic when the surface has it, else central differences of
/// the gradient.
/// `H(x) v`, analytic when the surface has it and otherwise a central
/// gradient difference with step `step` along `v`.
pub(crate) fn hessian_action<S: PointSurface>(
    surface: &S,
    x: ArrayView1<f64>,
    v: &Array1<f64>,
    step: f64,
    evaluations: &mut usize,
) -> Result<Array1<f64>, SaddleError> {
    if let Some(hv) = surface.hessian_vector(x, v.view())? {
        if hv.len() != x.len() || !hv.iter().all(|value| value.is_finite()) {
            return Err(SaddleError::NonFinite("surface Hessian action"));
        }
        return Ok(hv);
    }
    let norm = v.dot(v).sqrt();
    if norm == 0.0 {
        return Ok(Array1::zeros(x.len()));
    }
    let h = step / norm;
    let mut plus = x.to_owned();
    plus.scaled_add(h, v);
    let mut minus = x.to_owned();
    minus.scaled_add(-h, v);
    let (_, gp) = evaluate(surface, plus.view(), evaluations)?;
    let (_, gm) = evaluate(surface, minus.view(), evaluations)?;
    Ok((gp - gm) / (2.0 * h))
}

/// `H^{-1/2} b` by Lanczos with full reorthogonalization, for `H` given by
/// its action and positive on the Krylov space of `b`.
pub(crate) fn inverse_sqrt<F>(
    mut apply: F,
    b: &Array1<f64>,
    steps: usize,
    tolerance: f64,
) -> Result<Array1<f64>, SaddleError>
where
    F: FnMut(&Array1<f64>) -> Result<Array1<f64>, SaddleError>,
{
    let norm = b.dot(b).sqrt();
    if norm == 0.0 {
        return Ok(Array1::zeros(b.len()));
    }
    let mut basis = vec![b / norm];
    let mut alpha: Vec<f64> = Vec::with_capacity(steps);
    let mut beta: Vec<f64> = Vec::with_capacity(steps);
    let mut previous: Option<Array1<f64>> = None;
    for j in 0..steps {
        let mut w = apply(&basis[j])?;
        let a = basis[j].dot(&w);
        alpha.push(a);
        for _ in 0..2 {
            for q in &basis {
                let c = q.dot(&w);
                w.scaled_add(-c, q);
            }
        }
        let next = w.dot(&w).sqrt();
        let scale = alpha.iter().fold(0.0f64, |m, a| m.max(a.abs()));
        let exhausted = next <= 1.0e-12 * scale.max(f64::MIN_POSITIVE) || j + 1 == steps;
        if exhausted || (j + 1) % 4 == 0 {
            let z = krylov_inverse_sqrt(&alpha, &beta, &basis, norm)?;
            let converged = previous.as_ref().is_some_and(|p| {
                let change = &z - p;
                change.dot(&change).sqrt() <= tolerance * z.dot(&z).sqrt()
            });
            if exhausted || converged {
                return Ok(z);
            }
            previous = Some(z);
        }
        beta.push(next);
        basis.push(w / next);
    }
    Err(SaddleError::Solver("Lanczos ran out of steps".into()))
}

fn krylov_inverse_sqrt(
    alpha: &[f64],
    beta: &[f64],
    basis: &[Array1<f64>],
    norm: f64,
) -> Result<Array1<f64>, SaddleError> {
    let m = alpha.len();
    let (values, vectors) = tridiagonal_eigh(alpha, &beta[..m - 1])?;
    if values.iter().any(|&v| v.is_nan() || v <= 0.0) {
        return Err(SaddleError::Invalid(
            "the Hessian is not positive on the probed subspace".into(),
        ));
    }
    let mut coefficients = vec![0.0; m];
    for (k, &value) in values.iter().enumerate() {
        let weight = vectors[k] * norm / value.sqrt();
        for (i, c) in coefficients.iter_mut().enumerate() {
            *c += weight * vectors[i * m + k];
        }
    }
    let mut z = Array1::zeros(basis[0].len());
    for (c, q) in coefficients.iter().zip(basis) {
        z.scaled_add(*c, q);
    }
    Ok(z)
}

/// Eigenvalues and eigenvectors of the symmetric tridiagonal matrix with
/// diagonal `diagonal` and off-diagonal `off`, by implicit QL with
/// Wilkinson shifts. Component `i` of eigenvector `k` is `vectors[i m + k]`.
fn tridiagonal_eigh(diagonal: &[f64], off: &[f64]) -> Result<(Vec<f64>, Vec<f64>), SaddleError> {
    let m = diagonal.len();
    let mut d = diagonal.to_vec();
    let mut e = off.to_vec();
    e.resize(m, 0.0);
    let mut z = vec![0.0; m * m];
    for i in 0..m {
        z[i * m + i] = 1.0;
    }
    for l in 0..m {
        let mut iterations = 0;
        loop {
            let mut last = l;
            while last + 1 < m && e[last].abs() > f64::EPSILON * (d[last].abs() + d[last + 1].abs())
            {
                last += 1;
            }
            if last == l {
                break;
            }
            iterations += 1;
            if iterations > 60 {
                return Err(SaddleError::Solver(
                    "the tridiagonal eigensolver did not converge".into(),
                ));
            }
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = g.hypot(1.0);
            g = d[last] - d[l] + e[l] / (g + r.copysign(g));
            let (mut s, mut c, mut p) = (1.0, 1.0, 0.0);
            let mut deflated = false;
            for i in (l..last).rev() {
                let f = s * e[i];
                let b = c * e[i];
                r = f.hypot(g);
                e[i + 1] = r;
                if r == 0.0 {
                    d[i + 1] -= p;
                    e[last] = 0.0;
                    deflated = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
                for k in 0..m {
                    let f = z[k * m + i + 1];
                    z[k * m + i + 1] = s * z[k * m + i] + c * f;
                    z[k * m + i] = c * z[k * m + i] - s * f;
                }
            }
            if deflated {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[last] = 0.0;
        }
    }
    Ok((d, z))
}

fn mean_and_error(values: &[f64]) -> (f64, f64) {
    let m = values.len() as f64;
    let mean = values.iter().sum::<f64>() / m;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (m - 1.0);
    (mean, (variance / m).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};
    use rgsaddle::eigensolve::exact_eigh;

    /// A rotation of the plane of axes `a` and `b`.
    fn givens(n: usize, a: usize, b: usize, angle: f64) -> Array2<f64> {
        let mut g = Array2::eye(n);
        g[(a, a)] = angle.cos();
        g[(b, b)] = angle.cos();
        g[(a, b)] = -angle.sin();
        g[(b, a)] = angle.sin();
        g
    }

    #[test]
    fn tridiagonal_eigensolver_matches_the_dense_one() {
        let diagonal = [2.0, -1.0, 0.5, 3.0, 0.0, 1.5];
        let off = [0.7, 1.2, -0.4, 0.9, 2.0];
        let m = diagonal.len();
        let mut t = Array2::<f64>::zeros((m, m));
        for i in 0..m {
            t[(i, i)] = diagonal[i];
            if i + 1 < m {
                t[(i, i + 1)] = off[i];
                t[(i + 1, i)] = off[i];
            }
        }
        let (values, vectors) = tridiagonal_eigh(&diagonal, &off).unwrap();
        let mut sorted = values.clone();
        sorted.sort_by(f64::total_cmp);
        let (dense, _) = exact_eigh(t.view()).unwrap();
        for (a, b) in sorted.iter().zip(&dense) {
            assert!((a - b).abs() < 1e-12, "{a} against {b}");
        }
        for (k, value) in values.iter().enumerate() {
            let v = Array1::from_shape_fn(m, |i| vectors[i * m + k]);
            assert!((v.dot(&v) - 1.0).abs() < 1e-12);
            let residual = t.dot(&v) - &v * *value;
            assert!(residual.dot(&residual).sqrt() < 1e-12);
        }
    }

    #[test]
    fn lanczos_inverse_square_root_matches_the_eigendecomposition() {
        let q = givens(5, 0, 3, 0.7)
            .dot(&givens(5, 1, 4, -0.4))
            .dot(&givens(5, 2, 3, 1.1));
        let lambda = array![0.3, 1.0, 2.5, 7.0, 40.0];
        let h = q.t().dot(&Array2::from_diag(&lambda)).dot(&q);
        let b = array![0.4, -1.2, 0.7, 2.0, -0.3];
        let exact = q
            .t()
            .dot(&Array2::from_diag(&lambda.mapv(|l: f64| 1.0 / l.sqrt())))
            .dot(&q)
            .dot(&b);
        let z = inverse_sqrt(|v| Ok(h.dot(v)), &b, 5, 1.0e-12).unwrap();
        for (a, e) in z.iter().zip(&exact) {
            assert!((a - e).abs() < 1e-10, "{a} against {e}");
        }
        let negative = Array2::from_diag(&array![1.0, -1.0]);
        assert!(inverse_sqrt(|v| Ok(negative.dot(v)), &array![1.0, 1.0], 2, 1e-12).is_err());
    }

    /// In a rotated frame `y = Q x`, a double well `h (y₀² - 1)²` along
    /// `y₀` and three modes `½ λ_i y_i² + a_i y_i³ / 6 + b_i y_i⁴ / 24`
    /// coupled to it by `½ κ_i y₀ y_i²`. The saddle is at the origin and
    /// the reactant minimum at `y₀ = -1`, where every tensor is diagonal
    /// except `V_0ii = κ_i`.
    struct Coupled {
        q: Array2<f64>,
        well: f64,
        lambda: [f64; 3],
        cubic: [f64; 3],
        quartic: [f64; 3],
        coupling: [f64; 3],
    }

    impl Coupled {
        fn new() -> Self {
            Self {
                q: givens(4, 0, 1, 0.5)
                    .dot(&givens(4, 1, 2, -0.8))
                    .dot(&givens(4, 2, 3, 0.3)),
                well: 0.5,
                lambda: [1.5, 4.0, 0.8],
                cubic: [0.9, -2.0, 0.4],
                quartic: [3.0, 1.0, 0.6],
                coupling: [0.7, -1.5, 0.25],
            }
        }

        /// `A_R`, `A‡`, `S` and `|λ_u|` from the tensor contractions.
        fn exact(&self) -> (f64, f64, f64, f64) {
            let h = self.well;
            let one =
                |l: f64, a: f64, b: f64| -b / (8.0 * l * l) + 5.0 * a * a / (24.0 * l.powi(3));
            let plane: f64 = (0..3)
                .map(|i| one(self.lambda[i], self.cubic[i], self.quartic[i]))
                .sum();
            let slope = (0..3).map(|i| self.coupling[i] / self.lambda[i]).sum();
            let (v000, v0000, mu0) = (-24.0 * h, 24.0 * h, 8.0 * h);
            let mut reactant = -v0000 / (8.0 * mu0 * mu0) + v000 * v000 / (12.0 * mu0.powi(3));
            let mut trace0 = v000 / mu0;
            for (((&l, &a), &b), &k) in self
                .lambda
                .iter()
                .zip(&self.cubic)
                .zip(&self.quartic)
                .zip(&self.coupling)
            {
                let mu = l - k;
                reactant += -b / (8.0 * mu * mu);
                reactant += (a * a / mu.powi(3) + 3.0 * k * k / (mu0 * mu * mu)) / 12.0;
                reactant += a * a / (8.0 * mu.powi(3));
                trace0 += k / mu;
            }
            reactant += trace0 * trace0 / (8.0 * mu0);
            (reactant, plane, slope, 4.0 * h)
        }
    }

    impl PointSurface for Coupled {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let y = self.q.dot(&x);
            let h = self.well;
            let y0 = y[0];
            let mut e = h * (y0 * y0 - 1.0).powi(2);
            let mut gy = Array1::zeros(4);
            gy[0] = 4.0 * h * y0 * (y0 * y0 - 1.0);
            for i in 0..3 {
                let (l, a, b, k) = (
                    self.lambda[i],
                    self.cubic[i],
                    self.quartic[i],
                    self.coupling[i],
                );
                let yi = y[i + 1];
                e += 0.5 * l * yi * yi + a * yi.powi(3) / 6.0 + b * yi.powi(4) / 24.0;
                e += 0.5 * k * y0 * yi * yi;
                gy[i + 1] = l * yi + 0.5 * a * yi * yi + b * yi.powi(3) / 6.0 + k * y0 * yi;
                gy[0] += 0.5 * k * yi * yi;
            }
            Ok((e, self.q.t().dot(&gy)))
        }
    }

    #[test]
    fn coupled_well_coefficients_match_the_contractions() {
        let model = Coupled::new();
        let (reactant, plane, slope, curvature) = model.exact();
        let minimum = model.q.row(0).mapv(|v| -v);
        let normal = model.q.row(0).to_owned();
        let saddle = Hyperplane::new(Array1::zeros(4), normal).unwrap();
        let config = AnharmonicConfig::default().with_probes(8000).with_seed(3);
        let result = anharmonic_correction(&model, minimum.view(), &saddle, &config).unwrap();
        assert!((result.curvature - curvature).abs() < 1e-6);
        for (name, value, error, exact) in [
            ("A_R", result.reactant, result.reactant_error, reactant),
            ("A‡", result.saddle, result.saddle_error, plane),
            ("S", result.slope, result.slope_error, slope),
        ] {
            assert!(
                (value - exact).abs() < 4.0 * error + 1.0e-5,
                "{name} {value} ± {error} against {exact}"
            );
            assert!(error < 0.015, "{name} error {error}");
        }
        let coefficient = plane - reactant - slope * slope / (8.0 * curvature);
        assert!(
            (result.coefficient - coefficient).abs() < 4.0 * result.coefficient_error,
            "c₁ {} ± {} against {coefficient}",
            result.coefficient,
            result.coefficient_error
        );
        let (fixed, _) = result.fixed_log_ratio(0.1);
        assert!((fixed - 0.1 * (result.saddle - result.reactant)).abs() < 1e-15);
        assert!((result.offset(0.1) - 0.1 * result.slope / (2.0 * curvature)).abs() < 1e-6);
        // Probing the saddle as a minimum meets its unstable mode, and a
        // stable normal is not a saddle plane.
        let origin = Array1::zeros(4);
        assert!(anharmonic_correction(&model, origin.view(), &saddle, &config).is_err());
        let stable = Hyperplane::new(Array1::zeros(4), model.q.row(1).to_owned()).unwrap();
        assert!(anharmonic_correction(&model, minimum.view(), &stable, &config).is_err());
    }

    /// The contractions to the bit, so a rewrite of `Coupled::exact` keeps
    /// every rounding.
    #[test]
    fn coupled_well_contractions_are_pinned() {
        let (reactant, plane, slope, curvature) = Coupled::new().exact();
        assert_eq!(
            [reactant, plane, slope, curvature].map(f64::to_bits),
            [
                0xbf9a_837e_7e4d_1004,
                0xbfc4_eeee_eeee_eeef,
                0x3fd9_dddd_dddd_dddd,
                0x4000_0000_0000_0000,
            ]
        );
    }
}
