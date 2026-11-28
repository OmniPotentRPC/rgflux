//! Maps that carry configurations of one plane onto another, for targeted
//! free-energy perturbation (Jarzynski, Phys. Rev. E 65, 046122 (2002)),
//! BAR and MBAR between planes.
//!
//! [`Transport::Rigid`] is Eq. (7.1) of the OH-TST design survey: the
//! minimal rotation taking `ν_a` to `ν_b` about the pivot of `a`, then the
//! shift of the pivot off the held modes. It preserves the area measure,
//! so `Z_b / Z_a = ⟨exp(-ΔV/kT)⟩_a` exactly.
//!
//! [`Transport::Gaussian`] first applies, within plane `a`, the optimal
//! transport map between Gaussians with the moments of the two planes
//! (Eq. 12.2 of the survey; Olkin and Pukelsheim, Linear Algebra Appl. 48,
//! 257 (1982)),
//!
//! ```text
//! y ↦ m_b + L (y - m_a),   L = Σ_a^{-1/2} (Σ_a^{1/2} Σ_b Σ_a^{1/2})^{1/2} Σ_a^{-1/2},
//! ```
//!
//! with plane `b` pulled back onto `a` by the rigid map. Then
//! `Z_b / Z_a = |det L| ⟨exp(-ΔV/kT)⟩_a` exactly for any `L` fixed
//! independently of the configurations averaged, and on a quadratic `V`
//! the work is constant. The means are matched in full. The covariances
//! are matched in the leading principal directions of the pooled samples
//! and the map is the identity on the rest, so the moments come from a few
//! dozen samples and the Jacobian is the determinant of a small matrix.
//! The samples that fit the map are not the ones it is applied to: a map
//! fitted to its evaluation samples biases the estimate (Rizzi, Carloni
//! and Parrinello, J. Phys. Chem. Lett. 12, 9449 (2021)).
//!
//! A fitted map also carries the sampling noise of its moments, which in
//! many free dimensions with few samples costs more overlap than the map
//! recovers. [`GaussianTransport::shrink_to_rigid`] scales the fitted
//! change, the mean shift and `L - I`, by the positive-part James–Stein
//! factor (James and Stein, Proc. Fourth Berkeley Symp. Math. Statist.
//! Prob. 1, 361 (1961)) `(1 - noise / size)₊`. Here `size` is
//! `δᵀ Σ⁻¹ δ + 2 ‖ln L‖²`, twice the Kullback–Leibler divergence between
//! the fitted Gaussians to second order, and `noise` is the size of the
//! same fit between the even and odd halves of one plane's samples,
//! scaled to the full counts. The shrunk map is still affine with an
//! exact Jacobian, so the estimates stay exact for any weight.

use ndarray::{Array1, Array2, ArrayView1, Axis};

use rgsaddle::eigensolve::exact_eigh;
use rgsaddle::error::SaddleError;
use rgsaddle::projector::Hyperplane;
use rgsaddle::stats::Rng;

/// How samples of one plane are carried onto the next for BAR and MBAR.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
pub enum Transport {
    /// The rigid map of the module documentation: the minimal rotation of the
    /// normal, then the shift of the pivot. The default.
    #[default]
    Rigid,
    /// The Gaussian optimal-transport map within the plane, then the
    /// rigid map.
    Gaussian(GaussianTransport),
}

/// Settings of [`Transport::Gaussian`].
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianTransport {
    /// Principal directions of the pooled fitting samples in which the
    /// covariances are matched. It is capped by the plane's free
    /// dimensions and kept well below the fitting samples per plane.
    pub rank: usize,
    /// Weight from 0 to 1 that shrinks each fitted covariance toward the
    /// multiple of the identity with its trace.
    pub shrinkage: f64,
    /// Shrink each fitted map toward the rigid one by the positive-part
    /// James–Stein factor `1 - noise / size`, where `size` measures the
    /// fitted change and `noise` the change the same fit finds between two
    /// halves of one plane's samples, scaled to the full counts. Changes
    /// the samples do not resolve are then left to the rigid map.
    pub shrink_to_rigid: bool,
}

impl Default for GaussianTransport {
    fn default() -> Self {
        Self {
            rank: 4,
            shrinkage: 0.0,
            shrink_to_rigid: true,
        }
    }
}

impl GaussianTransport {
    /// Set [`Self::rank`].
    #[must_use]
    pub fn with_rank(mut self, rank: usize) -> Self {
        self.rank = rank;
        self
    }

    /// Set [`Self::shrinkage`].
    #[must_use]
    pub fn with_shrinkage(mut self, shrinkage: f64) -> Self {
        self.shrinkage = shrinkage;
        self
    }

    /// Set [`Self::shrink_to_rigid`].
    #[must_use]
    pub fn with_shrink_to_rigid(mut self, shrink_to_rigid: bool) -> Self {
        self.shrink_to_rigid = shrink_to_rigid;
        self
    }

    pub(crate) fn check(&self) -> Result<(), SaddleError> {
        if !(0.0..=1.0).contains(&self.shrinkage) {
            return Err(SaddleError::Invalid(
                "Gaussian transport shrinkage must lie in [0, 1]".into(),
            ));
        }
        Ok(())
    }
}

impl From<GaussianTransport> for Transport {
    fn from(gaussian: GaussianTransport) -> Self {
        Self::Gaussian(gaussian)
    }
}

/// The map from plane `a` onto plane `b`.
pub(crate) struct PlaneMap {
    rigid: RigidMap,
    affine: Option<Affine>,
    /// A Gaussian map was asked for and the rigid one is used instead.
    pub(crate) fallback: bool,
}

impl PlaneMap {
    pub(crate) fn rigid(a: &Hyperplane, b: &Hyperplane, held: &[Array1<f64>]) -> Self {
        Self {
            rigid: RigidMap::new(a, b, held),
            affine: None,
            fallback: false,
        }
    }

    /// The map that `transport` asks for, fitted to `fit_a` on `a` and
    /// `fit_b` on `b`.
    pub(crate) fn fit(
        transport: &Transport,
        a: &Hyperplane,
        b: &Hyperplane,
        held: &[Array1<f64>],
        fit_a: &[Array1<f64>],
        fit_b: &[Array1<f64>],
    ) -> Result<Self, SaddleError> {
        let mut map = Self::rigid(a, b, held);
        if let Transport::Gaussian(gaussian) = transport {
            let pulled: Vec<Array1<f64>> = fit_b.iter().map(|q| map.rigid.backward(q)).collect();
            map.affine = Affine::fit(gaussian, a, held, fit_a, &pulled)?;
            map.fallback = map.affine.is_none();
        }
        Ok(map)
    }

    pub(crate) fn forward(&self, q: &Array1<f64>) -> Array1<f64> {
        match &self.affine {
            Some(affine) => self.rigid.forward(&affine.apply(q)),
            None => self.rigid.forward(q),
        }
    }

    pub(crate) fn backward(&self, q: &Array1<f64>) -> Array1<f64> {
        let pulled = self.rigid.backward(q);
        match &self.affine {
            Some(affine) => affine.invert(&pulled),
            None => pulled,
        }
    }

    /// `ln |det J|` of [`Self::forward`] on the plane.
    pub(crate) fn log_jacobian(&self) -> f64 {
        self.affine.as_ref().map_or(0.0, |affine| affine.log_det)
    }

    /// The share of the fitted Gaussian change the map carries, 0 for the
    /// rigid map.
    pub(crate) fn weight(&self) -> f64 {
        self.affine.as_ref().map_or(0.0, |affine| affine.weight)
    }
}

/// The rigid map from plane `a` onto plane `b`.
struct RigidMap {
    pivot: Array1<f64>,
    shift: Array1<f64>,
    from: Array1<f64>,
    across: Option<Array1<f64>>,
    cos: f64,
    sin: f64,
}

impl RigidMap {
    fn new(a: &Hyperplane, b: &Hyperplane, held: &[Array1<f64>]) -> Self {
        let (na, nb) = (a.normal().to_owned(), b.normal().to_owned());
        let cos = na.dot(&nb).clamp(-1.0, 1.0);
        let mut across = &nb - &(&na * cos);
        let length = across.dot(&across).sqrt();
        let across = (length > 1.0e-14).then(|| {
            across /= length;
            across
        });
        let mut shift = &b.point() - &a.point();
        for t in held {
            let c = t.dot(&shift);
            shift.scaled_add(-c, t);
        }
        Self {
            pivot: a.point().to_owned(),
            shift,
            from: na,
            across,
            cos,
            sin: length,
        }
    }

    /// `R v`, or `Rᵀ v` when `sign` is negative.
    fn rotate(&self, v: &mut Array1<f64>, sign: f64) {
        let Some(u) = &self.across else {
            return;
        };
        let (a, b) = (self.from.dot(v), u.dot(v));
        let sin = sign * self.sin;
        v.scaled_add(sin * a, u);
        v.scaled_add(-sin * b, &self.from);
        v.scaled_add(self.cos - 1.0, &(&self.from * a + u * b));
    }

    fn forward(&self, q: &Array1<f64>) -> Array1<f64> {
        let mut v = q - &self.pivot;
        self.rotate(&mut v, 1.0);
        v + &self.pivot + &self.shift
    }

    fn backward(&self, q: &Array1<f64>) -> Array1<f64> {
        let mut v = q - &self.pivot - &self.shift;
        self.rotate(&mut v, -1.0);
        v + &self.pivot
    }
}

/// `y ↦ to + (y - from) + Wᵀ (L - I) W (y - from)` within plane `a`, `W`
/// orthonormal rows in the plane.
struct Affine {
    from: Array1<f64>,
    to: Array1<f64>,
    basis: Array2<f64>,
    forward: Array2<f64>,
    backward: Array2<f64>,
    log_det: f64,
    weight: f64,
}

/// The fitted Gaussian change from one sample set to another within a
/// plane: the means, the principal rows `W`, `L - I` in them, and its size
/// `δᵀ Σ⁻¹ δ + 2 ‖ln L‖²`, twice the Kullback–Leibler divergence between
/// the two Gaussians to second order.
struct Change {
    from: Array1<f64>,
    to: Array1<f64>,
    basis: Array2<f64>,
    excess: Array2<f64>,
    size: f64,
}

impl Change {
    fn fit(
        gaussian: &GaussianTransport,
        constraints: &[Array1<f64>],
        fit_a: &[Array1<f64>],
        pulled: &[Array1<f64>],
    ) -> Result<Option<Self>, SaddleError> {
        let free = fit_a[0].len() - constraints.len();
        let (mean_a, ya) = centred(fit_a, constraints);
        let (mean_b, yb) = centred(pulled, constraints);
        let rank = gaussian
            .rank
            .min(free)
            .min(fit_a.len().min(pulled.len()) - 1);
        let basis = leading_directions(&ya, &yb, rank, constraints)?;
        let rank = basis.nrows();
        let (na, nb) = ((ya.nrows() - 1) as f64, (yb.nrows() - 1) as f64);
        let mut shift = &mean_b - &mean_a;
        project_out(&mut shift, constraints);
        let along = basis.dot(&shift);
        let pooled = |w: ArrayView1<f64>| {
            0.5 * (ya.dot(&w).mapv(|x| x * x).sum() / na + yb.dot(&w).mapv(|x| x * x).sum() / nb)
        };
        let variances: Vec<f64> = basis.rows().into_iter().map(pooled).collect();
        let mut size: f64 = along.iter().zip(&variances).map(|(c, v)| c * c / v).sum();
        if free > rank {
            let trace = 0.5 * (ya.mapv(|x| x * x).sum() / na + yb.mapv(|x| x * x).sum() / nb);
            let rest = (trace - variances.iter().sum::<f64>()) / (free - rank) as f64;
            let outside = shift.dot(&shift) - along.dot(&along);
            if rest > 0.0 {
                size += outside.max(0.0) / rest;
            }
        }
        if rank == 0 {
            return Ok(Some(Self {
                from: mean_a,
                to: mean_b,
                basis,
                excess: Array2::zeros((0, 0)),
                size,
            }));
        }
        let moments = |y: &Array2<f64>| {
            let c = y.dot(&basis.t());
            let mut s = c.t().dot(&c) / (y.nrows() - 1) as f64;
            let trace = s.diag().sum() / rank as f64;
            s *= 1.0 - gaussian.shrinkage;
            for i in 0..rank {
                s[(i, i)] += gaussian.shrinkage * trace;
            }
            s
        };
        let (sa, sb) = (moments(&ya), moments(&yb));
        let Some((root, inverse_root, _)) = roots(&sa)? else {
            return Ok(None);
        };
        let m = root.dot(&sb).dot(&root);
        let Some((m_root, _, _)) = roots(&m)? else {
            return Ok(None);
        };
        let l = inverse_root.dot(&m_root).dot(&inverse_root);
        let (values, _) = exact_eigh(l.view())?;
        size += 2.0 * values.iter().map(|x| x.ln().powi(2)).sum::<f64>();
        Ok(Some(Self {
            from: mean_a,
            to: mean_b,
            basis,
            excess: l - Array2::<f64>::eye(rank),
            size,
        }))
    }
}

impl Affine {
    /// The Gaussian map from the samples `fit_a` of `a` to the samples
    /// `pulled` of `b` carried back onto `a`, shrunk toward the identity
    /// when [`GaussianTransport::shrink_to_rigid`] is set, or `None` when a
    /// fitted covariance is not positive definite.
    fn fit(
        gaussian: &GaussianTransport,
        a: &Hyperplane,
        held: &[Array1<f64>],
        fit_a: &[Array1<f64>],
        pulled: &[Array1<f64>],
    ) -> Result<Option<Self>, SaddleError> {
        gaussian.check()?;
        if fit_a.len() < 2 || pulled.len() < 2 {
            return Err(SaddleError::Invalid(
                "a Gaussian transport map needs two fitting samples on each plane".into(),
            ));
        }
        let mut constraints: Vec<Array1<f64>> = Vec::new();
        for direction in std::iter::once(a.normal().to_owned()).chain(held.iter().cloned()) {
            let mut v = direction;
            project_out(&mut v, &constraints);
            let norm = v.dot(&v).sqrt();
            if norm > 1.0e-10 {
                constraints.push(v / norm);
            }
        }
        let Some(change) = Change::fit(gaussian, &constraints, fit_a, pulled)? else {
            return Ok(None);
        };
        let weight = if gaussian.shrink_to_rigid && fit_a.len() >= 4 && pulled.len() >= 4 {
            let counts = 1.0 / fit_a.len() as f64 + 1.0 / pulled.len() as f64;
            let mut noise = Vec::with_capacity(2);
            for set in [fit_a, pulled] {
                let (even, odd): (Vec<Array1<f64>>, Vec<Array1<f64>>) = (
                    set.iter().step_by(2).cloned().collect(),
                    set.iter().skip(1).step_by(2).cloned().collect(),
                );
                if let Some(null) = Change::fit(gaussian, &constraints, &even, &odd)? {
                    let halves = 1.0 / even.len() as f64 + 1.0 / odd.len() as f64;
                    noise.push(null.size * counts / halves);
                }
            }
            if noise.is_empty() || change.size <= 0.0 {
                0.0
            } else {
                let noise = noise.iter().sum::<f64>() / noise.len() as f64;
                (1.0 - noise / change.size).max(0.0)
            }
        } else {
            1.0
        };
        Self::shrunk(change, weight).map(Some)
    }

    /// The map that carries `weight` of the fitted change: the mean shift
    /// scaled by it and `L` replaced by `I + weight (L - I)`.
    fn shrunk(change: Change, weight: f64) -> Result<Self, SaddleError> {
        let rank = change.basis.nrows();
        let to = &change.from + &((&change.to - &change.from) * weight);
        if rank == 0 {
            return Ok(Self {
                from: change.from,
                to,
                basis: change.basis,
                forward: Array2::zeros((0, 0)),
                backward: Array2::zeros((0, 0)),
                log_det: 0.0,
                weight,
            });
        }
        let (values, vectors) = exact_eigh(change.excess.view())?;
        let scaled = |f: &dyn Fn(f64) -> f64| {
            let mut out = vectors.clone();
            for (mut column, &x) in out.columns_mut().into_iter().zip(values.iter()) {
                column *= f(x);
            }
            out.dot(&vectors.t())
        };
        Ok(Self {
            from: change.from,
            to,
            basis: change.basis,
            forward: scaled(&|x| weight * x),
            backward: scaled(&|x| 1.0 / (1.0 + weight * x) - 1.0),
            log_det: values.iter().map(|x| (1.0 + weight * x).ln()).sum(),
            weight,
        })
    }

    fn apply(&self, q: &Array1<f64>) -> Array1<f64> {
        let y = q - &self.from;
        let correction = self.basis.t().dot(&self.forward.dot(&self.basis.dot(&y)));
        &self.to + &y + correction
    }

    fn invert(&self, q: &Array1<f64>) -> Array1<f64> {
        let z = q - &self.to;
        let correction = self.basis.t().dot(&self.backward.dot(&self.basis.dot(&z)));
        &self.from + &z + correction
    }
}

fn project_out(v: &mut Array1<f64>, directions: &[Array1<f64>]) {
    for d in directions {
        let c = d.dot(v);
        v.scaled_add(-c, d);
    }
}

/// The mean of `set` and its rows less the mean, off `constraints`.
fn centred(set: &[Array1<f64>], constraints: &[Array1<f64>]) -> (Array1<f64>, Array2<f64>) {
    let n = set[0].len();
    let mut rows = Array2::zeros((set.len(), n));
    for (mut row, q) in rows.rows_mut().into_iter().zip(set) {
        row.assign(q);
    }
    let mean = rows.mean_axis(Axis(0)).expect("non-empty");
    for mut row in rows.rows_mut() {
        let mut v = &row - &mean;
        project_out(&mut v, constraints);
        row.assign(&v);
    }
    (mean, rows)
}

/// Up to `rank` leading eigenvectors, as rows, of the pooled covariance
/// `(YₐᵀYₐ/Nₐ + Y_bᵀY_b/N_b) / 2`, by subspace iteration and
/// Rayleigh–Ritz, orthogonal to `constraints`. Directions without variance
/// are dropped.
fn leading_directions(
    ya: &Array2<f64>,
    yb: &Array2<f64>,
    rank: usize,
    constraints: &[Array1<f64>],
) -> Result<Array2<f64>, SaddleError> {
    let n = ya.ncols();
    if rank == 0 {
        return Ok(Array2::zeros((0, n)));
    }
    let width = (rank + 4).min(n - constraints.len());
    let apply = |v: &Array2<f64>| {
        let (na, nb) = (ya.nrows() as f64, yb.nrows() as f64);
        (ya.t().dot(&ya.dot(v)) / na + yb.t().dot(&yb.dot(v)) / nb) * 0.5
    };
    let mut rng = Rng::new(0x9e37_79b9);
    let mut v = Array2::from_shape_fn((n, width), |_| rng.normal());
    v = orthonormal_columns(v, constraints);
    for _ in 0..8 {
        v = orthonormal_columns(apply(&v), constraints);
    }
    let cv = apply(&v);
    let projected = v.t().dot(&cv);
    let (values, vectors) = exact_eigh(projected.view())?;
    let scale = values.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    let keep: Vec<usize> = (0..values.len())
        .rev()
        .filter(|&i| values[i] > 1.0e-12 * scale)
        .take(rank)
        .collect();
    let mut basis = Array2::zeros((keep.len(), n));
    for (row, &i) in keep.iter().enumerate() {
        basis.row_mut(row).assign(&v.dot(&vectors.column(i)));
    }
    Ok(basis)
}

/// Modified Gram–Schmidt on the columns of `v` off `constraints`; columns
/// that vanish are dropped.
fn orthonormal_columns(v: Array2<f64>, constraints: &[Array1<f64>]) -> Array2<f64> {
    let mut kept: Vec<Array1<f64>> = Vec::with_capacity(v.ncols());
    for column in v.columns() {
        let mut c = column.to_owned();
        let size = c.dot(&c).sqrt();
        project_out(&mut c, constraints);
        project_out(&mut c, &kept);
        project_out(&mut c, constraints);
        project_out(&mut c, &kept);
        let norm = c.dot(&c).sqrt();
        if norm > 1.0e-10 * size.max(f64::MIN_POSITIVE) {
            kept.push(c / norm);
        }
    }
    let mut out = Array2::zeros((v.nrows(), kept.len()));
    for (mut column, c) in out.columns_mut().into_iter().zip(&kept) {
        column.assign(c);
    }
    out
}

/// `S^{1/2}`, `S^{-1/2}` and `ln det S` of a symmetric `S`, or `None` when
/// it is not positive definite.
#[allow(clippy::type_complexity)]
fn roots(s: &Array2<f64>) -> Result<Option<(Array2<f64>, Array2<f64>, f64)>, SaddleError> {
    let (values, vectors) = exact_eigh(s.view())?;
    let largest = values.iter().fold(0.0_f64, |m, x| m.max(*x));
    let definite = largest > 0.0 && values.iter().all(|&x| x > 1.0e-12 * largest);
    if !definite {
        return Ok(None);
    }
    let scaled = |power: f64| {
        let mut out = vectors.clone();
        for (mut column, &x) in out.columns_mut().into_iter().zip(values.iter()) {
            column *= x.powf(power);
        }
        out.dot(&vectors.t())
    };
    Ok(Some((
        scaled(0.5),
        scaled(-0.5),
        values.iter().map(|x| x.ln()).sum(),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn the_rigid_map_takes_one_plane_onto_the_other() {
        let a = Hyperplane::new(array![0.2, -0.4, 0.1], array![1.0, 0.3, 0.0]).unwrap();
        let b = Hyperplane::new(array![0.9, 0.2, 0.5], array![0.4, 1.0, 0.0]).unwrap();
        let held = [array![0.0, 0.0, 1.0]];
        let map = PlaneMap::rigid(&a, &b, &held);
        let q = a.project(array![1.3, -0.2, 0.7].view());
        let r = a.project(array![-0.5, 0.8, 0.7].view());
        let (tq, tr) = (map.forward(&q), map.forward(&r));
        assert!(b.distance(tq.view()).abs() < 1e-12);
        assert!((tq[2] - q[2]).abs() < 1e-12);
        let (d0, d1) = (&q - &r, &tq - &tr);
        assert!((d0.dot(&d0) - d1.dot(&d1)).abs() < 1e-12);
        let back = map.backward(&tq);
        assert!((&back - &q).iter().all(|x| x.abs() < 1e-12));
        let same = PlaneMap::rigid(&a, &a, &held);
        assert!((&same.forward(&q) - &q).iter().all(|x| x.abs() < 1e-12));
        assert_eq!(map.log_jacobian(), 0.0);
    }

    /// Exact draws from `N(0, Σ)` restricted to a plane `ν·q = d`.
    fn plane_draws(
        sigma: &Array2<f64>,
        plane: &Hyperplane,
        count: usize,
        rng: &mut Rng,
    ) -> Vec<Array1<f64>> {
        let (values, vectors) = exact_eigh(sigma.view()).unwrap();
        let n = values.len();
        let nu = plane.normal().to_owned();
        let sn = sigma.dot(&nu);
        let offset = nu.dot(&plane.point());
        (0..count)
            .map(|_| {
                let xi = Array1::from_shape_fn(n, |i| values[i].sqrt() * rng.normal());
                let z = vectors.dot(&xi);
                &z + &(&sn * ((offset - nu.dot(&z)) / nu.dot(&sn)))
            })
            .collect()
    }

    /// Between planes through a correlated Gaussian, the fitted map moves
    /// the sample moments of `a` onto those of `b`, inverts exactly, and
    /// its Jacobian is the determinant of the map along the plane.
    #[test]
    fn the_gaussian_map_matches_the_moments_of_the_next_plane() {
        let k = array![
            [1.0, 0.3, 0.1, 0.0],
            [0.3, 2.0, 0.2, 0.1],
            [0.1, 0.2, 0.5, 0.05],
            [0.0, 0.1, 0.05, 3.0]
        ];
        let (values, vectors) = exact_eigh(k.view()).unwrap();
        let mut sigma = vectors.clone();
        for (mut column, &x) in sigma.columns_mut().into_iter().zip(values.iter()) {
            column /= x;
        }
        let sigma = sigma.dot(&vectors.t());
        let a = Hyperplane::new(array![0.0, 0.0, 0.0, 0.0], array![1.0, 0.0, 0.0, 0.0]).unwrap();
        let b = Hyperplane::new(array![0.3, 0.0, 0.0, 0.0], array![1.0, 0.5, 0.2, 0.0]).unwrap();
        let mut rng = Rng::new(11);
        let fit_a = plane_draws(&sigma, &a, 4000, &mut rng);
        let fit_b = plane_draws(&sigma, &b, 4000, &mut rng);
        let exact = GaussianTransport::default()
            .with_rank(3)
            .with_shrink_to_rigid(false);
        let map = PlaneMap::fit(&exact.clone().into(), &a, &b, &[], &fit_a, &fit_b).unwrap();
        assert!(!map.fallback);
        assert_eq!(map.weight(), 1.0);
        let shrunk = exact.with_shrink_to_rigid(true).into();
        let resolved = PlaneMap::fit(&shrunk, &a, &b, &[], &fit_a, &fit_b).unwrap();
        assert!(resolved.weight() > 0.8, "weight {}", resolved.weight());

        let mapped: Vec<Array1<f64>> = fit_a.iter().map(|q| map.forward(q)).collect();
        assert!(mapped.iter().all(|q| b.distance(q.view()).abs() < 1e-12));
        let mean = |set: &[Array1<f64>]| -> Array1<f64> {
            set.iter().fold(Array1::zeros(4), |acc, q| acc + q) / set.len() as f64
        };
        let covariance = |set: &[Array1<f64>]| -> Array2<f64> {
            let m = mean(set);
            set.iter().fold(Array2::zeros((4, 4)), |acc, q| {
                let d = (q - &m).insert_axis(Axis(1));
                acc + d.dot(&d.t())
            }) / (set.len() - 1) as f64
        };
        let (got, want) = (covariance(&mapped), covariance(&fit_b));
        assert!(
            (&mean(&mapped) - &mean(&fit_b))
                .iter()
                .all(|v| v.abs() < 1e-12)
        );
        assert!(
            (&got - &want).iter().all(|v| v.abs() < 1e-10),
            "{got} {want}"
        );

        let q = &fit_a[3];
        let back = map.backward(&map.forward(q));
        assert!((&back - q).iter().all(|v| v.abs() < 1e-12));

        // The Jacobian on the plane, from a basis of each plane.
        let basis = |plane: &Hyperplane| {
            let mut rows = Vec::new();
            for i in 0..4 {
                let mut e = Array1::zeros(4);
                e[i] = 1.0;
                let nu = plane.normal().to_owned();
                project_out(&mut e, &[nu]);
                project_out(&mut e, &rows);
                let norm = e.dot(&e).sqrt();
                if norm > 1e-8 {
                    rows.push(e / norm);
                }
            }
            rows
        };
        let (ea, eb) = (basis(&a), basis(&b));
        let h = 1e-6;
        let mut jacobian = Array2::zeros((3, 3));
        for (j, e) in ea.iter().enumerate() {
            let d = (map.forward(&(q + &(e * h))) - map.forward(&(q - &(e * h)))) / (2.0 * h);
            for (i, f) in eb.iter().enumerate() {
                jacobian[(i, j)] = f.dot(&d);
            }
        }
        let det = jacobian[(0, 0)]
            * (jacobian[(1, 1)] * jacobian[(2, 2)] - jacobian[(1, 2)] * jacobian[(2, 1)])
            - jacobian[(0, 1)]
                * (jacobian[(1, 0)] * jacobian[(2, 2)] - jacobian[(1, 2)] * jacobian[(2, 0)])
            + jacobian[(0, 2)]
                * (jacobian[(1, 0)] * jacobian[(2, 1)] - jacobian[(1, 1)] * jacobian[(2, 0)]);
        assert!(
            (det.abs().ln() - map.log_jacobian()).abs() < 1e-7,
            "{} against {}",
            det.abs().ln(),
            map.log_jacobian()
        );
    }

    /// Two sample sets of one plane differ only by noise, which the
    /// shrinkage leaves to the rigid map, while a shift of a fifth of a
    /// standard deviation in every direction is resolved.
    #[test]
    fn noise_is_left_to_the_rigid_map() {
        let n = 24;
        let sigma = Array2::from_diag(&Array1::from_shape_fn(n, |i| 0.5 + 0.1 * i as f64));
        let mut normal = Array1::zeros(n);
        normal[0] = 1.0;
        let a = Hyperplane::new(Array1::zeros(n), normal).unwrap();
        let mut rng = Rng::new(3);
        let fit_a = plane_draws(&sigma, &a, 128, &mut rng);
        let same = plane_draws(&sigma, &a, 128, &mut rng);
        let transport = Transport::from(GaussianTransport::default().with_rank(8));
        let null = PlaneMap::fit(&transport, &a, &a, &[], &fit_a, &same).unwrap();
        let shift = Array1::from_shape_fn(n, |i| {
            if i == 0 {
                0.0
            } else {
                0.2 * sigma[(i, i)].sqrt()
            }
        });
        let moved: Vec<Array1<f64>> = same.iter().map(|q| q + &shift).collect();
        let real = PlaneMap::fit(&transport, &a, &a, &[], &fit_a, &moved).unwrap();
        assert!(null.weight() < 0.2, "noise kept {}", null.weight());
        assert!(real.weight() > 0.5, "shift kept {}", real.weight());
    }

    /// With rank zero only the means move, and identical fits give the
    /// identity in the plane.
    #[test]
    fn a_rank_zero_map_only_shifts_the_mean() {
        let a = Hyperplane::new(array![0.0, 0.0, 0.0], array![0.0, 0.0, 1.0]).unwrap();
        let fit_a = vec![
            array![1.0, 0.0, 0.0],
            array![-1.0, 0.0, 0.0],
            array![0.0, 2.0, 0.0],
        ];
        let shifted: Vec<Array1<f64>> = fit_a.iter().map(|q| q + &array![0.5, 0.0, 0.0]).collect();
        let zero = Transport::from(GaussianTransport::default().with_rank(0));
        let map = PlaneMap::fit(&zero, &a, &a, &[], &fit_a, &shifted).unwrap();
        let q = array![0.2, 0.3, 0.0];
        assert!(
            (&map.forward(&q) - &array![0.7, 0.3, 0.0])
                .iter()
                .all(|v| v.abs() < 1e-14)
        );
        assert_eq!(map.log_jacobian(), 0.0);
        let full = Transport::from(GaussianTransport::default());
        let same = PlaneMap::fit(&full, &a, &a, &[], &fit_a, &fit_a).unwrap();
        assert!((&same.forward(&q) - &q).iter().all(|v| v.abs() < 1e-12));
        assert!(same.log_jacobian().abs() < 1e-12);
        let flat = vec![array![2.0, 0.0, 0.0], array![-2.0, 0.0, 0.0]];
        let flat_b = vec![array![0.0, 1.0, 0.0], array![0.0, -1.0, 0.0]];
        let degenerate = PlaneMap::fit(&full, &a, &a, &[], &flat, &flat_b).unwrap();
        assert!(degenerate.fallback);
    }
}
