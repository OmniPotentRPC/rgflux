//! Paths of planes for the reversible work.
//!
//! A [`PlanePath`] is a chain of segments. Each is a smooth family of
//! planes `Σ(s)`, `s ∈ [0, 1]`, with the derivative `[dp/ds; dν/ds]` that
//! [`PlaneSample::directional`](crate::ohtst::PlaneSample::directional)
//! contracts with the gradient of the plane free energy. Section 7.1 of
//! the OH-TST design survey splits the barrier into two parts, and the
//! constructors cover both:
//!
//! - [`PlanePath::curve`]: PAFI planes along a cubic spline through path
//!   points, the pivot on the curve and the unit tangent as the normal
//!   (Swinburne and Marinica, Phys. Rev. Lett. 120, 135503 (2018)), from
//!   the reactant minimum to the saddle along the minimum energy path.
//!   Clamped end normals make the last plane the saddle plane itself.
//! - [`PlanePath::legs`]: straight legs between given planes, the point on
//!   the chord and the normal on the great circle, such as the geodesic
//!   from the saddle plane to the optimum.
//! - [`PlanePath::trace`]: straight legs through the optimizer's own
//!   iterates, the path the 2001 paper integrates along.
//!
//! [`PlanePath::then`] joins paths that share an end plane, so one run can
//! go from the reactant to the optimal plane, and MBAR can combine every
//! node of it.

use ndarray::{Array1, ArrayView1, s};

use crate::ohtst::optimize::PlaneTrace;
use rgsaddle::error::SaddleError;
use rgsaddle::projector::Hyperplane;

/// Consecutive planes closer than this, in normal and in offset, are one
/// plane.
const SAME_PLANE: f64 = 1.0e-12;

/// A chain of plane families, each traversed for `s` from 0 to 1.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct PlanePath {
    segments: Vec<Segment>,
}

#[derive(Clone, Debug)]
enum Segment {
    Leg(Leg),
    Curve(CurvePiece),
}

impl PlanePath {
    /// Straight legs through `planes` in order.
    ///
    /// # Errors
    ///
    /// [`SaddleError::Invalid`] for fewer than two planes or for two consecutive
    /// planes with opposite normals; [`SaddleError::Shape`] for planes of different
    /// dimension.
    pub fn legs(planes: &[Hyperplane]) -> Result<Self, SaddleError> {
        if planes.len() < 2 {
            return Err(SaddleError::Invalid(
                "a work path needs at least two planes".into(),
            ));
        }
        let segments = planes
            .windows(2)
            .map(|pair| Leg::new(&pair[0], &pair[1]).map(Segment::Leg))
            .collect::<Result<_, _>>()?;
        Ok(Self { segments })
    }

    /// Straight legs through the planes the optimizer accepted, its
    /// [`PlaneOptimum::trace`](crate::ohtst::PlaneOptimum::trace).
    /// Entries that did not move the plane, such as a pivot recentred
    /// within it, are dropped.
    ///
    /// # Errors
    ///
    /// [`SaddleError::Invalid`] when the trace never moved the plane (fewer than two
    /// distinct planes), and the errors of [`PlanePath::legs`].
    pub fn trace(trace: &[PlaneTrace]) -> Result<Self, SaddleError> {
        let mut planes: Vec<Hyperplane> = Vec::with_capacity(trace.len());
        for record in trace {
            if planes
                .last()
                .is_none_or(|last| !same_plane(last, &record.plane, SAME_PLANE))
            {
                planes.push(record.plane.clone());
            }
        }
        if planes.len() < 2 {
            return Err(SaddleError::Invalid(
                "the optimizer trace never moved the plane".into(),
            ));
        }
        Self::legs(&planes)
    }

    /// PAFI planes along the cubic spline through `points`, parametrized
    /// by chord length: the plane at a point of the curve passes through it
    /// with the unit tangent as normal. A given end normal clamps the
    /// tangent there; it must point along the path. A free end has zero
    /// curvature.
    ///
    /// `points` are the path points, in the coordinates of the sampled surface, and the
    /// end normals are unit-length or normalized on input.
    ///
    /// # Errors
    ///
    /// [`SaddleError::Invalid`] for fewer than two points, for repeated consecutive
    /// points, or for an end normal that vanishes or does not point along the path;
    /// [`SaddleError::Shape`] for points of different or zero length, non-finite
    /// points, or an end normal of the wrong length.
    pub fn curve(
        points: &[Array1<f64>],
        start_normal: Option<ArrayView1<f64>>,
        end_normal: Option<ArrayView1<f64>>,
    ) -> Result<Self, SaddleError> {
        let pieces = spline(points, start_normal, end_normal)?;
        Ok(Self {
            segments: pieces.into_iter().map(Segment::Curve).collect(),
        })
    }

    /// This path followed by `next`, which must start on the plane this
    /// one ends on.
    ///
    /// # Errors
    ///
    /// [`SaddleError::Shape`] when the planes differ in dimension; [`SaddleError::Invalid`]
    /// when `next` does not start on the plane this path ends on.
    pub fn then(mut self, next: PlanePath) -> Result<Self, SaddleError> {
        let (end, start) = (self.end()?, next.start()?);
        if end.dim() != start.dim() {
            return Err(SaddleError::Shape(
                "joined plane paths differ in dimension".into(),
            ));
        }
        let scale = 1.0 + end.point().dot(&end.point()).sqrt();
        if !same_plane(&end, &start, 1.0e-9 * scale) {
            return Err(SaddleError::Invalid(
                "a joined plane path does not start on the plane the first one ends on".into(),
            ));
        }
        self.segments.extend(next.segments);
        Ok(self)
    }

    /// Number of segments: one per leg or curve piece, at least one for a built
    /// path.
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// Whether the path has no segment; never true for a built path.
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// The first plane of the path, the plane at `s = 0` of segment 0.
    ///
    /// # Errors
    ///
    /// The errors of [`PlanePath::plane`].
    pub fn start(&self) -> Result<Hyperplane, SaddleError> {
        self.plane(0, 0.0)
    }

    /// The last plane of the path, the plane at `s = 1` of the last segment.
    ///
    /// # Errors
    ///
    /// The errors of [`PlanePath::plane`].
    pub fn end(&self) -> Result<Hyperplane, SaddleError> {
        self.plane(self.segments.len() - 1, 1.0)
    }

    /// Dimension of the planes.
    pub fn dim(&self) -> usize {
        match &self.segments[0] {
            Segment::Leg(leg) => leg.dp.len(),
            Segment::Curve(piece) => piece.y0.len(),
        }
    }

    /// The plane at `s` on segment `k`.
    ///
    /// A leg returns its end plane for `s` outside `[0, 1]`.
    ///
    /// # Errors
    ///
    /// [`SaddleError::Invalid`] when the plane normal vanishes along a curve segment.
    ///
    /// # Panics
    ///
    /// Panics when `k >= self.len()`.
    pub fn plane(&self, k: usize, s: f64) -> Result<Hyperplane, SaddleError> {
        match &self.segments[k] {
            Segment::Leg(leg) => leg.plane(s),
            Segment::Curve(piece) => piece.plane(s),
        }
    }

    /// `[dp/ds; dν/ds]` at `s` on segment `k`.
    ///
    /// # Errors
    ///
    /// [`SaddleError::Invalid`] when the curve has a vanishing tangent.
    ///
    /// # Panics
    ///
    /// Panics when `k >= self.len()`.
    pub fn tangent(&self, k: usize, s: f64) -> Result<Array1<f64>, SaddleError> {
        match &self.segments[k] {
            Segment::Leg(leg) => Ok(leg.tangent(s)),
            Segment::Curve(piece) => piece.tangent(s),
        }
    }
}

/// `a` and `b` have normals within `tolerance` and offsets within
/// `tolerance` along the normal.
fn same_plane(a: &Hyperplane, b: &Hyperplane, tolerance: f64) -> bool {
    let turned = &a.normal() - &b.normal();
    let offset = a.normal().dot(&(&b.point() - &a.point()));
    turned.dot(&turned).sqrt() <= tolerance && offset.abs() <= tolerance
}

/// A straight leg: the point on the chord, the normal on the great circle,
///
/// `p(s) = p₀ + s (p₁ - p₀)`, `ν(s) = cos(sΩ) ν₀ + sin(sΩ) w`,
///
/// with `Ω = arccos(ν₀ · ν₁)` and `w` the unit part of `ν₁` orthogonal to
/// `ν₀`.
#[derive(Clone, Debug)]
struct Leg {
    from: Hyperplane,
    to: Hyperplane,
    dp: Array1<f64>,
    w: Array1<f64>,
    angle: f64,
}

impl Leg {
    fn new(from: &Hyperplane, to: &Hyperplane) -> Result<Self, SaddleError> {
        if from.dim() != to.dim() {
            return Err(SaddleError::Shape(
                "planes on a work path differ in dimension".into(),
            ));
        }
        let cosine = from.normal().dot(&to.normal()).clamp(-1.0, 1.0);
        let mut w = &to.normal() - &(&from.normal() * cosine);
        let norm = w.dot(&w).sqrt();
        if norm <= 1.0e-12 {
            if cosine < 0.0 {
                return Err(SaddleError::Invalid(
                    "opposite normals on a work path; orient them with Hyperplane::oriented_toward"
                        .into(),
                ));
            }
            w.fill(0.0);
        } else {
            w /= norm;
        }
        Ok(Self {
            from: from.clone(),
            to: to.clone(),
            dp: &to.point() - &from.point(),
            w,
            angle: cosine.acos(),
        })
    }

    fn plane(&self, s: f64) -> Result<Hyperplane, SaddleError> {
        if s <= 0.0 {
            return Ok(self.from.clone());
        }
        if s >= 1.0 {
            return Ok(self.to.clone());
        }
        let theta = s * self.angle;
        Hyperplane::new(
            &self.from.point() + &(&self.dp * s),
            &self.from.normal() * theta.cos() + &self.w * theta.sin(),
        )
    }

    fn tangent(&self, s: f64) -> Array1<f64> {
        let n = self.dp.len();
        let theta = s * self.angle;
        let mut t = Array1::zeros(2 * n);
        t.slice_mut(s![..n]).assign(&self.dp);
        t.slice_mut(s![n..])
            .assign(&((&self.w * theta.cos() - &self.from.normal() * theta.sin()) * self.angle));
        t
    }
}

/// One knot interval of a cubic spline `X(t)`, `t ∈ [0, h]`, from its end
/// values and second derivatives.
#[derive(Clone, Debug)]
struct CurvePiece {
    y0: Array1<f64>,
    y1: Array1<f64>,
    m0: Array1<f64>,
    m1: Array1<f64>,
    h: f64,
}

impl CurvePiece {
    fn position(&self, t: f64) -> Array1<f64> {
        let (h, u) = (self.h, self.h - t);
        &self.m0 * (u.powi(3) / (6.0 * h))
            + &self.m1 * (t.powi(3) / (6.0 * h))
            + (&self.y0 / h - &self.m0 * (h / 6.0)) * u
            + (&self.y1 / h - &self.m1 * (h / 6.0)) * t
    }

    fn first(&self, t: f64) -> Array1<f64> {
        let (h, u) = (self.h, self.h - t);
        &self.m1 * (t * t / (2.0 * h)) - &self.m0 * (u * u / (2.0 * h)) + (&self.y1 - &self.y0) / h
            - (&self.m1 - &self.m0) * (h / 6.0)
    }

    fn second(&self, t: f64) -> Array1<f64> {
        (&self.m0 * (self.h - t) + &self.m1 * t) / self.h
    }

    fn plane(&self, s: f64) -> Result<Hyperplane, SaddleError> {
        let t = s.clamp(0.0, 1.0) * self.h;
        Hyperplane::new(self.position(t), self.first(t))
    }

    fn tangent(&self, s: f64) -> Result<Array1<f64>, SaddleError> {
        let t = s.clamp(0.0, 1.0) * self.h;
        let velocity = self.first(t);
        let speed = velocity.dot(&velocity).sqrt();
        if !(speed.is_finite() && speed > 0.0) {
            return Err(SaddleError::Invalid(
                "the path curve has a vanishing tangent".into(),
            ));
        }
        let normal = &velocity / speed;
        let mut turn = self.second(t);
        let along = turn.dot(&normal);
        turn.scaled_add(-along, &normal);
        let n = velocity.len();
        let mut tangent = Array1::zeros(2 * n);
        tangent.slice_mut(s![..n]).assign(&(&velocity * self.h));
        tangent
            .slice_mut(s![n..])
            .assign(&(turn * (self.h / speed)));
        Ok(tangent)
    }
}

/// The cubic spline through `points` at their chord lengths, clamped to
/// the given unit end tangents, natural at a free end.
fn spline(
    points: &[Array1<f64>],
    start: Option<ArrayView1<f64>>,
    end: Option<ArrayView1<f64>>,
) -> Result<Vec<CurvePiece>, SaddleError> {
    if points.len() < 2 {
        return Err(SaddleError::Invalid(
            "a path curve needs at least two points".into(),
        ));
    }
    let n = points[0].len();
    if n == 0
        || points
            .iter()
            .any(|p| p.len() != n || !p.iter().all(|v| v.is_finite()))
    {
        return Err(SaddleError::Shape(
            "path points must share one finite, nonzero length".into(),
        ));
    }
    let h: Vec<f64> = points
        .windows(2)
        .map(|pair| {
            let d = &pair[1] - &pair[0];
            d.dot(&d).sqrt()
        })
        .collect();
    if h.iter().any(|&step| !(step.is_finite() && step > 0.0)) {
        return Err(SaddleError::Invalid(
            "consecutive path points must be distinct".into(),
        ));
    }
    let unit = |normal: Option<ArrayView1<f64>>, chord: Array1<f64>| {
        normal
            .map(|v| {
                if v.len() != n {
                    return Err(SaddleError::Shape(
                        "an end normal differs in length from the path".into(),
                    ));
                }
                let norm = v.dot(&v).sqrt();
                if !(norm.is_finite() && norm > 0.0) {
                    return Err(SaddleError::Invalid("an end normal vanishes".into()));
                }
                if v.dot(&chord) <= 0.0 {
                    return Err(SaddleError::Invalid(
                        "an end normal must point along the path".into(),
                    ));
                }
                Ok(&v / norm)
            })
            .transpose()
    };
    let k = points.len() - 1;
    let start = unit(start, &points[1] - &points[0])?;
    let end = unit(end, &points[k] - &points[k - 1])?;

    // Second derivatives M solve a tridiagonal system shared by every
    // coordinate.
    let size = k + 1;
    let mut lower = vec![0.0; size];
    let mut diagonal = vec![1.0; size];
    let mut upper = vec![0.0; size];
    let mut rhs: Vec<Array1<f64>> = vec![Array1::zeros(n); size];
    let slope = |i: usize| (&points[i + 1] - &points[i]) / h[i];
    for i in 1..k {
        lower[i] = h[i - 1];
        diagonal[i] = 2.0 * (h[i - 1] + h[i]);
        upper[i] = h[i];
        rhs[i] = (slope(i) - slope(i - 1)) * 6.0;
    }
    if let Some(d) = &start {
        diagonal[0] = 2.0 * h[0];
        upper[0] = h[0];
        rhs[0] = (slope(0) - d) * 6.0;
    }
    if let Some(d) = &end {
        lower[k] = h[k - 1];
        diagonal[k] = 2.0 * h[k - 1];
        rhs[k] = (d - &slope(k - 1)) * 6.0;
    }
    for i in 1..size {
        let w = lower[i] / diagonal[i - 1];
        diagonal[i] -= w * upper[i - 1];
        let previous = rhs[i - 1].clone();
        rhs[i].scaled_add(-w, &previous);
    }
    let mut second = vec![Array1::zeros(n); size];
    second[k] = &rhs[k] / diagonal[k];
    for i in (0..k).rev() {
        second[i] = (&rhs[i] - &(&second[i + 1] * upper[i])) / diagonal[i];
    }
    Ok((0..k)
        .map(|i| CurvePiece {
            y0: points[i].clone(),
            y1: points[i + 1].clone(),
            m0: second[i].clone(),
            m1: second[i + 1].clone(),
            h: h[i],
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// On a circle through equally spaced points, the PAFI normals turn
    /// with the radius and the clamped ends hold the given tangents.
    #[test]
    fn curve_follows_a_circle_and_holds_its_end_normals() {
        let points: Vec<Array1<f64>> = (0..=8)
            .map(|i| {
                let a = std::f64::consts::FRAC_PI_2 * i as f64 / 8.0;
                array![a.cos(), a.sin()]
            })
            .collect();
        let path = PlanePath::curve(
            &points,
            Some(array![0.0, 1.0].view()),
            Some(array![-1.0, 0.0].view()),
        )
        .unwrap();
        assert_eq!(path.len(), 8);
        let first = path.start().unwrap();
        assert!((first.normal()[1] - 1.0).abs() < 1e-14);
        let last = path.end().unwrap();
        assert!((last.normal()[0] + 1.0).abs() < 1e-14);
        assert!((&last.point() - &points[8]).iter().all(|v| v.abs() < 1e-14));
        for k in 0..8 {
            for s in [0.0, 0.3, 0.7] {
                let plane = path.plane(k, s).unwrap();
                let radius = plane.point().dot(&plane.point()).sqrt();
                assert!((radius - 1.0).abs() < 2e-4, "{radius}");
                assert!(plane.normal().dot(&plane.point()).abs() < 2e-3);
            }
        }
        // The normal turns by π/2 over the quarter circle, at the rate the
        // tangent reports.
        let mut turned = 0.0;
        for k in 0..8 {
            let steps = 64;
            for j in 0..steps {
                let s = (j as f64 + 0.5) / steps as f64;
                let t = path.tangent(k, s).unwrap();
                turned += t.slice(s![2..]).dot(&t.slice(s![2..])).sqrt() / steps as f64;
            }
        }
        assert!(
            (turned - std::f64::consts::FRAC_PI_2).abs() < 1e-3,
            "{turned}"
        );
    }

    /// The tangent is the derivative of the plane along the path.
    #[test]
    fn curve_tangent_is_the_derivative_of_the_plane() {
        let points = vec![
            array![0.0, 0.0, 0.0],
            array![0.5, 0.4, 0.1],
            array![1.2, 0.5, -0.2],
            array![1.6, 1.1, 0.0],
        ];
        let path = PlanePath::curve(&points, None, Some(array![0.2, 1.0, 0.3].view())).unwrap();
        let h = 1.0e-6;
        for k in 0..path.len() {
            for s in [0.2, 0.5, 0.8] {
                let (a, b) = (path.plane(k, s - h).unwrap(), path.plane(k, s + h).unwrap());
                let dp = (&b.point() - &a.point()) / (2.0 * h);
                let dn = (&b.normal() - &a.normal()) / (2.0 * h);
                let t = path.tangent(k, s).unwrap();
                assert!((&t.slice(s![..3]) - &dp).iter().all(|v| v.abs() < 1e-7));
                assert!((&t.slice(s![3..]) - &dn).iter().all(|v| v.abs() < 1e-7));
            }
        }
    }

    #[test]
    fn paths_join_only_on_a_shared_plane() {
        let a = Hyperplane::new(array![0.0, 0.0], array![1.0, 0.0]).unwrap();
        let b = Hyperplane::new(array![1.0, 0.0], array![1.0, 1.0]).unwrap();
        let c = Hyperplane::new(array![1.5, 0.5], array![0.0, 1.0]).unwrap();
        let joined = PlanePath::legs(&[a.clone(), b.clone()])
            .unwrap()
            .then(PlanePath::legs(&[b.recentered(array![0.5, 0.5].view()), c.clone()]).unwrap())
            .unwrap();
        assert_eq!(joined.len(), 2);
        assert_eq!(joined.end().unwrap(), c);
        assert!(
            PlanePath::legs(&[a.clone(), b])
                .unwrap()
                .then(PlanePath::legs(&[c.clone(), a.clone()]).unwrap())
                .is_err()
        );
        assert!(PlanePath::curve(&[array![0.0, 0.0]], None, None).is_err());
        assert!(
            PlanePath::curve(
                &[array![0.0, 0.0], array![1.0, 0.0]],
                Some(array![-1.0, 0.0].view()),
                None
            )
            .is_err()
        );
    }
}
