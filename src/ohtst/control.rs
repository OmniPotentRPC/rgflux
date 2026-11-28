//! Stein control variates for the plane estimators.
//!
//! On a plane without walls the restricted Boltzmann density `ρ` decays at
//! infinity, so for any smooth in-plane field `φ`,
//! `⟨φ · P∇V⟩ = kT ⟨∇_∥ · φ⟩`, with `P` the projector onto the plane's
//! free directions. Take a Hessian `H₀` fixed before sampling, usually the
//! saddle's, and on a plane with normal `ν` set `H_∥ = P H₀ P`,
//! `h = P H₀ ν` and `a = H_∥⁻¹ h`. With `φ = a` and `φ = (h · r) H_∥⁻¹ e_j`,
//! `r = q - p`, the estimators
//!
//! `⟨ν · ∇V⟩ ≈ mean[ν · ∇V - a · P∇V]`,
//!
//! `-⟨(ν · ∇V) r⟩ ≈ mean[-(ν · ∇V) r + (h · r) H_∥⁻¹ P∇V] - kT a`
//!
//! are unbiased whatever `H₀` is. On a harmonic surface with Hessian `H₀`
//! the first has zero variance, and so does the second when `p` is the
//! in-plane mean, which is why the optimizer pivots on the previous
//! centroid. These are the zero-variance control variates of Assaraf and
//! Caffarel (Phys. Rev. Lett. 83, 4682 (1999)) and of Mira, Solgi and
//! Imparato (Stat. Comput. 23, 653 (2013)), in the form of Eqs. (6.2) and
//! (6.3) of the OH-TST design survey. The solves use only Hessian actions,
//! by conjugate gradients in the plane.

use ndarray::Array1;

use super::anharmonic::hessian_action;
use rgsaddle::{PointSurface, SaddleError};

/// A Hessian fixed before sampling, for the control variates of
/// [`sample_plane`](crate::ohtst::sample_plane): the Hessian of the
/// sampled surface at [`Self::point`], usually the saddle.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct HarmonicReference {
    /// Where the Hessian is taken, in the coordinates of the sampled
    /// surface.
    pub point: Array1<f64>,
    /// Gradient-difference step for Hessian actions when the surface has
    /// no analytic one, in its length unit.
    pub hvp_step: f64,
    /// Relative residual at which the conjugate-gradient solves stop.
    pub tolerance: f64,
}

impl HarmonicReference {
    /// The Hessian at `point`, with a difference step of `1e-4` and a
    /// solve tolerance of `1e-8`.
    #[must_use]
    pub fn new(point: Array1<f64>) -> Self {
        Self {
            point,
            hvp_step: 1.0e-4,
            tolerance: 1.0e-8,
        }
    }

    /// Set [`Self::point`].
    #[must_use]
    pub fn with_point(mut self, point: Array1<f64>) -> Self {
        self.point = point;
        self
    }

    /// Set [`Self::hvp_step`].
    #[must_use]
    pub fn with_hvp_step(mut self, hvp_step: f64) -> Self {
        self.hvp_step = hvp_step;
        self
    }

    /// Set [`Self::tolerance`].
    #[must_use]
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    pub(crate) fn check(&self) -> Result<(), SaddleError> {
        if !self.point.iter().all(|x| x.is_finite()) {
            return Err(SaddleError::NonFinite("harmonic reference point"));
        }
        if !(self.hvp_step.is_finite() && self.hvp_step > 0.0) {
            return Err(SaddleError::Invalid(
                "the reference difference step must be finite and positive".into(),
            ));
        }
        if !(self.tolerance > 0.0 && self.tolerance < 1.0) {
            return Err(SaddleError::Invalid(
                "the reference solve tolerance must lie in (0, 1)".into(),
            ));
        }
        Ok(())
    }
}

/// The reference Hessian restricted to one plane.
pub(crate) struct PlaneControl<'a, S> {
    surface: &'a S,
    reference: &'a HarmonicReference,
    held: Vec<Array1<f64>>,
    /// `h = P H₀ ν`.
    pub(crate) coupling: Array1<f64>,
    /// `a = H_∥⁻¹ h`.
    pub(crate) coefficient: Array1<f64>,
    pub(crate) evaluations: usize,
}

impl<'a, S: PointSurface> PlaneControl<'a, S> {
    /// `held` is orthonormal and starts with the plane normal. `None` when
    /// `H_∥` is not positive along the first solve.
    pub(crate) fn new(
        surface: &'a S,
        reference: &'a HarmonicReference,
        held: &[Array1<f64>],
    ) -> Result<Option<Self>, SaddleError> {
        let normal = &held[0];
        if reference.point.len() != normal.len() {
            return Err(SaddleError::Shape(
                "the harmonic reference differs in length from the plane".into(),
            ));
        }
        let mut control = Self {
            surface,
            reference,
            held: held.to_vec(),
            coupling: Array1::zeros(normal.len()),
            coefficient: Array1::zeros(normal.len()),
            evaluations: 0,
        };
        let mut coupling = control.apply(normal)?;
        control.project(&mut coupling);
        let Some(coefficient) = control.solve(&coupling)? else {
            return Ok(None);
        };
        control.coupling = coupling;
        control.coefficient = coefficient;
        Ok(Some(control))
    }

    fn project(&self, v: &mut Array1<f64>) {
        for direction in &self.held {
            let c = direction.dot(v);
            v.scaled_add(-c, direction);
        }
    }

    fn apply(&mut self, v: &Array1<f64>) -> Result<Array1<f64>, SaddleError> {
        hessian_action(
            self.surface,
            self.reference.point.view(),
            v,
            self.reference.hvp_step,
            &mut self.evaluations,
        )
    }

    /// `H_∥⁻¹ b` for `b` in the plane. `None` when a search direction meets
    /// non-positive curvature.
    pub(crate) fn solve(&mut self, b: &Array1<f64>) -> Result<Option<Array1<f64>>, SaddleError> {
        let (surface, reference) = (self.surface, self.reference);
        let held = std::mem::take(&mut self.held);
        let mut evaluations = 0;
        let solution = conjugate_gradients(
            |v| {
                hessian_action(
                    surface,
                    reference.point.view(),
                    v,
                    reference.hvp_step,
                    &mut evaluations,
                )
            },
            b,
            &held,
            reference.tolerance,
        );
        self.held = held;
        self.evaluations += evaluations;
        solution
    }
}

/// Solve `P H P x = P b` on the complement of the orthonormal `held`
/// directions by conjugate gradients, for `H` given by its action. Stops
/// at a relative residual of `tolerance` or after `2 m + 10` steps, `m`
/// the free dimension. `None` when a search direction meets non-positive
/// curvature.
pub(crate) fn conjugate_gradients<F>(
    mut apply: F,
    b: &Array1<f64>,
    held: &[Array1<f64>],
    tolerance: f64,
) -> Result<Option<Array1<f64>>, SaddleError>
where
    F: FnMut(&Array1<f64>) -> Result<Array1<f64>, SaddleError>,
{
    let project = |v: &mut Array1<f64>| {
        for direction in held {
            let c = direction.dot(v);
            v.scaled_add(-c, direction);
        }
    };
    let n = b.len();
    let mut r = b.clone();
    project(&mut r);
    let target = tolerance * r.dot(&r).sqrt();
    let mut x = Array1::zeros(n);
    if target == 0.0 {
        return Ok(Some(x));
    }
    let mut p = r.clone();
    let mut rr = r.dot(&r);
    let free = n.saturating_sub(held.len());
    for _ in 0..2 * free + 10 {
        let mut ap = apply(&p)?;
        project(&mut ap);
        let curvature = p.dot(&ap);
        if curvature.is_nan() || curvature <= 0.0 {
            return Ok(None);
        }
        let step = rr / curvature;
        x.scaled_add(step, &p);
        r.scaled_add(-step, &ap);
        let next = r.dot(&r);
        if next.sqrt() <= target {
            break;
        }
        p *= next / rr;
        p += &r;
        rr = next;
    }
    project(&mut x);
    Ok(Some(x))
}
