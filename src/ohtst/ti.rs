//! Thermodynamic integration on one fixed hyperplane.

use ndarray::{Array1, Array2, ArrayView1};

use rgsaddle::error::SaddleError;
pub use rgsaddle::hyperplane_ti::{
    THERMODYNAMIC_DERIVATIVE, ThermodynamicIntegrationConfig, ThermodynamicIntegrationEstimate,
    ThermodynamicIntegrationNode, restricted_thermodynamic_integration, thermodynamic_integration,
};
use rgsaddle::minmode::PointSurface;
use rgsaddle::projector::{
    Hyperplane, HyperplaneEnsemble, HyperplaneSampleBudget, HyperplaneSampler,
};

use super::anharmonic::hessian_action;
use super::control::HarmonicReference;
use super::gaussian::HyperplaneObservable;
use super::{AnalyticSurfaceSampler, LangevinConfig};

/// Second-order Taylor expansion of a [`PointSurface`] about a
/// [`HarmonicReference`] point.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct HarmonicExpansion<'a, S> {
    /// Full potential whose Hessian defines the reference.
    pub surface: &'a S,
    /// Expansion point and Hessian-action step.
    pub reference: HarmonicReference,
}

impl<'a, S> HarmonicExpansion<'a, S> {
    /// Build the expansion at `reference.point`.
    #[must_use]
    pub fn new(surface: &'a S, reference: HarmonicReference) -> Self {
        Self { surface, reference }
    }

    /// Set [`Self::reference`].
    #[must_use]
    pub fn with_reference(mut self, reference: HarmonicReference) -> Self {
        self.reference = reference;
        self
    }
}

/// Energy of the Taylor harmonic model at `position`.
pub fn harmonic_energy_at<S: PointSurface>(
    surface: &S,
    reference: &HarmonicReference,
    position: ArrayView1<f64>,
    evaluations: &mut usize,
) -> Result<f64, SaddleError> {
    reference.check()?;
    if position.len() != reference.point.len() {
        return Err(SaddleError::Shape(
            "the harmonic reference differs in length from the position".into(),
        ));
    }
    let displacement = position.to_owned() - &reference.point;
    let (energy, gradient) = surface.eval(reference.point.view())?;
    *evaluations += 1;
    if gradient.len() != reference.point.len() {
        return Err(SaddleError::Shape(
            "the harmonic reference gradient differs in length from the position".into(),
        ));
    }
    if !energy.is_finite() || !gradient.iter().all(|value| value.is_finite()) {
        return Err(SaddleError::NonFinite(
            "energy or gradient at the harmonic reference",
        ));
    }
    let image = hessian_action(
        surface,
        reference.point.view(),
        &displacement,
        reference.hvp_step,
        evaluations,
    )?;
    let value = energy + gradient.dot(&displacement) + 0.5 * displacement.dot(&image);
    if !value.is_finite() {
        return Err(SaddleError::NonFinite("harmonic reference energy"));
    }
    Ok(value)
}

impl<S: PointSurface> PointSurface for HarmonicExpansion<'_, S> {
    fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.reference.check()?;
        if q.len() != self.reference.point.len() {
            return Err(SaddleError::Shape(
                "the harmonic expansion differs in length from the position".into(),
            ));
        }
        let displacement = q.to_owned() - &self.reference.point;
        let (energy, gradient) = self.surface.eval(self.reference.point.view())?;
        if gradient.len() != q.len() {
            return Err(SaddleError::Shape(
                "the harmonic expansion gradient differs in length from the position".into(),
            ));
        }
        let mut evaluations = 0;
        let image = hessian_action(
            self.surface,
            self.reference.point.view(),
            &displacement,
            self.reference.hvp_step,
            &mut evaluations,
        )?;
        let expanded = energy + gradient.dot(&displacement) + 0.5 * displacement.dot(&image);
        let expanded_gradient = &gradient + &image;
        if !expanded.is_finite() || !expanded_gradient.iter().all(|value| value.is_finite()) {
            return Err(SaddleError::NonFinite("harmonic expansion"));
        }
        Ok((expanded, expanded_gradient))
    }

    fn hessian_vector(
        &self,
        q: ArrayView1<f64>,
        vector: ArrayView1<f64>,
    ) -> Result<Option<Array1<f64>>, SaddleError> {
        self.reference.check()?;
        if q.len() != self.reference.point.len() || vector.len() != self.reference.point.len() {
            return Err(SaddleError::Shape(
                "the harmonic expansion Hessian action differs in length from the reference".into(),
            ));
        }
        let mut evaluations = 0;
        hessian_action(
            self.surface,
            self.reference.point.view(),
            &vector.to_owned(),
            self.reference.hvp_step,
            &mut evaluations,
        )
        .map(Some)
    }

    fn excluded_modes(&self, q: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        self.surface.excluded_modes(q)
    }
}

/// Mixed potential `V_lambda = (1-lambda) V_harmonic + lambda V_full`.
pub(crate) struct LambdaSurface<'a, S> {
    full: &'a S,
    harmonic: HarmonicExpansion<'a, S>,
    lambda: f64,
}

impl<'a, S: PointSurface> LambdaSurface<'a, S> {
    pub(crate) fn new(
        full: &'a S,
        reference: HarmonicReference,
        lambda: f64,
    ) -> Result<Self, SaddleError> {
        reference.check()?;
        if !(0.0..=1.0).contains(&lambda) {
            return Err(SaddleError::Invalid(
                "thermodynamic lambda must lie in [0, 1]".into(),
            ));
        }
        Ok(Self {
            full,
            harmonic: HarmonicExpansion::new(full, reference),
            lambda,
        })
    }
}

impl<S: PointSurface> PointSurface for LambdaSurface<'_, S> {
    fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (harmonic_energy, harmonic_gradient) = self.harmonic.eval(q)?;
        let (full_energy, full_gradient) = self.full.eval(q)?;
        let lambda = self.lambda;
        Ok((
            harmonic_energy.mul_add(1.0 - lambda, lambda * full_energy),
            &harmonic_gradient * (1.0 - lambda) + &full_gradient * lambda,
        ))
    }

    fn excluded_modes(&self, q: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        self.full.excluded_modes(q)
    }
}

/// Observable `V_full - V_harmonic` at a sampled configuration.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct ThermodynamicDerivative<'a, S> {
    /// Full potential in the sampling coordinates.
    pub full: &'a S,
    /// Expansion point for the harmonic reference.
    pub reference: HarmonicReference,
}

impl<'a, S> ThermodynamicDerivative<'a, S> {
    /// Name the observable [`THERMODYNAMIC_DERIVATIVE`].
    #[must_use]
    pub fn new(full: &'a S, reference: HarmonicReference) -> Self {
        Self { full, reference }
    }

    /// Set [`Self::reference`].
    #[must_use]
    pub fn with_reference(mut self, reference: HarmonicReference) -> Self {
        self.reference = reference;
        self
    }
}

impl<S: PointSurface> HyperplaneObservable for ThermodynamicDerivative<'_, S> {
    fn observe(&self, position: ArrayView1<f64>) -> Result<f64, SaddleError> {
        let mut evaluations = 0;
        let harmonic = harmonic_energy_at(self.full, &self.reference, position, &mut evaluations)?;
        let (full, _) = self.full.eval(position)?;
        let value = full - harmonic;
        if !value.is_finite() {
            return Err(SaddleError::NonFinite("thermodynamic derivative"));
        }
        Ok(value)
    }
}

/// Records `V_full - V_harmonic` from positions and attaches the observable.
pub struct ThermodynamicSampler<'a, E, S> {
    inner: E,
    derivative: ThermodynamicDerivative<'a, S>,
}

impl<'a, E, S> ThermodynamicSampler<'a, E, S> {
    /// Wrap `inner` with `derivative`.
    pub fn new(inner: E, derivative: ThermodynamicDerivative<'a, S>) -> Self {
        Self { inner, derivative }
    }

    /// Borrow the wrapped sampler.
    pub fn inner(&self) -> &E {
        &self.inner
    }

    /// Borrow the wrapped sampler mutably.
    pub fn inner_mut(&mut self) -> &mut E {
        &mut self.inner
    }
}

impl<'a, E: HyperplaneSampler, S: PointSurface> HyperplaneSampler
    for ThermodynamicSampler<'a, E, S>
{
    fn sample_hyperplane(
        &mut self,
        plane: &Hyperplane,
        budget: &HyperplaneSampleBudget,
    ) -> Result<HyperplaneEnsemble, SaddleError> {
        let requested = budget.lambda.is_some()
            || budget
                .observables
                .iter()
                .any(|name| name == THERMODYNAMIC_DERIVATIVE);
        let mut inner_budget = budget.clone();
        if requested
            && !inner_budget
                .observables
                .iter()
                .any(|name| name == THERMODYNAMIC_DERIVATIVE)
        {
            inner_budget
                .observables
                .push(THERMODYNAMIC_DERIVATIVE.to_string());
        }
        if requested {
            inner_budget.positions = true;
        }
        let mut ensemble = self.inner.sample_hyperplane(plane, &inner_budget)?;
        if !requested {
            return Ok(ensemble);
        }
        for observation in &mut ensemble.observations {
            if !observation
                .observables
                .contains_key(THERMODYNAMIC_DERIVATIVE)
            {
                let position = observation.position.as_ref().ok_or_else(|| {
                    SaddleError::Invalid(
                        "thermodynamic derivative needs positions in sampling records".into(),
                    )
                })?;
                let value = self.derivative.observe(position.view())?;
                observation
                    .observables
                    .insert(THERMODYNAMIC_DERIVATIVE.into(), value);
            }
            if !budget.positions {
                observation.position = None;
            }
        }
        Ok(ensemble)
    }

    fn energy(&mut self, position: ArrayView1<f64>) -> Result<Option<f64>, SaddleError> {
        self.inner.energy(position)
    }

    fn constrained_directions(
        &mut self,
        plane: &Hyperplane,
    ) -> Result<Vec<Array1<f64>>, SaddleError> {
        self.inner.constrained_directions(plane)
    }
}

/// Projected BAOAB/GHMC sampler on `V_lambda` built from a harmonic Taylor
/// expansion of `full`.
pub struct ThermodynamicSurfaceSampler<'a, S> {
    full: &'a S,
    reference: HarmonicReference,
    dynamics: LangevinConfig,
}

impl<'a, S: PointSurface> ThermodynamicSurfaceSampler<'a, S> {
    /// Configure Langevin dynamics with a harmonic reference at the basin
    /// minimum or saddle plane.
    pub fn new(
        full: &'a S,
        reference: HarmonicReference,
        dynamics: LangevinConfig,
    ) -> Result<Self, SaddleError> {
        reference.check()?;
        Ok(Self {
            full,
            reference,
            dynamics,
        })
    }
}

impl<S: PointSurface> HyperplaneSampler for ThermodynamicSurfaceSampler<'_, S> {
    fn sample_hyperplane(
        &mut self,
        plane: &Hyperplane,
        budget: &HyperplaneSampleBudget,
    ) -> Result<HyperplaneEnsemble, SaddleError> {
        let lambda = budget.lambda.ok_or_else(|| {
            SaddleError::Invalid("thermodynamic surface sampling needs a lambda request".into())
        })?;
        let surface = LambdaSurface::new(self.full, self.reference.clone(), lambda)?;
        let mut inner_budget = budget.clone();
        inner_budget.positions = true;
        let mut sampler = AnalyticSurfaceSampler::new(&surface, self.dynamics.clone());
        let mut ensemble = sampler.sample_hyperplane(plane, &inner_budget)?;
        let derivative = ThermodynamicDerivative::new(self.full, self.reference.clone());
        for observation in &mut ensemble.observations {
            let position = observation.position.as_ref().ok_or_else(|| {
                SaddleError::Invalid("thermodynamic surface sampler omitted a position".into())
            })?;
            let value = derivative.observe(position.view())?;
            observation
                .observables
                .insert(THERMODYNAMIC_DERIVATIVE.into(), value);
            if !budget.positions {
                observation.position = None;
            }
        }
        Ok(ensemble)
    }

    fn energy(&mut self, position: ArrayView1<f64>) -> Result<Option<f64>, SaddleError> {
        Ok(Some(self.full.eval(position)?.0))
    }

    fn constrained_directions(
        &mut self,
        plane: &Hyperplane,
    ) -> Result<Vec<Array1<f64>>, SaddleError> {
        AnalyticSurfaceSampler::new(self.full, self.dynamics.clone()).constrained_directions(plane)
    }
}

/// Analytic reference backend for
/// `V_lambda = (1-lambda) V_harmonic + lambda V_full`.
pub struct AnalyticThermodynamicSampler<'a, H, F> {
    harmonic: &'a H,
    full: &'a F,
    dynamics: LangevinConfig,
}

impl<'a, H: PointSurface, F: PointSurface> AnalyticThermodynamicSampler<'a, H, F> {
    /// Build the reference backend.
    pub fn new(harmonic: &'a H, full: &'a F, dynamics: LangevinConfig) -> Self {
        Self {
            harmonic,
            full,
            dynamics,
        }
    }
}

struct Interpolated<'a, H, F> {
    harmonic: &'a H,
    full: &'a F,
    lambda: f64,
}

impl<H: PointSurface, F: PointSurface> PointSurface for Interpolated<'_, H, F> {
    fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (harmonic_energy, harmonic_gradient) = self.harmonic.eval(q)?;
        let (full_energy, full_gradient) = self.full.eval(q)?;
        let lambda = self.lambda;
        Ok((
            (1.0 - lambda) * harmonic_energy + lambda * full_energy,
            harmonic_gradient * (1.0 - lambda) + full_gradient * lambda,
        ))
    }

    fn excluded_modes(&self, q: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        self.full.excluded_modes(q)
    }
}

impl<H: PointSurface, F: PointSurface> HyperplaneSampler
    for AnalyticThermodynamicSampler<'_, H, F>
{
    fn sample_hyperplane(
        &mut self,
        plane: &Hyperplane,
        budget: &HyperplaneSampleBudget,
    ) -> Result<HyperplaneEnsemble, SaddleError> {
        let lambda = budget.lambda.ok_or_else(|| {
            SaddleError::Invalid("analytic TI sampling needs a lambda request".into())
        })?;
        if !(0.0..=1.0).contains(&lambda) {
            return Err(SaddleError::Invalid(
                "thermodynamic lambda must lie in [0, 1]".into(),
            ));
        }
        let surface = Interpolated {
            harmonic: self.harmonic,
            full: self.full,
            lambda,
        };
        let mut inner_budget = budget.clone();
        inner_budget.positions = true;
        let mut sampler = AnalyticSurfaceSampler::new(&surface, self.dynamics.clone());
        let mut ensemble = sampler.sample_hyperplane(plane, &inner_budget)?;
        for observation in &mut ensemble.observations {
            let position = observation.position.as_ref().ok_or_else(|| {
                SaddleError::Invalid("analytic TI sampler omitted a position".into())
            })?;
            let (harmonic, _) = self.harmonic.eval(position.view())?;
            let (full, _) = self.full.eval(position.view())?;
            ensemble.evaluations += 2;
            observation
                .observables
                .insert(THERMODYNAMIC_DERIVATIVE.into(), full - harmonic);
            if !budget.positions {
                observation.position = None;
            }
        }
        Ok(ensemble)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;
    use crate::ohtst::{ExactGaussianPlaneSampler, QuadraticModel};
    use rgsaddle::minmode::PointSurface;
    use rgsaddle::projector::HyperplaneObservation;

    struct Harmonic;

    impl PointSurface for Harmonic {
        fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            Ok((0.5 * q[1] * q[1], array![0.0, q[1]]))
        }
    }

    #[derive(Clone, Copy)]
    struct Quartic;

    impl PointSurface for Quartic {
        fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            Ok((0.25 * q[1].powi(4), array![0.0, q[1].powi(3)]))
        }
    }

    #[test]
    fn thermodynamic_derivative_matches_full_minus_harmonic_energy() {
        let reference = HarmonicReference::new(array![0.0, 0.0]);
        let position = array![0.2, 0.4];
        let observable = ThermodynamicDerivative::new(&Quartic, reference.clone());
        let observed = observable.observe(position.view()).unwrap();
        let (full, _) = Quartic.eval(position.view()).unwrap();
        let mut evaluations = 0;
        let harmonic =
            harmonic_energy_at(&Quartic, &reference, position.view(), &mut evaluations).unwrap();
        assert_eq!(observed, full - harmonic);
    }

    #[test]
    fn thermodynamic_sampler_fills_the_observable_only_when_requested() {
        struct Bare {
            supply_value: Option<f64>,
        }

        impl HyperplaneSampler for Bare {
            fn sample_hyperplane(
                &mut self,
                _plane: &Hyperplane,
                budget: &HyperplaneSampleBudget,
            ) -> Result<HyperplaneEnsemble, SaddleError> {
                let mut observation = HyperplaneObservation::default();
                if budget.positions {
                    observation.position = Some(array![0.0, 0.3]);
                }
                if let Some(value) = self.supply_value {
                    observation
                        .observables
                        .insert(THERMODYNAMIC_DERIVATIVE.into(), value);
                }
                Ok(HyperplaneEnsemble::new(vec![observation], 3, 1.0))
            }
        }

        let derivative =
            ThermodynamicDerivative::new(&Quartic, HarmonicReference::new(array![0.0, 0.0]));
        let plane = Hyperplane::new(array![0.0, 0.0], array![1.0, 0.0]).unwrap();
        let idle = HyperplaneSampleBudget::default()
            .with_positions(false)
            .with_samples(1);
        let mut untouched =
            ThermodynamicSampler::new(Bare { supply_value: None }, derivative.clone());
        let ensemble = untouched.sample_hyperplane(&plane, &idle).unwrap();
        assert!(ensemble.observations[0].position.is_none());
        assert!(ensemble.observations[0].observables.is_empty());
        assert_eq!(ensemble.evaluations, 3);

        let mut filled = ThermodynamicSampler::new(Bare { supply_value: None }, derivative.clone());
        let requested = idle.clone().with_lambda(0.25);
        let ensemble = filled.sample_hyperplane(&plane, &requested).unwrap();
        assert!(ensemble.observations[0].position.is_none());
        let value = ensemble.observations[0].observables[THERMODYNAMIC_DERIVATIVE];
        let expected = derivative.observe(array![0.0, 0.3].view()).unwrap();
        assert_eq!(value, expected);

        let mut kept = ThermodynamicSampler::new(
            Bare {
                supply_value: Some(1.25),
            },
            derivative,
        );
        let ensemble = kept.sample_hyperplane(&plane, &requested).unwrap();
        assert!(ensemble.observations[0].position.is_none());
        assert_eq!(
            ensemble.observations[0].observables[THERMODYNAMIC_DERIVATIVE],
            1.25
        );
    }

    #[test]
    fn exact_gaussian_gives_zero_for_an_identical_reference() {
        let model = QuadraticModel::new(array![0.0, 0.0], array![[1.0, 0.0], [0.0, 2.0]]);
        let difference = |_q: ArrayView1<f64>| Ok(0.0);
        let mut sampler = ExactGaussianPlaneSampler::new(model)
            .unwrap()
            .with_observable(THERMODYNAMIC_DERIVATIVE, &difference);
        let plane = Hyperplane::new(array![0.0, 0.0], array![1.0, 0.0]).unwrap();
        let config = ThermodynamicIntegrationConfig::default()
            .with_nodes(4)
            .with_samples(200)
            .with_blocks(4);
        let estimate = thermodynamic_integration(&mut sampler, &plane, &config).unwrap();
        assert_eq!(estimate.free_energy, 0.0);
        assert_eq!(estimate.error, 0.0);
    }

    #[test]
    fn restricted_spelling_uses_the_same_sampler_estimator() {
        let model = QuadraticModel::new(array![0.0, 0.0], array![[1.0, 0.0], [0.0, 2.0]]);
        let difference = |_q: ArrayView1<f64>| Ok(0.0);
        let mut sampler = ExactGaussianPlaneSampler::new(model)
            .unwrap()
            .with_observable(THERMODYNAMIC_DERIVATIVE, &difference);
        let plane = Hyperplane::new(array![0.0, 0.0], array![1.0, 0.0]).unwrap();
        let config = ThermodynamicIntegrationConfig::default()
            .with_nodes(2)
            .with_samples(100)
            .with_blocks(4);
        let estimate = restricted_thermodynamic_integration(&mut sampler, &plane, &config).unwrap();
        assert_eq!(estimate.free_energy, 0.0);
    }

    #[test]
    fn anharmonic_free_energy_matches_the_partition_functions() {
        let thermal = 0.5;
        let dynamics = LangevinConfig::default()
            .with_thermal(thermal)
            .with_time_step(0.03)
            .with_friction(2.0);
        let mut sampler = AnalyticThermodynamicSampler::new(&Harmonic, &Quartic, dynamics);
        let plane = Hyperplane::new(array![0.0, 0.0], array![1.0, 0.0]).unwrap();
        let config = ThermodynamicIntegrationConfig::default()
            .with_thermal(thermal)
            .with_nodes(6)
            .with_thermalize(500)
            .with_stride(8)
            .with_samples(4_000)
            .with_blocks(16)
            .with_seed(41);
        let estimate = thermodynamic_integration(&mut sampler, &plane, &config).unwrap();
        // Direct endpoint partition integrals, evaluated independently of TI.
        let width = 8.0;
        let intervals = 200_000;
        let integrate = |potential: &dyn Fn(f64) -> f64| {
            let h = 2.0 * width / intervals as f64;
            (0..=intervals)
                .map(|index| {
                    let y = -width + index as f64 * h;
                    let coefficient = if index == 0 || index == intervals {
                        1.0
                    } else if index % 2 == 0 {
                        2.0
                    } else {
                        4.0
                    };
                    coefficient * (-potential(y) / thermal).exp()
                })
                .sum::<f64>()
                * h
                / 3.0
        };
        let harmonic = integrate(&|y| 0.5 * y * y);
        let quartic = integrate(&|y| 0.25 * y.powi(4));
        let expected = -thermal * (quartic / harmonic).ln();
        assert!(
            (estimate.free_energy - expected).abs() < 4.0 * estimate.error + 0.01,
            "{} ± {} against {expected}",
            estimate.free_energy,
            estimate.error
        );
    }

    #[test]
    fn hessian_vector_rejects_the_wrong_length() {
        let surface = Harmonic;
        let expansion = HarmonicExpansion::new(&surface, HarmonicReference::new(array![0.0, 0.0]));
        let short_vector =
            PointSurface::hessian_vector(&expansion, array![0.0, 0.0].view(), array![1.0].view());
        assert!(
            matches!(short_vector, Err(SaddleError::Shape(_))),
            "{short_vector:?}"
        );
        let short_point =
            PointSurface::hessian_vector(&expansion, array![0.0].view(), array![1.0, 0.0].view());
        assert!(
            matches!(short_point, Err(SaddleError::Shape(_))),
            "{short_point:?}"
        );
    }
}
