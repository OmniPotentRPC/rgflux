//! Exact constrained sampling of quadratic surfaces.

use std::collections::BTreeMap;

use ndarray::{Array1, Array2, ArrayView1};

use rgsaddle::error::SaddleError;
use rgsaddle::minmode::{MinModeConfig, PointSurface};
use rgsaddle::projector::{
    Hyperplane, HyperplaneEnsemble, HyperplaneObservation, HyperplaneSampleBudget,
    HyperplaneSampler, TangentProjector,
};
use rgsaddle::stats::Rng;

/// A quadratic potential
/// `V(q) = energy + gradient·(q-center) + ½(q-center)ᵀ H (q-center)`.
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct QuadraticModel {
    /// Expansion point.
    pub center: Array1<f64>,
    /// Potential at [`Self::center`].
    pub energy: f64,
    /// Gradient at [`Self::center`].
    pub gradient: Array1<f64>,
    /// Symmetric Hessian.
    pub hessian: Array2<f64>,
}

impl QuadraticModel {
    /// Construct a zero-gradient quadratic model.
    pub fn new(center: Array1<f64>, hessian: Array2<f64>) -> Self {
        let gradient = Array1::zeros(center.len());
        Self {
            center,
            energy: 0.0,
            gradient,
            hessian,
        }
    }

    /// Set [`Self::center`].
    #[must_use]
    pub fn with_center(mut self, center: Array1<f64>) -> Self {
        self.center = center;
        self
    }

    /// Set [`Self::energy`].
    #[must_use]
    pub fn with_energy(mut self, energy: f64) -> Self {
        self.energy = energy;
        self
    }

    /// Set [`Self::gradient`].
    #[must_use]
    pub fn with_gradient(mut self, gradient: Array1<f64>) -> Self {
        self.gradient = gradient;
        self
    }

    /// Set [`Self::hessian`].
    #[must_use]
    pub fn with_hessian(mut self, hessian: Array2<f64>) -> Self {
        self.hessian = hessian;
        self
    }

    fn check(&self) -> Result<(), SaddleError> {
        let n = self.center.len();
        if n == 0 || self.gradient.len() != n || self.hessian.dim() != (n, n) {
            return Err(SaddleError::Shape(
                "a quadratic model needs a square Hessian and matching vectors".into(),
            ));
        }
        if !self.energy.is_finite()
            || self
                .center
                .iter()
                .chain(&self.gradient)
                .chain(&self.hessian)
                .any(|value| !value.is_finite())
        {
            return Err(SaddleError::NonFinite("quadratic model"));
        }
        for row in 0..n {
            for column in row + 1..n {
                let scale = self.hessian[(row, column)]
                    .abs()
                    .max(self.hessian[(column, row)].abs())
                    .max(1.0);
                if (self.hessian[(row, column)] - self.hessian[(column, row)]).abs()
                    > 1.0e-12 * scale
                {
                    return Err(SaddleError::Invalid(
                        "the quadratic Hessian must be symmetric".into(),
                    ));
                }
            }
        }
        Ok(())
    }
}

impl PointSurface for QuadraticModel {
    fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.check()?;
        if q.len() != self.center.len() {
            return Err(SaddleError::Shape(
                "quadratic point differs in length from the model".into(),
            ));
        }
        let displacement = &q - &self.center;
        let image = self.hessian.dot(&displacement);
        let gradient = &self.gradient + &image;
        let energy =
            self.energy + self.gradient.dot(&displacement) + 0.5 * displacement.dot(&image);
        Ok((energy, gradient))
    }

    fn hessian_vector(
        &self,
        q: ArrayView1<f64>,
        vector: ArrayView1<f64>,
    ) -> Result<Option<Array1<f64>>, SaddleError> {
        self.check()?;
        if q.len() != self.center.len() || vector.len() != self.center.len() {
            return Err(SaddleError::Shape(
                "quadratic Hessian action differs in length from the model".into(),
            ));
        }
        Ok(Some(self.hessian.dot(&vector)))
    }
}

/// A named scalar measured at an exact Gaussian sample.
pub trait HyperplaneObservable: Send + Sync {
    /// Evaluate the scalar at `position`.
    fn observe(&self, position: ArrayView1<f64>) -> Result<f64, SaddleError>;
}

impl<F> HyperplaneObservable for F
where
    F: Fn(ArrayView1<f64>) -> Result<f64, SaddleError> + Send + Sync,
{
    fn observe(&self, position: ArrayView1<f64>) -> Result<f64, SaddleError> {
        self(position)
    }
}

/// Independent exact samples of a quadratic potential restricted to a plane.
pub struct ExactGaussianPlaneSampler<'a> {
    model: QuadraticModel,
    observables: BTreeMap<String, &'a dyn HyperplaneObservable>,
}

impl ExactGaussianPlaneSampler<'_> {
    /// Build an exact sampler for `model`.
    pub fn new(model: QuadraticModel) -> Result<Self, SaddleError> {
        model.check()?;
        Ok(Self {
            model,
            observables: BTreeMap::new(),
        })
    }
}

impl<'a> ExactGaussianPlaneSampler<'a> {
    /// Add a named scalar observable.
    #[must_use]
    pub fn with_observable(
        mut self,
        name: impl Into<String>,
        observable: &'a dyn HyperplaneObservable,
    ) -> Self {
        self.observables.insert(name.into(), observable);
        self
    }
}

impl HyperplaneSampler for ExactGaussianPlaneSampler<'_> {
    fn sample_hyperplane(
        &mut self,
        plane: &Hyperplane,
        budget: &HyperplaneSampleBudget,
    ) -> Result<HyperplaneEnsemble, SaddleError> {
        self.model.check()?;
        if plane.dim() != self.model.center.len() {
            return Err(SaddleError::Shape(
                "the Gaussian model differs in length from the plane".into(),
            ));
        }
        if !(budget.thermal.is_finite() && budget.thermal > 0.0) {
            return Err(SaddleError::Invalid(
                "Gaussian sampling needs a finite positive temperature".into(),
            ));
        }
        let free = plane.dim().saturating_sub(1);
        let (modes, curvatures, actions) = if free == 0 {
            (Array2::zeros((plane.dim(), 0)), Array1::zeros(0), 0)
        } else {
            let rows = Array2::zeros((0, plane.dim()));
            let block = plane.lowest_modes(
                &self.model,
                rows.view(),
                free,
                &MinModeConfig::default().with_rotation_tol(1.0e-12),
            )?;
            (block.modes, block.curvatures, block.actions)
        };
        if let Some(value) = curvatures
            .iter()
            .copied()
            .find(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(SaddleError::Invalid(format!(
                "the Gaussian plane has non-positive free curvature {value:.3e}"
            )));
        }
        let (_, gradient_at_plane) = self.model.eval(plane.point())?;
        let linear = modes.t().dot(&gradient_at_plane);
        let mean: Array1<f64> = linear
            .iter()
            .zip(&curvatures)
            .map(|(&coefficient, &curvature)| -coefficient / curvature)
            .collect();
        let sigma: Array1<f64> = curvatures
            .iter()
            .map(|&curvature| (budget.thermal / curvature).sqrt())
            .collect();
        let mut rng = Rng::new(budget.seed);
        let mut observations = Vec::with_capacity(budget.samples);
        for _ in 0..budget.samples {
            let mut coordinates = mean.clone();
            for (value, width) in coordinates.iter_mut().zip(&sigma) {
                *value += width * rng.normal();
            }
            let position = plane.point().to_owned() + modes.dot(&coordinates);
            let (_, gradient) = self.model.eval(position.view())?;
            let force = -&gradient;
            let projected_force = TangentProjector::project(plane, force.view());
            let mut observables = BTreeMap::new();
            observables.insert(
                "potential.harmonic".into(),
                self.model.eval(position.view())?.0,
            );
            for (name, observable) in &self.observables {
                observables.insert(name.clone(), observable.observe(position.view())?);
            }
            let mut observation = HyperplaneObservation::default()
                .with_constraint_force(plane.normal().dot(&force))
                .with_projected_force(projected_force);
            observation.position = budget.positions.then_some(position);
            observation.observables = observables;
            observations.push(observation);
        }
        Ok(HyperplaneEnsemble::new(
            observations,
            budget.samples + actions,
            1.0,
        ))
    }

    fn energy(&mut self, position: ArrayView1<f64>) -> Result<Option<f64>, SaddleError> {
        Ok(Some(self.model.eval(position)?.0))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn exact_samples_match_the_restricted_gaussian_moments() {
        let model = QuadraticModel::new(
            array![0.0, 0.0, 0.0],
            array![[2.0, 0.3, 0.0], [0.3, 1.0, 0.2], [0.0, 0.2, 0.5]],
        );
        let plane = Hyperplane::new(array![0.6, 0.0, 0.0], array![1.0, 0.0, 0.0]).unwrap();
        let mut sampler = ExactGaussianPlaneSampler::new(model).unwrap();
        let budget = HyperplaneSampleBudget::default()
            .with_thermal(0.4)
            .with_seed(7)
            .with_samples(20_000)
            .with_positions(true);
        let ensemble = sampler.sample_hyperplane(&plane, &budget).unwrap();
        let mean_y = ensemble
            .observations
            .iter()
            .map(|observation| observation.position.as_ref().unwrap()[1])
            .sum::<f64>()
            / budget.samples as f64;
        let expected_y = -0.3 * 0.6 * 0.5 / (0.5 - 0.2 * 0.2);
        assert!((mean_y - expected_y).abs() < 0.01, "{mean_y}");
        assert!(ensemble.observations.iter().all(|observation| {
            plane
                .distance(observation.position.as_ref().unwrap().view())
                .abs()
                < 1.0e-12
        }));
    }

    #[test]
    fn a_non_positive_plane_curvature_is_rejected() {
        let model = QuadraticModel::new(array![0.0, 0.0], array![[1.0, 0.0], [0.0, -1.0]]);
        let plane = Hyperplane::new(array![0.0, 0.0], array![1.0, 0.0]).unwrap();
        let mut sampler = ExactGaussianPlaneSampler::new(model).unwrap();
        let error = sampler
            .sample_hyperplane(&plane, &HyperplaneSampleBudget::default())
            .unwrap_err();
        assert!(error.to_string().contains("non-positive"), "{error}");
    }
}
