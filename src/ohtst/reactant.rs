//! The density of the reactant ensemble on a plane, `ρ = Q_Z / Q_R`.
//!
//! The rate of Jóhannesson and Jónsson (Eqs. 2, 3 and 21) needs the ratio
//! of the configurational integral over a reference plane `Z_R` to that of
//! the reactant region. In mass-weighted coordinates it is the probability
//! density of `ξ = ν · q` at the plane, per unit length along its normal.
//!
//! The reactant region is the part of space behind every wall: a proposal
//! past a wall is rejected and the velocity reversed, so the walls
//! reflect. A system with several exit channels needs a wall in each; the
//! optimal plane, or one a little past it, is the natural wall.
//!
//! The density comes from the mean force on `ξ`. Its marginal is
//! `ρ(ξ) ∝ exp(-F(ξ)/kT)` with `F'(ξ) = ⟨ν · ∇V⟩_ξ`, so integrating from the
//! far side,
//!
//! ```text
//! ρ(ξ*) = -⟨θ(ξ* - ξ) ν · ∇V⟩_R / kT,
//! ```
//!
//! exact when the density vanishes on the far side and no wall cuts the
//! region `ξ < ξ*`. The symmetric form, half the mean of
//! `sign(ξ - ξ*) ν · ∇V / kT`, has a smaller variance but is high by half
//! the density at the walls, which is not small at high temperature.
//!
//! The crossing estimate of the paper's Eq. 22 is returned beside it: the
//! steps that cross the plane, each weighted by `1 / |Δξ|`, over the number
//! of steps. Its weights have a slowly decaying tail, so its block error
//! is only a guide.

use ndarray::{Array1, ArrayView1};

use crate::ohtst::sample::{Dynamics, LangevinConfig, held_directions};
use rgsaddle::error::SaddleError;
use rgsaddle::minmode::PointSurface;
use rgsaddle::projector::Hyperplane;
use rgsaddle::stats::{Rng, block_mean};

/// The reactant density on a plane and its errors.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct ReactantDensity {
    /// The reference plane.
    pub plane: Hyperplane,
    /// The sampling temperature `kT`.
    pub thermal: f64,
    /// `Q_Z / Q_R` per unit mass-weighted length along the normal, from
    /// the mean force.
    pub density: f64,
    /// Block standard error of [`Self::density`].
    pub density_error: f64,
    /// The crossing estimate of the same density, Jóhannesson and Jónsson
    /// (2001), Eq. 22. Each crossing carries the weight `1/|ξ̇|`, whose
    /// second moment under the flux-weighted velocities diverges
    /// logarithmically, so its variance is infinite. Use
    /// [`basin_density`](crate::ohtst::basin_density) or
    /// [`Self::density`] instead.
    pub crossing_density: f64,
    /// Block standard error of [`Self::crossing_density`]; a guide only,
    /// since the variance it estimates is infinite.
    pub crossing_error: f64,
    /// Fraction of records behind the plane.
    pub behind: f64,
    /// Integrated autocorrelation time of the force series, in records.
    pub tau: f64,
    /// Records.
    pub records: usize,
    /// Surface evaluations.
    pub evaluations: usize,
    /// Accepted fraction of the proposals.
    pub acceptance: f64,
    /// Proposals rejected at a wall.
    pub wall_hits: usize,
    /// Configuration after the last step.
    pub last: Array1<f64>,
}

/// Sample the reactant region behind `walls` from `start` and return the
/// density of the ensemble on `plane`.
pub fn reactant_density<S: PointSurface>(
    surface: &S,
    plane: &Hyperplane,
    start: ArrayView1<f64>,
    walls: &[Hyperplane],
    config: &LangevinConfig,
) -> Result<ReactantDensity, SaddleError> {
    config.check()?;
    let n = plane.dim();
    if start.len() != n || walls.iter().any(|wall| wall.dim() != n) {
        return Err(SaddleError::Shape(
            "reactant start, plane and walls differ in length".into(),
        ));
    }
    let normal = plane.normal().to_owned();
    let target = normal.dot(&plane.point());
    let held = held_directions(surface, start, None, config.hold_excluded)?;
    let overlap = held
        .iter()
        .map(|t| t.dot(&normal).powi(2))
        .sum::<f64>()
        .sqrt();
    if overlap > 1.0e-6 {
        return Err(SaddleError::Invalid(
            "the plane normal has a component along a held mode; take Hyperplane::orthogonal_to first"
                .into(),
        ));
    }
    let rng = Rng::new(config.seed);
    let mut dynamics = Dynamics::new(surface, start.to_owned(), held, walls, None, config, rng)?;
    for _ in 0..config.burn_in {
        dynamics.step()?;
    }
    let mut force = Vec::with_capacity(config.records);
    let mut crossing = Vec::with_capacity(config.records);
    let mut behind = 0usize;
    let mut xi = normal.dot(&dynamics.q);
    for _ in 0..config.records {
        let mut weight = 0.0;
        for _ in 0..config.stride {
            dynamics.step()?;
            let next = normal.dot(&dynamics.q);
            if (xi < target) != (next < target) {
                weight += 1.0 / (next - xi).abs();
            }
            xi = next;
        }
        crossing.push(weight / config.stride as f64);
        if xi < target {
            behind += 1;
            force.push(normal.dot(&dynamics.gradient));
        } else {
            force.push(0.0);
        }
    }
    let beta = 1.0 / config.thermal;
    let (mean, error, tau, _) = block_mean(&force, config.blocks);
    let (crossing_density, crossing_error, _, _) = block_mean(&crossing, config.blocks);
    Ok(ReactantDensity {
        plane: plane.clone(),
        thermal: config.thermal,
        density: -beta * mean,
        density_error: beta * error,
        crossing_density,
        crossing_error,
        behind: behind as f64 / config.records as f64,
        tau,
        records: config.records,
        evaluations: dynamics.evaluations,
        acceptance: dynamics.acceptance(),
        wall_hits: dynamics.wall_hits,
        last: dynamics.q.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// `V = ½ (2 x² + y²)`: `x` is normal with variance `kT / 2`.
    struct Well;

    impl PointSurface for Well {
        fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            Ok((
                0.5 * (2.0 * q[0] * q[0] + q[1] * q[1]),
                array![2.0 * q[0], q[1]],
            ))
        }
    }

    /// The standard normal distribution function, by Simpson's rule.
    fn normal_cdf(z: f64) -> f64 {
        let steps = 20_000;
        let lo = -12.0;
        let h = (z - lo) / steps as f64;
        let f = |t: f64| (-0.5 * t * t).exp() / std::f64::consts::TAU.sqrt();
        let mut sum = f(lo) + f(z);
        for i in 1..steps {
            sum += f(lo + i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 };
        }
        sum * h / 3.0
    }

    #[test]
    fn density_of_a_truncated_gaussian() {
        let kt = 0.5;
        let sigma = (kt / 2.0f64).sqrt();
        let plane = Hyperplane::new(array![0.3, 0.0], array![1.0, 0.0]).unwrap();
        let wall = Hyperplane::new(array![0.6, 0.0], array![1.0, 0.0]).unwrap();
        let config = LangevinConfig::default()
            .with_thermal(kt)
            .with_time_step(0.3)
            .with_burn_in(500)
            .with_stride(3)
            .with_records(40_000)
            .with_seed(5);
        let gauss =
            |x: f64| (-0.5 * (x / sigma).powi(2)).exp() / (sigma * std::f64::consts::TAU.sqrt());
        let free = reactant_density(&Well, &plane, array![0.0, 0.0].view(), &[], &config).unwrap();
        let walled = reactant_density(
            &Well,
            &plane,
            array![0.0, 0.0].view(),
            std::slice::from_ref(&wall),
            &config,
        )
        .unwrap();
        let truncated = gauss(0.3) / normal_cdf(0.6 / sigma);
        for (result, exact) in [(&free, gauss(0.3)), (&walled, truncated)] {
            assert!(
                (result.density - exact).abs() < 4.0 * result.density_error,
                "density {} ± {}, exact {exact}",
                result.density,
                result.density_error
            );
            assert!(result.density_error < 0.03 * exact);
            assert!((result.crossing_density - exact).abs() < 4.0 * result.crossing_error);
        }
        assert_eq!(free.wall_hits, 0);
        assert!(walled.wall_hits > 0);
        assert!(
            reactant_density(&Well, &plane, array![0.7, 0.0].view(), &[wall], &config).is_err()
        );
    }
}
