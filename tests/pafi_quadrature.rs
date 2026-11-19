//! Zero-noise PAFI profiles whose free energy is known in closed form.
//!
//! A one-dimensional constraint freezes the only coordinate, so each plane
//! mean is the path gradient and the batch-means error is zero. What remains
//! is truncation of the integral. The reported plane and barrier errors have
//! to cover that truncation.

use std::f64::consts::PI;

use ndarray::{Array1, Array2, ArrayView1};
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::engine::{DynamicsEngine, ThermostatConfig};
use rgflux::pafi::{FreeEnergyProfile, PafiConfig, pafi_profile};
use rgflux::stats::StatsConfig;

const PLANES: usize = 9;
const THERMAL_ENERGY: f64 = 0.2;

/// `V(x) = 1 + cos(pi x)`, wells at `x = ±1` and a barrier of height 2 at
/// `x = 0`. The path runs from the left minimum to the barrier.
struct CosineDoubleWell;

impl rgsaddle::PointSurface for CosineDoubleWell {
    fn eval(
        &self,
        point: ArrayView1<'_, f64>,
    ) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
        let x = point[0];
        let energy = 1.0 + (PI * x).cos();
        let gradient = Array1::from_elem(1, -PI * (PI * x).sin());
        Ok((energy, gradient))
    }
}

/// Straight-path image of the harmonic-limit free energy on the curved
/// anisotropic well `κ(θ) = a_x cos²θ + a_y sin²θ`. Planes through the origin
/// have `F(θ) - F(θ₀) = (kT/2) ln(κ(θ)/κ(θ₀))` with `θ₀ = pi/2` and `κ(θ₀) = a_y`.
struct HarmonicLimit {
    ax: f64,
    ay: f64,
    thermal_energy: f64,
}

impl rgsaddle::PointSurface for HarmonicLimit {
    fn eval(
        &self,
        point: ArrayView1<'_, f64>,
    ) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
        let theta = point[0];
        let kappa = self.ax * theta.cos().powi(2) + self.ay * theta.sin().powi(2);
        let kappa_prime = (self.ay - self.ax) * (2.0 * theta).sin();
        let energy = 0.5 * self.thermal_energy * (kappa / self.ay).ln();
        let gradient = Array1::from_elem(1, 0.5 * self.thermal_energy * kappa_prime / kappa);
        Ok((energy, gradient))
    }
}

fn straight_path(start: f64, end: f64) -> Array2<f64> {
    let mut path = Array2::zeros((PLANES, 1));
    for plane in 0..PLANES {
        let fraction = plane as f64 / (PLANES - 1) as f64;
        path[[plane, 0]] = start + (end - start) * fraction;
    }
    path
}

fn profile_on(
    surface: impl rgsaddle::PointSurface + Send + 'static,
    path: Array2<f64>,
) -> FreeEnergyProfile {
    let mut engine = NativeEngine::new(
        surface,
        NativeConfig::default()
            .with_masses(Array1::from_elem(1, 1.0))
            .with_thermostat(
                ThermostatConfig::default()
                    .with_temperature(THERMAL_ENERGY)
                    .with_boltzmann(1.0),
            )
            .with_seed(11),
    )
    .expect("engine");
    engine.shoot_velocities(1).expect("velocities");
    let config = PafiConfig::default()
        .with_planes(PLANES)
        .with_burn_in_steps(1)
        .with_sample_steps(2)
        .with_time_step(0.01)
        .with_zero_temperature_tol(f64::MAX)
        .with_stats(StatsConfig::default().with_min_batches(2));
    pafi_profile(&mut engine, path.view(), &config).expect("profile")
}

fn assert_covers_analytic(profile: &FreeEnergyProfile, analytic: &[f64], barrier: f64) {
    assert_eq!(profile.free_energy.len(), analytic.len());
    for (plane, (&value, &truth)) in profile.free_energy.iter().zip(analytic).enumerate() {
        let discrepancy = (value - truth).abs();
        assert!(
            discrepancy <= profile.standard_error[plane],
            "plane {plane}: discrepancy {discrepancy} exceeds standard error {}",
            profile.standard_error[plane]
        );
    }
    let discrepancy = (profile.barrier - barrier).abs();
    assert!(
        discrepancy <= profile.barrier_error,
        "barrier discrepancy {discrepancy} exceeds barrier error {}",
        profile.barrier_error
    );
}

#[test]
fn harmonic_limit_profile_stays_inside_its_reported_error() {
    let start = PI / 2.0;
    let end = 0.0;
    let analytic: Vec<f64> = (0..PLANES)
        .map(|plane| {
            let fraction = plane as f64 / (PLANES - 1) as f64;
            let theta = start + (end - start) * fraction;
            let kappa = 3.0 * theta.cos().powi(2) + theta.sin().powi(2);
            0.5 * THERMAL_ENERGY * kappa.ln()
        })
        .collect();
    let barrier = *analytic.last().expect("endpoint");
    let profile = profile_on(
        HarmonicLimit {
            ax: 3.0,
            ay: 1.0,
            thermal_energy: THERMAL_ENERGY,
        },
        straight_path(start, end),
    );
    assert_covers_analytic(&profile, &analytic, barrier);
}

#[test]
fn cosine_double_well_profile_stays_inside_its_reported_error() {
    let start = -1.0;
    let end = 0.0;
    let analytic: Vec<f64> = (0..PLANES)
        .map(|plane| {
            let fraction = plane as f64 / (PLANES - 1) as f64;
            let x = start + (end - start) * fraction;
            (PI * x).cos() - (PI * start).cos()
        })
        .collect();
    let barrier = *analytic.last().expect("endpoint");
    let profile = profile_on(CosineDoubleWell, straight_path(start, end));
    assert_covers_analytic(&profile, &analytic, barrier);
}
