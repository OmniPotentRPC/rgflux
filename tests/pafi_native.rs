//! Analytic PAFI checks through the native constrained dynamics engine.

use std::f64::consts::PI;

use ndarray::{Array1, Array2, ArrayView1};
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::engine::{DynamicsEngine, ThermostatConfig};
use rgflux::pafi::{FreeEnergyProfile, PafiConfig, pafi_profile};
use rgflux::stats::StatsConfig;

struct WideningDoubleWell;
struct CircularValley {
    ax: f64,
    ay: f64,
    radius: f64,
    along_path: f64,
}

impl rgsaddle::PointSurface for WideningDoubleWell {
    fn eval(&self, r: ArrayView1<'_, f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
        let x = r[0];
        let y = r[1];
        let well = (1.0 - x * x).powi(2);
        let k = 1.5 + 0.5 * x;
        let energy = well + 0.5 * k * y * y;
        let gx = -4.0 * x * (1.0 - x * x) + 0.25 * y * y;
        let gy = k * y;
        Ok((energy, Array1::from_vec(vec![gx, gy])))
    }
}

impl rgsaddle::PointSurface for CircularValley {
    fn eval(&self, r: ArrayView1<'_, f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
        // Masses are `(1, 4)`, so `q = (x, 2 y)` is mass-weighted. The floor
        // of the valley is the circle `|q| = radius`. Transverse stiffness is
        // `ax` at angle 0 and `ay` at `pi / 2`. `along_path` is a shallow well
        // in the arc so the first image is a positive definite minimum.
        let qx = r[0];
        let qy = 2.0 * r[1];
        let rho_sq = qx * qx + qy * qy;
        let rho = rho_sq.sqrt();
        if !rho.is_finite() || rho == 0.0 {
            return Err(rgsaddle::SaddleError::NonFinite("circular valley radius"));
        }
        let theta = qy.atan2(qx);
        let radial = rho - self.radius;
        let lambda = (self.ax * qx * qx + self.ay * qy * qy) / rho_sq;
        let angle = std::f64::consts::FRAC_PI_2 - theta;
        let energy = 0.5 * lambda * radial * radial
            + 0.5 * self.along_path * self.radius * self.radius * angle * angle;
        let d_rho_dqx = qx / rho;
        let d_rho_dqy = qy / rho;
        let d_theta_dqx = -qy / rho_sq;
        let d_theta_dqy = qx / rho_sq;
        let d_lambda_d_theta = 2.0 * (self.ay - self.ax) * (qy / rho) * (qx / rho);
        let radial_sq = radial * radial;
        let d_trans_dqx =
            0.5 * d_lambda_d_theta * d_theta_dqx * radial_sq + lambda * radial * d_rho_dqx;
        let d_trans_dqy =
            0.5 * d_lambda_d_theta * d_theta_dqy * radial_sq + lambda * radial * d_rho_dqy;
        let well = self.along_path * self.radius * self.radius * angle;
        let gradient = Array1::from_vec(vec![
            d_trans_dqx - well * d_theta_dqx,
            2.0 * (d_trans_dqy - well * d_theta_dqy),
        ]);
        Ok((energy, gradient))
    }
}

fn line_path(start: f64, end: f64, planes: usize) -> Array2<f64> {
    let mut path = Array2::<f64>::zeros((planes, 2));
    for i in 0..planes {
        let t = (i as f64) / ((planes - 1) as f64);
        path[[i, 0]] = start + (end - start) * t;
        path[[i, 1]] = 0.0;
    }
    path
}

fn tilted_path(start: f64, end: f64, planes: usize, slope: f64) -> Array2<f64> {
    let mut path = Array2::<f64>::zeros((planes, 2));
    for i in 0..planes {
        let t = (i as f64) / ((planes - 1) as f64);
        let x = start + (end - start) * t;
        path[[i, 0]] = x;
        path[[i, 1]] = slope * (x - start);
    }
    path
}

fn integrate_reactant_partition(arc_lengths: &[f64], free_energy: &[f64], kbt: f64) -> f64 {
    let dividing = free_energy.len() - 1;
    let mut zr_half = 0.0_f64;
    for i in 1..=dividing {
        let ds = arc_lengths[i] - arc_lengths[i - 1];
        let w0 = (-free_energy[i - 1] / kbt).exp();
        let w1 = (-free_energy[i] / kbt).exp();
        zr_half += 0.5 * ds * (w0 + w1);
    }
    2.0 * zr_half
}

fn mass_weighted_circular_path(
    radius: f64,
    theta_start: f64,
    theta_end: f64,
    images: usize,
) -> Array2<f64> {
    let mut path = Array2::zeros((images, 2));
    for i in 0..images {
        let fraction = (i as f64) / ((images - 1) as f64);
        let theta = theta_start + fraction * (theta_end - theta_start);
        path[[i, 0]] = radius * theta.cos();
        path[[i, 1]] = 0.5 * radius * theta.sin();
    }
    path
}

fn run_barrier_case(
    masses: Vec<f64>,
    seed: u64,
    planes: usize,
    burn_in_steps: usize,
    sample_steps: usize,
    time_step: f64,
) -> (f64, f64, f64) {
    let thermal = 0.2_f64;
    let analytic_barrier = 1.0 + 0.5 * thermal * 1.5_f64.ln();
    let path = line_path(-1.0, 0.0, planes);
    let config = PafiConfig::default()
        .with_planes(planes)
        .with_burn_in_steps(burn_in_steps)
        .with_sample_steps(sample_steps)
        .with_time_step(time_step)
        .with_stats(StatsConfig::default().with_min_batches(2))
        .with_seed(seed);

    let mut engine = NativeEngine::new(
        WideningDoubleWell,
        NativeConfig::default()
            .with_masses(Array1::from_vec(masses))
            .with_seed(seed)
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(1.5)
                    .with_temperature(thermal)
                    .with_boltzmann(1.0),
            ),
    )
    .expect("native engine");
    engine.shoot_velocities(seed).expect("initial velocities");

    let profile_result = pafi_profile(&mut engine, path.view(), &config);
    assert!(
        profile_result.is_ok(),
        "pafi_profile returned {profile_result:?}",
    );
    let profile = profile_result.expect("checked above");
    let z = (profile.barrier - analytic_barrier).abs() / profile.barrier_error;
    (profile.barrier, profile.barrier_error, z)
}

#[test]
fn pafi_widening_barrier_matches_analytic_within_reported_se() {
    let (barrier, se, z) = run_barrier_case(vec![1.0, 1.0], 11, 31, 10_000, 10_000, 0.001);
    eprintln!("isotropic widening barrier={barrier:.8} se={se:.8} z={z:.4}");
    assert!(z <= 2.0, "barrier {barrier:.8} ± {se:.8} gives z={z:.4}");
}

#[test]
fn pafi_widening_barrier_mass_anisotropic_matches_analytic_within_reported_se() {
    let (barrier, se, z) = run_barrier_case(vec![1.0, 4.0], 21, 31, 20_000, 10_000, 0.001);
    eprintln!("anisotropic widening barrier={barrier:.8} se={se:.8} z={z:.4}");
    assert!(z <= 2.0, "barrier {barrier:.8} ± {se:.8} gives z={z:.4}");
}

const CURVED_THERMAL: f64 = 0.2;
/// Power of two, so the circle `|q| = radius` is exact in floating point and
/// the reactant gradient at the first image is a true zero.
const CURVED_RADIUS: f64 = 1.0;
const CURVED_STIFFNESS_END: f64 = 120.0;
const CURVED_STIFFNESS_START: f64 = 40.0;
/// Curvature along the arc. Small next to `kT`, large next to the zero-mode
/// floor `sqrt(epsilon) * stiffness`, so the floor is a minimum and the
/// potential rise does not hide the transverse entropy.
const CURVED_ALONG_PATH: f64 = 1.0e-4;
const CURVED_PLANES: usize = 9;
const CURVED_SEED: u64 = 1_234;

/// Largest disagreement, as a fraction of the analytic barrier, between an
/// end plane of the curved path and the same plane seen from a path that
/// continues past it. Common random numbers make the two runs share their
/// sampling noise, so the residual is not statistical: it is 0.03% of the
/// barrier with not-a-knot endpoints and 7.8% with natural ones.
const END_PLANE_AGREEMENT: f64 = 1.0 / 64.0;

/// Profile along the quarter circle `theta` in `[pi / 2, 0]`, sampled at
/// `CURVED_PLANES + extra_images` equally spaced images and planes. With
/// `extra_images = 1` the arc continues one image spacing past `theta = 0`,
/// so the plane at `theta = 0` is an interior plane of the longer path while
/// every plane before it keeps its position, tangent and per-plane seed.
fn curved_profile(extra_images: usize) -> FreeEnergyProfile {
    let theta_start = PI * 0.5;
    let theta_end = 0.0_f64;
    let images = CURVED_PLANES + extra_images;
    let spacing = (theta_end - theta_start) / ((CURVED_PLANES - 1) as f64);
    let path = mass_weighted_circular_path(
        CURVED_RADIUS,
        theta_start,
        theta_end + spacing * (extra_images as f64),
        images,
    );
    let config = PafiConfig::default()
        .with_planes(images)
        .with_burn_in_steps(3_000)
        .with_sample_steps(54_000)
        .with_stats(StatsConfig::default().with_min_batches(4))
        .with_seed(CURVED_SEED);
    let mut engine = NativeEngine::new(
        CircularValley {
            ax: CURVED_STIFFNESS_END,
            ay: CURVED_STIFFNESS_START,
            radius: CURVED_RADIUS,
            along_path: CURVED_ALONG_PATH,
        },
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0, 4.0]))
            .with_seed(CURVED_SEED)
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(1.5)
                    .with_temperature(CURVED_THERMAL)
                    .with_boltzmann(1.0),
            ),
    )
    .expect("native engine");
    engine
        .shoot_velocities(CURVED_SEED)
        .expect("initial velocities");
    pafi_profile(&mut engine, path.view(), &config).expect("curved profile")
}

#[test]
fn circular_valley_starts_at_a_minimum() {
    let start = mass_weighted_circular_path(CURVED_RADIUS, PI * 0.5, 0.0, 2);
    let mut engine = NativeEngine::new(
        CircularValley {
            ax: CURVED_STIFFNESS_END,
            ay: CURVED_STIFFNESS_START,
            radius: CURVED_RADIUS,
            along_path: CURVED_ALONG_PATH,
        },
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0, 4.0]))
            .with_thermostat(
                ThermostatConfig::default()
                    .with_temperature(CURVED_THERMAL)
                    .with_boltzmann(1.0),
            ),
    )
    .expect("native engine");
    engine
        .set_positions(start.row(0))
        .expect("set the first image");
    let (_, forces) = engine.energy_and_forces().expect("forces");
    let peak = forces
        .iter()
        .fold(0.0_f64, |acc, force| acc.max(force.abs()));
    assert!(
        peak <= rgflux::stats::SERIES_FLAT_TOL,
        "valley reactant force peak {peak}"
    );
    let config = PafiConfig::default()
        .with_planes(2)
        .with_burn_in_steps(1)
        .with_sample_steps(2);
    pafi_profile(&mut engine, start.view(), &config).expect("valley reactant is a minimum");
}

#[test]
fn curved_mass_weighted_harmonic_profile_requires_jacobian() {
    let (profile, longer) = std::thread::scope(|scope| {
        let base = scope.spawn(|| curved_profile(0));
        let longer = scope.spawn(|| curved_profile(1));
        (
            base.join().expect("curved profile thread"),
            longer.join().expect("longer curved profile thread"),
        )
    });
    // On the valley floor the radial spring is harmonic in the flat measure
    // of the plane, so its free energy changes by `1/2 kT ln(k_end / k_start)`.
    // The shallow well adds `1/2 * along_path * (radius * pi / 2)^2`.
    let arc = CURVED_RADIUS * PI * 0.5;
    let analytic_barrier = 0.5 * CURVED_ALONG_PATH * arc * arc
        + 0.5 * CURVED_THERMAL * (CURVED_STIFFNESS_END / CURVED_STIFFNESS_START).ln();
    let z = (profile.barrier - analytic_barrier).abs() / profile.barrier_error;
    eprintln!(
        "curved harmonic barrier={:.8} se={:.8} analytic={:.8} z={:.4}",
        profile.barrier, profile.barrier_error, analytic_barrier, z
    );
    assert_eq!(profile.dividing_index, CURVED_PLANES - 1);
    assert!(
        z <= 4.0,
        "curved barrier {:.8} ± {:.8} versus analytic {:.8} gives z={:.4}",
        profile.barrier,
        profile.barrier_error,
        analytic_barrier,
        z
    );

    // Natural spline endpoints zero the normal curvature `dn/ds` on the end
    // planes. That shifts the last free-energy increment by far less than the
    // sampling error, so compare against the longer path, which sees the same
    // plane with the curvature of an interior plane.
    let tolerance = END_PLANE_AGREEMENT * analytic_barrier;
    let worst = (1..CURVED_PLANES)
        .map(|k| {
            let increment = profile.free_energy[k] - profile.free_energy[k - 1];
            let reference = longer.free_energy[k] - longer.free_energy[k - 1];
            (increment - reference).abs()
        })
        .fold(0.0_f64, f64::max);
    eprintln!("end-plane disagreement={worst:.8} tolerance={tolerance:.8}");
    assert!(
        worst <= tolerance,
        "free-energy increments differ by up to {worst:.8} from those of a longer path, \
         above the tolerance {tolerance:.8}"
    );
}

#[test]
fn normalized_pafi_rate_matches_analytic_profile() {
    let thermal = 0.2_f64;
    let planes = 41usize;
    let path = line_path(-1.0, 0.0, planes);
    let config = PafiConfig::default()
        .with_planes(planes)
        .with_burn_in_steps(10_000)
        .with_sample_steps(10_000)
        .with_time_step(0.001)
        .with_stats(StatsConfig::default().with_min_batches(2))
        .with_seed(7);

    let mut engine = NativeEngine::new(
        WideningDoubleWell,
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0, 4.0]))
            .with_seed(7)
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(1.3)
                    .with_temperature(thermal)
                    .with_boltzmann(1.0),
            ),
    )
    .expect("native engine");
    engine.shoot_velocities(7).expect("initial velocities");

    let profile_result = pafi_profile(&mut engine, path.view(), &config);
    assert!(
        profile_result.is_ok(),
        "pafi_profile returned {profile_result:?}",
    );
    let profile = profile_result.expect("checked above");
    let rate = profile.normalized_rate;
    let rate_error = profile.normalized_rate_error;
    let dividing = profile.dividing_index;
    assert!(rate_error.is_finite() && rate_error >= 0.0);
    assert!(dividing > 0);
    let reconstructed_zr =
        integrate_reactant_partition(&profile.arc_lengths, &profile.free_energy, thermal);
    assert!(
        (profile.reactant_partition - reconstructed_zr).abs() <= 1.0e-12,
        "reactant partition mismatch: {:.16e} vs {:.16e}",
        profile.reactant_partition,
        reconstructed_zr
    );
    assert!(
        (profile.reactant_log_partition - reconstructed_zr.ln()).abs() <= 1.0e-12,
        "reactant log partition mismatch: {:.16e} vs {:.16e}",
        profile.reactant_log_partition,
        reconstructed_zr.ln()
    );

    let intervals = 20_000usize;
    let dx = 1.0 / (intervals as f64);
    let analytic_free_energy =
        |x: f64| (1.0 - x * x).powi(2) + 0.5 * thermal * (1.5 + 0.5 * x).ln();
    let reference = analytic_free_energy(-1.0);
    let mut partition_sum = 0.0_f64;
    for i in 0..=intervals {
        let x = -1.0 + (i as f64) * dx;
        let weight = if i == 0 || i == intervals {
            1.0
        } else if i % 2 == 0 {
            2.0
        } else {
            4.0
        };
        partition_sum += weight * (-(analytic_free_energy(x) - reference) / thermal).exp();
    }
    let analytic_partition = 2.0 * dx * partition_sum / 3.0;
    let analytic_barrier = analytic_free_energy(0.0) - reference;
    let analytic_rate =
        (thermal / (2.0 * PI)).sqrt() * (-analytic_barrier / thermal).exp() / analytic_partition;
    let relative_error = (rate - analytic_rate).abs() / analytic_rate;
    eprintln!(
        "normalized rate={rate:.8e} se={rate_error:.8e} analytic={analytic_rate:.8e} relative error={relative_error:.4}",
    );
    assert!(
        relative_error <= 0.05,
        "rate {rate:.8e} ± {rate_error:.8e} versus analytic {analytic_rate:.8e} has relative error {relative_error:.4}",
    );
}

#[test]
fn widening_well_reports_the_reactant_curvature() {
    let thermal = 0.2_f64;
    let planes = 41usize;
    let path = line_path(-1.0, 0.0, planes);
    let config = PafiConfig::default()
        .with_planes(planes)
        .with_burn_in_steps(10_000)
        .with_sample_steps(10_000)
        .with_time_step(0.001)
        .with_stats(StatsConfig::default().with_min_batches(2))
        .with_seed(7);

    let mut engine = NativeEngine::new(
        WideningDoubleWell,
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0, 4.0]))
            .with_seed(7)
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(1.3)
                    .with_temperature(thermal)
                    .with_boltzmann(1.0),
            ),
    )
    .expect("native engine");
    engine.shoot_velocities(7).expect("initial velocities");

    let profile = pafi_profile(&mut engine, path.view(), &config).expect("profile");
    // V_xx = 8 at x = -1 along the unit normal, and the mass on x is 1.
    let omega_0 = 8.0_f64.sqrt();
    assert!((profile.reactant_frequency - omega_0).abs() <= 1.0e-6 * omega_0);
    let harmonic_rate = omega_0 / (2.0 * PI) * (-profile.barrier / thermal).exp();
    let gap = (profile.normalized_rate - harmonic_rate).abs();
    assert!(
        gap > 2.0 * profile.normalized_rate_error,
        "harmonic gap {gap:.4e} does not exceed two standard errors {:.4e}",
        2.0 * profile.normalized_rate_error
    );
}

#[test]
fn sampling_refinement_restores_analytic_coverage() {
    let (coarse_barrier, coarse_se, coarse_z) =
        run_barrier_case(vec![1.0, 4.0], 91, 31, 20_000, 2_500, 0.001);
    let (fine_barrier, fine_se, fine_z) =
        run_barrier_case(vec![1.0, 4.0], 91, 31, 20_000, 10_000, 0.001);
    eprintln!(
        "refinement coarse barrier={coarse_barrier:.8} se={coarse_se:.8} z={coarse_z:.4}; fine barrier={fine_barrier:.8} se={fine_se:.8} z={fine_z:.4}",
    );
    assert!(
        coarse_z > 4.0 && fine_z <= 2.0,
        "refinement should move the analytic value into two-SE coverage: coarse z={coarse_z:.4}, fine barrier {fine_barrier:.8} ± {fine_se:.8} gives z={fine_z:.4}",
    );
}

#[test]
fn arc_lengths_reflect_mass_weighting() {
    let thermal = 0.03_f64;
    let planes = 9usize;
    let slope = 0.3_f64;
    let start = -1.0_f64;
    let end = 0.0_f64;
    let path = tilted_path(start, end, planes, slope);
    let spacing = 1.0 / ((planes - 1) as f64);
    // The path is `y = slope * (x - start)`, so it leaves the minimum at
    // `(-1, 0)`. Along it the potential is
    // `V = (1 - x^2)^2 + 0.5 * (1.5 + 0.5 * x) * (slope * (x - start))^2`.
    // The quartic part has third derivative `4 * 3 * 2 * x`, and the cubic part
    // `0.5 * 0.5 * slope^2 * x^3` has the constant `3 * 2 * 0.5 * 0.5 * slope^2`.
    // The path force `V'` therefore has curvature at most `third_derivative_bound`,
    // and the trapezoid error over `[start, end]` is at most
    // `spacing^2 * (end - start) * third_derivative_bound / 12`.
    // Masses only rescale arc length uniformly on a straight path, so the same
    // bound holds for both mass sets.
    let third_derivative_bound =
        4.0 * 3.0 * 2.0 * start.abs().max(end.abs()) + 3.0 * 2.0 * 0.5 * 0.5 * slope * slope;
    let trapezoid_error_bound = spacing * spacing * (end - start) * third_derivative_bound / 12.0;
    let config = PafiConfig::default()
        .with_planes(planes)
        .with_burn_in_steps(200)
        .with_sample_steps(64)
        .with_time_step(0.001)
        .with_zero_temperature_tol(trapezoid_error_bound)
        .with_stats(StatsConfig::default().with_min_batches(2))
        .with_seed(99);

    let make_engine = |masses: Vec<f64>, seed: u64| {
        let mut engine = NativeEngine::new(
            WideningDoubleWell,
            NativeConfig::default()
                .with_masses(Array1::from_vec(masses))
                .with_seed(seed)
                .with_thermostat(
                    ThermostatConfig::default()
                        .with_gamma(1.3)
                        .with_temperature(thermal)
                        .with_boltzmann(1.0),
                ),
        )
        .expect("native engine");
        engine.shoot_velocities(seed).expect("initial velocities");
        engine
    };

    let mut unit_mass = make_engine(vec![1.0, 1.0], 99);
    let mut skew_mass = make_engine(vec![1.0, 4.0], 100);
    let p1 = pafi_profile(&mut unit_mass, path.view(), &config).expect("profile unit");
    let p2 = pafi_profile(&mut skew_mass, path.view(), &config).expect("profile skew");
    let s1 = *p1.arc_lengths.last().expect("last arc");
    let s2 = *p2.arc_lengths.last().expect("last arc");
    assert!(
        (s1 - s2).abs() > 1.0e-3,
        "mass-weighted arc lengths should differ: s1={s1}, s2={s2}",
    );
}
