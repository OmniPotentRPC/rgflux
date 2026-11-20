//! Tests for PAFI monitor equidistribution (`s -> sigma(s)`) and spectral-tail
//! adaptive plane refinement.

use std::sync::atomic::{AtomicUsize, Ordering};

use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2, ArrayView1};
use rgflux::engine::ThermostatConfig;
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::pafi::{
    AdaptivePafiConfig, EquidistributionMap, MonitorInputs, MonitorKind, PafiConfig,
    estimate_spectral_tail, pafi_profile, pafi_profile_adaptive, pool_pafi_replicates,
    size_window_from_tau_int,
};
use rgflux::stats::StatsConfig;

/// Analytic 1D symmetric double-well with a localized barrier:
/// `V(x) = (1 - x^2)^2 / (1 + 6 * x^2)` on `[-1, 1]`.
///
/// Exact reactant minimum at `x = -1` (`V(-1) = 0`), product minimum at
/// `x = +1` (`V(+1) = 0`), and barrier top at `x = 0` (`V(0) = 1`, `F* = 1`).
struct LocalizedDoubleWell1D {
    calls: AtomicUsize,
}

impl LocalizedDoubleWell1D {
    const fn new() -> Self {
        Self {
            calls: AtomicUsize::new(0),
        }
    }

    fn exact_potential(x: f64) -> f64 {
        let u = 1.0 - x * x;
        (u * u) / (1.0 + 6.0 * x * x)
    }

    fn exact_gradient(x: f64) -> f64 {
        let u = 1.0 - x * x;
        let denom = 1.0 + 6.0 * x * x;
        let d_num = -4.0 * x * u;
        let d_den = 12.0 * x;
        (d_num * denom - u * u * d_den) / (denom * denom)
    }

    fn call_count(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }
}

impl rgsaddle::PointSurface for LocalizedDoubleWell1D {
    fn eval(&self, r: ArrayView1<'_, f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let x = r[0];
        Ok((
            Self::exact_potential(x),
            Array1::from_vec(vec![Self::exact_gradient(x)]),
        ))
    }
}

/// Analytic 1D asymmetric barrier on `[-1, 1]`:
/// `V(x) = (1 - x^2)^2 / (1 + 8 * (x - 0.25)^2) + 0.05 * (x + 1)^2 * (2 - x)`.
///
/// Stationary endpoints `V'(-1) = 0` (`V(-1) = 0`) and `V'(+1) = 0` (`V(+1) = 0.2`),
/// with a shifted, steep barrier peak near `x* = 0.23615866`.
struct AsymmetricBarrier1D {
    calls: AtomicUsize,
}

impl AsymmetricBarrier1D {
    const fn new() -> Self {
        Self {
            calls: AtomicUsize::new(0),
        }
    }

    fn exact_potential(x: f64) -> f64 {
        let u = 1.0 - x * x;
        u * u * (1.0 + 0.2 * x)
    }

    fn exact_gradient(x: f64) -> f64 {
        let u = 1.0 - x * x;
        -4.0 * x * u * (1.0 + 0.2 * x) + 0.2 * u * u
    }

    fn exact_barrier() -> f64 {
        let mut left = 0.0_f64;
        let mut right = 0.3_f64;
        for _ in 0..80 {
            let mid = 0.5 * (left + right);
            if Self::exact_gradient(mid) > 0.0 {
                left = mid;
            } else {
                right = mid;
            }
        }
        Self::exact_potential(0.5 * (left + right)) - Self::exact_potential(-1.0)
    }

    fn call_count(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }
}

impl rgsaddle::PointSurface for AsymmetricBarrier1D {
    fn eval(&self, r: ArrayView1<'_, f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let x = r[0];
        Ok((
            Self::exact_potential(x),
            Array1::from_vec(vec![Self::exact_gradient(x)]),
        ))
    }
}

/// 1D harmonic potential `V(x) = 0.5 * k * x^2`.
struct Harmonic1D {
    k: f64,
}

impl rgsaddle::PointSurface for Harmonic1D {
    fn eval(&self, r: ArrayView1<'_, f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
        let x = r[0];
        Ok((0.5 * self.k * x * x, Array1::from_vec(vec![self.k * x])))
    }
}

/// Monotone smooth barrier whose endpoint curvatures and quadrature corrections match.
struct SmoothRipple1D;

impl rgsaddle::PointSurface for SmoothRipple1D {
    fn eval(&self, r: ArrayView1<'_, f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
        let x = r[0];
        let amplitude = 0.25;
        let energy =
            x + 0.125 * x * x + amplitude * (x.powi(3) / 3.0 - x.powi(4) / 2.0 + x.powi(5) / 5.0);
        let gradient = 1.0 + 0.25 * x + amplitude * x * x * (1.0 - x) * (1.0 - x);
        Ok((energy, Array1::from_vec(vec![gradient])))
    }
}

/// 2D symmetric double-well with symmetric transverse stiffening:
/// `V(x, y) = (1 - x^2)^2 + 0.5 * (1.5 + 0.5 * x^2) * y^2`.
struct SymmetricWell2D;

impl rgsaddle::PointSurface for SymmetricWell2D {
    fn eval(&self, r: ArrayView1<'_, f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
        let x = r[0];
        let y = r[1];
        let u = 1.0 - x * x;
        let ky = 1.5 + 0.5 * x * x;
        let energy = u * u + 0.5 * ky * y * y;
        let gx = -4.0 * x * u + 0.5 * x * y * y;
        let gy = ky * y;
        Ok((energy, Array1::from_vec(vec![gx, gy])))
    }
}

/// Smooth double well with a thermally fluctuating transverse stiffness.
struct ThermalBarrier2D;

impl rgsaddle::PointSurface for ThermalBarrier2D {
    fn eval(&self, r: ArrayView1<'_, f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
        let x = r[0];
        let y = r[1];
        let u = 1.0 - x * x;
        let transverse = 1.0 + 8.0 * x * x;
        let energy = 0.1 * u * u + 0.5 * transverse * y * y;
        let gx = -0.4 * x * u + 8.0 * x * y * y;
        let gy = transverse * y;
        Ok((energy, Array1::from_vec(vec![gx, gy])))
    }
}

struct AnisotropicHarmonic2D;

impl rgsaddle::PointSurface for AnisotropicHarmonic2D {
    fn eval(&self, r: ArrayView1<'_, f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
        Ok((
            0.5 * (r[0] * r[0] + 3.0 * r[1] * r[1]),
            Array1::from_vec(vec![r[0], 3.0 * r[1]]),
        ))
    }
}

fn path_1d(start: f64, end: f64, images: usize) -> Array2<f64> {
    let mut path = Array2::zeros((images, 1));
    for i in 0..images {
        let t = (i as f64) / ((images - 1) as f64);
        path[[i, 0]] = start + t * (end - start);
    }
    path
}

#[test]
fn equidistribution_map_is_strictly_monotone_scale_invariant_and_never_collapses() {
    // Exact asymmetric case. The monitor integral is 4 over a path of length 2,
    // so the defensive floor is 2 and sigma(1) = 3/8. Inverting equal sigma
    // increments gives non-uniform planes (0, 2/3, 6/5, 8/5, 2).
    let exact_map =
        EquidistributionMap::new(&[0.0, 1.0, 2.0], &[0.0, 2.0, 4.0]).expect("exact map");
    assert_abs_diff_eq!(exact_map.floor(), 2.0, epsilon = 1.0e-15);
    assert_abs_diff_eq!(exact_map.sigma(1.0).expect("sigma(1)"), 3.0 / 8.0);
    let exact_planes = exact_map.place_planes(5).expect("exact planes");
    for (&actual, expected) in exact_planes
        .iter()
        .zip([0.0, 2.0 / 3.0, 6.0 / 5.0, 8.0 / 5.0, 2.0])
    {
        assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-14);
    }

    let knots: Vec<f64> = (0..=40).map(|k| (k as f64) * 0.05).collect();
    // Extreme localized spike at the midpoint plus flat zero tails.
    let mut spiky_monitor = vec![0.0_f64; knots.len()];
    spiky_monitor[19] = 500.0;
    spiky_monitor[20] = 1.0e6;
    spiky_monitor[21] = 500.0;

    let map = EquidistributionMap::new(&knots, &spiky_monitor).expect("equidistribution map");
    assert!(
        map.floor() > 0.0,
        "principled floor must be positive for a non-zero monitor"
    );
    assert_abs_diff_eq!(map.sigma(0.0).expect("sigma(0)"), 0.0, epsilon = 1.0e-15);
    assert_abs_diff_eq!(map.sigma(2.0).expect("sigma(L)"), 1.0, epsilon = 1.0e-15);

    // Strict monotonicity of sigma(s) and round-trip inversion.
    let mut prev_sigma = 0.0_f64;
    for step in 1..=200 {
        let s = (step as f64) * 0.01;
        let sig = map.sigma(s).expect("sigma");
        assert!(
            sig > prev_sigma,
            "sigma must strictly increase: sigma({s}) = {sig} <= {prev_sigma}"
        );
        let recovered = map.inverse(sig).expect("inverse");
        assert_abs_diff_eq!(recovered, s, epsilon = 1.0e-12);
        prev_sigma = sig;
    }

    // Planes bunch near the spike (s = 1.0) while obeying the certified spacing floor.
    let n_planes = 17usize;
    let placed = map.place_planes(n_planes).expect("place planes");
    let min_bound = map.min_spacing_bound(n_planes).expect("min spacing bound");
    assert!(min_bound > 0.0, "certified min spacing bound must be > 0");
    let mut min_ds = f64::INFINITY;
    let mut max_ds = 0.0_f64;
    for k in 1..n_planes {
        let ds = placed[k] - placed[k - 1];
        assert!(
            ds >= min_bound * (1.0 - 64.0 * f64::EPSILON),
            "plane interval {k} collapsed: ds={ds} < bound={min_bound}"
        );
        min_ds = min_ds.min(ds);
        max_ds = max_ds.max(ds);
    }
    let mid_ds = (placed[n_planes / 2] - placed[n_planes / 2 - 1])
        .min(placed[n_planes / 2 + 1] - placed[n_planes / 2]);
    let end_ds = placed[1] - placed[0];
    assert!(
        mid_ds < 0.25 * end_ds,
        "planes must bunch at the monitor spike: mid_ds={mid_ds}, end_ds={end_ds}"
    );

    // Scale invariance: multiplying the monitor by 1e4 leaves sigma and placed planes unchanged.
    let scaled_monitor: Vec<f64> = spiky_monitor.iter().map(|&w| w * 1.0e4).collect();
    let scaled_map = EquidistributionMap::new(&knots, &scaled_monitor).expect("scaled map");
    let scaled_planes = scaled_map.place_planes(n_planes).expect("scaled planes");
    for (a, b) in placed.iter().zip(&scaled_planes) {
        assert_abs_diff_eq!(a, b, epsilon = 1.0e-12);
    }

    // All MonitorKind variants produce valid, strictly increasing plane placements.
    let gradients: Vec<f64> = knots
        .iter()
        .map(|&s| LocalizedDoubleWell1D::exact_gradient(s - 1.0))
        .collect();
    let curvatures: Vec<f64> = knots
        .iter()
        .map(|&s| (-4.0 * (s - 1.0) * (s - 1.0)).exp())
        .collect();
    let variances: Vec<f64> = knots
        .iter()
        .map(|&s| 0.01 + 0.5 * (-8.0 * (s - 1.0).powi(2)).exp())
        .collect();
    let inputs = MonitorInputs::new(&knots)
        .with_force_gradients(&gradients)
        .with_path_curvatures(&curvatures)
        .with_tau_weighted_variances(&variances);
    for kind in [
        MonitorKind::ForceGradient,
        MonitorKind::PathCurvature,
        MonitorKind::AutocorrelatedVariance,
        MonitorKind::Combined,
    ] {
        let kind_map = EquidistributionMap::from_inputs(&inputs, kind).expect("monitor map");
        let kind_planes = kind_map.place_planes(11).expect("kind planes");
        for k in 1..kind_planes.len() {
            assert!(kind_planes[k] > kind_planes[k - 1]);
        }
    }
}

#[test]
fn harmonic_limit_has_zero_spectral_tail_and_exact_profile() {
    let k_spring = 2.5_f64;
    let length = 1.6_f64;
    let path = path_1d(0.0, length, 9);
    let mut engine = NativeEngine::new(
        Harmonic1D { k: k_spring },
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0]))
            .with_thermostat(ThermostatConfig::default().with_temperature(0.1)),
    )
    .expect("harmonic engine");

    let adaptive_cfg = AdaptivePafiConfig::default()
        .with_base(
            PafiConfig::default()
                .with_planes(5)
                .with_burn_in_steps(2)
                .with_sample_steps(8)
                .with_target_error(1.0e-6)
                .with_stats(StatsConfig::default().with_min_batches(2)),
        )
        .with_max_planes(25)
        .with_max_refinements(4);

    let result =
        pafi_profile_adaptive(&mut engine, path.view(), &adaptive_cfg).expect("harmonic adaptive");
    let exact_barrier = 0.5 * k_spring * length * length;
    let kbt = 0.1_f64;
    let exact_log_rate = 0.5 * k_spring.ln() - std::f64::consts::TAU.ln() - exact_barrier / kbt;
    let exact_rate = exact_log_rate.exp();
    let log_rate_roundoff =
        256.0 * f64::EPSILON * (1.0 + exact_log_rate.abs() + exact_barrier / kbt);
    let rate_roundoff = exact_rate * log_rate_roundoff;
    assert_abs_diff_eq!(result.profile.barrier, exact_barrier, epsilon = 1.0e-12);
    assert_abs_diff_eq!(
        result.profile.normalized_log_rate,
        exact_log_rate,
        epsilon = log_rate_roundoff
    );
    assert_abs_diff_eq!(
        result
            .profile
            .normalized_rate
            .expect("representable harmonic rate"),
        exact_rate,
        epsilon = rate_roundoff
    );
    assert!(
        result.tail.tail_indicator <= 1.0e-12,
        "harmonic limit must have zero spectral tail indicator, got {}",
        result.tail.tail_indicator
    );
    assert!(
        result.tail.flagged_intervals.is_empty(),
        "harmonic limit must flag no intervals, got {:?}",
        result.tail.flagged_intervals
    );
    assert_eq!(
        result.refinement_rounds, 0,
        "harmonic limit must stop without adding refinement planes"
    );
    for (&s, &f) in result
        .profile
        .arc_lengths
        .iter()
        .zip(&result.profile.free_energy)
    {
        assert_abs_diff_eq!(f, 0.5 * k_spring * s * s, epsilon = 1.0e-12);
    }

    // Verify spectral tail estimator flags intervals on a coarse under-resolved quartic barrier.
    let coarse_arcs = [0.0, 0.5, 1.0, 1.5, 2.0];
    let coarse_grads: Vec<f64> = coarse_arcs
        .iter()
        .map(|&s| LocalizedDoubleWell1D::exact_gradient(s - 1.0))
        .collect();
    let coarse_tail =
        estimate_spectral_tail(&coarse_arcs, &coarse_grads, 1.0e-4).expect("coarse tail");
    let localized_tail = estimate_spectral_tail(&coarse_arcs, &[0.0, 0.0, 10.0, 0.0, 0.0], 2.0)
        .expect("localized tail");
    assert_eq!(
        localized_tail.flagged_intervals,
        vec![1, 2, 3],
        "interval selection must remain local rather than flagging every interval"
    );
    assert!(
        coarse_tail.tail_indicator > 1.0e-2,
        "coarse 5-plane localized double well must have non-trivial spectral tail indicator, got {}",
        coarse_tail.tail_indicator
    );
    assert!(
        !coarse_tail.flagged_intervals.is_empty(),
        "coarse 5-plane localized double well must flag intervals for refinement"
    );
}

#[test]
fn adaptive_insertions_enforce_the_full_spacing_certificate() {
    let path = path_1d(-1.0, 1.0, 17);
    let config = AdaptivePafiConfig::default()
        .with_base(
            PafiConfig::default()
                .with_planes(5)
                .with_burn_in_steps(2)
                .with_sample_steps(8)
                .with_zero_temperature_tol(0.5)
                .with_stats(StatsConfig::default().with_min_batches(2)),
        )
        .with_max_planes(11)
        .with_max_refinements(4)
        .with_monitor(MonitorKind::ForceGradient);
    let mut engine = NativeEngine::new(
        LocalizedDoubleWell1D::new(),
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0]))
            .with_thermostat(ThermostatConfig::default().with_temperature(0.1)),
    )
    .expect("spacing engine");
    let adaptive =
        pafi_profile_adaptive(&mut engine, path.view(), &config).expect("adaptive spacing profile");
    assert!(adaptive.refinement_rounds > 0);

    let bound = adaptive
        .equidistribution
        .min_spacing_bound(config.max_planes)
        .expect("spacing certificate");
    for pair in adaptive.profile.arc_lengths.windows(2) {
        let gap = pair[1] - pair[0];
        let tolerance = 64.0 * f64::EPSILON * (1.0 + pair[0].abs() + pair[1].abs() + bound);
        assert!(
            gap + tolerance >= bound,
            "adaptive gap {gap} violated full spacing certificate {bound}"
        );
    }
}

#[test]
fn zero_error_smooth_barrier_refines_below_a_fixed_absolute_threshold() {
    let path = path_1d(0.0, 1.0, 17);
    let base = PafiConfig::default()
        .with_planes(5)
        .with_burn_in_steps(2)
        .with_sample_steps(8)
        .with_zero_temperature_tol(0.5)
        .with_stats(StatsConfig::default().with_min_batches(2));
    let config = AdaptivePafiConfig::default()
        .with_base(base.clone())
        .with_max_planes(9)
        .with_max_refinements(1)
        .with_monitor(MonitorKind::PathCurvature);
    let make_engine = || {
        NativeEngine::new(
            SmoothRipple1D,
            NativeConfig::default()
                .with_masses(Array1::from_vec(vec![1.0]))
                .with_thermostat(ThermostatConfig::default().with_temperature(0.1)),
        )
        .expect("smooth ripple engine")
    };
    let mut coarse_engine = make_engine();
    let coarse = pafi_profile(&mut coarse_engine, path.view(), &base).expect("coarse profile");
    assert_eq!(coarse.barrier_error, 0.0);
    let coarse_arcs = [0.0, 0.25, 0.5, 0.75, 1.0];
    let coarse_gradients: Vec<f64> = coarse_arcs
        .iter()
        .map(|&x| 1.0 + 0.25 * x + 0.25 * x * x * (1.0 - x) * (1.0 - x))
        .collect();
    let coarse_tail = estimate_spectral_tail(
        &coarse_arcs,
        &coarse_gradients,
        64.0 * f64::EPSILON * (1.0 + coarse.barrier.abs()),
    )
    .expect("coarse smooth tail");
    assert!(coarse_tail.tail_indicator < 0.05);

    let mut engine = make_engine();

    let adaptive =
        pafi_profile_adaptive(&mut engine, path.view(), &config).expect("smooth adaptive profile");
    assert!(
        adaptive.tail.tail_indicator < 0.05,
        "tail must remain below the mutant fixed threshold"
    );
    assert_eq!(adaptive.refinement_rounds, 1);
    assert_eq!(adaptive.profile.arc_lengths.len(), 7);

    let loose_config = config.clone().with_base(base.with_target_error(1.0));
    let mut loose_engine = make_engine();
    let loose = pafi_profile_adaptive(&mut loose_engine, path.view(), &loose_config)
        .expect("loose-target adaptive profile");
    assert_eq!(loose.refinement_rounds, 1);
    assert_eq!(loose.profile.arc_lengths.len(), 7);
    assert_eq!(loose.profile.arc_lengths, adaptive.profile.arc_lengths);
}

#[test]
fn tight_target_error_does_not_change_thermal_spectral_refinement() {
    let mut path = Array2::zeros((17, 2));
    for i in 0..17 {
        path[[i, 0]] = -1.0 + 2.0 * (i as f64) / 16.0;
    }
    let base = PafiConfig::default()
        .with_planes(5)
        .with_burn_in_steps(200)
        .with_sample_steps(400)
        .with_time_step(0.005)
        .with_max_rounds(1)
        .with_zero_temperature_tol(0.5)
        .with_stats(StatsConfig::default().with_min_batches(4));
    let reference_config = AdaptivePafiConfig::default()
        .with_base(base.clone())
        .with_max_planes(15)
        .with_max_refinements(4)
        .with_monitor(MonitorKind::ForceGradient);
    let targeted_config = reference_config
        .clone()
        .with_base(base.with_target_error(1.0e-12));
    let make_engine = || {
        NativeEngine::new(
            ThermalBarrier2D,
            NativeConfig::default()
                .with_masses(Array1::from_vec(vec![1.0, 1.0]))
                .with_thermostat(
                    ThermostatConfig::default()
                        .with_gamma(1.5)
                        .with_temperature(0.5),
                ),
        )
        .expect("thermal barrier engine")
    };

    let mut reference_engine = make_engine();
    let reference = pafi_profile_adaptive(&mut reference_engine, path.view(), &reference_config)
        .expect("reference adaptive profile");
    let mut targeted_engine = make_engine();
    let targeted = pafi_profile_adaptive(&mut targeted_engine, path.view(), &targeted_config)
        .expect("targeted adaptive profile");

    assert!(reference.profile.barrier_error > 0.0);
    assert_eq!(targeted.refinement_rounds, reference.refinement_rounds);
    assert_eq!(targeted.profile.arc_lengths, reference.profile.arc_lengths);
}

#[test]
fn symmetric_hop_preserves_product_equals_reactant_free_energy() {
    let path = path_1d(-1.0, 1.0, 17);
    let mut engine = NativeEngine::new(
        LocalizedDoubleWell1D::new(),
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0]))
            .with_thermostat(ThermostatConfig::default().with_temperature(0.1)),
    )
    .expect("1d symmetric engine");

    let cfg = AdaptivePafiConfig::default()
        .with_base(
            PafiConfig::default()
                .with_planes(5)
                .with_burn_in_steps(2)
                .with_sample_steps(8)
                .with_target_error(2.0e-3)
                .with_zero_temperature_tol(0.5)
                .with_stats(StatsConfig::default().with_min_batches(2)),
        )
        .with_max_planes(25)
        .with_max_refinements(5)
        .with_monitor(MonitorKind::ForceGradient);

    let adaptive =
        pafi_profile_adaptive(&mut engine, path.view(), &cfg).expect("symmetric adaptive");
    assert!(
        adaptive.refinement_rounds >= 1 && adaptive.profile.arc_lengths.len() > 5,
        "symmetric hop must refine beyond the initial 5 planes"
    );
    let f_reactant = adaptive.profile.free_energy[0];
    let f_product = *adaptive
        .profile
        .free_energy
        .last()
        .expect("product plane exists");
    assert_abs_diff_eq!(f_reactant, 0.0, epsilon = 1.0e-15);
    assert_abs_diff_eq!(f_product, f_reactant, epsilon = 1.0e-13);

    // Check plane coordinate reflection symmetry s_{N-1-k} = L - s_k.
    let arcs = &adaptive.profile.arc_lengths;
    let total_arc = *arcs.last().expect("total arc");
    for k in 0..arcs.len() {
        assert_abs_diff_eq!(
            arcs[k] + arcs[arcs.len() - 1 - k],
            total_arc,
            epsilon = 1.0e-12
        );
    }

    // 2D thermal symmetric hop: F(product) == F(reactant) within statistical SE.
    let mut path_2d = Array2::zeros((17, 2));
    for i in 0..17 {
        path_2d[[i, 0]] = -1.0 + 2.0 * (i as f64) / 16.0;
    }
    let mut engine_2d = NativeEngine::new(
        SymmetricWell2D,
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0, 1.0]))
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(1.5)
                    .with_temperature(0.1),
            ),
    )
    .expect("2d symmetric engine");
    let cfg_2d = AdaptivePafiConfig::default()
        .with_base(
            PafiConfig::default()
                .with_planes(7)
                .with_burn_in_steps(1_000)
                .with_sample_steps(4_000)
                .with_time_step(0.002)
                .with_zero_temperature_tol(0.2)
                .with_stats(StatsConfig::default().with_min_batches(4)),
        )
        .with_max_planes(15)
        .with_max_refinements(3);
    let adaptive_2d =
        pafi_profile_adaptive(&mut engine_2d, path_2d.view(), &cfg_2d).expect("2d adaptive");
    let end_idx = adaptive_2d.profile.free_energy.len() - 1;
    let end_f = adaptive_2d.profile.free_energy[end_idx];
    let end_se = adaptive_2d.profile.standard_error[end_idx];
    assert!(
        end_f.abs() <= 2.5 * end_se,
        "2D symmetric hop product free energy {end_f} exceeds 2.5 * SE ({end_se})"
    );
}

#[test]
fn adaptive_driver_uses_the_selected_monitor() {
    let mut path = Array2::zeros((17, 2));
    for i in 0..17 {
        let x = 0.5 + (i as f64) / 16.0;
        path[[i, 0]] = x;
        path[[i, 1]] = 0.3 * x * x;
    }
    let base = PafiConfig::default()
        .with_planes(7)
        .with_burn_in_steps(2)
        .with_sample_steps(8)
        .with_zero_temperature_tol(1.0)
        .with_stats(StatsConfig::default().with_min_batches(2));
    let run = |monitor| {
        let mut engine = NativeEngine::new(
            AnisotropicHarmonic2D,
            NativeConfig::default()
                .with_masses(Array1::ones(2))
                .with_thermostat(ThermostatConfig::default().with_temperature(0.1)),
        )
        .expect("curved harmonic engine");
        pafi_profile_adaptive(
            &mut engine,
            path.view(),
            &AdaptivePafiConfig::default()
                .with_base(base.clone())
                .with_max_planes(17)
                .with_max_refinements(1)
                .with_monitor(monitor),
        )
        .expect("adaptive monitor profile")
    };

    let force = run(MonitorKind::ForceGradient);
    let curvature = run(MonitorKind::PathCurvature);
    let uniform = run(MonitorKind::AutocorrelatedVariance);
    let force_arcs = &force.profile.arc_lengths;
    let curvature_arcs = &curvature.profile.arc_lengths;
    let uniform_arcs = &uniform.profile.arc_lengths;
    for result in [&force, &curvature, &uniform] {
        assert!(
            result.refinement_rounds > 0 && result.profile.arc_lengths.len() > 7,
            "the monitor regression must exercise an actual plane-insertion round"
        );
    }
    let force_curvature_distance: f64 = force_arcs
        .iter()
        .zip(curvature_arcs)
        .map(|(a, b)| (a - b).abs())
        .sum();
    let curvature_uniform_distance: f64 = curvature_arcs
        .iter()
        .zip(uniform_arcs)
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        force_curvature_distance > 1.0e-3,
        "PathCurvature placement must differ from ForceGradient placement"
    );
    assert!(
        curvature_uniform_distance > 1.0e-3,
        "the adaptive driver must not ignore PathCurvature and place uniform planes"
    );
    let expected_force_monitor = MonitorInputs::new(&force.profile.arc_lengths)
        .with_force_gradients(force.profile.gradients())
        .evaluate(MonitorKind::ForceGradient)
        .expect("expected force-gradient monitor");
    let force_monitor_error: f64 = force
        .equidistribution
        .monitor_values()
        .iter()
        .zip(expected_force_monitor)
        .map(|(actual, expected)| (actual - expected).abs())
        .sum();
    assert!(
        force_monitor_error <= 1.0e-12,
        "the refinement driver must retain the selected ForceGradient monitor"
    );
    let selected_monitor = curvature.equidistribution.monitor_values();
    // Rebuild the selected monitor from the raw geometric channel returned
    // before monitor selection. This independently checks the driver's
    // mapping without pinning implementation-generated golden numbers.
    let expected_path_curvature = MonitorInputs::new(&curvature.profile.arc_lengths)
        .with_path_curvatures(&curvature.final_path_curvatures)
        .evaluate(MonitorKind::PathCurvature)
        .expect("expected path-curvature monitor");
    let path_curvature_error = selected_monitor
        .iter()
        .zip(&expected_path_curvature)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        selected_monitor.len() == expected_path_curvature.len() && path_curvature_error <= 1.0e-12,
        "PathCurvature final monitor changed: {selected_monitor:?}"
    );
    assert!(
        selected_monitor.iter().any(|&value| value > 1.0e-6),
        "the final driver map must retain non-zero PathCurvature values"
    );
    let force_monitor = MonitorInputs::new(&curvature.profile.arc_lengths)
        .with_force_gradients(curvature.profile.gradients())
        .evaluate(MonitorKind::ForceGradient)
        .expect("force-gradient monitor");
    let selected_force_distance: f64 = selected_monitor
        .iter()
        .zip(force_monitor)
        .map(|(selected, force)| (selected - force).abs())
        .sum();
    assert!(
        selected_force_distance > 1.0e-3,
        "PathCurvature must not be replaced by ForceGradient in the adaptive driver"
    );
    let variance_monitor = uniform.equidistribution.monitor_values();
    // The fixed-seed force samples are intentionally private, so the profile
    // exposes their final pre-selection variance channel. Re-evaluating that
    // channel here checks monitor selection and normalization independently.
    let mut expected_variance = MonitorInputs::new(&uniform.profile.arc_lengths)
        .with_tau_weighted_variances(&uniform.final_tau_weighted_variances)
        .evaluate(MonitorKind::AutocorrelatedVariance)
        .expect("expected autocorrelated-variance monitor");
    for i in 0..(expected_variance.len() / 2) {
        let mirror = expected_variance.len() - 1 - i;
        let average = 0.5 * (expected_variance[i] + expected_variance[mirror]);
        expected_variance[i] = average;
        expected_variance[mirror] = average;
    }
    let variance_error = variance_monitor
        .iter()
        .zip(&expected_variance)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        variance_monitor.len() == expected_variance.len() && variance_error <= 1.0e-12,
        "AutocorrelatedVariance final monitor changed: {variance_monitor:?}"
    );
}

#[test]
fn adaptive_plane_placement_tradeoffs_are_reported_honestly() {
    let path = path_1d(-1.0, 1.0, 65);
    let burn_in = 2usize;
    let samples = 8usize;

    // 1. Analytic 1D double-well: V(x) = (1 - x^2)^2 / (1 + 6 * x^2), exact F* = 1.0.
    let exact_dw_barrier = 1.0_f64;
    let dw_adaptive_cfg = AdaptivePafiConfig::default()
        .with_base(
            PafiConfig::default()
                .with_planes(5)
                .with_burn_in_steps(burn_in)
                .with_sample_steps(samples)
                .with_target_error(4.0e-3)
                .with_zero_temperature_tol(0.5)
                .with_stats(StatsConfig::default().with_min_batches(2)),
        )
        .with_max_planes(17)
        .with_max_refinements(5)
        .with_monitor(MonitorKind::ForceGradient);

    let mut dw_adaptive_engine = NativeEngine::new(
        LocalizedDoubleWell1D::new(),
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0]))
            .with_thermostat(ThermostatConfig::default().with_temperature(0.1)),
    )
    .expect("dw adaptive engine");
    let dw_adaptive = pafi_profile_adaptive(&mut dw_adaptive_engine, path.view(), &dw_adaptive_cfg)
        .expect("dw adaptive profile");
    assert!(
        dw_adaptive.profile.quadrature_residual > 0.0,
        "adaptive profile must retain its zero-temperature quadrature residual"
    );
    let dw_adaptive_err = (dw_adaptive.profile.barrier - exact_dw_barrier).abs();
    let dw_adaptive_planes = dw_adaptive.profile.arc_lengths.len();
    let dw_adaptive_calls = dw_adaptive_engine.surface().call_count();

    // Find the minimum even-spacing plane count that reaches the same F* error.
    let mut dw_even_planes = 0usize;
    let mut dw_even_calls = 0usize;
    let mut dw_even_err = f64::INFINITY;
    for candidate_planes in (5..=99).step_by(2) {
        let mut engine = NativeEngine::new(
            LocalizedDoubleWell1D::new(),
            NativeConfig::default()
                .with_masses(Array1::from_vec(vec![1.0]))
                .with_thermostat(ThermostatConfig::default().with_temperature(0.1)),
        )
        .expect("dw even engine");
        let cfg = PafiConfig::default()
            .with_planes(candidate_planes)
            .with_burn_in_steps(burn_in)
            .with_sample_steps(samples)
            .with_zero_temperature_tol(0.5)
            .with_stats(StatsConfig::default().with_min_batches(2));
        let prof = pafi_profile(&mut engine, path.view(), &cfg).expect("dw even profile");
        let err = (prof.barrier - exact_dw_barrier).abs();
        if err <= dw_adaptive_err {
            dw_even_planes = candidate_planes;
            dw_even_calls = engine.surface().call_count();
            dw_even_err = err;
            break;
        }
    }
    eprintln!(
        "1D double-well matched error: adaptive planes={dw_adaptive_planes} calls={dw_adaptive_calls} err={dw_adaptive_err:.6e} vs even planes={dw_even_planes} calls={dw_even_calls} err={dw_even_err:.6e}"
    );
    assert_eq!(
        (dw_adaptive_planes, dw_adaptive_calls),
        (15, 670),
        "double-well adaptive surface-evaluation regression changed"
    );
    assert_eq!(
        (dw_even_planes, dw_even_calls),
        (27, 329),
        "double-well even-grid surface-evaluation regression changed"
    );
    assert!(
        dw_adaptive_err <= 5.0e-3,
        "1D double-well adaptive error {dw_adaptive_err} exceeded target"
    );
    assert!(
        dw_even_planes > dw_adaptive_planes,
        "adaptive ({dw_adaptive_planes} planes) must use fewer planes than even spacing ({dw_even_planes} planes) at matched error"
    );
    assert!(
        dw_adaptive_calls > dw_even_calls,
        "adaptive sampling must honestly report its larger force-call cost ({dw_adaptive_calls} vs {dw_even_calls})"
    );

    // 2. Analytic 1D asymmetric barrier with known exact F(s) and F*.
    let exact_asym_barrier = AsymmetricBarrier1D::exact_barrier();
    let asym_adaptive_cfg = AdaptivePafiConfig::default()
        .with_base(
            PafiConfig::default()
                .with_planes(5)
                .with_burn_in_steps(burn_in)
                .with_sample_steps(samples)
                .with_target_error(5.0e-3)
                .with_zero_temperature_tol(0.5)
                .with_stats(StatsConfig::default().with_min_batches(2)),
        )
        .with_max_planes(19)
        .with_max_refinements(5)
        .with_monitor(MonitorKind::ForceGradient);

    let mut asym_adaptive_engine = NativeEngine::new(
        AsymmetricBarrier1D::new(),
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0]))
            .with_thermostat(ThermostatConfig::default().with_temperature(0.1)),
    )
    .expect("asym adaptive engine");
    let asym_adaptive =
        pafi_profile_adaptive(&mut asym_adaptive_engine, path.view(), &asym_adaptive_cfg)
            .expect("asym adaptive profile");
    let asym_adaptive_err = (asym_adaptive.profile.barrier - exact_asym_barrier).abs();
    let asym_adaptive_planes = asym_adaptive.profile.arc_lengths.len();
    let asym_adaptive_calls = asym_adaptive_engine.surface().call_count();

    let mut asym_even_planes = 0usize;
    let mut asym_even_calls = 0usize;
    let mut asym_even_err = f64::INFINITY;
    for candidate_planes in 5..=120 {
        let mut engine = NativeEngine::new(
            AsymmetricBarrier1D::new(),
            NativeConfig::default()
                .with_masses(Array1::from_vec(vec![1.0]))
                .with_thermostat(ThermostatConfig::default().with_temperature(0.1)),
        )
        .expect("asym even engine");
        let cfg = PafiConfig::default()
            .with_planes(candidate_planes)
            .with_burn_in_steps(burn_in)
            .with_sample_steps(samples)
            .with_zero_temperature_tol(0.5)
            .with_stats(StatsConfig::default().with_min_batches(2));
        let prof = pafi_profile(&mut engine, path.view(), &cfg).expect("asym even profile");
        let err = (prof.barrier - exact_asym_barrier).abs();
        if err <= asym_adaptive_err {
            asym_even_planes = candidate_planes;
            asym_even_calls = engine.surface().call_count();
            asym_even_err = err;
            break;
        }
    }
    eprintln!(
        "1D asymmetric barrier matched error: adaptive planes={asym_adaptive_planes} calls={asym_adaptive_calls} err={asym_adaptive_err:.6e} vs even planes={asym_even_planes} calls={asym_even_calls} err={asym_even_err:.6e}"
    );
    assert_eq!(
        (asym_adaptive_planes, asym_adaptive_calls),
        (15, 670),
        "asymmetric adaptive surface-evaluation regression changed"
    );
    assert_eq!(
        (asym_even_planes, asym_even_calls),
        (12, 149),
        "asymmetric even-grid surface-evaluation regression changed"
    );
    assert!(
        asym_adaptive_err <= 8.0e-3,
        "1D asymmetric barrier adaptive error {asym_adaptive_err} exceeded target"
    );
    assert!(
        asym_adaptive.refinement_rounds > 0,
        "asymmetric barrier must trigger spectral-tail refinement"
    );
    assert!(
        asym_even_planes < asym_adaptive_planes,
        "uniform placement must honestly report fewer planes ({asym_even_planes} vs {asym_adaptive_planes})"
    );
    assert!(
        asym_even_calls < asym_adaptive_calls,
        "uniform placement must honestly report fewer force calls ({asym_even_calls} vs {asym_adaptive_calls})"
    );
    assert!(
        asym_even_err < asym_adaptive_err,
        "uniform placement must honestly report lower error ({asym_even_err} vs {asym_adaptive_err})"
    );
}

#[test]
fn tau_int_window_sizing_scales_with_autocorrelation_and_batches() {
    let stats = StatsConfig::default().with_min_batches(16);
    let (burn, sample) = size_window_from_tau_int(12.5, 40, 128, &stats).expect("window size");
    // g = ceil(2 * 12.5) = 25; sample >= 16 * 25 = 400; burn >= 8 * 25 = 200.
    assert_eq!(sample, 400);
    assert_eq!(burn, 200);
}

/// 2D analytic double-well with linear transverse coupling:
/// `V(x, y) = (1 - x^2)^2 + alpha * x * y + 0.5 * k_y * y^2`.
///
/// On each plane `x = const`, `y ~ N(-alpha * x / k_y, k_B T / k_y)`, so the exact
/// analytical mean constraint force is `g_exact(x) = -4 * x * (1 - x^2) - (alpha^2 / k_y) * x`
/// and the instantaneous force carries Gaussian thermal variance `alpha^2 * k_B T / k_y`.
struct CoupledDoubleWell2D {
    alpha: f64,
    k_y: f64,
}

impl CoupledDoubleWell2D {
    fn exact_mean_gradient(&self, x: f64) -> f64 {
        -4.0 * x * (1.0 - x * x) - (self.alpha * self.alpha / self.k_y) * x
    }
}

impl rgsaddle::PointSurface for CoupledDoubleWell2D {
    fn eval(&self, r: ArrayView1<'_, f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
        let x = r[0];
        let y = r[1];
        let u = 1.0 - x * x;
        let energy = u * u + self.alpha * x * y + 0.5 * self.k_y * y * y;
        let gx = -4.0 * x * u + self.alpha * y;
        let gy = self.alpha * x + self.k_y * y;
        Ok((energy, Array1::from_vec(vec![gx, gy])))
    }
}

/// Exact 1D potential of mean force obtained by analytically integrating out `y`
/// in [`CoupledDoubleWell2D`]: `F(x) = (1 - x^2)^2 + 0.5 * (alpha^2 / k_y) * (1 - x^2)`.
struct ExactCoupledPmf1D {
    quad_coeff: f64,
}

impl rgsaddle::PointSurface for ExactCoupledPmf1D {
    fn eval(&self, r: ArrayView1<'_, f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
        let x = r[0];
        let u = 1.0 - x * x;
        Ok((
            u * u + 0.5 * self.quad_coeff * u,
            Array1::from_vec(vec![-4.0 * x * u - self.quad_coeff * x]),
        ))
    }
}

#[test]
fn pooled_rate_eliminates_jensen_bias_on_analytic_double_well() {
    let surface = CoupledDoubleWell2D {
        alpha: 1.1,
        k_y: 1.0,
    };
    let kbt = 0.12_f64;
    let planes = 15usize;
    let n_replicates = 80usize;

    let mut path_2d = Array2::zeros((17, 2));
    for i in 0..17 {
        path_2d[[i, 0]] = -1.0 + 2.0 * (i as f64) / 16.0;
    }

    let mut replicates = Vec::with_capacity(n_replicates);
    for rep in 0..n_replicates {
        let mut engine = NativeEngine::new(
            CoupledDoubleWell2D {
                alpha: surface.alpha,
                k_y: surface.k_y,
            },
            NativeConfig::default()
                .with_masses(Array1::from_vec(vec![1.0, 1.0]))
                .with_thermostat(
                    ThermostatConfig::default()
                        .with_gamma(1.5)
                        .with_temperature(kbt),
                )
                .with_seed(10_000 + (rep as u64) * 7_919),
        )
        .expect("coupled 2d engine");
        let cfg = PafiConfig::default()
            .with_planes(planes)
            .with_burn_in_steps(100)
            .with_sample_steps(12)
            .with_time_step(0.08)
            .with_zero_temperature_tol(0.25)
            .with_stats(StatsConfig::default().with_min_batches(3))
            .with_seed(50_000 + (rep as u64) * 104_729);
        let prof = pafi_profile(&mut engine, path_2d.view(), &cfg).expect("replicate profile");
        replicates.push(prof);
    }

    // Build the exact infinite-sample reference profile on the same 15-plane knot grid.
    let path_1d_pmf = path_1d(-1.0, 1.0, 17);
    let mut exact_engine = NativeEngine::new(
        ExactCoupledPmf1D {
            quad_coeff: (surface.alpha * surface.alpha) / surface.k_y,
        },
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0]))
            .with_thermostat(ThermostatConfig::default().with_temperature(kbt)),
    )
    .expect("exact 1d pmf engine");
    let exact_profile = pafi_profile(
        &mut exact_engine,
        path_1d_pmf.view(),
        &PafiConfig::default()
            .with_planes(planes)
            .with_burn_in_steps(2)
            .with_sample_steps(8)
            .with_zero_temperature_tol(0.25)
            .with_stats(StatsConfig::default().with_min_batches(2)),
    )
    .expect("exact profile");
    for (g_ref, &s) in exact_profile
        .gradients()
        .iter()
        .zip(&exact_profile.arc_lengths)
    {
        assert_abs_diff_eq!(
            *g_ref,
            surface.exact_mean_gradient(s - 1.0),
            epsilon = 1.0e-12
        );
    }
    let exact_rate = exact_profile
        .normalized_rate
        .expect("representable exact rate");

    // 1. Per-replicate arithmetic mean of k^(r) is inflated by positive Jensen bias:
    //    E[exp(-F* / kT)] > exp(-E[F*] / kT).
    let n_f64 = n_replicates as f64;
    let arith_mean_rate: f64 = replicates
        .iter()
        .map(|p| p.normalized_rate.expect("representable replicate rate"))
        .sum::<f64>()
        / n_f64;
    let arith_var: f64 = replicates
        .iter()
        .map(|p| {
            let d = p.normalized_rate.expect("representable replicate rate") - arith_mean_rate;
            d * d
        })
        .sum::<f64>()
        / (n_f64 - 1.0);
    let arith_se = (arith_var / n_f64).sqrt();
    let arith_z = (arith_mean_rate - exact_rate) / arith_se;

    // 2. Pooled estimator integrates the replicate-averaged plane forces before exponentiating
    //    and MUST pass within 2 SE of the exact rate.
    let pooled = pool_pafi_replicates(&replicates, kbt).expect("pooled profile");
    let pooled_rate = pooled.normalized_rate.expect("representable pooled rate");
    let pooled_rate_error = pooled
        .normalized_rate_error
        .expect("representable pooled rate error");
    let pooled_z = (pooled_rate - exact_rate).abs() / pooled_rate_error;

    eprintln!(
        "Analytic 2D double-well Jensen check: exact_k={exact_rate:.6e}, arith_mean_k={arith_mean_rate:.6e} +- {arith_se:.6e} (z=+{arith_z:.2}), pooled_k={pooled_rate:.6e} +- {pooled_rate_error:.6e} (|z|={pooled_z:.2})"
    );

    let separation_roundoff = 64.0 * f64::EPSILON * arith_mean_rate.abs().max(pooled_rate.abs());
    assert!(
        arith_mean_rate - pooled_rate > separation_roundoff,
        "per-replicate arithmetic mean must differ from and exceed the force-pooled rate"
    );
    assert!(
        pooled_z <= 2.0,
        "pooled rate must agree with exact analytical rate within 2 SE, got |z| = {pooled_z:.3}"
    );
}
