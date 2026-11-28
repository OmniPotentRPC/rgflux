//! Statistical budget and seed control shared by the PAFI integration tests.
//!
//! Family: each call to [`assert_coverage`], [`assert_below`] or
//! [`assert_dividing_plane`] in the two PAFI test files is one statistical
//! assertion. There are [`FAMILY_SIZE`] (`m`) of them, checked by
//! `statistical_family_size_matches_call_sites`.
//!
//! Budget: the suite false failure rate is [`SUITE_ALPHA`] (`alpha`, one in a
//! thousand). The Bonferroni inequality splits `alpha` evenly across the `m`
//! assertions, giving every assertion level `alpha / m` ([`ASSERTION_ALPHA`]).
//!
//! Coverage: planes are seeded independently and each plane standard error is
//! block averaged over [`BLOCKS_PER_PLANE`] blocks, each long compared with
//! the autocorrelation time. In the batch means central limit theorem the
//! block means on each plane are asymptotically independent normal with
//! `nu = BLOCKS_PER_PLANE - 1` degrees of freedom. The barrier is a linear
//! combination of the independent plane means and the rate is linear to first
//! order, so the ratio `(estimate - target) / SE` follows a Student `t`
//! distribution with effective degrees of freedom at least `nu = 39`. The
//! Student `t` upper tail decreases as the degrees of freedom grow, so the
//! two sided quantile `t_{39, 1 - alpha / (2 m)}` ([`T_QUANTILE`]) is an
//! upper bound on the critical value ([`Z_CRIT`]). Targets are the Hermite
//! integral of the analytic mean force on the same planes, the expectation of
//! this estimator.
//!
//! Batch-means caps: for a variance sum of independent normal block means
//! with at least `nu = 39` effective degrees of freedom, `(SE / sigma)^2` is
//! bounded in the upper tail by `chi^2_nu / nu`. At one sided level
//! `alpha / m`, a pure batch-means standard error, or the ratio of two
//! independent ones, is at most `sigma * sqrt(chi^2_{39, 1 - alpha / m} / 39)`
//! ([`CHI2_SE_FACTOR`], [`se_cap`]).
//!
//! Curved barrier cap: the reported standard error on that path is not a pure
//! batch-means error. It is `hypot(S, T)`, with `S` the Hermite propagation
//! of the plane batch-means errors and `T` the absolute Hermite-versus-
//! trapezoid gap of the measured plane means. [`curved_barrier_se_cap`] is
//! the union of a Chernoff bound on `S^2` and a Mills bound on `T`, and
//! [`curved_barrier_se_tail`] is that union's probability. The widening well
//! uses the same union ([`widening_barrier_se_cap`]): its gap is the larger
//! piece. [`end_plane_disagreement_cap`] is a union of Mills tails on the
//! curved increment differences.
//!
//! Seeds: `PAFI_SEED_OFFSET` shifts every seed by a multiple of a large odd
//! constant, so a sweep over offsets draws distinct reproducible suites. PAFI
//! seeds plane `k` of round `r` with `seed + 4099 k + 1000003 r`, so two
//! profiles given the same seed sample identical noise on every plane they
//! share. A bound that sums the variances of two profiles, such as
//! [`end_plane_disagreement_cap`], describes independent profiles, and a sweep
//! of it must give each profile its own stream with [`independent_seed`].
//!
//! Sweeps: [`offset_sweep`] draws a capped quantity at a fixed list of
//! offsets and [`assert_sweep_within_budget`] requires the empirical failure
//! rate of each cap to stay within [`ASSERTION_ALPHA`]. The curved, widening,
//! rate, refinement, end-plane and rgpot caps each have an ignored sweep test
//! that runs with `cargo test --features rgpot -- --ignored sweep`. That
//! command runs all six tests: five native sweeps and one rgpot sweep.

#![allow(dead_code)]

use std::f64::consts::{PI, TAU};

use rgflux::pafi::{HermiteIntegral, MeanForceIntegral};
use rgflux::stats::StatsConfig;

/// Suite level false failure rate `alpha`.
pub const SUITE_ALPHA: f64 = 1.0 / 1_000.0;

/// Number of statistical assertions `m` in the PAFI test suite.
pub const FAMILY_SIZE: usize = 18;

/// Per assertion level `alpha / m` under the Bonferroni bound.
pub const ASSERTION_ALPHA: f64 = SUITE_ALPHA / (FAMILY_SIZE as f64);

/// Batch means blocks per plane, so the standard error has at least `39`
/// degrees of freedom.
pub const BLOCKS_PER_PLANE: usize = 40;

/// Degrees of freedom `nu = BLOCKS_PER_PLANE - 1`.
pub const DEGREES_OF_FREEDOM: usize = BLOCKS_PER_PLANE - 1;

/// Two sided Student `t` quantile `t_{39, 1 - alpha / (2 m)}` with
/// `alpha = 1/1000` and `m = 18` (`1 - alpha / (2 m) = 1 - 1/36000`).
pub const T_QUANTILE: f64 = 4.523_716;

/// Two sided critical value for `|estimate - target| / SE`.
pub const Z_CRIT: f64 = T_QUANTILE;

/// Upper chi-square quantile `chi^2_{39, 1 - alpha / m}` with `alpha = 1/1000`
/// and `m = 18` (`1 - alpha / m = 1 - 1/18000`).
pub const CHI2_QUANTILE: f64 = 82.727_831;

/// `sqrt(chi^2_{39, 1 - alpha / m} / 39)`.
pub const CHI2_SE_FACTOR: f64 = 1.456_443;

/// Share of the assertion level spent on the sampling-error Chernoff bound.
const SAMPLING_TAIL_SHARE: f64 = 0.2;

/// Share of the assertion level spent on the quadrature-gap Mills bound.
const GAP_TAIL_SHARE: f64 = 0.8;

/// Upper cap on a pure batch-means standard error whose expectation is
/// `reference`, at one sided level `alpha / m`.
pub const fn se_cap(reference: f64) -> f64 {
    reference * CHI2_SE_FACTOR
}

/// Upper cap on a block-averaged mean difference with expectation `reference`
/// and coefficient of variation `cv`, from the Student `t` quantile at
/// `nu = 39`.
pub const fn t_cap(reference: f64, cv: f64) -> f64 {
    reference * (1.0 + T_QUANTILE * cv)
}

/// Deterministic seed for suite offset `offset`, using the same mix as
/// [`seed`].
pub fn seed_with_offset(base: u64, offset: u64) -> u64 {
    base.wrapping_add(offset.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

/// Result of an empirical offset sweep over several capped quantities.
pub struct SweepReport {
    /// Number of seed offsets that were drawn.
    pub offsets: u64,
    /// Failures of each check: a value that is not finite, not positive, or
    /// above its cap.
    pub failures: Vec<u64>,
    /// Largest `value / cap` seen for each check.
    pub worst_ratio: Vec<f64>,
}

/// Runs `trial` at offsets `0..offsets` and tallies each check against its
/// cap.
///
/// `trial` maps one offset to `(value, cap)` pairs, one per check, in a fixed
/// order of length `checks`. Offsets are spread over worker threads. Every
/// trial is seeded from its offset alone, so the report does not depend on
/// the worker count or the scheduling.
pub fn offset_sweep<F>(offsets: u64, checks: usize, trial: F) -> SweepReport
where
    F: Fn(u64) -> Vec<(f64, f64)> + Sync,
{
    let workers = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .clamp(1, 8);
    let trial = &trial;
    let partials = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for worker in 0..workers {
            handles.push(scope.spawn(move || {
                let mut failures = vec![0_u64; checks];
                let mut worst = vec![0.0_f64; checks];
                let mut offset = worker as u64;
                while offset < offsets {
                    let outcomes = trial(offset);
                    assert_eq!(
                        outcomes.len(),
                        checks,
                        "sweep trial must return every check"
                    );
                    for (index, (value, cap)) in outcomes.into_iter().enumerate() {
                        if !(value.is_finite() && value > 0.0 && value <= cap) {
                            failures[index] += 1;
                        }
                        worst[index] = worst[index].max(value / cap);
                    }
                    offset += workers as u64;
                }
                (failures, worst)
            }));
        }
        handles
            .into_iter()
            .map(|handle| handle.join().expect("sweep thread"))
            .collect::<Vec<_>>()
    });
    let mut failures = vec![0_u64; checks];
    let mut worst_ratio = vec![0.0_f64; checks];
    for (local_failures, local_worst) in partials {
        for index in 0..checks {
            failures[index] += local_failures[index];
            worst_ratio[index] = worst_ratio[index].max(local_worst[index]);
        }
    }
    SweepReport {
        offsets,
        failures,
        worst_ratio,
    }
}

/// Asserts that every check of a sweep failed at a rate within
/// [`ASSERTION_ALPHA`], the budget of one statistical assertion.
///
/// With fewer than `1 / ASSERTION_ALPHA` offsets a single failure already
/// exceeds the budget, so the sweep accepts no failures.
pub fn assert_sweep_within_budget(name: &str, labels: &[&str], report: &SweepReport) {
    assert_eq!(labels.len(), report.failures.len(), "one label per check");
    for (index, label) in labels.iter().enumerate() {
        let failures = report.failures[index];
        let rate = (failures as f64) / (report.offsets as f64);
        eprintln!(
            "{name} sweep {label}: failures={failures} seeds={} rate={rate:.6e} budget={ASSERTION_ALPHA:.6e} worst_ratio={:.6}",
            report.offsets, report.worst_ratio[index]
        );
        assert!(
            rate <= ASSERTION_ALPHA,
            "{name} sweep {label}: empirical failure rate {rate:.6e} exceeds {ASSERTION_ALPHA:.6e}"
        );
    }
}

/// SplitMix64 finalizer, a bijection on `u64` with full avalanche.
fn splitmix64(value: u64) -> u64 {
    let mut z = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Seed of the random stream labelled `stream` that belongs to `base`.
///
/// Distinct labels give seeds that are unrelated at the scale of PAFI's
/// per-plane and per-round increments, so profiles drawn from them are
/// independent. Using one seed for two profiles instead makes them share the
/// noise of every plane they have in common.
pub fn independent_seed(base: u64, stream: u64) -> u64 {
    splitmix64(splitmix64(base) ^ splitmix64(stream.wrapping_add(1)))
}

/// Deterministic seed for suite offset `PAFI_SEED_OFFSET` (default zero).
pub fn seed(base: u64) -> u64 {
    let offset = match std::env::var("PAFI_SEED_OFFSET") {
        Ok(text) => text
            .parse::<u64>()
            .expect("PAFI_SEED_OFFSET must be an unsigned integer"),
        Err(_) => 0,
    };
    seed_with_offset(base, offset)
}

/// Batch means settings with exactly [`BLOCKS_PER_PLANE`] blocks per plane.
pub fn block_stats(samples_per_plane: usize) -> StatsConfig {
    assert_eq!(
        samples_per_plane % BLOCKS_PER_PLANE,
        0,
        "samples per plane must be a multiple of {BLOCKS_PER_PLANE}"
    );
    StatsConfig::default().with_batch_size(Some(samples_per_plane / BLOCKS_PER_PLANE))
}

/// One statistical assertion: `|estimate - target| <= Z_CRIT * standard_error`.
pub fn assert_coverage(label: &str, estimate: f64, standard_error: f64, target: f64) {
    assert!(
        standard_error.is_finite() && standard_error > 0.0,
        "{label}: standard error {standard_error:.8e} must be finite and positive"
    );
    let z = (estimate - target).abs() / standard_error;
    eprintln!(
        "{label}: estimate={estimate:.8e} se={standard_error:.8e} target={target:.8e} z={z:.4}"
    );
    assert!(
        z <= Z_CRIT,
        "{label}: {estimate:.8e} ± {standard_error:.8e} versus {target:.8e} gives z={z:.4} > {Z_CRIT:.4}"
    );
}

/// One statistical assertion: `value <= cap` for a random standard error or
/// standard error ratio.
pub fn assert_below(label: &str, value: f64, cap: f64) {
    eprintln!("{label}: value={value:.8e} cap={cap:.8e}");
    assert!(
        value.is_finite() && value > 0.0 && value <= cap,
        "{label}: {value:.8e} must lie in (0, {cap:.8e}]"
    );
}

/// Rejects a value that must exceed `cap` when a known regression is injected.
pub fn assert_above(label: &str, value: f64, cap: f64) {
    eprintln!("{label}: value={value:.8e} cap={cap:.8e}");
    assert!(
        value.is_finite() && value > cap,
        "{label}: regressed value {value:.8e} must exceed cap {cap:.8e}"
    );
}

/// One statistical assertion: `dividing_index == expected` on a sampled profile.
pub fn assert_dividing_plane(label: &str, dividing_index: usize, expected: usize) {
    eprintln!("{label}: dividing_index={dividing_index} expected={expected}");
    assert_eq!(
        dividing_index, expected,
        "{label}: dividing plane {dividing_index} must equal {expected}"
    );
}

/// Curved harmonic profile whose reported barrier standard error this module
/// bounds. These are the settings of `curved_profile` in `pafi_native.rs`.
const CURVED_PLANES: usize = 9;
const CURVED_SAMPLES: usize = 54_000;
const CURVED_TIME_STEP: f64 = 0.01;
const CURVED_RADIUS: f64 = 1.2;
const CURVED_AX: f64 = 3.0;
const CURVED_AY: f64 = 1.0;
const CURVED_THERMAL: f64 = 0.2;
const CURVED_GAMMA: f64 = 1.5;
const CURVED_MASSES: [f64; 2] = [1.0, 4.0];

struct SeModel {
    /// Hermite integral of the exact mean force, the estimator's expectation.
    mean_barrier: f64,
    /// Absolute Hermite-versus-trapezoid gap `tau_0` of the exact mean force.
    gap: f64,
    /// Standard deviation of that gap under the plane-mean noise.
    gap_sigma: f64,
    /// Coefficients `c_i = w_i^2 sigma_i^2` of the weighted chi-square `S^2`.
    coefficients: Vec<f64>,
}

struct CurvedCap {
    cap: f64,
    tail: f64,
}

/// Upper cap on the curved barrier's reported standard error.
///
/// The reported error at the last plane is `hypot(S, T)`. `S^2` propagates
/// the plane batch-means variances through the Hermite weights `w_i`, and
/// `T` is the absolute gap between the Hermite and trapezoid integrals of
/// the measured plane means.
///
/// On this path the potential is harmonic and each hyperplane leaves one
/// Gaussian coordinate. The projected force is a quadratic of that
/// coordinate, so its variance is a closed combination of the Gaussian
/// moments. The coordinate is an underdamped Langevin mode: the thermostat
/// multiplies every Cartesian velocity by `exp(-gamma dt)`, which multiplies
/// the tangential velocity by the same factor, and the squared frequency is
/// the curvature of that coordinate. The position autocorrelation of that
/// mode is the standard underdamped (or overdamped) Green-Kubo function, and
/// the force autocorrelation follows by Wick's theorem. Summing it across
/// one batch gives `Var(batch mean)`; dividing by the batch count gives
/// `sigma_i^2`, the expected squared batch-means error. A block is many
/// correlation times, so the forty batch means are independent and
/// `nu sigma_hat_i^2 / sigma_i^2` is chi-square with `nu = 39` degrees of
/// freedom. Planes use independent seeds, so
///
/// `S^2 = sum_i c_i Z_i`, `c_i = w_i^2 sigma_i^2`, `nu Z_i ~ chi^2_nu`,
/// `T = |tau_0 + eta|`, `eta ~ Normal(0, sigma_tau^2)`,
///
/// with `tau_0` the gap of the exact mean force and
/// `sigma_tau^2 = sum_i delta_i^2 sigma_i^2`. The Hermite-minus-trapezoid
/// weights are `delta_i`. `S^2` is independent of `eta`.
///
/// The moment generating function of `c Z` is `(1 - 2 t c / nu)^(-nu / 2)`.
/// Markov's inequality at the `t` that minimises it is an upper bound on
/// `P(S^2 > q)`. The Gaussian Mills ratio `phi(z) / z` is an upper bound on
/// the normal tail, and therefore on `P(T > t)` after splitting the absolute
/// value into its two sides. One fifth of `alpha / m` is spent on the
/// Chernoff bound and four fifths on the Mills bound. If both hold,
/// `hypot(S, T)` cannot exceed `hypot(sqrt(q), t)`, so
///
/// `P(SE > curved_barrier_se_cap()) <= alpha / m`.
///
/// Scaling a measured sampling mean by `sqrt(chi^2_nu / nu)` bounds `S`
/// alone. The reported error also carries the quadrature gap, which on this
/// path is the same order as `S`, so that product is not the tail of `SE`.
pub fn curved_barrier_se_cap() -> f64 {
    curved_cap().cap
}

/// Chernoff-plus-Mills upper bound on the probability that the curved
/// barrier standard error exceeds [`curved_barrier_se_cap`].
pub fn curved_barrier_se_tail() -> f64 {
    curved_cap().tail
}

/// Widening-well profile whose reported barrier standard error this module
/// bounds. Arc length is the unit-mass `x` distance from `-1` to `0`.
const WIDENING_PLANES: usize = 21;
const WIDENING_THERMAL: f64 = 0.2;
const WIDENING_TIME_STEP: f64 = 0.05;
const WIDENING_MASS_X: f64 = 1.0;

/// Upper cap on the widening well's reported barrier standard error.
///
/// The straight path has no normal curvature, so the projected force on the
/// plane at `x` is the Cartesian force. Only the transverse coordinate `y`
/// fluctuates: it is an underdamped Langevin oscillator of stiffness
/// `1.5 + 0.5 x` and mass `mass_y`, and the force fluctuation is
/// `y^2 / 4`. The same Green-Kubo batch variance, Hermite weight propagation
/// and Chernoff-plus-Mills union as [`curved_barrier_se_cap`] then give
///
/// `P(SE > widening_barrier_se_cap(...)) <= alpha / m`.
///
/// The Hermite-versus-trapezoid gap of this mean force is about `2.5e-3`,
/// larger than the sampling piece, so a chi-square factor times the sampling
/// error alone sits below the reported standard error.
pub fn widening_barrier_se_cap(samples: usize, mass_y: f64, gamma: f64) -> f64 {
    widening_barrier_se_cap_at(samples, mass_y, gamma, WIDENING_THERMAL)
}

/// [`widening_barrier_se_cap`] at thermal energy `thermal`.
///
/// The transverse variance is `kT / k(x)` and the mean force carries
/// `kT / (4 k(x))`, so a colder well has a smaller gap and a smaller
/// sampling piece. The tail bound is otherwise the one documented on
/// [`widening_barrier_se_cap`].
pub fn widening_barrier_se_cap_at(samples: usize, mass_y: f64, gamma: f64, thermal: f64) -> f64 {
    reported_se_bound(
        &widening_se_model(samples, mass_y, gamma, thermal),
        ASSERTION_ALPHA,
    )
    .cap
}

/// Upper cap on the widening well's normalized-rate standard error.
///
/// The reported rate error is `rate * barrier_error / kT`. The barrier
/// standard error is bounded as in [`widening_barrier_se_cap`], on four
/// fifths of `alpha / m`. The rate prefactor is the closed Gaussian rate at
/// the Hermite barrier, inflated by a one sided normal tail of the Hermite
/// estimator on the remaining fifth: the estimator, not the reported standard
/// error, is what enters the exponential, and only a low barrier raises the
/// rate.
pub fn widening_rate_se_cap(samples: usize, mass_y: f64, gamma: f64) -> f64 {
    let model = widening_se_model(samples, mass_y, gamma, WIDENING_THERMAL);
    let rate_share = 0.2;
    let barrier_budget = (1.0 - rate_share) * ASSERTION_ALPHA;
    let barrier_cap = reported_se_bound(&model, barrier_budget).cap;
    let sigma = model.coefficients.iter().sum::<f64>().sqrt();
    let z = smallest_one_sided_normal_quantile(rate_share * ASSERTION_ALPHA);
    let omega = 8.0_f64.sqrt();
    let rate = omega / TAU * (-model.mean_barrier / WIDENING_THERMAL).exp();
    let rate_high = rate * (z * sigma / WIDENING_THERMAL).exp();
    rate_high * barrier_cap / WIDENING_THERMAL
}

/// Upper cap on the ratio of two widening reported standard errors.
///
/// The fine error is bounded on half of `alpha / m`. The coarse error is at
/// least the quadrature gap, whose lower tail takes the other half. The
/// ratio of those two bounds is an upper bound on the reported-error ratio.
/// The gap does not shrink with the sample count, so this ratio is not the
/// sampling-error ratio `sqrt(N_coarse / N_fine)`.
pub fn widening_se_ratio_cap(
    coarse_samples: usize,
    fine_samples: usize,
    mass_y: f64,
    gamma: f64,
) -> f64 {
    let half = 0.5 * ASSERTION_ALPHA;
    let coarse = widening_se_model(coarse_samples, mass_y, gamma, WIDENING_THERMAL);
    let fine = widening_se_model(fine_samples, mass_y, gamma, WIDENING_THERMAL);
    let fine_cap = reported_se_bound(&fine, half).cap;
    let floor = largest_gap_floor(coarse.gap, coarse.gap_sigma, half);
    fine_cap / floor
}

/// Upper cap on the curved end-plane increment disagreement.
///
/// The short path ends at `theta = 0`. The longer path continues one image
/// spacing, so the same image is an interior plane and the natural-spline
/// curvature changes on every plane. Each of the eight increment differences
/// is normal with the mean of those two Hermite integrals. The profiles are
/// separate integrations, so the variances add. A union of Mills tails bounds
/// the maximum absolute difference by `alpha / m`.
pub fn end_plane_disagreement_cap() -> f64 {
    let short = curved_plane_table(CURVED_PLANES, 0.0);
    let spacing = -PI * 0.5 / ((CURVED_PLANES - 1) as f64);
    let long = curved_plane_table(CURVED_PLANES + 1, spacing);
    let short_rows = cumulative_weight_rows(&short.arcs);
    let long_rows = cumulative_weight_rows(&long.arcs);
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    let mut guard = 0;
    while end_plane_union(&short, &long, &short_rows, &long_rows, hi) > ASSERTION_ALPHA {
        hi *= 2.0;
        guard += 1;
        assert!(guard < 40, "end-plane union bound did not fall");
    }
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if end_plane_union(&short, &long, &short_rows, &long_rows, mid) > ASSERTION_ALPHA {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi
}

fn curved_cap() -> CurvedCap {
    reported_se_bound(&curved_se_model(), ASSERTION_ALPHA)
}

/// Cap on `hypot(S, T)` whose Chernoff-plus-Mills tail is at most `budget`.
///
/// One fifth of `budget` bounds `S^2` and four fifths bound `T`.
fn reported_se_bound(model: &SeModel, budget: f64) -> CurvedCap {
    let nu = DEGREES_OF_FREEDOM as f64;
    let sampling_level = SAMPLING_TAIL_SHARE * budget;
    let gap_level = GAP_TAIL_SHARE * budget;
    let sampling_quantile = smallest_chernoff_quantile(&model.coefficients, nu, sampling_level);
    let gap_quantile = smallest_gap_quantile(model.gap, model.gap_sigma, gap_level);
    let sampling_tail = chernoff_tail(&model.coefficients, nu, sampling_quantile);
    let gap_tail = gap_survival(model.gap, model.gap_sigma, gap_quantile);
    CurvedCap {
        cap: (sampling_quantile + gap_quantile * gap_quantile).sqrt(),
        tail: sampling_tail + gap_tail,
    }
}

fn curved_se_model() -> SeModel {
    let (knots, path) = curved_knots_and_path();
    let second = spline_second(&knots, &path);
    let n = CURVED_PLANES;
    let total = knots[n - 1];
    let mut arcs = Vec::with_capacity(n);
    let mut means = Vec::with_capacity(n);
    let mut mean_variances = Vec::with_capacity(n);
    let batches = BLOCKS_PER_PLANE;
    let batch_len = CURVED_SAMPLES / batches;
    for k in 0..n {
        let arc = total * (k as f64) / ((n - 1) as f64);
        arcs.push(arc);
        let (anchor, tangent, curvature) = spline_evaluate(&knots, &path, &second, arc);
        let (mean, mean_variance) = plane_mean_and_variance(anchor, tangent, curvature, batch_len);
        means.push(mean);
        mean_variances.push(mean_variance / (batches as f64));
    }
    assemble_se_model(&arcs, &means, &mean_variances)
}

fn assemble_se_model(arcs: &[f64], means: &[f64], mean_variances: &[f64]) -> SeModel {
    let hermite = hermite_weights(arcs);
    let trapezoid = trapezoid_weights(arcs);
    let mut mean_barrier = 0.0_f64;
    let mut gap = 0.0_f64;
    let mut gap_variance = 0.0_f64;
    let mut coefficients = Vec::with_capacity(arcs.len());
    for i in 0..arcs.len() {
        mean_barrier += hermite[i] * means[i];
        let delta = hermite[i] - trapezoid[i];
        gap += delta * means[i];
        gap_variance += delta * delta * mean_variances[i];
        coefficients.push(hermite[i] * hermite[i] * mean_variances[i]);
    }
    SeModel {
        mean_barrier,
        gap: gap.abs(),
        gap_sigma: gap_variance.sqrt(),
        coefficients,
    }
}

fn widening_se_model(samples: usize, mass_y: f64, gamma: f64, thermal: f64) -> SeModel {
    let n = WIDENING_PLANES;
    let batches = BLOCKS_PER_PLANE;
    assert_eq!(
        samples % batches,
        0,
        "widening samples must fill the batches"
    );
    let batch_len = samples / batches;
    let denominator = (n - 1) as f64;
    let mut arcs = Vec::with_capacity(n);
    let mut means = Vec::with_capacity(n);
    let mut mean_variances = Vec::with_capacity(n);
    for k in 0..n {
        // `m_x = 1`, so arc length is the `x` distance and the projected
        // force is the Cartesian `x` force. The only random piece is `y^2 / 4`.
        let arc = (k as f64) / denominator * WIDENING_MASS_X.sqrt();
        let x = -1.0 + arc;
        let stiffness = 1.5 + 0.5 * x;
        let deterministic = -4.0 * x * (1.0 - x * x);
        let variance_y = thermal / stiffness;
        let omega_squared = stiffness / mass_y;
        let batch_variance = batch_mean_variance(
            0.0,
            0.25,
            variance_y,
            omega_squared,
            batch_len,
            gamma,
            WIDENING_TIME_STEP,
        );
        arcs.push(arc);
        means.push(deterministic + 0.25 * variance_y);
        mean_variances.push(batch_variance / (batches as f64));
    }
    assemble_se_model(&arcs, &means, &mean_variances)
}

struct PlaneTable {
    arcs: Vec<f64>,
    means: Vec<f64>,
    variances: Vec<f64>,
}

fn curved_plane_table(images: usize, theta_end: f64) -> PlaneTable {
    let (knots, path) = circular_knots(images, theta_end);
    let second = spline_second(&knots, &path);
    let total = knots[images - 1];
    let batches = BLOCKS_PER_PLANE;
    let batch_len = CURVED_SAMPLES / batches;
    let mut arcs = Vec::with_capacity(images);
    let mut means = Vec::with_capacity(images);
    let mut variances = Vec::with_capacity(images);
    for k in 0..images {
        let arc = total * (k as f64) / ((images - 1) as f64);
        arcs.push(arc);
        let (anchor, tangent, curvature) = spline_evaluate(&knots, &path, &second, arc);
        let (mean, batch_variance) = plane_mean_and_variance(anchor, tangent, curvature, batch_len);
        means.push(mean);
        variances.push(batch_variance / (batches as f64));
    }
    PlaneTable {
        arcs,
        means,
        variances,
    }
}

fn curved_knots_and_path() -> (Vec<f64>, Vec<[f64; 2]>) {
    circular_knots(CURVED_PLANES, 0.0)
}

fn circular_knots(images: usize, theta_end: f64) -> (Vec<f64>, Vec<[f64; 2]>) {
    let mut path = Vec::with_capacity(images);
    let mut knots = Vec::with_capacity(images);
    knots.push(0.0);
    for i in 0..images {
        let fraction = (i as f64) / ((images - 1) as f64);
        let theta = PI * 0.5 + fraction * (theta_end - PI * 0.5);
        path.push([
            CURVED_RADIUS * theta.cos(),
            0.5 * CURVED_RADIUS * theta.sin(),
        ]);
        if i > 0 {
            let dx = path[i][0] - path[i - 1][0];
            let dy = path[i][1] - path[i - 1][1];
            let chord = (CURVED_MASSES[0] * dx * dx + CURVED_MASSES[1] * dy * dy).sqrt();
            let previous = knots[i - 1];
            knots.push(previous + chord);
        }
    }
    (knots, path)
}

fn spline_second(knots: &[f64], path: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let n = path.len();
    let interior = n - 2;
    let mut lower = vec![0.0_f64; interior];
    let mut diagonal = vec![0.0_f64; interior];
    let mut upper = vec![0.0_f64; interior];
    let mut rhs = vec![[0.0_f64; 2]; interior];
    for k in 0..interior {
        let i = k + 1;
        let left = knots[i] - knots[i - 1];
        let right = knots[i + 1] - knots[i];
        lower[k] = left;
        diagonal[k] = 2.0 * (left + right);
        upper[k] = right;
        for axis in 0..2 {
            rhs[k][axis] = 6.0
                * ((path[i + 1][axis] - path[i][axis]) / right
                    - (path[i][axis] - path[i - 1][axis]) / left);
        }
    }
    let first = knots[1] - knots[0];
    let second_width = knots[2] - knots[1];
    diagonal[0] += first * (first + second_width) / second_width;
    upper[0] -= first * first / second_width;
    let penultimate = knots[n - 2] - knots[n - 3];
    let last = knots[n - 1] - knots[n - 2];
    lower[interior - 1] -= last * last / penultimate;
    diagonal[interior - 1] += last * (penultimate + last) / penultimate;
    for k in 1..interior {
        let factor = lower[k] / diagonal[k - 1];
        diagonal[k] -= factor * upper[k - 1];
        let previous = rhs[k - 1];
        for (value, below) in rhs[k].iter_mut().zip(previous) {
            *value -= factor * below;
        }
    }
    let mut second = vec![[0.0_f64; 2]; n];
    for k in (0..interior).rev() {
        let i = k + 1;
        for axis in 0..2 {
            let next = if k + 1 < interior {
                upper[k] * second[i + 1][axis]
            } else {
                0.0
            };
            second[i][axis] = (rhs[k][axis] - next) / diagonal[k];
        }
    }
    let (near_start, next_start) = (second[1], second[2]);
    second[0] = std::array::from_fn(|axis| {
        ((first + second_width) * near_start[axis] - first * next_start[axis]) / second_width
    });
    let (near_end, next_end) = (second[n - 2], second[n - 3]);
    second[n - 1] = std::array::from_fn(|axis| {
        ((penultimate + last) * near_end[axis] - last * next_end[axis]) / penultimate
    });
    second
}

fn spline_evaluate(
    knots: &[f64],
    path: &[[f64; 2]],
    second: &[[f64; 2]],
    s: f64,
) -> ([f64; 2], [f64; 2], [f64; 2]) {
    let last = knots.len() - 1;
    let mut segment = last - 1;
    for i in 0..last {
        if s <= knots[i + 1] {
            segment = i;
            break;
        }
    }
    let left = knots[segment];
    let right = knots[segment + 1];
    let width = right - left;
    let a = (right - s) / width;
    let b = (s - left) / width;
    let mut position = [0.0_f64; 2];
    let mut tangent = [0.0_f64; 2];
    let mut curvature = [0.0_f64; 2];
    for axis in 0..2 {
        let y0 = path[segment][axis];
        let y1 = path[segment + 1][axis];
        let m0 = second[segment][axis];
        let m1 = second[segment + 1][axis];
        position[axis] =
            a * y0 + b * y1 + ((a * a * a - a) * m0 + (b * b * b - b) * m1) * width * width / 6.0;
        tangent[axis] = (y1 - y0) / width
            + width * ((-3.0 * a * a + 1.0) * m0 + (3.0 * b * b - 1.0) * m1) / 6.0;
        curvature[axis] = a * m0 + b * m1;
    }
    (position, tangent, curvature)
}

/// Exact mean and `Var(batch mean)` of the projected force on one plane.
fn plane_mean_and_variance(
    anchor: [f64; 2],
    tangent: [f64; 2],
    curvature: [f64; 2],
    batch_len: usize,
) -> (f64, f64) {
    let mut normal_raw = [0.0_f64; 2];
    let mut curvature_raw = [0.0_f64; 2];
    for axis in 0..2 {
        normal_raw[axis] = CURVED_MASSES[axis] * tangent[axis];
        curvature_raw[axis] = CURVED_MASSES[axis] * curvature[axis];
    }
    let normal_norm = (normal_raw[0] * normal_raw[0] + normal_raw[1] * normal_raw[1]).sqrt();
    let normal = [normal_raw[0] / normal_norm, normal_raw[1] / normal_norm];
    let projection = normal[0] * curvature_raw[0] + normal[1] * curvature_raw[1];
    let dn_ds = [
        (curvature_raw[0] - normal[0] * projection) / normal_norm,
        (curvature_raw[1] - normal[1] * projection) / normal_norm,
    ];
    let dc_ds = normal[0] * tangent[0] + normal[1] * tangent[1];
    let mut tangent_direction = [normal[1], -normal[0]];
    let tangent_norm = (tangent_direction[0] * tangent_direction[0]
        + tangent_direction[1] * tangent_direction[1])
        .sqrt();
    tangent_direction[0] /= tangent_norm;
    tangent_direction[1] /= tangent_norm;

    let grad = |point: [f64; 2]| [CURVED_AX * point[0], 4.0 * CURVED_AY * point[1]];
    let metric = |vector: [f64; 2]| {
        vector[0] * vector[0] / CURVED_MASSES[0] + vector[1] * vector[1] / CURVED_MASSES[1]
    };
    let normal_metric = metric(normal);
    let force_at = |xi: f64| {
        let point = [
            anchor[0] + xi * tangent_direction[0],
            anchor[1] + xi * tangent_direction[1],
        ];
        let gradient = grad(point);
        let lambda = (normal[0] * gradient[0] / CURVED_MASSES[0]
            + normal[1] * gradient[1] / CURVED_MASSES[1])
            / normal_metric;
        let jacobian = xi * (tangent_direction[0] * dn_ds[0] + tangent_direction[1] * dn_ds[1]);
        lambda * (dc_ds - jacobian)
    };
    let stiffness = {
        let hu = grad(tangent_direction);
        tangent_direction[0] * hu[0] + tangent_direction[1] * hu[1]
    };
    let slope = tangent_direction[0] * grad(anchor)[0] + tangent_direction[1] * grad(anchor)[1];
    let mean_xi = -slope / stiffness;
    let variance_xi = CURVED_THERMAL / stiffness;
    let at_zero = force_at(0.0);
    let at_plus = force_at(1.0);
    let at_minus = force_at(-1.0);
    let quadratic = 0.5 * (at_plus + at_minus) - at_zero;
    let linear = 0.5 * (at_plus - at_minus);
    let mean = at_zero + linear * mean_xi + quadratic * (mean_xi * mean_xi + variance_xi);
    let centered_slope = linear + 2.0 * quadratic * mean_xi;
    let accel = |xi: f64| {
        let point = [
            anchor[0] + xi * tangent_direction[0],
            anchor[1] + xi * tangent_direction[1],
        ];
        let gradient = grad(point);
        let physical = [-gradient[0], -gradient[1]];
        let lambda = -(normal[0] * physical[0] / CURVED_MASSES[0]
            + normal[1] * physical[1] / CURVED_MASSES[1])
            / normal_metric;
        let ax = (physical[0] + lambda * normal[0]) / CURVED_MASSES[0];
        let ay = (physical[1] + lambda * normal[1]) / CURVED_MASSES[1];
        tangent_direction[0] * ax + tangent_direction[1] * ay
    };
    let omega_squared = -(accel(1.0) - accel(0.0));
    let batch_variance = batch_mean_variance(
        centered_slope,
        quadratic,
        variance_xi,
        omega_squared,
        batch_len,
        CURVED_GAMMA,
        CURVED_TIME_STEP,
    );
    (mean, batch_variance)
}

fn batch_mean_variance(
    centered_slope: f64,
    quadratic: f64,
    variance_xi: f64,
    omega_squared: f64,
    batch_len: usize,
    gamma: f64,
    time_step: f64,
) -> f64 {
    let decay = 0.5 * gamma;
    let discriminant = decay * decay - omega_squared;
    let mut var_sum = 0.0_f64;
    for lag in 0..batch_len {
        let time = (lag as f64) * time_step;
        let correlation = if discriminant < 0.0 {
            let frequency = (-discriminant).sqrt();
            let shape = decay / frequency;
            (-decay * time).exp() * (time * frequency).cos()
                + shape * (-decay * time).exp() * (time * frequency).sin()
        } else {
            let split = discriminant.sqrt();
            let slow = -decay + split;
            let fast = -decay - split;
            (fast * (slow * time).exp() - slow * (fast * time).exp()) / (fast - slow)
        };
        let position = variance_xi * correlation;
        let covariance = centered_slope * centered_slope * position
            + 2.0 * quadratic * quadratic * position * position;
        let copies = if lag == 0 {
            batch_len as f64
        } else {
            2.0 * ((batch_len - lag) as f64)
        };
        var_sum += copies * covariance;
    }
    var_sum / ((batch_len * batch_len) as f64)
}

fn hermite_weights(arcs: &[f64]) -> Vec<f64> {
    let n = arcs.len();
    let mut weights = vec![0.0_f64; n];
    for i in 0..n {
        let mut gradient = vec![0.0_f64; n];
        gradient[i] = 1.0;
        let estimate = HermiteIntegral
            .cumulative(arcs, &gradient)
            .expect("Hermite weight");
        weights[i] = estimate.values[n - 1];
    }
    weights
}

fn trapezoid_weights(arcs: &[f64]) -> Vec<f64> {
    let mut weights = vec![0.0_f64; arcs.len()];
    for k in 0..arcs.len() - 1 {
        let width = arcs[k + 1] - arcs[k];
        weights[k] += 0.5 * width;
        weights[k + 1] += 0.5 * width;
    }
    weights
}

fn cumulative_weight_rows(arcs: &[f64]) -> Vec<Vec<f64>> {
    let n = arcs.len();
    let mut rows = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        let mut gradient = vec![0.0_f64; n];
        gradient[i] = 1.0;
        let estimate = HermiteIntegral
            .cumulative(arcs, &gradient)
            .expect("Hermite weight row");
        for (row, value) in rows.iter_mut().zip(&estimate.values) {
            row[i] = *value;
        }
    }
    rows
}

fn node_value(rows: &[Vec<f64>], means: &[f64], node: usize) -> f64 {
    rows[node]
        .iter()
        .zip(means)
        .map(|(weight, mean)| weight * mean)
        .sum()
}

fn increment_variance(rows: &[Vec<f64>], variances: &[f64], node: usize) -> f64 {
    let mut total = 0.0_f64;
    for (i, variance) in variances.iter().enumerate() {
        let weight = rows[node][i] - rows[node - 1][i];
        total += weight * weight * variance;
    }
    total
}

/// Predicted mean and variance of the difference between the short and the long
/// curved profile at every increment `1..CURVED_PLANES`, for independently
/// seeded profiles.
///
/// The variance is the sum of the two increment variances, the quantity
/// [`end_plane_disagreement_cap`] bounds. It is the variance of the difference
/// only when the two profiles use independent random streams: draws that share
/// per-plane seeds are positively correlated and have a much smaller
/// variance.
pub fn end_plane_difference_prediction() -> Vec<(f64, f64)> {
    let short = curved_plane_table(CURVED_PLANES, 0.0);
    let spacing = -PI * 0.5 / ((CURVED_PLANES - 1) as f64);
    let long = curved_plane_table(CURVED_PLANES + 1, spacing);
    let short_rows = cumulative_weight_rows(&short.arcs);
    let long_rows = cumulative_weight_rows(&long.arcs);
    (1..CURVED_PLANES)
        .map(|node| {
            let short_increment = node_value(&short_rows, &short.means, node)
                - node_value(&short_rows, &short.means, node - 1);
            let long_increment = node_value(&long_rows, &long.means, node)
                - node_value(&long_rows, &long.means, node - 1);
            let variance = increment_variance(&short_rows, &short.variances, node)
                + increment_variance(&long_rows, &long.variances, node);
            (short_increment - long_increment, variance)
        })
        .collect()
}

fn end_plane_union(
    short: &PlaneTable,
    long: &PlaneTable,
    short_rows: &[Vec<f64>],
    long_rows: &[Vec<f64>],
    level: f64,
) -> f64 {
    let mut total = 0.0_f64;
    for node in 1..CURVED_PLANES {
        let short_increment = node_value(short_rows, &short.means, node)
            - node_value(short_rows, &short.means, node - 1);
        let long_increment =
            node_value(long_rows, &long.means, node) - node_value(long_rows, &long.means, node - 1);
        let mean = short_increment - long_increment;
        let variance = increment_variance(short_rows, &short.variances, node)
            + increment_variance(long_rows, &long.variances, node);
        total += gap_survival(mean.abs(), variance.sqrt(), level);
    }
    total
}

/// Smallest `z` whose one-sided Mills bound `phi(z) / z` is at most `target`.
///
/// The rate prefactor `exp(-B / kT)` grows when the Hermite barrier estimate
/// `B_hat ~ Normal(B, sigma^2)` falls, so only the lower tail of `B_hat`
/// matters. With `P(B_hat < B - z sigma) = Phi(-z)` and the Mills inequality
/// `Phi(-z) <= phi(z) / z` for `z > 0`, the event `B_hat < B - z sigma` has
/// probability at most `phi(z) / z`. The upper tail never raises the
/// prefactor, so no factor of two enters. Solving `2 phi(z) / z <= target`
/// instead spends half of the budget on a tail the cap does not need, raises
/// `z`, and makes the cap looser than its failure budget requires.
fn smallest_one_sided_normal_quantile(target: f64) -> f64 {
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    let mut guard = 0;
    while mills_tail(hi) > target {
        hi += 1.0;
        guard += 1;
        assert!(guard < 40, "normal Mills bound did not fall");
    }
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if mills_tail(mid) > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi
}

/// Largest `floor <= center` with `P(|center + eta| < floor) <= target`.
///
/// For `floor <= center` that probability is at most the one-sided Mills
/// bound `phi(z) / z` at `z = (center - floor) / sigma`.
fn largest_gap_floor(center: f64, sigma: f64, target: f64) -> f64 {
    if sigma <= 0.0 {
        return center;
    }
    assert!(
        mills_tail(center / sigma) <= target,
        "quadrature gap is not separated from zero at this level"
    );
    let mut lo = 0.0_f64;
    let mut hi = center;
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if mills_tail((center - mid) / sigma) > target {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    lo
}

fn chernoff_tail(weights: &[f64], nu: f64, quantile: f64) -> f64 {
    let max_weight = weights.iter().copied().fold(0.0_f64, f64::max);
    if max_weight == 0.0 {
        return if quantile > 0.0 { 0.0 } else { 1.0 };
    }
    let max_t = 0.5 * nu / max_weight;
    let mut lo = 0.0_f64;
    let mut hi = max_t * (1.0 - 1.0e-9);
    for _ in 0..80 {
        let left = lo + (hi - lo) / 3.0;
        let right = hi - (hi - lo) / 3.0;
        if chernoff_exponent(weights, nu, quantile, left)
            < chernoff_exponent(weights, nu, quantile, right)
        {
            hi = right;
        } else {
            lo = left;
        }
    }
    chernoff_exponent(weights, nu, quantile, 0.5 * (lo + hi)).exp()
}

fn chernoff_exponent(weights: &[f64], nu: f64, quantile: f64, t: f64) -> f64 {
    let mut exponent = -t * quantile;
    for &weight in weights {
        if weight == 0.0 {
            continue;
        }
        let argument = 1.0 - 2.0 * t * weight / nu;
        if argument <= 0.0 {
            return f64::INFINITY;
        }
        exponent += -0.5 * nu * argument.ln();
    }
    exponent
}

fn smallest_chernoff_quantile(weights: &[f64], nu: f64, target: f64) -> f64 {
    let mean: f64 = weights.iter().sum();
    let mut lo = 0.0_f64;
    let mut hi = mean.max(1.0e-18);
    let mut guard = 0;
    while chernoff_tail(weights, nu, hi) > target {
        hi *= 2.0;
        guard += 1;
        assert!(guard < 80, "Chernoff bound did not reach the sampling tail");
    }
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if chernoff_tail(weights, nu, mid) > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi
}

fn mills_tail(z: f64) -> f64 {
    if z <= 0.0 {
        return 1.0;
    }
    let density = (-0.5 * z * z).exp() / TAU.sqrt();
    (density / z).min(1.0)
}

fn gap_survival(center: f64, sigma: f64, level: f64) -> f64 {
    if sigma <= 0.0 {
        return if level >= center { 0.0 } else { 1.0 };
    }
    mills_tail((level - center) / sigma) + mills_tail((level + center) / sigma)
}

fn smallest_gap_quantile(center: f64, sigma: f64, target: f64) -> f64 {
    let mut lo = 0.0_f64;
    let mut hi = center + 8.0 * sigma.max(1.0e-18);
    let mut guard = 0;
    while gap_survival(center, sigma, hi) > target {
        hi = center + (hi - center) * 1.5 + sigma;
        guard += 1;
        assert!(guard < 80, "Mills bound did not reach the gap tail");
    }
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if gap_survival(center, sigma, mid) > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi
}

#[test]
fn curved_standard_error_model_matches_the_langevin_gap() {
    let model = curved_se_model();
    let gap = 1.789_332_695_915_132_2e-3;
    let sampling: f64 = 1.132_230_265_816_751_4e-5;
    let gap_sigma = 6.617_110_711_598_744e-4;
    let sampling_model: f64 = model.coefficients.iter().sum();
    assert!(
        (model.gap - gap).abs() <= 1.0e-12 * gap,
        "gap {:.16e} versus {:.16e}",
        model.gap,
        gap
    );
    assert!(
        (sampling_model - sampling).abs() <= 1.0e-9 * sampling,
        "sampling variance {sampling_model:.16e} versus {sampling:.16e}"
    );
    assert!(
        (model.gap_sigma - gap_sigma).abs() <= 1.0e-9 * gap_sigma,
        "gap sigma {:.16e} versus {:.16e}",
        model.gap_sigma,
        gap_sigma
    );
    let tail = curved_barrier_se_tail();
    assert!(
        tail <= ASSERTION_ALPHA,
        "curved standard-error tail {tail:.6e} exceeds {ASSERTION_ALPHA:.6e}"
    );
}

#[test]
fn widening_gap_matches_the_hermite_trapezoid_difference() {
    let samples = 10_000;
    let model = widening_se_model(samples, 1.0, 1.5, WIDENING_THERMAL);
    let n = WIDENING_PLANES;
    let arcs: Vec<f64> = (0..n).map(|i| (i as f64) / ((n - 1) as f64)).collect();
    let gradients: Vec<f64> = arcs
        .iter()
        .map(|arc| {
            let x = -1.0 + *arc;
            -4.0 * x * (1.0 - x * x) + 0.25 * WIDENING_THERMAL / (1.5 + 0.5 * x)
        })
        .collect();
    let estimate = HermiteIntegral
        .cumulative(&arcs, &gradients)
        .expect("widening hermite");
    let gap = estimate.error[n - 1];
    let barrier = estimate.values[n - 1];
    assert!(
        (model.gap - gap).abs() <= 1.0e-12 * gap,
        "gap {:.16e} versus hermite {:.16e}",
        model.gap,
        gap
    );
    assert!(
        (model.mean_barrier - barrier).abs() <= 1.0e-12 * barrier,
        "barrier {:.16e} versus hermite {:.16e}",
        model.mean_barrier,
        barrier
    );
    let sampling = model.coefficients.iter().sum::<f64>().sqrt();
    let predicted = sampling.hypot(model.gap);
    let cap = widening_barrier_se_cap(samples, 1.0, 1.5);
    assert!(
        predicted < cap,
        "predicted standard error {predicted:.8e} is not under the cap {cap:.8e}"
    );
}

#[test]
fn one_sided_mills_quantile_matches_its_inversion() {
    // `phi(z) / z = alpha / (5 m)` solved with scipy.optimize.brentq, and the
    // exact `scipy.stats.norm.isf(alpha / (5 m))` that the Mills bound must
    // not undercut.
    let target = 0.2 * ASSERTION_ALPHA;
    let mills_root = 4.252_330_921_149;
    let exact = 4.241_306_705_633;
    let two_sided_root = 4.404_356_789_847;
    let z = smallest_one_sided_normal_quantile(target);
    assert!(
        (z - mills_root).abs() <= 1.0e-9,
        "one sided quantile {z:.12} versus {mills_root:.12}"
    );
    assert!(
        mills_tail(z) <= target,
        "Mills tail at the quantile exceeds the target"
    );
    assert!(
        z > exact && z < two_sided_root,
        "quantile {z:.12} must lie between the exact one sided {exact:.12} and the two sided {two_sided_root:.12}"
    );
}
