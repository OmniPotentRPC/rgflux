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
//! Coverage: planes are seeded independently and each plane SE is block
//! averaged over [`BLOCKS_PER_PLANE`] blocks, each long compared with the
//! autocorrelation time. In the batch means central limit theorem of
//! Schmeiser (1982) and Flegal, Haran, and Jones (2008), the block means on
//! each plane are asymptotically independent normal with `nu = BLOCKS_PER_PLANE - 1`
//! degrees of freedom. The barrier is a linear combination of the independent
//! plane means and the rate is linear to first order, so by the
//! Welch-Satterthwaite theorem (Welch, 1947; Satterthwaite, 1946) the ratio
//! `(estimate - target) / SE` follows a Student `t` distribution with
//! effective degrees of freedom at least `nu = 39` (achieved when a single
//! plane dominates the variance, and strictly larger when variance is spread
//! across planes). Because the Student `t` upper tail decreases monotonically
//! with degrees of freedom, the two sided quantile `t_{39, 1 - alpha / (2 m)}`
//! ([`T_QUANTILE`]) is an upper bound on the exact two sided critical value
//! ([`Z_CRIT`]). Targets are the expectation of the estimator: the trapezoid
//! sum of the analytic mean force on the same planes.
//!
//! SE caps: for a variance sum of independent normal block means with at least
//! `nu = BLOCKS_PER_PLANE - 1` effective degrees of freedom (Satterthwaite,
//! 1946), the squared ratio `(SE / sigma)^2` is bounded in the upper tail by
//! `chi^2_nu / nu` (Casella and Berger, 2002, Theorem 5.3.1). Therefore, at
//! one sided level `alpha / m`, a reported SE or the ratio of two independent
//! SEs is bounded above by `reference * sqrt(chi^2_{39, 1 - alpha / m} / 39)`
//! ([`CHI2_SE_FACTOR`], [`se_cap`]). This bound depends only on the degrees of
//! freedom `nu = 39` and the per assertion level `alpha / m`, with no fitted
//! excess factor.
//!
//! End plane disagreement cap: `end_plane_disagreement` is the magnitude of the
//! shift in the last free energy increment between the path and a path one
//! image longer, which share per plane seeds. Unlike a standard error, it is a
//! linear mean difference across `BLOCKS_PER_PLANE` blocks, so `(X - mu) / sd`
//! follows Student `t` with `nu = 39` degrees of freedom. At two sided level
//! `alpha / m`, [`t_cap`] bounds it by `reference * (1 + T_QUANTILE * cv)`
//! (`1.7645 * reference`, below `2 * reference` so a doubled estimator error
//! is rejected).
//!
//! Seeds: `PAFI_SEED_OFFSET` shifts every seed by a multiple of a large odd
//! constant, so a sweep over offsets draws independent suites.
//!
//! ## References
//!
//! - Bonferroni, C. E. (1936). *Teoria statistica delle classi e calcolo delle
//!   probabilità*. Pubblicazioni del R Istituto Superiore di Scienze
//!   Economiche e Commerciali di Firenze, 8, 3 to 62.
//! - Casella, G., and Berger, R. L. (2002). *Statistical Inference* (2nd ed.).
//!   Duxbury, Theorem 5.3.1.
//! - Flegal, J. M., Haran, M., and Jones, G. L. (2008). *Markov Chain Monte
//!   Carlo: Can We Trust the Third Significant Figure?* Statistical Science,
//!   23, 250 to 260. <https://doi.org/10.1214/08-STS257>
//! - Satterthwaite, F. E. (1946). *An Approximate Distribution of Estimates of
//!   Variance Components*. Biometrics Bulletin, 2, 110 to 114.
//!   <https://doi.org/10.2307/3002019>
//! - Schmeiser, B. (1982). *Batch Size Effects in the Analysis of Simulation
//!   Output*. Operations Research, 30, 556 to 568.
//!   <https://doi.org/10.1287/opre.30.3.556>
//! - Welch, B. L. (1947). *The Generalization of 'Student's' Problem when
//!   Several Different Population Variances are Involved*. Biometrika, 34,
//!   28 to 35. <https://doi.org/10.1093/biomet/34.1-2.28>

#![allow(dead_code)]

use rgflux::stats::StatsConfig;

/// Suite level false failure rate `alpha`.
pub const SUITE_ALPHA: f64 = 1.0 / 1_000.0;

/// Number of statistical assertions `m` in the PAFI test suite.
pub const FAMILY_SIZE: usize = 18;

/// Per assertion level `alpha / m` under the Bonferroni bound.
pub const ASSERTION_ALPHA: f64 = SUITE_ALPHA / (FAMILY_SIZE as f64);

/// Batch means blocks per plane, so the SE has at least `39` degrees of freedom.
pub const BLOCKS_PER_PLANE: usize = 40;

/// Degrees of freedom `nu = BLOCKS_PER_PLANE - 1` of a single plane batch means
/// variance estimator, and the Welch-Satterthwaite lower bound across planes.
pub const DEGREES_OF_FREEDOM: usize = BLOCKS_PER_PLANE - 1;

/// Two sided Student `t` quantile `t_{39, 1 - alpha / (2 m)}` with
/// `alpha = 1/1000` and `m = 18` (`1 - alpha / (2 m) = 1 - 1 / 36000`).
pub const T_QUANTILE: f64 = 4.523_716;

/// Two sided critical value for `|estimate - target| / SE`, equal to
/// [`T_QUANTILE`] directly from the Student `t` pivot with `39` degrees of
/// freedom.
pub const Z_CRIT: f64 = T_QUANTILE;

/// Upper chi-square quantile `chi^2_{39, 1 - alpha / m}` with `alpha = 1/1000`
/// and `m = 18` (`1 - alpha / m = 1 - 1 / 18000`).
pub const CHI2_QUANTILE: f64 = 82.727_831;

/// Upper bound factor `sqrt(chi^2_{39, 1 - alpha / m} / 39)` for the ratio of a
/// batch means standard error to its expectation at level `alpha / m`.
pub const CHI2_SE_FACTOR: f64 = 1.456_443;

/// Upper cap on a reported SE or SE ratio with expectation `reference` at one
/// sided level `alpha / m`, derived from `sqrt(chi^2_{39, 1 - alpha / m} / 39)`.
pub const fn se_cap(reference: f64) -> f64 {
    reference * CHI2_SE_FACTOR
}

/// Upper cap on a mean difference with expectation `reference` and coefficient
/// of variation `cv` over [`BLOCKS_PER_PLANE`] blocks, derived from the
/// Student `t` quantile [`T_QUANTILE`] at `nu = 39` degrees of freedom.
pub const fn t_cap(reference: f64, cv: f64) -> f64 {
    reference * (1.0 + T_QUANTILE * cv)
}

/// Deterministic seed for suite offset `PAFI_SEED_OFFSET` (default zero).
pub fn seed(base: u64) -> u64 {
    let offset = match std::env::var("PAFI_SEED_OFFSET") {
        Ok(text) => text
            .parse::<u64>()
            .expect("PAFI_SEED_OFFSET must be an unsigned integer"),
        Err(_) => 0,
    };
    base.wrapping_add(offset.wrapping_mul(0x9E37_79B9_7F4A_7C15))
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

/// One statistical assertion: `value <= cap` for a random SE or SE ratio.
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
