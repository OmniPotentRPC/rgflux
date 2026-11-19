//! Statistical budget and seed control shared by the PAFI integration tests.
//!
//! Family: each call to [`assert_coverage`] or [`assert_below`] in the two
//! PAFI test files is one statistical assertion. There are [`FAMILY_SIZE`]
//! (`m`) of them, checked by `statistical_family_size_matches_call_sites`.
//!
//! Budget: the suite false failure rate is [`SUITE_ALPHA`] (`alpha`, one in a
//! thousand). Bonferroni gives every assertion level `alpha / m`.
//!
//! Coverage: planes are seeded independently and each plane SE is block
//! averaged over [`BLOCKS_PER_PLANE`] blocks, each long compared with the
//! autocorrelation time. The barrier is linear in the plane means and the
//! rate to first order, so `(estimate - target) / SE` is Student `t` with at
//! least `BLOCKS_PER_PLANE - 1` degrees of freedom (Welch and Satterthwaite),
//! and the two sided quantile at `alpha / (2 m)` is [`T_QUANTILE`]. A
//! sampling spread of `s` times the mean SE scales that quantile by `s`, so
//! [`PIVOT_MARGIN`] must be at least `s`. The measured spread over mean SE
//! was at most `1.07` in 4096 seeds per configuration. The margin `1.25`
//! also covers the largest studentised deviation seen in any configuration,
//! `5.42 = 1.21 * T_QUANTILE`. Targets are the expectation of the estimator:
//! the trapezoid sum of the analytic mean force on the same planes. The mean
//! of the estimates differed from the target by at most `0.06` of the mean
//! SE (quadrature, spline and time step), within `0.02` of the mean SE of
//! sampling error.
//!
//! SE caps: per configuration the mean SE and its coefficient of variation
//! `cv` were measured over 4096 seeds, and the cap is
//! `SE * (1 + TAIL_EXCESS * z * cv)` with `z = Phi^-1(1 - alpha / m)`.
//! [`TAIL_EXCESS`] widens the normal tail: the largest value of an ensemble
//! was up to `1.6` times the normal prediction at its rank. No SE, SE ratio or
//! disagreement reached its cap.
//!
//! Seeds: `PAFI_SEED_OFFSET` shifts every seed by a multiple of a large odd
//! constant, so a sweep over offsets draws independent suites.

#![allow(dead_code)]

use rgflux::stats::StatsConfig;

/// Suite level false failure rate `alpha`.
pub const SUITE_ALPHA: f64 = 1.0 / 1_000.0;

/// Number of statistical assertions `m` in the PAFI test suite.
pub const FAMILY_SIZE: usize = 16;

/// Per assertion level `alpha / m`.
pub const ASSERTION_ALPHA: f64 = SUITE_ALPHA / (FAMILY_SIZE as f64);

/// Batch means blocks per plane, so the SE has at least `39` degrees of freedom.
pub const BLOCKS_PER_PLANE: usize = 40;

/// `t_{39, 1 - alpha / (2 m)}` with `alpha = 1/1000` and `m = 16`.
pub const T_QUANTILE: f64 = 4.486;

/// Margin over [`T_QUANTILE`] for the finite block variance deficit and the
/// heaviest measured studentised deviation.
pub const PIVOT_MARGIN: f64 = 1.25;

/// Two sided critical value for `|estimate - target| / SE`.
pub const Z_CRIT: f64 = PIVOT_MARGIN * T_QUANTILE;

/// Standard normal quantile `Phi^-1(1 - alpha / m)` for one sided caps.
pub const Z_ONE_SIDED: f64 = 3.836;

/// Widening of the normal tail for the measured right skew of the reported SE.
pub const TAIL_EXCESS: f64 = 2.0;

/// Upper cap on a reported SE with measured mean `reference` and coefficient
/// of variation `cv`, at level `alpha / m`.
pub const fn se_cap(reference: f64, cv: f64) -> f64 {
    reference * (1.0 + TAIL_EXCESS * Z_ONE_SIDED * cv)
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
