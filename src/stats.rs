//! Batch means, effective sample size (`n_eff`), and stationarity diagnostics.
//!
//! Correlated molecular-dynamics and Markov-chain observables underestimate
//! uncertainty when treated as independent draws. This module provides
//! non-overlapping batch means, Geyer's initial-monotone-sequence effective
//! sample size, and Geweke's two-window stationarity check.
//!
//! Note: implementation moves from rgsaddle for spline-propagated profile
//! covariance helpers; the core batch-means, `n_eff`, and stationarity
//! estimators used by `kappa::bennett_chandler` and engine validation
//! live here directly.
//!
//! ## Example
//!
//! ```rust
//! use rgflux::stats::{StatsConfig, batch_means, check_stationarity, effective_sample_size};
//!
//! let samples: Vec<f64> = (0..128)
//!     .map(|k| if k % 2 == 0 { 1.0 } else { -1.0 })
//!     .collect();
//! let est = batch_means(&samples, &StatsConfig::default())?;
//! assert_eq!(est.mean, 0.0);
//! let n_eff = effective_sample_size(&samples)?;
//! assert!(n_eff >= 1.0);
//! let stat = check_stationarity(&samples, &StatsConfig::default())?;
//! assert!(stat.is_stationary);
//! # Ok::<(), rgflux::FluxError>(())
//! ```
//!
//! ## References
//!
//! - Goswami, R. (2025). *Bayesian hierarchical models for quantitative
//!   estimates for performance metrics applied to saddle search algorithms*.
//!   AIP Advances, 15, 085210. <https://doi.org/10.1063/5.0283639>
//! - Flyvbjerg, H., and Petersen, H. G. (1989). *Error estimates on averages
//!   of correlated data*. The Journal of Chemical Physics, 91, 461–466.
//!   <https://doi.org/10.1063/1.457480>
//! - Geyer, C. J. (1992). *Practical Markov Chain Monte Carlo*. Statistical
//!   Science, 7, 473–483. <https://doi.org/10.1214/ss/1177011137>
//! - Geweke, J. (1992). *Evaluating the Accuracy of Sampling-Based Approaches
//!   to the Calculation of Posterior Moments*. In *Bayesian Statistics 4*,
//!   Oxford University Press, 169–193.
//!   <https://doi.org/10.1093/oso/9780198522669.003.0010>
//! - Flegal, J. M., Haran, M., and Jones, G. L. (2008). *Markov Chain Monte
//!   Carlo: Can We Trust the Third Significant Figure?* Statistical Science,
//!   23, 250–260. <https://doi.org/10.1214/08-STS257>
//! - Chodera, J. D. (2016). *A Simple Method for Automated Equilibration
//!   Detection in Molecular Simulations*. Journal of Chemical Theory and
//!   Computation, 12, 1799–1805. <https://doi.org/10.1021/acs.jctc.5b00784>

use crate::error::FluxError;

/// Relative flatness tolerance for sample spread: `64 * f64::EPSILON` (`2^-46`),
/// bounding accumulated IEEE-754 binary64 rounding in centered deviations.
pub const SERIES_FLAT_TOL: f64 = 64.0 * f64::EPSILON;

/// Relative variance floor: [`SERIES_FLAT_TOL`] squared (`2^-92`), below which
/// sample second moments represent floating-point rounding noise.
pub const VARIANCE_REL_FLOOR: f64 = SERIES_FLAT_TOL * SERIES_FLAT_TOL;

/// Default target floor on batch count when automatic batch sizing is active.
pub const DEFAULT_MIN_BATCHES: usize = 16;

/// Default leading fraction of the window in Geweke's stationarity test.
pub const DEFAULT_FIRST_FRACTION: f64 = 0.25;

/// Default trailing fraction of the window in Geweke's stationarity test.
pub const DEFAULT_LAST_FRACTION: f64 = 0.50;

/// Default maximum `|z|` score before rejecting stationarity.
pub const DEFAULT_MAX_Z_SCORE: f64 = 3.0;

/// Minimum window length required to split two non-overlapping sub-windows
/// with at least two observations per sub-window.
pub const MIN_STATIONARITY_SAMPLES: usize = 4;

/// Configuration for batch-means standard errors and stationarity checks.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct StatsConfig {
    /// Target minimum number of non-overlapping batches when `n` is large enough.
    pub min_batches: usize,
    /// Explicit batch width, or `None` for automatic `max(ceil(2 * tau_int), floor(sqrt(n)))`.
    pub batch_size: Option<usize>,
    /// Leading window fraction `(0, 1)` compared in Geweke's stationarity test.
    pub stationarity_first_fraction: f64,
    /// Trailing window fraction `(0, 1)` compared in Geweke's stationarity test.
    pub stationarity_last_fraction: f64,
    /// Maximum allowed `|z|` score in [`check_stationarity`].
    pub max_z_score: f64,
}

impl Default for StatsConfig {
    fn default() -> Self {
        Self {
            min_batches: DEFAULT_MIN_BATCHES,
            batch_size: None,
            stationarity_first_fraction: DEFAULT_FIRST_FRACTION,
            stationarity_last_fraction: DEFAULT_LAST_FRACTION,
            max_z_score: DEFAULT_MAX_Z_SCORE,
        }
    }
}

with_setters!(StatsConfig {
    with_min_batches => min_batches: val usize,
    with_batch_size => batch_size: opt usize,
    with_stationarity_first_fraction => stationarity_first_fraction: val f64,
    with_stationarity_last_fraction => stationarity_last_fraction: val f64,
    with_max_z_score => max_z_score: val f64,
});

/// Result of a non-overlapping batch-means uncertainty estimate.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct BatchEstimate {
    /// Full-sample arithmetic mean.
    pub mean: f64,
    /// Block-averaged standard error of [`Self::mean`].
    pub standard_error: f64,
    /// Asymptotic variance constant `sigma_BM^2 = n * SE^2`.
    pub variance: f64,
    /// Number of non-overlapping batches formed.
    pub batch_count: usize,
    /// Observations per batch.
    pub batch_size: usize,
    /// Total observations in the input slice.
    pub n_samples: usize,
    /// Geyer initial-monotone-sequence effective sample size `n_eff`.
    pub n_eff: f64,
    /// Estimated integrated autocorrelation time `tau_int` (`g = 2 * tau_int >= 1`).
    pub tau_int: f64,
}

/// Diagnostic returned by [`check_stationarity`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct StationarityReport {
    /// Whether `z_score <= max_z_score`.
    pub is_stationary: bool,
    /// Absolute Geweke two-window score `|mean_first - mean_second| / hypot(se_first, se_second)`.
    pub z_score: f64,
    /// Configured `max_z_score` threshold.
    pub max_z_score: f64,
    /// Ordinary-least-squares slope per sample index across the window.
    pub drift_slope: f64,
    /// Mean of the leading sub-window.
    pub first_mean: f64,
    /// Mean of the trailing sub-window.
    pub second_mean: f64,
}

fn validate_samples(samples: &[f64]) -> Result<(), FluxError> {
    if samples.is_empty() {
        return Err(FluxError::EmptyInput("stats sample slice is empty"));
    }
    for &x in samples {
        if !x.is_finite() {
            return Err(FluxError::NonFinite("stats sample slice"));
        }
    }
    Ok(())
}

fn validate_config(config: &StatsConfig) -> Result<(), FluxError> {
    if config.min_batches < 2 {
        return Err(FluxError::InvalidConfig(
            "StatsConfig::min_batches must be at least 2",
        ));
    }
    if let Some(bs) = config.batch_size
        && bs == 0
    {
        return Err(FluxError::InvalidConfig(
            "StatsConfig::batch_size must be at least 1",
        ));
    }
    if !config.stationarity_first_fraction.is_finite()
        || !config.stationarity_last_fraction.is_finite()
        || config.stationarity_first_fraction <= 0.0
        || config.stationarity_last_fraction <= 0.0
        || config.stationarity_first_fraction + config.stationarity_last_fraction >= 1.0
            && (config.stationarity_first_fraction + config.stationarity_last_fraction - 1.0)
                > SERIES_FLAT_TOL
    {
        return Err(FluxError::InvalidConfig(
            "StatsConfig stationarity fractions must be positive and sum to at most 1",
        ));
    }
    if !config.max_z_score.is_finite() || config.max_z_score <= 0.0 {
        return Err(FluxError::InvalidConfig(
            "StatsConfig::max_z_score must be finite and positive",
        ));
    }
    Ok(())
}

fn mean_and_max_dev(samples: &[f64]) -> (f64, f64) {
    let n = samples.len() as f64;
    let mean = samples.iter().copied().sum::<f64>() / n;
    let mut max_dev = 0.0_f64;
    for &x in samples {
        let dev = (x - mean).abs();
        if dev > max_dev {
            max_dev = dev;
        }
    }
    (mean, max_dev)
}

/// Integrated autocorrelation time `tau_int` and statistical inefficiency `g = 2 * tau_int`
/// using Geyer's (1992) initial positive and monotone sequence estimator.
fn geyer_inefficiency(samples: &[f64], mean: f64) -> (f64, f64) {
    let n = samples.len();
    if n < 4 {
        return (0.5, n as f64);
    }
    let scale = 1.0 + mean.abs();
    let gamma0 = samples
        .iter()
        .map(|&x| {
            let d = x - mean;
            d * d
        })
        .sum::<f64>()
        / (n as f64);
    if gamma0 <= VARIANCE_REL_FLOOR * scale * scale {
        return (0.5, n as f64);
    }

    let autocov = |lag: usize| -> f64 {
        let mut acc = 0.0_f64;
        for i in 0..(n - lag) {
            acc += (samples[i] - mean) * (samples[i + lag] - mean);
        }
        acc / (n as f64)
    };

    let mut pair_sum_prev = gamma0 + autocov(1);
    if pair_sum_prev <= gamma0 {
        return (0.5, n as f64);
    }

    let mut sum_pairs = pair_sum_prev;
    let mut m = 1_usize;
    while 2 * m + 1 < n / 2 {
        let raw_pair = autocov(2 * m) + autocov(2 * m + 1);
        if raw_pair <= 0.0 {
            break;
        }
        let monotone_pair = if raw_pair < pair_sum_prev {
            raw_pair
        } else {
            pair_sum_prev
        };
        sum_pairs += monotone_pair;
        pair_sum_prev = monotone_pair;
        m += 1;
    }

    let raw_g = (-gamma0 + 2.0 * sum_pairs) / gamma0;
    let g = if raw_g < 1.0 { 1.0 } else { raw_g };
    let tau_int = 0.5 * g;
    let raw_neff = (n as f64) / g;
    let n_eff = if raw_neff < 1.0 {
        1.0
    } else if raw_neff > n as f64 {
        n as f64
    } else {
        raw_neff
    };
    (tau_int, n_eff)
}

/// Estimate the effective sample size `n_eff = n / (2 * tau_int)` of a stationary series.
///
/// Uses Geyer's (1992) initial monotone sequence estimator. Constant or
/// anti-correlated series return `samples.len() as f64`.
pub fn effective_sample_size(samples: &[f64]) -> Result<f64, FluxError> {
    validate_samples(samples)?;
    let (mean, max_dev) = mean_and_max_dev(samples);
    let scale = 1.0 + mean.abs();
    if max_dev <= SERIES_FLAT_TOL * scale {
        return Ok(samples.len() as f64);
    }
    let (_tau_int, n_eff) = geyer_inefficiency(samples, mean);
    Ok(n_eff)
}

/// Compute the mean, non-overlapping batch-means standard error, and `n_eff` of `samples`.
///
/// Follows Flegal, Haran, and Jones (2008): `sigma_BM^2 = (b / (a - 1)) * sum (Y_j - Y_bar)^2`
/// over `a` batches of width `b`, yielding `SE = sqrt(sigma_BM^2 / n)`.
pub fn batch_means(samples: &[f64], config: &StatsConfig) -> Result<BatchEstimate, FluxError> {
    validate_samples(samples)?;
    validate_config(config)?;

    let n = samples.len();
    let (mean, max_dev) = mean_and_max_dev(samples);
    if n == 1 {
        return Ok(BatchEstimate {
            mean,
            standard_error: 0.0,
            variance: 0.0,
            batch_count: 1,
            batch_size: 1,
            n_samples: 1,
            n_eff: 1.0,
            tau_int: 0.5,
        });
    }

    let scale = 1.0 + mean.abs();
    if max_dev <= SERIES_FLAT_TOL * scale {
        return Ok(BatchEstimate {
            mean,
            standard_error: 0.0,
            variance: 0.0,
            batch_count: n,
            batch_size: 1,
            n_samples: n,
            n_eff: n as f64,
            tau_int: 0.5,
        });
    }

    let (tau_int, n_eff) = geyer_inefficiency(samples, mean);

    let batch_size = match config.batch_size {
        Some(bs) => {
            if n / bs < 2 {
                return Err(FluxError::InvalidConfig(
                    "StatsConfig::batch_size leaves fewer than 2 batches",
                ));
            }
            bs
        }
        None => {
            let corr_width = {
                let raw = (2.0 * tau_int).ceil() as usize;
                if raw < 1 { 1 } else { raw }
            };
            if n < 2 * config.min_batches && corr_width <= 1 {
                1
            } else {
                let sqrt_width = {
                    let raw = (n as f64).sqrt().floor() as usize;
                    if raw < 1 { 1 } else { raw }
                };
                let target = if corr_width > sqrt_width {
                    corr_width
                } else {
                    sqrt_width
                };
                let max_for_min_batches = if n >= 2 * config.min_batches {
                    n / config.min_batches
                } else {
                    n / 2
                };
                let chosen = if corr_width > max_for_min_batches {
                    corr_width
                } else if target > max_for_min_batches {
                    max_for_min_batches
                } else {
                    target
                };
                let half_n = n / 2;
                if chosen > half_n {
                    half_n
                } else if chosen < 1 {
                    1
                } else {
                    chosen
                }
            }
        }
    };

    let batch_count = n / batch_size;
    let mut batch_means_vec = Vec::with_capacity(batch_count);
    for b in 0..batch_count {
        let start = b * batch_size;
        let end = start + batch_size;
        let bm = samples[start..end].iter().copied().sum::<f64>() / (batch_size as f64);
        batch_means_vec.push(bm);
    }
    let grand_bm = batch_means_vec.iter().copied().sum::<f64>() / (batch_count as f64);
    let sq_sum = batch_means_vec
        .iter()
        .map(|&bm| {
            let d = bm - grand_bm;
            d * d
        })
        .sum::<f64>();
    let variance = (batch_size as f64) * sq_sum / ((batch_count - 1) as f64);
    let standard_error = (variance / (n as f64)).sqrt();

    Ok(BatchEstimate {
        mean,
        standard_error,
        variance,
        batch_count,
        batch_size,
        n_samples: n,
        n_eff,
        tau_int,
    })
}

fn ols_slope(samples: &[f64], mean_y: f64) -> f64 {
    let n = samples.len();
    if n < 2 {
        return 0.0;
    }
    let mean_x = 0.5 * ((n - 1) as f64);
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for (i, &y) in samples.iter().enumerate() {
        let dx = (i as f64) - mean_x;
        num += dx * (y - mean_y);
        den += dx * dx;
    }
    if den <= 0.0 { 0.0 } else { num / den }
}

/// Test stationarity of a candidate plateau or equilibrium window via Geweke's two-window score.
///
/// Compares the leading `stationarity_first_fraction` and trailing
/// `stationarity_last_fraction` sub-windows using their batch-means standard
/// errors, alongside the ordinary-least-squares linear drift slope.
pub fn check_stationarity(
    samples: &[f64],
    config: &StatsConfig,
) -> Result<StationarityReport, FluxError> {
    validate_samples(samples)?;
    validate_config(config)?;

    let n = samples.len();
    if n < MIN_STATIONARITY_SAMPLES {
        return Err(FluxError::InvalidConfig(
            "check_stationarity requires at least 4 samples",
        ));
    }

    let (mean, max_dev) = mean_and_max_dev(samples);
    let scale = 1.0 + mean.abs();
    if max_dev <= SERIES_FLAT_TOL * scale {
        return Ok(StationarityReport {
            is_stationary: true,
            z_score: 0.0,
            max_z_score: config.max_z_score,
            drift_slope: 0.0,
            first_mean: mean,
            second_mean: mean,
        });
    }

    let raw_first = ((n as f64) * config.stationarity_first_fraction).floor() as usize;
    let n_first = if raw_first < 2 { 2 } else { raw_first };
    let raw_second = ((n as f64) * config.stationarity_last_fraction).floor() as usize;
    let max_second = n - n_first;
    let n_second = if raw_second < 2 {
        2
    } else if raw_second > max_second {
        max_second
    } else {
        raw_second
    };

    let sub_config = StatsConfig::default().with_min_batches(config.min_batches);
    let first_est = batch_means(&samples[..n_first], &sub_config)?;
    let second_est = batch_means(&samples[n - n_second..], &sub_config)?;
    let drift_slope = ols_slope(samples, mean);

    let diff = (first_est.mean - second_est.mean).abs();
    let denom = first_est.standard_error.hypot(second_est.standard_error);
    let z_score = if denom <= SERIES_FLAT_TOL * scale {
        if diff <= SERIES_FLAT_TOL * scale {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        diff / denom
    };

    Ok(StationarityReport {
        is_stationary: z_score <= config.max_z_score,
        z_score,
        max_z_score: config.max_z_score,
        drift_slope,
        first_mean: first_est.mean,
        second_mean: second_est.mean,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn box_muller(rng: &mut StdRng) -> f64 {
        let u1: f64 = 1.0 - rng.random::<f64>();
        let u2: f64 = rng.random::<f64>();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    #[test]
    fn rejects_empty_and_non_finite_inputs() {
        assert_eq!(
            batch_means(&[], &StatsConfig::default()),
            Err(FluxError::EmptyInput("stats sample slice is empty"))
        );
        assert_eq!(
            effective_sample_size(&[1.0, f64::NAN]),
            Err(FluxError::NonFinite("stats sample slice"))
        );
    }

    #[test]
    fn constant_series_has_zero_error_and_is_stationary() {
        let xs = vec![0.75_f64; 64];
        let est = batch_means(&xs, &StatsConfig::default()).expect("valid constant series");
        assert_eq!(est.mean, 0.75);
        assert_eq!(est.standard_error, 0.0);
        assert_eq!(est.n_eff, 64.0);
        let rep = check_stationarity(&xs, &StatsConfig::default()).expect("stationary");
        assert!(rep.is_stationary);
        assert_eq!(rep.z_score, 0.0);
    }

    #[test]
    fn ar1_chain_reduces_n_eff_and_detects_linear_drift() {
        let mut rng = StdRng::seed_from_u64(42);
        let rho = 0.8_f64;
        let sigma_eps = (1.0 - rho * rho).sqrt();
        let n = 4_000_usize;
        let mut x = 0.0_f64;
        let mut chain = Vec::with_capacity(n);
        for _ in 0..n {
            x = rho * x + sigma_eps * box_muller(&mut rng);
            chain.push(x);
        }

        let est = batch_means(&chain, &StatsConfig::default()).expect("ar1 batch means");
        // Exact asymptotic variance of AR(1) with marginal variance 1 is (1+rho)/(1-rho) = 9,
        // so n_eff ~ 4000 / 9 = 444.4 and SE ~ 3 / sqrt(4000) = 0.0474.
        assert!(est.n_eff > 200.0 && est.n_eff < 900.0);
        assert!(est.standard_error > 0.03 && est.standard_error < 0.07);

        let stationary = check_stationarity(&chain, &StatsConfig::default()).expect("stationary");
        assert!(stationary.is_stationary);

        let drifting: Vec<f64> = chain
            .iter()
            .enumerate()
            .map(|(k, &v)| v + 0.002 * (k as f64))
            .collect();
        let drifted = check_stationarity(&drifting, &StatsConfig::default()).expect("drift check");
        assert!(!drifted.is_stationary);
        assert!(drifted.z_score > DEFAULT_MAX_Z_SCORE);
        assert!(drifted.drift_slope > 0.0);
    }
}
