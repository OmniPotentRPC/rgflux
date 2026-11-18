//! Forward Flux Sampling (FFS) across ordered reaction-coordinate interfaces.
//!
//! Computes rare-event transition rates `k_AB = Phi_A0 * prod_i P(lambda_{i+1} | lambda_i)`
//! in equilibrium or non-equilibrium stochastic dynamics without requiring
//! detailed balance or knowledge of the stationary phase-space density.
//!
//! ## References
//!
//! - Allen, R. J., Frenkel, D., and ten Wolde, P. R. (2006). *Simulating rare
//!   events in equilibrium or nonequilibrium stochastic systems*. The Journal
//!   of Chemical Physics, 124, 024102. <https://doi.org/10.1063/1.2140273>
//! - Allen, R. J., Frenkel, D., and ten Wolde, P. R. (2006). *Forward flux
//!   sampling-type schemes for simulating rare events: Efficiency analysis*.
//!   The Journal of Chemical Physics, 124, 194111.
//!   <https://doi.org/10.1063/1.2198827>
//! - Allen, R. J., Warren, P. B., and ten Wolde, P. R. (2005). *Sampling Rare
//!   Switching Events in Biochemical Networks*. Physical Review Letters, 94,
//!   018104. <https://doi.org/10.1103/PhysRevLett.94.018104>
//! - Allen, R. J., Valeriani, C., and ten Wolde, P. R. (2009). *Forward flux
//!   sampling for rare event simulations*. Journal of Physics: Condensed
//!   Matter, 21, 463102. <https://doi.org/10.1088/0953-8984/21/46/463102>

use ndarray::ArrayView1;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::engine::{DynamicsEngine, EngineSnapshot};
use crate::error::FluxError;
use crate::stats::{StatsConfig, batch_means};

/// Configuration for direct Forward Flux Sampling across ordered interfaces.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct FfsConfig {
    /// Strictly monotone interface levels `lambda_0 < lambda_1 < ... < lambda_n`.
    pub interfaces: Vec<f64>,
    /// Target crossing configurations collected at each interface.
    pub points_per_interface: usize,
    /// Trial trajectories launched per interface to estimate `P(lambda_{i+1} | lambda_i)`.
    pub trials_per_interface: usize,
    /// Maximum integration steps per trial trajectory before aborting.
    pub max_trial_steps: usize,
    /// Maximum integration steps spent collecting the initial flux crossings.
    pub max_flux_steps: usize,
    /// Burn-in integration steps discarded before flux counting starts.
    pub burn_in_steps: usize,
    /// Integration timestep `dt > 0`.
    pub time_step: f64,
    /// Deterministic PRNG seed.
    pub seed: u64,
    /// Batch-means controls for the initial escape flux `Phi_{A,0}`.
    pub stats: StatsConfig,
}

impl Default for FfsConfig {
    fn default() -> Self {
        Self {
            interfaces: vec![-0.5, 0.0, 0.5],
            points_per_interface: 128,
            trials_per_interface: 256,
            max_trial_steps: 10_000,
            max_flux_steps: 200_000,
            burn_in_steps: 0,
            time_step: 0.005,
            seed: 1,
            stats: StatsConfig::default(),
        }
    }
}

with_setters!(FfsConfig {
    with_interfaces => interfaces: val Vec<f64>,
    with_points_per_interface => points_per_interface: val usize,
    with_trials_per_interface => trials_per_interface: val usize,
    with_max_trial_steps => max_trial_steps: val usize,
    with_max_flux_steps => max_flux_steps: val usize,
    with_burn_in_steps => burn_in_steps: val usize,
    with_time_step => time_step: val f64,
    with_seed => seed: val u64,
    with_stats => stats: val StatsConfig,
});

/// Forward Flux Sampling rate and per-interface conditional advancement probabilities.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct FfsEstimate {
    /// Initial positive crossing flux `Phi_{A,0}` out of basin `A` across `lambda_0`.
    pub initial_flux: f64,
    /// Conditional probabilities `P(lambda_{i+1} | lambda_i)` across consecutive interfaces.
    pub interface_probabilities: Vec<f64>,
    /// Composite forward transition rate `k_{AB}`.
    pub rate: f64,
    /// Propagated standard error on [`Self::rate`].
    pub standard_error: f64,
}

#[derive(Clone, Debug)]
struct PhasePoint {
    positions: Vec<f64>,
    velocities: Vec<f64>,
}

#[derive(Clone, Copy, Debug, Default)]
struct TrialCluster {
    trials: u64,
    successes: u64,
}

fn validate(config: &FfsConfig) -> Result<(), FluxError> {
    if config.interfaces.len() < 2 {
        return Err(FluxError::InvalidConfig(
            "FfsConfig::interfaces must contain at least two levels",
        ));
    }
    if config.interfaces.iter().any(|v| !v.is_finite()) {
        return Err(FluxError::NonFinite("FfsConfig::interfaces"));
    }
    if config.interfaces.windows(2).any(|pair| pair[1] <= pair[0]) {
        return Err(FluxError::InvalidConfig(
            "FfsConfig::interfaces must be strictly increasing",
        ));
    }
    if config.points_per_interface < 2 {
        return Err(FluxError::InvalidConfig(
            "FfsConfig::points_per_interface must be at least 2",
        ));
    }
    if config.trials_per_interface < 1 {
        return Err(FluxError::InvalidConfig(
            "FfsConfig::trials_per_interface must be positive",
        ));
    }
    if config.max_trial_steps < 1 || config.max_flux_steps < 1 {
        return Err(FluxError::InvalidConfig(
            "FfsConfig step budgets must be positive",
        ));
    }
    if !config.time_step.is_finite() || config.time_step <= 0.0 {
        return Err(FluxError::InvalidConfig(
            "FfsConfig::time_step must be finite and positive",
        ));
    }
    Ok(())
}

fn set_point<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    point: &PhasePoint,
) -> Result<(), FluxError> {
    engine.set_positions(ArrayView1::from(&point.positions))?;
    engine.set_velocities(ArrayView1::from(&point.velocities))?;
    Ok(())
}

fn capture_point<E: DynamicsEngine + ?Sized>(engine: &E) -> PhasePoint {
    PhasePoint {
        positions: engine.positions().to_vec(),
        velocities: engine.velocities().to_vec(),
    }
}

fn flux_from_intervals(intervals: &[f64], stats: &StatsConfig) -> Result<(f64, f64), FluxError> {
    let n = intervals.len();
    if n < 2 {
        return Err(FluxError::InvalidConfig(
            "FFS flux intervals require at least two crossings",
        ));
    }
    if intervals.iter().any(|d| !d.is_finite() || *d <= 0.0) {
        return Err(FluxError::InvalidConfig(
            "FFS flux intervals must be finite and positive",
        ));
    }
    let interval_est = batch_means(intervals, stats)?;
    if interval_est.mean <= 0.0 {
        return Err(FluxError::InvalidConfig(
            "FFS flux intervals must have positive mean interval",
        ));
    }
    let flux = 1.0 / interval_est.mean;
    let se = interval_est.standard_error / (interval_est.mean * interval_est.mean);
    if !flux.is_finite() || !se.is_finite() {
        return Err(FluxError::NonFinite("FFS initial flux estimate"));
    }
    Ok((flux, se))
}

fn stage_probability(clusters: &[TrialCluster]) -> Result<(f64, f64), FluxError> {
    if clusters.len() < 2 {
        return Err(FluxError::InvalidConfig(
            "FFS stage estimate requires at least two starting points",
        ));
    }
    let trials: u64 = clusters.iter().map(|c| c.trials).sum();
    let successes: u64 = clusters.iter().map(|c| c.successes).sum();
    if trials == 0 || successes == 0 {
        return Err(FluxError::InvalidConfig(
            "FFS stage estimate requires at least one successful trial",
        ));
    }
    let k = trials as f64;
    let p = successes as f64 / k;
    let m = clusters.len() as f64;
    let spread = clusters
        .iter()
        .map(|c| {
            let residual = c.successes as f64 - p * c.trials as f64;
            residual * residual
        })
        .sum::<f64>();
    let se = (m / (m - 1.0) * spread).sqrt() / k;
    Ok((p, se))
}

fn finalize_and_restore<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    initial: EngineSnapshot,
    result: Result<FfsEstimate, FluxError>,
) -> Result<FfsEstimate, FluxError> {
    initial.restore_result(engine, result)
}

/// Run Forward Flux Sampling on `engine` using order parameter `order_param`.
pub fn forward_flux_rate<E, F>(
    engine: &mut E,
    order_param: F,
    config: &FfsConfig,
) -> Result<FfsEstimate, FluxError>
where
    E: DynamicsEngine + ?Sized,
    F: Fn(ArrayView1<'_, f64>) -> f64,
{
    validate(config)?;
    let interfaces = &config.interfaces;
    let lambda_a = interfaces[0];
    let product_level = interfaces[interfaces.len() - 1];
    let dt = config.time_step;
    let mut rng = StdRng::seed_from_u64(config.seed);

    let initial = EngineSnapshot::capture(engine);
    let initial_point = capture_point(engine);
    let run = (|| -> Result<FfsEstimate, FluxError> {
        let start_lambda = order_param(engine.positions());
        if !start_lambda.is_finite() {
            return Err(FluxError::NonFinite("FFS order parameter at initial state"));
        }
        if start_lambda > lambda_a {
            return Err(FluxError::InvalidConfig(
                "FFS initial state must begin in basin A (lambda <= lambda_0)",
            ));
        }

        // Drive flux sampling from the config seed as well as trial launches.
        engine.reseed(rng.random())?;

        let mut points: Vec<PhasePoint> = Vec::with_capacity(config.points_per_interface);
        let mut intervals: Vec<f64> = Vec::with_capacity(config.points_per_interface);
        let mut armed = true;
        let mut burn_in = config.burn_in_steps;
        let mut since_crossing = 0.0_f64;
        let mut flux_steps = 0_usize;

        while points.len() < config.points_per_interface {
            if flux_steps >= config.max_flux_steps {
                return Err(FluxError::InvalidConfig(
                    "FFS flux budget exceeded before collecting interface points",
                ));
            }
            let _ = engine.step(dt)?;
            flux_steps += 1;
            let counting = burn_in == 0;
            if counting {
                since_crossing += dt;
            } else {
                burn_in -= 1;
            }
            let lambda = order_param(engine.positions());
            if !lambda.is_finite() {
                return Err(FluxError::NonFinite("FFS order parameter during flux run"));
            }
            if lambda <= lambda_a {
                armed = true;
            } else if armed {
                armed = false;
                if counting {
                    intervals.push(since_crossing);
                    since_crossing = 0.0;
                    points.push(capture_point(engine));
                }
            }
            if lambda >= product_level {
                set_point(engine, &initial_point)?;
                armed = true;
            }
        }
        let (initial_flux, initial_flux_se) = flux_from_intervals(&intervals, &config.stats)?;

        let mut stage_probabilities = Vec::with_capacity(interfaces.len() - 1);
        let mut stage_ses = Vec::with_capacity(interfaces.len() - 1);
        let mut stage_points = points;

        for stage in 0..(interfaces.len() - 1) {
            if stage_points.len() < 2 {
                return Err(FluxError::InvalidConfig(
                    "FFS stage has fewer than two starting points",
                ));
            }
            let target = interfaces[stage + 1];
            let is_last = stage + 1 == interfaces.len() - 1;
            let mut clusters = vec![TrialCluster::default(); stage_points.len()];
            let mut reached = Vec::new();

            for _ in 0..config.trials_per_interface {
                let index = rng.random_range(0..stage_points.len());
                set_point(engine, &stage_points[index])?;
                engine.reseed(rng.random())?;

                let mut trial_steps = 0_usize;
                let mut success = false;
                loop {
                    if trial_steps >= config.max_trial_steps {
                        return Err(FluxError::InvalidConfig(
                            "FFS trial budget exceeded before commitment",
                        ));
                    }
                    let _ = engine.step(dt)?;
                    trial_steps += 1;
                    let lambda = order_param(engine.positions());
                    if !lambda.is_finite() {
                        return Err(FluxError::NonFinite("FFS order parameter during trial"));
                    }
                    if lambda >= target {
                        success = true;
                        if !is_last {
                            reached.push(capture_point(engine));
                        }
                        break;
                    }
                    if lambda <= lambda_a {
                        break;
                    }
                }
                clusters[index].trials += 1;
                if success {
                    clusters[index].successes += 1;
                }
            }

            let (probability, se) = stage_probability(&clusters)?;
            stage_probabilities.push(probability);
            stage_ses.push(se);
            stage_points = reached;
        }

        let mut rate = initial_flux;
        let mut relative_se_sq = (initial_flux_se / initial_flux).powi(2);
        for i in 0..stage_probabilities.len() {
            let p = stage_probabilities[i];
            let se = stage_ses[i];
            rate *= p;
            relative_se_sq += (se / p).powi(2);
        }
        let standard_error = rate * relative_se_sq.sqrt();

        Ok(FfsEstimate {
            initial_flux,
            interface_probabilities: stage_probabilities,
            rate,
            standard_error,
        })
    })();

    finalize_and_restore(engine, initial, run)
}

#[cfg(test)]
mod tests {
    use super::{TrialCluster, flux_from_intervals, stage_probability};
    use crate::error::FluxError;
    use crate::stats::StatsConfig;

    #[test]
    fn flux_from_intervals_matches_hand_calculation() {
        let intervals = [2.0, 3.0, 5.0];
        let stats = StatsConfig::default().with_batch_size(1);
        let (flux, se) = flux_from_intervals(&intervals, &stats).expect("valid intervals");
        let expected_flux = 3.0 / 10.0;
        let mean = 10.0 / 3.0;
        let variance =
            ((2.0_f64 - mean).powi(2) + (3.0_f64 - mean).powi(2) + (5.0_f64 - mean).powi(2))
                / 2.0_f64;
        let expected_se = expected_flux * variance.sqrt() / (mean * (3.0_f64).sqrt());
        assert!((flux - expected_flux).abs() <= f64::EPSILON);
        assert!((se - expected_se).abs() <= f64::EPSILON);
    }

    #[test]
    fn flux_from_intervals_rejects_less_than_two_crossings() {
        let stats = StatsConfig::default().with_batch_size(1);
        let err = flux_from_intervals(&[1.0], &stats).expect_err("n < 2");
        assert!(
            matches!(err, FluxError::InvalidConfig(msg) if msg.contains("at least two crossings"))
        );
    }

    #[test]
    fn flux_from_intervals_uses_stats_batch_size_for_se() {
        let intervals = [1.0, 2.0, 4.0, 8.0];
        let one = flux_from_intervals(&intervals, &StatsConfig::default().with_batch_size(1))
            .expect("batch 1");
        let two = flux_from_intervals(&intervals, &StatsConfig::default().with_batch_size(2))
            .expect("batch 2");
        assert!((one.0 - two.0).abs() <= f64::EPSILON);
        assert_ne!(one.1.to_bits(), two.1.to_bits());
    }

    #[test]
    fn stage_probability_matches_hand_calculation() {
        let clusters = [
            TrialCluster {
                trials: 2,
                successes: 2,
            },
            TrialCluster {
                trials: 2,
                successes: 0,
            },
        ];
        let (p, se) = stage_probability(&clusters).expect("valid clusters");
        assert!((p - 0.5).abs() <= f64::EPSILON);
        assert!((se - 0.5).abs() <= f64::EPSILON);
    }

    #[test]
    fn flux_from_intervals_rejects_non_finite_and_non_positive_intervals() {
        let stats = StatsConfig::default().with_batch_size(1);
        for bad in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
            let err = flux_from_intervals(&[1.0, bad, 2.0], &stats).expect_err("bad interval");
            assert!(
                matches!(err, FluxError::InvalidConfig(msg) if msg.contains("finite and positive")),
                "{bad}"
            );
        }
    }

    #[test]
    fn flux_from_intervals_rejects_non_finite_standard_error_with_finite_flux() {
        // Flat series: se = 0 and mean^2 underflows, so se / mean^2 is NaN
        // while the flux 1 / mean = 1e200 stays finite.
        let stats = StatsConfig::default().with_batch_size(1);
        let err = flux_from_intervals(&[1.0e-200; 4], &stats).expect_err("NaN standard error");
        assert_eq!(err, FluxError::NonFinite("FFS initial flux estimate"));
    }

    #[test]
    fn stage_probability_rejects_zero_successes_among_real_trials() {
        let clusters = [TrialCluster {
            trials: 3,
            successes: 0,
        }; 2];
        let err = stage_probability(&clusters).expect_err("no successes");
        assert!(
            matches!(err, FluxError::InvalidConfig(msg) if msg.contains("at least one successful trial"))
        );
    }
}
