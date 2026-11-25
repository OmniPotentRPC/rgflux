//! Multithermal on-the-fly probability enhanced sampling (OPES).
//!
//! [`OpesMultithermal`] is the expanded-ensemble OPES bias of Invernizzi,
//! Piaggi, and Parrinello (2020) with the multithermal expansion, the method
//! PLUMED provides as `OPES_EXPANDED` over `ECV_MULTITHERMAL`. A simulation
//! thermostatted at `T0` (inverse temperature `beta0 = 1 / (k_B T0)`) samples
//! a uniform mixture of the canonical ensembles at `T_1 <= ... <= T_N`.
//!
//! ## Equations
//!
//! The expansion collective variables are `Delta u_k(x) = (beta_k - beta0) U(x)`
//! with `U` the potential energy. After `n` updates the bias is
//!
//! ```text
//! V_n(U) = -(1 / beta0) ln[ (1 / N) sum_k exp(-Delta u_k + beta0 Delta F_n(k)) ]
//! ```
//!
//! and the free-energy differences are the reweighted estimates
//!
//! ```text
//! exp(-beta0 Delta F_n(k)) = sum_t exp(beta0 V_t) exp(-Delta u_k(t)) / sum_t exp(beta0 V_t)
//! ```
//!
//! over the samples taken every `pace` steps, with `V_t` the bias in force
//! when sample `t` was taken. `Delta F(k)` is in energy units, as in the PLUMED
//! `DELTAFS` file, and converges to `-(1 / beta0) ln(Z(beta_k) / Z(beta0))`
//! for configurational partition functions `Z`. Before the first update the
//! estimates are initialized from `observation_steps` unbiased samples with
//! the same expression at `V_t = 0`. The running constant
//! `rct = (1 / beta0) ln <exp(beta0 V)>` is kept only for restarts.
//!
//! The bias depends on the coordinates only through `U`, so
//!
//! ```text
//! -grad(U + V(U)) = (1 + dV/dU) F,   1 + dV/dU = sum_k w_k beta_k / beta0,
//! ```
//!
//! with `F = -grad U`, `w_k` the normalized mixture weights
//! `w_k proportional to exp(-Delta u_k + beta0 Delta F(k))`, and no collective
//! variable gradient. PLUMED applies the same factor as `1 - f_U`, where
//! `f_U = -dV/dU` is the force on its `ENERGY` value.
//!
//! A frame sampled with bias `V_t` reweights to any `T = 1 / (k_B beta)` in
//! `[T_1, T_N]` with the logarithmic weight
//!
//! ```text
//! ln w_t(T) = beta0 V_t + (beta0 - beta) U_t,
//! ```
//!
//! which is the sum of the PLUMED `REWEIGHT_BIAS` and `REWEIGHT_TEMP_PRESS`
//! log weights.
//!
//! ## Engine wiring
//!
//! [`OpesMultithermalSurface`] wraps any [`ForceSurface`] and shares one
//! [`SharedOpesMultithermal`] with [`run_opes_multithermal`], which steps a
//! [`DynamicsEngine`] built over that surface (for example
//! [`crate::engine::native::NativeEngine`]), feeds the unbiased energy of
//! every step to [`OpesMultithermal::update`], and records frames for
//! [`MultithermalReweighting`].
//!
//! ## Example
//!
//! ```rust
//! use ndarray::{Array1, ArrayView1};
//! use rgflux::engine::native::{NativeConfig, NativeEngine};
//! use rgflux::engine::{DynamicsEngine, ThermostatConfig};
//! use rgflux::opes::{
//!     OpesMultithermal, OpesMultithermalConfig, OpesMultithermalSurface, OpesRunConfig,
//!     SharedOpesMultithermal, run_opes_multithermal,
//! };
//!
//! struct DoubleWell;
//! impl rgsaddle::PointSurface for DoubleWell {
//!     fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
//!         let a = x[0] * x[0] - 1.0;
//!         Ok((a * a, Array1::from_vec(vec![4.0 * x[0] * a])))
//!     }
//! }
//!
//! let thermostat = ThermostatConfig::default().with_temperature(0.3);
//! let config = OpesMultithermalConfig::range(0.3, 1.0)
//!     .with_temp_steps(4)
//!     .with_pace(10)
//!     .with_observation_steps(10);
//! let shared = SharedOpesMultithermal::new(OpesMultithermal::new(config, &thermostat)?);
//! let surface = OpesMultithermalSurface::new(DoubleWell, shared.clone());
//! let native = NativeConfig::default().with_thermostat(thermostat).with_seed(3);
//! let mut engine = NativeEngine::new(surface, native)?;
//! engine.set_positions(Array1::from_vec(vec![-1.0]).view())?;
//! let run = OpesRunConfig::default().with_steps(2_000).with_time_step(0.01);
//! let trajectory = run_opes_multithermal(&mut engine, &shared, &run)?;
//! let weights = trajectory.weights(0.6)?;
//! assert_eq!(weights.len(), trajectory.potential_energies.len());
//! assert_eq!(trajectory.delta_f().len(), 4);
//! # Ok::<(), rgflux::FluxError>(())
//! ```
//!
//! ## References
//!
//! - Invernizzi, M., Piaggi, P. M., and Parrinello, M. (2020). *Unified
//!   Approach to Enhanced Sampling*. Physical Review X, 10, 041034.
//!   <https://doi.org/10.1103/PhysRevX.10.041034>
//! - Invernizzi, M., and Parrinello, M. (2020). *Rethinking Metadynamics:
//!   From Bias Potentials to Probability Distributions*. The Journal of
//!   Physical Chemistry Letters, 11, 2731-2736.
//!   <https://doi.org/10.1021/acs.jpclett.0c00497>
//! - Tribello, G. A., Bonomi, M., Branduardi, D., Camilloni, C., and Bussi, G.
//!   (2014). *PLUMED 2: New feathers for an old bird*. Computer Physics
//!   Communications, 185, 604-613. <https://doi.org/10.1016/j.cpc.2013.09.018>
//! - Kish, L. (1965). *Survey Sampling*. John Wiley and Sons, New York.

use std::io::{BufRead, Write};
use std::sync::{Arc, Mutex, MutexGuard};

use ndarray::Array1;

use crate::engine::{DynamicsEngine, ForceSurface, ThermostatConfig, validate_finite_slice};
use crate::error::FluxError;
use crate::stats::{StatsConfig, batch_means};

/// Default bias update stride in MD steps.
///
/// PLUMED makes `PACE` compulsory; `500` is the value in its
/// `OPES_EXPANDED` and `ECV_MULTITHERMAL` examples.
pub const DEFAULT_PACE: usize = 500;

/// Default count of unbiased `pace` samples before the first bias, PLUMED's
/// `OBSERVATION_STEPS` default.
pub const DEFAULT_OBSERVATION_STEPS: usize = 100;

/// Default stride, in bias updates, between recorded `Delta F` estimates,
/// PLUMED's `PRINT_STRIDE` default.
pub const DEFAULT_PRINT_STRIDE: usize = 100;

/// Field name of the temperature column in a state file.
///
/// PLUMED names the column after the expansion argument, so a PLUMED input
/// that restarts from an `rgflux` state labels the expansion `ecv` and the
/// energy `ene`.
pub const STATE_ENERGY_FIELD: &str = "ecv.ene";

/// Root tolerance on `N_eff / N - 1/2` in automatic temperature spacing,
/// the value PLUMED uses.
const NEFF_ROOT_TOL: f64 = 1e-4;

/// Iteration cap on the regula falsi search for the automatic spacing.
const NEFF_ROOT_MAX_ITERATIONS: usize = 100_000;

/// Relative slack on the reweighting range: temperatures recovered from the
/// inverse-temperature ladder carry a few units of rounding.
const RANGE_REL_TOL: f64 = 64.0 * f64::EPSILON;

/// Spacing of an automatic or `temp_steps` temperature ladder.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TemperatureSpacing {
    /// Constant ratio `T_{k+1} / T_k`, PLUMED's default.
    #[default]
    Geometric,
    /// Equal steps in inverse temperature `beta`, PLUMED's `NO_GEOM_SPACING`.
    LinearInverse,
}

/// Configuration of [`OpesMultithermal`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct OpesMultithermalConfig {
    /// Simulation temperature `T0 > 0`, PLUMED's `TEMP`.
    ///
    /// `None` takes [`ThermostatConfig::temperature`]. Default: `None`.
    pub temperature: Option<f64>,
    /// Lowest target temperature `T_min > 0`, PLUMED's `TEMP_MIN`.
    ///
    /// `None` uses `T0`. At least one of `temp_min` and `temp_max` must be
    /// set unless `temp_set_all` is. Default: `None`.
    pub temp_min: Option<f64>,
    /// Highest target temperature `T_max >= T_min`, PLUMED's `TEMP_MAX`.
    ///
    /// `None` uses `T0`. Default: `None`.
    pub temp_max: Option<f64>,
    /// Number of target temperatures `>= 1`, PLUMED's `TEMP_STEPS`.
    ///
    /// `None` estimates the count from the observation samples, as PLUMED
    /// does, unless `T_min == T_max`, which uses one temperature.
    /// Default: `None`.
    pub temp_steps: Option<usize>,
    /// Explicit non-decreasing target temperatures (at least two, all
    /// `> 0`), PLUMED's `TEMP_SET_ALL`.
    ///
    /// Excludes `temp_min`, `temp_max`, and `temp_steps`. Default: `None`.
    pub temp_set_all: Option<Vec<f64>>,
    /// Ladder spacing for `temp_steps` and automatic ladders.
    ///
    /// Default: [`TemperatureSpacing::Geometric`].
    pub spacing: TemperatureSpacing,
    /// MD steps between bias updates (`>= 1`), PLUMED's `PACE`.
    ///
    /// Default: [`DEFAULT_PACE`] (`500`).
    pub pace: usize,
    /// Unbiased `pace` samples collected before the first bias (`>= 1`),
    /// PLUMED's `OBSERVATION_STEPS`.
    ///
    /// Default: [`DEFAULT_OBSERVATION_STEPS`] (`100`).
    pub observation_steps: usize,
    /// Bias updates between recorded `Delta F` estimates (`>= 1`), PLUMED's
    /// `PRINT_STRIDE`.
    ///
    /// Default: [`DEFAULT_PRINT_STRIDE`] (`100`).
    pub print_stride: usize,
}

impl Default for OpesMultithermalConfig {
    fn default() -> Self {
        Self {
            temperature: None,
            temp_min: None,
            temp_max: None,
            temp_steps: None,
            temp_set_all: None,
            spacing: TemperatureSpacing::Geometric,
            pace: DEFAULT_PACE,
            observation_steps: DEFAULT_OBSERVATION_STEPS,
            print_stride: DEFAULT_PRINT_STRIDE,
        }
    }
}

with_setters!(OpesMultithermalConfig {
    with_temperature => temperature: opt f64,
    with_temp_min => temp_min: opt f64,
    with_temp_max => temp_max: opt f64,
    with_temp_steps => temp_steps: opt usize,
    with_temp_set_all => temp_set_all: opt Vec<f64>,
    with_spacing => spacing: val TemperatureSpacing,
    with_pace => pace: val usize,
    with_observation_steps => observation_steps: val usize,
    with_print_stride => print_stride: val usize,
});

impl OpesMultithermalConfig {
    /// Default configuration targeting `[temp_min, temp_max]`.
    #[must_use]
    pub fn range(temp_min: f64, temp_max: f64) -> Self {
        Self::default()
            .with_temp_min(temp_min)
            .with_temp_max(temp_max)
    }

    /// Default configuration targeting exactly `temperatures`.
    #[must_use]
    pub fn set_all(temperatures: Vec<f64>) -> Self {
        Self::default().with_temp_set_all(temperatures)
    }

    /// Check every field against its documented domain.
    pub fn validate(&self) -> Result<(), FluxError> {
        let positive = |value: Option<f64>, message: &'static str| match value {
            Some(t) if !t.is_finite() => Err(FluxError::NonFinite(message)),
            Some(t) if t <= 0.0 => Err(FluxError::InvalidConfig(message)),
            _ => Ok(()),
        };
        positive(
            self.temperature,
            "OpesMultithermalConfig::temperature must be finite and positive",
        )?;
        positive(
            self.temp_min,
            "OpesMultithermalConfig::temp_min must be finite and positive",
        )?;
        positive(
            self.temp_max,
            "OpesMultithermalConfig::temp_max must be finite and positive",
        )?;
        if self.pace == 0 {
            return Err(FluxError::InvalidConfig(
                "OpesMultithermalConfig::pace must be at least 1",
            ));
        }
        if self.observation_steps == 0 {
            return Err(FluxError::InvalidConfig(
                "OpesMultithermalConfig::observation_steps must be at least 1",
            ));
        }
        if self.print_stride == 0 {
            return Err(FluxError::InvalidConfig(
                "OpesMultithermalConfig::print_stride must be at least 1",
            ));
        }
        if self.temp_steps == Some(0) {
            return Err(FluxError::InvalidConfig(
                "OpesMultithermalConfig::temp_steps must be at least 1",
            ));
        }
        if let Some(temps) = &self.temp_set_all {
            if self.temp_min.is_some() || self.temp_max.is_some() || self.temp_steps.is_some() {
                return Err(FluxError::InvalidConfig(
                    "OpesMultithermalConfig::temp_set_all excludes temp_min, temp_max, and temp_steps",
                ));
            }
            if temps.len() < 2 {
                return Err(FluxError::InvalidConfig(
                    "OpesMultithermalConfig::temp_set_all needs at least 2 temperatures",
                ));
            }
            for &t in temps {
                positive(
                    Some(t),
                    "OpesMultithermalConfig::temp_set_all entries must be finite and positive",
                )?;
            }
            if temps.windows(2).any(|pair| pair[0] > pair[1]) {
                return Err(FluxError::InvalidConfig(
                    "OpesMultithermalConfig::temp_set_all must be non-decreasing",
                ));
            }
            return Ok(());
        }
        if self.temp_min.is_none() && self.temp_max.is_none() {
            return Err(FluxError::InvalidConfig(
                "OpesMultithermalConfig needs temp_min, temp_max, or temp_set_all",
            ));
        }
        if let (Some(lo), Some(hi)) = (self.temp_min, self.temp_max)
            && hi < lo
        {
            return Err(FluxError::InvalidConfig(
                "OpesMultithermalConfig::temp_max must not be below temp_min",
            ));
        }
        Ok(())
    }
}

/// Bias value and energy derivative at one potential energy.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct BiasEvaluation {
    /// Bias `V(U)` in energy units.
    pub bias: f64,
    /// Derivative `dV/dU` (dimensionless).
    pub derivative: f64,
}

impl BiasEvaluation {
    /// Factor `1 + dV/dU` that turns the unbiased force into the biased one.
    #[must_use]
    pub fn force_scale(&self) -> f64 {
        1.0 + self.derivative
    }
}

/// Effect of one [`OpesMultithermal::update`] call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum OpesUpdate {
    /// The step was skipped: the first step, or not a multiple of `pace`.
    Idle,
    /// An unbiased observation sample was stored.
    Observed,
    /// The last observation sample initialized the ladder and `Delta F`.
    Initialized,
    /// `Delta F` and `rct` absorbed one biased sample.
    Updated {
        /// `true` on the updates PLUMED prints to `DELTAFS`.
        record: bool,
    },
}

/// MD step position of a bias.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StepRecord {
    /// Built by [`OpesMultithermal::new`] and never updated.
    Fresh,
    /// Restored from a state without a step, such as a PLUMED state.
    Unknown,
    /// Last MD step passed to [`OpesMultithermal::update`].
    At(u64),
}

/// Multithermal OPES bias state: temperature ladder, `Delta F`, and counters.
#[derive(Clone, Debug, PartialEq)]
pub struct OpesMultithermal {
    config: OpesMultithermalConfig,
    boltzmann: f64,
    temperature: f64,
    kbt: f64,
    /// `beta_k - beta0` per target temperature.
    der: Vec<f64>,
    automatic: bool,
    delta_f: Vec<f64>,
    rct: f64,
    counter: u64,
    obs_steps: usize,
    observations: Vec<f64>,
    first_step: bool,
    step: StepRecord,
}

impl OpesMultithermal {
    /// Build an uninitialized bias at the simulation temperature `T0`.
    ///
    /// `k_B` comes from `thermostat`, and `T0` from
    /// [`OpesMultithermalConfig::temperature`] or the thermostat; only these
    /// two thermostat values are checked, and the thermostat temperature
    /// only when the configuration does not override it. A target
    /// temperature so far from `T0` that `beta_k - beta0`, or the
    /// temperature recovered from it, is not finite returns
    /// [`FluxError::NonFinite`], and so does a ladder whose spacing
    /// overflows between finite ends.
    pub fn new(
        config: OpesMultithermalConfig,
        thermostat: &ThermostatConfig,
    ) -> Result<Self, FluxError> {
        config.validate()?;
        let boltzmann = thermostat.boltzmann;
        if !boltzmann.is_finite() {
            return Err(FluxError::NonFinite("ThermostatConfig::boltzmann"));
        }
        if boltzmann <= 0.0 {
            return Err(FluxError::InvalidConfig(
                "ThermostatConfig::boltzmann must be positive",
            ));
        }
        let temperature = config.temperature.unwrap_or(thermostat.temperature);
        if !temperature.is_finite() {
            return Err(FluxError::NonFinite("OPES simulation temperature"));
        }
        if temperature <= 0.0 {
            return Err(FluxError::InvalidConfig(
                "OPES simulation temperature must be positive",
            ));
        }
        let kbt = boltzmann * temperature;
        // A subnormal k_B T0 is positive but has an infinite inverse.
        if !kbt.is_finite() || kbt <= 0.0 || !(1.0 / kbt).is_finite() {
            return Err(FluxError::NonFinite("OPES thermal energy k_B * T0"));
        }
        let (der, automatic) = initial_ladder(&config, temperature, kbt)?;
        let obs_steps = config.observation_steps;
        Ok(Self {
            config,
            boltzmann,
            temperature,
            kbt,
            der,
            automatic,
            delta_f: Vec::new(),
            rct: 0.0,
            counter: 0,
            obs_steps,
            observations: Vec::new(),
            first_step: true,
            step: StepRecord::Fresh,
        })
    }

    /// Configuration the bias was built from.
    pub fn config(&self) -> &OpesMultithermalConfig {
        &self.config
    }

    /// Simulation temperature `T0`.
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Boltzmann constant `k_B`.
    pub fn boltzmann(&self) -> f64 {
        self.boltzmann
    }

    /// Thermal energy `k_B T0`.
    pub fn kbt(&self) -> f64 {
        self.kbt
    }

    /// `true` once the ladder and `Delta F` are set and the bias acts.
    pub fn is_initialized(&self) -> bool {
        !self.delta_f.is_empty()
    }

    /// Target temperatures `T_k = T0 / (k_B T0 (beta_k - beta0) + 1)`.
    ///
    /// An automatic ladder is empty until initialization.
    pub fn temperatures(&self) -> Vec<f64> {
        if self.automatic {
            return Vec::new();
        }
        self.der
            .iter()
            .map(|&d| self.temperature / (d * self.kbt + 1.0))
            .collect()
    }

    /// Reweighting range `(T_min, T_max)` of the configured ladder.
    pub fn temperature_range(&self) -> (f64, f64) {
        let first = self.der[0];
        let last = self.der[self.der.len() - 1];
        (
            self.temperature / (first * self.kbt + 1.0),
            self.temperature / (last * self.kbt + 1.0),
        )
    }

    /// `Delta F(k)` in energy units, one per target temperature; empty
    /// before initialization.
    pub fn delta_f(&self) -> &[f64] {
        &self.delta_f
    }

    /// Running constant `rct = (1 / beta0) ln <exp(beta0 V)>`.
    pub fn rct(&self) -> f64 {
        self.rct
    }

    /// Samples absorbed into `Delta F`, counting the observation phase as one.
    pub fn counter(&self) -> u64 {
        self.counter
    }

    /// Last MD step passed to [`Self::update`], or the step stored in the
    /// state it was restored from; `None` for a new bias and for a state
    /// without a step, such as one written by PLUMED.
    pub fn last_step(&self) -> Option<u64> {
        match self.step {
            StepRecord::At(step) => Some(step),
            StepRecord::Fresh | StepRecord::Unknown => None,
        }
    }

    /// Bias `V(U)` and `dV/dU` at potential energy `energy`; zero before
    /// initialization.
    pub fn bias(&self, energy: f64) -> Result<BiasEvaluation, FluxError> {
        if !energy.is_finite() {
            return Err(FluxError::NonFinite("OPES potential energy"));
        }
        if self.delta_f.is_empty() {
            return Ok(BiasEvaluation {
                bias: 0.0,
                derivative: 0.0,
            });
        }
        let mut diff_max = -f64::MAX;
        for (d, df) in self.der.iter().zip(&self.delta_f) {
            let diff = -(d * energy) + df / self.kbt;
            if diff > diff_max {
                diff_max = diff;
            }
        }
        let mut sum = 0.0_f64;
        let mut der_sum = 0.0_f64;
        for (d, df) in self.der.iter().zip(&self.delta_f) {
            let add = (-(d * energy) + df / self.kbt - diff_max).exp();
            sum += add;
            der_sum -= d * add;
        }
        let bias = -self.kbt * (diff_max + (sum / self.delta_f.len() as f64).ln());
        let force = self.kbt * der_sum / sum;
        if !bias.is_finite() || !force.is_finite() {
            return Err(FluxError::NonFinite("OPES bias"));
        }
        Ok(BiasEvaluation {
            bias,
            derivative: -force,
        })
    }

    /// Feed the unbiased potential energy of MD step `step`.
    ///
    /// Mirrors PLUMED's `OPES_EXPANDED` update: the first call is skipped
    /// unless `observation_steps == 1`, only multiples of `pace` act, the
    /// first `observation_steps` of those are unbiased observations, and every
    /// later one absorbs the sample with the bias in force at that step.
    ///
    /// Steps must not decrease. A repeat of [`Self::last_step`] is idle, so
    /// a run continuing from the last step does not absorb that
    /// configuration twice; an earlier step returns
    /// [`FluxError::InvalidConfig`]. Finite inputs whose update overflows
    /// `rct` or `Delta F` return [`FluxError::NonFinite`]. An update that
    /// returns an error leaves the bias unchanged, including
    /// [`Self::last_step`], so the same step can be retried.
    pub fn update(&mut self, step: u64, energy: f64) -> Result<OpesUpdate, FluxError> {
        if !energy.is_finite() {
            return Err(FluxError::NonFinite("OPES potential energy"));
        }
        let repeat = match self.step {
            StepRecord::At(last) if step < last => {
                return Err(FluxError::InvalidConfig(
                    "OPES update step precedes the last step the bias absorbed",
                ));
            }
            StepRecord::At(last) => step == last,
            StepRecord::Fresh | StepRecord::Unknown => false,
        };
        let outcome = self.absorb(step, energy, repeat)?;
        self.first_step = false;
        self.step = StepRecord::At(step);
        Ok(outcome)
    }

    fn absorb(&mut self, step: u64, energy: f64, repeat: bool) -> Result<OpesUpdate, FluxError> {
        if self.first_step && self.obs_steps != 1 {
            return Ok(OpesUpdate::Idle);
        }
        if repeat || !step.is_multiple_of(self.config.pace as u64) {
            return Ok(OpesUpdate::Idle);
        }
        if self.obs_steps > 0 {
            if self.observations.len() + 1 == self.obs_steps {
                self.init_from_observations(energy)?;
                return Ok(OpesUpdate::Initialized);
            }
            self.observations.push(energy);
            return Ok(OpesUpdate::Observed);
        }
        let current = self.bias(energy)?.bias;
        self.update_delta_f(current, energy)?;
        let record = (self.counter - 1).is_multiple_of(self.config.print_stride as u64);
        Ok(OpesUpdate::Updated { record })
    }

    /// Fix the ladder and `Delta F` from the stored observations and `last`;
    /// on error nothing changes.
    fn init_from_observations(&mut self, last: f64) -> Result<(), FluxError> {
        let obs: Vec<f64> = self.observations.iter().copied().chain([last]).collect();
        let der = if self.automatic {
            let steps = estimate_num_steps(self.der[0], self.der[1], &obs)?;
            ladder_steps(
                self.der[0],
                self.der[1],
                steps,
                self.config.spacing,
                1.0 / self.kbt,
            )?
        } else {
            self.der.clone()
        };
        let kbt = self.kbt;
        let mut delta_f: Vec<f64> = der.iter().map(|d| kbt * (d * obs[0])).collect();
        for (t, &energy) in obs.iter().enumerate().skip(1) {
            let ln_t = (t as f64).ln();
            let shrink = (-1.0 / (1.0 + t as f64)).ln_1p();
            for (df, d) in delta_f.iter_mut().zip(&der) {
                let diff = -(d * energy) + *df / kbt - ln_t;
                if diff > 0.0 {
                    *df -= kbt * (diff + (-diff).exp().ln_1p() + shrink);
                } else {
                    *df -= kbt * (diff.exp().ln_1p() + shrink);
                }
            }
        }
        validate_finite_slice(
            ndarray::ArrayView1::from(&delta_f[..]),
            "OPES initial Delta F",
        )?;
        self.der = der;
        self.automatic = false;
        self.observations.clear();
        self.delta_f = delta_f;
        self.counter = 1;
        self.obs_steps = 0;
        Ok(())
    }

    /// Absorb one biased sample; on error nothing changes.
    fn update_delta_f(&mut self, bias: f64, energy: f64) -> Result<(), FluxError> {
        let counter = self.counter.checked_add(1).ok_or(FluxError::InvalidConfig(
            "OPES update counter overflows u64",
        ))?;
        let kbt = self.kbt;
        let ln_prev = (counter as f64 - 1.0).ln();
        let arg = (bias - self.rct) / kbt - ln_prev;
        let increment = if arg > 0.0 {
            kbt * (arg + (-arg).exp().ln_1p())
        } else {
            kbt * arg.exp().ln_1p()
        };
        let rct = self.rct + (increment + kbt * (-1.0 / counter as f64).ln_1p());
        if !rct.is_finite() {
            return Err(FluxError::NonFinite("OPES updated rct"));
        }
        let delta_f: Vec<f64> = self
            .delta_f
            .iter()
            .zip(&self.der)
            .map(|(&df, d)| {
                let diff = -(d * energy) + (bias - self.rct + df) / kbt - ln_prev;
                if diff > 0.0 {
                    df + (increment - kbt * (diff + (-diff).exp().ln_1p()))
                } else {
                    df + (increment - kbt * diff.exp().ln_1p())
                }
            })
            .collect();
        validate_finite_slice(
            ndarray::ArrayView1::from(&delta_f[..]),
            "OPES updated Delta F",
        )?;
        self.counter = counter;
        self.delta_f = delta_f;
        self.rct = rct;
        Ok(())
    }

    /// Write the PLUMED `STATE_WFILE` layout: `time`, `counter`, `rct`, and
    /// one `temperature DeltaF` row per target.
    ///
    /// Temperatures are written with six significant digits, as PLUMED
    /// labels them, so PLUMED can restart from the file. Comment lines
    /// `# rgflux <key> <value>` add `T0`, `k_B`, the reweighting range, and
    /// [`Self::last_step`] when known, which [`Self::from_state`] checks and
    /// restores. PLUMED skips comment lines, so the file stays a valid
    /// PLUMED state.
    pub fn write_state<W: Write>(&self, writer: &mut W, time: f64) -> Result<(), FluxError> {
        if !self.is_initialized() {
            return Err(FluxError::InvalidConfig(
                "OPES state is written only after initialization",
            ));
        }
        if !time.is_finite() {
            return Err(FluxError::NonFinite("OPES state time"));
        }
        let io = |err: std::io::Error| FluxError::Surface(format!("OPES state write: {err}"));
        writeln!(writer, "#! FIELDS {STATE_ENERGY_FIELD} DeltaF").map_err(io)?;
        writeln!(writer, "#! SET time  {time:e}").map_err(io)?;
        writeln!(writer, "#! SET counter  {}", self.counter).map_err(io)?;
        writeln!(writer, "#! SET rct  {:e}", self.rct).map_err(io)?;
        let (temp_min, temp_max) = self.temperature_range();
        writeln!(
            writer,
            "{STATE_META_PREFIX} temperature {:e}",
            self.temperature
        )
        .map_err(io)?;
        writeln!(writer, "{STATE_META_PREFIX} boltzmann {:e}", self.boltzmann).map_err(io)?;
        writeln!(writer, "{STATE_META_PREFIX} temp_min {temp_min:e}").map_err(io)?;
        writeln!(writer, "{STATE_META_PREFIX} temp_max {temp_max:e}").map_err(io)?;
        if let Some(step) = self.last_step() {
            writeln!(writer, "{STATE_META_PREFIX} step {step}").map_err(io)?;
        }
        for (t, df) in self.temperatures().iter().zip(&self.delta_f) {
            writeln!(writer, "  {} {df:e}", format_c_general(*t)).map_err(io)?;
        }
        Ok(())
    }

    /// Restart from a state written by [`Self::write_state`]; read a state
    /// written by PLUMED with [`Self::from_plumed_state`].
    ///
    /// `config` and `thermostat` must describe the same ladder: the count of
    /// rows fixes an automatic ladder, and every temperature label must match
    /// the one this configuration produces. The last state in a file holding
    /// several is read. As in PLUMED, the first [`Self::update`] after a
    /// restart is skipped. A `counter` of `0`, which no initialized bias
    /// has, or `u64::MAX`, which the next update would overflow, returns
    /// [`FluxError::InvalidConfig`].
    ///
    /// `Delta F` and `rct` hold only at the `T0` and `k_B` they were
    /// estimated at. A state from [`Self::write_state`] records both and the
    /// reweighting range, and a mismatch with `config` and `thermostat`
    /// beyond a relative `64 * f64::EPSILON` returns
    /// [`FluxError::InvalidConfig`]; so does a state with only some of these
    /// entries. The stored step becomes [`Self::last_step`], which
    /// [`run_opes_multithermal`] continues from. A PLUMED state carries
    /// none of these entries, and the six-digit labels alone do not fix
    /// `T0` or `k_B`, so it returns [`FluxError::InvalidConfig`] here; read
    /// it with [`Self::from_plumed_state`].
    pub fn from_state<R: BufRead>(
        config: OpesMultithermalConfig,
        thermostat: &ThermostatConfig,
        reader: R,
    ) -> Result<Self, FluxError> {
        Self::restore(config, thermostat, reader, None)
    }

    /// Restart from a PLUMED state, given the `T0` (`temperature`) and
    /// `k_B` (`boltzmann`) it was written at.
    ///
    /// A PLUMED `STATE_WFILE` records neither, so the caller attests them,
    /// for example from the `TEMP` and the units of the PLUMED input. Both
    /// must be finite and positive and match the `T0` and `k_B` of `config`
    /// and `thermostat` within a relative `64 * f64::EPSILON`; otherwise
    /// [`FluxError::InvalidConfig`] is returned. The ladder is then checked
    /// as in [`Self::from_state`]. A state that also carries the `rgflux`
    /// entries is checked against them too. Without a stored step
    /// [`Self::last_step`] is `None`, and a run from the bias needs
    /// [`OpesRunConfig::start_step`].
    pub fn from_plumed_state<R: BufRead>(
        config: OpesMultithermalConfig,
        thermostat: &ThermostatConfig,
        reader: R,
        temperature: f64,
        boltzmann: f64,
    ) -> Result<Self, FluxError> {
        for value in [temperature, boltzmann] {
            if !value.is_finite() {
                return Err(FluxError::NonFinite("attested OPES state T0 and k_B"));
            }
            if value <= 0.0 {
                return Err(FluxError::InvalidConfig(
                    "attested OPES state T0 and k_B must be positive",
                ));
            }
        }
        Self::restore(config, thermostat, reader, Some((temperature, boltzmann)))
    }

    fn restore<R: BufRead>(
        config: OpesMultithermalConfig,
        thermostat: &ThermostatConfig,
        reader: R,
        attested: Option<(f64, f64)>,
    ) -> Result<Self, FluxError> {
        let mut bias = Self::new(config, thermostat)?;
        let state = parse_state(reader)?;
        let meta = state.meta.thermal()?;
        let stored = match (&meta, attested) {
            (_, Some(pair)) => pair,
            (Some(meta), None) => (meta.temperature, meta.boltzmann),
            (None, None) => {
                return Err(FluxError::InvalidConfig(
                    "OPES state has no rgflux T0 and k_B entries; read a PLUMED state with from_plumed_state",
                ));
            }
        };
        if !same_value(stored.0, bias.temperature) {
            return Err(FluxError::InvalidConfig(
                "OPES state was written at a different simulation temperature T0",
            ));
        }
        if !same_value(stored.1, bias.boltzmann) {
            return Err(FluxError::InvalidConfig(
                "OPES state was written with a different Boltzmann constant k_B",
            ));
        }
        if let Some(meta) = &meta {
            if !same_value(meta.temperature, bias.temperature) {
                return Err(FluxError::InvalidConfig(
                    "OPES state was written at a different simulation temperature T0",
                ));
            }
            if !same_value(meta.boltzmann, bias.boltzmann) {
                return Err(FluxError::InvalidConfig(
                    "OPES state was written with a different Boltzmann constant k_B",
                ));
            }
        }
        if bias.automatic {
            bias.der = ladder_steps(
                bias.der[0],
                bias.der[1],
                state.labels.len(),
                bias.config.spacing,
                1.0 / bias.kbt,
            )?;
            bias.automatic = false;
        }
        let expected: Vec<String> = bias
            .temperatures()
            .iter()
            .map(|&t| format_c_general(t))
            .collect();
        if expected.len() != state.labels.len() {
            return Err(FluxError::InvalidConfig(
                "OPES state has a different number of temperatures than the configuration",
            ));
        }
        if expected != state.labels {
            return Err(FluxError::InvalidConfig(
                "OPES state temperatures differ from the configured ladder",
            ));
        }
        if let Some(meta) = &meta {
            let (temp_min, temp_max) = bias.temperature_range();
            if !same_value(meta.temp_min, temp_min) || !same_value(meta.temp_max, temp_max) {
                return Err(FluxError::InvalidConfig(
                    "OPES state was written for a different temperature range",
                ));
            }
        }
        bias.delta_f = state.delta_f;
        bias.counter = state.counter;
        bias.rct = state.rct;
        bias.obs_steps = 0;
        bias.step = state.meta.step.map_or(StepRecord::Unknown, StepRecord::At);
        Ok(bias)
    }
}

/// Prefix of the `rgflux` entries in a state file, a comment to PLUMED.
const STATE_META_PREFIX: &str = "# rgflux";

/// `rgflux` entries of one state block.
#[derive(Default)]
struct StateMeta {
    temperature: Option<f64>,
    boltzmann: Option<f64>,
    temp_min: Option<f64>,
    temp_max: Option<f64>,
    step: Option<u64>,
}

/// `T0`, `k_B`, and the reweighting range a state was written with.
struct ThermalMeta {
    temperature: f64,
    boltzmann: f64,
    temp_min: f64,
    temp_max: f64,
}

impl StateMeta {
    /// All four thermal entries, `None` when there are none, or an error
    /// when only some are present.
    fn thermal(&self) -> Result<Option<ThermalMeta>, FluxError> {
        match (
            self.temperature,
            self.boltzmann,
            self.temp_min,
            self.temp_max,
        ) {
            (Some(temperature), Some(boltzmann), Some(temp_min), Some(temp_max)) => {
                Ok(Some(ThermalMeta {
                    temperature,
                    boltzmann,
                    temp_min,
                    temp_max,
                }))
            }
            (None, None, None, None) => Ok(None),
            _ => Err(FluxError::InvalidConfig(
                "OPES state needs all of the rgflux temperature, boltzmann, temp_min, and temp_max entries",
            )),
        }
    }
}

/// State block being read.
#[derive(Default)]
struct StateBlock {
    counter: Option<u64>,
    rct: Option<f64>,
    labels: Vec<String>,
    delta_f: Vec<f64>,
    meta: StateMeta,
}

struct ParsedState {
    labels: Vec<String>,
    delta_f: Vec<f64>,
    counter: u64,
    rct: f64,
    meta: StateMeta,
}

fn same_value(stored: f64, configured: f64) -> bool {
    (stored - configured).abs() <= RANGE_REL_TOL * stored.abs().max(configured.abs())
}

fn parse_state<R: BufRead>(reader: R) -> Result<ParsedState, FluxError> {
    let mut last: Option<ParsedState> = None;
    let mut current: Option<StateBlock> = None;
    let finish = |block: StateBlock| match block {
        StateBlock {
            counter: Some(counter),
            rct: Some(rct),
            labels,
            delta_f,
            meta,
        } if !labels.is_empty() => Ok(ParsedState {
            labels,
            delta_f,
            counter,
            rct,
            meta,
        }),
        _ => Err(FluxError::InvalidConfig(
            "OPES state block lacks counter, rct, or DeltaF rows",
        )),
    };
    for line in reader.lines() {
        let line = line.map_err(|err| FluxError::Surface(format!("OPES state read: {err}")))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("#! FIELDS") {
            let fields: Vec<&str> = rest.split_whitespace().collect();
            if fields.len() != 2 || fields[1] != "DeltaF" {
                return Err(FluxError::InvalidConfig(
                    "OPES state FIELDS must be one temperature column and DeltaF",
                ));
            }
            if let Some(block) = current.take() {
                last = Some(finish(block)?);
            }
            current = Some(StateBlock::default());
            continue;
        }
        let Some(block) = current.as_mut() else {
            return Err(FluxError::InvalidConfig(
                "OPES state must start with a FIELDS header",
            ));
        };
        if let Some(rest) = trimmed.strip_prefix("#! SET") {
            let mut parts = rest.split_whitespace();
            let (Some(name), Some(value)) = (parts.next(), parts.next()) else {
                return Err(FluxError::InvalidConfig("malformed OPES state SET line"));
            };
            match name {
                "counter" => {
                    let counter: u64 = value.parse().map_err(|_| {
                        FluxError::InvalidConfig("OPES state counter is not an integer")
                    })?;
                    // An initialized bias has counted at least the
                    // observation phase, and the next update adds one.
                    if counter == 0 || counter == u64::MAX {
                        return Err(FluxError::InvalidConfig(
                            "OPES state counter must lie between 1 and u64::MAX - 1",
                        ));
                    }
                    block.counter = Some(counter);
                }
                "rct" => block.rct = Some(parse_finite(value, "OPES state rct")?),
                _ => {}
            }
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix(STATE_META_PREFIX) {
            let mut parts = rest.split_whitespace();
            let (Some(name), Some(value), None) = (parts.next(), parts.next(), parts.next()) else {
                return Err(FluxError::InvalidConfig("malformed OPES state rgflux line"));
            };
            let meta = &mut block.meta;
            match name {
                "temperature" => meta.temperature = Some(parse_finite(value, "OPES state T0")?),
                "boltzmann" => meta.boltzmann = Some(parse_finite(value, "OPES state k_B")?),
                "temp_min" => meta.temp_min = Some(parse_finite(value, "OPES state temp_min")?),
                "temp_max" => meta.temp_max = Some(parse_finite(value, "OPES state temp_max")?),
                "step" => {
                    meta.step = Some(value.parse().map_err(|_| {
                        FluxError::InvalidConfig("OPES state step is not an integer")
                    })?);
                }
                _ => {}
            }
            continue;
        }
        if trimmed.starts_with('#') {
            continue;
        }
        let mut parts = trimmed.split_whitespace();
        let (Some(label), Some(value), None) = (parts.next(), parts.next(), parts.next()) else {
            return Err(FluxError::InvalidConfig(
                "OPES state rows hold one temperature and one DeltaF",
            ));
        };
        block.labels.push(label.to_owned());
        block
            .delta_f
            .push(parse_finite(value, "OPES state DeltaF")?);
    }
    if let Some(block) = current.take() {
        last = Some(finish(block)?);
    }
    last.ok_or(FluxError::EmptyInput("OPES state file has no state"))
}

fn parse_finite(text: &str, context: &'static str) -> Result<f64, FluxError> {
    let value: f64 = text
        .parse()
        .map_err(|_| FluxError::InvalidConfig("OPES state value is not a number"))?;
    if !value.is_finite() {
        return Err(FluxError::NonFinite(context));
    }
    Ok(value)
}

/// `beta_k - beta0` ladder from the configuration, and whether the count
/// still has to be estimated from observations.
fn initial_ladder(
    config: &OpesMultithermalConfig,
    temp0: f64,
    kbt: f64,
) -> Result<(Vec<f64>, bool), FluxError> {
    // Valid temperatures far from T0 can overflow T0 / T or the division
    // by k_B T0, or lose T in the rounding of T0 / (der k_B T0 + 1).
    let der_of = |t: f64| {
        let der = (temp0 / t - 1.0) / kbt;
        let back = temp0 / (der * kbt + 1.0);
        if der.is_finite() && back.is_finite() && back > 0.0 {
            Ok(der)
        } else {
            Err(FluxError::NonFinite(
                "OPES target temperature gives a non-finite beta_k - beta0",
            ))
        }
    };
    if let Some(temps) = &config.temp_set_all {
        let der = temps
            .iter()
            .map(|&t| der_of(t))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok((der, false));
    }
    let temp_min = config.temp_min.unwrap_or(temp0);
    let temp_max = config.temp_max.unwrap_or(temp0);
    if temp_max < temp_min {
        return Err(FluxError::InvalidConfig(
            "OPES temp_max must not be below temp_min",
        ));
    }
    let lo = der_of(temp_min)?;
    let hi = der_of(temp_max)?;
    let steps = match config.temp_steps {
        Some(n) => Some(n),
        None if temp_min == temp_max => Some(1),
        None => None,
    };
    match steps {
        Some(n) => Ok((ladder_steps(lo, hi, n, config.spacing, 1.0 / kbt)?, false)),
        None => Ok((vec![lo, hi], true)),
    }
}

/// PLUMED's `ExpansionCVs::getSteps` over `lambda = beta - beta0` with
/// `shift = beta0`.
fn ladder_steps(
    lambda_min: f64,
    lambda_max: f64,
    steps: usize,
    spacing: TemperatureSpacing,
    shift: f64,
) -> Result<Vec<f64>, FluxError> {
    if steps == 0 {
        return Err(FluxError::InvalidConfig(
            "OPES temperature ladder needs at least one step",
        ));
    }
    if lambda_min == lambda_max && steps > 1 {
        return Err(FluxError::InvalidConfig(
            "OPES temp_steps > 1 needs temp_min != temp_max",
        ));
    }
    let last = (steps - 1) as f64;
    let ladder = match spacing {
        _ if steps == 1 => vec![(lambda_min + lambda_max) / 2.0],
        TemperatureSpacing::Geometric => {
            let lo = lambda_min + shift;
            let hi = lambda_max + shift;
            if lo <= 0.0 || hi <= 0.0 {
                return Err(FluxError::InvalidConfig(
                    "geometric OPES spacing needs positive inverse temperatures",
                ));
            }
            let (ln_lo, ln_hi) = (lo.ln(), hi.ln());
            (0..steps)
                .map(|k| (ln_lo + k as f64 * (ln_hi - ln_lo) / last).exp() - shift)
                .collect()
        }
        TemperatureSpacing::LinearInverse => (0..steps)
            .map(|k| lambda_min + k as f64 * (lambda_max - lambda_min) / last)
            .collect(),
    };
    // Finite ends can still overflow their sum, their difference, or
    // lambda + shift.
    validate_finite_slice(
        ndarray::ArrayView1::from(&ladder[..]),
        "OPES temperature ladder",
    )?;
    Ok(ladder)
}

/// PLUMED's `ExpansionCVs::estimateNumSteps`: the ladder spacing is the sum
/// of the half widths at which the Kish effective sample size of the
/// observations reweighted to the end temperatures halves.
fn estimate_num_steps(left: f64, right: f64, obs: &[f64]) -> Result<usize, FluxError> {
    if left == 0.0 && right == 0.0 {
        return Ok(1);
    }
    let mut left_hwhm = if left != 0.0 {
        neff_half_width(left, obs)?
    } else {
        0.0
    };
    let mut right_hwhm = if right != 0.0 {
        neff_half_width(right, obs)?
    } else {
        0.0
    };
    if left_hwhm == 0.0 {
        right_hwhm *= 2.0;
    }
    if right_hwhm == 0.0 {
        left_hwhm *= 2.0;
    }
    let spacing = left_hwhm + right_hwhm;
    if spacing == 0.0 {
        return Ok(2);
    }
    let steps = ((right - left).abs() / spacing).ceil();
    if !steps.is_finite() {
        return Err(FluxError::NonFinite("OPES automatic temperature steps"));
    }
    if steps < 2.0 {
        return Ok(2);
    }
    if steps > usize::MAX as f64 {
        return Err(FluxError::InvalidConfig(
            "OPES automatic temperature steps overflow",
        ));
    }
    Ok(steps as usize)
}

fn neff_fraction(delta: f64, obs: &[f64]) -> f64 {
    let shift = if delta < 0.0 {
        obs.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    } else {
        obs.iter().copied().fold(f64::INFINITY, f64::min)
    };
    let mut sum_w = 0.0_f64;
    let mut sum_w2 = 0.0_f64;
    for &u in obs {
        let w = (-delta * (u - shift)).exp();
        sum_w += w;
        sum_w2 += w * w;
    }
    sum_w * sum_w / sum_w2 / obs.len() as f64 - 0.5
}

fn neff_half_width(side: f64, obs: &[f64]) -> Result<f64, FluxError> {
    let mut a = 0.0_f64;
    let mut func_a = 0.5_f64;
    let mut b = side;
    let mut func_b = neff_fraction(side, obs);
    if !func_b.is_finite() {
        return Err(FluxError::NonFinite("OPES effective sample size"));
    }
    if func_b >= 0.0 {
        return Ok(0.0);
    }
    if b < 0.0 {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut func_a, &mut func_b);
    }
    let mut c = a;
    let mut func_c = func_a;
    let mut iterations = 0_usize;
    while func_c.abs() > NEFF_ROOT_TOL {
        if func_a * func_c > 0.0 {
            a = c;
            func_a = func_c;
        } else {
            b = c;
            func_b = func_c;
        }
        c = (a * func_b - b * func_a) / (func_b - func_a);
        func_c = neff_fraction(c, obs);
        iterations += 1;
        if !func_c.is_finite() || iterations > NEFF_ROOT_MAX_ITERATIONS {
            return Err(FluxError::InvalidConfig(
                "OPES automatic temperature spacing did not converge",
            ));
        }
    }
    Ok(c.abs())
}

/// Format like the C++ stream default (`%g` with six significant digits),
/// the form PLUMED uses for temperature labels.
fn format_c_general(value: f64) -> String {
    const PRECISION: i32 = 6;
    if value == 0.0 {
        return if value.is_sign_negative() { "-0" } else { "0" }.to_owned();
    }
    let sci = format!("{:.*e}", (PRECISION - 1) as usize, value);
    let (mantissa, exponent) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let strip = |text: &str| -> String {
        if text.contains('.') {
            text.trim_end_matches('0').trim_end_matches('.').to_owned()
        } else {
            text.to_owned()
        }
    };
    if !(-4..PRECISION).contains(&exponent) {
        let sign = if exponent < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", strip(mantissa), exponent.abs())
    } else {
        let decimals = (PRECISION - 1 - exponent) as usize;
        strip(&format!("{value:.decimals$}"))
    }
}

/// One evaluation of [`OpesMultithermalSurface`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct SurfaceEvaluation {
    /// Positions the surface was evaluated at.
    pub positions: Array1<f64>,
    /// Unbiased potential energy `U`.
    pub potential_energy: f64,
    /// Bias `V(U)` in force at the evaluation.
    pub bias: f64,
    /// Force factor `1 + dV/dU`.
    pub force_scale: f64,
    /// Biased energy `U + V(U)` returned to the engine.
    pub biased_energy: f64,
}

struct SharedState {
    bias: OpesMultithermal,
    last: Option<SurfaceEvaluation>,
}

/// Thread-safe handle shared by [`OpesMultithermalSurface`] and
/// [`run_opes_multithermal`]; clones refer to the same bias.
#[derive(Clone)]
pub struct SharedOpesMultithermal {
    inner: Arc<Mutex<SharedState>>,
}

impl std::fmt::Debug for SharedOpesMultithermal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedOpesMultithermal")
            .finish_non_exhaustive()
    }
}

impl SharedOpesMultithermal {
    /// Share `bias`.
    pub fn new(bias: OpesMultithermal) -> Self {
        Self {
            inner: Arc::new(Mutex::new(SharedState { bias, last: None })),
        }
    }

    fn lock(&self) -> Result<MutexGuard<'_, SharedState>, FluxError> {
        self.inner
            .lock()
            .map_err(|_| FluxError::InvalidConfig("OPES bias lock is poisoned"))
    }

    /// Copy of the current bias state.
    pub fn snapshot(&self) -> Result<OpesMultithermal, FluxError> {
        Ok(self.lock()?.bias.clone())
    }

    /// Run `f` on the shared bias.
    pub fn with_bias<R>(&self, f: impl FnOnce(&OpesMultithermal) -> R) -> Result<R, FluxError> {
        Ok(f(&self.lock()?.bias))
    }

    /// Run `f` on the shared bias mutably, for example to restore a state.
    pub fn with_bias_mut<R>(
        &self,
        f: impl FnOnce(&mut OpesMultithermal) -> R,
    ) -> Result<R, FluxError> {
        Ok(f(&mut self.lock()?.bias))
    }

    /// Most recent surface evaluation, if any.
    pub fn last_evaluation(&self) -> Result<Option<SurfaceEvaluation>, FluxError> {
        Ok(self.lock()?.last.clone())
    }
}

/// [`ForceSurface`] adding the multithermal OPES bias to an inner surface.
///
/// Returns `U + V(U)` and `(1 + dV/dU) F`, the exact negative gradient of the
/// biased energy, and stores the unbiased evaluation in the shared handle.
/// A finite `U` and `V(U)` whose sum, or a scaled force, is not finite
/// returns [`FluxError::NonFinite`] and stores nothing.
pub struct OpesMultithermalSurface<S> {
    inner: S,
    shared: SharedOpesMultithermal,
}

impl<S: ForceSurface> OpesMultithermalSurface<S> {
    /// Bias `inner` with the shared bias.
    pub fn new(inner: S, shared: SharedOpesMultithermal) -> Self {
        Self { inner, shared }
    }

    /// Borrow the unbiased surface.
    pub fn inner(&self) -> &S {
        &self.inner
    }

    /// Borrow the shared bias handle.
    pub fn shared(&self) -> &SharedOpesMultithermal {
        &self.shared
    }
}

impl<S: ForceSurface> ForceSurface for OpesMultithermalSurface<S> {
    fn energy_and_forces(
        &self,
        positions: ndarray::ArrayView1<'_, f64>,
        cell: Option<&[f64; 9]>,
    ) -> Result<(f64, Array1<f64>), FluxError> {
        let (energy, forces) = self.inner.energy_and_forces(positions, cell)?;
        let mut state = self.shared.lock()?;
        let eval = state.bias.bias(energy)?;
        let scale = eval.force_scale();
        let biased_energy = energy + eval.bias;
        if !biased_energy.is_finite() {
            return Err(FluxError::NonFinite("OPES biased energy"));
        }
        let biased_forces = forces.mapv(|f| scale * f);
        validate_finite_slice(biased_forces.view(), "OPES biased forces")?;
        state.last = Some(SurfaceEvaluation {
            positions: positions.to_owned(),
            potential_energy: energy,
            bias: eval.bias,
            force_scale: scale,
            biased_energy,
        });
        Ok((biased_energy, biased_forces))
    }
}

/// Integration and recording settings of [`run_opes_multithermal`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct OpesRunConfig {
    /// MD steps to integrate after the starting configuration.
    ///
    /// Default: `10_000`.
    pub steps: usize,
    /// Integration timestep `dt > 0` in engine time units.
    ///
    /// Default: `0.005`.
    pub time_step: f64,
    /// MD steps between stored frames (`>= 1`); the starting configuration
    /// is always the first frame.
    ///
    /// Default: `1`.
    pub frame_stride: usize,
    /// Store frame positions alongside the energies.
    ///
    /// Default: `true`.
    pub record_positions: bool,
    /// MD step index of the starting configuration.
    ///
    /// `None` continues from [`OpesMultithermal::last_step`] of the shared
    /// bias, which [`OpesMultithermal::from_state`] restores, and starts a
    /// new bias at step `0`. A value must equal that last step when the bias
    /// has one; otherwise [`run_opes_multithermal`] returns
    /// [`FluxError::InvalidConfig`] before integrating, since the `pace`
    /// cadence counts absolute steps. A bias restored from a state without
    /// a step, such as a PLUMED state, needs a value. The start step plus
    /// [`Self::steps`] must not exceed `u64::MAX`; otherwise the run returns
    /// [`FluxError::InvalidConfig`] before touching the engine.
    ///
    /// Default: `None`.
    pub start_step: Option<u64>,
}

impl Default for OpesRunConfig {
    fn default() -> Self {
        Self {
            steps: 10_000,
            time_step: 0.005,
            frame_stride: 1,
            record_positions: true,
            start_step: None,
        }
    }
}

with_setters!(OpesRunConfig {
    with_steps => steps: val usize,
    with_time_step => time_step: val f64,
    with_frame_stride => frame_stride: val usize,
    with_record_positions => record_positions: val bool,
    with_start_step => start_step: opt u64,
});

/// `Delta F` estimates at one recorded update.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct DeltaFRecord {
    /// MD step of the update.
    pub step: u64,
    /// `rct` after the update.
    pub rct: f64,
    /// `Delta F(k)` after the update, in energy units.
    pub delta_f: Vec<f64>,
}

/// Frames and final bias state of [`run_opes_multithermal`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct OpesTrajectory {
    /// MD step of each frame.
    pub steps: Vec<u64>,
    /// Unbiased potential energy `U_t` of each frame.
    pub potential_energies: Vec<f64>,
    /// Bias `V_t` in force at each frame, before that step's update.
    pub biases: Vec<f64>,
    /// Force factor `1 + dV/dU` at each frame.
    pub force_scales: Vec<f64>,
    /// Frame positions; empty when `record_positions` is `false`.
    pub positions: Vec<Array1<f64>>,
    /// `Delta F` at initialization and every `print_stride` updates.
    pub delta_f_history: Vec<DeltaFRecord>,
    /// Final bias state.
    pub bias: OpesMultithermal,
    /// MD step of the last integrated configuration.
    pub last_step: u64,
}

impl OpesTrajectory {
    /// Reweighting over every stored frame.
    pub fn reweighting(&self) -> Result<MultithermalReweighting, FluxError> {
        MultithermalReweighting::from_bias(
            &self.bias,
            self.potential_energies.clone(),
            self.biases.clone(),
        )
    }

    /// Normalized weights at `target_temperature`, one per stored frame.
    pub fn weights(&self, target_temperature: f64) -> Result<Vec<f64>, FluxError> {
        self.reweighting()?.weights(target_temperature)
    }

    /// Final `Delta F(k)` per target temperature.
    pub fn delta_f(&self) -> &[f64] {
        self.bias.delta_f()
    }

    /// Target temperatures of the final ladder.
    pub fn temperatures(&self) -> Vec<f64> {
        self.bias.temperatures()
    }
}

/// Run multithermal OPES dynamics on `engine` with forces from an
/// [`OpesMultithermalSurface`] sharing `shared`.
///
/// The starting configuration is MD step [`OpesRunConfig::start_step`], or
/// the last step of the shared bias when that is `None`; each of the
/// `steps` steps that follow is integrated with the bias in force, and then
/// its unbiased energy updates the bias, so the force of step `n` uses the
/// estimates before step `n`'s update, as in PLUMED. The engine thermostat
/// must sample the canonical ensemble at the bias temperature `T0`. Returns
/// [`FluxError::InvalidConfig`] when the engine forces do not come from the
/// shared surface, or when the start step disagrees with the bias.
pub fn run_opes_multithermal<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    shared: &SharedOpesMultithermal,
    config: &OpesRunConfig,
) -> Result<OpesTrajectory, FluxError> {
    if !config.time_step.is_finite() || config.time_step <= 0.0 {
        return Err(FluxError::InvalidConfig(
            "OpesRunConfig::time_step must be finite and positive",
        ));
    }
    if config.frame_stride == 0 {
        return Err(FluxError::InvalidConfig(
            "OpesRunConfig::frame_stride must be at least 1",
        ));
    }
    let engine_kbt = engine.thermostat().require_canonical_sampling()?;
    let bias_kbt = shared.with_bias(OpesMultithermal::kbt)?;
    if (engine_kbt - bias_kbt).abs() > RANGE_REL_TOL * bias_kbt {
        return Err(FluxError::InvalidConfig(
            "engine thermostat k_B T differs from the OPES temperature T0",
        ));
    }
    let start_step = match (config.start_step, shared.with_bias(|bias| bias.step)?) {
        (None, StepRecord::Fresh) => 0,
        (None, StepRecord::At(last)) => last,
        (None, StepRecord::Unknown) => {
            return Err(FluxError::InvalidConfig(
                "OPES state has no MD step; set OpesRunConfig::start_step",
            ));
        }
        (Some(step), StepRecord::At(last)) if step != last => {
            return Err(FluxError::InvalidConfig(
                "OpesRunConfig::start_step differs from the last MD step of the OPES bias",
            ));
        }
        (Some(step), _) => step,
    };
    let end_step = u64::try_from(config.steps)
        .ok()
        .and_then(|steps| start_step.checked_add(steps))
        .ok_or(FluxError::InvalidConfig(
            "OPES run would step past u64::MAX; lower OpesRunConfig::steps or start_step",
        ))?;

    let start = engine.positions().to_owned();
    engine.set_positions(start.view())?;
    let (biased_energy, _) = engine.energy_and_forces()?;

    let mut trajectory = OpesTrajectory {
        steps: Vec::new(),
        potential_energies: Vec::new(),
        biases: Vec::new(),
        force_scales: Vec::new(),
        positions: Vec::new(),
        delta_f_history: Vec::new(),
        bias: shared.snapshot()?,
        last_step: start_step,
    };
    let mut process = |engine: &E, step: u64, biased: f64| -> Result<(), FluxError> {
        let eval = matched_evaluation(engine, shared, biased)?;
        if (step - start_step).is_multiple_of(config.frame_stride as u64) {
            trajectory.steps.push(step);
            trajectory.potential_energies.push(eval.potential_energy);
            trajectory.biases.push(eval.bias);
            trajectory.force_scales.push(eval.force_scale);
            if config.record_positions {
                trajectory.positions.push(eval.positions.clone());
            }
        }
        let record = shared.with_bias_mut(|bias| {
            let outcome = bias.update(step, eval.potential_energy)?;
            let record = matches!(
                outcome,
                OpesUpdate::Initialized | OpesUpdate::Updated { record: true }
            );
            Ok::<_, FluxError>(record.then(|| DeltaFRecord {
                step,
                rct: bias.rct(),
                delta_f: bias.delta_f().to_vec(),
            }))
        })??;
        if let Some(record) = record {
            trajectory.delta_f_history.push(record);
        }
        Ok(())
    };

    process(engine, start_step, biased_energy)?;
    for previous in start_step..end_step {
        let report = engine.step(config.time_step)?;
        process(engine, previous + 1, report.potential_energy)?;
    }
    trajectory.bias = shared.snapshot()?;
    trajectory.last_step = end_step;
    Ok(trajectory)
}

fn matched_evaluation<E: DynamicsEngine + ?Sized>(
    engine: &E,
    shared: &SharedOpesMultithermal,
    biased_energy: f64,
) -> Result<SurfaceEvaluation, FluxError> {
    let not_routed =
        FluxError::InvalidConfig("engine forces do not come from the shared OPES surface");
    let Some(eval) = shared.last_evaluation()? else {
        return Err(not_routed);
    };
    let positions = engine.positions();
    let same_positions = eval.positions.len() == positions.len()
        && eval
            .positions
            .iter()
            .zip(positions.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits());
    if !same_positions || eval.biased_energy.to_bits() != biased_energy.to_bits() {
        return Err(not_routed);
    }
    Ok(eval)
}

/// Reweighted mean of one observable at a target temperature.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct ReweightedAverage {
    /// Self-normalized weighted mean `sum_t w_t A_t`.
    pub mean: f64,
    /// Batch-means standard error of the linearized ratio estimator.
    pub standard_error: f64,
    /// Kish effective sample size `(sum w)^2 / sum w^2` of the weights.
    pub effective_sample_size: f64,
}

/// Temperature reweighting of frames sampled under a multithermal OPES bias.
#[derive(Clone, Debug, PartialEq)]
pub struct MultithermalReweighting {
    potential_energies: Vec<f64>,
    biases: Vec<f64>,
    beta0: f64,
    boltzmann: f64,
    kbt: f64,
    temp_min: f64,
    temp_max: f64,
}

impl MultithermalReweighting {
    /// Reweight frames with unbiased energies `potential_energies` and
    /// in-force biases `biases`, simulated at `temperature` with Boltzmann
    /// constant `boltzmann`, to temperatures in `[temp_min, temp_max]`.
    ///
    /// Besides each value being finite and positive, `k_B T` and
    /// `1 / (k_B T)` must be finite and positive at `temperature`,
    /// `temp_min`, and `temp_max`; otherwise [`FluxError::NonFinite`] is
    /// returned instead of weights built on an invalid `beta`.
    pub fn new(
        potential_energies: Vec<f64>,
        biases: Vec<f64>,
        boltzmann: f64,
        temperature: f64,
        temp_min: f64,
        temp_max: f64,
    ) -> Result<Self, FluxError> {
        if potential_energies.is_empty() {
            return Err(FluxError::EmptyInput("reweighting frames are empty"));
        }
        if biases.len() != potential_energies.len() {
            return Err(FluxError::Shape {
                expected: potential_energies.len(),
                got: biases.len(),
                context: "reweighting biases",
            });
        }
        validate_finite_slice(
            ndarray::ArrayView1::from(&potential_energies[..]),
            "reweighting potential energies",
        )?;
        validate_finite_slice(ndarray::ArrayView1::from(&biases[..]), "reweighting biases")?;
        for value in [boltzmann, temperature, temp_min, temp_max] {
            if !value.is_finite() {
                return Err(FluxError::NonFinite("reweighting temperatures"));
            }
            if value <= 0.0 {
                return Err(FluxError::InvalidConfig(
                    "reweighting temperatures and k_B must be positive",
                ));
            }
        }
        if temp_max < temp_min {
            return Err(FluxError::InvalidConfig(
                "reweighting temp_max must not be below temp_min",
            ));
        }
        // Each factor can be finite and positive while the product
        // underflows to zero or overflows, and a tiny k_B T overflows beta.
        for t in [temperature, temp_min, temp_max] {
            let thermal = boltzmann * t;
            if !thermal.is_finite() || thermal <= 0.0 || !(1.0 / thermal).is_finite() {
                return Err(FluxError::NonFinite(
                    "reweighting k_B T and its inverse must be finite and positive",
                ));
            }
        }
        let kbt = boltzmann * temperature;
        Ok(Self {
            potential_energies,
            biases,
            beta0: 1.0 / kbt,
            boltzmann,
            kbt,
            temp_min,
            temp_max,
        })
    }

    /// Reweight frames recorded under `bias` to its ladder range.
    pub fn from_bias(
        bias: &OpesMultithermal,
        potential_energies: Vec<f64>,
        biases: Vec<f64>,
    ) -> Result<Self, FluxError> {
        let (temp_min, temp_max) = bias.temperature_range();
        Self::new(
            potential_energies,
            biases,
            bias.boltzmann(),
            bias.temperature(),
            temp_min,
            temp_max,
        )
    }

    /// Number of frames.
    pub fn len(&self) -> usize {
        self.potential_energies.len()
    }

    /// `true` when there are no frames; never for a constructed value.
    pub fn is_empty(&self) -> bool {
        self.potential_energies.is_empty()
    }

    /// Reweighting range `(T_min, T_max)`.
    pub fn temperature_range(&self) -> (f64, f64) {
        (self.temp_min, self.temp_max)
    }

    fn target_beta(&self, target: f64) -> Result<f64, FluxError> {
        if !target.is_finite() {
            return Err(FluxError::NonFinite("reweighting target temperature"));
        }
        let lo = self.temp_min * (1.0 - RANGE_REL_TOL);
        let hi = self.temp_max * (1.0 + RANGE_REL_TOL);
        if target < lo || target > hi {
            return Err(FluxError::InvalidConfig(
                "reweighting target temperature lies outside the OPES ladder",
            ));
        }
        let beta = 1.0 / (self.boltzmann * target);
        if !beta.is_finite() || beta <= 0.0 {
            return Err(FluxError::NonFinite(
                "reweighting target inverse temperature",
            ));
        }
        Ok(beta)
    }

    /// Unnormalized logarithmic weights `beta0 V_t + (beta0 - beta) U_t`,
    /// one per frame.
    ///
    /// Finite energies and biases can still overflow these products, so a
    /// non-finite log weight returns [`FluxError::NonFinite`].
    pub fn log_weights(&self, target_temperature: f64) -> Result<Vec<f64>, FluxError> {
        let beta = self.target_beta(target_temperature)?;
        let log_weights: Vec<f64> = self
            .potential_energies
            .iter()
            .zip(&self.biases)
            .map(|(&u, &v)| self.beta0 * v + (self.beta0 - beta) * u)
            .collect();
        validate_finite_slice(
            ndarray::ArrayView1::from(&log_weights[..]),
            "reweighting log weights",
        )?;
        Ok(log_weights)
    }

    /// Weights at `target_temperature` normalized to sum to one, aligned one
    /// to one with the frames.
    pub fn weights(&self, target_temperature: f64) -> Result<Vec<f64>, FluxError> {
        let mut weights = self.log_weights(target_temperature)?;
        let max = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0_f64;
        for w in weights.iter_mut() {
            *w = (*w - max).exp();
            sum += *w;
        }
        for w in weights.iter_mut() {
            *w /= sum;
        }
        validate_finite_slice(
            ndarray::ArrayView1::from(&weights[..]),
            "reweighting weights",
        )?;
        Ok(weights)
    }

    /// Kish effective sample size `(sum w)^2 / sum w^2` at the target.
    pub fn effective_sample_size(&self, target_temperature: f64) -> Result<f64, FluxError> {
        Ok(kish(&self.weights(target_temperature)?))
    }

    /// Reweighted mean of `observable` (one value per frame) at the target.
    ///
    /// The standard error is [`crate::stats::batch_means`] of
    /// `n w_t (A_t - mean)`, the linearization of the self-normalized ratio
    /// estimator, so frame correlation enters through the batches.
    pub fn average(
        &self,
        target_temperature: f64,
        observable: &[f64],
        stats: &StatsConfig,
    ) -> Result<ReweightedAverage, FluxError> {
        if observable.len() != self.len() {
            return Err(FluxError::Shape {
                expected: self.len(),
                got: observable.len(),
                context: "reweighted observable",
            });
        }
        validate_finite_slice(
            ndarray::ArrayView1::from(observable),
            "reweighted observable",
        )?;
        let weights = self.weights(target_temperature)?;
        let mean: f64 = weights.iter().zip(observable).map(|(w, a)| w * a).sum();
        let n = weights.len() as f64;
        let linearized: Vec<f64> = weights
            .iter()
            .zip(observable)
            .map(|(w, a)| n * w * (a - mean))
            .collect();
        let standard_error = batch_means(&linearized, stats)?.standard_error;
        Ok(ReweightedAverage {
            mean,
            standard_error,
            effective_sample_size: kish(&weights),
        })
    }

    /// `Delta F(T) = -(1 / beta0) ln <exp(-(beta - beta0) U)>_0` from the
    /// frames, in the energy units and convention of
    /// [`OpesMultithermal::delta_f`].
    pub fn delta_f(&self, target_temperature: f64) -> Result<f64, FluxError> {
        let target = self.log_weights(target_temperature)?;
        let reference: Vec<f64> = self.biases.iter().map(|&v| self.beta0 * v).collect();
        validate_finite_slice(
            ndarray::ArrayView1::from(&reference[..]),
            "reweighting log weights",
        )?;
        let delta_f = -self.kbt * (log_sum_exp(&target) - log_sum_exp(&reference));
        if !delta_f.is_finite() {
            return Err(FluxError::NonFinite("reweighting Delta F"));
        }
        Ok(delta_f)
    }
}

fn kish(weights: &[f64]) -> f64 {
    let sum: f64 = weights.iter().sum();
    let sum_sq: f64 = weights.iter().map(|w| w * w).sum();
    sum * sum / sum_sq
}

fn log_sum_exp(values: &[f64]) -> f64 {
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    max + values.iter().map(|v| (v - max).exp()).sum::<f64>().ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayView1;

    struct DoubleWell2D;

    impl rgsaddle::PointSurface for DoubleWell2D {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
            let a = x[0] * x[0] - 1.0;
            Ok((
                a * a + 0.5 * x[1] * x[1],
                Array1::from_vec(vec![4.0 * x[0] * a, x[1]]),
            ))
        }
    }

    fn thermostat(t: f64) -> ThermostatConfig {
        ThermostatConfig::default().with_temperature(t)
    }

    fn initialized_bias() -> OpesMultithermal {
        let config = OpesMultithermalConfig::range(0.5, 2.0)
            .with_temp_steps(5)
            .with_pace(1)
            .with_observation_steps(3);
        let mut bias = OpesMultithermal::new(config, &thermostat(0.8)).expect("bias");
        for (step, u) in [0.4, 0.1, 0.7, 0.2].into_iter().enumerate() {
            bias.update(step as u64, u).expect("update");
        }
        assert!(bias.is_initialized());
        bias
    }

    #[test]
    fn biased_force_is_the_negative_gradient_of_the_biased_energy() {
        let bias = initialized_bias();
        let shared = SharedOpesMultithermal::new(bias);
        let surface = OpesMultithermalSurface::new(DoubleWell2D, shared);
        let h = 1e-5;
        for point in [[-1.2, 0.3], [0.1, -0.7], [0.8, 1.4], [-0.4, 0.0]] {
            let x = Array1::from_vec(point.to_vec());
            let (_, force) = surface.energy_and_forces(x.view(), None).expect("eval");
            for i in 0..2 {
                let mut plus = x.clone();
                let mut minus = x.clone();
                plus[i] += h;
                minus[i] -= h;
                let (ep, _) = surface.energy_and_forces(plus.view(), None).expect("plus");
                let (em, _) = surface
                    .energy_and_forces(minus.view(), None)
                    .expect("minus");
                let fd = -(ep - em) / (2.0 * h);
                assert!(
                    (fd - force[i]).abs() <= 1e-8 * (1.0 + force[i].abs()),
                    "coordinate {i} at {point:?}: finite difference {fd}, force {}",
                    force[i]
                );
            }
        }
    }

    #[test]
    fn surface_refuses_a_biased_energy_that_overflows() {
        struct Flat(f64);
        impl rgsaddle::PointSurface for Flat {
            fn eval(
                &self,
                _: ArrayView1<f64>,
            ) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
                Ok((self.0, Array1::from_vec(vec![1.0])))
            }
        }
        // At the cold end k_B T0 (beta_k - beta0) = 0.6, so V(U) is about
        // 0.6 U for very negative U: U and V are finite and U + V is not.
        let energy = -1.5e308;
        let shared = SharedOpesMultithermal::new(initialized_bias());
        assert!(shared.with_bias(|b| b.bias(energy)).expect("lock").is_ok());
        let surface = OpesMultithermalSurface::new(Flat(energy), shared.clone());
        let x = Array1::from_vec(vec![0.0]);
        assert_eq!(
            surface.energy_and_forces(x.view(), None).err(),
            Some(FluxError::NonFinite("OPES biased energy"))
        );
        assert_eq!(shared.last_evaluation().expect("lock"), None);
    }

    #[test]
    fn bias_derivative_matches_finite_differences_and_mixture_identity() {
        let bias = initialized_bias();
        let beta0 = 1.0 / bias.kbt();
        let betas: Vec<f64> = bias.temperatures().iter().map(|t| 1.0 / t).collect();
        let h = 1e-6;
        for u in [-2.0, 0.0, 0.35, 1.7, 6.0] {
            let eval = bias.bias(u).expect("bias");
            let fd = (bias.bias(u + h).expect("plus").bias - bias.bias(u - h).expect("minus").bias)
                / (2.0 * h);
            assert!((fd - eval.derivative).abs() <= 1e-8, "u = {u}");
            let logits: Vec<f64> = bias
                .der
                .iter()
                .zip(bias.delta_f())
                .map(|(d, df)| -d * u + beta0 * df)
                .collect();
            let lse = log_sum_exp(&logits);
            let mixture: f64 = logits
                .iter()
                .zip(&betas)
                .map(|(l, b)| (l - lse).exp() * b / beta0)
                .sum();
            assert!((mixture - eval.force_scale()).abs() <= 1e-12, "u = {u}");
            let direct = -(lse - (logits.len() as f64).ln()) / beta0;
            assert!((direct - eval.bias).abs() <= 1e-12, "u = {u}");
        }
    }

    #[test]
    fn bias_is_zero_until_initialized_and_first_step_is_skipped() {
        let config = OpesMultithermalConfig::range(1.0, 2.0)
            .with_temp_steps(3)
            .with_pace(2)
            .with_observation_steps(2);
        let mut bias = OpesMultithermal::new(config, &thermostat(1.0)).expect("bias");
        assert_eq!(bias.bias(3.0).expect("bias").bias, 0.0);
        assert_eq!(bias.update(0, 1.0), Ok(OpesUpdate::Idle));
        assert_eq!(bias.update(1, 1.0), Ok(OpesUpdate::Idle));
        assert_eq!(bias.update(2, 1.0), Ok(OpesUpdate::Observed));
        assert_eq!(bias.update(4, 1.5), Ok(OpesUpdate::Initialized));
        assert_eq!(bias.counter(), 1);
        assert_eq!(
            bias.update(6, 1.2),
            Ok(OpesUpdate::Updated { record: false })
        );
        assert_eq!(bias.counter(), 2);
    }

    #[test]
    fn observation_and_update_recursions_match_direct_averages() {
        let config = OpesMultithermalConfig::range(0.6, 1.8)
            .with_temp_steps(4)
            .with_pace(1)
            .with_observation_steps(5)
            .with_print_stride(1);
        let t0 = 0.9;
        let mut bias = OpesMultithermal::new(config, &thermostat(t0)).expect("bias");
        let obs = [0.3, 1.1, -0.2, 0.8, 0.5];
        let later = [0.9, 0.05, 1.6, 0.4, 0.75, 1.2];
        let mut step = 0_u64;
        bias.update(step, 0.0).expect("skipped");
        for &u in &obs {
            step += 1;
            bias.update(step, u).expect("observe");
        }
        let beta0 = 1.0 / t0;
        let der = bias.der.clone();
        let mut samples: Vec<(f64, f64)> = vec![];
        for &u in &later {
            let v = bias.bias(u).expect("bias").bias;
            samples.push((u, v));
            step += 1;
            bias.update(step, u).expect("update");
        }
        for (k, d) in der.iter().enumerate() {
            let unbiased: Vec<f64> = obs.iter().map(|u| -d * u).collect();
            let init = -(log_sum_exp(&unbiased) - (obs.len() as f64).ln()) / beta0;
            let mut num = vec![-beta0 * init];
            let mut den = vec![0.0];
            for &(u, v) in &samples {
                num.push(beta0 * v - d * u);
                den.push(beta0 * v);
            }
            let direct = -(log_sum_exp(&num) - log_sum_exp(&den)) / beta0;
            assert!(
                (direct - bias.delta_f()[k]).abs() <= 1e-12,
                "k = {k}: direct {direct}, recursion {}",
                bias.delta_f()[k]
            );
        }
        let den: Vec<f64> = std::iter::once(0.0)
            .chain(samples.iter().map(|&(_, v)| beta0 * v))
            .collect();
        let rct = (log_sum_exp(&den) - (den.len() as f64).ln()) / beta0;
        assert!((rct - bias.rct()).abs() <= 1e-12);
    }

    #[test]
    fn ladders_follow_geometric_and_linear_inverse_spacing() {
        let geometric = OpesMultithermal::new(
            OpesMultithermalConfig::range(1.0, 8.0).with_temp_steps(4),
            &thermostat(2.0),
        )
        .expect("geometric");
        let temps = geometric.temperatures();
        for (got, want) in temps.iter().zip([1.0, 2.0, 4.0, 8.0]) {
            assert!((got - want).abs() <= 1e-12 * want);
        }
        let linear = OpesMultithermal::new(
            OpesMultithermalConfig::range(1.0, 4.0)
                .with_temp_steps(4)
                .with_spacing(TemperatureSpacing::LinearInverse),
            &thermostat(2.0),
        )
        .expect("linear");
        let betas: Vec<f64> = linear.temperatures().iter().map(|t| 1.0 / t).collect();
        for (got, want) in betas.iter().zip([1.0, 0.75, 0.5, 0.25]) {
            assert!((got - want).abs() <= 1e-12);
        }
        let single =
            OpesMultithermal::new(OpesMultithermalConfig::range(1.5, 1.5), &thermostat(1.5))
                .expect("single");
        assert_eq!(single.temperatures(), vec![1.5]);
    }

    #[test]
    fn automatic_ladder_uses_the_effective_sample_size_spacing() {
        let config = OpesMultithermalConfig::range(1.0, 3.0)
            .with_pace(1)
            .with_observation_steps(64);
        let mut bias = OpesMultithermal::new(config, &thermostat(1.0)).expect("bias");
        assert!(bias.temperatures().is_empty());
        bias.update(0, 0.0).expect("skip");
        for t in 0..64_u64 {
            let u = 20.0 + 3.0 * ((t as f64) * 0.7).sin();
            bias.update(t + 1, u).expect("observe");
        }
        let temps = bias.temperatures();
        assert!(temps.len() >= 2);
        assert!((temps[0] - 1.0).abs() <= 1e-12);
        assert!((temps[temps.len() - 1] - 3.0).abs() <= 1e-12);
        let ratio = temps[1] / temps[0];
        for pair in temps.windows(2) {
            assert!((pair[1] / pair[0] - ratio).abs() <= 1e-12);
        }
    }

    #[test]
    fn invalid_configurations_are_rejected() {
        let t = thermostat(1.0);
        let reject = |config: OpesMultithermalConfig| {
            assert!(matches!(
                OpesMultithermal::new(config, &t),
                Err(FluxError::InvalidConfig(_) | FluxError::NonFinite(_))
            ));
        };
        reject(OpesMultithermalConfig::default());
        reject(OpesMultithermalConfig::range(2.0, 1.0));
        reject(OpesMultithermalConfig::range(1.0, 2.0).with_pace(0));
        reject(OpesMultithermalConfig::range(1.0, 2.0).with_observation_steps(0));
        reject(OpesMultithermalConfig::range(1.0, 2.0).with_print_stride(0));
        reject(OpesMultithermalConfig::range(1.0, 2.0).with_temp_steps(0));
        reject(OpesMultithermalConfig::range(1.0, 1.0).with_temp_steps(3));
        reject(OpesMultithermalConfig::range(f64::NAN, 2.0));
        reject(OpesMultithermalConfig::set_all(vec![1.0]));
        reject(OpesMultithermalConfig::set_all(vec![2.0, 1.0]));
        reject(OpesMultithermalConfig::set_all(vec![1.0, 2.0]).with_temp_min(1.0));
        reject(OpesMultithermalConfig::range(1.0, 2.0).with_temperature(-1.0));
    }

    #[test]
    fn state_round_trips_and_rejects_a_different_ladder() {
        let bias = initialized_bias();
        let mut buffer = Vec::new();
        bias.write_state(&mut buffer, 12.5).expect("write");
        let text = String::from_utf8(buffer.clone()).expect("utf8");
        assert!(text.starts_with("#! FIELDS ecv.ene DeltaF\n"));
        let restored =
            OpesMultithermal::from_state(bias.config().clone(), &thermostat(0.8), &buffer[..])
                .expect("restore");
        assert_eq!(restored.delta_f(), bias.delta_f());
        assert_eq!(restored.rct().to_bits(), bias.rct().to_bits());
        assert_eq!(restored.counter(), bias.counter());
        assert_eq!(restored.temperatures(), bias.temperatures());
        let other = bias.config().clone().with_temp_max(3.0);
        assert!(OpesMultithermal::from_state(other, &thermostat(0.8), &buffer[..]).is_err());
        let automatic = bias.config().clone().with_temp_steps(None);
        let restored_auto = OpesMultithermal::from_state(automatic, &thermostat(0.8), &buffer[..])
            .expect("automatic restore");
        assert_eq!(restored_auto.temperatures().len(), 5);
        assert_eq!(restored.last_step(), Some(3));
        assert_eq!(restored_auto.last_step(), Some(3));
    }

    fn state_text(bias: &OpesMultithermal) -> String {
        let mut buffer = Vec::new();
        bias.write_state(&mut buffer, 0.0).expect("write");
        String::from_utf8(buffer).expect("utf8")
    }

    #[test]
    fn state_records_t0_boltzmann_range_and_step() {
        let bias = initialized_bias();
        let text = state_text(&bias);
        for line in [
            "# rgflux temperature 8e-1\n",
            "# rgflux boltzmann 1e0\n",
            "# rgflux temp_min 5e-1\n",
            "# rgflux temp_max 2e0\n",
            "# rgflux step 3\n",
        ] {
            assert!(text.contains(line), "missing {line:?} in\n{text}");
        }
    }

    #[test]
    fn state_rejects_a_changed_t0_boltzmann_or_range() {
        let bias = initialized_bias();
        let text = state_text(&bias);
        let config = bias.config().clone();
        let restore = |config: OpesMultithermalConfig, thermostat: &ThermostatConfig| {
            OpesMultithermal::from_state(config, thermostat, text.as_bytes())
        };
        assert!(restore(config.clone(), &thermostat(0.8)).is_ok());
        assert_eq!(
            restore(config.clone(), &thermostat(0.9)).err(),
            Some(FluxError::InvalidConfig(
                "OPES state was written at a different simulation temperature T0"
            ))
        );
        assert_eq!(
            restore(config.clone().with_temperature(0.9), &thermostat(0.8)).err(),
            Some(FluxError::InvalidConfig(
                "OPES state was written at a different simulation temperature T0"
            ))
        );
        let other_kb = thermostat(0.8).with_boltzmann(8.617_333_262e-5);
        assert_eq!(
            restore(config.clone(), &other_kb).err(),
            Some(FluxError::InvalidConfig(
                "OPES state was written with a different Boltzmann constant k_B"
            ))
        );
        // Same six-digit labels, different range.
        assert_eq!(
            restore(config.with_temp_max(2.000_000_1), &thermostat(0.8)).err(),
            Some(FluxError::InvalidConfig(
                "OPES state was written for a different temperature range"
            ))
        );
    }

    fn without_rgflux_entries(text: &str) -> String {
        text.lines()
            .filter(|line| !line.starts_with(STATE_META_PREFIX))
            .map(|line| format!("{line}\n"))
            .collect()
    }

    #[test]
    fn state_without_rgflux_entries_needs_attested_t0_and_boltzmann() {
        let bias = initialized_bias();
        let plumed = without_rgflux_entries(&state_text(&bias));
        let config = bias.config().clone();
        assert_eq!(
            OpesMultithermal::from_state(config.clone(), &thermostat(0.8), plumed.as_bytes()).err(),
            Some(FluxError::InvalidConfig(
                "OPES state has no rgflux T0 and k_B entries; read a PLUMED state with from_plumed_state"
            ))
        );
        // The labels match a thermostat at 0.9 as well, so only the
        // attestation tells the two apart.
        let attest = |thermostat: &ThermostatConfig, t0: f64, kb: f64| {
            OpesMultithermal::from_plumed_state(
                config.clone(),
                thermostat,
                plumed.as_bytes(),
                t0,
                kb,
            )
        };
        assert!(attest(&thermostat(0.9), 0.9, 1.0).is_ok());
        assert_eq!(
            attest(&thermostat(0.9), 0.8, 1.0).err(),
            Some(FluxError::InvalidConfig(
                "OPES state was written at a different simulation temperature T0"
            ))
        );
        let other_kb = thermostat(0.8).with_boltzmann(2.0);
        assert_eq!(
            attest(&other_kb, 0.8, 1.0).err(),
            Some(FluxError::InvalidConfig(
                "OPES state was written with a different Boltzmann constant k_B"
            ))
        );
        assert!(matches!(
            attest(&thermostat(0.8), f64::NAN, 1.0),
            Err(FluxError::NonFinite(_))
        ));
        assert!(matches!(
            attest(&thermostat(0.8), 0.8, 0.0),
            Err(FluxError::InvalidConfig(_))
        ));

        let restored = attest(&thermostat(0.8), 0.8, 1.0).expect("attested PLUMED state");
        assert_eq!(restored.delta_f(), bias.delta_f());
        assert_eq!(restored.rct().to_bits(), bias.rct().to_bits());
        assert_eq!(restored.counter(), bias.counter());
        assert_eq!(restored.last_step(), None);
        assert_eq!(restored.step, StepRecord::Unknown);

        let text = state_text(&bias);
        let with_entries = OpesMultithermal::from_plumed_state(
            config.clone(),
            &thermostat(0.8),
            text.as_bytes(),
            0.8,
            1.0,
        )
        .expect("attested rgflux state");
        assert_eq!(with_entries.last_step(), Some(3));
        assert!(matches!(
            OpesMultithermal::from_plumed_state(
                config.with_temperature(0.9),
                &thermostat(0.8),
                text.as_bytes(),
                0.9,
                1.0
            ),
            Err(FluxError::InvalidConfig(_))
        ));

        let partial: String = text
            .lines()
            .filter(|line| !line.starts_with("# rgflux boltzmann"))
            .map(|line| format!("{line}\n"))
            .collect();
        assert!(matches!(
            OpesMultithermal::from_state(
                bias.config().clone(),
                &thermostat(0.8),
                partial.as_bytes()
            ),
            Err(FluxError::InvalidConfig(_))
        ));
    }

    #[test]
    fn state_rejects_a_counter_of_zero_or_u64_max() {
        let bias = initialized_bias();
        let text = state_text(&bias);
        let counter_line = format!("#! SET counter  {}", bias.counter());
        assert!(text.contains(&counter_line));
        let with_counter =
            |counter: &str| text.replace(&counter_line, &format!("#! SET counter  {counter}"));
        let restore = |state: &str| {
            OpesMultithermal::from_state(bias.config().clone(), &thermostat(0.8), state.as_bytes())
        };
        for bad in ["0".to_owned(), u64::MAX.to_string()] {
            assert_eq!(
                restore(&with_counter(&bad)).err(),
                Some(FluxError::InvalidConfig(
                    "OPES state counter must lie between 1 and u64::MAX - 1"
                )),
                "counter {bad}"
            );
        }
        for good in [1, u64::MAX - 1] {
            let mut restored = restore(&with_counter(&good.to_string())).expect("counter in range");
            assert_eq!(restored.counter(), good);
            restored.update(4, 0.9).expect("skipped first update");
            assert_eq!(
                restored.update(5, 0.9),
                Ok(OpesUpdate::Updated {
                    record: (good % restored.config().print_stride as u64) == 0
                })
            );
            assert_eq!(restored.counter(), good + 1);
            assert!(restored.rct().is_finite());
            assert!(restored.delta_f().iter().all(|df| df.is_finite()));
        }
    }

    #[test]
    fn an_update_past_the_largest_counter_is_refused() {
        let mut bias = initialized_bias();
        bias.counter = u64::MAX;
        let before = bias.clone();
        assert_eq!(
            bias.update(4, 0.9),
            Err(FluxError::InvalidConfig(
                "OPES update counter overflows u64"
            ))
        );
        assert_eq!(bias, before);
        assert_eq!(bias.counter(), u64::MAX);
        assert_eq!(bias.last_step(), Some(3));
        // The refused step stays unabsorbed, so a retry once the counter
        // has room updates instead of idling as a repeat.
        bias.counter = u64::MAX - 1;
        assert_eq!(
            bias.update(4, 0.9),
            Ok(OpesUpdate::Updated {
                record: (u64::MAX - 1).is_multiple_of(bias.config().print_stride as u64)
            })
        );
        assert_eq!(bias.counter(), u64::MAX);
        assert_eq!(bias.last_step(), Some(4));
    }

    #[test]
    fn an_update_that_overflows_rct_changes_nothing() {
        let mut bias = initialized_bias();
        // A finite rct of -f64::MAX, which a state file can hold, overflows
        // (V - rct) / k_B T0, so rct and Delta F would both be non-finite.
        bias.rct = -f64::MAX;
        let before = bias.clone();
        assert_eq!(
            bias.update(4, 0.9),
            Err(FluxError::NonFinite("OPES updated rct"))
        );
        assert_eq!(bias, before);
        assert_eq!(bias.counter(), 1);
        assert_eq!(bias.last_step(), Some(3));
        // The refused step stays unabsorbed, so a retry is not a repeat.
        assert_eq!(
            bias.update(4, 0.9),
            Err(FluxError::NonFinite("OPES updated rct"))
        );
        assert_eq!(bias, before);
    }

    #[test]
    fn an_update_that_overflows_delta_f_alone_changes_nothing() {
        let mut bias = initialized_bias();
        bias.rct = -f64::MAX;
        let before = bias.clone();
        // At U = 1e308 the bias is finite and (V - rct) / k_B T0 = 1.5e308
        // is too, so rct stays finite, but adding -(beta_k - beta0) U at
        // the hot end of the ladder overflows Delta F there.
        assert!(bias.bias(1e308).is_ok());
        assert_eq!(
            bias.update(4, 1e308),
            Err(FluxError::NonFinite("OPES updated Delta F"))
        );
        assert_eq!(bias, before);
        assert_eq!(bias.counter(), 1);
        assert_eq!(bias.last_step(), Some(3));
        assert_eq!(
            bias.update(4, 1e308),
            Err(FluxError::NonFinite("OPES updated Delta F"))
        );
        assert_eq!(bias, before);
    }

    #[test]
    fn a_failed_initialization_keeps_the_observations_and_the_step() {
        let config = OpesMultithermalConfig::range(0.1, 2.0)
            .with_temp_steps(5)
            .with_pace(1)
            .with_observation_steps(2);
        let mut bias = OpesMultithermal::new(config, &thermostat(0.8)).expect("bias");
        assert_eq!(bias.update(0, 0.4), Ok(OpesUpdate::Idle));
        assert_eq!(bias.update(1, 0.1), Ok(OpesUpdate::Observed));
        // (beta_k - beta0) U overflows at T_k = 0.1, so the initial Delta F
        // is not finite.
        assert_eq!(
            bias.update(2, -1e308),
            Err(FluxError::NonFinite("OPES initial Delta F"))
        );
        assert!(!bias.is_initialized());
        assert_eq!(bias.last_step(), Some(1));
        assert_eq!(bias.update(2, 0.7), Ok(OpesUpdate::Initialized));
        let mut reference =
            OpesMultithermal::new(bias.config().clone(), &thermostat(0.8)).expect("bias");
        for (step, u) in [0.4, 0.1, 0.7].into_iter().enumerate() {
            reference.update(step as u64, u).expect("update");
        }
        assert_eq!(bias, reference);
    }

    #[test]
    fn new_rejects_a_subnormal_thermal_energy_with_an_infinite_inverse() {
        // k_B T0 = 1e-320 is positive but subnormal, so 1 / (k_B T0) = inf.
        let thermostat = ThermostatConfig::default()
            .with_boltzmann(1e-160)
            .with_temperature(1e-160);
        let config = OpesMultithermalConfig::range(0.5e-160, 2e-160).with_temp_steps(3);
        assert_eq!(
            OpesMultithermal::new(config.clone(), &thermostat).err(),
            Some(FluxError::NonFinite("OPES thermal energy k_B * T0"))
        );
        let finite = ThermostatConfig::default()
            .with_boltzmann(1e-140)
            .with_temperature(1e-160);
        let bias = OpesMultithermal::new(config, &finite).expect("k_B T0 = 1e-300");
        assert!(bias.temperatures().iter().all(|t| t.is_finite()));
    }

    #[test]
    fn new_checks_only_the_simulation_temperature_it_uses() {
        let config = OpesMultithermalConfig::range(0.5, 2.0).with_temp_steps(3);
        for unused in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
            let bias =
                OpesMultithermal::new(config.clone().with_temperature(0.8), &thermostat(unused))
                    .expect("the configured T0 replaces the thermostat temperature");
            assert_eq!(bias.temperature(), 0.8);
        }
        assert_eq!(
            OpesMultithermal::new(config.clone(), &thermostat(f64::NAN)).err(),
            Some(FluxError::NonFinite("OPES simulation temperature"))
        );
        assert!(matches!(
            OpesMultithermal::new(config.clone(), &thermostat(-1.0)),
            Err(FluxError::InvalidConfig(_))
        ));
        let bad_kb = thermostat(0.8).with_boltzmann(f64::NAN);
        assert_eq!(
            OpesMultithermal::new(config.with_temperature(0.8), &bad_kb).err(),
            Some(FluxError::NonFinite("ThermostatConfig::boltzmann"))
        );
    }

    #[test]
    fn new_rejects_a_target_temperature_whose_beta_offset_overflows() {
        let message = Some(FluxError::NonFinite(
            "OPES target temperature gives a non-finite beta_k - beta0",
        ));
        // k_B T0 = 1, so T0 / T = 1e310 overflows at T = 1e-10.
        let cold = ThermostatConfig::default()
            .with_boltzmann(1e-300)
            .with_temperature(1e300);
        for config in [
            OpesMultithermalConfig::range(1e-10, 1.0).with_temp_steps(3),
            OpesMultithermalConfig::range(1e-10, 1.0)
                .with_temp_steps(3)
                .with_spacing(TemperatureSpacing::LinearInverse),
            OpesMultithermalConfig::range(1e-10, 1.0),
            OpesMultithermalConfig::set_all(vec![1e-10, 1.0]),
        ] {
            assert_eq!(OpesMultithermal::new(config, &cold).err(), message);
        }
        // k_B T0 = 1, so T0 / T = 1e-310 rounds away in T0 / T - 1 and the
        // recovered temperature is infinite.
        let hot = ThermostatConfig::default()
            .with_boltzmann(1e300)
            .with_temperature(1e-300);
        assert_eq!(
            OpesMultithermal::new(OpesMultithermalConfig::set_all(vec![1.0, 1e10]), &hot).err(),
            message
        );
        let bias = OpesMultithermal::new(
            OpesMultithermalConfig::range(1e290, 1e300).with_temp_steps(3),
            &cold,
        )
        .expect("T0 / T = 1e10");
        let temperatures = bias.temperatures();
        assert!((temperatures[0] / 1e290 - 1.0).abs() <= 1e-12);
        assert!((temperatures[2] / 1e300 - 1.0).abs() <= 1e-12);
    }

    #[test]
    fn new_rejects_a_ladder_that_overflows_between_finite_ends() {
        // beta0 = 1e307 and beta_k - beta0 = 1.7e308 at T0 / 18 are finite,
        // but their sum, the sum of equal ends, and the difference to the
        // hot end near -1e307 are not.
        let thermostat = ThermostatConfig::default()
            .with_boltzmann(1e-307)
            .with_temperature(1.0);
        let cold = 1.0 / 18.0;
        for config in [
            OpesMultithermalConfig::range(cold, 1.0).with_temp_steps(3),
            OpesMultithermalConfig::range(cold, cold),
            OpesMultithermalConfig::range(cold, 1e10)
                .with_temp_steps(3)
                .with_spacing(TemperatureSpacing::LinearInverse),
        ] {
            assert_eq!(
                OpesMultithermal::new(config, &thermostat).err(),
                Some(FluxError::NonFinite("OPES temperature ladder"))
            );
        }
        let bias = OpesMultithermal::new(
            OpesMultithermalConfig::range(1.0 / 17.0, 1.0).with_temp_steps(3),
            &thermostat,
        )
        .expect("beta_k = 1.7e308 at T0 / 17");
        assert!(
            bias.temperatures()
                .iter()
                .all(|t| t.is_finite() && *t > 0.0)
        );
    }

    #[test]
    fn equal_range_ends_give_a_single_temperature() {
        let config = OpesMultithermalConfig::range(1.2, 1.2)
            .with_pace(1)
            .with_observation_steps(2);
        let mut bias = OpesMultithermal::new(config, &thermostat(0.8)).expect("bias");
        assert_eq!(bias.temperatures().len(), 1);
        for (step, u) in [0.4, 0.1, 0.7].into_iter().enumerate() {
            bias.update(step as u64, u).expect("update");
        }
        assert!(bias.is_initialized());
        assert_eq!(bias.delta_f().len(), 1);
        let temperatures = bias.temperatures();
        assert_eq!(temperatures.len(), 1);
        assert!((temperatures[0] - 1.2).abs() <= 1e-12);
    }

    #[test]
    fn updates_ignore_a_repeated_step_and_reject_an_earlier_one() {
        let mut bias = initialized_bias();
        let counter = bias.counter();
        let delta_f = bias.delta_f().to_vec();
        assert_eq!(bias.update(3, 0.9), Ok(OpesUpdate::Idle));
        assert_eq!(bias.counter(), counter);
        assert_eq!(bias.delta_f(), &delta_f[..]);
        assert!(matches!(
            bias.update(2, 0.9),
            Err(FluxError::InvalidConfig(_))
        ));
        assert_eq!(
            bias.update(4, 0.9),
            Ok(OpesUpdate::Updated { record: false })
        );
        assert_eq!(bias.last_step(), Some(4));
    }

    #[test]
    fn c_general_format_matches_the_stream_default() {
        for (value, text) in [
            (300.0, "300"),
            (0.8, "0.8"),
            (339.2899, "339.29"),
            (1234567.0, "1.23457e+06"),
            (0.0001, "0.0001"),
            (0.00001234, "1.234e-05"),
            (2.0, "2"),
            (999999.5, "1e+06"),
        ] {
            assert_eq!(format_c_general(value), text);
        }
    }

    #[test]
    fn reweighting_rejects_a_zero_infinite_or_uninvertible_thermal_energy() {
        let build = |kb: f64, t: f64, lo: f64, hi: f64| {
            MultithermalReweighting::new(vec![0.1, 0.5], vec![0.0, -0.2], kb, t, lo, hi)
        };
        let message = Some(FluxError::NonFinite(
            "reweighting k_B T and its inverse must be finite and positive",
        ));
        // k_B T underflows to zero, overflows to infinity, or is subnormal
        // with an infinite beta, at T0 or at an end of the range.
        assert_eq!(build(1e-200, 1e-200, 1e-200, 2e-200).err(), message);
        assert_eq!(build(1e200, 1e200, 1e200, 2e200).err(), message);
        assert_eq!(build(1e-160, 1e-160, 1e-160, 2e-160).err(), message);
        assert_eq!(build(1e-300, 1e10, 1e-20, 1e10).err(), message);
        assert_eq!(build(1e300, 1.0, 1.0, 1e10).err(), message);

        let scaled = build(1e-150, 1e150, 0.5e150, 2e150).expect("k_B T = 1");
        let reference = build(1.0, 1.0, 0.5, 2.0).expect("reference");
        for (a, b) in scaled
            .weights(1.5e150)
            .expect("scaled weights")
            .iter()
            .zip(reference.weights(1.5).expect("weights"))
        {
            assert!(a.is_finite());
            assert!((a - b).abs() <= 1e-12);
        }
    }

    #[test]
    fn weights_are_normalized_and_reject_targets_outside_the_ladder() {
        let rw = MultithermalReweighting::new(
            vec![0.1, 0.5, 1.2, 0.3],
            vec![0.0, -0.2, -0.6, -0.1],
            1.0,
            1.0,
            0.8,
            2.0,
        )
        .expect("reweighting");
        let w = rw.weights(1.4).expect("weights");
        assert_eq!(w.len(), 4);
        assert!((w.iter().sum::<f64>() - 1.0).abs() <= 1e-15);
        let lw = rw.log_weights(1.4).expect("log weights");
        let ratio = (lw[2] - lw[0]).exp();
        assert!((w[2] / w[0] - ratio).abs() <= 1e-12 * ratio);
        assert!(rw.weights(0.5).is_err());
        assert!(rw.weights(2.5).is_err());
        let ess = rw.effective_sample_size(1.0).expect("ess");
        assert!(ess > 1.0 && ess <= 4.0);
    }

    #[test]
    fn reweighting_refuses_log_weights_that_overflow() {
        // beta0 = 2, so beta0 V_t = 2e308 overflows for finite inputs.
        let rw =
            MultithermalReweighting::new(vec![0.0, 0.0], vec![1e308, 1e308], 1.0, 0.5, 0.4, 1.0)
                .expect("reweighting");
        let message = Some(FluxError::NonFinite("reweighting log weights"));
        assert_eq!(rw.log_weights(0.5).err(), message);
        assert_eq!(rw.weights(0.5).err(), message);
        assert_eq!(rw.effective_sample_size(0.5).err(), message);
        assert_eq!(rw.delta_f(0.5).err(), message);
        assert!(
            rw.average(0.5, &[1.0, 2.0], &StatsConfig::default())
                .is_err()
        );

        // (beta0 - beta) U_t overflows on its own at the cold end.
        let rw = MultithermalReweighting::new(vec![1e308, 0.0], vec![0.0, 0.0], 1.0, 1.0, 0.1, 2.0)
            .expect("reweighting");
        assert_eq!(rw.weights(0.1).err(), message);
        assert_eq!(rw.weights(1.0).expect("beta = beta0"), vec![0.5, 0.5]);

        // Log weights far apart but finite still normalize.
        let rw =
            MultithermalReweighting::new(vec![0.0, 0.0], vec![4e307, -4e307], 1.0, 1.0, 0.5, 2.0)
                .expect("reweighting");
        assert_eq!(rw.weights(1.0).expect("weights"), vec![1.0, 0.0]);
    }
}
