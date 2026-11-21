# Changelog

All notable changes to `rgflux` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-11-21

### Breaking

- `FreeEnergyProfile::reactant_partition`, `normalized_rate` and
  `normalized_rate_error` are `Option<f64>` instead of `f64`.
  `reactant_log_partition` and `normalized_log_rate` stay authoritative for every
  finite value, and a finite log stays on the profile.
  `reactant_partition` and `normalized_rate` are `None` when the exponential of
  that finite log overflows or underflows.
  `normalized_rate_error` holds `exp(normalized_log_rate + ln(sigma))`, where
  sigma is `barrier_error` divided by k_B T; it is `Some(0.0)` when sigma is zero
  and `None` only when that exponential overflows, underflows, or is not finite.
- `PafiConfig::zero_temperature_tol` is `Option<f64>` instead of `f64`.
  `None`, the default, scales the tolerance with `planes`.
  An explicit value is used at every plane count, including a value equal to the
  tolerance at the default plane count.

### Added

- `kappa::bennett_chandler::shoot_reactive_trajectories` and `shoot_bennett_chandler` support opt-in early commitment through `ShootingConfig::kappa`. Fixed-length shooting remains the default. Enabling the optimization requires an explicit positive checkpoint count and signed-distance threshold; invalid criteria return a typed configuration error.
- `OhtstConfig::reactant_fmax` sets the largest force component accepted at the OH-TST reactant minimum, default `1e-6`, so a reactant relaxed to the rounding floor of its forces gives a rate instead of stalling. The reactant is relaxed with the `rgmin` L-BFGS solver, and a relaxation that ends above the tolerance returns `FluxError::ReactantNotRelaxed`. `ohtst::relax_reactant` relaxes a reactant once, and `ohtst::optimize_hyperplane_from_minimum` evaluates an exit from a supplied `ReactantMinimum` after checking its force against the tolerance.

### Developer

- The PAFI rate standard-error cap now bounds only the lower tail of the
  Hermite barrier estimate, with a one sided normal quantile in place of the
  two sided one, so the cap meets its stated failure budget. The standard-error
  caps have opt-in fixed-seed sweeps that require the observed failure rate to
  stay within the per-assertion budget. Run
  `cargo test --features rgpot -- --ignored sweep` to execute all six: five
  native sweep tests and one rgpot sweep test.

### Fixed

- Hermite mean-force integration now rejects non-finite gradients with `FluxError::NonFinite`, matching `SimpsonIntegral` and preventing NaN or infinity values from producing a successful cumulative profile.
- PAFI integrates the plane means with the cubic Hermite rule and takes the barrier from the global maximum of that interpolant, including a maximum that is not next to the highest sample. Two planes reduce the rule to the trapezoid rule, three planes are exact for a quadratic gradient, and four or more planes are exact for a cubic gradient. `reactant_log_partition` and `normalized_log_rate` stay the log of the reactant partition and of the normalized rate, now `ln(sqrt(2 π kT) / ω_0)` and `ln(ω_0) - ln(2π) - ΔF / kT` from the reactant Hessian at the first image. `FreeEnergyProfile` also reports `dividing_arc`, `reactant_frequency`, and `quadrature_residual`. The zero-temperature check remains the trapezoid comparison and keeps its default tolerance.
- Report the PAFI truncation instead of a silent zero. `standard_error` and `barrier_error` combine the batch-means plane-force error with the absolute difference between the cubic Hermite integral and the trapezoid rule on the same nodes. `MeanForceIntegral` is the seam for another deterministic rule; `SimpsonIntegral` is composite Simpson and rejects an even plane count rather than falling back. There is no in-crate Gaussian process.
- The free-energy `standard_error` and `barrier_error` are a scaled
  Euclidean norm of the weighted plane uncertainties. A finite uncertainty
  whose square overflows stays finite.

## [0.1.0] - 2026-11-19

### Added

- Workspace and CI scaffold mirroring `rgsaddle` (`edition = "2024"`, `rust-version = "1.88"`, optional `rgpot`, `rgpot-session`, and `lammpc` features, Org-mode + Shibuya Sphinx documentation, and `towncrier` newsfragments).
- `engine::DynamicsEngine` object-safe trait, `ForceSurface` adapter for `rgsaddle::PointSurface` and `rgmin`/`eindir-core` objectives, `engine::native` BAOAB Langevin reference integrator with RATTLE and SHAKE hyperplane constraints, `engine::lammpc` direct LAMMPS backend over `lammpc_session_*`, `rgpot` static PES wrapper, and `rgpot-session` skeleton over `rgpot`'s MD session ABI (`openmm`, `nblib`, `lammps`).
- `stats` batch-means standard error, Geyer initial-positive-sequence effective sample size (`n_eff`), and Geweke two-window stationarity diagnostics.
- `kappa::bennett_chandler` reactive-flux transmission coefficient estimator over recorded trajectories and generic `DynamicsEngine` shooting driver, plus `kappa::kramers_transmission_coefficient`.
- `hyperplane::sample_hyperplane` holonomic hyperplane mean constraint force (`<lambda> = dA/dc`) and dividing-surface configuration sampler.
- `ohtst::optimize_hyperplane` fixed-normal dividing-surface placement and harmonic hyperplanar TST rate estimator with delta-method uncertainty propagation.
- `ffs::forward_flux_rate` direct Forward Flux Sampling transition rate estimator across ordered reaction-coordinate interfaces with batch-means initial flux uncertainty and cluster-variance stage probabilities.
- `pafi::pafi_profile` and `pafi::pafi_profile_dyn` free-energy profile, barrier, dividing plane, and normalized rate along a reaction path. Planes are orthogonal to a mass-metric spline through the images, with a zero-temperature path-force check, optional adaptive rounds to a target error, propagated standard errors, and engine restoration on return.
- Public `#[non_exhaustive]` configuration types and documented module layouts for `hyperplane`, `pafi`, `ohtst`, `kappa`, `ffs`, and `network`.

### Changed

- Depend on the crates.io releases of `rgsaddle` 0.5.0, `rgmin` 0.6.0, and `rgpot-core` 3.7.0 with caret requirements, so downstream builds can unify patch releases. Only the `lammpc` feature enables the `rgpot-core` `profile` and `schema` features, so `rgpot` and `rgpot-session` build without the `capnp` CLI.
- The value stored by default in `PafiConfig::zero_temperature_tol` is about 2.6% instead of 1%. The check divides the path-force integral discrepancy by the larger of one engine energy unit and the endpoint energy change magnitude. While the field retains its default value, the effective tolerance scales as `2 (pi^2 / 12) / (planes - 1)^2`; an explicitly different value is used unchanged.
- `cargo test` builds with `opt-level = 2` through `[profile.test]`, and the curved PAFI integration test compares the last free-energy increment with the one from a longer path, so natural spline endpoints fail it. It takes 0.7 s instead of 44 s.

### Fixed

- PAFI mass-metric projection, spline geometry, adaptive sampling, profile integration, uncertainty propagation, and engine-state restoration.
- `ffs::forward_flux_rate` restores positions, velocities, the periodic cell, the thermostat, and the holonomic constraint on return via `EngineSnapshot`, leaving the stochastic stream where the flux and trial reseeds placed it. `ohtst::optimize_hyperplane` delegates the flux mass to `rgsaddle::flux_mass`, keeping length mismatches on `FluxError::Shape` and non-positive or non-finite masses on `FluxError::InvalidConfig`.
- `stats::batch_means` returns the first sample as the mean of a series that is constant up to rounding (`SERIES_FLAT_TOL`), avoiding floating-point summation drift on repeated non-dyadic values.
- Compute the PAFI reactant partition in log space with shifted log-sum-exp trapezoid quadrature so deep free-energy wells no longer overflow the partition integral. Propagate `ln Z_R` through normalized rate evaluation and uncertainty propagation while retaining legacy linear-rate and partition fields for compatibility.
