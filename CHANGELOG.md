# Changelog

All notable changes to `rgflux` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Workspace and CI scaffold mirroring `rgsaddle` (`edition = "2024"`, `rust-version = "1.88"`, pinned git dependencies on `rgsaddle` and `rgmin` `develop`, optional `rgpot`, `rgpot-session`, and `lammpc` features, Org-mode + Shibuya Sphinx documentation, and `towncrier` newsfragments).
- `engine::DynamicsEngine` object-safe trait, `ForceSurface` adapter for `rgsaddle::PointSurface` and `rgmin`/`eindir-core` objectives, `engine::native` BAOAB Langevin reference integrator with RATTLE and SHAKE hyperplane constraints, `engine::lammpc` direct LAMMPS backend over `lammpc_session_*`, `rgpot` static PES wrapper, and `rgpot-session` skeleton over `rgpot`'s MD session ABI (`openmm`, `nblib`, `lammps`).
- `stats` batch-means standard error, Geyer initial-positive-sequence effective sample size (`n_eff`), and Geweke two-window stationarity diagnostics.
- `kappa::bennett_chandler` reactive-flux transmission coefficient estimator over recorded trajectories and generic `DynamicsEngine` shooting driver, plus `kappa::kramers_transmission_coefficient`.
- `hyperplane::sample_hyperplane` holonomic hyperplane mean constraint force (`<lambda> = dA/dc`) and dividing-surface configuration sampler.
- `ohtst::optimize_hyperplane` fixed-normal dividing-surface placement and harmonic hyperplanar TST rate estimator with delta-method uncertainty propagation.
- `ffs::forward_flux_rate` direct Forward Flux Sampling transition rate estimator across ordered reaction-coordinate interfaces with batch-means initial flux uncertainty and cluster-variance stage probabilities.
- `pafi::pafi_profile` and `pafi::pafi_profile_dyn` free-energy profile, barrier, dividing plane, and normalized rate along a reaction path. Planes are orthogonal to a mass-metric spline through the images, with a zero-temperature path-force check, optional adaptive rounds to a target error, propagated standard errors, and engine restoration on return.
- Public `#[non_exhaustive]` configuration types and documented module layouts for `hyperplane`, `pafi`, `ohtst`, `kappa`, `ffs`, and `network`.

### Changed

- The default `PafiConfig::zero_temperature_tol` is about 2.6% instead of 1%. The trapezoid error of the zero-temperature check falls as `(planes - 1)^-2` and is 1.29% on a quarter-turn circular arc at the default 9 planes, so the new default is twice that value.
- `cargo test` builds with `opt-level = 2` through `[profile.test]`, and the curved PAFI integration test compares the last free-energy increment with the one from a longer path, so natural spline endpoints fail it. It takes 0.7 s instead of 44 s.

### Fixed

- PAFI mass-metric projection, spline geometry, adaptive sampling, profile integration, uncertainty propagation, and engine-state restoration.

- `ffs::forward_flux_rate` restores positions, velocities, the periodic cell, the thermostat, and the holonomic constraint on return via `EngineSnapshot`, leaving the stochastic stream where the flux and trial reseeds placed it. `ohtst::optimize_hyperplane` delegates the flux mass to `rgsaddle::flux_mass`, keeping length mismatches on `FluxError::Shape` and non-positive or non-finite masses on `FluxError::InvalidConfig`.
- `stats::batch_means` returns the first sample as the mean of a series that is constant up to rounding (`SERIES_FLAT_TOL`), avoiding floating-point summation drift on repeated non-dyadic values.
