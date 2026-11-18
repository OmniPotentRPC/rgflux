# Changelog

All notable changes to `rgflux` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Workspace and CI scaffold mirroring `rgsaddle` (`edition = "2024"`, `rust-version = "1.88"`, pinned git dependencies on `rgsaddle` and `rgmin` `develop`, optional `rgpot` and `rgpot-session` features, Org-mode + Shibuya Sphinx documentation, and `towncrier` newsfragments).
- `engine::DynamicsEngine` object-safe trait, `ForceSurface` adapter for `rgsaddle::PointSurface` and `rgmin`/`eindir-core` objectives, `engine::native` BAOAB Langevin reference integrator with RATTLE and SHAKE hyperplane constraints, `rgpot` static PES wrapper, and `rgpot-session` skeleton over `rgpot`'s MD session ABI (`openmm`, `nblib`, `lammps`).
- `stats` batch-means standard error, Geyer initial-positive-sequence effective sample size (`n_eff`), and Geweke two-window stationarity diagnostics.
- `kappa::bennett_chandler` reactive-flux transmission coefficient estimator over recorded trajectories and generic `DynamicsEngine` shooting driver.
- Public `#[non_exhaustive]` configuration types and documented module layouts for `hyperplane`, `pafi`, `ohtst`, `kappa`, `ffs`, and `network`.
