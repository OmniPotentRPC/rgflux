# rgflux

`rgflux` computes rare-event kinetics on top of [`rgsaddle`](https://github.com/OmniPotentRPC/rgsaddle) (bands, minimum-mode and index-1 saddle searches) and [`rgmin`](https://github.com/OmniPotentRPC/rgmin) (first-order and Riemannian steppers), with [`rgpot`](https://github.com/OmniPotentRPC/rgpot) static potentials, MD sessions, and direct `lammpc` sessions as optional features.

Kinetics code is engine-agnostic: trajectory samplers drive the `engine::DynamicsEngine` trait and static potential surfaces enter through `engine::ForceSurface` (implemented for every `rgsaddle::PointSurface`, `eindir_core::DifferentiableObjective<f64>` via `engine::EindirSurface`, and `rgpot` callback potential via `engine::rgpot::RgpotSurface`), while post-hoc estimators operate on plain trajectory slices.

## Install

```sh
cargo add rgflux
```

Optional Cargo features (for example `cargo add rgflux --features rgpot`):

- `rgpot`: enables `engine::rgpot::RgpotSurface` and `RgpotConfig` wrapping `rgpot-core` DLPack potential callbacks.
- `rgpot-session`: enables `engine::rgpot_session::RgpotSessionEngine`, `RgpotSessionConfig`, and `SessionEngineKind` (`openmm`, `nblib`, `lammps`).
- `lammpc`: enables `engine::lammpc::LammpcEngine`, `LammpcConfig`, and `MASS_VELOCITY_TO_EV` over `liblammpc` (`lammpc_session_*`), and needs the `capnp` command-line tool at build time.

## Modules

- `engine`: object-safe `DynamicsEngine` trait, `ForceSurface`, `EindirSurface`, `HyperplaneConstraint`, `ThermostatConfig`, `SurfaceVelocityPolicy`, `StepReport`, the in-process `engine::native` BAOAB Langevin + RATTLE/SHAKE reference backend, the optional `rgpot` static PES wrapper, the optional direct `engine::lammpc` backend over `lammpc_session_*`, and the optional `rgpot-session` adapter wrapping `rgpot`'s engine-agnostic MD session ABI (`openmm`, `nblib`, `lammps`).
- `stats`: non-overlapping batch means, Geyer initial-monotone-sequence effective sample size (`n_eff`), and Geweke two-window stationarity diagnostics.
- `kappa` and `kappa::bennett_chandler`: Bennett-Chandler reactive-flux transmission coefficient estimator over recorded dividing-surface velocity and side histories, generic trajectory shooting over any `DynamicsEngine`, and Kramers' spatial-diffusion barrier-recrossing factor (`kramers_transmission_coefficient`).
- `hyperplane`: holonomic hyperplane mean constraint force (`<lambda> = dA/dc`) and dividing-surface configuration sampling (`sample_hyperplane`); the in-plane virial, rotational torque, and zero-variance control variates are not yet implemented.
- `pafi`: Projected Average Force Integration along reaction paths (`PafiConfig`, `FreeEnergyProfile`, `pafi_profile`, `pafi_profile_dyn`). Samples hyperplanes orthogonal to a mass-metric spline through the path images, integrates the mean force with a cubic Hermite rule into a free-energy profile, and returns the barrier, the dividing plane and the normalized rate from the projected reactant frequency, with standard errors, after a zero-temperature check of the path force.
- `ohtst`: OH-TST with a fixed hyperplane normal (`optimize_hyperplane`). Relaxes the reactant minimum with `rgmin` L-BFGS to the physical force tolerance `OhtstConfig::reactant_fmax` (or takes a relaxed `ReactantMinimum` through `optimize_hyperplane_from_minimum`), brackets and bisects the stationary dividing hyperplane along the seed normal, samples the curvature probe plane, and evaluates the harmonic hyperplanar TST rate with `rgsaddle::flux_mass` and a delta-method standard error; variational normal optimization is not yet implemented.
- `ffs`: direct Forward Flux Sampling (`forward_flux_rate`) across ordered reaction-coordinate interfaces with batch-means initial escape flux uncertainty, cluster-variance stage probabilities, and full `EngineSnapshot` state restoration.
- `network`: symmetry-resolved kinetic rate networks and kinetic Monte Carlo rate tables (`NetworkConfig`, `RateTableConfig`, `RateEdge`, `build_rate_table`). `RateEdge::new` and the configuration types are implemented; `build_rate_table` is not yet implemented and returns `FluxError::Unimplemented`.

## Quick example

```rust
use rgflux::kappa::bennett_chandler::{
    BennettChandlerConfig, ReactiveTrajectory, Side, estimate_bennett_chandler,
};

fn main() -> Result<(), rgflux::FluxError> {
    let trajectories = vec![
        ReactiveTrajectory::new(1.2, vec![Side::Product; 16])?,
        ReactiveTrajectory::new(-0.9, vec![Side::Reactant; 16])?,
        ReactiveTrajectory::new(0.8, vec![Side::Product; 16])?,
    ];
    let estimate = estimate_bennett_chandler(&trajectories, &BennettChandlerConfig::default())?;
    assert_eq!(estimate.plateau, 1.0);
    Ok(())
}
```

## Development checks

```sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets --features rgpot,rgpot-session -- -D warnings
cargo test --locked --features rgpot,rgpot-session
ci/check_rustdoc.sh
python3 docs/check_paths.py
python3 ci/check_dependency_pins.py
emacs --batch -l docs/export.el
sphinx-build -b html -n -W --keep-going docs/source docs/build
```

## License

MIT License. Copyright (c) 2026 Rohit Goswami.
