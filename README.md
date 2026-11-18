# rgflux

`rgflux` computes rare-event kinetics on top of [`rgsaddle`](https://github.com/OmniPotentRPC/rgsaddle) (bands, minimum-mode and index-1 saddle searches) and [`rgmin`](https://github.com/OmniPotentRPC/rgmin) (first-order and Riemannian steppers), with [`rgpot`](https://github.com/OmniPotentRPC/rgpot) static potentials and MD sessions as optional features.

Kinetics code is engine-agnostic: trajectory samplers drive the `engine::DynamicsEngine` trait and static potential surfaces enter through `engine::ForceSurface` (implemented for every `rgsaddle::PointSurface`, `eindir_core::DifferentiableObjective<f64>`, and `rgpot` callback potential), while post-hoc estimators operate on plain trajectory slices.

## Modules

- `engine` — object-safe `DynamicsEngine` trait, `ForceSurface`, the in-process `engine::native` BAOAB Langevin + RATTLE/SHAKE reference backend, the optional `rgpot` static PES wrapper, and the optional `rgpot-session` adapter wrapping `rgpot`'s engine-agnostic MD session ABI (`openmm`, `nblib`, `lammps`).
- `stats` — batch means, Geyer effective sample size (`n_eff`), and Geweke two-window stationarity diagnostics.
- `kappa` and `kappa::bennett_chandler` — Bennett-Chandler reactive-flux transmission coefficient estimator over recorded dividing-surface velocity and side histories, plus generic trajectory shooting over any `DynamicsEngine`.
- `hyperplane` — holonomic hyperplane mean constraint force sampling with retained dividing-surface configurations; the torque and zero-variance control variates are not yet ported (*implementation moves from rgsaddle*).
- `pafi` — Projected Average Force Integration along reaction paths (*implementation moves from rgsaddle*).
- `ohtst` — OH-TST with a fixed hyperplane normal: the offset is placed on the stationary surface along the seed normal and the harmonic hyperplanar TST rate is reported; the normal is not variationally optimized.
- `ffs` — Forward Flux Sampling across ordered interfaces.
- `network` — symmetry-resolved kinetic rate networks and kinetic Monte Carlo rate tables (*implementation moves from rgsaddle*).

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
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --all-features
```

## License

MIT License. Copyright (c) 2026 Rohit Goswami.
