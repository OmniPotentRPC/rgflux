//! Rare-event kinetics over [`rgsaddle`] and [`rgmin`].
//!
//! `rgflux` separates rate theory and trajectory sampling from potential
//! energy surfaces and molecular dynamics codes. Surfaces enter through
//! [`rgsaddle::PointSurface`], [`eindir_core::DifferentiableObjective`], or
//! the optional `rgpot` feature, while samplers drive [`engine::DynamicsEngine`].
//!
//! ## Modules
//!
//! - [`engine`]: [`DynamicsEngine`] trait, [`ForceSurface`], the pure-Rust
//!   [`engine::native`] BAOAB + RATTLE/SHAKE backend, and optional `rgpot`
//!   and `rgpot-session` adapters.
//! - [`stats`]: non-overlapping batch means, Geyer effective sample size
//!   (`n_eff`), and Geweke two-window stationarity checks.
//! - [`kappa`] and [`kappa::bennett_chandler`]: Bennett-Chandler reactive-flux
//!   transmission coefficient estimator and generic [`DynamicsEngine`] shooter.
//! - [`hyperplane`]: holonomic hyperplane free-energy gradient sampling.
//! - [`pafi`]: Projected Average Force Integration along reaction paths.
//! - [`ohtst`]: OH-TST with a fixed hyperplane normal; the normal is not optimized.
//! - [`ffs`]: Forward Flux Sampling across ordered interfaces.
//! - [`network`]: symmetry-resolved kinetic networks and kMC rate tables.
//!
//! ## References
//!
//! - Goswami, R. (2025). *Efficient Exploration of Chemical Kinetics*.
//!   arXiv:2510.21368. <https://doi.org/10.48550/arXiv.2510.21368>
//! - Goswami, R., Gunde, M., and Jónsson, H. (2026). *Enhanced climbing image
//!   nudged elastic band method with Hessian eigenmode alignment*. Frontiers
//!   in Chemistry, 14, 1807063. <https://doi.org/10.3389/fchem.2026.1807063>

#![deny(missing_docs)]

#[macro_use]
mod config;

pub mod engine;
pub mod error;
pub mod ffs;
pub mod hyperplane;
pub mod kappa;
pub mod network;
pub mod ohtst;
pub mod pafi;
pub mod stats;

pub use engine::{
    DynamicsEngine, EindirSurface, ForceSurface, HyperplaneConstraint, StepReport,
    SurfaceVelocityPolicy, ThermostatConfig,
};
pub use error::FluxError;
