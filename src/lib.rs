//! Rare-event kinetics over [`rgsaddle`] and [`rgmin`].
//!
//! `rgflux` separates rate theory and trajectory sampling from potential
//! energy surfaces and molecular dynamics codes. Surfaces enter through
//! [`rgsaddle::PointSurface`], [`eindir_core::DifferentiableObjective`], or
//! the optional `rgpot` feature, while samplers drive [`engine::DynamicsEngine`].
//!
//! ## Features
//!
//! - `rgpot`: static potential surfaces over `rgpot-core` callbacks.
//! - `rgpot-session`: engine-agnostic MD session adapter over `rgpot-core`.
//! - `lammpc`: direct LAMMPS engine, `engine::lammpc::LammpcEngine`, over
//!   `liblammpc`. It needs the `capnp` CLI at build time.
//!
//! ## Modules
//!
//! - [`engine`]: [`DynamicsEngine`] trait, [`ForceSurface`], the pure-Rust
//!   [`engine::native`] BAOAB + RATTLE/SHAKE backend, the optional
//!   `engine::lammpc` backend over `lammpc_session_*` (the `lammpc` feature),
//!   and optional `rgpot` and `rgpot-session` adapters.
//! - [`stats`]: non-overlapping batch means, Geyer effective sample size
//!   (`n_eff`), and Geweke two-window stationarity checks.
//! - [`kappa`] and [`kappa::bennett_chandler`]: Bennett-Chandler reactive-flux
//!   transmission coefficient estimator, generic [`DynamicsEngine`] shooter,
//!   and Kramers spatial-diffusion factor.
//! - [`hyperplane`]: holonomic hyperplane free-energy gradient (`dA/dc`) and
//!   dividing-surface configuration sampling via [`hyperplane::sample_hyperplane`].
//! - [`pafi`]: Projected Average Force Integration along reaction paths
//!   ([`pafi::PafiConfig`], [`pafi::FreeEnergyProfile`], [`pafi::pafi_profile`],
//!   [`pafi::group_displacement_scan`], [`pafi::MeanForceIntegral`]).
//! - [`ohtst`]: optimal hyperplanar transition state theory. The plane is
//!   moved and turned until its free energy is a maximum
//!   ([`ohtst::PlaneOptimizer`]), and [`ohtst::ohtst_rate`] combines the basin
//!   density, the reversible work to the optimum and the flux through it.
//!   [`adaptive_ohtst`] places the sampling along a string of planes.
//! - [`ffs`]: direct Forward Flux Sampling across ordered interfaces via
//!   [`ffs::forward_flux_rate`].
//! - [`opes`]: multithermal OPES expanded-ensemble bias on the potential
//!   energy ([`opes::OpesMultithermal`]), its biased [`ForceSurface`]
//!   ([`opes::OpesMultithermalSurface`]), the [`DynamicsEngine`] driver
//!   [`opes::run_opes_multithermal`], and per-frame temperature reweighting
//!   ([`opes::MultithermalReweighting`]).
//! - [`network`]: symmetry-resolved kinetic networks and kMC rate tables
//!   (`build_rate_table` is not yet implemented).
//! - [`cu_reference`]: Daw and Chandross copper vacancy mobility. The
//!   published column is the packaged Table II record. The lattice comes
//!   from [`cu_reference::run_npt_lattice`], which refuses until an
//!   rgpot-core release exports `eam_table_from_file` and
//!   `eam_table_from_data`.
//!
//! ## Quick example
//!
//! ```rust
//! use rgflux::kappa::bennett_chandler::{
//!     BennettChandlerConfig, ReactiveTrajectory, Side, estimate_bennett_chandler,
//! };
//!
//! let trajectories = vec![
//!     ReactiveTrajectory::new(1.2, vec![Side::Product; 16])?,
//!     ReactiveTrajectory::new(-0.9, vec![Side::Reactant; 16])?,
//!     ReactiveTrajectory::new(0.8, vec![Side::Product; 16])?,
//! ];
//! let estimate = estimate_bennett_chandler(&trajectories, &BennettChandlerConfig::default())?;
//! assert_eq!(estimate.plateau, 1.0);
//! # Ok::<(), rgflux::FluxError>(())
//! ```
//!
//! ## References
//!
//! - Goswami, R. (2025). *Efficient Exploration of Chemical Kinetics*.
//!   arXiv:2510.21368. <https://doi.org/10.48550/arXiv.2510.21368>
//! - Goswami, R., Gunde, M., and Jónsson, H. (2026). *Enhanced climbing image
//!   nudged elastic band method with Hessian eigenmode alignment*. Frontiers
//!   in Chemistry, 14, 1807063. <https://doi.org/10.3389/fchem.2026.1807063>
//! - Goswami, R., and Jónsson, H. (2026). *Adaptive Pruning for Increased
//!   Robustness and Reduced Computational Overhead in Gaussian Process
//!   Accelerated Saddle Point Searches*. ChemPhysChem, 27, e202500730.
//!   <https://doi.org/10.1002/cphc.202500730>
//! - Goswami, R., Masterov, M., Kamath, S., Peña-Torres, A., and Jónsson, H.
//!   (2025). *Efficient Implementation of Gaussian Process Regression
//!   Accelerated Saddle Point Searches with Application to Molecular
//!   Reactions*. Journal of Chemical Theory and Computation, 21(16),
//!   7935-7943. <https://doi.org/10.1021/acs.jctc.5c00866>
//! - Goswami, R. (2026). *Reproducible orchestration of best practices for
//!   reaction path optimization with the nudged elastic band*. MethodsX, 16,
//!   103899. <https://doi.org/10.1016/j.mex.2026.103899>
//! - Invernizzi, M., Piaggi, P. M., and Parrinello, M. (2020). *Unified
//!   Approach to Enhanced Sampling*. Physical Review X, 10, 041034.
//!   <https://doi.org/10.1103/PhysRevX.10.041034>
//! - Daw, M. S., and Chandross, M. (2021). *Sluggish diffusion in random
//!   equimolar FCC alloys*. Physical Review Materials, 5(4), 043603.
//!   <https://doi.org/10.1103/PhysRevMaterials.5.043603>

#![deny(missing_docs)]

#[macro_use]
mod config;

pub mod adaptive_ohtst;
pub mod cu_reference;
pub mod defect_pair;
pub mod engine;
pub mod error;
pub mod ffs;
pub mod hyperplane;
pub mod kappa;
pub mod network;
pub mod ohtst;
pub mod opes;
pub mod pafi;
pub mod pair_network;
pub mod stats;

pub use engine::{
    DynamicsEngine, EindirSurface, ForceSurface, HyperplaneConstraint, StepReport,
    SurfaceVelocityPolicy, ThermostatConfig,
};
pub use error::FluxError;
