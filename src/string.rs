//! Finite-temperature strings and strings with swarms of trajectories.
//!
//! Both algorithms sample dynamics on hyperplanes through the current images,
//! orthogonal to centered path tangents. The finite-temperature method evolves
//! each image with the mean in-plane force. The swarm method instead evolves it
//! with the mean displacement of independent constrained trajectories. Images
//! are redistributed with [`rgsaddle::reparametrize`], the string utility
//! exposed by the pinned `rgsaddle` dependency.
//!
//! ## References
//!
//! - E, W., Ren, W., and Vanden-Eijnden, E. (2005). *Finite Temperature
//!   String Method for the Study of Rare Events*. The Journal of Physical
//!   Chemistry B, 109, 6688-6693. <https://doi.org/10.1021/jp0455430>
//! - Vanden-Eijnden, E., and Venturoli, M. (2009). *Revisiting the finite
//!   temperature string method for the calculation of reaction tubes and
//!   free energies*. The Journal of Chemical Physics, 130, 194103.
//!   <https://doi.org/10.1063/1.3130083>
//! - Pan, A. C., Sezer, D., and Roux, B. (2008). *Finding Transition Pathways
//!   Using the String Method with Swarms of Trajectories*. The Journal of
//!   Physical Chemistry B, 112, 3432-3440. <https://doi.org/10.1021/jp0777059>

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rgmin::vecops::nrm2;
use rgsaddle::{Reparametrization, reparametrize};

use crate::engine::{DynamicsEngine, EngineSnapshot, validate_finite_slice};
use crate::error::FluxError;
use crate::hyperplane::{
    HyperplaneConfig, HyperplaneSample, plane_from_path_image, sample_hyperplane,
};

const MINIMUM_IMAGES: usize = 3;
const MINIMUM_ITERATIONS: usize = 1;
const MINIMUM_SWARM_SIZE: usize = 2;
const IMAGE_SEED_STRIDE: u64 = 0x9e37_79b9_7f4a_7c15;
const ITERATION_SEED_STRIDE: u64 = 0xbf58_476d_1ce4_e5b9;
const SWARM_SEED_STRIDE: u64 = 0x94d0_49bb_1331_11eb;
const DEFAULT_EVOLUTION_STEP: f64 = 0.05;
const DEFAULT_FORCE_TOLERANCE: f64 = 0.001;

/// Configuration for the finite-temperature string method.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct FiniteTemperatureStringConfig {
    /// Maximum sampling and reparametrization cycles.
    pub max_iterations: usize,
    /// Consecutive converged cycles required before stopping.
    pub patience: usize,
    /// Explicit evolution step multiplying the mean projected force.
    pub evolution_step: f64,
    /// Maximum Euclidean norm of an interior mean projected force at convergence.
    pub force_tolerance: f64,
    /// Keep the reactant and product endpoint images fixed.
    pub fix_endpoints: bool,
    /// Shared constrained-dynamics sampling configuration.
    pub sampling: HyperplaneConfig,
    /// Base seed mixed with the iteration and image indices.
    pub seed: u64,
}

impl Default for FiniteTemperatureStringConfig {
    fn default() -> Self {
        Self {
            max_iterations: 32,
            patience: 3,
            evolution_step: DEFAULT_EVOLUTION_STEP,
            force_tolerance: DEFAULT_FORCE_TOLERANCE,
            fix_endpoints: true,
            sampling: HyperplaneConfig::default(),
            seed: 1,
        }
    }
}

with_setters!(FiniteTemperatureStringConfig {
    with_max_iterations => max_iterations: val usize,
    with_patience => patience: val usize,
    with_evolution_step => evolution_step: val f64,
    with_force_tolerance => force_tolerance: val f64,
    with_fix_endpoints => fix_endpoints: val bool,
    with_sampling => sampling: val HyperplaneConfig,
    with_seed => seed: val u64,
});

/// Result of finite-temperature string evolution.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct FiniteTemperatureString {
    /// Final, equal-arc path images.
    pub images: Array2<f64>,
    /// Mean projected force at every final-cycle image.
    pub mean_projected_forces: Array2<f64>,
    /// Mean sampled position at every final-cycle image.
    pub mean_positions: Array2<f64>,
    /// Position covariance at every final-cycle image.
    pub position_covariances: Vec<Array2<f64>>,
    /// Number of completed cycles.
    pub iterations: usize,
    /// Whether the configured convergence condition was met.
    pub converged: bool,
}

/// Configuration for the string method with swarms of trajectories.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct SwarmStringConfig {
    /// Maximum swarm and reparametrization cycles.
    pub max_iterations: usize,
    /// Consecutive converged cycles required before stopping.
    pub patience: usize,
    /// Number of independent constrained trajectories per interior image.
    pub swarm_size: usize,
    /// Relaxation multiplying the mean swarm displacement.
    pub evolution_step: f64,
    /// Maximum Euclidean norm of an interior mean displacement at convergence.
    pub displacement_tolerance: f64,
    /// Keep the reactant and product endpoint images fixed.
    pub fix_endpoints: bool,
    /// Shared constrained-dynamics sampling configuration.
    pub sampling: HyperplaneConfig,
    /// Base seed mixed with iteration, image, and swarm indices.
    pub seed: u64,
}

impl Default for SwarmStringConfig {
    fn default() -> Self {
        Self {
            max_iterations: 32,
            patience: 3,
            swarm_size: 8,
            evolution_step: DEFAULT_EVOLUTION_STEP,
            displacement_tolerance: DEFAULT_FORCE_TOLERANCE,
            fix_endpoints: true,
            sampling: HyperplaneConfig::default(),
            seed: 1,
        }
    }
}

with_setters!(SwarmStringConfig {
    with_max_iterations => max_iterations: val usize,
    with_patience => patience: val usize,
    with_swarm_size => swarm_size: val usize,
    with_evolution_step => evolution_step: val f64,
    with_displacement_tolerance => displacement_tolerance: val f64,
    with_fix_endpoints => fix_endpoints: val bool,
    with_sampling => sampling: val HyperplaneConfig,
    with_seed => seed: val u64,
});

/// Result of string evolution with swarms of trajectories.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct SwarmString {
    /// Final, equal-arc path images.
    pub images: Array2<f64>,
    /// Mean sampled displacement at every final-cycle image.
    pub mean_displacements: Array2<f64>,
    /// Number of completed cycles.
    pub iterations: usize,
    /// Whether the configured convergence condition was met.
    pub converged: bool,
}

/// Evolve a finite-temperature string with shared hyperplane sampling.
pub fn finite_temperature_string<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    initial_images: ArrayView2<'_, f64>,
    config: &FiniteTemperatureStringConfig,
) -> Result<FiniteTemperatureString, FluxError> {
    validate_path(engine, initial_images)?;
    validate_common(
        config.max_iterations,
        config.patience,
        config.evolution_step,
        config.force_tolerance,
    )?;
    let snapshot = EngineSnapshot::capture(engine);
    let result = finite_temperature_string_inner(engine, initial_images, config);
    snapshot.restore_result(engine, result)
}

fn finite_temperature_string_inner<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    initial_images: ArrayView2<'_, f64>,
    config: &FiniteTemperatureStringConfig,
) -> Result<FiniteTemperatureString, FluxError> {
    let mut images = initial_images.to_owned();
    let mut mean_forces = Array2::zeros(images.raw_dim());
    let mut mean_positions = images.clone();
    let mut covariances = vec![Array2::zeros((engine.dof(), engine.dof())); images.nrows()];
    let mut converged_cycles = 0_usize;
    let mut converged = false;
    let mut iterations = 0_usize;

    for iteration in 0..config.max_iterations {
        let tangents = path_tangents(images.view())?;
        let first = if config.fix_endpoints { 1 } else { 0 };
        let end = if config.fix_endpoints {
            images.nrows() - 1
        } else {
            images.nrows()
        };
        let mut maximum_force = 0.0_f64;
        for image_index in first..end {
            let seed = mixed_seed(config.seed, iteration, image_index, 0);
            let sampled = sample_image(
                engine,
                images.row(image_index),
                tangents.row(image_index),
                &config.sampling,
                seed,
            )?;
            mean_forces
                .row_mut(image_index)
                .assign(&sampled.mean_projected_force);
            mean_positions
                .row_mut(image_index)
                .assign(&sampled.mean_position);
            covariances[image_index] = sampled.position_covariance;
            let force_norm = nrm2(sampled.mean_projected_force.view());
            maximum_force = maximum_force.max(force_norm);
            let update = sampled.mean_projected_force * config.evolution_step;
            images.row_mut(image_index).scaled_add(1.0, &update);
        }
        reparametrize(&mut images, None, None, Reparametrization::Linear);
        iterations = iteration + 1;
        if maximum_force <= config.force_tolerance {
            converged_cycles += 1;
            if converged_cycles >= config.patience {
                converged = true;
                break;
            }
        } else {
            converged_cycles = 0;
        }
    }

    Ok(FiniteTemperatureString {
        images,
        mean_projected_forces: mean_forces,
        mean_positions,
        position_covariances: covariances,
        iterations,
        converged,
    })
}

/// Evolve a string with independent swarms of constrained trajectories.
pub fn swarm_string<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    initial_images: ArrayView2<'_, f64>,
    config: &SwarmStringConfig,
) -> Result<SwarmString, FluxError> {
    validate_path(engine, initial_images)?;
    validate_common(
        config.max_iterations,
        config.patience,
        config.evolution_step,
        config.displacement_tolerance,
    )?;
    if config.swarm_size < MINIMUM_SWARM_SIZE {
        return Err(FluxError::InvalidConfig(
            "SwarmStringConfig::swarm_size must be at least 2",
        ));
    }
    let snapshot = EngineSnapshot::capture(engine);
    let result = swarm_string_inner(engine, initial_images, config);
    snapshot.restore_result(engine, result)
}

fn swarm_string_inner<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    initial_images: ArrayView2<'_, f64>,
    config: &SwarmStringConfig,
) -> Result<SwarmString, FluxError> {
    let mut images = initial_images.to_owned();
    let mut mean_displacements = Array2::zeros(images.raw_dim());
    let mut converged_cycles = 0_usize;
    let mut converged = false;
    let mut iterations = 0_usize;

    for iteration in 0..config.max_iterations {
        let tangents = path_tangents(images.view())?;
        mean_displacements.fill(0.0);
        let first = if config.fix_endpoints { 1 } else { 0 };
        let end = if config.fix_endpoints {
            images.nrows() - 1
        } else {
            images.nrows()
        };
        let mut maximum_displacement = 0.0_f64;
        for image_index in first..end {
            let image = images.row(image_index).to_owned();
            for swarm_index in 0..config.swarm_size {
                let seed = mixed_seed(config.seed, iteration, image_index, swarm_index);
                let sampled = sample_image(
                    engine,
                    image.view(),
                    tangents.row(image_index),
                    &config.sampling,
                    seed,
                )?;
                let displacement = sampled.mean_position - &image;
                mean_displacements
                    .row_mut(image_index)
                    .scaled_add((config.swarm_size as f64).recip(), &displacement);
            }
            let mean_displacement = mean_displacements.row(image_index);
            maximum_displacement = maximum_displacement.max(nrm2(mean_displacement));
            let update = mean_displacement.to_owned() * config.evolution_step;
            images.row_mut(image_index).scaled_add(1.0, &update);
        }
        reparametrize(&mut images, None, None, Reparametrization::Linear);
        iterations = iteration + 1;
        if maximum_displacement <= config.displacement_tolerance {
            converged_cycles += 1;
            if converged_cycles >= config.patience {
                converged = true;
                break;
            }
        } else {
            converged_cycles = 0;
        }
    }

    Ok(SwarmString {
        images,
        mean_displacements,
        iterations,
        converged,
    })
}

fn sample_image<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    image: ArrayView1<'_, f64>,
    tangent: ArrayView1<'_, f64>,
    sampling: &HyperplaneConfig,
    seed: u64,
) -> Result<HyperplaneSample, FluxError> {
    let plane = plane_from_path_image(image, tangent)?;
    engine.set_constraint(None)?;
    engine.set_positions(image)?;
    let mut image_sampling = sampling.clone();
    image_sampling.seed = seed;
    sample_hyperplane(engine, &plane, &image_sampling)
}

fn path_tangents(images: ArrayView2<'_, f64>) -> Result<Array2<f64>, FluxError> {
    let mut tangents = Array2::zeros(images.raw_dim());
    for index in 0..images.nrows() {
        let mut tangent: Array1<f64> = if index == 0 {
            &images.row(1) - &images.row(0)
        } else if index + 1 == images.nrows() {
            &images.row(index) - &images.row(index - 1)
        } else {
            &images.row(index + 1) - &images.row(index - 1)
        };
        let norm = nrm2(tangent.view());
        if norm <= f64::MIN_POSITIVE.sqrt() {
            return Err(FluxError::InvalidConfig(
                "string contains a zero-length centered tangent",
            ));
        }
        tangent /= norm;
        tangents.row_mut(index).assign(&tangent);
    }
    Ok(tangents)
}

fn validate_path<E: DynamicsEngine + ?Sized>(
    engine: &E,
    images: ArrayView2<'_, f64>,
) -> Result<(), FluxError> {
    if images.nrows() < MINIMUM_IMAGES {
        return Err(FluxError::InvalidConfig(
            "a string requires at least three images",
        ));
    }
    if images.ncols() != engine.dof() {
        return Err(FluxError::Shape {
            expected: engine.dof(),
            got: images.ncols(),
            context: "string image dimension",
        });
    }
    for image in images.rows() {
        validate_finite_slice(image, "string image")?;
    }
    Ok(())
}

fn validate_common(
    max_iterations: usize,
    patience: usize,
    evolution_step: f64,
    tolerance: f64,
) -> Result<(), FluxError> {
    if max_iterations < MINIMUM_ITERATIONS {
        return Err(FluxError::InvalidConfig(
            "string max_iterations must be at least 1",
        ));
    }
    if patience < MINIMUM_ITERATIONS {
        return Err(FluxError::InvalidConfig(
            "string patience must be at least 1",
        ));
    }
    if !evolution_step.is_finite() || evolution_step <= 0.0 {
        return Err(FluxError::InvalidConfig(
            "string evolution_step must be finite and positive",
        ));
    }
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(FluxError::InvalidConfig(
            "string tolerance must be finite and non-negative",
        ));
    }
    Ok(())
}

fn mixed_seed(base: u64, iteration: usize, image: usize, swarm: usize) -> u64 {
    base.wrapping_add((iteration as u64).wrapping_mul(ITERATION_SEED_STRIDE))
        .wrapping_add((image as u64).wrapping_mul(IMAGE_SEED_STRIDE))
        .wrapping_add((swarm as u64).wrapping_mul(SWARM_SEED_STRIDE))
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::path_tangents;

    #[test]
    fn centered_tangents_follow_a_bent_string() {
        let images = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]];
        let tangents = path_tangents(images.view()).expect("valid string");
        assert_eq!(tangents.row(0), array![1.0, 0.0]);
        assert_eq!(tangents.row(2), array![0.0, 1.0]);
        let root_half = 0.5_f64.sqrt();
        for component in tangents.row(1) {
            assert!((component - root_half).abs() <= f64::EPSILON);
        }
    }

    #[test]
    fn coincident_neighbors_fail_before_sampling() {
        let images = array![[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]];
        assert!(path_tangents(images.view()).is_err());
    }
}
