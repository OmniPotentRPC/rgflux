//! Fixed-cell scan of the displacement between two atom groups.
//!
//! The collective variable is the signed displacement of the mass-weighted
//! centres of mass along a unit direction `n`,
//!
//! `s = n · (R_B - R_A)`,
//! `R_G = (sum_{i in G} m_i r_i) / (sum_{i in G} m_i)`.
//!
//! Its Cartesian gradient is constant. On atom `i` of group B it is
//! `(m_i / M_B) n`, on atom `i` of group A it is `-(m_i / M_A) n`, and it is
//! zero elsewhere. `M_G` is the mass of group G. A uniform translation adds
//! the same vector to `R_A` and `R_B`, so `s` and `grad s` are unchanged.
//!
//! The Euclidean norm `||grad s||_2 = sqrt(sum_i c_i^2)`, with
//! `c_i = ± m_i / M_G`, rescales the flat hyperplane. The sampler constrains
//! the unit normal `ν = grad s / ||grad s||_2` at offset `c = s / ||grad s||_2`,
//! so `ν · x = c` is the same plane as `grad s · x = s`. The holonomic
//! multiplier `λ` of that unit-normal plane obeys `<λ> = dF/dc` (Sprik and
//! Ciccotti 1998). Because `dc = ds / ||grad s||_2`,
//!
//! `dF/ds = <λ> / ||grad s||_2`.
//!
//! `λ` is positive when the potential increases with `s`. The scan reports
//! that quotient, not the raw multiplier, and integrates it with
//! [`super::HermiteIntegral`].
//!
//! With a periodic cell the displacement uses one reference per window,
//! `d = d_ref + minimum_image(R_B - R_A - d_ref)` and `d_ref = s n`. Atoms
//! inside a group are unwrapped relative to the group's first atom before
//! the centres are formed, and group B is shifted by a lattice vector so the
//! raw centres sit on that branch. The cell itself is not changed.

use ndarray::{Array1, ArrayView1};

use crate::engine::{DynamicsEngine, EngineSnapshot, HyperplaneConstraint, validate_finite_slice};
use crate::error::FluxError;
use crate::hyperplane::HyperplaneConfig;
use crate::hyperplane::sample_hyperplane;

use super::HermiteIntegral;
use super::MeanForceIntegral;

/// Free-energy scan along the displacement between two atom groups.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct GroupDisplacementProfile {
    /// Window values of `s = n · (R_B - R_A)`, strictly increasing.
    pub displacements: Vec<f64>,
    /// Mean force `dF/ds = <λ> / ||grad s||_2` on each window.
    pub mean_forces: Vec<f64>,
    /// Block-means standard error of [`Self::mean_forces`].
    pub standard_errors: Vec<f64>,
    /// Cubic Hermite integral of the mean force, starting at 0 on the first window.
    pub free_energy: Vec<f64>,
    /// Absolute Hermite-minus-trapezoid truncation of [`Self::free_energy`].
    pub free_energy_error: Vec<f64>,
}

/// Sample `dF/ds` on each window and integrate it with the cubic Hermite rule.
///
/// `group_a` and `group_b` are 0-based atom indices, one entry per atom, with
/// no repeats and no shared atom. `rigid_groups` lists atoms that must stay
/// in one piece under the collective variable: every atom of a rigid group
/// has to lie wholly in A, wholly in B, or wholly in neither. A direction
/// whose gradient assigns two of those memberships returns
/// [`FluxError::InvalidConfig`]. Pass `&[]` when the session has no rigid
/// groups.
///
/// `direction` is a non-zero 3-vector and is normalised. `windows` are the
/// target values of `s`, at least two, finite, and strictly increasing.
/// Sampling uses [`sample_hyperplane`] with `config`; the cell already
/// installed on `engine` is left as it is. The engine's positions,
/// velocities, cell, thermostat, and constraint are restored on return.
///
/// # Example
///
/// Two atoms and `V = s^2 / 2` with `s = x_B - x_A`. The mean force equals `s`.
///
/// ```rust
/// use ndarray::{Array1, ArrayView1};
/// use rgflux::engine::native::{NativeConfig, NativeEngine};
/// use rgflux::engine::{DynamicsEngine, ThermostatConfig};
/// use rgflux::hyperplane::HyperplaneConfig;
/// use rgflux::pafi::group_displacement_scan;
///
/// struct PairSpring;
/// impl rgsaddle::PointSurface for PairSpring {
///     fn eval(
///         &self,
///         x: ArrayView1<'_, f64>,
///     ) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
///         let s = x[3] - x[0];
///         let mut gradient = Array1::zeros(6);
///         gradient[0] = -s;
///         gradient[3] = s;
///         Ok((0.5 * s * s, gradient))
///     }
/// }
///
/// let mut engine = NativeEngine::new(
///     PairSpring,
///     NativeConfig::default()
///         .with_masses(Array1::from_elem(6, 1.0))
///         .with_thermostat(ThermostatConfig::default().with_gamma(1.0).with_temperature(0.0)),
/// )?;
/// engine.set_positions(Array1::from_vec(vec![0.0, 0.0, 0.0, 0.5, 0.0, 0.0]).view())?;
/// let config = HyperplaneConfig::default()
///     .with_burn_in_steps(0)
///     .with_sample_steps(2)
///     .with_time_step(0.05);
/// let profile = group_displacement_scan(
///     &mut engine,
///     &[0],
///     &[1],
///     Array1::from_vec(vec![1.0, 0.0, 0.0]).view(),
///     &[0.0, 0.5],
///     &[],
///     &config,
/// )?;
/// assert!((profile.mean_forces[1] - 0.5).abs() < 1.0e-8);
/// # Ok::<(), rgflux::FluxError>(())
/// ```
///
/// # Errors
///
/// - [`FluxError::EmptyInput`] when either group is empty.
/// - [`FluxError::Shape`] when `direction` does not have length 3.
/// - [`FluxError::InvalidConfig`] when the groups overlap or repeat an atom,
///   a window list is shorter than two or is not strictly increasing, the
///   direction is zero, the coordinates are not Cartesian atoms with equal
///   per-axis masses, the cell lengths are not positive, or a rigid group is
///   split by the collective variable.
/// - [`FluxError::NonFinite`] when a direction component, window, cell entry,
///   or mean force is not finite.
/// - Any error from [`sample_hyperplane`] or from restoring the engine.
///
/// # References
///
/// - Sprik, M., and Ciccotti, G. (1998). *Free energy from constrained
///   molecular dynamics*. The Journal of Chemical Physics, 109, 7737–7744.
///   <https://doi.org/10.1063/1.477419>
/// - Swinburne, T. D., and Marinica, M.-C. (2018). *Unsupervised Calculation
///   of Free Energy Barriers in Large Crystalline Systems*. Physical Review
///   Letters, 120, 135503. <https://doi.org/10.1103/PhysRevLett.120.135503>
pub fn group_displacement_scan<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    group_a: &[usize],
    group_b: &[usize],
    direction: ArrayView1<'_, f64>,
    windows: &[f64],
    rigid_groups: &[&[usize]],
    config: &HyperplaneConfig,
) -> Result<GroupDisplacementProfile, FluxError> {
    let direction = unit_direction(direction)?;
    validate_windows(windows)?;
    let cell = engine.cell().copied();
    validate_cell(cell.as_ref())?;
    let atom_mass = cartesian_masses(engine.masses())?;
    let coeff = coefficients(atom_mass.len(), &atom_mass, group_a, group_b, rigid_groups)?;
    let (gradient, norm) = collective_gradient(atom_mass.len(), &coeff, direction);
    if !(norm.is_finite() && norm > 0.0) {
        return Err(FluxError::InvalidConfig(
            "group displacement gradient has zero norm",
        ));
    }
    let origin = engine.positions().to_owned();
    let prepared = WindowScan {
        origin: &origin,
        gradient: &gradient,
        norm,
        direction,
        atom_mass: &atom_mass,
        group_a,
        group_b,
        windows,
        cell,
        config,
    };
    let snapshot = EngineSnapshot::capture(engine);
    let result = sample_windows(engine, &prepared);
    snapshot.restore_result(engine, result)
}

struct WindowScan<'a> {
    origin: &'a Array1<f64>,
    gradient: &'a Array1<f64>,
    norm: f64,
    direction: [f64; 3],
    atom_mass: &'a [f64],
    group_a: &'a [usize],
    group_b: &'a [usize],
    windows: &'a [f64],
    cell: Option<[f64; 9]>,
    config: &'a HyperplaneConfig,
}

fn sample_windows<E: DynamicsEngine + ?Sized>(
    engine: &mut E,
    scan: &WindowScan<'_>,
) -> Result<GroupDisplacementProfile, FluxError> {
    let mut mean_forces = Vec::with_capacity(scan.windows.len());
    let mut standard_errors = Vec::with_capacity(scan.windows.len());
    for (index, &window) in scan.windows.iter().enumerate() {
        let frame = aligned_positions(
            scan.origin,
            scan.direction,
            scan.group_a,
            scan.group_b,
            scan.atom_mass,
            window,
            scan.cell.as_ref(),
        );
        engine.set_constraint(None)?;
        engine.set_positions(frame.view())?;
        let plane = HyperplaneConstraint::from_direction(scan.gradient.view(), window / scan.norm)?;
        let window_config = scan.config.clone().with_seed(
            scan.config
                .seed
                .wrapping_add((index as u64).wrapping_mul(4_099)),
        );
        let sample = sample_hyperplane(engine, &plane, &window_config)?;
        let mean = sample.mean_force / scan.norm;
        let error = sample.standard_error / scan.norm;
        if !mean.is_finite() || !error.is_finite() {
            return Err(FluxError::NonFinite("group displacement mean force"));
        }
        mean_forces.push(mean);
        standard_errors.push(error);
    }
    let integral = HermiteIntegral.cumulative(scan.windows, &mean_forces)?;
    Ok(GroupDisplacementProfile {
        displacements: scan.windows.to_vec(),
        mean_forces,
        standard_errors,
        free_energy: integral.values,
        free_energy_error: integral.error,
    })
}

fn cartesian_masses(masses: ArrayView1<'_, f64>) -> Result<Vec<f64>, FluxError> {
    let dof = masses.len();
    if dof < 6 || !dof.is_multiple_of(3) {
        return Err(FluxError::InvalidConfig(
            "group displacement requires at least two Cartesian atoms",
        ));
    }
    let mut atom_mass = Vec::with_capacity(dof / 3);
    for atom in 0..dof / 3 {
        let base = atom * 3;
        let mass = masses[base];
        if masses[base + 1] != mass || masses[base + 2] != mass {
            return Err(FluxError::InvalidConfig(
                "group displacement requires equal x, y, and z masses on each atom",
            ));
        }
        atom_mass.push(mass);
    }
    Ok(atom_mass)
}

fn coefficients(
    n_atoms: usize,
    atom_mass: &[f64],
    group_a: &[usize],
    group_b: &[usize],
    rigid_groups: &[&[usize]],
) -> Result<Vec<f64>, FluxError> {
    if group_a.is_empty() || group_b.is_empty() {
        return Err(FluxError::EmptyInput("group displacement group"));
    }
    let mut side = vec![0_u8; n_atoms];
    assign(&mut side, group_a, 1)?;
    assign(&mut side, group_b, 2)?;
    for rigid in rigid_groups {
        check_rigid(&side, rigid)?;
    }
    let mass_a: f64 = group_a.iter().map(|&atom| atom_mass[atom]).sum();
    let mass_b: f64 = group_b.iter().map(|&atom| atom_mass[atom]).sum();
    let mut coeff = vec![0.0; n_atoms];
    for &atom in group_a {
        coeff[atom] = -atom_mass[atom] / mass_a;
    }
    for &atom in group_b {
        coeff[atom] = atom_mass[atom] / mass_b;
    }
    Ok(coeff)
}

fn assign(side: &mut [u8], group: &[usize], label: u8) -> Result<(), FluxError> {
    for &atom in group {
        if atom >= side.len() {
            return Err(FluxError::InvalidConfig(
                "group displacement atom index is outside the session",
            ));
        }
        if side[atom] == label {
            return Err(FluxError::InvalidConfig(
                "group displacement group repeats an atom",
            ));
        }
        if side[atom] != 0 {
            return Err(FluxError::InvalidConfig(
                "group displacement groups overlap",
            ));
        }
        side[atom] = label;
    }
    Ok(())
}

fn check_rigid(side: &[u8], group: &[usize]) -> Result<(), FluxError> {
    if group.is_empty() {
        return Err(FluxError::InvalidConfig(
            "group displacement rigid constraint group is empty",
        ));
    }
    let mut seen = vec![false; side.len()];
    let mut membership: Option<u8> = None;
    for &atom in group {
        if atom >= side.len() {
            return Err(FluxError::InvalidConfig(
                "group displacement atom index is outside the session",
            ));
        }
        if seen[atom] {
            return Err(FluxError::InvalidConfig(
                "group displacement rigid constraint group repeats an atom",
            ));
        }
        seen[atom] = true;
        match membership {
            None => membership = Some(side[atom]),
            Some(previous) if previous != side[atom] => {
                return Err(FluxError::InvalidConfig(
                    "group-displacement direction splits a rigid constraint group",
                ));
            }
            Some(_) => {}
        }
    }
    Ok(())
}

fn unit_direction(direction: ArrayView1<'_, f64>) -> Result<[f64; 3], FluxError> {
    if direction.len() != 3 {
        return Err(FluxError::Shape {
            expected: 3,
            got: direction.len(),
            context: "group displacement direction",
        });
    }
    validate_finite_slice(direction, "group displacement direction")?;
    let mut normal = [direction[0], direction[1], direction[2]];
    let norm = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if !(norm.is_finite() && norm > f64::MIN_POSITIVE.sqrt()) {
        return Err(FluxError::InvalidConfig(
            "group displacement direction has zero norm",
        ));
    }
    for component in &mut normal {
        *component /= norm;
    }
    Ok(normal)
}

fn validate_windows(windows: &[f64]) -> Result<(), FluxError> {
    if windows.len() < 2 {
        return Err(FluxError::InvalidConfig(
            "group displacement needs at least two windows",
        ));
    }
    for pair in windows.windows(2) {
        if !pair[0].is_finite() || !pair[1].is_finite() {
            return Err(FluxError::NonFinite("group displacement window"));
        }
        if pair[1] <= pair[0] {
            return Err(FluxError::InvalidConfig(
                "group displacement windows must strictly increase",
            ));
        }
    }
    Ok(())
}

fn validate_cell(cell: Option<&[f64; 9]>) -> Result<(), FluxError> {
    let Some(cell) = cell else {
        return Ok(());
    };
    for entry in cell {
        if !entry.is_finite() {
            return Err(FluxError::NonFinite("group displacement cell"));
        }
    }
    if cell[0] <= 0.0 || cell[4] <= 0.0 || cell[8] <= 0.0 {
        return Err(FluxError::InvalidConfig(
            "group displacement cell lengths must be positive",
        ));
    }
    Ok(())
}

fn collective_gradient(n_atoms: usize, coeff: &[f64], direction: [f64; 3]) -> (Array1<f64>, f64) {
    let mut gradient = Array1::zeros(n_atoms * 3);
    let mut coeff_sq = 0.0;
    for atom in 0..n_atoms {
        let weight = coeff[atom];
        coeff_sq += weight * weight;
        for axis in 0..3 {
            gradient[3 * atom + axis] = weight * direction[axis];
        }
    }
    (gradient, coeff_sq.sqrt())
}

fn aligned_positions(
    origin: &Array1<f64>,
    direction: [f64; 3],
    group_a: &[usize],
    group_b: &[usize],
    atom_mass: &[f64],
    window: f64,
    cell: Option<&[f64; 9]>,
) -> Array1<f64> {
    let mut positions = origin.clone();
    let Some(cell) = cell else {
        return positions;
    };
    unwrap_group(&mut positions, group_a, cell);
    unwrap_group(&mut positions, group_b, cell);
    let delta = sub(
        centre(&positions, atom_mass, group_b),
        centre(&positions, atom_mass, group_a),
    );
    let reference = scale(direction, window);
    let adjusted = add(reference, minimum_image(sub(delta, reference), cell));
    let shift = sub(adjusted, delta);
    for &atom in group_b {
        for axis in 0..3 {
            positions[3 * atom + axis] += shift[axis];
        }
    }
    positions
}

fn unwrap_group(positions: &mut Array1<f64>, group: &[usize], cell: &[f64; 9]) {
    let Some(&first) = group.first() else {
        return;
    };
    let origin = atom_position(positions, first);
    for &atom in group.iter().skip(1) {
        let image = minimum_image(sub(atom_position(positions, atom), origin), cell);
        for axis in 0..3 {
            positions[3 * atom + axis] = origin[axis] + image[axis];
        }
    }
}

fn centre(positions: &Array1<f64>, atom_mass: &[f64], group: &[usize]) -> [f64; 3] {
    let mut acc = [0.0; 3];
    let mut total = 0.0;
    for &atom in group {
        let mass = atom_mass[atom];
        total += mass;
        let place = atom_position(positions, atom);
        for axis in 0..3 {
            acc[axis] += mass * place[axis];
        }
    }
    scale(acc, 1.0 / total)
}

fn atom_position(positions: &Array1<f64>, atom: usize) -> [f64; 3] {
    [
        positions[3 * atom],
        positions[3 * atom + 1],
        positions[3 * atom + 2],
    ]
}

fn minimum_image(delta: [f64; 3], cell: &[f64; 9]) -> [f64; 3] {
    let (xx, xy, xz) = (cell[0], cell[3], cell[6]);
    let (yy, yz, zz) = (cell[4], cell[7], cell[8]);
    let fz = delta[2] / zz;
    let fy = (delta[1] - yz * fz) / yy;
    let fx = (delta[0] - xy * fy - xz * fz) / xx;
    let (ix, iy, iz) = (fx.round(), fy.round(), fz.round());
    [
        delta[0] - (ix * xx + iy * xy + iz * xz),
        delta[1] - (iy * yy + iz * yz),
        delta[2] - iz * zz,
    ]
}

fn add(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

fn sub(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn scale(value: [f64; 3], factor: f64) -> [f64; 3] {
    [value[0] * factor, value[1] * factor, value[2] * factor]
}
