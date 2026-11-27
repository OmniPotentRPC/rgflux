//! Analytic checks for the two-group displacement scan.
//!
//! The potential is `V = k (s - s_eq)^2 / 2` with the same mass-weighted
//! displacement the scan constrains. On that surface the mean force is
//! `k (s - s_eq)` at every temperature, because the constraint holds `s`
//! fixed and the multiplier is constant.

use approx::assert_abs_diff_eq;
use ndarray::{Array1, ArrayView1};
use rgflux::FluxError;
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::engine::{DynamicsEngine, ThermostatConfig};
use rgflux::hyperplane::HyperplaneConfig;
use rgflux::pafi::{GroupDisplacementProfile, group_displacement_scan};

const TOL: f64 = 1.0e-8;

struct DisplacementSpring {
    k: f64,
    s_eq: f64,
    atom_mass: Vec<f64>,
    group_a: Vec<usize>,
    group_b: Vec<usize>,
    direction: [f64; 3],
    periodic_length: Option<f64>,
}

impl DisplacementSpring {
    fn displacement(&self, x: ArrayView1<'_, f64>, cell: Option<&[f64; 9]>) -> f64 {
        let (com_a, com_b) = centres(x, &self.atom_mass, &self.group_a, &self.group_b);
        let mut delta = [0.0; 3];
        for axis in 0..3 {
            delta[axis] = com_b[axis] - com_a[axis];
        }
        if let Some(cell) = cell {
            delta = minimum_image(delta, cell);
        } else if let Some(length) = self.periodic_length {
            delta[0] -= length * (delta[0] / length).round();
        }
        self.direction[0] * delta[0] + self.direction[1] * delta[1] + self.direction[2] * delta[2]
    }
}

impl rgsaddle::PointSurface for DisplacementSpring {
    fn eval(&self, x: ArrayView1<'_, f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
        self.eval_in_cell(x, &[0.0; 9])
    }

    fn eval_in_cell(
        &self,
        x: ArrayView1<'_, f64>,
        cell: &[f64; 9],
    ) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
        let periodic = cell.iter().any(|entry| *entry != 0.0);
        let s = self.displacement(x, periodic.then_some(cell));
        let gap = s - self.s_eq;
        let mut gradient = Array1::zeros(x.len());
        // Derivative of the raw (or branch-local) displacement. Away from a
        // periodic cut it equals the collective gradient of `s`.
        let (weight_a, weight_b) = group_weights(&self.atom_mass, &self.group_a, &self.group_b);
        for (atom, weight) in weight_a {
            for axis in 0..3 {
                gradient[3 * atom + axis] += self.k * gap * weight * self.direction[axis];
            }
        }
        for (atom, weight) in weight_b {
            for axis in 0..3 {
                gradient[3 * atom + axis] += self.k * gap * weight * self.direction[axis];
            }
        }
        Ok((0.5 * self.k * gap * gap, gradient))
    }
}

fn centres(
    x: ArrayView1<'_, f64>,
    atom_mass: &[f64],
    group_a: &[usize],
    group_b: &[usize],
) -> ([f64; 3], [f64; 3]) {
    (centre(x, atom_mass, group_a), centre(x, atom_mass, group_b))
}

fn centre(x: ArrayView1<'_, f64>, atom_mass: &[f64], group: &[usize]) -> [f64; 3] {
    let mut acc = [0.0; 3];
    let mut total = 0.0;
    for &atom in group {
        let mass = atom_mass[atom];
        total += mass;
        for axis in 0..3 {
            acc[axis] += mass * x[3 * atom + axis];
        }
    }
    for value in &mut acc {
        *value /= total;
    }
    acc
}

type WeightedAtoms = Vec<(usize, f64)>;

fn group_weights(
    atom_mass: &[f64],
    group_a: &[usize],
    group_b: &[usize],
) -> (WeightedAtoms, WeightedAtoms) {
    let total = |group: &[usize]| group.iter().map(|&atom| atom_mass[atom]).sum::<f64>();
    let weights = |group: &[usize], sign: f64| {
        let mass = total(group);
        group
            .iter()
            .map(|&atom| (atom, sign * atom_mass[atom] / mass))
            .collect()
    };
    (weights(group_a, -1.0), weights(group_b, 1.0))
}

fn minimum_image(delta: [f64; 3], cell: &[f64; 9]) -> [f64; 3] {
    let xx = cell[0];
    let xy = cell[3];
    let xz = cell[6];
    let yy = cell[4];
    let yz = cell[7];
    let zz = cell[8];
    let fz = delta[2] / zz;
    let fy = (delta[1] - yz * fz) / yy;
    let fx = (delta[0] - xy * fy - xz * fz) / xx;
    let ix = fx.round();
    let iy = fy.round();
    let iz = fz.round();
    [
        delta[0] - (ix * xx + iy * xy + iz * xz),
        delta[1] - (iy * yy + iz * yz),
        delta[2] - iz * zz,
    ]
}

fn coordinate_masses(atom_mass: &[f64]) -> Array1<f64> {
    let mut masses = Vec::with_capacity(atom_mass.len() * 3);
    for &mass in atom_mass {
        masses.extend_from_slice(&[mass, mass, mass]);
    }
    Array1::from_vec(masses)
}

fn positions_from_x(xs: &[f64]) -> Array1<f64> {
    let mut values = Vec::with_capacity(xs.len() * 3);
    for &x in xs {
        values.extend_from_slice(&[x, 0.0, 0.0]);
    }
    Array1::from_vec(values)
}

fn config() -> HyperplaneConfig {
    HyperplaneConfig::default()
        .with_burn_in_steps(0)
        .with_sample_steps(2)
        .with_retention_stride(1)
        .with_time_step(0.05)
        .with_seed(11)
}

fn build_engine(
    surface: DisplacementSpring,
    cell: Option<[f64; 9]>,
    xs: &[f64],
) -> NativeEngine<DisplacementSpring> {
    let masses = coordinate_masses(&surface.atom_mass);
    let mut config = NativeConfig::default().with_masses(masses).with_thermostat(
        ThermostatConfig::default()
            .with_gamma(1.0)
            .with_temperature(0.0),
    );
    if let Some(cell) = cell {
        config = config.with_cell(cell);
    }
    let mut engine = NativeEngine::new(surface, config).expect("engine");
    engine
        .set_positions(positions_from_x(xs).view())
        .expect("positions");
    engine
}

fn scan(
    engine: &mut impl DynamicsEngine,
    group_a: &[usize],
    group_b: &[usize],
    windows: &[f64],
    rigid: &[&[usize]],
) -> GroupDisplacementProfile {
    let direction = Array1::from_vec(vec![1.0, 0.0, 0.0]);
    group_displacement_scan(
        engine,
        group_a,
        group_b,
        direction.view(),
        windows,
        rigid,
        &config(),
    )
    .expect("scan")
}

fn assert_forces(profile: &GroupDisplacementProfile, windows: &[f64], expected: &[f64]) {
    assert_eq!(profile.displacements, windows);
    assert_eq!(profile.mean_forces.len(), expected.len());
    for (got, want) in profile.mean_forces.iter().zip(expected) {
        assert_abs_diff_eq!(*got, *want, epsilon = TOL);
    }
}

#[test]
fn multiplier_sign_and_scaling_match_an_analytic_two_group_spring() {
    // Group A is atoms 0 and 1 (masses 1 and 3). Group B is atom 2 (mass 2).
    // R_A = (1*0 + 3*4) / 4 = 3, so s = x_2 - 3.
    let atom_mass = vec![1.0, 3.0, 2.0];
    let group_a = [0, 1];
    let group_b = [2];
    let windows = [1.0, 4.0, 7.0];
    let s_eq = 4.0;
    let make = |k: f64| DisplacementSpring {
        k,
        s_eq,
        atom_mass: atom_mass.clone(),
        group_a: group_a.to_vec(),
        group_b: group_b.to_vec(),
        direction: [1.0, 0.0, 0.0],
        periodic_length: None,
    };
    let mut engine = build_engine(make(1.5), None, &[0.0, 4.0, 7.0]);
    let origin = engine.positions().to_owned();
    let profile = scan(&mut engine, &group_a, &group_b, &windows, &[&[0, 1]]);
    let expected: Vec<f64> = windows.iter().map(|s| 1.5 * (s - s_eq)).collect();
    assert_forces(&profile, &windows, &expected);
    assert!(
        profile.mean_forces[0] < 0.0,
        "force below equilibrium must be negative"
    );
    assert_abs_diff_eq!(profile.mean_forces[1], 0.0, epsilon = TOL);
    assert!(
        profile.mean_forces[2] > 0.0,
        "force above equilibrium must be positive"
    );

    let mut doubled = build_engine(make(3.0), None, &[0.0, 4.0, 7.0]);
    let scaled = scan(&mut doubled, &group_a, &group_b, &windows, &[]);
    for (force, window) in scaled.mean_forces.iter().zip(windows) {
        assert_abs_diff_eq!(*force, 3.0 * (window - s_eq), epsilon = TOL);
    }
    // Linear mean force: Hermite matches `k/2 * ((s-s_eq)^2 - (s0-s_eq)^2)`.
    let k = 1.5;
    for (value, window) in profile.free_energy.iter().zip(windows) {
        let analytic = 0.5 * k * ((window - s_eq).powi(2) - (windows[0] - s_eq).powi(2));
        assert_abs_diff_eq!(*value, analytic, epsilon = TOL);
    }
    assert_eq!(engine.positions(), origin.view());
    assert!(engine.constraint().is_none());
}

#[test]
fn exchanging_the_groups_negates_the_displacement_and_the_force() {
    let atom_mass = vec![1.0, 3.0, 2.0];
    let windows = [1.0, 4.0, 7.0];
    let surface = DisplacementSpring {
        k: 1.5,
        s_eq: 4.0,
        atom_mass,
        group_a: vec![0, 1],
        group_b: vec![2],
        direction: [1.0, 0.0, 0.0],
        periodic_length: None,
    };
    let mut forward = build_engine(surface, None, &[0.0, 4.0, 7.0]);
    // Rebuild an identical surface for the exchanged call. `engine` owns the first.
    let surface = DisplacementSpring {
        k: 1.5,
        s_eq: 4.0,
        atom_mass: vec![1.0, 3.0, 2.0],
        group_a: vec![0, 1],
        group_b: vec![2],
        direction: [1.0, 0.0, 0.0],
        periodic_length: None,
    };
    let profile = scan(&mut forward, &[0, 1], &[2], &windows, &[]);
    let mut backward = build_engine(surface, None, &[0.0, 4.0, 7.0]);
    let exchanged_windows: Vec<f64> = windows.iter().rev().map(|s| -s).collect();
    let exchanged = scan(&mut backward, &[2], &[0, 1], &exchanged_windows, &[]);
    for (got, original) in exchanged
        .mean_forces
        .iter()
        .zip(profile.mean_forces.iter().rev())
    {
        assert_abs_diff_eq!(*got, -original, epsilon = TOL);
    }
}

#[test]
fn a_uniform_translation_does_not_change_the_mean_force() {
    let surface = || DisplacementSpring {
        k: 2.0,
        s_eq: 1.0,
        atom_mass: vec![1.0, 3.0, 2.0],
        group_a: vec![0, 1],
        group_b: vec![2],
        direction: [1.0, 0.0, 0.0],
        periodic_length: None,
    };
    let windows = [0.0, 1.0, 2.0];
    let mut local = build_engine(surface(), None, &[0.0, 4.0, 4.0]);
    let local_profile = scan(&mut local, &[0, 1], &[2], &windows, &[]);
    let mut shifted = build_engine(surface(), None, &[0.0, 4.0, 4.0]);
    let mut translated = shifted.positions().to_owned();
    for atom in 0..3 {
        translated[3 * atom] += 2.5;
        translated[3 * atom + 1] += -1.25;
        translated[3 * atom + 2] += 0.5;
    }
    shifted.set_positions(translated.view()).expect("translate");
    let shifted_profile = scan(&mut shifted, &[0, 1], &[2], &windows, &[]);
    for (left, right) in local_profile
        .mean_forces
        .iter()
        .zip(&shifted_profile.mean_forces)
    {
        assert_abs_diff_eq!(left, right, epsilon = TOL);
    }
}

#[test]
fn a_periodic_cell_keeps_one_displacement_reference() {
    let length = 10.0;
    let cell = [length, 0.0, 0.0, 0.0, length, 0.0, 0.0, 0.0, length];
    let surface = || DisplacementSpring {
        k: 1.0,
        s_eq: 0.0,
        atom_mass: vec![1.0, 4.0],
        group_a: vec![0],
        group_b: vec![1],
        direction: [1.0, 0.0, 0.0],
        periodic_length: Some(length),
    };
    let windows = [1.0, 9.0];
    let mut periodic = build_engine(surface(), Some(cell), &[0.0, 9.0]);
    let profile = scan(&mut periodic, &[0], &[1], &windows, &[]);
    // Minimum image of 9 in a cell of length 10 is -1, so the branch-local
    // force is `k * s_image`, not `k * 9`.
    assert_abs_diff_eq!(profile.mean_forces[0], 1.0, epsilon = TOL);
    assert_abs_diff_eq!(profile.mean_forces[1], -1.0, epsilon = TOL);

    let mut open = build_engine(
        DisplacementSpring {
            periodic_length: None,
            ..surface()
        },
        None,
        &[0.0, 9.0],
    );
    let raw = scan(&mut open, &[0], &[1], &windows, &[]);
    assert_abs_diff_eq!(raw.mean_forces[1], 9.0, epsilon = TOL);

    let mut image = build_engine(surface(), Some(cell), &[0.0, -1.0]);
    let from_image = scan(&mut image, &[0], &[1], &windows, &[]);
    assert_abs_diff_eq!(
        from_image.mean_forces[1],
        profile.mean_forces[1],
        epsilon = TOL
    );

    let mut translated = build_engine(surface(), Some(cell), &[0.0, 9.0]);
    let mut shifted = translated.positions().to_owned();
    for atom in 0..2 {
        shifted[3 * atom] += 1.5;
        shifted[3 * atom + 1] += -0.25;
    }
    translated.set_positions(shifted.view()).expect("translate");
    let shifted_profile = scan(&mut translated, &[0], &[1], &windows, &[]);
    assert_abs_diff_eq!(
        shifted_profile.mean_forces[0],
        profile.mean_forces[0],
        epsilon = TOL
    );
    assert_abs_diff_eq!(
        shifted_profile.mean_forces[1],
        profile.mean_forces[1],
        epsilon = TOL
    );
}

#[test]
fn empty_overlapping_and_split_rigid_groups_are_rejected() {
    let surface = DisplacementSpring {
        k: 1.0,
        s_eq: 0.0,
        atom_mass: vec![1.0, 1.0, 1.0],
        group_a: vec![0],
        group_b: vec![1],
        direction: [1.0, 0.0, 0.0],
        periodic_length: None,
    };
    let mut engine = build_engine(surface, None, &[0.0, 1.0, 2.0]);
    let direction = Array1::from_vec(vec![1.0, 0.0, 0.0]);
    let windows = [0.0, 1.0];
    let reject = |engine: &mut NativeEngine<DisplacementSpring>,
                  group_a: &[usize],
                  group_b: &[usize],
                  rigid: &[&[usize]]| {
        group_displacement_scan(
            engine,
            group_a,
            group_b,
            direction.view(),
            &windows,
            rigid,
            &config(),
        )
        .expect_err("rejected scan")
    };
    assert!(matches!(
        reject(&mut engine, &[], &[1], &[]),
        FluxError::EmptyInput(_)
    ));
    assert!(matches!(
        reject(&mut engine, &[0], &[0], &[]),
        FluxError::InvalidConfig(message) if message.contains("overlap")
    ));
    let split = reject(&mut engine, &[0], &[1], &[&[0, 1]]);
    assert!(
        matches!(split, FluxError::InvalidConfig(message) if message.contains("rigid")),
        "split rigid group returned {split:?}"
    );
}
