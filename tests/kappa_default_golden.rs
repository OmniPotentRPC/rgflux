//! Default shooting results pinned to the values produced before early
//! basin commitment became an option.
//!
//! The constants below were produced by the shooter on `develop` before
//! [`KappaConfig::early_commitment`] existed. A default configuration must
//! still reproduce every trajectory side, the final engine state, and the
//! estimate bit for bit.

use ndarray::{Array1, Array2, ArrayView1};
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::engine::{DynamicsEngine, HyperplaneConstraint, ThermostatConfig};
use rgflux::kappa::KappaConfig;
use rgflux::kappa::bennett_chandler::{
    ShootingConfig, Side, shoot_bennett_chandler, shoot_reactive_trajectories,
};

struct DoubleWell2D;

impl rgsaddle::PointSurface for DoubleWell2D {
    fn eval(&self, r: ArrayView1<f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
        let (x, y) = (r[0], r[1]);
        let dx = x * x - 1.0;
        Ok((
            dx * dx + y * y,
            Array1::from_vec(vec![4.0 * x * dx, 2.0 * y]),
        ))
    }
}

fn fnv(hash: &mut u64, word: u64) {
    for byte in word.to_le_bytes() {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

fn fixture() -> (
    NativeEngine<DoubleWell2D>,
    HyperplaneConstraint,
    Array2<f64>,
) {
    let plane = HyperplaneConstraint::new(Array1::from_vec(vec![1.0, 0.0]), 0.0).expect("plane");
    let engine = NativeEngine::new(
        DoubleWell2D,
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0, 1.0]))
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(0.5)
                    .with_temperature(0.3),
            )
            .with_constraint(plane.clone())
            .with_seed(4242),
    )
    .expect("engine");
    let mut starts = Array2::zeros((6, 2));
    for (row, y) in [-0.4, -0.2, -0.05, 0.05, 0.2, 0.4].into_iter().enumerate() {
        starts[(row, 1)] = y;
    }
    (engine, plane, starts)
}

fn config() -> ShootingConfig {
    ShootingConfig::default()
        .with_steps(200)
        .with_time_step(0.02)
        .with_antithetic(true)
        .with_seed(4242)
}

fn trajectory_hash(config: &ShootingConfig) -> u64 {
    let (mut engine, plane, starts) = fixture();
    let trajectories =
        shoot_reactive_trajectories(&mut engine, &plane, starts.view(), config).expect("shoot");
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for trajectory in &trajectories {
        fnv(&mut hash, trajectory.initial_normal_velocity.to_bits());
        for side in &trajectory.sides {
            fnv(&mut hash, u64::from(*side == Side::Product));
        }
    }
    for value in engine.positions().iter().chain(engine.velocities().iter()) {
        fnv(&mut hash, value.to_bits());
    }
    hash
}

fn estimate_bits(config: &ShootingConfig) -> [u64; 3] {
    let (mut engine, plane, starts) = fixture();
    let est = shoot_bennett_chandler(&mut engine, &plane, starts.view(), config).expect("estimate");
    let mut kappa_hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in &est.kappa_t {
        fnv(&mut kappa_hash, value.to_bits());
    }
    [
        kappa_hash,
        est.plateau.to_bits(),
        est.standard_error.to_bits(),
    ]
}

const GOLDEN_TRAJECTORIES: u64 = 0xe7ed_967d_3d2c_c200;
const GOLDEN_KAPPA_T: u64 = 0xd6ad_c869_7013_5079;
const GOLDEN_PLATEAU: u64 = 0x3fda_9b69_a373_52a5;
const GOLDEN_STANDARD_ERROR: u64 = 0x3fd3_faa1_bcb6_c6db;

#[test]
fn default_config_matches_develop_bit_for_bit() {
    assert_eq!(trajectory_hash(&config()), GOLDEN_TRAJECTORIES);
    let bits = estimate_bits(&config());
    assert_eq!(bits[0], GOLDEN_KAPPA_T);
    assert_eq!(bits[1], GOLDEN_PLATEAU);
    assert_eq!(bits[2], GOLDEN_STANDARD_ERROR);
}

#[test]
fn window_without_opt_in_matches_develop_bit_for_bit() {
    for window in [0, 1, 4, 64, 200, 10_000] {
        let cfg = config().with_commit_checkpoints(window);
        assert_eq!(
            trajectory_hash(&cfg),
            GOLDEN_TRAJECTORIES,
            "window {window}"
        );
        let bits = estimate_bits(&cfg);
        assert_eq!(
            bits,
            [GOLDEN_KAPPA_T, GOLDEN_PLATEAU, GOLDEN_STANDARD_ERROR]
        );
    }
}

#[test]
fn opted_in_commitment_changes_these_trajectories() {
    let cfg = config().with_kappa(KappaConfig::default().with_early_commitment(true));
    assert_ne!(trajectory_hash(&cfg), GOLDEN_TRAJECTORIES);
    let opted_in_with_zero = config().with_kappa(
        KappaConfig::default()
            .with_early_commitment(true)
            .with_commit_checkpoints(0),
    );
    assert_eq!(trajectory_hash(&opted_in_with_zero), GOLDEN_TRAJECTORIES);
}
