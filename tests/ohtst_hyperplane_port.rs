use ndarray::{Array1, ArrayView1};

use rgflux::FluxError;
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::engine::{DynamicsEngine, ForceSurface, HyperplaneConstraint, ThermostatConfig};
use rgflux::hyperplane::{HyperplaneConfig, sample_hyperplane};
use rgflux::ohtst::{OhtstConfig, optimize_hyperplane};

struct DoubleWell2D;

impl ForceSurface for DoubleWell2D {
    fn energy_and_forces(
        &self,
        positions: ArrayView1<'_, f64>,
        _cell: Option<&[f64; 9]>,
    ) -> Result<(f64, Array1<f64>), FluxError> {
        if positions.len() != 2 {
            return Err(FluxError::Shape {
                expected: 2,
                got: positions.len(),
                context: "DoubleWell2D positions",
            });
        }
        let x = positions[0];
        let y = positions[1];
        let well = x * x - 1.0;
        Ok((
            0.25 * well * well + 0.5 * y * y,
            Array1::from_vec(vec![-x * well, -y]),
        ))
    }
}

fn engine_at(position: [f64; 2], seed: u64) -> Result<NativeEngine<DoubleWell2D>, FluxError> {
    let mut engine = NativeEngine::new(
        DoubleWell2D,
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0, 1.0]))
            .with_seed(seed),
    )?;
    engine.set_positions(Array1::from_vec(position.to_vec()).view())?;
    Ok(engine)
}

#[test]
fn hyperplane_sampling_returns_finite_force_and_configurations() -> Result<(), FluxError> {
    let plane = HyperplaneConstraint::from_direction(Array1::from_vec(vec![1.0, 0.0]).view(), 0.2)?;
    let config = HyperplaneConfig::default()
        .with_burn_in_steps(150)
        .with_sample_steps(800)
        .with_retention_stride(20)
        .with_time_step(0.01)
        .with_seed(11);

    let mut engine = engine_at([0.2, 0.0], 11)?;
    let sampled = sample_hyperplane(&mut engine, &plane, &config)?;

    assert!(sampled.mean_force.is_finite());
    assert!(sampled.standard_error.is_finite());
    assert!(sampled.configurations.nrows() > 0);
    assert_eq!(sampled.configurations.ncols(), 2);
    Ok(())
}

#[test]
fn hyperplane_mean_force_changes_with_plane_offset() -> Result<(), FluxError> {
    let config = HyperplaneConfig::default()
        .with_burn_in_steps(120)
        .with_sample_steps(600)
        .with_retention_stride(30)
        .with_time_step(0.01)
        .with_seed(21);

    let mut left = engine_at([-0.3, 0.0], 21)?;
    let mut right = engine_at([0.3, 0.0], 21)?;

    let left_plane =
        HyperplaneConstraint::from_direction(Array1::from_vec(vec![1.0, 0.0]).view(), -0.3)?;
    let right_plane =
        HyperplaneConstraint::from_direction(Array1::from_vec(vec![1.0, 0.0]).view(), 0.3)?;

    let left_sample = sample_hyperplane(&mut left, &left_plane, &config)?;
    let right_sample = sample_hyperplane(&mut right, &right_plane, &config)?;

    assert!((left_sample.mean_force - right_sample.mean_force).abs() > 1.0e-6);
    Ok(())
}

#[test]
fn ohtst_optimization_returns_positive_finite_rate() -> Result<(), FluxError> {
    let mut engine = engine_at([0.0, 0.0], 33)?;
    let seed_plane =
        HyperplaneConstraint::from_direction(Array1::from_vec(vec![1.0, 0.0]).view(), 0.0)?;
    let config = OhtstConfig::default().with_seed(33);

    let estimate = optimize_hyperplane(&mut engine, &seed_plane, &config)?;

    assert!(estimate.rate.is_finite() && estimate.rate > 0.0);
    assert!(estimate.standard_error.is_finite() && estimate.standard_error >= 0.0);
    Ok(())
}

fn engine_state<E: DynamicsEngine>(
    engine: &E,
) -> (
    Array1<f64>,
    Array1<f64>,
    Option<HyperplaneConstraint>,
    ThermostatConfig,
) {
    (
        engine.positions().to_owned(),
        engine.velocities().to_owned(),
        engine.constraint().cloned(),
        *engine.thermostat(),
    )
}

struct RisingLine;

impl ForceSurface for RisingLine {
    fn energy_and_forces(
        &self,
        positions: ArrayView1<'_, f64>,
        _cell: Option<&[f64; 9]>,
    ) -> Result<(f64, Array1<f64>), FluxError> {
        if positions.len() != 1 {
            return Err(FluxError::Shape {
                expected: 1,
                got: positions.len(),
                context: "RisingLine positions",
            });
        }
        Ok((positions[0], Array1::from_vec(vec![-1.0])))
    }
}

#[test]
fn optimize_hyperplane_restores_engine_when_the_search_fails() -> Result<(), FluxError> {
    let mut engine = NativeEngine::new(
        RisingLine,
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0]))
            .with_seed(5),
    )?;
    engine.set_constraint(Some(HyperplaneConstraint::from_direction(
        Array1::from_vec(vec![1.0]).view(),
        0.4,
    )?))?;
    engine.set_positions(Array1::from_vec(vec![0.2]).view())?;
    engine.shoot_velocities(5)?;
    let before = engine_state(&engine);
    let err = optimize_hyperplane(
        &mut engine,
        &HyperplaneConstraint::from_direction(Array1::from_vec(vec![1.0]).view(), 0.0)?,
        &OhtstConfig::default().with_max_outer_steps(1).with_seed(5),
    )
    .unwrap_err();
    assert!(matches!(err, FluxError::InvalidConfig(_)));
    assert_eq!(engine_state(&engine), before);
    Ok(())
}
