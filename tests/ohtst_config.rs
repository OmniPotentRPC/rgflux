//! Configuration defaults, setters and `validate_config` rejections of OH-TST.

use rgflux::FluxError;
use rgflux::ohtst::OhtstConfig;
use rgflux::stats::StatsConfig;

mod ohtst_support;

use ohtst_support::*;

#[test]
fn default_configuration_values_are_stable() {
    let config = OhtstConfig::default();
    assert_eq!(config.max_outer_steps, 32);
    assert_eq!(config.patience, 3);
    assert_eq!(config.max_translation, 0.25);
    assert_eq!(config.max_rotation, 0.25);
    assert_eq!(config.min_overlap, 0.03);
    assert!(config.use_mbar);
    assert_eq!(config.seed, 1);
    assert_eq!(config.stats, StatsConfig::default());
    assert_eq!(config.burn_in_steps, 256);
    assert_eq!(config.min_sample_steps, 128);
    assert_eq!(config.sample_step_factor, 64);
    assert_eq!(config.retention_stride, 16);
    assert_eq!(config.time_step, 0.01);
}

#[test]
fn setters_write_their_own_field() {
    let stats = StatsConfig::default().with_min_batches(7);
    let config = OhtstConfig::default()
        .with_max_outer_steps(1)
        .with_patience(2)
        .with_max_translation(3.0)
        .with_max_rotation(4.0)
        .with_min_overlap(5.0)
        .with_use_mbar(false)
        .with_seed(6)
        .with_stats(stats.clone())
        .with_burn_in_steps(8)
        .with_min_sample_steps(9)
        .with_sample_step_factor(10)
        .with_retention_stride(11)
        .with_time_step(12.0);
    assert_eq!(config.max_outer_steps, 1);
    assert_eq!(config.patience, 2);
    assert_eq!(config.max_translation, 3.0);
    assert_eq!(config.max_rotation, 4.0);
    assert_eq!(config.min_overlap, 5.0);
    assert!(!config.use_mbar);
    assert_eq!(config.seed, 6);
    assert_eq!(config.stats, stats);
    assert_eq!(config.burn_in_steps, 8);
    assert_eq!(config.min_sample_steps, 9);
    assert_eq!(config.sample_step_factor, 10);
    assert_eq!(config.retention_stride, 11);
    assert_eq!(config.time_step, 12.0);
}

fn rejected(config: OhtstConfig) -> FluxError {
    let mut engine = well_engine(-HALF_WIDTH);
    let err = error(&mut engine, &plane(&[1.0], 0.0), &config);
    assert!(
        engine.log.step_sizes.is_empty(),
        "no sampling before validation"
    );
    err
}

fn accepted(config: OhtstConfig) {
    let mut engine = well_engine(-HALF_WIDTH);
    estimate(&mut engine, &plane(&[1.0], 0.0), &config);
}

#[test]
fn validate_config_rejects_zero_outer_steps() {
    assert_eq!(
        rejected(quiet().with_max_outer_steps(0)),
        invalid("OhtstConfig::max_outer_steps must be at least 1")
    );
    accepted(quiet().with_max_outer_steps(1));
}

#[test]
fn validate_config_rejects_unusable_translation_limits() {
    let message = "OhtstConfig::max_translation must be finite and positive";
    for bad in [0.0, -0.25, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            rejected(quiet().with_max_translation(bad)),
            invalid(message)
        );
    }
    accepted(quiet().with_max_translation(0.5));
}

#[test]
fn validate_config_rejects_unusable_rotation_limits() {
    let message = "OhtstConfig::max_rotation must be finite and non-negative";
    for bad in [-0.25, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(rejected(quiet().with_max_rotation(bad)), invalid(message));
    }
    accepted(quiet().with_max_rotation(0.0));
}

#[test]
fn validate_config_rejects_short_sample_floors() {
    let message = "OhtstConfig::min_sample_steps must be at least 2";
    for bad in [0, 1] {
        assert_eq!(
            rejected(quiet().with_min_sample_steps(bad)),
            invalid(message)
        );
    }
    accepted(quiet().with_min_sample_steps(2).with_sample_step_factor(0));
}

#[test]
fn validate_config_rejects_zero_retention_stride() {
    assert_eq!(
        rejected(quiet().with_retention_stride(0)),
        invalid("OhtstConfig::retention_stride must be at least 1")
    );
    accepted(quiet().with_retention_stride(1));
}

#[test]
fn validate_config_rejects_unusable_time_steps() {
    let message = "OhtstConfig::time_step must be finite and positive";
    for bad in [0.0, -0.01, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(rejected(quiet().with_time_step(bad)), invalid(message));
    }
    accepted(quiet().with_time_step(f64::MIN_POSITIVE));
}

#[test]
fn default_reactant_force_tolerance_is_the_rgmin_lbfgs_tolerance() {
    let config = OhtstConfig::default();
    assert_eq!(config.reactant_fmax, 1.0e-6);
    assert_eq!(config.reactant_fmax, rgmin::Lbfgs::default().gtol);
}

#[test]
fn reactant_force_tolerance_setter_writes_its_field() {
    let config = OhtstConfig::default().with_reactant_fmax(3.5e-9);
    assert_eq!(config.reactant_fmax, 3.5e-9);
    assert_eq!(
        config.with_reactant_fmax(1.0e-6),
        OhtstConfig::default(),
        "the setter touches no other field"
    );
}

#[test]
fn validate_config_rejects_unusable_reactant_force_tolerances() {
    let message = "OhtstConfig::reactant_fmax must be finite and positive";
    for bad in [
        0.0,
        -0.0,
        -1.0e-6,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        assert_eq!(rejected(quiet().with_reactant_fmax(bad)), invalid(message));
    }
    accepted(quiet().with_reactant_fmax(f64::MIN_POSITIVE));
}

#[test]
fn seed_plane_must_be_a_unit_plane_in_the_engine_dimension() {
    let mut engine = well_engine(-HALF_WIDTH);
    let mut bent = plane(&[1.0], 0.0);
    bent.normal[0] = 2.0;
    assert_eq!(
        error(&mut engine, &bent, &quiet()),
        invalid("hyperplane normal must have unit Euclidean norm")
    );
    assert_eq!(
        error(&mut engine, &plane(&[1.0, 0.0], 0.0), &quiet()),
        FluxError::Shape {
            expected: 1,
            got: 2,
            context: "OH-TST seed plane"
        }
    );
    assert!(engine.log.step_sizes.is_empty());
}

#[test]
fn zero_thermal_energy_is_rejected() {
    let mut engine = well_engine(-HALF_WIDTH).with_temperature(0.0);
    assert_eq!(
        error(&mut engine, &plane(&[1.0], 0.0), &quiet()),
        invalid("OH-TST requires a positive thermal energy k_B * T")
    );
}
