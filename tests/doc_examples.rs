//! Compile and execute the worked examples from `docs/orgmode/getting-started.org`
//! and `docs/orgmode/howto.org`, and verify that the `.org` source blocks stay in sync.

use ndarray::{Array1, Array2, ArrayView1};
use rgflux::FluxError;
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::engine::{DynamicsEngine, EindirSurface, HyperplaneConstraint, ThermostatConfig};
use rgflux::ffs::{FfsConfig, forward_flux_rate};
use rgflux::hyperplane::{HyperplaneConfig, sample_hyperplane};
use rgflux::kappa::bennett_chandler::{
    BennettChandlerConfig, ReactiveTrajectory, ShootingConfig, Side, estimate_bennett_chandler,
    shoot_bennett_chandler,
};
use rgflux::kappa::kramers_transmission_coefficient;
use rgflux::ohtst::{OhtstConfig, optimize_hyperplane};
use rgflux::pafi::{PafiConfig, pafi_profile};
use rgflux::stats::StatsConfig;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn extract_rust_blocks(org_source: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut current: Option<Vec<&str>> = None;
    for line in org_source.lines() {
        if line.trim() == "#+begin_src rust" {
            current = Some(Vec::new());
        } else if line.trim() == "#+end_src" {
            if let Some(lines) = current.take() {
                blocks.push(lines.join("\n"));
            }
        } else if let Some(lines) = &mut current {
            lines.push(line);
        }
    }
    blocks
}

#[test]
fn org_guide_rust_blocks_compile_and_run() {
    let getting_started = include_str!("../docs/orgmode/getting-started.org");
    let gs_blocks = extract_rust_blocks(getting_started);
    assert_eq!(gs_blocks.len(), 2, "getting-started.org has 2 Rust blocks");
    run_extracted_blocks("getting_started", &gs_blocks);

    let howto = include_str!("../docs/orgmode/howto.org");
    let howto_blocks = extract_rust_blocks(howto);
    assert_eq!(howto_blocks.len(), 8, "howto.org has 8 Rust blocks");
    run_extracted_blocks("howto", &howto_blocks);
}

fn run_extracted_blocks(label: &str, blocks: &[String]) {
    for block in blocks {
        assert!(
            block.contains("fn main() -> Result<(), rgflux::FluxError>"),
            "missing `main` in extracted {label} block:\n{block}"
        );
    }
    let sandbox = make_sandbox_dir(label);
    write_sandbox_manifest(&sandbox);
    let bins_dir = sandbox.join("src").join("bin");
    fs::create_dir_all(&bins_dir).expect("create bin directory");
    for (index, block) in blocks.iter().enumerate() {
        let file = bins_dir.join(format!("block_{index}.rs"));
        fs::write(file, block).expect("write extracted Rust block");
    }
    for (index, _) in blocks.iter().enumerate() {
        let output = Command::new("cargo")
            .arg("run")
            .arg("--quiet")
            .arg("--bin")
            .arg(format!("block_{index}"))
            .current_dir(&sandbox)
            .env(
                "CARGO_TARGET_DIR",
                Path::new(env!("CARGO_MANIFEST_DIR")).join("target/doc_examples"),
            )
            .output()
            .expect("run extracted Rust block");
        assert!(
            output.status.success(),
            "{label} block {index} failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    fs::remove_dir_all(&sandbox).expect("remove sandbox");
}

fn make_sandbox_dir(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time")
        .as_nanos();
    std::env::temp_dir().join(format!("rgflux_doc_examples_{label}_{nanos}"))
}

fn write_sandbox_manifest(sandbox: &Path) {
    fs::create_dir_all(sandbox).expect("create sandbox");
    let manifest = format!(
        r#"[package]
name = "rgflux-doc-examples"
version = "0.0.0"
edition = "2024"

[dependencies]
ndarray = "0.17"
rgsaddle = {{ version = "0.5.0", default-features = false }}
rgmin = {{ version = "0.6.0", default-features = false }}
rgflux = {{ path = "{}" }}
"#,
        env!("CARGO_MANIFEST_DIR")
    );
    fs::write(sandbox.join("Cargo.toml"), manifest).expect("write sandbox manifest");
}

#[test]
fn reference_pafi_defaults_match_the_code() {
    let reference = include_str!("../docs/orgmode/reference.org");
    let defaults = PafiConfig::default();
    let pafi_rows = extract_pafi_reference_rows(reference);
    assert_eq!(pafi_rows.len(), 9, "expected all PAFI rows to be present");
    assert_row_contains(
        &pafi_rows,
        "=planes=",
        format!("=usize= | ={}=", defaults.planes).as_str(),
    );
    assert_row_contains(
        &pafi_rows,
        "=burn_in_steps=",
        format!("=usize= | ={}=", defaults.burn_in_steps).as_str(),
    );
    assert_row_contains(&pafi_rows, "=sample_steps=", "=usize= | =4_000=");
    assert_row_contains(
        &pafi_rows,
        "=time_step=",
        format!("=f64= | ={}=", defaults.time_step).as_str(),
    );
    assert_row_contains(&pafi_rows, "=target_error=", "=Option<f64>= | =None=");
    assert_row_contains(
        &pafi_rows,
        "=max_rounds=",
        format!("=usize= | ={}=", defaults.max_rounds).as_str(),
    );
    assert_row_contains(
        &pafi_rows,
        "=zero_temperature_tol=",
        "=Option<f64>= | =None=",
    );
    assert!(defaults.zero_temperature_tol.is_none());
    let tolerance_row = pafi_rows
        .iter()
        .find(|row| row.contains("=zero_temperature_tol="))
        .expect("zero_temperature_tol row");
    let (documented, tolerance) = documented_approximate_value(tolerance_row);
    let intervals = (defaults.planes - 1) as f64;
    let effective =
        2.0 * (std::f64::consts::PI * std::f64::consts::PI / 12.0) / (intervals * intervals);
    let difference = (effective - documented).abs();
    assert!(
        difference <= tolerance,
        "effective default {effective} disagrees with documented {documented} by {difference}, above {tolerance}",
    );
    assert_row_contains(
        &pafi_rows,
        "=seed=",
        format!("=u64= | ={}=", defaults.seed).as_str(),
    );
    assert_row_contains(
        &pafi_rows,
        "=stats=",
        "=StatsConfig= | =StatsConfig::default()=",
    );
    assert_eq!(defaults.sample_steps, 4_000);
    assert!(defaults.target_error.is_none());
}

#[test]
fn reference_pafi_row_drop_is_detected() {
    let reference = include_str!("../docs/orgmode/reference.org");
    let dropped = reference
        .lines()
        .filter(|line| !line.contains("| =target_error= |"))
        .collect::<Vec<_>>()
        .join("\n");
    let rows = extract_pafi_reference_rows(&dropped);
    assert_eq!(rows.len(), 8);
}

fn extract_pafi_reference_rows(reference: &str) -> Vec<String> {
    let mut rows = Vec::new();
    let mut in_pafi_table = false;
    for line in reference.lines() {
        if line.trim() == "** =pafi::PafiConfig=" {
            in_pafi_table = true;
            continue;
        }
        if in_pafi_table && line.starts_with("** ") {
            break;
        }
        if in_pafi_table && line.starts_with("| =") {
            rows.push(line.to_string());
        }
    }
    rows
}

fn documented_approximate_value(row: &str) -> (f64, f64) {
    let marker = "\\approx ";
    let start = row
        .find(marker)
        .unwrap_or_else(|| panic!("row has no approximate value:\n{row}"));
    let rest = &row[start + marker.len()..];
    let mut end = 0usize;
    for (index, ch) in rest.char_indices() {
        if ch.is_ascii_digit() || ch == '.' {
            end = index + ch.len_utf8();
        } else {
            break;
        }
    }
    assert!(end > 0, "approximate value is empty:\n{row}");
    let numeric = &rest[..end];
    let value = numeric
        .parse::<f64>()
        .unwrap_or_else(|_| panic!("could not parse `{numeric}` from:\n{row}"));
    let decimals = numeric.split('.').nth(1).map(str::len).unwrap_or(0);
    let unit = 10_f64.powi(-(decimals as i32));
    (value, 0.5 * unit)
}

fn assert_row_contains(rows: &[String], field: &str, fragment: &str) {
    let row = rows
        .iter()
        .find(|row| row.contains(field))
        .unwrap_or_else(|| panic!("missing row for {field}"));
    assert!(
        row.contains(fragment),
        "row for {field} lacks expected fragment `{fragment}`:\n{row}"
    );
}

#[test]
fn getting_started_bennett_chandler_example_runs() -> Result<(), FluxError> {
    let trajectories = vec![
        ReactiveTrajectory::new(1.2, vec![Side::Product; 16])?,
        ReactiveTrajectory::new(-0.9, vec![Side::Reactant; 16])?,
        ReactiveTrajectory::new(0.8, vec![Side::Product; 16])?,
    ];
    let estimate = estimate_bennett_chandler(&trajectories, &BennettChandlerConfig::default())?;
    assert_eq!(estimate.plateau, 1.0);
    assert_eq!(estimate.standard_error, 0.0);
    assert!(estimate.stationarity.is_stationary);
    Ok(())
}

#[test]
fn getting_started_ohtst_example_runs() -> Result<(), FluxError> {
    struct QuarticDoubleWell1D;

    impl rgsaddle::PointSurface for QuarticDoubleWell1D {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
            let u = x[0];
            let well = u * u - 1.0;
            Ok((0.25 * well * well, Array1::from_vec(vec![u * well])))
        }
    }

    let cfg = NativeConfig::default()
        .with_masses(Array1::from_vec(vec![1.0]))
        .with_thermostat(
            ThermostatConfig::default()
                .with_gamma(1.0)
                .with_temperature(0.1),
        )
        .with_seed(17);
    let mut engine = NativeEngine::new(QuarticDoubleWell1D, cfg)?;
    engine.set_positions(Array1::from_vec(vec![-1.0]).view())?;

    let seed_plane = HyperplaneConstraint::from_direction(Array1::from_vec(vec![1.0]).view(), 0.0)?;
    let ohtst_cfg = OhtstConfig::default()
        .with_burn_in_steps(64)
        .with_min_sample_steps(128)
        .with_sample_step_factor(4)
        .with_seed(17);
    let estimate = optimize_hyperplane(&mut engine, &seed_plane, &ohtst_cfg)?;

    assert_eq!(estimate.plane.offset, 0.0);
    assert!(estimate.rate > 0.0 && estimate.standard_error.is_finite());
    Ok(())
}

#[test]
fn howto_hyperplane_sampling_example_runs() -> Result<(), FluxError> {
    struct DoubleWell2D;

    impl rgsaddle::PointSurface for DoubleWell2D {
        fn eval(&self, r: ArrayView1<f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
            let (x, y) = (r[0], r[1]);
            let well = x * x - 1.0;
            Ok((
                0.25 * well * well + 0.5 * y * y,
                Array1::from_vec(vec![x * well, y]),
            ))
        }
    }

    let mut engine = NativeEngine::new(
        DoubleWell2D,
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0, 1.0]))
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(1.0)
                    .with_temperature(0.2),
            )
            .with_seed(11),
    )?;
    engine.set_positions(Array1::from_vec(vec![-0.3, 0.0]).view())?;

    let plane =
        HyperplaneConstraint::from_direction(Array1::from_vec(vec![1.0, 0.0]).view(), -0.3)?;
    let cfg = HyperplaneConfig::default()
        .with_burn_in_steps(100)
        .with_sample_steps(400)
        .with_retention_stride(20)
        .with_time_step(0.01)
        .with_seed(11);

    let sample = sample_hyperplane(&mut engine, &plane, &cfg)?;
    assert!(sample.mean_force.is_finite());
    assert!(sample.standard_error >= 0.0);
    assert_eq!(sample.configurations.dim(), (20, 2));
    assert!(engine.constraint().is_none());
    Ok(())
}

#[test]
fn howto_ohtst_oblique_example_runs() -> Result<(), FluxError> {
    struct ObliqueDoubleWell2D;

    impl rgsaddle::PointSurface for ObliqueDoubleWell2D {
        fn eval(&self, r: ArrayView1<f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
            let u = 0.6 * r[0] + 0.8 * r[1];
            let v = -0.8 * r[0] + 0.6 * r[1];
            let well = u * u - 1.0;
            let du = u * well;
            let dv = 2.0 * v;
            Ok((
                0.25 * well * well + v * v,
                Array1::from_vec(vec![0.6 * du - 0.8 * dv, 0.8 * du + 0.6 * dv]),
            ))
        }
    }

    let mut engine = NativeEngine::new(
        ObliqueDoubleWell2D,
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![2.0, 8.0]))
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(1.0)
                    .with_temperature(0.1),
            )
            .with_seed(33),
    )?;
    engine.set_positions(Array1::from_vec(vec![-0.6, -0.8]).view())?;

    let seed_plane =
        HyperplaneConstraint::from_direction(Array1::from_vec(vec![3.0, 4.0]).view(), -0.4)?;
    let cfg = OhtstConfig::default()
        .with_max_outer_steps(16)
        .with_max_translation(0.25)
        .with_burn_in_steps(64)
        .with_min_sample_steps(128)
        .with_sample_step_factor(8)
        .with_time_step(0.01)
        .with_seed(33);

    let est = optimize_hyperplane(&mut engine, &seed_plane, &cfg)?;
    assert!(est.plane.offset.abs() < 1e-12);
    assert!(est.rate > 0.0 && est.standard_error.is_finite());
    Ok(())
}

#[test]
fn howto_ffs_example_runs() -> Result<(), FluxError> {
    struct QuarticDoubleWell1D;

    impl rgsaddle::PointSurface for QuarticDoubleWell1D {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
            let dx = x[0] * x[0] - 1.0;
            Ok((dx * dx, Array1::from_vec(vec![4.0 * x[0] * dx])))
        }
    }

    let cfg = NativeConfig::default()
        .with_masses(Array1::from_vec(vec![1.0]))
        .with_thermostat(
            ThermostatConfig::default()
                .with_gamma(2.0)
                .with_temperature(0.9),
        )
        .with_seed(11);
    let mut engine = NativeEngine::new(QuarticDoubleWell1D, cfg)?;
    engine.set_positions(Array1::from_vec(vec![-1.0]).view())?;
    engine.shoot_velocities(11)?;

    let ffs_cfg = FfsConfig::default()
        .with_interfaces(vec![-0.8, -0.4, -0.1, 0.2])
        .with_points_per_interface(16)
        .with_trials_per_interface(64)
        .with_max_trial_steps(3_000)
        .with_max_flux_steps(400_000)
        .with_burn_in_steps(50)
        .with_time_step(0.01)
        .with_seed(11);

    let est = forward_flux_rate(&mut engine, |x| x[0], &ffs_cfg)?;
    assert!(est.initial_flux > 0.0);
    assert_eq!(est.interface_probabilities.len(), 3);
    assert!(est.rate > 0.0 && est.standard_error > 0.0);
    assert_eq!(engine.positions()[0], -1.0);
    Ok(())
}

#[test]
fn howto_pafi_widening_well_example_runs() -> Result<(), FluxError> {
    struct WideningWell2D;

    impl rgsaddle::PointSurface for WideningWell2D {
        fn eval(&self, r: ArrayView1<f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
            let (x, y) = (r[0], r[1]);
            let well = 1.0 - x * x;
            let k = 1.5 + 0.5 * x;
            Ok((
                well * well + 0.5 * k * y * y,
                Array1::from_vec(vec![-4.0 * x * well + 0.25 * y * y, k * y]),
            ))
        }
    }

    let temperature = 0.2;
    let mut engine = NativeEngine::new(
        WideningWell2D,
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0, 1.0]))
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(1.5)
                    .with_temperature(temperature),
            )
            .with_seed(19),
    )?;

    let config = PafiConfig::default()
        .with_planes(33)
        .with_burn_in_steps(500)
        .with_sample_steps(8_000)
        .with_stats(StatsConfig::default().with_min_batches(4))
        .with_seed(19);
    let images = 5;
    let mut path = Array2::<f64>::zeros((images, 2));
    for k in 0..images {
        path[[k, 0]] = -1.0 + (k as f64) / ((images - 1) as f64);
    }

    let profile = pafi_profile(&mut engine, path.view(), &config)?;
    assert_eq!(profile.arc_lengths.len(), config.planes);
    assert_eq!(profile.free_energy[0], 0.0);
    assert_eq!(profile.dividing_index, config.planes - 1);

    let analytic = 1.0 + 0.5 * temperature * 1.5_f64.ln();
    assert!(profile.barrier_error > 0.0);
    assert!((profile.barrier - analytic).abs() < 5.0 * profile.barrier_error);
    assert!(
        profile.normalized_rate.is_some_and(|rate| rate > 0.0)
            && profile
                .normalized_rate_error
                .is_some_and(|error| error > 0.0)
    );
    Ok(())
}

#[test]
fn howto_bennett_chandler_shooting_example_runs() -> Result<(), FluxError> {
    struct SeparableDoubleWell2D;

    impl rgsaddle::PointSurface for SeparableDoubleWell2D {
        fn eval(&self, r: ArrayView1<f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
            let (x, y) = (r[0], r[1]);
            let dx = x * x - 1.0;
            Ok((
                dx * dx + y * y,
                Array1::from_vec(vec![4.0 * x * dx, 2.0 * y]),
            ))
        }
    }

    let gamma = 6.0;
    let omega_b = 2.0;
    let kramers = kramers_transmission_coefficient(gamma, omega_b)?;

    let dividing_surface = HyperplaneConstraint::new(Array1::from_vec(vec![1.0, 0.0]), 0.0)?;
    let mut engine = NativeEngine::new(
        SeparableDoubleWell2D,
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0, 1.0]))
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(gamma)
                    .with_temperature(0.08),
            )
            .with_seed(42),
    )?;

    let surface_sample = sample_hyperplane(
        &mut engine,
        &dividing_surface,
        &HyperplaneConfig::default()
            .with_burn_in_steps(100)
            .with_sample_steps(400)
            .with_retention_stride(10)
            .with_time_step(0.02)
            .with_seed(42),
    )?;

    let shoot_cfg = ShootingConfig::default()
        .with_steps(120)
        .with_time_step(0.05)
        .with_antithetic(true)
        .with_seed(99);
    let bc = shoot_bennett_chandler(
        &mut engine,
        &dividing_surface,
        surface_sample.configurations.view(),
        &shoot_cfg,
    )?;

    assert!(bc.plateau > 0.0 && bc.standard_error > 0.0);
    assert!(kramers > 0.3 && kramers < 0.31);
    Ok(())
}

#[test]
fn howto_eindir_surface_example_runs() -> Result<(), FluxError> {
    let oracle = rgmin::Oracle::unbounded(2, |x| {
        let energy = 0.5 * (x[0] * x[0] + 4.0 * x[1] * x[1]);
        let grad = Array1::from_vec(vec![x[0], 4.0 * x[1]]);
        (energy, grad)
    });
    let surface = EindirSurface::new(oracle);
    let mut engine = NativeEngine::new(
        surface,
        NativeConfig::default()
            .with_masses(Array1::from_vec(vec![1.0, 2.0]))
            .with_seed(7),
    )?;
    engine.shoot_velocities(7)?;
    let report = engine.step(0.01)?;
    assert!(report.potential_energy.is_finite() && report.kinetic_energy.is_finite());
    Ok(())
}
