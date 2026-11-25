//! PLUMED oracle for multithermal OPES.
//!
//! An `rgflux` run on the 2D double well records one frame per bias update.
//! `plumed driver` replays those frames through `OPES_EXPANDED` over
//! `ECV_MULTITHERMAL`, so both codes see the same energies, and the test
//! compares the per-frame bias, the bias force, every printed `Delta F`, the
//! final state, the reweighted averages, and restarts across the two codes.
//!
//! PLUMED is a test-only oracle. The test skips, printing the reason, unless
//! `RGFLUX_PLUMED` names a `plumed` executable, or one is on `PATH`, built
//! with the `opes` module. With `RGFLUX_REQUIRE_PLUMED=1` a missing or
//! unusable PLUMED fails the test instead, as in the CI oracle job. It was
//! verified against PLUMED 2.10.2 with the input in
//! `tests/data/opes_plumed/plumed.dat`.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use ndarray::{Array1, ArrayView1};
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::engine::{DynamicsEngine, ForceSurface, ThermostatConfig};
use rgflux::opes::{
    MultithermalReweighting, OpesMultithermal, OpesMultithermalConfig, OpesMultithermalSurface,
    OpesRunConfig, OpesTrajectory, SharedOpesMultithermal, TemperatureSpacing,
    run_opes_multithermal,
};
use rgflux::stats::StatsConfig;

const TEMPLATE: &str = include_str!("data/opes_plumed/plumed.dat");
const T0: f64 = 0.3;
const PACE: usize = 50;
const OBSERVATION_STEPS: usize = 100;
const PRINT_STRIDE: usize = 10;
const HALF_STEPS: usize = 200_000;
const TARGETS: [f64; 4] = [0.3, 0.5, 0.8, 1.2];
/// Agreement bound, relative to `1 + |value|`. The codes evaluate `U` with
/// different arithmetic, so a few units of `1e-15` per frame accumulate.
const AGREEMENT: f64 = 1e-9;

struct DoubleWell2D;

impl rgsaddle::PointSurface for DoubleWell2D {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
        let a = x[0] * x[0] - 1.0;
        Ok((
            a * a + 0.5 * x[1] * x[1],
            Array1::from_vec(vec![4.0 * x[0] * a, x[1]]),
        ))
    }
}

struct Case {
    name: &'static str,
    config: OpesMultithermalConfig,
    ladder: &'static str,
}

fn cases() -> Vec<Case> {
    let base = |config: OpesMultithermalConfig| {
        config
            .with_pace(PACE)
            .with_observation_steps(OBSERVATION_STEPS)
            .with_print_stride(PRINT_STRIDE)
    };
    vec![
        Case {
            name: "geometric TEMP_STEPS=6",
            config: base(OpesMultithermalConfig::range(T0, 1.2).with_temp_steps(6)),
            ladder: "TEMP_MIN=0.3 TEMP_MAX=1.2 TEMP_STEPS=6",
        },
        Case {
            name: "automatic NO_GEOM_SPACING",
            config: base(
                OpesMultithermalConfig::range(T0, 1.2)
                    .with_spacing(TemperatureSpacing::LinearInverse),
            ),
            ladder: "TEMP_MIN=0.3 TEMP_MAX=1.2 NO_GEOM_SPACING",
        },
        Case {
            name: "TEMP_SET_ALL",
            config: base(OpesMultithermalConfig::set_all(vec![
                0.3, 0.4, 0.55, 0.8, 1.2,
            ])),
            ladder: "TEMP_SET_ALL=0.3,0.4,0.55,0.8,1.2",
        },
    ]
}

/// A `plumed` with the `opes` module, or why there is none.
fn plumed_executable() -> Result<PathBuf, String> {
    let candidate = std::env::var_os("RGFLUX_PLUMED")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("plumed"));
    let output = Command::new(&candidate)
        .args(["config", "module", "opes"])
        .output()
        .map_err(|err| {
            format!(
                "cannot run {} ({err}); set RGFLUX_PLUMED or put plumed on PATH",
                candidate.display()
            )
        })?;
    let text = String::from_utf8_lossy(&output.stdout);
    if output.status.success() && text.contains("opes on") {
        Ok(candidate)
    } else {
        Err(format!(
            "{} lacks the opes module (`plumed config module opes` printed {:?}); configure PLUMED with --enable-modules=opes",
            candidate.display(),
            text.trim()
        ))
    }
}

fn plumed_required() -> bool {
    std::env::var("RGFLUX_REQUIRE_PLUMED").is_ok_and(|v| !v.is_empty() && v != "0")
}

fn plumed_version(plumed: &Path) -> String {
    let output = Command::new(plumed)
        .args(["info", "--long-version"])
        .output()
        .expect("plumed info");
    String::from_utf8_lossy(&output.stdout).trim().to_owned()
}

fn scratch(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("rgflux_opes_plumed_{label}_{nanos}"));
    fs::create_dir_all(&dir).expect("scratch directory");
    dir
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= AGREEMENT * (1.0 + a.abs().max(b.abs()))
}

/// Numeric rows of a PLUMED output file, skipping `#!` headers.
fn rows(path: &Path) -> Vec<Vec<f64>> {
    fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
        .lines()
        .filter(|line| !line.trim_start().starts_with('#') && !line.trim().is_empty())
        .map(|line| {
            line.split_whitespace()
                .map(|v| v.parse().expect("number"))
                .collect()
        })
        .collect()
}

/// Bias forces on atom 1 per frame from `--dump-forces`.
fn dumped_forces(path: &Path) -> Vec<[f64; 2]> {
    let text = fs::read_to_string(path).expect("forces");
    let lines: Vec<&str> = text.lines().collect();
    lines
        .chunks(3)
        .map(|frame| {
            let parts: Vec<f64> = frame[2]
                .split_whitespace()
                .skip(1)
                .map(|v| v.parse().expect("force"))
                .collect();
            [parts[0], parts[1]]
        })
        .collect()
}

fn write_frames(path: &Path, trajectory: &OpesTrajectory) {
    let mut text = String::new();
    for p in &trajectory.positions {
        text.push_str(&format!("1\n100 100 100\nX {:e} {:e} 0\n", p[0], p[1]));
    }
    fs::write(path, text).expect("write frames");
}

fn averages_block() -> (String, String) {
    let mut lines = String::new();
    let mut args = Vec::new();
    for (i, t) in TARGETS.iter().enumerate() {
        lines.push_str(&format!(
            "rw{i}: REWEIGHT_TEMP_PRESS TEMP={T0} REWEIGHT_TEMP={t} ENERGY=ene\n\
             avu{i}: AVERAGE ARG=ene LOGWEIGHTS=rb,rw{i}\n\
             avx{i}: AVERAGE ARG=x2 LOGWEIGHTS=rb,rw{i}\n"
        ));
        args.push(format!("avu{i},avx{i}"));
    }
    (lines, args.join(","))
}

fn run_plumed(plumed: &Path, dir: &Path, case: &Case, restart_from: Option<&Path>) {
    let (averages, average_args) = averages_block();
    let restart = restart_from.map_or(String::new(), |_| "RESTART".to_owned());
    let state_rfile =
        restart_from.map_or(String::new(), |p| format!("STATE_RFILE={}", p.display()));
    let input = TEMPLATE
        .replace("@RESTART@", &restart)
        .replace("@TEMP@", &T0.to_string())
        .replace("@LADDER@", case.ladder)
        .replace("@PACE@", &PACE.to_string())
        .replace("@OBS@", &OBSERVATION_STEPS.to_string())
        .replace("@PRINT@", &PRINT_STRIDE.to_string())
        .replace("@STATE_RFILE@", &state_rfile)
        .replace("@AVERAGES@", &averages)
        .replace("@AVERAGE_ARGS@", &average_args);
    fs::write(dir.join("plumed.dat"), input).expect("write plumed input");
    let output = Command::new(plumed)
        .args([
            "driver",
            "--ixyz",
            "traj.xyz",
            "--plumed",
            "plumed.dat",
            "--timestep",
            "1",
            "--trajectory-stride",
            &PACE.to_string(),
            "--dump-forces",
            "FORCES",
            "--dump-forces-fmt",
            "%.17g",
        ])
        .current_dir(dir)
        .output()
        .expect("run plumed driver");
    assert!(
        output.status.success(),
        "plumed driver failed in {}:\n{}",
        dir.display(),
        String::from_utf8_lossy(&output.stdout)
    );
}

struct Comparison {
    bias: f64,
    force: f64,
    deltafs: f64,
    rows: usize,
}

/// Compare one PLUMED replay with the `rgflux` frames it replayed.
fn compare_replay(dir: &Path, trajectory: &OpesTrajectory, first_step: u64) -> Comparison {
    let colvar = rows(&dir.join("COLVAR"));
    assert_eq!(colvar.len(), trajectory.steps.len());
    let mut worst = Comparison {
        bias: 0.0,
        force: 0.0,
        deltafs: 0.0,
        rows: 0,
    };
    for (row, (u, v)) in colvar
        .iter()
        .zip(trajectory.potential_energies.iter().zip(&trajectory.biases))
    {
        assert!(close(row[1], *u), "energy {} vs {u}", row[1]);
        assert!(close(row[3], *v), "bias {} vs {v}", row[3]);
        worst.bias = worst.bias.max((row[3] - v).abs());
    }
    let forces = dumped_forces(&dir.join("FORCES"));
    assert_eq!(forces.len(), trajectory.steps.len());
    for ((plumed_force, p), scale) in forces
        .iter()
        .zip(&trajectory.positions)
        .zip(&trajectory.force_scales)
    {
        let (_, unbiased) = DoubleWell2D
            .energy_and_forces(p.view(), None)
            .expect("unbiased force");
        for i in 0..2 {
            let ours = (scale - 1.0) * unbiased[i];
            assert!(
                close(plumed_force[i], ours),
                "bias force {} vs {ours}",
                plumed_force[i]
            );
            worst.force = worst.force.max((plumed_force[i] - ours).abs());
        }
    }
    let deltafs = rows(&dir.join("DELTAFS"));
    let history: Vec<_> = trajectory
        .delta_f_history
        .iter()
        .filter(|record| record.step >= first_step)
        .collect();
    assert_eq!(deltafs.len(), history.len(), "printed DELTAFS rows");
    for (row, record) in deltafs.iter().zip(history) {
        assert_eq!(row[0] as u64 + first_step, record.step);
        assert!(close(row[1], record.rct));
        for (plumed, ours) in row[2..].iter().zip(&record.delta_f) {
            assert!(close(*plumed, *ours), "Delta F {plumed} vs {ours}");
            worst.deltafs = worst.deltafs.max((plumed - ours).abs());
        }
    }
    worst.rows = deltafs.len();
    worst
}

fn simulate(case: &Case) -> (OpesTrajectory, OpesTrajectory, Vec<u8>) {
    let thermostat = ThermostatConfig::default()
        .with_temperature(T0)
        .with_gamma(1.0);
    let shared = SharedOpesMultithermal::new(
        OpesMultithermal::new(case.config.clone(), &thermostat).expect("bias"),
    );
    let surface = OpesMultithermalSurface::new(DoubleWell2D, shared.clone());
    let native = NativeConfig::default()
        .with_masses(Array1::from_vec(vec![1.0, 1.0]))
        .with_thermostat(thermostat)
        .with_seed(21);
    let mut engine = NativeEngine::new(surface, native).expect("engine");
    engine
        .set_positions(Array1::from_vec(vec![-1.0, 0.0]).view())
        .expect("start");
    engine.shoot_velocities(21).expect("velocities");
    let first_run = OpesRunConfig::default()
        .with_steps(HALF_STEPS)
        .with_time_step(0.01)
        .with_frame_stride(PACE);
    let first = run_opes_multithermal(&mut engine, &shared, &first_run).expect("first half");

    let mut state = Vec::new();
    first
        .bias
        .write_state(&mut state, HALF_STEPS as f64)
        .expect("state");
    let restored = OpesMultithermal::from_state(case.config.clone(), &thermostat, &state[..])
        .expect("restore");
    shared
        .with_bias_mut(|bias| *bias = restored)
        .expect("install restored bias");
    let second_run = first_run.with_start_step(HALF_STEPS as u64);
    let second = run_opes_multithermal(&mut engine, &shared, &second_run).expect("second half");
    (first, second, state)
}

#[test]
fn plumed_opes_expanded_agrees_with_rgflux() {
    let plumed = match plumed_executable() {
        Ok(plumed) => plumed,
        Err(reason) => {
            assert!(
                !plumed_required(),
                "RGFLUX_REQUIRE_PLUMED is set, so the PLUMED oracle may not skip: {reason}"
            );
            // The test harness captures `eprintln!` from passing tests; a
            // direct write to the stderr handle is always shown.
            let _ = writeln!(std::io::stderr(), "\nskipping the PLUMED oracle: {reason}");
            return;
        }
    };
    eprintln!("PLUMED {}", plumed_version(&plumed));
    let thermostat = ThermostatConfig::default().with_temperature(T0);
    for case in cases() {
        let (first, second, state) = simulate(&case);

        let dir1 = scratch("first");
        write_frames(&dir1.join("traj.xyz"), &first);
        run_plumed(&plumed, &dir1, &case, None);
        let c1 = compare_replay(&dir1, &first, 0);

        let plumed_state = fs::read(dir1.join("STATE")).expect("PLUMED state");
        let from_plumed =
            OpesMultithermal::from_state(case.config.clone(), &thermostat, &plumed_state[..])
                .expect("rgflux reads the PLUMED state");
        assert_eq!(from_plumed.counter(), first.bias.counter());
        assert_eq!(from_plumed.last_step(), None);
        assert_eq!(from_plumed.temperatures(), first.bias.temperatures());
        for (a, b) in from_plumed.delta_f().iter().zip(first.bias.delta_f()) {
            assert!(close(*a, *b));
        }

        let state_text = String::from_utf8_lossy(&state);
        for key in ["temperature", "boltzmann", "temp_min", "temp_max", "step"] {
            assert!(
                state_text.contains(&format!("# rgflux {key} ")),
                "PLUMED restarts from an rgflux state without its {key} entry"
            );
        }
        let dir2 = scratch("restart");
        let rgflux_state = dir2.join("RGFLUX_STATE");
        fs::write(&rgflux_state, &state).expect("write rgflux state");
        write_frames(&dir2.join("traj.xyz"), &second);
        run_plumed(&plumed, &dir2, &case, Some(&rgflux_state));
        let c2 = compare_replay(&dir2, &second, HALF_STEPS as u64);
        let final_plumed = OpesMultithermal::from_state(
            case.config.clone(),
            &thermostat,
            &fs::read(dir2.join("STATE")).expect("state")[..],
        )
        .expect("final PLUMED state");
        assert_eq!(final_plumed.counter(), second.bias.counter());

        eprintln!(
            "\n{}: {} temperatures; max |bias| diff {:.1e}, max |bias force| diff {:.1e}, max |Delta F| diff {:.1e} over {} DELTAFS rows; after restart from the rgflux state {:.1e}, {:.1e}, {:.1e} over {} rows",
            case.name,
            second.temperatures().len(),
            c1.bias,
            c1.force,
            c1.deltafs,
            c1.rows,
            c2.bias,
            c2.force,
            c2.deltafs,
            c2.rows
        );
        eprintln!("| T | Delta F rgflux | Delta F PLUMED | abs diff |");
        for ((t, ours), plumed) in second
            .temperatures()
            .iter()
            .zip(second.delta_f())
            .zip(final_plumed.delta_f())
        {
            assert!(close(*ours, *plumed));
            eprintln!(
                "| {t:.6} | {ours:+.15e} | {plumed:+.15e} | {:.1e} |",
                (ours - plumed).abs()
            );
        }

        let averages = rows(&dir1.join("AVERAGES"));
        let last = averages.last().expect("averages");
        let rw = MultithermalReweighting::from_bias(
            &first.bias,
            first.potential_energies[1..].to_vec(),
            first.biases[1..].to_vec(),
        )
        .expect("reweighting");
        let x2: Vec<f64> = first.positions[1..].iter().map(|p| p[0] * p[0]).collect();
        let stats = StatsConfig::default();
        eprintln!("| T | <U> rgflux | <U> PLUMED | <x^2> rgflux | <x^2> PLUMED | max abs diff |");
        for (i, &t) in TARGETS.iter().enumerate() {
            let u = rw
                .average(t, &first.potential_energies[1..], &stats)
                .expect("<U>")
                .mean;
            let x = rw.average(t, &x2, &stats).expect("<x^2>").mean;
            let (pu, px) = (last[1 + 2 * i], last[2 + 2 * i]);
            assert!(close(u, pu), "<U> at {t}: {u} vs {pu}");
            assert!(close(x, px), "<x^2> at {t}: {x} vs {px}");
            eprintln!(
                "| {t} | {u:.12} | {pu:.12} | {x:.12} | {px:.12} | {:.1e} |",
                (u - pu).abs().max((x - px).abs())
            );
        }
        fs::remove_dir_all(&dir1).expect("remove scratch");
        fs::remove_dir_all(&dir2).expect("remove scratch");
    }
}
