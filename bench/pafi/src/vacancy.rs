use std::collections::BTreeMap;
use std::ffi::CString;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use rgflux::FluxError;
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::engine::{ForceSurface, ThermostatConfig};
use rgflux::pafi::{PafiConfig, pafi_profile};
use rgsaddle::{BandConfig, BandSession, BandStatus, BandSurface, Cell, SaddleError};
use serde::{Deserialize, Serialize};

const LATTICE: f64 = 3.92;
const MASS: f64 = 195.084;
const NEB_IMAGES: usize = 9;

#[derive(Clone, Serialize, Deserialize)]
struct VacancyPath {
    generator_sha: String,
    cell: [f64; 9],
    atomic_numbers: Vec<i32>,
    images: Vec<Vec<f64>>,
    neb_iterations: usize,
    neb_max_force: f64,
}

#[derive(Default)]
struct Metrics {
    calls: AtomicU64,
    nanoseconds: Mutex<u128>,
}

struct PythonMorse {
    evaluate: Py<PyAny>,
    metrics: Arc<Metrics>,
}

impl PythonMorse {
    fn new(
        site_packages: &str,
        atom_count: usize,
        cell: [f64; 9],
        threads: usize,
    ) -> Result<Self, String> {
        let evaluate = Python::attach(|py| -> PyResult<Py<PyAny>> {
            let sys = PyModule::import(py, "sys")?;
            sys.getattr("path")?
                .call_method1("insert", (0, site_packages))?;
            let source = CString::new(format!(
                r#"
import ctypes
import ctypes.util
import numpy as np
import rgpot

_gomp = ctypes.CDLL(ctypes.util.find_library("gomp"))
_gomp.omp_set_num_threads.argtypes = [ctypes.c_int]
_gomp.omp_set_num_threads({threads})
_atoms = np.full({atom_count}, 78, dtype=np.int32)
_cell = np.asarray({cell:?}, dtype=np.float64).reshape(3, 3)
_potential = rgpot.MorsePot(De=0.7102, a=1.6047, re=2.897, cutoff=5.5)

def evaluate(flat):
    positions = np.asarray(flat, dtype=np.float64).reshape({atom_count}, 3)
    energy, forces, variance = _potential(positions, _atoms, _cell)
    return float(energy), np.asarray(forces).reshape(-1).tolist()
"#
            ))
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
            let filename = CString::new("vacancy_surface.py").expect("literal");
            let module_name = CString::new("vacancy_surface").expect("literal");
            let module = PyModule::from_code(
                py,
                source.as_c_str(),
                filename.as_c_str(),
                module_name.as_c_str(),
            )?;
            module.getattr("evaluate").map(Bound::unbind)
        })
        .map_err(|error| error.to_string())?;
        Ok(Self {
            evaluate,
            metrics: Arc::new(Metrics::default()),
        })
    }

    fn calculate(&self, positions: ArrayView1<'_, f64>) -> Result<(f64, Vec<f64>), String> {
        let started = Instant::now();
        let result = Python::attach(|py| {
            self.evaluate
                .call1(py, (positions.to_vec(),))?
                .extract::<(f64, Vec<f64>)>(py)
        })
        .map_err(|error| error.to_string());
        self.metrics.calls.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut elapsed) = self.metrics.nanoseconds.lock() {
            *elapsed += started.elapsed().as_nanos();
        }
        result
    }
}

impl ForceSurface for PythonMorse {
    fn energy_and_forces(
        &self,
        positions: ArrayView1<'_, f64>,
        _cell: Option<&[f64; 9]>,
    ) -> Result<(f64, Array1<f64>), FluxError> {
        let (energy, forces) = self.calculate(positions).map_err(FluxError::Surface)?;
        Ok((energy, Array1::from_vec(forces)))
    }
}

impl BandSurface for PythonMorse {
    fn eval(
        &self,
        positions: ArrayView2<'_, f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (row_index, row) in positions.rows().into_iter().enumerate() {
            let (energy, forces) = self.calculate(row).map_err(SaddleError::Surface)?;
            energies[row_index] = energy;
            for (column, force) in forces.into_iter().enumerate() {
                gradients[(row_index, column)] = -force;
            }
        }
        Ok(())
    }
}

fn required(arguments: &BTreeMap<String, String>, name: &str) -> Result<String, String> {
    arguments
        .get(name)
        .cloned()
        .ok_or_else(|| format!("required argument {name} was not supplied"))
}

fn parse_flags(values: impl Iterator<Item = String>) -> Result<BTreeMap<String, String>, String> {
    let mut parsed = BTreeMap::new();
    let mut values = values;
    while let Some(flag) = values.next() {
        if !flag.starts_with("--") {
            return Err(format!("unexpected positional argument: {flag}"));
        }
        let value = values
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        parsed.insert(flag, value);
    }
    Ok(parsed)
}

fn fcc_endpoints() -> ([f64; 9], Vec<i32>, Vec<f64>, Vec<f64>) {
    let cell = [
        3.0 * LATTICE,
        0.0,
        0.0,
        0.0,
        4.0 * LATTICE,
        0.0,
        0.0,
        0.0,
        4.0 * LATTICE,
    ];
    let bases = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ];
    let vacancy = [LATTICE, LATTICE, LATTICE];
    let hopping = [1.5 * LATTICE, 1.5 * LATTICE, LATTICE];
    let mut start = Vec::new();
    let mut hopping_offset = 0usize;
    for ix in 0..3 {
        for iy in 0..4 {
            for iz in 0..4 {
                for basis in bases {
                    let point = [
                        (ix as f64 + basis[0]) * LATTICE,
                        (iy as f64 + basis[1]) * LATTICE,
                        (iz as f64 + basis[2]) * LATTICE,
                    ];
                    if point == vacancy {
                        continue;
                    }
                    if point == hopping {
                        hopping_offset = start.len();
                    }
                    start.extend_from_slice(&point);
                }
            }
        }
    }
    let mut end = start.clone();
    end[hopping_offset..hopping_offset + 3].copy_from_slice(&vacancy);
    let atom_count = start.len() / 3;
    (cell, vec![78; atom_count], start, end)
}

fn generate(arguments: &BTreeMap<String, String>) -> Result<(), String> {
    let site_packages = required(arguments, "--site-packages")?;
    let output = PathBuf::from(required(arguments, "--output")?);
    let generator_sha = required(arguments, "--rgsaddle-sha")?;
    let (cell, atomic_numbers, start, end) = fcc_endpoints();
    let dof = start.len();
    let initial = Array2::from_shape_fn((NEB_IMAGES, dof), |(image, coordinate)| {
        let fraction = image as f64 / (NEB_IMAGES - 1) as f64;
        start[coordinate] + fraction * (end[coordinate] - start[coordinate])
    });
    let surface = PythonMorse::new(&site_packages, atomic_numbers.len(), cell, 1)?;
    let cell_matrix = Cell([
        [cell[0], cell[1], cell[2]],
        [cell[3], cell[4], cell[5]],
        [cell[6], cell[7], cell[8]],
    ]);
    let config = BandConfig::default()
        .with_cell(cell_matrix)
        .with_min_mode(None)
        .with_saddle_check(None);
    let mut session = BandSession::new(config, initial).map_err(|error| error.to_string())?;
    let mut final_iteration = 0usize;
    let mut final_force = f64::INFINITY;
    for _ in 0..2_000 {
        let report = session.step(&surface).map_err(|error| error.to_string())?;
        final_iteration = report.iteration;
        final_force = report.max_force;
        if report.status == BandStatus::Converged {
            let path = VacancyPath {
                generator_sha,
                cell,
                atomic_numbers,
                images: session
                    .positions()
                    .rows()
                    .into_iter()
                    .map(|row| row.to_vec())
                    .collect(),
                neb_iterations: final_iteration,
                neb_max_force: final_force,
            };
            fs::write(
                output,
                serde_json::to_string_pretty(&path).map_err(|error| error.to_string())? + "\n",
            )
            .map_err(|error| error.to_string())?;
            return Ok(());
        }
    }
    Err(format!(
        "NEB did not converge in {final_iteration} iterations; max force {final_force}"
    ))
}

fn numbers(values: &[f64]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|value| format!("{value:.17e}"))
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn run_profile(arguments: &BTreeMap<String, String>) -> Result<(), String> {
    let site_packages = required(arguments, "--site-packages")?;
    let path_file = PathBuf::from(required(arguments, "--path")?);
    let planes: usize = required(arguments, "--planes")?
        .parse()
        .map_err(|_| "--planes must be an integer")?;
    let temperature: f64 = required(arguments, "--temperature")?
        .parse()
        .map_err(|_| "--temperature must be a number")?;
    let samples: usize = required(arguments, "--samples")?
        .parse()
        .map_err(|_| "--samples must be an integer")?;
    let burn_in: usize = required(arguments, "--burn-in")?
        .parse()
        .map_err(|_| "--burn-in must be an integer")?;
    let seed: u64 = required(arguments, "--seed")?
        .parse()
        .map_err(|_| "--seed must be an integer")?;
    let threads: usize = required(arguments, "--threads")?
        .parse()
        .map_err(|_| "--threads must be an integer")?;
    let time_step: f64 = required(arguments, "--time-step")?
        .parse()
        .map_err(|_| "--time-step must be a number")?;
    let friction: f64 = required(arguments, "--friction")?
        .parse()
        .map_err(|_| "--friction must be a number")?;
    let tolerance: f64 = required(arguments, "--zero-temperature-tolerance")?
        .parse()
        .map_err(|_| "--zero-temperature-tolerance must be a number")?;
    let path: VacancyPath =
        serde_json::from_str(&fs::read_to_string(path_file).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    let path_array = Array2::from_shape_vec(
        (path.images.len(), path.images[0].len()),
        path.images.into_iter().flatten().collect(),
    )
    .map_err(|error| error.to_string())?;
    let surface = PythonMorse::new(
        &site_packages,
        path.atomic_numbers.len(),
        path.cell,
        threads,
    )?;
    let metrics = Arc::clone(&surface.metrics);
    let thermostat = ThermostatConfig::default()
        .with_gamma(friction)
        .with_temperature(temperature)
        .with_boltzmann(1.0);
    let mut engine = NativeEngine::new(
        surface,
        NativeConfig::default()
            .with_masses(Array1::from_elem(path_array.ncols(), MASS))
            .with_cell(path.cell)
            .with_seed(seed)
            .with_thermostat(thermostat),
    )
    .map_err(|error| error.to_string())?;
    let config = PafiConfig::default()
        .with_planes(planes)
        .with_burn_in_steps(burn_in)
        .with_sample_steps(samples)
        .with_time_step(time_step)
        .with_zero_temperature_tol(tolerance)
        .with_seed(seed);
    let started = Instant::now();
    let profile = pafi_profile(&mut engine, path_array.view(), &config);
    let wall = started.elapsed();
    let potential_nanoseconds = u64::try_from(
        *metrics
            .nanoseconds
            .lock()
            .map_err(|error| error.to_string())?,
    )
    .unwrap_or(u64::MAX);
    let potential = Duration::from_nanos(potential_nanoseconds);
    let overhead = wall.saturating_sub(potential);
    let common = format!(
        "\"engine\":\"rgflux\",\"system\":\"fcc-vacancy\",\"planes\":{planes},\"temperature\":{temperature:.17e},\"samples\":{samples},\"burn_in\":{burn_in},\"seed\":{seed},\"requested_threads\":{threads},\"effective_threads\":{threads},\"time_step\":{time_step:.17e},\"friction\":{friction:.17e},\"zero_temperature_tolerance\":{tolerance:.17e},\"wall_seconds\":{:.17e},\"potential_seconds\":{:.17e},\"overhead_seconds\":{:.17e},\"force_calls\":{},\"atom_count\":{}",
        wall.as_secs_f64(),
        potential.as_secs_f64(),
        overhead.as_secs_f64(),
        metrics.calls.load(Ordering::Relaxed),
        path.atomic_numbers.len(),
    );
    match profile {
        Ok(profile) => println!(
            "{{{common},\"status\":\"ok\",\"arc_lengths\":{},\"free_energy\":{},\"standard_error\":{},\"barrier\":{:.17e},\"barrier_error\":{:.17e}}}",
            numbers(&profile.arc_lengths),
            numbers(&profile.free_energy),
            numbers(&profile.standard_error),
            profile.barrier,
            profile.barrier_error,
        ),
        Err(error) => println!(
            "{{{common},\"status\":\"rejected\",\"error\":{}}}",
            serde_json::to_string(&error.to_string()).map_err(|error| error.to_string())?
        ),
    }
    Ok(())
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(command) = args.next() else {
        eprintln!("required command generate or run was not supplied");
        return ExitCode::from(2);
    };
    let result = parse_flags(args).and_then(|arguments| match command.as_str() {
        "generate" => generate(&arguments),
        "run" => run_profile(&arguments),
        _ => Err("command must be generate or run".to_string()),
    });
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(1)
        }
    }
}
