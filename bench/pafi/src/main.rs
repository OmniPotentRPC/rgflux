use std::collections::BTreeMap;
use std::env;
use std::process::ExitCode;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ndarray::{Array1, Array2, ArrayView1};
use rgflux::FluxError;
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::engine::{ForceSurface, ThermostatConfig};
use rgflux::pafi::{PafiConfig, pafi_profile};

#[derive(Default)]
struct Metrics {
    calls: AtomicU64,
    nanoseconds: Mutex<u128>,
}

struct MorseSurface {
    metrics: Arc<Metrics>,
}

impl ForceSurface for MorseSurface {
    fn energy_and_forces(
        &self,
        positions: ArrayView1<'_, f64>,
        _cell: Option<&[f64; 9]>,
    ) -> Result<(f64, Array1<f64>), FluxError> {
        let started = Instant::now();
        let mut gradient = vec![0.0; positions.len()];
        let energy =
            rgpot_pair::morse_gradient(1.0, 6.0, 1.0, 4.0, &positions.to_vec(), &mut gradient);
        self.metrics.calls.fetch_add(1, Ordering::Relaxed);
        let mut elapsed =
            self.metrics.nanoseconds.lock().map_err(|_| {
                FluxError::Surface("benchmark timing mutex was poisoned".to_string())
            })?;
        *elapsed += started.elapsed().as_nanos();
        Ok((energy, Array1::from_vec(gradient).mapv(|value| -value)))
    }
}

struct Arguments {
    case_name: String,
    planes: usize,
    temperature: f64,
    samples: usize,
    burn_in: usize,
    seed: u64,
    threads: usize,
    effective_threads: usize,
    time_step: f64,
    friction: f64,
    zero_temperature_tolerance: f64,
}

fn parse_arguments() -> Result<Arguments, String> {
    let mut values = BTreeMap::new();
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        if !flag.starts_with("--") {
            return Err(format!("unexpected positional argument: {flag}"));
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        values.insert(flag, value);
    }
    let take = |name: &str| {
        values
            .get(name)
            .cloned()
            .ok_or_else(|| format!("required argument {name} was not supplied"))
    };
    let parse = |name: &str| -> Result<f64, String> {
        take(name)?
            .parse()
            .map_err(|_| format!("{name} must be a number"))
    };
    let parse_usize = |name: &str| -> Result<usize, String> {
        take(name)?
            .parse()
            .map_err(|_| format!("{name} must be a positive integer"))
    };
    let parse_u64 = |name: &str| -> Result<u64, String> {
        take(name)?
            .parse()
            .map_err(|_| format!("{name} must be an unsigned integer"))
    };
    let expected = [
        "--case",
        "--planes",
        "--temperature",
        "--samples",
        "--burn-in",
        "--seed",
        "--threads",
        "--effective-threads",
        "--time-step",
        "--friction",
        "--zero-temperature-tolerance",
    ];
    if values.len() != expected.len() {
        let unknown: Vec<_> = values
            .keys()
            .filter(|key| !expected.contains(&key.as_str()))
            .cloned()
            .collect();
        if !unknown.is_empty() {
            return Err(format!("unknown arguments: {}", unknown.join(", ")));
        }
    }
    let case_name = take("--case")?;
    if case_name != "straight" && case_name != "curved" {
        return Err("--case must be straight or curved".to_string());
    }
    Ok(Arguments {
        case_name,
        planes: parse_usize("--planes")?,
        temperature: parse("--temperature")?,
        samples: parse_usize("--samples")?,
        burn_in: parse_usize("--burn-in")?,
        seed: parse_u64("--seed")?,
        threads: parse_usize("--threads")?,
        effective_threads: parse_usize("--effective-threads")?,
        time_step: parse("--time-step")?,
        friction: parse("--friction")?,
        zero_temperature_tolerance: parse("--zero-temperature-tolerance")?,
    })
}

fn dimer_path(case_name: &str, images: usize) -> Array2<f64> {
    let mut path = Array2::zeros((images, 6));
    for image in 0..images {
        let fraction = image as f64 / (images - 1) as f64;
        let (distance, angle) = if case_name == "straight" {
            (1.0 + 0.4 * fraction, 0.0)
        } else {
            (1.0, std::f64::consts::FRAC_PI_2 * fraction)
        };
        let half_x = 0.5 * distance * angle.cos();
        let half_y = 0.5 * distance * angle.sin();
        path[[image, 0]] = -half_x;
        path[[image, 1]] = -half_y;
        path[[image, 3]] = half_x;
        path[[image, 4]] = half_y;
    }
    path
}

fn static_energies(path: &Array2<f64>) -> Vec<f64> {
    path.rows()
        .into_iter()
        .map(|row| {
            let mut gradient = vec![0.0; row.len()];
            rgpot_pair::morse_gradient(1.0, 6.0, 1.0, 4.0, &row.to_vec(), &mut gradient)
        })
        .collect()
}

fn numbers(values: &[f64]) -> String {
    let body = values
        .iter()
        .map(|value| format!("{value:.17e}"))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{body}]")
}

fn escaped(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

fn run(arguments: &Arguments) -> String {
    let path = dimer_path(&arguments.case_name, arguments.planes);
    let energies = static_energies(&path);
    let metrics = Arc::new(Metrics::default());
    let surface = MorseSurface {
        metrics: Arc::clone(&metrics),
    };
    let thermostat = ThermostatConfig::default()
        .with_gamma(arguments.friction)
        .with_temperature(arguments.temperature)
        .with_boltzmann(1.0);
    let mut engine = match NativeEngine::new(
        surface,
        NativeConfig::default()
            .with_masses(Array1::from_elem(6, 1.0))
            .with_seed(arguments.seed)
            .with_thermostat(thermostat),
    ) {
        Ok(engine) => engine,
        Err(error) => {
            return format!(
                "{{\"engine\":\"rgflux\",\"status\":\"error\",\"error\":\"{}\"}}",
                escaped(&error.to_string())
            );
        }
    };
    let config = PafiConfig::default()
        .with_planes(arguments.planes)
        .with_burn_in_steps(arguments.burn_in)
        .with_sample_steps(arguments.samples)
        .with_time_step(arguments.time_step)
        .with_zero_temperature_tol(arguments.zero_temperature_tolerance)
        .with_seed(arguments.seed);
    let started = Instant::now();
    let result = pafi_profile(&mut engine, path.view(), &config);
    let wall = started.elapsed();
    let potential_nanoseconds =
        u64::try_from(*metrics.nanoseconds.lock().expect("timing mutex fits")).unwrap_or(u64::MAX);
    let potential = Duration::from_nanos(potential_nanoseconds);
    let overhead = wall.saturating_sub(potential);
    let common = format!(
        "\"engine\":\"rgflux\",\"case\":\"{}\",\"planes\":{},\"temperature\":{:.17e},\"samples\":{},\"burn_in\":{},\"seed\":{},\"requested_threads\":{},\"effective_threads\":{},\"time_step\":{:.17e},\"friction\":{:.17e},\"zero_temperature_tolerance\":{:.17e},\"wall_seconds\":{:.17e},\"potential_seconds\":{:.17e},\"overhead_seconds\":{:.17e},\"force_calls\":{},\"minimum_energy\":{:.17e},\"saddle_energy\":{:.17e}",
        arguments.case_name,
        arguments.planes,
        arguments.temperature,
        arguments.samples,
        arguments.burn_in,
        arguments.seed,
        arguments.threads,
        arguments.effective_threads,
        arguments.time_step,
        arguments.friction,
        arguments.zero_temperature_tolerance,
        wall.as_secs_f64(),
        potential.as_secs_f64(),
        overhead.as_secs_f64(),
        metrics.calls.load(Ordering::Relaxed),
        energies.iter().copied().fold(f64::INFINITY, f64::min),
        energies.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    );
    match result {
        Ok(profile) => format!(
            "{{{common},\"status\":\"ok\",\"arc_lengths\":{},\"free_energy\":{},\"standard_error\":{},\"barrier\":{:.17e},\"barrier_error\":{:.17e}}}",
            numbers(&profile.arc_lengths),
            numbers(&profile.free_energy),
            numbers(&profile.standard_error),
            profile.barrier,
            profile.barrier_error,
        ),
        Err(error) => format!(
            "{{{common},\"status\":\"rejected\",\"error\":\"{}\"}}",
            escaped(&error.to_string())
        ),
    }
}

fn main() -> ExitCode {
    match parse_arguments() {
        Ok(arguments) => {
            println!("{}", run(&arguments));
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Arguments, run};

    #[test]
    fn reproduces_zero_width_reactant_partition() {
        let record = run(&Arguments {
            case_name: "curved".to_string(),
            planes: 17,
            temperature: 0.05,
            samples: 64,
            burn_in: 128,
            seed: 1_036_330,
            threads: 1,
            effective_threads: 1,
            time_step: 0.002,
            friction: 1.0,
            zero_temperature_tolerance: 0.01,
        });
        let expected = "invalid configuration: PAFI reactant partition must be finite and positive";
        assert!(
            record.contains("\"status\":\"rejected\"") && record.contains(expected),
            "zero-width reactant partition at dividing plane index 0 was not reproduced: {record}"
        );
    }
}
