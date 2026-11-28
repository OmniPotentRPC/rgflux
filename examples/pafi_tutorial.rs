//! PAFI tutorial: free-energy profile and rate on an analytic double well.
//!
//! `V(x, y) = (x² - 1)² + ½ k(x) y²` with `k(x) = 2.5 + 1.5 x`. The
//! valley floor is the straight line `y = 0`: at fixed `x` the minimum
//! over `y` is at `y = 0`, where `V = (x² - 1)²`, the plain double well.
//! Minima sit at `x = ∓1` (both `V = 0`), the saddle at the origin
//! (`V = 1`). The transverse stiffness differs at the two minima,
//! `k(-1) = 1.0` and `k(+1) = 4.0`, and at the saddle `k(0) = 2.5`, so the
//! barrier picks up a temperature-dependent entropic term. The path is a
//! straight line, so the reactant-to-saddle half of the valley is the
//! soft, floppy side and the saddle-to-product half is the stiff side.
//!
//! Run directly for the walkthrough in `tutorials/pafi`:
//!
//! ```console
//! $ cargo run --release --example pafi_tutorial
//! ```
//!
//! Run with two paths for the JSON mode `docs/orgmode/tutorials/pafi.org`
//! uses from Python: the first argument is a config file, the second is
//! where the result is written.
//!
//! ```console
//! $ cargo run --release --example pafi_tutorial -- config.json result.json
//! ```

use std::env;
use std::fs;

use ndarray::{Array1, ArrayView1, array};
use rgflux::defect_pair::attempt_frequency;
use rgflux::pafi::cartesian::{PafiConfig, pafi_profile, zero_temperature_barrier};
use rgsaddle::{ArcPath, PointSurface, SaddleError};

/// `k(x) = 2.5 + 1.5 x`.
fn stiffness(x: f64) -> f64 {
    2.5 + 1.5 * x
}

struct CurvedWell;

impl PointSurface for CurvedWell {
    fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (x, y) = (q[0], q[1]);
        let k = stiffness(x);
        let well = x * x - 1.0;
        let energy = well * well + 0.5 * k * y * y;
        let dk_dx = 1.5;
        let dvdx = 4.0 * x * well + 0.5 * dk_dx * y * y;
        let dvdy = k * y;
        Ok((energy, array![dvdx, dvdy]))
    }
}

/// `n_images` points on the exact valley floor, the straight line `y = 0`
/// from `x0` to `x1`, reactant first.
fn valley_path(x0: f64, x1: f64, n_images: usize) -> Vec<Array1<f64>> {
    (0..n_images)
        .map(|i| {
            let t = i as f64 / (n_images - 1) as f64;
            let x = x0 + (x1 - x0) * t;
            array![x, 0.0]
        })
        .collect()
}

/// The reactant's attempt frequency along the path tangent at `arc = 0`,
/// from the exact Hessian of the energy, unit mass.
fn reactant_attempt_frequency(path: &ArcPath) -> Result<f64, SaddleError> {
    let geometry = path.geometry(path.bounds().0)?;
    let tangent_norm = geometry.tangent.dot(&geometry.tangent).sqrt();
    let direction = &geometry.tangent / tangent_norm;
    let (x, y) = (geometry.positions[0], geometry.positions[1]);
    let (dx, dy) = (direction[0], direction[1]);
    let hxx = 12.0 * x * x - 4.0;
    let hxy = 1.5 * y;
    let hyy = stiffness(x);
    let lambda = hxx * dx * dx + 2.0 * hxy * dx * dy + hyy * dy * dy;
    attempt_frequency(lambda, 1.0, 1.0)
}

/// Barrier of the exact transverse-Gaussian free energy at `kt`, used only
/// to check the sampled profile: with `y` integrated out analytically,
/// `F(x) = (x² - 1)² - ½ kt ln(2π kt / k(x))`, so
/// `ΔF = F(0) - F(-1) = 1 + ½ kt ln(k(0) / k(-1))`.
fn closed_form_barrier(kt: f64) -> f64 {
    1.0 + 0.5 * kt * (stiffness(0.0) / stiffness(-1.0)).ln()
}

struct TutorialConfig {
    kt: f64,
    n_images: usize,
    n_planes: usize,
    pafi: PafiConfig,
}

impl Default for TutorialConfig {
    fn default() -> Self {
        let kt = 0.1;
        Self {
            kt,
            n_images: 21,
            n_planes: 49,
            pafi: PafiConfig::default()
                .with_thermal(kt)
                .with_mobility(1.0)
                .with_time_step(2.0e-3)
                .with_thermalize(2_000)
                .with_stride(4)
                .with_samples(30_000)
                .with_blocks(24)
                .with_seed(7)
                .with_fix_com(false)
                .with_retain(0),
        }
    }
}

struct TutorialResult {
    barrier: f64,
    barrier_error: f64,
    closed_form: f64,
    cold_barrier: f64,
    nu: f64,
    rate: f64,
    tau_max: f64,
    n_eff_min: f64,
}

fn run(config: &TutorialConfig) -> Result<TutorialResult, SaddleError> {
    let surface = CurvedWell;
    let images = valley_path(-1.0, 1.0, config.n_images);
    let path = ArcPath::from_positions(&images)?;

    let cold = zero_temperature_barrier(&surface, &path, &path.arcs())?;

    let arcs: Vec<f64> = {
        let (first, last) = path.bounds();
        (0..config.n_planes)
            .map(|k| first + (last - first) * k as f64 / (config.n_planes - 1) as f64)
            .collect()
    };
    let (profile, planes) = pafi_profile(&surface, &path, &arcs, &config.pafi)?;

    let nu = reactant_attempt_frequency(&path)?;
    // Arrhenius law in the surface's own units: attempt_frequency already
    // carries 1 / time, and PafiConfig::thermal is kT in the surface's
    // energy unit, so this is k = nu * exp(-barrier / kT) with no unit
    // conversion (unlike crate::arrhenius, which is fixed to eV and K).
    let rate = nu * (-profile.barrier / config.kt).exp();

    let tau_max = planes.iter().fold(0.0_f64, |m, p| m.max(p.tau));
    let n_eff_min = planes.iter().fold(f64::INFINITY, |m, p| m.min(p.n_eff));

    Ok(TutorialResult {
        barrier: profile.barrier,
        barrier_error: profile.barrier_error,
        closed_form: closed_form_barrier(config.kt),
        cold_barrier: cold.barrier,
        nu,
        rate,
        tau_max,
        n_eff_min,
    })
}

fn print_report(config: &TutorialConfig, result: &TutorialResult) {
    println!(
        "zero-temperature barrier along the path: {:.6} (potential barrier is 1)",
        result.cold_barrier
    );
    println!(
        "PAFI barrier at kT = {}: {:.5} +/- {:.5}",
        config.kt, result.barrier, result.barrier_error
    );
    println!(
        "closed-form transverse-Gaussian barrier at this kT: {:.5}",
        result.closed_form
    );
    println!("reactant attempt frequency (unit mass): {:.5}", result.nu);
    println!("rate k = nu * exp(-barrier / kT): {:.6e}", result.rate);
    println!(
        "largest plane autocorrelation time: {:.2} samples; smallest n_eff: {:.1}",
        result.tau_max, result.n_eff_min
    );
}

/// Reads one `"key": value` field out of a flat JSON object. Good enough
/// for the fixed, caller-controlled schema this example reads; it is not
/// a general JSON parser.
fn json_field(text: &str, key: &str) -> Result<f64, SaddleError> {
    let needle = format!("\"{key}\"");
    let at = text
        .find(&needle)
        .ok_or_else(|| SaddleError::Invalid(format!("missing field {key}")))?;
    let after_colon = text[at + needle.len()..]
        .find(':')
        .ok_or_else(|| SaddleError::Invalid(format!("missing colon after {key}")))?;
    let rest = &text[at + needle.len() + after_colon + 1..];
    let end = rest.find([',', '}', '\n']).unwrap_or(rest.len());
    rest[..end]
        .trim()
        .parse::<f64>()
        .map_err(|e| SaddleError::Invalid(format!("field {key} is not a number: {e}")))
}

fn run_json(input: &str, output: &str) -> Result<(), SaddleError> {
    let text = fs::read_to_string(input)
        .map_err(|e| SaddleError::Invalid(format!("cannot read {input}: {e}")))?;
    let config = TutorialConfig {
        kt: json_field(&text, "kt")?,
        n_images: json_field(&text, "n_images")? as usize,
        n_planes: json_field(&text, "n_planes")? as usize,
        pafi: PafiConfig::default()
            .with_thermal(json_field(&text, "kt")?)
            .with_mobility(1.0)
            .with_time_step(2.0e-3)
            .with_thermalize(json_field(&text, "thermalize")? as usize)
            .with_stride(json_field(&text, "stride")? as usize)
            .with_samples(json_field(&text, "samples")? as usize)
            .with_blocks(json_field(&text, "blocks")? as usize)
            .with_seed(json_field(&text, "seed")? as u64)
            .with_fix_com(false)
            .with_retain(0),
    };
    let result = run(&config)?;
    let out = format!(
        "{{\n  \"cold_barrier\": {},\n  \"barrier\": {},\n  \"barrier_error\": {},\n  \"closed_form\": {},\n  \"nu\": {},\n  \"rate\": {},\n  \"tau_max\": {},\n  \"n_eff_min\": {}\n}}\n",
        result.cold_barrier,
        result.barrier,
        result.barrier_error,
        result.closed_form,
        result.nu,
        result.rate,
        result.tau_max,
        result.n_eff_min,
    );
    fs::write(output, out)
        .map_err(|e| SaddleError::Invalid(format!("cannot write {output}: {e}")))?;
    Ok(())
}

fn main() -> Result<(), SaddleError> {
    let args: Vec<String> = env::args().collect();
    if args.len() >= 3 {
        return run_json(&args[1], &args[2]);
    }
    let config = TutorialConfig::default();
    let result = run(&config)?;
    print_report(&config, &result);
    Ok(())
}
