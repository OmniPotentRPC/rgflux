//! Bennett–Chandler transmission coefficient.
//!
//! Bennett, in Algorithms for Chemical Computations, ACS Symposium Series
//! 46, 63 (1977), doi:10.1021/bk-1977-0046.ch004, and Chandler, J. Chem.
//! Phys. 68, 2959 (1978), doi:10.1063/1.436049, write the recrossing factor as the plateau of the
//! flux correlation through a dividing surface. For a surface with unit
//! normal `τ` and configurations drawn from the constrained ensemble,
//!
//! `κ(t) = ⟨ v_τ(0) θ(s(t)) ⟩ / ⟨ v_τ(0) θ(v_τ(0)) ⟩`,
//!
//! where `s(t) = (x(t) - x_surface) · τ` and `θ(s)` is 1 on the product
//! side. The numerator keeps the sign of the initial velocity, so a
//! trajectory that starts toward the reactant and finishes in the product
//! counts against the flux. The denominator is the positive flux. The
//! average is over Maxwell–Boltzmann velocities. Trajectories are
//! inertial Langevin, BAOAB,
//!
//! `m dv = (-∇E - m γ v) dt + sqrt(2 m γ kT) dW`,
//!
//! integrated by a velocity-Verlet / Ornstein–Uhlenbeck split. `γ = 0` is
//! Newtonian. The reported `κ` is `κ(t)` at the requested plateau record.
//! The standard error is the block standard error of that ratio.
//!
//! Units follow the surface: `kT` and energies in eV, positions in
//! Angstrom, masses in amu, and time in `sqrt(amu Angstrom^2 / eV)`, which
//! is 10.18 fs.

use ndarray::{Array1, ArrayView1};

use rgsaddle::error::SaddleError;
use rgsaddle::minmode::PointSurface;
use rgsaddle::stats::Rng;

/// Inertial shooting from one dividing surface.
///
/// Defaults: `thermal = 0.02585` eV (300 K), `mass = 1.0` amu,
/// `friction = 0.0` (Newtonian), `time_step = 1.0e-3`, `steps = 200`,
/// `record_every = 10`, `shots = 64`, `blocks = 4`, `seed = 1`,
/// `plateau_record = 0`.
///
/// Non-exhaustive: start from [`ShootingConfig::default`] and chain the
/// `with_*` setters.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ShootingConfig {
    /// `kT` in eV (0.02585 eV is 300 K). Must be positive and finite.
    pub thermal: f64,
    /// Mass of every degree of freedom, in amu. Must be positive and finite.
    pub mass: f64,
    /// Langevin friction `γ` in inverse time (1/(10.18 fs) for eV, Angstrom
    /// and amu). Zero is Newtonian. Must be finite and nonnegative.
    pub friction: f64,
    /// Velocity-Verlet time step in the time unit of the surface (10.18 fs
    /// for eV, Angstrom and amu). Must be positive and finite.
    pub time_step: f64,
    /// Integration steps after the initial velocity is drawn. At least one.
    pub steps: usize,
    /// Steps between stored values of `κ(t)`. At least one.
    pub record_every: usize,
    /// Shots. Starts are cycled if there are fewer configurations than
    /// shots. At least one, and at least `blocks`.
    pub shots: usize,
    /// Blocks of shots for the standard error of the plateau value. At
    /// least two. Each block holds `shots / blocks` shots; a remainder is not
    /// used by the error estimate.
    pub blocks: usize,
    /// Seed of the velocity draws. The result is a function of the seed.
    pub seed: u64,
    /// Index into the stored records that is the plateau. Zero is the
    /// first record, at time `record_every * time_step`. It must be below
    /// `ceil(steps / record_every)`.
    pub plateau_record: usize,
}

with_setters!(ShootingConfig {
    with_thermal => thermal: val f64,
    with_mass => mass: val f64,
    with_friction => friction: val f64,
    with_time_step => time_step: val f64,
    with_steps => steps: val usize,
    with_record_every => record_every: val usize,
    with_shots => shots: val usize,
    with_blocks => blocks: val usize,
    with_seed => seed: val u64,
    with_plateau_record => plateau_record: val usize,
});

impl Default for ShootingConfig {
    fn default() -> Self {
        Self {
            thermal: 0.02585,
            mass: 1.0,
            friction: 0.0,
            time_step: 1.0e-3,
            steps: 200,
            record_every: 10,
            shots: 64,
            blocks: 4,
            seed: 1,
            plateau_record: 0,
        }
    }
}

/// `κ(t)` and the plateau value.
///
/// `time` and `kappa_of_t` have `ceil(steps / record_every)` entries.
#[derive(Clone, Debug)]
pub struct Transmission {
    /// Times of the stored records, `(r + 1) * record_every * time_step`
    /// for record `r`, in the unit of [`ShootingConfig::time_step`].
    pub time: Vec<f64>,
    /// `κ(t)` at those times, dimensionless.
    pub kappa_of_t: Vec<f64>,
    /// `κ` at [`ShootingConfig::plateau_record`], dimensionless.
    pub kappa: f64,
    /// Block standard error of `kappa`, dimensionless.
    pub error: f64,
    /// Number of shots (out of [`ShootingConfig::shots`]) whose initial
    /// velocity along the tangent was positive and that entered the
    /// denominator.
    pub forward: usize,
}

/// Shoot from `starts` and return the Bennett–Chandler factor.
///
/// `tangent` is the product direction. It is normalised. `reference` is the
/// point on the dividing surface that defines `s = 0`; every start should
/// already lie on that plane.
///
/// `surface` is a [`PointSurface`] returning the energy in eV and the
/// gradient in eV/Angstrom. `starts` are configurations on the dividing
/// surface, each a flat coordinate vector in Angstrom; `reference` and
/// `tangent` have the same length. The mass is the same for every
/// coordinate.
///
/// # Errors
///
/// - [`SaddleError::Invalid`] when `starts` is empty, `thermal`, `mass`,
///   `time_step` or a count in `config` is out of range (see the field
///   docs), `friction` is negative or not finite, `tangent` has zero or
///   non-finite norm, `plateau_record` is past the trajectory, no shot
///   drew a positive initial velocity, or fewer than two blocks have a
///   positive flux.
/// - [`SaddleError::Shape`] when `reference`, `tangent` and every start do
///   not share a length, or the surface returns a gradient of another
///   length or with a non-finite entry.
/// - Any error the surface returns.
///
/// # Examples
///
/// ```
/// use ndarray::{Array1, ArrayView1, array};
/// use rgsaddle::minmode::PointSurface;
/// use rgsaddle::{SaddleError};
/// use rgflux::kappa::surface::{ShootingConfig, bennett_chandler};
///
/// /// V = (x^2 - 1)^2 + 2 y^2 + 2 z^2 in eV and Angstrom: saddle at the origin.
/// struct DoubleWell;
///
/// impl PointSurface for DoubleWell {
///     fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
///         let energy = (x[0] * x[0] - 1.0).powi(2) + 2.0 * x[1] * x[1] + 2.0 * x[2] * x[2];
///         let gradient = array![4.0 * x[0] * (x[0] * x[0] - 1.0), 4.0 * x[1], 4.0 * x[2]];
///         Ok((energy, gradient))
///     }
/// }
///
/// // Dividing surface x = 0 through the saddle, product side toward +x.
/// let starts = [array![0.0, 0.1, 0.0], array![0.0, -0.1, 0.05]];
/// let reference = array![0.0, 0.0, 0.0];
/// let tangent = array![1.0, 0.0, 0.0];
/// let config = ShootingConfig::default().with_shots(16).with_blocks(2);
/// let out = bennett_chandler(&DoubleWell, &starts, reference.view(), tangent.view(), &config)?;
/// assert!(out.kappa > 0.0 && out.kappa <= 1.0);
/// assert_eq!(out.kappa_of_t.len(), 20);
/// # Ok::<(), SaddleError>(())
/// ```
pub fn bennett_chandler<S: PointSurface>(
    surface: &S,
    starts: &[Array1<f64>],
    reference: ArrayView1<f64>,
    tangent: ArrayView1<f64>,
    config: &ShootingConfig,
) -> Result<Transmission, SaddleError> {
    if starts.is_empty() {
        return Err(SaddleError::Invalid(
            "Bennett-Chandler needs at least one dividing-surface configuration".into(),
        ));
    }
    if !(config.thermal.is_finite() && config.thermal > 0.0) {
        return Err(SaddleError::Invalid(
            "Bennett-Chandler kT must be finite and positive".into(),
        ));
    }
    if !(config.mass.is_finite() && config.mass > 0.0) {
        return Err(SaddleError::Invalid(
            "Bennett-Chandler mass must be finite and positive".into(),
        ));
    }
    if !(config.friction.is_finite() && config.friction >= 0.0) {
        return Err(SaddleError::Invalid(
            "Bennett-Chandler friction must be finite and non-negative".into(),
        ));
    }
    if !(config.time_step.is_finite() && config.time_step > 0.0) || config.steps == 0 {
        return Err(SaddleError::Invalid(
            "Bennett-Chandler needs a positive time step and at least one step".into(),
        ));
    }
    if config.record_every == 0 || config.shots == 0 || config.blocks < 2 {
        return Err(SaddleError::Invalid(
            "Bennett-Chandler needs a record interval, one shot, and at least two blocks".into(),
        ));
    }
    if config.shots < config.blocks {
        return Err(SaddleError::Invalid(
            "Bennett-Chandler needs at least one shot per block".into(),
        ));
    }
    let n = reference.len();
    if tangent.len() != n || starts.iter().any(|start| start.len() != n) {
        return Err(SaddleError::Shape(
            "Bennett-Chandler reference, tangent and starts must share a length".into(),
        ));
    }
    let tangent_norm = tangent.dot(&tangent).sqrt();
    if !(tangent_norm.is_finite() && tangent_norm > 0.0) {
        return Err(SaddleError::Invalid(
            "Bennett-Chandler tangent vanishes".into(),
        ));
    }
    let tangent = &tangent / tangent_norm;
    let n_record = config.steps.div_ceil(config.record_every);
    if config.plateau_record >= n_record {
        return Err(SaddleError::Invalid(
            "Bennett-Chandler plateau record is past the trajectory".into(),
        ));
    }
    let mut rng = Rng::new(config.seed);
    let sigma_v = (config.thermal / config.mass).sqrt();
    let mut weights = vec![0.0; config.shots];
    let mut side = vec![vec![0.0; n_record]; config.shots];
    let mut forward = 0usize;
    for shot in 0..config.shots {
        let start = &starts[shot % starts.len()];
        let mut velocity = Array1::zeros(n);
        for value in velocity.iter_mut() {
            *value = sigma_v * rng.normal();
        }
        let v_parallel = velocity.dot(&tangent);
        if v_parallel > 0.0 {
            forward += 1;
        }
        weights[shot] = v_parallel;
        let mut x = start.clone();
        let mut force = force_of(surface, x.view())?;
        let half = 0.5 * config.time_step / config.mass;
        for step in 0..config.steps {
            velocity = &velocity + &(&force * half);
            x = &x + &(&velocity * (0.5 * config.time_step));
            if config.friction > 0.0 {
                let decay = (-config.friction * config.time_step).exp();
                let noise = (config.thermal / config.mass * (1.0 - decay * decay)).sqrt();
                for value in velocity.iter_mut() {
                    *value = decay * *value + noise * rng.normal();
                }
            }
            x = &x + &(&velocity * (0.5 * config.time_step));
            force = force_of(surface, x.view())?;
            velocity = &velocity + &(&force * half);
            if (step + 1).is_multiple_of(config.record_every) {
                let record = (step + 1) / config.record_every - 1;
                let coordinate = (&x - &reference).dot(&tangent);
                side[shot][record] = if coordinate > 0.0 { 1.0 } else { 0.0 };
            }
        }
    }
    if forward == 0 {
        return Err(SaddleError::Invalid(
            "Bennett-Chandler drew no positive flux".into(),
        ));
    }
    let mut kappa_of_t = Vec::with_capacity(n_record);
    let mut time = Vec::with_capacity(n_record);
    for record in 0..n_record {
        let mut numer = 0.0;
        let mut denom = 0.0;
        for (weight, path) in weights.iter().zip(&side).take(config.shots) {
            numer += weight * path[record];
            if *weight > 0.0 {
                denom += weight;
            }
        }
        kappa_of_t.push(numer / denom);
        time.push((record + 1) as f64 * config.record_every as f64 * config.time_step);
    }
    let record = config.plateau_record;
    let width = config.shots / config.blocks;
    let mut block_kappa = Vec::with_capacity(config.blocks);
    for block in 0..config.blocks {
        let mut numer = 0.0;
        let mut denom = 0.0;
        for shot in block * width..(block + 1) * width {
            numer += weights[shot] * side[shot][record];
            if weights[shot] > 0.0 {
                denom += weights[shot];
            }
        }
        if denom > 0.0 {
            block_kappa.push(numer / denom);
        }
    }
    if block_kappa.len() < 2 {
        return Err(SaddleError::Invalid(
            "Bennett-Chandler blocks have no positive flux".into(),
        ));
    }
    let mean = block_kappa.iter().sum::<f64>() / block_kappa.len() as f64;
    let variance = block_kappa
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / (block_kappa.len() - 1) as f64;
    Ok(Transmission {
        kappa: kappa_of_t[record],
        error: (variance / block_kappa.len() as f64).sqrt(),
        time,
        kappa_of_t,
        forward,
    })
}

fn force_of<S: PointSurface>(surface: &S, x: ArrayView1<f64>) -> Result<Array1<f64>, SaddleError> {
    let (_energy, gradient) = surface.eval(x)?;
    if gradient.len() != x.len() || !gradient.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::Shape(
            "Bennett-Chandler gradient must match the coordinate".into(),
        ));
    }
    Ok(-gradient)
}
