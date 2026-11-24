//! Generalized Langevin colored-noise thermostat.
//!
//! The propagator is the one in Ceriotti, Bussi, and Parrinello, J. Chem.
//! Theory Comput. 6, 1170 (2010), as implemented by i-PI's `ThermoGLE`: the
//! physical momentum is stacked with auxiliary momenta, `T = exp(-Δt A)`
//! advances the stack, and the noise amplitude is the Hermitian square root
//! of `kT (C - T C Tᵀ)` with the canonical covariance `C = kT I`.
//!
//! [`ColoredNoise::SmartSampling`] is the 7×7 drift published for the most
//! efficient sampling over a 2 ps window. It is copied from the GLE4MD
//! parameter set reproduced in the atomistic cookbook
//! (<https://atomistic-cookbook.org/examples/thermostats/thermostats.html>)
//! and is not refitted here. The matrix is stored in atomic units. Engine
//! time is converted with [`atomic_units_per_engine_time`], from the CODATA
//! atomic unit of time and [`rgsaddle::strain_rate::angular_scale_rad_s`].
//!
//! White-noise Langevin remains [`ColoredNoise::White`]. A smart-sampling
//! step ignores `gamma`: the published matrix is the friction.

use rand::rngs::StdRng;

use crate::engine::box_muller_pair;
use crate::error::FluxError;

/// Dimension of the published smart-sampling drift, physical momentum plus auxiliaries.
pub const GLE_DIM: usize = 7;

/// CODATA 2022 atomic unit of time, `ℏ / E_h`, in seconds.
pub const ATOMIC_UNIT_OF_TIME_S: f64 = 2.4188843265857e-17;

/// How the velocity Ornstein–Uhlenbeck step is colored.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ColoredNoise {
    /// Scalar Langevin friction `gamma`.
    #[default]
    White,
    /// Published 2 ps smart-sampling GLE. See [`SMART_SAMPLING_A_AU`].
    SmartSampling,
}

/// Drift `A` in atomic units for [`ColoredNoise::SmartSampling`].
///
/// Row-major, copied from the cookbook's `mode="gle"` thermostat. The
/// generator's target was the most efficient sampling on a 2 ps window
/// (GLE4MD, <https://gle4md.org>).
pub const SMART_SAMPLING_A_AU: [[f64; GLE_DIM]; GLE_DIM] = [
    [
        8.191023526179e-4,
        8.328506066524e-3,
        1.657771834013e-3,
        9.736989925341e-4,
        2.841803794895e-4,
        -3.176846864198e-5,
        -2.967010478210e-4,
    ],
    [
        -8.389856546341e-4,
        2.405526974742e-2,
        -1.507872374848e-2,
        2.589784240185e-3,
        1.516783633362e-3,
        -5.958833418565e-4,
        4.198422349789e-4,
    ],
    [
        7.798710586406e-4,
        1.507872374848e-2,
        8.569039501219e-3,
        6.001000899602e-3,
        1.062029383877e-3,
        1.093939147968e-3,
        -2.661575532976e-3,
    ],
    [
        -9.676783161546e-4,
        -2.589784240185e-3,
        -6.001000899602e-3,
        2.680459336535e-5,
        -5.214694469742e-5,
        4.231304910751e-4,
        -2.104894919743e-5,
    ],
    [
        -2.841997149166e-4,
        -1.516783633362e-3,
        -1.062029383877e-3,
        5.214694469742e-5,
        1.433903506353e-9,
        -4.241574212449e-5,
        7.910178912362e-5,
    ],
    [
        3.333208286893e-5,
        5.958833418565e-4,
        -1.093939147968e-3,
        -4.231304910751e-4,
        4.241574212449e-5,
        2.385554468441e-8,
        -3.139255482869e-5,
    ],
    [
        2.967533789056e-4,
        -4.198422349789e-4,
        2.661575532976e-3,
        2.104894919743e-5,
        -7.910178912362e-5,
        3.139255482869e-5,
        2.432567259684e-11,
    ],
];

/// Atomic units of time in one engine time unit.
///
/// One engine time is `1 / angular_scale` seconds. Dividing by the atomic
/// unit of time is the factor that converts a drift from atomic units into
/// engine units: `A_engine = A_au * atomic_units_per_engine_time()`.
#[must_use]
pub fn atomic_units_per_engine_time() -> f64 {
    1.0 / (ATOMIC_UNIT_OF_TIME_S * rgsaddle::strain_rate::angular_scale_rad_s())
}

/// Auxiliary state of a GLE thermostat, one stack per degree of freedom.
#[derive(Clone, Debug)]
pub struct GleThermostat {
    active: bool,
    drift: [[f64; GLE_DIM]; GLE_DIM],
    cached_dt: u64,
    cached_kbt: u64,
    /// `0` when auxiliaries belong to Cartesian coordinates. Otherwise the
    /// FNV-1a key of the mass-independent Cartesian normal they were drawn for.
    basis_key: u64,
    transition: [[f64; GLE_DIM]; GLE_DIM],
    diffusion: [[f64; GLE_DIM]; GLE_DIM],
    auxiliaries: Vec<f64>,
    auxiliaries_ready: bool,
}

impl GleThermostat {
    /// A thermostat that does not color the noise.
    #[must_use]
    pub fn inactive() -> Self {
        Self {
            active: false,
            drift: [[0.0; GLE_DIM]; GLE_DIM],
            cached_dt: u64::MAX,
            cached_kbt: u64::MAX,
            basis_key: 0,
            transition: identity(),
            diffusion: [[0.0; GLE_DIM]; GLE_DIM],
            auxiliaries: Vec::new(),
            auxiliaries_ready: false,
        }
    }

    /// Install `noise` for `dof` coordinates, dropping any cached propagator.
    pub fn sync(&mut self, noise: ColoredNoise, dof: usize) {
        match noise {
            ColoredNoise::White => {
                self.active = false;
                self.auxiliaries.clear();
                self.auxiliaries_ready = false;
            }
            ColoredNoise::SmartSampling => {
                self.active = true;
                let factor = atomic_units_per_engine_time();
                self.drift = scale(&SMART_SAMPLING_A_AU, factor);
                self.cached_dt = u64::MAX;
                self.cached_kbt = u64::MAX;
                self.basis_key = 0;
                self.auxiliaries
                    .resize(dof.saturating_mul(GLE_DIM - 1), 0.0);
                self.auxiliaries_ready = false;
            }
        }
    }

    /// Forget the auxiliary momenta so the next step redraws them.
    pub fn invalidate(&mut self) {
        self.auxiliaries_ready = false;
    }

    /// Whether a colored-noise step is installed.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.active
    }

    /// One GLE velocity update. `velocities` are Cartesian; the stack is mass-weighted.
    ///
    /// Unconstrained coordinates only. A holonomic constraint has to use
    /// [`Self::apply_on_tangent`]: rattling the physical velocity after a full
    /// Cartesian step drops momentum the auxiliary stack still carries, and
    /// the next step puts that imbalance back into the free directions.
    pub fn apply_in_place(
        &mut self,
        velocities: &mut [f64],
        masses: &[f64],
        rng: &mut StdRng,
        kbt: f64,
        dt: f64,
    ) -> Result<(), FluxError> {
        if !self.active {
            return Ok(());
        }
        if self.basis_key != 0 {
            self.basis_key = 0;
            self.auxiliaries_ready = false;
        }
        if velocities.len() != masses.len() {
            return Err(FluxError::Shape {
                expected: masses.len(),
                got: velocities.len(),
                context: "GLE velocities",
            });
        }
        if !dt.is_finite() || dt <= 0.0 || !kbt.is_finite() || kbt < 0.0 {
            return Err(FluxError::InvalidConfig(
                "GLE step requires a positive timestep and non-negative kT",
            ));
        }
        self.ensure_propagator(dt, kbt)?;
        let extra = GLE_DIM - 1;
        let width = velocities.len() * extra;
        if self.auxiliaries.len() != width {
            self.auxiliaries.resize(width, 0.0);
            self.auxiliaries_ready = false;
        }
        if !self.auxiliaries_ready {
            self.draw_auxiliaries(rng, kbt);
        }
        let mut normal = vec![0.0; velocities.len() * GLE_DIM];
        fill_standard_normals(&mut normal, rng);
        for i in 0..velocities.len() {
            let root = masses[i].sqrt();
            if !(root.is_finite() && root > 0.0) {
                return Err(FluxError::InvalidConfig("GLE mass must be positive"));
            }
            let mut s = [0.0; GLE_DIM];
            s[0] = velocities[i] * root;
            for component in 0..extra {
                s[component + 1] = self.auxiliaries[i * extra + component];
            }
            let mut next = [0.0; GLE_DIM];
            for (next_row, (transition_row, diffusion_row)) in next
                .iter_mut()
                .zip(self.transition.iter().zip(&self.diffusion))
            {
                let mut acc = 0.0;
                for (col, (transition, diffusion)) in
                    transition_row.iter().zip(diffusion_row).enumerate()
                {
                    acc += transition * s[col];
                    acc += diffusion * normal[i * GLE_DIM + col];
                }
                *next_row = acc;
            }
            if !next.iter().all(|value| value.is_finite()) {
                return Err(FluxError::NonFinite("GLE momentum"));
            }
            velocities[i] = next[0] / root;
            for component in 0..extra {
                self.auxiliaries[i * extra + component] = next[component + 1];
            }
        }
        Ok(())
    }

    /// GLE on the mass-weighted tangent space of a unit Cartesian `normal`.
    ///
    /// The constrained momentum is identically zero and has no auxiliary
    /// stack. The free momenta are the Householder coordinates orthogonal to
    /// `M^{-1/2} n`, so each carries the canonical variance `kT` and the
    /// physical velocity satisfies `n · v = 0` when this returns.
    pub fn apply_on_tangent(
        &mut self,
        velocities: &mut [f64],
        masses: &[f64],
        normal: &[f64],
        rng: &mut StdRng,
        kbt: f64,
        dt: f64,
    ) -> Result<(), FluxError> {
        if !self.active {
            return Ok(());
        }
        if velocities.len() != masses.len() || normal.len() != masses.len() {
            return Err(FluxError::Shape {
                expected: masses.len(),
                got: velocities.len().max(normal.len()),
                context: "GLE tangent velocities",
            });
        }
        if !dt.is_finite() || dt <= 0.0 || !kbt.is_finite() || kbt < 0.0 {
            return Err(FluxError::InvalidConfig(
                "GLE step requires a positive timestep and non-negative kT",
            ));
        }
        let dof = velocities.len();
        if dof == 0 {
            return Err(FluxError::EmptyInput("GLE tangent velocities"));
        }
        let axis = mass_weighted_axis(normal, masses)?;
        let mirror = Householder::from_axis(&axis);
        if dof == 1 {
            velocities[0] = 0.0;
            self.auxiliaries.clear();
            self.auxiliaries_ready = false;
            self.basis_key = 0;
            return Ok(());
        }
        let key = normal_key(normal);
        let tangent = dof - 1;
        let extra = GLE_DIM - 1;
        let width = tangent * extra;
        if self.basis_key != key || self.auxiliaries.len() != width {
            self.basis_key = key;
            self.auxiliaries.resize(width, 0.0);
            self.auxiliaries_ready = false;
        }
        self.ensure_propagator(dt, kbt)?;
        if !self.auxiliaries_ready {
            self.draw_auxiliaries(rng, kbt);
        }
        let mut u = vec![0.0; dof];
        for i in 0..dof {
            let root = masses[i].sqrt();
            if !(root.is_finite() && root > 0.0) {
                return Err(FluxError::InvalidConfig("GLE mass must be positive"));
            }
            u[i] = velocities[i] * root;
        }
        let mut parallel = 0.0;
        for i in 0..dof {
            parallel += axis[i] * u[i];
        }
        for i in 0..dof {
            u[i] -= parallel * axis[i];
        }
        let mut mode = vec![0.0; dof];
        mirror.apply(&u, &mut mode);
        let mut noise = vec![0.0; tangent * GLE_DIM];
        fill_standard_normals(&mut noise, rng);
        for a in 0..tangent {
            let mut s = [0.0; GLE_DIM];
            s[0] = mode[a + 1];
            for component in 0..extra {
                s[component + 1] = self.auxiliaries[a * extra + component];
            }
            let mut next = [0.0; GLE_DIM];
            for (next_row, (transition_row, diffusion_row)) in next
                .iter_mut()
                .zip(self.transition.iter().zip(&self.diffusion))
            {
                let mut acc = 0.0;
                for (col, (transition, diffusion)) in
                    transition_row.iter().zip(diffusion_row).enumerate()
                {
                    acc += transition * s[col];
                    acc += diffusion * noise[a * GLE_DIM + col];
                }
                *next_row = acc;
            }
            if !next.iter().all(|value| value.is_finite()) {
                return Err(FluxError::NonFinite("GLE tangent momentum"));
            }
            mode[a + 1] = next[0];
            for component in 0..extra {
                self.auxiliaries[a * extra + component] = next[component + 1];
            }
        }
        mode[0] = 0.0;
        mirror.apply(&mode, &mut u);
        for i in 0..dof {
            velocities[i] = u[i] / masses[i].sqrt();
            if !velocities[i].is_finite() {
                return Err(FluxError::NonFinite("GLE tangent velocity"));
            }
        }
        Ok(())
    }

    fn draw_auxiliaries(&mut self, rng: &mut StdRng, kbt: f64) {
        let mut noise = vec![0.0; self.auxiliaries.len()];
        fill_standard_normals(&mut noise, rng);
        let width = kbt.sqrt();
        for (slot, draw) in self.auxiliaries.iter_mut().zip(noise) {
            *slot = width * draw;
        }
        self.auxiliaries_ready = true;
    }

    fn ensure_propagator(&mut self, dt: f64, kbt: f64) -> Result<(), FluxError> {
        if self.cached_dt == dt.to_bits() && self.cached_kbt == kbt.to_bits() {
            return Ok(());
        }
        let transition = expm(&scale(&self.drift, -dt));
        let gram = mat_mul(&transition, &transpose(&transition));
        let mut covariance = identity::<GLE_DIM>();
        for i in 0..GLE_DIM {
            for j in 0..GLE_DIM {
                covariance[i][j] -= gram[i][j];
            }
        }
        let diffusion = scale(&spd_factor(&covariance), kbt.sqrt());
        if transition.iter().flatten().any(|value| !value.is_finite())
            || diffusion.iter().flatten().any(|value| !value.is_finite())
        {
            return Err(FluxError::NonFinite("GLE propagator"));
        }
        self.transition = transition;
        self.diffusion = diffusion;
        self.cached_dt = dt.to_bits();
        self.cached_kbt = kbt.to_bits();
        Ok(())
    }
}

fn fill_standard_normals(out: &mut [f64], rng: &mut StdRng) {
    let mut index = 0;
    while index < out.len() {
        let (left, right) = box_muller_pair(rng);
        out[index] = left;
        if index + 1 < out.len() {
            out[index + 1] = right;
        }
        index += 2;
    }
}

fn identity<const N: usize>() -> [[f64; N]; N] {
    let mut out = [[0.0; N]; N];
    for (index, row) in out.iter_mut().enumerate() {
        row[index] = 1.0;
    }
    out
}

fn scale<const N: usize>(matrix: &[[f64; N]; N], factor: f64) -> [[f64; N]; N] {
    let mut out = [[0.0; N]; N];
    for (out_row, in_row) in out.iter_mut().zip(matrix) {
        for (out_value, in_value) in out_row.iter_mut().zip(in_row) {
            *out_value = in_value * factor;
        }
    }
    out
}

fn transpose<const N: usize>(matrix: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut out = [[0.0; N]; N];
    for (column, row) in matrix.iter().enumerate() {
        for (row_index, value) in row.iter().enumerate() {
            out[row_index][column] = *value;
        }
    }
    out
}

fn mat_mul<const N: usize>(left: &[[f64; N]; N], right: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut out = [[0.0; N]; N];
    for (out_row, left_row) in out.iter_mut().zip(left) {
        for (left_value, right_row) in left_row.iter().zip(right) {
            for (out_value, right_value) in out_row.iter_mut().zip(right_row) {
                *out_value += left_value * right_value;
            }
        }
    }
    out
}

fn add<const N: usize>(left: &[[f64; N]; N], right: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut out = [[0.0; N]; N];
    for (out_row, (left_row, right_row)) in out.iter_mut().zip(left.iter().zip(right)) {
        for (out_value, (left_value, right_value)) in
            out_row.iter_mut().zip(left_row.iter().zip(right_row))
        {
            *out_value = left_value + right_value;
        }
    }
    out
}

/// Scaling-and-squaring exponential. The series is taken at order 12 after
/// the 1-norm has been scaled below one half.
fn expm<const N: usize>(matrix: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut norm = 0.0_f64;
    for row in matrix {
        let row_sum = row.iter().map(|value| value.abs()).sum::<f64>();
        if row_sum > norm {
            norm = row_sum;
        }
    }
    let mut squares = 0_u32;
    let mut scaled = norm;
    while scaled > 0.5 && squares < 32 {
        scaled *= 0.5;
        squares += 1;
    }
    let factor = 0.5_f64.powi(squares as i32);
    let scaled_matrix = scale(matrix, factor);
    let mut term = identity::<N>();
    let mut sum = identity::<N>();
    for k in 1..=12 {
        term = scale(&mat_mul(&term, &scaled_matrix), 1.0 / f64::from(k));
        sum = add(&sum, &term);
    }
    for _ in 0..squares {
        sum = mat_mul(&sum, &sum);
    }
    sum
}

/// Symmetric factor `S` with `S Sᵀ = A` for a symmetric `A`.
///
/// Negative eigenvalues from roundoff are clipped to zero. Jacobi rotations
/// are enough at the GLE dimension of 7.
fn spd_factor<const N: usize>(matrix: &[[f64; N]; N]) -> [[f64; N]; N] {
    let (values, vectors) = jacobi_eigen(matrix);
    let mut scaled = [[0.0; N]; N];
    for (col, value) in values.iter().enumerate() {
        let root = value.max(0.0).sqrt();
        for (scaled_row, vector_row) in scaled.iter_mut().zip(&vectors) {
            scaled_row[col] = vector_row[col] * root;
        }
    }
    scaled
}

#[allow(clippy::needless_range_loop)]
fn jacobi_eigen<const N: usize>(matrix: &[[f64; N]; N]) -> ([f64; N], [[f64; N]; N]) {
    let mut dense = *matrix;
    for i in 0..N {
        for j in (i + 1)..N {
            let mean = 0.5 * (dense[i][j] + dense[j][i]);
            dense[i][j] = mean;
            dense[j][i] = mean;
        }
    }
    let mut vectors = identity::<N>();
    for _ in 0..(8 * N * N) {
        let mut pivot_row = 0;
        let mut pivot_col = 1.min(N - 1);
        let mut largest = 0.0_f64;
        for i in 0..N {
            for j in (i + 1)..N {
                let value = dense[i][j].abs();
                if value > largest {
                    largest = value;
                    pivot_row = i;
                    pivot_col = j;
                }
            }
        }
        if largest <= 1.0e-15 * (1.0 + dense_scale(&dense)) {
            break;
        }
        let app = dense[pivot_row][pivot_row];
        let aqq = dense[pivot_col][pivot_col];
        let apq = dense[pivot_row][pivot_col];
        let tau = (aqq - app) / (2.0 * apq);
        let tangent = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let cosine = 1.0 / (1.0 + tangent * tangent).sqrt();
        let sine = tangent * cosine;
        for i in 0..N {
            if i != pivot_row && i != pivot_col {
                let aip = dense[i][pivot_row];
                let aiq = dense[i][pivot_col];
                dense[i][pivot_row] = cosine * aip - sine * aiq;
                dense[pivot_row][i] = dense[i][pivot_row];
                dense[i][pivot_col] = sine * aip + cosine * aiq;
                dense[pivot_col][i] = dense[i][pivot_col];
            }
            let vip = vectors[i][pivot_row];
            let viq = vectors[i][pivot_col];
            vectors[i][pivot_row] = cosine * vip - sine * viq;
            vectors[i][pivot_col] = sine * vip + cosine * viq;
        }
        dense[pivot_row][pivot_row] =
            cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq;
        dense[pivot_col][pivot_col] =
            sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq;
        dense[pivot_row][pivot_col] = 0.0;
        dense[pivot_col][pivot_row] = 0.0;
    }
    let mut values = [0.0; N];
    for i in 0..N {
        values[i] = dense[i][i];
    }
    (values, vectors)
}

fn dense_scale<const N: usize>(matrix: &[[f64; N]; N]) -> f64 {
    matrix
        .iter()
        .flatten()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()))
}

fn normal_key(normal: &[f64]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for value in normal {
        hash = hash
            .wrapping_mul(0x100000001b3)
            .wrapping_add(value.to_bits());
    }
    if hash == 0 { 1 } else { hash }
}

fn mass_weighted_axis(normal: &[f64], masses: &[f64]) -> Result<Vec<f64>, FluxError> {
    let mut axis = vec![0.0; normal.len()];
    let mut norm_sq = 0.0;
    for i in 0..normal.len() {
        let root = masses[i].sqrt();
        if !(root.is_finite() && root > 0.0 && normal[i].is_finite()) {
            return Err(FluxError::InvalidConfig(
                "GLE tangent axis needs a finite normal and positive masses",
            ));
        }
        axis[i] = normal[i] / root;
        norm_sq += axis[i] * axis[i];
    }
    let norm = norm_sq.sqrt();
    if !(norm.is_finite() && norm > 0.0) {
        return Err(FluxError::InvalidConfig("GLE constraint normal vanished"));
    }
    for component in &mut axis {
        *component /= norm;
    }
    Ok(axis)
}

/// Householder reflector whose first column is `-sign(axis_0) * axis`.
struct Householder {
    vector: Vec<f64>,
    scale: f64,
}

impl Householder {
    fn from_axis(axis: &[f64]) -> Self {
        let mut vector = axis.to_vec();
        let sign = if axis[0] >= 0.0 { 1.0 } else { -1.0 };
        vector[0] += sign;
        let mut norm_sq = 0.0;
        for component in &vector {
            norm_sq += component * component;
        }
        Self {
            vector,
            scale: 2.0 / norm_sq,
        }
    }

    fn apply(&self, input: &[f64], output: &mut [f64]) {
        let projection: f64 = self
            .vector
            .iter()
            .zip(input)
            .map(|(component, value)| component * value)
            .sum();
        let coeff = self.scale * projection;
        for ((slot, value), component) in output.iter_mut().zip(input).zip(&self.vector) {
            *slot = value - coeff * component;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn the_published_matrix_is_the_cookbook_smart_set() {
        assert_eq!(SMART_SAMPLING_A_AU.len(), 7);
        assert_eq!(SMART_SAMPLING_A_AU[0].len(), 7);
        let sum: f64 = SMART_SAMPLING_A_AU.iter().flatten().copied().sum();
        assert!((sum - 4.340502209920356e-2).abs() < 1e-15, "{sum}");
        assert!((SMART_SAMPLING_A_AU[1][1] - 2.405526974742e-2).abs() < 1e-18);
        assert!(SMART_SAMPLING_A_AU[0][1] != SMART_SAMPLING_A_AU[1][0]);
    }

    #[test]
    fn one_engine_time_is_a_few_hundred_atomic_units() {
        let factor = atomic_units_per_engine_time();
        assert!(
            (400.0..450.0).contains(&factor),
            "atomic units per engine time {factor}"
        );
        let drift = scale(&SMART_SAMPLING_A_AU, factor);
        let ratio = drift[0][0] / SMART_SAMPLING_A_AU[0][0];
        assert!((ratio - factor).abs() < 1e-9 * factor);
    }

    #[test]
    fn the_exponential_and_the_factor_match_diagonal_matrices() {
        let exp_zero = expm(&[[0.0, 0.0], [0.0, 0.0]]);
        assert!((exp_zero[0][0] - 1.0).abs() < 1e-12);
        assert!(exp_zero[0][1].abs() < 1e-12);
        let exp_diag = expm(&[[0.2, 0.0], [0.0, -0.5]]);
        assert!((exp_diag[0][0] - 0.2_f64.exp()).abs() < 1e-12);
        assert!((exp_diag[1][1] - (-0.5_f64).exp()).abs() < 1e-12);
        assert!(exp_diag[0][1].abs() < 1e-12);
        let root = spd_factor(&[[4.0, 0.0], [0.0, 9.0]]);
        let rebuilt = mat_mul(&root, &transpose(&root));
        assert!((rebuilt[0][0] - 4.0).abs() < 1e-10);
        assert!((rebuilt[1][1] - 9.0).abs() < 1e-10);
        assert!(rebuilt[0][1].abs() < 1e-10);
        let coupled = spd_factor(&[[2.0, 1.0], [1.0, 2.0]]);
        let coupled_rebuilt = mat_mul(&coupled, &transpose(&coupled));
        assert!((coupled_rebuilt[0][0] - 2.0).abs() < 1e-8);
        assert!((coupled_rebuilt[0][1] - 1.0).abs() < 1e-8);
        assert!((coupled_rebuilt[1][1] - 2.0).abs() < 1e-8);
    }

    #[test]
    fn a_free_particle_keeps_the_canonical_momentum_variance() {
        let mut gle = GleThermostat::inactive();
        gle.sync(ColoredNoise::SmartSampling, 1);
        let mut velocity = [0.0];
        let mass = [1.7];
        let kbt = 0.8;
        let mut rng = StdRng::seed_from_u64(11);
        let mut moment = 0.0;
        let mut second = 0.0;
        let burn = 5_000;
        let samples = 40_000;
        for step in 0..(burn + samples) {
            gle.apply_in_place(&mut velocity, &mass, &mut rng, kbt, 0.05)
                .expect("gle");
            if step >= burn {
                let momentum = velocity[0] * mass[0].sqrt();
                moment += momentum;
                second += momentum * momentum;
            }
        }
        let mean = moment / f64::from(samples);
        let variance = second / f64::from(samples) - mean * mean;
        let relative = (variance - kbt).abs() / kbt;
        assert!(
            relative < 0.12,
            "mass-weighted momentum variance {variance} vs kT {kbt}"
        );
    }
}
