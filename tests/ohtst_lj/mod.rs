//! Periodic fcc Lennard-Jones cells for the OH-TST reactant scaling tests.
//!
//! Reduced units: `sigma = epsilon = mass = 1`. The pair potential is shifted
//! so that energy and force both vanish at the cutoff, which keeps the force
//! the exact negative gradient of the energy. Pairs come from a Verlet list
//! with a skin, rebuilt once any atom has moved half the skin.

#![allow(dead_code)]

use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rgflux::FluxError;
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::engine::{
    DynamicsEngine, HyperplaneConstraint, StepReport, SurfaceVelocityPolicy, ThermostatConfig,
};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

pub const CUTOFF: f64 = 2.5;
pub const SKIN: f64 = 0.3;
pub const LATTICE: f64 = 1.55;

struct Pairs {
    reference: Vec<f64>,
    list: Vec<(usize, usize, [f64; 3])>,
}

pub struct LjCell {
    side: f64,
    pairs: Mutex<Option<Pairs>>,
    evaluations: AtomicUsize,
    best_fmax: Mutex<f64>,
}

fn pair_force(r: f64) -> f64 {
    let inv6 = r.powi(-6);
    24.0 * (2.0 * inv6 * inv6 - inv6) / r
}

fn pair_energy(r: f64) -> f64 {
    let inv6 = r.powi(-6);
    4.0 * (inv6 * inv6 - inv6)
}

impl LjCell {
    pub fn new(cells_per_side: usize) -> Self {
        Self {
            side: LATTICE * cells_per_side as f64,
            pairs: Mutex::new(None),
            evaluations: AtomicUsize::new(0),
            best_fmax: Mutex::new(f64::INFINITY),
        }
    }

    pub fn evaluations(&self) -> usize {
        self.evaluations.load(Ordering::Relaxed)
    }

    pub fn best_fmax(&self) -> f64 {
        *self.best_fmax.lock().expect("fmax lock")
    }

    fn build(&self, x: &[f64]) -> Pairs {
        let reach = CUTOFF + SKIN;
        let bins = (self.side / reach).floor() as usize;
        if bins >= 3 {
            return self.build_binned(x, bins);
        }
        self.build_all_pairs(x)
    }

    /// Minimum-image pairs from a grid of bins at least `CUTOFF + SKIN`
    /// wide, so every partner lies in one of the 27 surrounding bins.
    fn build_binned(&self, x: &[f64], bins: usize) -> Pairs {
        let atoms = x.len() / 3;
        let reach = CUTOFF + SKIN;
        let width = self.side / bins as f64;
        let bin_of = |v: f64| {
            let wrapped = v - self.side * (v / self.side).floor();
            ((wrapped / width) as usize).min(bins - 1)
        };
        let mut members = vec![Vec::new(); bins * bins * bins];
        let mut home = Vec::with_capacity(atoms);
        for i in 0..atoms {
            let b = [bin_of(x[3 * i]), bin_of(x[3 * i + 1]), bin_of(x[3 * i + 2])];
            members[(b[0] * bins + b[1]) * bins + b[2]].push(i);
            home.push(b);
        }
        let mut list = Vec::new();
        for i in 0..atoms {
            for dx in [bins - 1, 0, 1] {
                for dy in [bins - 1, 0, 1] {
                    for dz in [bins - 1, 0, 1] {
                        let b = [
                            (home[i][0] + dx) % bins,
                            (home[i][1] + dy) % bins,
                            (home[i][2] + dz) % bins,
                        ];
                        for &j in &members[(b[0] * bins + b[1]) * bins + b[2]] {
                            if j <= i {
                                continue;
                            }
                            let mut shift = [0.0; 3];
                            let mut r2 = 0.0;
                            for a in 0..3 {
                                let d = x[3 * j + a] - x[3 * i + a];
                                shift[a] = -self.side * (d / self.side).round();
                                r2 += (d + shift[a]).powi(2);
                            }
                            if r2 < reach * reach {
                                list.push((i, j, shift));
                            }
                        }
                    }
                }
            }
        }
        Pairs {
            reference: x.to_vec(),
            list,
        }
    }

    fn build_all_pairs(&self, x: &[f64]) -> Pairs {
        let atoms = x.len() / 3;
        let reach = CUTOFF + SKIN;
        let shifts: Vec<[f64; 3]> = if self.side >= 2.0 * reach {
            vec![[0.0; 3]]
        } else {
            assert!(self.side >= reach, "cell too small for one image shell");
            let k = [-self.side, 0.0, self.side];
            let mut shifts = Vec::with_capacity(27);
            for a in k {
                for b in k {
                    for c in k {
                        shifts.push([a, b, c]);
                    }
                }
            }
            shifts
        };
        let minimum_image = shifts.len() == 1;
        let mut list = Vec::new();
        for i in 0..atoms {
            for j in i + 1..atoms {
                let mut d = [0.0; 3];
                for a in 0..3 {
                    d[a] = x[3 * j + a] - x[3 * i + a];
                }
                for s in &shifts {
                    let mut shift = *s;
                    if minimum_image {
                        for a in 0..3 {
                            shift[a] = -self.side * (d[a] / self.side).round();
                        }
                    }
                    let r2: f64 = (0..3).map(|a| (d[a] + shift[a]).powi(2)).sum();
                    if r2 < reach * reach {
                        list.push((i, j, shift));
                    }
                }
            }
        }
        Pairs {
            reference: x.to_vec(),
            list,
        }
    }

    fn stale(pairs: &Pairs, x: &[f64]) -> bool {
        let limit = (0.5 * SKIN) * (0.5 * SKIN);
        x.chunks(3).zip(pairs.reference.chunks(3)).any(|(p, q)| {
            let moved: f64 = p.iter().zip(q).map(|(a, b)| (a - b).powi(2)).sum();
            moved > limit
        })
    }
}

impl rgsaddle::PointSurface for LjCell {
    fn eval(&self, x: ArrayView1<'_, f64>) -> Result<(f64, Array1<f64>), rgsaddle::SaddleError> {
        let x = x.as_slice().expect("contiguous positions");
        self.evaluations.fetch_add(1, Ordering::Relaxed);
        let mut guard = self.pairs.lock().expect("pair lock");
        if guard.as_ref().is_none_or(|p| Self::stale(p, x)) {
            *guard = Some(self.build(x));
        }
        let pairs = guard.as_ref().expect("built pairs");
        let (e_cut, f_cut) = (pair_energy(CUTOFF), pair_force(CUTOFF));
        let mut energy = 0.0;
        let mut gradient = vec![0.0; x.len()];
        for &(i, j, shift) in &pairs.list {
            let mut d = [0.0; 3];
            for a in 0..3 {
                d[a] = x[3 * j + a] - x[3 * i + a] + shift[a];
            }
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if r2 >= CUTOFF * CUTOFF {
                continue;
            }
            let r = r2.sqrt();
            energy += pair_energy(r) - e_cut + (r - CUTOFF) * f_cut;
            let push = (pair_force(r) - f_cut) / r;
            for a in 0..3 {
                gradient[3 * i + a] += push * d[a];
                gradient[3 * j + a] -= push * d[a];
            }
        }
        let fmax = gradient.iter().fold(0.0_f64, |m, g| m.max(g.abs()));
        let mut best = self.best_fmax.lock().expect("fmax lock");
        *best = best.min(fmax);
        Ok((energy, Array1::from_vec(gradient)))
    }
}

/// Perfect fcc sites of an `n x n x n` cell, every force zero up to rounding.
pub fn lattice(cells_per_side: usize) -> Array1<f64> {
    let basis = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ];
    let n = cells_per_side;
    let mut x = Vec::with_capacity(12 * n * n * n);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                for b in &basis {
                    x.push(LATTICE * (i as f64 + b[0]));
                    x.push(LATTICE * (j as f64 + b[1]));
                    x.push(LATTICE * (k as f64 + b[2]));
                }
            }
        }
    }
    Array1::from_vec(x)
}

/// Lattice sites displaced by independent uniform offsets in `[-amplitude, amplitude)`.
pub fn perturbed(cells_per_side: usize, amplitude: f64, seed: u64) -> Array1<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    lattice(cells_per_side).mapv(|v| v + rng.random_range(-amplitude..amplitude))
}

/// Delegates to a native LJ engine and fails the first Langevin step or
/// hyperplane constraint, which no OH-TST stage before sampling uses.
pub struct StopAtSampling {
    pub inner: NativeEngine<LjCell>,
    pub stopped: Option<(Instant, usize)>,
}

impl StopAtSampling {
    pub fn new(inner: NativeEngine<LjCell>) -> Self {
        Self {
            inner,
            stopped: None,
        }
    }

    fn stop(&mut self) -> FluxError {
        if self.stopped.is_none() {
            self.stopped = Some((Instant::now(), self.inner.surface().evaluations()));
        }
        FluxError::InvalidConfig("stopped at the first sampling call")
    }
}

impl DynamicsEngine for StopAtSampling {
    fn dof(&self) -> usize {
        self.inner.dof()
    }
    fn masses(&self) -> ArrayView1<'_, f64> {
        self.inner.masses()
    }
    fn positions(&self) -> ArrayView1<'_, f64> {
        self.inner.positions()
    }
    fn set_positions(&mut self, positions: ArrayView1<'_, f64>) -> Result<(), FluxError> {
        self.inner.set_positions(positions)
    }
    fn velocities(&self) -> ArrayView1<'_, f64> {
        self.inner.velocities()
    }
    fn set_velocities(&mut self, velocities: ArrayView1<'_, f64>) -> Result<(), FluxError> {
        self.inner.set_velocities(velocities)
    }
    fn cell(&self) -> Option<&[f64; 9]> {
        self.inner.cell()
    }
    fn set_cell(&mut self, cell: Option<[f64; 9]>) -> Result<(), FluxError> {
        self.inner.set_cell(cell)
    }
    fn thermostat(&self) -> &ThermostatConfig {
        self.inner.thermostat()
    }
    fn set_thermostat(&mut self, thermostat: ThermostatConfig) -> Result<(), FluxError> {
        self.inner.set_thermostat(thermostat)
    }
    fn constraint(&self) -> Option<&HyperplaneConstraint> {
        self.inner.constraint()
    }
    fn set_constraint(
        &mut self,
        constraint: Option<HyperplaneConstraint>,
    ) -> Result<(), FluxError> {
        if constraint.is_some() {
            return Err(self.stop());
        }
        self.inner.set_constraint(None)
    }
    fn energy_and_forces(&mut self) -> Result<(f64, Array1<f64>), FluxError> {
        self.inner.energy_and_forces()
    }
    fn step(&mut self, _dt: f64) -> Result<StepReport, FluxError> {
        Err(self.stop())
    }
    fn reseed(&mut self, seed: u64) -> Result<(), FluxError> {
        self.inner.reseed(seed)
    }
    fn shoot_velocities(&mut self, _seed: u64) -> Result<(), FluxError> {
        Err(self.stop())
    }
    fn reflect_velocity(&mut self, _normal: ArrayView1<'_, f64>) -> Result<f64, FluxError> {
        Err(self.stop())
    }
    fn apply_dividing_surface(
        &mut self,
        _normal: ArrayView1<'_, f64>,
        _policy: SurfaceVelocityPolicy,
    ) -> Result<f64, FluxError> {
        Err(self.stop())
    }
}

pub fn engine(cells_per_side: usize, start: &Array1<f64>) -> NativeEngine<LjCell> {
    let mut engine = NativeEngine::new(
        LjCell::new(cells_per_side),
        NativeConfig::default()
            .with_masses(Array1::ones(start.len()))
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(1.0)
                    .with_temperature(0.05),
            )
            .with_seed(11),
    )
    .expect("engine");
    engine.set_positions(start.view()).expect("start");
    engine
}
