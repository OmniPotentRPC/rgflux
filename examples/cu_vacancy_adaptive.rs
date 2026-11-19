//! Cu vacancy hop physical check with Mishin Cu01 EAM via rgpot's in-process
//! `EAMTablePot` at the own-temperature NPT lattice at 1000 K and 1200 K.
//!
//! Compares even hyperplane spacing (`pafi_profile`) with adaptive placement
//! (`pafi_profile_adaptive`: monitor equidistribution + spectral-tail refinement)
//! using plane sampling windows sized from the measured `tau_int` of the
//! constraint force, and compares `k / Gamma` against Daw and Chandross,
//! Phys. Rev. Materials 5, 043603 (2021), Table II (`|z| <= 2` and `SE <= 3%`
//! under the replicate rule: pilot `14`, then `ceil(1.1 * n * (se / 0.03)^2)`
//! capped at `2048`).

use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Instant;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rgflux::FluxError;
use rgflux::engine::native::{NativeConfig, NativeEngine};
use rgflux::engine::rgpot::{RgpotConfig, RgpotSurface};
use rgflux::engine::{DynamicsEngine, ForceSurface, HyperplaneConstraint, ThermostatConfig};
use rgflux::pafi::{
    AdaptivePafiConfig, MonitorKind, PafiConfig, pafi_profile, pafi_profile_adaptive,
    size_window_from_tau_int,
};
use rgflux::stats::{StatsConfig, batch_means};
use rgsaddle::strain_rate::{BOLTZMANN_EV_PER_K, angular_scale_rad_s};
use rgsaddle::{
    BandConfig, BandStatus, BandSurface, Cell, IdppConfig, PointSurface, RelaxConfig, SaddleError,
    SolidNeb, finite_difference_cauchy_stress, idpp_path, relax_to_tolerance,
};

/// Equilibrium 0 K lattice constant of Mishin Cu01 EAM (Angstrom).
const CU_LATTICE_A0: f64 = 3.615;
/// Standard atomic weight of copper (amu).
const CU_MASS_AMU: f64 = 63.546;
/// Atomic number of copper.
const CU_Z: i32 = 29;
/// FCC nearest-neighbour coordination number.
const FCC_COORDINATION: f64 = 12.0;

/// Daw and Chandross, Phys. Rev. Materials 5, 043603 (2021), Table II:
/// `M0 = 190 A^2/ps`, `sigma(ln M0) = 0.13`, `Qm = 0.67 eV`, `sigma(Qm) = 0.013 eV`.
const PUBLISHED_M0_A2_PER_PS: f64 = 190.0;
const PUBLISHED_LN_M0_SIGMA: f64 = 0.13;
const PUBLISHED_QM_EV: f64 = 0.67;
const PUBLISHED_QM_SIGMA_EV: f64 = 0.013;

const PILOT_REPLICATES: usize = 14;
const MAX_REPLICATES: usize = 2048;
const TARGET_REL_SE: f64 = 0.03;
const EVEN_PLANES: usize = 25;
const ADAPTIVE_INIT_PLANES: usize = 9;
const ADAPTIVE_MAX_PLANES: usize = 15;
const BASE_SEED: u64 = 17_029;

/// Replicate growth rule: pilot `14`, then `ceil(1.1 * n * (se / 0.03)^2)`, capped at `2048`.
fn next_replicate_count(n: usize, relative_se: f64) -> usize {
    if relative_se <= TARGET_REL_SE {
        return n;
    }
    let ratio = relative_se / TARGET_REL_SE;
    let grown = (1.1 * (n as f64) * ratio * ratio).ceil() as usize;
    grown.clamp(n, MAX_REPLICATES)
}

/// Mishin, Mehl, Papaconstantopoulos, Voter, and Kress, Phys. Rev. B 63, 224106 (2001)
/// Table I parameters for Cu EAM1, tabulated onto the 10000-knot DYNAMO grid and
/// evaluated in-process by `rgpot::EAMTablePot`.
fn build_mishin_cu01_surface(n_atoms: usize, cell: [f64; 9]) -> Result<RgpotSurface, FluxError> {
    const RC: f64 = 5.506_786;
    const H_CUT: f64 = 0.500_37;
    const E1: f64 = 2.014_58e2;
    const E2: f64 = 6.592_88e-3;
    const R01: f64 = 0.835_91;
    const R02: f64 = 4.468_67;
    const ALPHA1: f64 = 2.977_58;
    const ALPHA2: f64 = 1.549_27;
    const DELTA: f64 = 0.862_25e-2;
    const RS: [f64; 3] = [2.24, 1.80, 1.20];
    const S_REP: [f64; 3] = [4.0, 40.0, 1.15e3];
    const DENSITY_A: f64 = 3.803_62;
    const R03: f64 = -2.198_85;
    const R04: f64 = -2.619_84e2;
    const BETA1: f64 = 0.173_94;
    const BETA2: f64 = 5.356_61e2;
    const F0: f64 = -2.282_35;
    const F2: f64 = 1.355_35;
    const Q1: f64 = -1.277_75;
    const Q2: f64 = -0.860_74;
    const Q3: f64 = 1.788_04;
    const Q4: f64 = 2.975_71;
    const EMB_Q1: f64 = 0.40;
    const EMB_Q2: f64 = 0.30;

    let psi = |r: f64| -> f64 {
        let x = (r - RC) / H_CUT;
        if x >= 0.0 {
            0.0
        } else {
            let x2 = x * x;
            let x4 = x2 * x2;
            x4 / (1.0 + x4)
        }
    };
    let morse = |r: f64, r0: f64, alpha: f64| -> f64 {
        let z = -alpha * (r - r0);
        (2.0 * z).exp() - 2.0 * z.exp()
    };
    let pair_v = |r: f64| -> f64 {
        if r >= RC {
            return 0.0;
        }
        let mut v = (E1 * morse(r, R01, ALPHA1) + E2 * morse(r, R02, ALPHA2) + DELTA) * psi(r);
        for (&rs, &s) in RS.iter().zip(&S_REP) {
            if r < rs {
                let g = rs - r;
                v -= s * g * g * g * g;
            }
        }
        v
    };
    let rho_raw = |r: f64| -> f64 {
        if r >= RC {
            return 0.0;
        }
        let d3 = r - R03;
        let g1 = (-BETA1 * d3 * d3).exp();
        let arg = -BETA2 * (r - R04);
        let g2 = if arg < -700.0 { 0.0 } else { arg.exp() };
        (DENSITY_A * g1 + g2) * psi(r)
    };
    let embed_f = |rho: f64| -> f64 {
        let x = rho - 1.0;
        let x2 = x * x;
        let x3 = x2 * x;
        if rho < 1.0 {
            F0 + 0.5 * F2 * x2 + Q1 * x3 + Q2 * x2 * x2 + Q3 * x2 * x3 + Q4 * x3 * x3
        } else {
            (F0 + 0.5 * F2 * x2 + Q1 * x3 + EMB_Q1 * x2 * x2) / (1.0 + EMB_Q2 * x3)
        }
    };

    let nrho = 10_000usize;
    let drho = 0.0012_f64;
    let nr = 10_000usize;
    let dr = RC / (nr as f64);
    let rho_eq = 12.0 * rho_raw(CU_LATTICE_A0 / 2.0_f64.sqrt())
        + 6.0 * rho_raw(CU_LATTICE_A0)
        + 24.0 * rho_raw(CU_LATTICE_A0 * 1.5_f64.sqrt())
        + 12.0 * rho_raw(CU_LATTICE_A0 * 2.0_f64.sqrt());

    let frho: Vec<f64> = (0..nrho).map(|i| embed_f((i as f64) * drho)).collect();
    let rhor: Vec<f64> = (0..nr).map(|i| rho_raw((i as f64) * dr) / rho_eq).collect();
    let z2r: Vec<f64> = (0..nr)
        .map(|i| {
            let r = (i as f64) * dr;
            if i == 0 { 0.0 } else { r * pair_v(r) }
        })
        .collect();

    let cfg = RgpotConfig::default()
        .with_atomic_numbers(vec![CU_Z; n_atoms])
        .with_default_cell(cell);
    RgpotSurface::from_eam_table_memory(CU_Z, drho, dr, RC, &frho, &rhor, &z2r, true, 1.0, cfg)
}

struct CountingSurface {
    inner: RgpotSurface,
    cell: [f64; 9],
    calls: AtomicUsize,
}

impl CountingSurface {
    fn new(n_atoms: usize, cell: [f64; 9]) -> Result<Self, FluxError> {
        Ok(Self {
            inner: build_mishin_cu01_surface(n_atoms, cell)?,
            cell,
            calls: AtomicUsize::new(0),
        })
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }
}

impl PointSurface for CountingSurface {
    fn eval(&self, x: ArrayView1<'_, f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let (energy, forces) = self
            .inner
            .energy_and_forces(x, Some(&self.cell))
            .map_err(|e| SaddleError::Surface(e.to_string()))?;
        Ok((energy, forces.mapv(|f| -f)))
    }

    fn eval_in_cell(
        &self,
        x: ArrayView1<'_, f64>,
        cell: &[f64; 9],
    ) -> Result<(f64, Array1<f64>), SaddleError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let (energy, forces) = self
            .inner
            .energy_and_forces(x, Some(cell))
            .map_err(|e| SaddleError::Surface(e.to_string()))?;
        Ok((energy, forces.mapv(|f| -f)))
    }
}

impl BandSurface for CountingSurface {
    fn eval(
        &self,
        positions: ArrayView2<'_, f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.rows().into_iter().enumerate() {
            let (e, g) = PointSurface::eval(self, row)?;
            energies[i] = e;
            gradients.row_mut(i).assign(&g);
        }
        Ok(())
    }
}

fn fcc_crystal(lattice: f64, nx: usize, ny: usize, nz: usize) -> (Array1<f64>, [f64; 9]) {
    let basis = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ];
    let mut pos = Vec::with_capacity(nx * ny * nz * 12);
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                for b in basis {
                    pos.push(lattice * ((ix as f64) + b[0]));
                    pos.push(lattice * ((iy as f64) + b[1]));
                    pos.push(lattice * ((iz as f64) + b[2]));
                }
            }
        }
    }
    let cell = [
        lattice * (nx as f64),
        0.0,
        0.0,
        0.0,
        lattice * (ny as f64),
        0.0,
        0.0,
        0.0,
        lattice * (nz as f64),
    ];
    (Array1::from_vec(pos), cell)
}

fn vacancy_endpoints(
    lattice: f64,
    nx: usize,
    ny: usize,
    nz: usize,
) -> (Array1<f64>, Array1<f64>, [f64; 9]) {
    let basis = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ];
    let mut pos = Vec::with_capacity((nx * ny * nz * 4 - 1) * 3);
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                for b in basis {
                    if ix == 0 && iy == 0 && iz == 0 && b == [0.0, 0.0, 0.0] {
                        continue;
                    }
                    pos.push(lattice * ((ix as f64) + b[0]));
                    pos.push(lattice * ((iy as f64) + b[1]));
                    pos.push(lattice * ((iz as f64) + b[2]));
                }
            }
        }
    }
    let cell = [
        lattice * (nx as f64),
        0.0,
        0.0,
        0.0,
        lattice * (ny as f64),
        0.0,
        0.0,
        0.0,
        lattice * (nz as f64),
    ];
    let reactant = Array1::from_vec(pos);
    let mut product = reactant.clone();
    product[0] = 0.0;
    product[1] = 0.0;
    product[2] = 0.0;
    (reactant, product, cell)
}

fn m3_of(cell: [f64; 9]) -> [[f64; 3]; 3] {
    [
        [cell[0], cell[1], cell[2]],
        [cell[3], cell[4], cell[5]],
        [cell[6], cell[7], cell[8]],
    ]
}

fn hydrostatic_tension(
    surface: &CountingSurface,
    positions: ArrayView1<'_, f64>,
    cell: [f64; 9],
) -> Result<f64, FluxError> {
    let s = finite_difference_cauchy_stress(surface, positions, m3_of(cell), 1.0e-4)?;
    Ok((s[0][0] + s[1][1] + s[2][2]) / 3.0)
}

fn npt_lattice_at(temperature_k: f64, seed: u64) -> Result<(f64, f64), FluxError> {
    let repeats = 4usize;
    let (positions, cell0) = fcc_crystal(CU_LATTICE_A0, repeats, repeats, repeats);
    let n_atoms = positions.len() / 3;
    let surface = CountingSurface::new(n_atoms, cell0)?;
    let base_tension = hydrostatic_tension(&surface, positions.view(), cell0)?;
    let strain = 1.0e-3;
    let factor = 1.0 + strain;
    let moved = positions.mapv(|v| v * factor);
    let mut cell_s = cell0;
    cell_s[0] *= factor;
    cell_s[4] *= factor;
    cell_s[8] *= factor;
    let stretched_tension = hydrostatic_tension(&surface, moved.view(), cell_s)?;
    let bulk_modulus = (stretched_tension - base_tension) / (3.0 * strain);

    let mut engine = NativeEngine::new(
        surface,
        NativeConfig::default()
            .with_masses(Array1::from_elem(positions.len(), CU_MASS_AMU))
            .with_cell(cell0)
            .with_seed(seed)
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(1.0)
                    .with_temperature(temperature_k)
                    .with_boltzmann(BOLTZMANN_EV_PER_K),
            ),
    )?;
    engine.set_positions(positions.view())?;
    engine.shoot_velocities(seed)?;

    let dt = 2.0e-15 * angular_scale_rad_s();
    let tau = 0.5e-12 * angular_scale_rad_s();
    let steps = 2_000usize;
    let stride = 20usize;
    let mut edges = Vec::new();
    for step in 0..steps {
        engine.step(dt)?;
        if (step + 1) % stride != 0 {
            continue;
        }
        let cell_cur = *engine.cell().expect("cell");
        let hydro = hydrostatic_tension(engine.surface(), engine.positions(), cell_cur)?;
        let vol = cell_cur[0] * cell_cur[4] * cell_cur[8];
        let kinetic = -(2.0 * engine.kinetic_energy()) / (3.0 * vol);
        let mu = (1.0 - ((stride as f64) * dt / tau) * (hydro + kinetic) / bulk_modulus)
            .clamp(0.5, 1.5)
            .cbrt();
        let new_pos = engine.positions().mapv(|v| v * mu);
        engine.set_positions(new_pos.view())?;
        let mut next_cell = cell_cur;
        next_cell[0] *= mu;
        next_cell[4] *= mu;
        next_cell[8] *= mu;
        engine.set_cell(Some(next_cell))?;
        if step >= steps / 2 {
            edges.push(next_cell[0] / (repeats as f64));
        }
    }
    let est = batch_means(&edges, &StatsConfig::default().with_min_batches(4))?;
    Ok((est.mean, est.standard_error))
}

fn unwrap_path(images: &Array2<f64>, cell: [f64; 9]) -> Array2<f64> {
    let mut out = images.clone();
    let lengths = [cell[0], cell[4], cell[8]];
    for row in 1..out.nrows() {
        for atom in 0..(out.ncols() / 3) {
            for (axis, &len) in lengths.iter().enumerate() {
                let idx = 3 * atom + axis;
                let delta = out[[row, idx]] - out[[row - 1, idx]];
                out[[row, idx]] -= len * (delta / len).round();
            }
        }
    }
    out
}

fn prepare_vacancy_path(lattice: f64) -> Result<(Array2<f64>, [f64; 9], f64), FluxError> {
    let (reactant, product, cell) = vacancy_endpoints(lattice, 4, 4, 4);
    let n_atoms = reactant.len() / 3;
    let surface = CountingSurface::new(n_atoms, cell)?;
    let mut relax_cfg = RelaxConfig::default();
    relax_cfg.force_tol = 1.0e-3;
    relax_cfg.max_steps = 2_000;
    relax_cfg.max_atom_move = 0.1;
    let r_rel = relax_to_tolerance(&surface, reactant, m3_of(cell), &relax_cfg)?;
    let p_rel = relax_to_tolerance(&surface, product, m3_of(cell), &relax_cfg)?;

    let mic_cell = rgsaddle::mic::Cell::from_vesin(m3_of(cell))
        .map_err(|e| FluxError::Surface(e.to_string()))?;
    let mut idpp_cfg = IdppConfig::default();
    idpp_cfg.max_iterations = 200;
    let initial = idpp_path(
        r_rel.positions.view(),
        p_rel.positions.view(),
        7,
        Some(&mic_cell),
        &idpp_cfg,
    )?;

    let mut band_cfg = BandConfig::default();
    band_cfg.cell = Some(Cell(m3_of(cell)));
    band_cfg.force_tol = 5.0e-3;
    band_cfg.max_move = 0.15;
    band_cfg.min_mode = None;
    band_cfg.saddle_check = None;
    let mut neb = SolidNeb::frozen(band_cfg, initial)?;
    for _ in 0..800 {
        let rep = neb.step_surface(&surface)?;
        if rep.status == BandStatus::Converged {
            break;
        }
    }
    let path = unwrap_path(&neb.cartesian()?, cell);
    let mut saddle_e = f64::NEG_INFINITY;
    for row in path.rows() {
        let (e, _) = PointSurface::eval(&surface, row)?;
        saddle_e = saddle_e.max(e);
    }
    let barrier = saddle_e - r_rel.energy;
    Ok((path, cell, barrier))
}

fn published_gamma_hz(temperature_k: f64) -> f64 {
    let kbt = BOLTZMANN_EV_PER_K * temperature_k;
    let m = PUBLISHED_M0_A2_PER_PS * (-PUBLISHED_QM_EV / kbt).exp();
    FCC_COORDINATION * m / (CU_LATTICE_A0 * CU_LATTICE_A0) * 1.0e12
}

fn published_relative_sigma(temperature_k: f64) -> f64 {
    let kbt = BOLTZMANN_EV_PER_K * temperature_k;
    (PUBLISHED_LN_M0_SIGMA * PUBLISHED_LN_M0_SIGMA + (PUBLISHED_QM_SIGMA_EV / kbt).powi(2)).sqrt()
}

fn comparison_z(rate_hz: f64, se_hz: f64, temperature_k: f64) -> f64 {
    let gamma = published_gamma_hz(temperature_k);
    let gamma_se = gamma * published_relative_sigma(temperature_k);
    (rate_hz - gamma) / se_hz.hypot(gamma_se)
}

#[derive(Clone, Copy, Debug)]
struct ReplicateSample {
    replicate: usize,
    seed: u64,
    planes: usize,
    f_star_ev: f64,
    f_star_se_ev: f64,
    tail_bound_ev: f64,
    rate_hz: f64,
    force_calls: usize,
    wall_s: f64,
}

fn mean_and_se(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().copied().sum::<f64>() / n;
    let var = values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, (var / n).sqrt())
}

#[allow(clippy::too_many_arguments)]
fn run_campaign(
    temperature_k: f64,
    t_index: usize,
    adaptive: bool,
    path: &Array2<f64>,
    cell: [f64; 9],
    burn_in: usize,
    sample_steps: usize,
    scale_hz: f64,
    stats_cfg: &StatsConfig,
) -> Result<(Vec<ReplicateSample>, f64), FluxError> {
    let campaign_start = Instant::now();
    let mut target_n = PILOT_REPLICATES;
    let mut samples: Vec<ReplicateSample> = Vec::new();
    let workers = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(8);

    loop {
        let mut cursor = samples.len();
        while cursor < target_n {
            let batch_end = (cursor + workers).min(target_n);
            let batch_res: Result<Vec<ReplicateSample>, FluxError> = thread::scope(|scope| {
                let mut handles = Vec::with_capacity(batch_end - cursor);
                for rep in cursor..batch_end {
                    let tag = if adaptive { 2_u64 } else { 1_u64 };
                    let seed = BASE_SEED
                        + (t_index as u64) * 1_000_003
                        + tag * 100_003
                        + (rep as u64) * 7_919;
                    handles.push(scope.spawn(move || -> Result<ReplicateSample, FluxError> {
                        let n_atoms = path.ncols() / 3;
                        let surface = CountingSurface::new(n_atoms, cell)?;
                        let mut engine = NativeEngine::new(
                            surface,
                            NativeConfig::default()
                                .with_masses(Array1::from_elem(path.ncols(), CU_MASS_AMU))
                                .with_cell(cell)
                                .with_seed(seed)
                                .with_thermostat(
                                    ThermostatConfig::default()
                                        .with_gamma(1.0)
                                        .with_temperature(temperature_k)
                                        .with_boltzmann(BOLTZMANN_EV_PER_K),
                                ),
                        )?;
                        let dt = 2.0e-15 * angular_scale_rad_s();
                        let t0 = Instant::now();
                        if adaptive {
                            let cfg = AdaptivePafiConfig::default()
                                .with_base(
                                    PafiConfig::default()
                                        .with_planes(ADAPTIVE_INIT_PLANES)
                                        .with_burn_in_steps(burn_in)
                                        .with_sample_steps(sample_steps)
                                        .with_time_step(dt)
                                        .with_zero_temperature_tol(0.05)
                                        .with_seed(seed)
                                        .with_stats(stats_cfg.clone()),
                                )
                                .with_max_planes(ADAPTIVE_MAX_PLANES)
                                .with_max_refinements(3)
                                .with_monitor(MonitorKind::Combined);
                            let res = pafi_profile_adaptive(&mut engine, path.view(), &cfg)?;
                            let wall_s = t0.elapsed().as_secs_f64();
                            let calls = engine.surface().calls();
                            Ok(ReplicateSample {
                                replicate: rep,
                                seed,
                                planes: res.profile.arc_lengths.len(),
                                f_star_ev: res.profile.barrier,
                                f_star_se_ev: res.profile.barrier_error,
                                tail_bound_ev: res.tail.tail_bound,
                                rate_hz: res.profile.normalized_rate * scale_hz,
                                force_calls: calls,
                                wall_s,
                            })
                        } else {
                            let cfg = PafiConfig::default()
                                .with_planes(EVEN_PLANES)
                                .with_burn_in_steps(burn_in)
                                .with_sample_steps(sample_steps)
                                .with_time_step(dt)
                                .with_zero_temperature_tol(0.05)
                                .with_seed(seed)
                                .with_stats(stats_cfg.clone());
                            let prof = pafi_profile(&mut engine, path.view(), &cfg)?;
                            let wall_s = t0.elapsed().as_secs_f64();
                            let calls = engine.surface().calls();
                            Ok(ReplicateSample {
                                replicate: rep,
                                seed,
                                planes: prof.arc_lengths.len(),
                                f_star_ev: prof.barrier,
                                f_star_se_ev: prof.barrier_error,
                                tail_bound_ev: 0.0,
                                rate_hz: prof.normalized_rate * scale_hz,
                                force_calls: calls,
                                wall_s,
                            })
                        }
                    }));
                }
                handles
                    .into_iter()
                    .map(|h| h.join().expect("replicate thread panicked"))
                    .collect()
            });
            samples.extend(batch_res?);
            cursor = batch_end;
        }

        let f_vals: Vec<f64> = samples.iter().map(|s| s.f_star_ev).collect();
        let k_vals: Vec<f64> = samples.iter().map(|s| s.rate_hz).collect();
        let (f_mean, f_se) = mean_and_se(&f_vals);
        let (k_mean, k_se) = mean_and_se(&k_vals);
        let f_rel_se = f_se / f_mean;
        let k_rel_se = k_se / k_mean;
        eprintln!(
            "  [{temperature_k} K adaptive={adaptive}] n={} F*={f_mean:.6}±{f_se:.6} (f_rel={f_rel_se:.4}) k={k_mean:.4e}±{k_se:.4e} (k_rel={k_rel_se:.4})",
            samples.len()
        );
        let next_n = next_replicate_count(samples.len(), f_rel_se);
        if next_n == samples.len() || samples.len() >= MAX_REPLICATES {
            break;
        }
        target_n = next_n;
    }

    Ok((samples, campaign_start.elapsed().as_secs_f64()))
}

fn main() -> Result<(), FluxError> {
    let out_dir = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/cu-vacancy-adaptive"));
    fs::create_dir_all(&out_dir).expect("create output dir");

    let mut lattice_csv =
        File::create(out_dir.join("cu_vacancy_npt_lattice.csv")).expect("lattice csv");
    writeln!(
        lattice_csv,
        "temperature_K,atoms,lattice_A,lattice_se_A,strain,static_barrier_eV,published_gamma_Hz,published_rel_sigma"
    )
    .expect("lattice header");

    let mut summary_csv =
        File::create(out_dir.join("cu_vacancy_even_vs_adaptive.csv")).expect("summary csv");
    writeln!(
        summary_csv,
        "temperature_K,method,planes_mean,tau_int_steps,burn_in_steps,sample_steps,replicates_n,f_star_eV,f_star_se_eV,f_star_rel_se,tail_bound_eV,rate_Hz,rate_se_Hz,rate_rel_se,published_gamma_Hz,k_over_gamma,z_score,pass,force_calls_per_rep,total_force_calls,wall_time_s"
    )
    .expect("summary header");

    let mut rep_csv =
        File::create(out_dir.join("cu_vacancy_replicates.csv")).expect("replicates csv");
    writeln!(
        rep_csv,
        "temperature_K,method,replicate,seed,planes,f_star_eV,f_star_se_eV,tail_bound_eV,rate_Hz,force_calls,wall_s"
    )
    .expect("replicates header");

    let stats_cfg = StatsConfig::default();

    for (idx, &temp) in [1000.0_f64, 1200.0].iter().enumerate() {
        let (a_t, a_se) = npt_lattice_at(temp, BASE_SEED + idx as u64)?;
        let strain = a_t / CU_LATTICE_A0 - 1.0;
        let (path, cell, static_barrier) = prepare_vacancy_path(a_t)?;
        let gamma_pub = published_gamma_hz(temp);
        let gamma_rel_sigma = published_relative_sigma(temp);

        writeln!(
            lattice_csv,
            "{temp:.1},{},{a_t:.8},{a_se:.8e},{strain:.8e},{static_barrier:.8},{gamma_pub:.8e},{gamma_rel_sigma:.8}",
            path.ncols() / 3
        )
        .expect("lattice row");
        lattice_csv.flush().expect("flush lattice");

        // Measure tau_int on the saddle hyperplane to size the plane sampling window.
        let n_atoms = path.ncols() / 3;
        let surface = CountingSurface::new(n_atoms, cell)?;
        let mut pilot_engine = NativeEngine::new(
            surface,
            NativeConfig::default()
                .with_masses(Array1::from_elem(path.ncols(), CU_MASS_AMU))
                .with_cell(cell)
                .with_seed(BASE_SEED + 50 + idx as u64)
                .with_thermostat(
                    ThermostatConfig::default()
                        .with_gamma(1.0)
                        .with_temperature(temp)
                        .with_boltzmann(BOLTZMANN_EV_PER_K),
                ),
        )?;
        let saddle_row = path.row(3).to_owned();
        let dir = &path.row(4) - &path.row(2);
        let norm = dir.dot(&dir).sqrt();
        let normal = dir.mapv(|v| v / norm);
        let offset = normal.dot(&saddle_row);
        let dt = 2.0e-15 * angular_scale_rad_s();
        pilot_engine.set_constraint(Some(HyperplaneConstraint::new(normal, offset)?))?;
        pilot_engine.set_positions(saddle_row.view())?;
        pilot_engine.shoot_velocities(BASE_SEED + 50 + idx as u64)?;
        for _ in 0..32 {
            pilot_engine.step(dt)?;
        }
        let mut pilot_forces = Vec::with_capacity(128);
        for _ in 0..128 {
            let rep = pilot_engine.step(dt)?;
            pilot_forces.push(rep.constraint_force.unwrap_or(0.0));
        }
        let pilot_est = batch_means(&pilot_forces, &stats_cfg)?;
        let (burn_in, sample_steps) =
            size_window_from_tau_int(pilot_est.tau_int, 24, 96, &stats_cfg)?;
        let scale_hz = angular_scale_rad_s() * FCC_COORDINATION;

        eprintln!(
            "T={temp} K: a(T)={a_t:.6} A, Em={static_barrier:.6} eV, tau_int={:.2} -> burn={burn_in}, sample={sample_steps}",
            pilot_est.tau_int
        );

        for (method, adaptive) in [("even", false), ("adaptive", true)] {
            let (samples, wall_time_s) = run_campaign(
                temp,
                idx,
                adaptive,
                &path,
                cell,
                burn_in,
                sample_steps,
                scale_hz,
                &stats_cfg,
            )?;
            for s in &samples {
                writeln!(
                    rep_csv,
                    "{temp:.1},{method},{},{},{},{:.8},{:.8e},{:.8e},{:.8e},{},{:.6}",
                    s.replicate,
                    s.seed,
                    s.planes,
                    s.f_star_ev,
                    s.f_star_se_ev,
                    s.tail_bound_ev,
                    s.rate_hz,
                    s.force_calls,
                    s.wall_s
                )
                .expect("rep row");
            }
            rep_csv.flush().expect("flush rep");

            let n_rep = samples.len();
            let planes_mean = samples.iter().map(|s| s.planes as f64).sum::<f64>() / (n_rep as f64);
            let f_vals: Vec<f64> = samples.iter().map(|s| s.f_star_ev).collect();
            let k_vals: Vec<f64> = samples.iter().map(|s| s.rate_hz).collect();
            let (f_mean, f_se) = mean_and_se(&f_vals);
            let (k_mean, k_se) = mean_and_se(&k_vals);
            let f_rel_se = f_se / f_mean;
            let k_rel_se = k_se / k_mean;
            let tail_mean = samples.iter().map(|s| s.tail_bound_ev).sum::<f64>() / (n_rep as f64);
            let total_calls: usize = samples.iter().map(|s| s.force_calls).sum();
            let calls_per_rep = (total_calls as f64) / (n_rep as f64);
            let k_over_gamma = k_mean / gamma_pub;
            let z = comparison_z(k_mean, k_se, temp);
            let pass = z.abs() <= 2.0 && f_rel_se <= TARGET_REL_SE;

            writeln!(
                summary_csv,
                "{temp:.1},{method},{planes_mean:.2},{:.4},{burn_in},{sample_steps},{n_rep},{f_mean:.8},{f_se:.8e},{f_rel_se:.6e},{tail_mean:.8e},{k_mean:.8e},{k_se:.8e},{k_rel_se:.6e},{gamma_pub:.8e},{k_over_gamma:.6},{z:.6},{pass},{calls_per_rep:.1},{total_calls},{wall_time_s:.4}",
                pilot_est.tau_int
            )
            .expect("summary row");
            summary_csv.flush().expect("flush summary");

            eprintln!(
                "T={temp} K {method}: planes={planes_mean:.1} n={n_rep} F*={f_mean:.6}±{f_se:.6} (rel={:.2}%) k/Gamma={k_over_gamma:.4} (rel_se={:.2}%) z={z:.4} pass={pass} calls/rep={calls_per_rep:.0} total_calls={total_calls} wall={wall_time_s:.2}s",
                f_rel_se * 100.0,
                k_rel_se * 100.0
            );
            assert!(
                pass,
                "T={temp} K {method} failed pass criteria: z={z:.4}, f_rel_se={f_rel_se:.4}, k_rel_se={k_rel_se:.4}"
            );
        }
    }
    Ok(())
}
