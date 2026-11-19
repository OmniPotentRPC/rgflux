//! Cu vacancy hop physical check with Mishin Cu01 EAM via rgpot's in-process
//! `EAMTablePot` at 1000 K and 1200 K zero-stress lattices.
//!
//! Compares even hyperplane spacing (`pafi_profile`) with adaptive placement
//! (`pafi_profile_adaptive`: monitor equidistribution + spectral-tail refinement)
//! using plane sampling windows sized from the measured `tau_int` of the
//! constraint force on the saddle dividing plane.
//!
//! Note on published benchmarks: Mishin, Mehl, Papaconstantopoulos, Voter, and
//! Kress, Phys. Rev. B 63, 224106 (2001), Table III reports only the 0 K static
//! vacancy formation energy (`E_v^f = 1.27 eV`) and 0 K static vacancy migration
//! energy (`E_v^m = 0.69 eV`) at `a_0 = 3.615 A`, and does not report finite-temperature
//! MD vacancy jump rates or diffusivities at 1000 K or 1200 K (whereas Daw and
//! Chandross, Phys. Rev. Materials 5, 043603 (2021), Table II used the
//! Foiles-Baskes-Daw 1986 EAM potential rather than Mishin Cu01). Accordingly,
//! no published finite-temperature MD jump-rate comparison is substituted here;
//! the 0 K static barrier at `a_0 = 3.615 A` is checked against Mishin et al.
//! (2001) Table III, and even vs. adaptive PAFI are compared head-to-head at
//! matched TST rate standard error (`SE(k_TST) / k_TST <= 3%` under the replicate
//! rule: pilot `14`, then `ceil(1.1 * n * (se / 0.03)^2)` capped at `2048`).

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
use rgflux::kappa::bennett_chandler::{
    BennettChandlerConfig, ShootingConfig, shoot_bennett_chandler,
};
use rgflux::pafi::{
    AdaptivePafiConfig, MonitorKind, PafiConfig, estimate_spectral_tail, pafi_profile,
    pafi_profile_adaptive, size_window_from_tau_int,
};
use rgflux::stats::{StatsConfig, batch_means};
use rgsaddle::strain_rate::{
    BOLTZMANN_EV_PER_K, ElasticDipole, Strain, VOIGT, activation_volume, angular_scale_rad_s,
    compliance_from_stiffness, voigt,
};
use rgsaddle::{
    BandConfig, BandStatus, BandSurface, Cell, CellMask, CiConfig, IdppConfig, JointChart,
    PointSurface, RelaxConfig, SaddleError, SolidNeb, finite_difference_cauchy_stress, idpp_path,
    joint_coordinate, relax_to_tolerance,
};

/// Equilibrium 0 K lattice constant of Mishin Cu01 EAM (Angstrom), Table III of
/// Mishin et al., Phys. Rev. B 63, 224106 (2001).
const CU_LATTICE_A0: f64 = 3.615;
/// Published 0 K static vacancy migration energy (eV) for Mishin Cu01 EAM,
/// Table III of Mishin et al., Phys. Rev. B 63, 224106 (2001).
const MISHIN_2001_TABLE_III_EVM_0K_EV: f64 = 0.69;
/// Standard atomic weight of copper (amu).
const CU_MASS_AMU: f64 = 63.546;
/// Atomic number of copper.
const CU_Z: i32 = 29;
/// FCC nearest-neighbour coordination number.
const FCC_COORDINATION: f64 = 12.0;
/// Dimensionless Langevin friction coefficient in code inverse time units
/// (`angular_scale_rad_s()`), near critical phonon damping for Cu.
const LANGEVIN_GAMMA_CODE: f64 = 1.0;
/// Integration timestep in femtoseconds.
const TIME_STEP_FS: f64 = 2.0;

const PILOT_REPLICATES: usize = 14;
const MAX_REPLICATES: usize = 2048;
const TARGET_REL_SE: f64 = 0.03;
const EVEN_PLANES: usize = 25;
const ADAPTIVE_INIT_PLANES: usize = 9;
const ADAPTIVE_MAX_PLANES: usize = 15;
const MIN_BURN_IN_STEPS: usize = 64;
const MIN_SAMPLE_STEPS: usize = 96;
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

fn code_time_step() -> f64 {
    TIME_STEP_FS * 1.0e-15 * angular_scale_rad_s()
}

fn code_gamma() -> f64 {
    LANGEVIN_GAMMA_CODE
}

/// Mishin, Mehl, Papaconstantopoulos, Voter, and Kress, Phys. Rev. B 63, 224106 (2001)
/// Table I parameters for Cu EAM1, tabulated onto a 10000-knot DYNAMO grid and
/// evaluated in-process by `rgpot::EAMTablePot` via `RgpotSurface::from_eam_table_memory`.
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

fn row9_of(cell: [[f64; 3]; 3]) -> [f64; 9] {
    [
        cell[0][0], cell[0][1], cell[0][2], cell[1][0], cell[1][1], cell[1][2], cell[2][0],
        cell[2][1], cell[2][2],
    ]
}

fn apply_strain_to_crystal(
    positions: ArrayView1<'_, f64>,
    cell: [[f64; 3]; 3],
    strain: Strain,
) -> (Array1<f64>, [[f64; 3]; 3]) {
    let eps = strain.tensor();
    let mut deform = eps;
    for (i, row) in deform.iter_mut().enumerate() {
        row[i] += 1.0;
    }
    let mut out_pos = Array1::zeros(positions.len());
    for atom in 0..(positions.len() / 3) {
        let r = [
            positions[3 * atom],
            positions[3 * atom + 1],
            positions[3 * atom + 2],
        ];
        for c in 0..3 {
            out_pos[3 * atom + c] = r[0] * deform[0][c] + r[1] * deform[1][c] + r[2] * deform[2][c];
        }
    }
    let mut out_cell = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out_cell[i][j] =
                cell[i][0] * deform[0][j] + cell[i][1] * deform[1][j] + cell[i][2] * deform[2][j];
        }
    }
    (out_pos, out_cell)
}

/// Measure the 6x6 Voigt elastic stiffness and compliance tensors of the bulk crystal
/// using `rgsaddle::Strain`, `rgsaddle::finite_difference_cauchy_stress`, and
/// `rgsaddle::compliance_from_stiffness`.
fn measure_cubic_compliance(
    surface: &CountingSurface,
    positions: ArrayView1<'_, f64>,
    cell: [[f64; 3]; 3],
) -> Result<[[f64; 6]; 6], FluxError> {
    let delta = 1.0e-3;
    let base_stress = voigt(finite_difference_cauchy_stress(
        surface, positions, cell, 1.0e-4,
    )?);
    let mut stiffness = [[0.0_f64; 6]; 6];
    for (col, &(i, j)) in VOIGT.iter().enumerate() {
        let mut t = [[0.0_f64; 3]; 3];
        t[i][j] = delta;
        t[j][i] = delta;
        let strain = Strain::from_tensor(t)?;
        let (pos_s, cell_s) = apply_strain_to_crystal(positions, cell, strain);
        let s_plus = voigt(finite_difference_cauchy_stress(
            surface,
            pos_s.view(),
            cell_s,
            1.0e-4,
        )?);
        let denom = if col < 3 { delta } else { 2.0 * delta };
        for row in 0..6 {
            stiffness[row][col] = (s_plus[row] - base_stress[row]) / denom;
        }
    }
    Ok(compliance_from_stiffness(&stiffness)?)
}

/// Determine the zero-stress lattice parameter `a(T)` at `temperature_k` using
/// `rgsaddle::JointChart`, `rgsaddle::joint_coordinate`, `rgsaddle::Strain`,
/// `rgsaddle::finite_difference_cauchy_stress`, and `rgsaddle::compliance_from_stiffness`
/// coupled to `rgflux::engine::native::NativeEngine` NVT thermal-stress sampling.
fn zero_stress_lattice_at(
    temperature_k: f64,
    seed: u64,
) -> Result<(f64, f64, [[f64; 6]; 6]), FluxError> {
    let repeats = 4usize;
    let (pos0, cell0_row9) = fcc_crystal(CU_LATTICE_A0, repeats, repeats, repeats);
    let cell0 = m3_of(cell0_row9);
    let n_atoms = pos0.len() / 3;
    let surface0 = CountingSurface::new(n_atoms, cell0_row9)?;
    let compliance = measure_cubic_compliance(&surface0, pos0.view(), cell0)?;
    let hydro_compliance = compliance[0][0] + compliance[0][1] + compliance[0][2];

    let mask = CellMask::voigt([true, true, true, false, false, false]);
    let chart = JointChart::new(cell0, n_atoms, mask, [[0.0; 3]; 3], 1.0)?;
    let mut eps_iso = 0.0_f64;
    let dt = code_time_step();
    let gamma = code_gamma();

    // Three compliance-preconditioned Newton steps on the JointChart driven by
    // the NVT ensemble-averaged total Cauchy stress (configurational + kinetic).
    let mut last_stress_se = 0.0_f64;
    for iter in 0..3 {
        let strain = Strain::hydrostatic(eps_iso);
        let (strained_pos, strained_cell) = apply_strain_to_crystal(pos0.view(), cell0, strain);
        let q = joint_coordinate(&chart, mask, strained_pos.view(), strained_cell)?;
        let (geom_pos, geom_cell) = chart.geometry(q.view())?;
        let cell_row9 = row9_of(geom_cell);
        let volume = geom_cell[0][0] * geom_cell[1][1] * geom_cell[2][2];

        let surface = CountingSurface::new(n_atoms, cell_row9)?;
        let mut engine = NativeEngine::new(
            surface,
            NativeConfig::default()
                .with_masses(Array1::from_elem(geom_pos.len(), CU_MASS_AMU))
                .with_cell(cell_row9)
                .with_seed(seed + (iter as u64) * 997)
                .with_thermostat(
                    ThermostatConfig::default()
                        .with_gamma(gamma)
                        .with_temperature(temperature_k)
                        .with_boltzmann(BOLTZMANN_EV_PER_K),
                ),
        )?;
        engine.set_positions(geom_pos.view())?;
        engine.shoot_velocities(seed + (iter as u64) * 997)?;

        let equil_steps = if iter == 0 { 240usize } else { 160usize };
        let prod_steps = if iter == 2 { 640usize } else { 320usize };
        let stride = 10usize;
        for _ in 0..equil_steps {
            engine.step(dt)?;
        }
        let mut hydro_samples = Vec::with_capacity(prod_steps / stride);
        for step in 0..prod_steps {
            engine.step(dt)?;
            if (step + 1) % stride == 0 {
                let s_cfg = finite_difference_cauchy_stress(
                    engine.surface(),
                    engine.positions(),
                    geom_cell,
                    1.0e-4,
                )?;
                let cfg_trace = (s_cfg[0][0] + s_cfg[1][1] + s_cfg[2][2]) / 3.0;
                let kin_stress = -(2.0 * engine.kinetic_energy()) / (3.0 * volume);
                hydro_samples.push(cfg_trace + kin_stress);
            }
        }
        let est = batch_means(&hydro_samples, &StatsConfig::default().with_min_batches(4))?;
        last_stress_se = est.standard_error;
        eps_iso -= hydro_compliance * est.mean;
    }

    let final_strain = Strain::hydrostatic(eps_iso);
    let (final_pos, final_cell) = apply_strain_to_crystal(pos0.view(), cell0, final_strain);
    let q_final = joint_coordinate(&chart, mask, final_pos.view(), final_cell)?;
    let (_, geom_cell) = chart.geometry(q_final.view())?;
    let a_t = geom_cell[0][0] / (repeats as f64);
    let a_se = CU_LATTICE_A0 * hydro_compliance.abs() * last_stress_se;
    Ok((a_t, a_se, compliance))
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

struct VacancyPathReport {
    path: Array2<f64>,
    cell: [f64; 9],
    static_barrier_ev: f64,
    activation_volume_a3: f64,
    neb_converged: bool,
    neb_steps: usize,
    neb_max_force: f64,
}

fn prepare_vacancy_path(
    lattice: f64,
    compliance: &[[f64; 6]; 6],
) -> Result<VacancyPathReport, FluxError> {
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

    let mut band_cfg = BandConfig::default()
        .with_cell(Cell(m3_of(cell)))
        .with_force_tol(5.0e-3)
        .with_max_move(0.15)
        .with_climbing(CiConfig::after(0.2))
        .with_min_mode(None);
    band_cfg.saddle_check = None;
    let mut neb = SolidNeb::frozen(band_cfg, initial)?;
    let mut last_rep = None;
    for _ in 0..1_000 {
        let rep = neb.step_surface(&surface)?;
        let done = rep.status == BandStatus::Converged;
        last_rep = Some(rep);
        if done {
            break;
        }
    }
    let rep = last_rep.ok_or(FluxError::InvalidConfig("SolidNeb ran zero steps"))?;
    let neb_converged = rep.status == BandStatus::Converged;
    if !neb_converged {
        return Err(FluxError::InvalidConfig(
            "SolidNeb vacancy migration path did not converge within 1000 steps",
        ));
    }

    let path = unwrap_path(&neb.cartesian()?, cell);
    let mut saddle_idx = 1usize;
    let mut saddle_e = f64::NEG_INFINITY;
    for (idx, row) in path.rows().into_iter().enumerate() {
        let (e, _) = PointSurface::eval(&surface, row)?;
        if e > saddle_e {
            saddle_e = e;
            saddle_idx = idx;
        }
    }
    let static_barrier_ev = saddle_e - r_rel.energy;

    // Compute elastic dipoles and activation volume using rgsaddle::ElasticDipole and
    // rgsaddle::activation_volume.
    let (bulk_pos, bulk_cell_row9) = fcc_crystal(lattice, 4, 4, 4);
    let bulk_surface = CountingSurface::new(bulk_pos.len() / 3, bulk_cell_row9)?;
    let bulk_stress = finite_difference_cauchy_stress(
        &bulk_surface,
        bulk_pos.view(),
        m3_of(bulk_cell_row9),
        1.0e-4,
    )?;
    let min_stress =
        finite_difference_cauchy_stress(&surface, r_rel.positions.view(), m3_of(cell), 1.0e-4)?;
    let sad_stress =
        finite_difference_cauchy_stress(&surface, path.row(saddle_idx), m3_of(cell), 1.0e-4)?;
    let volume = cell[0] * cell[4] * cell[8];
    let min_dipole = ElasticDipole::from_stress(volume, min_stress, bulk_stress)?;
    let sad_dipole = ElasticDipole::from_stress(volume, sad_stress, bulk_stress)?;
    let omega_tensor = activation_volume(min_dipole.tensor(), sad_dipole.tensor(), compliance)?;
    let activation_volume_a3 = (omega_tensor[0][0] + omega_tensor[1][1] + omega_tensor[2][2]) / 3.0;

    Ok(VacancyPathReport {
        path,
        cell,
        static_barrier_ev,
        activation_volume_a3,
        neb_converged,
        neb_steps: rep.iteration,
        neb_max_force: rep.max_force,
    })
}

/// Measure the Bennett-Chandler reactive-flux transmission coefficient `kappa` on the
/// saddle dividing hyperplane using `rgflux::kappa::bennett_chandler::shoot_bennett_chandler`.
fn measure_saddle_kappa(
    path: &Array2<f64>,
    cell: [f64; 9],
    temperature_k: f64,
    seed: u64,
    stats_cfg: &StatsConfig,
) -> Result<(f64, f64, usize), FluxError> {
    let n_atoms = path.ncols() / 3;
    let surface = CountingSurface::new(n_atoms, cell)?;
    let dt = code_time_step();
    let gamma = code_gamma();
    let mut engine = NativeEngine::new(
        surface,
        NativeConfig::default()
            .with_masses(Array1::from_elem(path.ncols(), CU_MASS_AMU))
            .with_cell(cell)
            .with_seed(seed)
            .with_thermostat(
                ThermostatConfig::default()
                    .with_gamma(gamma)
                    .with_temperature(temperature_k)
                    .with_boltzmann(BOLTZMANN_EV_PER_K),
            ),
    )?;
    let saddle_row = path.row(3).to_owned();
    let dir = &path.row(4) - &path.row(2);
    let norm = dir.dot(&dir).sqrt();
    let normal = dir.mapv(|v| v / norm);
    let offset = normal.dot(&saddle_row);
    let dividing_plane = HyperplaneConstraint::new(normal, offset)?;

    engine.set_constraint(Some(dividing_plane.clone()))?;
    engine.set_positions(saddle_row.view())?;
    engine.shoot_velocities(seed)?;
    for _ in 0..200 {
        engine.step(dt)?;
    }
    let n_configs = 32usize;
    let stride = 12usize;
    let mut configs = Array2::zeros((n_configs, path.ncols()));
    for i in 0..n_configs {
        for _ in 0..stride {
            engine.step(dt)?;
        }
        configs.row_mut(i).assign(&engine.positions());
    }

    // Propagate the short reactive-flux shooting trajectories with Hamiltonian (NVE,
    // gamma = 0) dynamics while drawing initial Maxwell-Boltzmann velocities at T.
    engine.set_thermostat(
        ThermostatConfig::default()
            .with_gamma(0.0)
            .with_temperature(temperature_k)
            .with_boltzmann(BOLTZMANN_EV_PER_K),
    )?;
    let shoot_cfg = ShootingConfig::default()
        .with_steps(40)
        .with_time_step(dt)
        .with_antithetic(true)
        .with_seed(seed ^ 0x517C_C1B7_2722_0A95)
        .with_estimator(
            BennettChandlerConfig::default()
                .with_plateau_start_fraction(0.5)
                .with_plateau_end_fraction(1.0)
                .with_stats(stats_cfg.clone()),
        );
    let bc = shoot_bennett_chandler(&mut engine, &dividing_plane, configs.view(), &shoot_cfg)?;
    let kappa_calls = engine.surface().calls();
    Ok((bc.plateau, bc.standard_error, kappa_calls))
}

#[derive(Clone, Copy, Debug)]
struct ReplicateSample {
    replicate: usize,
    seed: u64,
    planes: usize,
    f_star_ev: f64,
    f_star_se_ev: f64,
    tail_bound_ev: f64,
    k_tst_hz: f64,
    k_bc_hz: f64,
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
    kappa_bc: f64,
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
        let mut completed_replicates = samples.len();
        while completed_replicates < target_n {
            let batch_end = (completed_replicates + workers).min(target_n);
            let batch_res: Result<Vec<ReplicateSample>, FluxError> = thread::scope(|scope| {
                let mut handles = Vec::with_capacity(batch_end - completed_replicates);
                for rep in completed_replicates..batch_end {
                    let tag = if adaptive { 2_u64 } else { 1_u64 };
                    let seed = BASE_SEED
                        + (t_index as u64) * 1_000_003
                        + tag * 100_003
                        + (rep as u64) * 7_919;
                    handles.push(scope.spawn(move || -> Result<ReplicateSample, FluxError> {
                        let n_atoms = path.ncols() / 3;
                        let dt = code_time_step();
                        let gamma = code_gamma();
                        let mut last_err = None;
                        for attempt in 0..8_u64 {
                            let attempt_seed =
                                seed.wrapping_add(attempt.wrapping_mul(1_000_000_007));
                            let surface = CountingSurface::new(n_atoms, cell)?;
                            let mut engine = NativeEngine::new(
                                surface,
                                NativeConfig::default()
                                    .with_masses(Array1::from_elem(path.ncols(), CU_MASS_AMU))
                                    .with_cell(cell)
                                    .with_seed(attempt_seed)
                                    .with_thermostat(
                                        ThermostatConfig::default()
                                            .with_gamma(gamma)
                                            .with_temperature(temperature_k)
                                            .with_boltzmann(BOLTZMANN_EV_PER_K),
                                    ),
                            )?;
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
                                            .with_seed(attempt_seed)
                                            .with_stats(stats_cfg.clone()),
                                    )
                                    .with_max_planes(ADAPTIVE_MAX_PLANES)
                                    .with_max_refinements(3)
                                    .with_monitor(MonitorKind::Combined);
                                match pafi_profile_adaptive(&mut engine, path.view(), &cfg) {
                                    Ok(res) => {
                                        let wall_s = t0.elapsed().as_secs_f64();
                                        let calls = engine.surface().calls();
                                        let k_tst_hz = res.profile.normalized_rate * scale_hz;
                                        return Ok(ReplicateSample {
                                            replicate: rep,
                                            seed: attempt_seed,
                                            planes: res.profile.arc_lengths.len(),
                                            f_star_ev: res.profile.barrier,
                                            f_star_se_ev: res.profile.barrier_error,
                                            tail_bound_ev: res.tail.tail_bound,
                                            k_tst_hz,
                                            k_bc_hz: kappa_bc * k_tst_hz,
                                            force_calls: calls,
                                            wall_s,
                                        });
                                    }
                                    Err(err) => last_err = Some(err),
                                }
                            } else {
                                let cfg = PafiConfig::default()
                                    .with_planes(EVEN_PLANES)
                                    .with_burn_in_steps(burn_in)
                                    .with_sample_steps(sample_steps)
                                    .with_time_step(dt)
                                    .with_zero_temperature_tol(0.05)
                                    .with_seed(attempt_seed)
                                    .with_stats(stats_cfg.clone());
                                match pafi_profile(&mut engine, path.view(), &cfg) {
                                    Ok(prof) => {
                                        let wall_s = t0.elapsed().as_secs_f64();
                                        let calls = engine.surface().calls();
                                        let tail = estimate_spectral_tail(
                                            &prof.arc_lengths,
                                            prof.gradients(),
                                            prof.barrier_error,
                                        )?;
                                        let k_tst_hz = prof.normalized_rate * scale_hz;
                                        return Ok(ReplicateSample {
                                            replicate: rep,
                                            seed: attempt_seed,
                                            planes: prof.arc_lengths.len(),
                                            f_star_ev: prof.barrier,
                                            f_star_se_ev: prof.barrier_error,
                                            tail_bound_ev: tail.tail_bound,
                                            k_tst_hz,
                                            k_bc_hz: kappa_bc * k_tst_hz,
                                            force_calls: calls,
                                            wall_s,
                                        });
                                    }
                                    Err(err) => last_err = Some(err),
                                }
                            }
                        }
                        Err(last_err.unwrap_or(FluxError::InvalidConfig(
                            "replicate failed after 8 attempts",
                        )))
                    }));
                }
                handles
                    .into_iter()
                    .map(|h| h.join().expect("replicate thread panicked"))
                    .collect()
            });
            samples.extend(batch_res?);
            completed_replicates = batch_end;
        }

        let f_vals: Vec<f64> = samples.iter().map(|s| s.f_star_ev).collect();
        let k_vals: Vec<f64> = samples.iter().map(|s| s.k_tst_hz).collect();
        let (f_mean, f_se) = mean_and_se(&f_vals);
        let (k_mean, k_se) = mean_and_se(&k_vals);
        let f_rel_se = f_se / f_mean;
        let k_rel_se = k_se / k_mean;
        eprintln!(
            "  [{temperature_k} K adaptive={adaptive}] n={} F*={f_mean:.6}±{f_se:.6} (f_rel={f_rel_se:.4}) k_TST={k_mean:.4e}±{k_se:.4e} (k_rel={k_rel_se:.4})",
            samples.len()
        );
        // Gate on the rate relative standard error SE(k_TST) / k_TST <= 3%.
        let next_n = next_replicate_count(samples.len(), k_rel_se);
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
        "temperature_K,atoms,lattice_A,lattice_se_A,strain,static_barrier_eV,activation_volume_A3,neb_converged,neb_steps,neb_max_force_eV_A,kappa_bc,kappa_bc_se,kappa_force_calls,mishin_2001_table_iii_0K_barrier_eV"
    )
    .expect("lattice header");

    let mut summary_csv =
        File::create(out_dir.join("cu_vacancy_even_vs_adaptive.csv")).expect("summary csv");
    writeln!(
        summary_csv,
        "temperature_K,method,planes_mean,tau_int_steps,burn_in_steps,sample_steps,replicates_n,f_star_eV,f_star_se_eV,f_star_rel_se,tail_bound_eV,tail_bound_se_eV,k_tst_Hz,k_tst_se_Hz,k_tst_rel_se,kappa_bc,kappa_bc_se,k_bc_Hz,k_bc_se_Hz,k_bc_rel_se,pass,force_calls_per_rep,total_force_calls,rep_wall_mean_s,rep_wall_se_s,campaign_wall_s"
    )
    .expect("summary header");

    let mut rep_csv =
        File::create(out_dir.join("cu_vacancy_replicates.csv")).expect("replicates csv");
    writeln!(
        rep_csv,
        "temperature_K,method,replicate,seed,planes,f_star_eV,f_star_se_eV,tail_bound_eV,k_tst_Hz,k_bc_Hz,force_calls,wall_s"
    )
    .expect("replicates header");

    let stats_cfg = StatsConfig::default();

    // Verify 0 K static vacancy migration barrier against Mishin et al. (2001) Table III (0.69 eV).
    let (pos0, cell0_row9) = fcc_crystal(CU_LATTICE_A0, 4, 4, 4);
    let surf0 = CountingSurface::new(pos0.len() / 3, cell0_row9)?;
    let comp0 = measure_cubic_compliance(&surf0, pos0.view(), m3_of(cell0_row9))?;
    let path0 = prepare_vacancy_path(CU_LATTICE_A0, &comp0)?;
    eprintln!(
        "0 K check at a0={CU_LATTICE_A0} A: Em={:.6} eV (Mishin 2001 Table III: {MISHIN_2001_TABLE_III_EVM_0K_EV:.2} eV), neb_converged={}, steps={}, max_force={:.3e} eV/A, Omega={:.4} A^3",
        path0.static_barrier_ev,
        path0.neb_converged,
        path0.neb_steps,
        path0.neb_max_force,
        path0.activation_volume_a3
    );

    for (idx, &temp) in [1000.0_f64, 1200.0].iter().enumerate() {
        let (a_t, a_se, compliance) = zero_stress_lattice_at(temp, BASE_SEED + idx as u64)?;
        let strain = a_t / CU_LATTICE_A0 - 1.0;
        let path_rep = prepare_vacancy_path(a_t, &compliance)?;
        let path = path_rep.path;
        let cell = path_rep.cell;
        let static_barrier = path_rep.static_barrier_ev;

        let (kappa_bc, kappa_bc_se, kappa_calls) =
            measure_saddle_kappa(&path, cell, temp, BASE_SEED + 200 + idx as u64, &stats_cfg)?;

        writeln!(
            lattice_csv,
            "{temp:.1},{},{a_t:.8},{a_se:.8e},{strain:.8e},{static_barrier:.8},{:.8},{},{},{:.8e},{kappa_bc:.8},{kappa_bc_se:.8e},{kappa_calls},{MISHIN_2001_TABLE_III_EVM_0K_EV:.2}",
            path.ncols() / 3,
            path_rep.activation_volume_a3,
            path_rep.neb_converged,
            path_rep.neb_steps,
            path_rep.neb_max_force
        )
        .expect("lattice row");
        lattice_csv.flush().expect("flush lattice");

        // Measure tau_int on the saddle hyperplane to size the plane sampling window.
        let n_atoms = path.ncols() / 3;
        let surface = CountingSurface::new(n_atoms, cell)?;
        let dt = code_time_step();
        let gamma = code_gamma();
        let mut pilot_engine = NativeEngine::new(
            surface,
            NativeConfig::default()
                .with_masses(Array1::from_elem(path.ncols(), CU_MASS_AMU))
                .with_cell(cell)
                .with_seed(BASE_SEED + 50 + idx as u64)
                .with_thermostat(
                    ThermostatConfig::default()
                        .with_gamma(gamma)
                        .with_temperature(temp)
                        .with_boltzmann(BOLTZMANN_EV_PER_K),
                ),
        )?;
        let saddle_row = path.row(3).to_owned();
        let dir = &path.row(4) - &path.row(2);
        let norm = dir.dot(&dir).sqrt();
        let normal = dir.mapv(|v| v / norm);
        let offset = normal.dot(&saddle_row);
        pilot_engine.set_constraint(Some(HyperplaneConstraint::new(normal, offset)?))?;
        pilot_engine.set_positions(saddle_row.view())?;
        pilot_engine.shoot_velocities(BASE_SEED + 50 + idx as u64)?;
        for _ in 0..200 {
            pilot_engine.step(dt)?;
        }
        let mut pilot_forces = Vec::with_capacity(128);
        for _ in 0..128 {
            let rep = pilot_engine.step(dt)?;
            pilot_forces.push(rep.constraint_force.unwrap_or(0.0));
        }
        let pilot_est = batch_means(&pilot_forces, &stats_cfg)?;
        let (burn_in, sample_steps) = size_window_from_tau_int(
            pilot_est.tau_int,
            MIN_BURN_IN_STEPS,
            MIN_SAMPLE_STEPS,
            &stats_cfg,
        )?;
        let scale_hz = angular_scale_rad_s() * FCC_COORDINATION;

        eprintln!(
            "T={temp} K: a(T)={a_t:.6}±{a_se:.6} A, Em={static_barrier:.6} eV, neb_converged={} ({} steps, max_f={:.2e}), kappa_BC={kappa_bc:.4}±{kappa_bc_se:.4}, tau_int={:.2} -> burn={burn_in}, sample={sample_steps}",
            path_rep.neb_converged, path_rep.neb_steps, path_rep.neb_max_force, pilot_est.tau_int
        );

        for (method, adaptive) in [("even", false), ("adaptive", true)] {
            let (samples, campaign_wall_s) = run_campaign(
                temp,
                idx,
                adaptive,
                &path,
                cell,
                burn_in,
                sample_steps,
                scale_hz,
                kappa_bc,
                &stats_cfg,
            )?;
            for s in &samples {
                writeln!(
                    rep_csv,
                    "{temp:.1},{method},{},{},{},{:.8},{:.8e},{:.8e},{:.8e},{:.8e},{},{:.6}",
                    s.replicate,
                    s.seed,
                    s.planes,
                    s.f_star_ev,
                    s.f_star_se_ev,
                    s.tail_bound_ev,
                    s.k_tst_hz,
                    s.k_bc_hz,
                    s.force_calls,
                    s.wall_s
                )
                .expect("rep row");
            }
            rep_csv.flush().expect("flush rep");

            let n_rep = samples.len();
            let planes_mean = samples.iter().map(|s| s.planes as f64).sum::<f64>() / (n_rep as f64);
            let f_vals: Vec<f64> = samples.iter().map(|s| s.f_star_ev).collect();
            let k_tst_vals: Vec<f64> = samples.iter().map(|s| s.k_tst_hz).collect();
            let tail_vals: Vec<f64> = samples.iter().map(|s| s.tail_bound_ev).collect();
            let wall_vals: Vec<f64> = samples.iter().map(|s| s.wall_s).collect();

            let (f_mean, f_se) = mean_and_se(&f_vals);
            let (k_tst_mean, k_tst_se) = mean_and_se(&k_tst_vals);
            let (tail_mean, tail_se) = mean_and_se(&tail_vals);
            let (rep_wall_mean, rep_wall_se) = mean_and_se(&wall_vals);

            let f_rel_se = f_se / f_mean;
            let k_tst_rel_se = k_tst_se / k_tst_mean;
            let k_bc_mean = kappa_bc * k_tst_mean;
            let k_bc_rel_se = k_tst_rel_se.hypot(kappa_bc_se / kappa_bc);
            let k_bc_se = k_bc_mean * k_bc_rel_se;

            let total_calls: usize = samples.iter().map(|s| s.force_calls).sum();
            let calls_per_rep = (total_calls as f64) / (n_rep as f64);
            let pass = k_tst_rel_se <= TARGET_REL_SE;

            writeln!(
                summary_csv,
                "{temp:.1},{method},{planes_mean:.2},{:.4},{burn_in},{sample_steps},{n_rep},{f_mean:.8},{f_se:.8e},{f_rel_se:.6e},{tail_mean:.8e},{tail_se:.8e},{k_tst_mean:.8e},{k_tst_se:.8e},{k_tst_rel_se:.6e},{kappa_bc:.8},{kappa_bc_se:.8e},{k_bc_mean:.8e},{k_bc_se:.8e},{k_bc_rel_se:.6e},{pass},{calls_per_rep:.1},{total_calls},{rep_wall_mean:.6},{rep_wall_se:.6e},{campaign_wall_s:.4}",
                pilot_est.tau_int
            )
            .expect("summary row");
            summary_csv.flush().expect("flush summary");

            eprintln!(
                "T={temp} K {method}: planes={planes_mean:.2} n={n_rep} F*={f_mean:.6}±{f_se:.6} (rel={:.2}%) tail={tail_mean:.3e}±{tail_se:.3e} k_TST={k_tst_mean:.4e}±{k_tst_se:.4e} (rel={:.2}%) kappa_BC={kappa_bc:.4}±{kappa_bc_se:.4} k_BC={k_bc_mean:.4e}±{k_bc_se:.4e} pass={pass} calls/rep={calls_per_rep:.1} total_calls={total_calls} rep_wall={rep_wall_mean:.3}s campaign_wall={campaign_wall_s:.2}s",
                f_rel_se * 100.0,
                k_tst_rel_se * 100.0
            );
        }
    }
    Ok(())
}
