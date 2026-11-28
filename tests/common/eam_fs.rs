// One-element `pair_style eam/fs` tables in an orthorhombic periodic box.
//
// The file layout is the Finnis-Sinclair setfl of the LAMMPS pair_eam page
// (docs.lammps.org/pair_eam.html): three comment lines, the element line,
// `Nrho drho Nr dr cutoff`, then `Z mass a lattice`, F(ρ) on Nrho points
// and ρ(r) on Nr points, and finally r φ(r) on Nr points. The tables go
// through `PairEAM::interpolate`, the seven-coefficient cubic of LAMMPS,
// and are looked up the way `PairEAM::compute` does, with F extended
// linearly above ρmax. Energies and forces therefore follow
// `pair_style eam/fs` to rounding, which is what the reference PAFI code
// samples. rgpot's `EAMTablePot` takes the same steps.

use std::cell::RefCell;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

/// `eam/fs` table after `PairEAM::interpolate`. Index 1 is `r = 0` or
/// `ρ = 0`, as in LAMMPS; index 0 is unused.
struct EamTable {
    nrho: usize,
    nr: usize,
    rdrho: f64,
    rdr: f64,
    rhomax: f64,
    cutoff: f64,
    frho: Vec<[f64; 7]>,
    rhor: Vec<[f64; 7]>,
    z2r: Vec<[f64; 7]>,
}

impl EamTable {
    fn read(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        Self::parse(&text).map_err(|error| format!("{}: {error}", path.display()))
    }

    fn parse(text: &str) -> Result<Self, String> {
        let mut lines = text.lines().skip(3);
        let count: usize = lines
            .next()
            .and_then(|line| line.split_whitespace().next())
            .and_then(|token| token.parse().ok())
            .ok_or("eam/fs: no element count")?;
        if count != 1 {
            return Err(format!("eam/fs: {count} elements, this reader takes one"));
        }
        let grid = numbers(lines.next().ok_or("eam/fs: no grid line")?)?;
        if grid.len() < 5 {
            return Err("eam/fs: the grid line needs Nrho drho Nr dr cutoff".into());
        }
        let (nrho, drho, nr, dr, cutoff) = (
            grid[0] as usize,
            grid[1],
            grid[2] as usize,
            grid[3],
            grid[4],
        );
        if nrho < 5 || nr < 5 || !(drho > 0.0 && dr > 0.0 && cutoff > 0.0) {
            return Err("eam/fs: the grid is too small or not positive".into());
        }
        lines.next().ok_or("eam/fs: no element line")?;
        let mut tokens = lines.flat_map(str::split_whitespace);
        let mut take = |n: usize, what: &str| -> Result<Vec<f64>, String> {
            let mut values = vec![0.0; n + 1];
            for slot in values.iter_mut().skip(1) {
                let token = tokens
                    .next()
                    .ok_or_else(|| format!("eam/fs: the file ends in {what}"))?;
                *slot = token
                    .parse()
                    .map_err(|_| format!("eam/fs {what}: {token}"))?;
            }
            Ok(values)
        };
        let frho = take(nrho, "F(rho)")?;
        let rhor = take(nr, "rho(r)")?;
        let z2r = take(nr, "r phi(r)")?;
        Ok(Self {
            nrho,
            nr,
            rdrho: 1.0 / drho,
            rdr: 1.0 / dr,
            rhomax: (nrho - 1) as f64 * drho,
            cutoff,
            frho: interpolate(&frho, drho),
            rhor: interpolate(&rhor, dr),
            z2r: interpolate(&z2r, dr),
        })
    }

    fn r_knot(&self, r: f64) -> (usize, f64) {
        let p = r * self.rdr + 1.0;
        let m = (p as usize).min(self.nr - 1);
        (m, (p - m as f64).min(1.0))
    }

    fn rho_knot(&self, rho: f64) -> (usize, f64) {
        let p = rho * self.rdrho + 1.0;
        let m = (p as i64).clamp(1, self.nrho as i64 - 1) as usize;
        (m, (p - m as f64).min(1.0))
    }
}

fn numbers(line: &str) -> Result<Vec<f64>, String> {
    line.split_whitespace()
        .map(|token| token.parse::<f64>().map_err(|_| format!("eam/fs: {token}")))
        .collect()
}

/// `PairEAM::interpolate`: coefficients 6, 5, 4, 3 are the value and the
/// cubic in the knot fraction, 2, 1, 0 its derivative per unit argument.
fn interpolate(f: &[f64], delta: f64) -> Vec<[f64; 7]> {
    let n = f.len() - 1;
    let mut s = vec![[0.0; 7]; n + 1];
    for (row, value) in s.iter_mut().zip(f).skip(1) {
        row[6] = *value;
    }
    s[1][5] = s[2][6] - s[1][6];
    s[2][5] = 0.5 * (s[3][6] - s[1][6]);
    s[n - 1][5] = 0.5 * (s[n][6] - s[n - 2][6]);
    s[n][5] = s[n][6] - s[n - 1][6];
    for m in 3..=n - 2 {
        s[m][5] = ((s[m - 2][6] - s[m + 2][6]) + 8.0 * (s[m + 1][6] - s[m - 1][6])) / 12.0;
    }
    for m in 1..n {
        s[m][4] = 3.0 * (s[m + 1][6] - s[m][6]) - 2.0 * s[m][5] - s[m + 1][5];
        s[m][3] = s[m][5] + s[m + 1][5] - 2.0 * (s[m + 1][6] - s[m][6]);
    }
    s[n][4] = 0.0;
    s[n][3] = 0.0;
    for row in s.iter_mut().skip(1) {
        row[2] = row[5] / delta;
        row[1] = 2.0 * row[4] / delta;
        row[0] = 3.0 * row[3] / delta;
    }
    s
}

fn value(c: &[f64; 7], p: f64) -> f64 {
    ((c[3] * p + c[4]) * p + c[5]) * p + c[6]
}

fn slope(c: &[f64; 7], p: f64) -> f64 {
    (c[0] * p + c[1]) * p + c[2]
}

/// Pairs within `cutoff + skin` of a reference configuration.
struct Verlet {
    owner: u64,
    reference: Vec<f64>,
    pairs: Vec<(u32, u32)>,
}

thread_local! {
    static VERLET: RefCell<Option<Verlet>> = const { RefCell::new(None) };
}

static NEXT_CRYSTAL: AtomicU64 = AtomicU64::new(1);

/// Periodic orthorhombic crystal on one `eam/fs` table.
///
/// Each thread keeps a Verlet list with a skin of [`Self::SKIN`] Å and
/// rebuilds it when an atom has moved half the skin, so every pair inside
/// the cutoff is always on the list.
struct EamCrystal {
    table: EamTable,
    lengths: [f64; 3],
    id: u64,
}

impl EamCrystal {
    const SKIN: f64 = 1.0;

    fn new(table: EamTable, lengths: [f64; 3]) -> Result<Self, String> {
        let reach = table.cutoff + Self::SKIN;
        if lengths
            .iter()
            .any(|length| !(length.is_finite() && 2.0 * reach < *length))
        {
            return Err(format!(
                "every box length must exceed twice the cutoff plus skin, {:.3} Å",
                2.0 * reach
            ));
        }
        Ok(Self {
            table,
            lengths,
            id: NEXT_CRYSTAL.fetch_add(1, Ordering::Relaxed),
        })
    }

    fn image(&self, d: &mut [f64; 3]) {
        for (component, length) in d.iter_mut().zip(self.lengths) {
            *component -= length * (*component / length).round();
        }
    }

    fn delta(&self, x: &[f64], i: usize, j: usize) -> [f64; 3] {
        let mut d = [
            x[3 * i] - x[3 * j],
            x[3 * i + 1] - x[3 * j + 1],
            x[3 * i + 2] - x[3 * j + 2],
        ];
        self.image(&mut d);
        d
    }

    fn build(&self, x: &[f64]) -> Verlet {
        let atoms = x.len() / 3;
        let reach = (self.table.cutoff + Self::SKIN).powi(2);
        let mut pairs = Vec::new();
        for i in 0..atoms {
            for j in i + 1..atoms {
                let d = self.delta(x, i, j);
                if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < reach {
                    pairs.push((i as u32, j as u32));
                }
            }
        }
        Verlet {
            owner: self.id,
            reference: x.to_vec(),
            pairs,
        }
    }

    fn stale(&self, list: &Verlet, x: &[f64]) -> bool {
        if list.owner != self.id || list.reference.len() != x.len() {
            return true;
        }
        let limit = (0.5 * Self::SKIN).powi(2);
        (0..x.len() / 3).any(|atom| {
            let mut d = [
                x[3 * atom] - list.reference[3 * atom],
                x[3 * atom + 1] - list.reference[3 * atom + 1],
                x[3 * atom + 2] - list.reference[3 * atom + 2],
            ];
            self.image(&mut d);
            d[0] * d[0] + d[1] * d[1] + d[2] * d[2] > limit
        })
    }

    /// Energy and Cartesian gradient, `PairEAM::compute` on a full list.
    fn energy_gradient(&self, x: &[f64]) -> (f64, Vec<f64>) {
        VERLET.with(|cell| {
            let mut slot = cell.borrow_mut();
            if slot.as_ref().is_none_or(|list| self.stale(list, x)) {
                *slot = Some(self.build(x));
            }
            let list = slot.as_ref().expect("Verlet list");
            self.evaluate(x, &list.pairs)
        })
    }

    fn evaluate(&self, x: &[f64], pairs: &[(u32, u32)]) -> (f64, Vec<f64>) {
        let table = &self.table;
        let atoms = x.len() / 3;
        let cutsq = table.cutoff * table.cutoff;
        let mut rho = vec![0.0; atoms];
        let mut near = Vec::with_capacity(pairs.len());
        for &(i, j) in pairs {
            let (i, j) = (i as usize, j as usize);
            let d = self.delta(x, i, j);
            let rsq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if rsq < cutsq {
                let r = rsq.sqrt();
                let (m, p) = table.r_knot(r);
                let density = value(&table.rhor[m], p);
                rho[i] += density;
                rho[j] += density;
                near.push((i, j, d, r, m, p));
            }
        }
        let mut energy = 0.0;
        let mut fp = vec![0.0; atoms];
        for (atom, density) in rho.iter().enumerate() {
            let (m, p) = table.rho_knot(*density);
            let c = &table.frho[m];
            fp[atom] = slope(c, p);
            let mut embedding = value(c, p);
            if *density > table.rhomax {
                embedding += fp[atom] * (density - table.rhomax);
            }
            energy += embedding;
        }
        let mut gradient = vec![0.0; x.len()];
        for (i, j, d, r, m, p) in near {
            let rho_p = slope(&table.rhor[m], p);
            let c = &table.z2r[m];
            let z2p = slope(c, p);
            let z2 = value(c, p);
            let recip = 1.0 / r;
            let phi = z2 * recip;
            let phip = z2p * recip - phi * recip;
            let psip = fp[i] * rho_p + fp[j] * rho_p + phip;
            energy += phi;
            let scale = psip * recip;
            for k in 0..3 {
                gradient[3 * i + k] += d[k] * scale;
                gradient[3 * j + k] -= d[k] * scale;
            }
        }
        (energy, gradient)
    }
}

impl rgsaddle::PointSurface for EamCrystal {
    fn eval(
        &self,
        x: ndarray::ArrayView1<f64>,
    ) -> Result<(f64, ndarray::Array1<f64>), rgsaddle::SaddleError> {
        if x.is_empty() || !x.len().is_multiple_of(3) {
            return Err(rgsaddle::SaddleError::Shape(
                "EAM positions must be whole atoms".into(),
            ));
        }
        let (energy, gradient) = match x.as_slice() {
            Some(slice) => self.energy_gradient(slice),
            None => self.energy_gradient(&x.to_vec()),
        };
        if !energy.is_finite() {
            return Err(rgsaddle::SaddleError::NonFinite("EAM energy"));
        }
        Ok((energy, ndarray::Array1::from(gradient)))
    }

    fn excluded_modes(
        &self,
        x: ndarray::ArrayView1<f64>,
    ) -> Result<ndarray::Array2<f64>, rgsaddle::SaddleError> {
        let atoms = x.len() / 3;
        let mut modes = ndarray::Array2::zeros((3, x.len()));
        let weight = (atoms as f64).sqrt().recip();
        for atom in 0..atoms {
            for k in 0..3 {
                modes[[k, 3 * atom + k]] = weight;
            }
        }
        Ok(modes)
    }
}

/// Orthorhombic box lengths and positions in atom-id order from a LAMMPS
/// `atomic` data file whose box starts at the origin. Image flags are
/// ignored, as `gather_atoms` returns wrapped coordinates.
fn read_lammps_data(path: &Path) -> Result<([f64; 3], Vec<f64>), String> {
    let text =
        std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut atoms = None;
    let mut lengths = [0.0; 3];
    let mut lines = text.lines();
    for line in lines.by_ref() {
        let words: Vec<&str> = line.split_whitespace().collect();
        match words.as_slice() {
            [count, "atoms"] => atoms = count.parse::<usize>().ok(),
            [lo, hi, low, high] if low.ends_with("lo") && high.ends_with("hi") => {
                let axis = match *low {
                    "xlo" => 0,
                    "ylo" => 1,
                    "zlo" => 2,
                    _ => return Err(format!("{}: box line {line}", path.display())),
                };
                let lo: f64 = lo.parse().map_err(|_| format!("box: {line}"))?;
                let hi: f64 = hi.parse().map_err(|_| format!("box: {line}"))?;
                if lo != 0.0 {
                    return Err(format!("{}: the box must start at 0", path.display()));
                }
                lengths[axis] = hi - lo;
            }
            [tilt, ..] if *tilt == "xy" || words.contains(&"xy") => {
                return Err(format!("{}: triclinic boxes are not read", path.display()));
            }
            ["Atoms", ..] => break,
            _ => {}
        }
    }
    let atoms = atoms.ok_or_else(|| format!("{}: no atom count", path.display()))?;
    let mut positions = vec![f64::NAN; 3 * atoms];
    let mut seen = 0usize;
    for line in lines {
        let words: Vec<&str> = line.split_whitespace().collect();
        if words.is_empty() {
            if seen == 0 {
                continue;
            }
            break;
        }
        if words.len() < 5 {
            return Err(format!("{}: atom line {line}", path.display()));
        }
        let id: usize = words[0].parse().map_err(|_| format!("atom id: {line}"))?;
        if id == 0 || id > atoms {
            return Err(format!("{}: atom id {id} out of range", path.display()));
        }
        for k in 0..3 {
            positions[3 * (id - 1) + k] = words[2 + k]
                .parse()
                .map_err(|_| format!("atom position: {line}"))?;
        }
        seen += 1;
    }
    if seen != atoms || positions.iter().any(|value| !value.is_finite()) {
        return Err(format!("{}: {seen} of {atoms} atoms read", path.display()));
    }
    Ok((lengths, positions))
}
