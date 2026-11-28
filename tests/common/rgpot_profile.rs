use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rgpot_core::profile::{ProfileRequest, ProfileSession};
use rgsaddle::{BandSurface, PointSurface, SaddleError};

thread_local! {
    static SESSIONS: RefCell<HashMap<u64, ProfileSession>> = RefCell::new(HashMap::new());
}

static NEXT_SESSION: AtomicU64 = AtomicU64::new(1);

/// A persistent, thread-local rgpot profile session exposed as a
/// [`PointSurface`].
///
/// `ProfileSession` intentionally does not claim `Send`: native backends
/// may hold thread-affine state. Each rayon worker therefore opens its own
/// session on first use while repeated force calls on that worker reuse it.
pub struct RgpotProfileSurface {
    id: u64,
    prefix: String,
    library: PathBuf,
    config: Vec<u8>,
    atomic_numbers: Vec<i32>,
    box_matrix: [f64; 9],
}

impl RgpotProfileSurface {
    pub fn new(
        prefix: impl Into<String>,
        library: impl Into<PathBuf>,
        config: Vec<u8>,
        atomic_numbers: Vec<i32>,
        box_matrix: [f64; 9],
    ) -> Self {
        Self {
            id: NEXT_SESSION.fetch_add(1, Ordering::Relaxed),
            prefix: prefix.into(),
            library: library.into(),
            config,
            atomic_numbers,
            box_matrix,
        }
    }

    fn evaluate(
        &self,
        positions: &[f64],
    ) -> Result<rgpot_core::profile::ProfileEvaluation, SaddleError> {
        SESSIONS.with(|sessions| {
            let mut sessions = sessions.borrow_mut();
            let session = match sessions.entry(self.id) {
                Entry::Occupied(slot) => slot.into_mut(),
                Entry::Vacant(slot) => {
                    let session = unsafe {
                        ProfileSession::load(
                            &self.prefix,
                            Some(Path::new(&self.library)),
                            &self.config,
                        )
                    }
                    .map_err(|error| {
                        SaddleError::Solver(format!("rgpot {} profile: {error}", self.prefix))
                    })?;
                    slot.insert(session)
                }
            };
            session
                .evaluate(&ProfileRequest {
                    positions,
                    atomic_numbers: &self.atomic_numbers,
                    box_matrix: Some(&self.box_matrix),
                    length_unit: "angstrom",
                    energy_unit: "eV",
                })
                .map_err(|error| {
                    SaddleError::Solver(format!("rgpot {} evaluation: {error}", self.prefix))
                })
        })
    }
}

impl PointSurface for RgpotProfileSurface {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let positions = x.as_slice().ok_or_else(|| {
            SaddleError::Shape("rgpot profile positions must be contiguous".into())
        })?;
        let result = self.evaluate(positions)?;
        if result.forces.len() != positions.len() {
            return Err(SaddleError::Shape(format!(
                "rgpot returned {} force components for {} coordinates",
                result.forces.len(),
                positions.len()
            )));
        }
        Ok((
            result.energy,
            Array1::from_iter(result.forces.into_iter().map(|force| -force)),
        ))
    }
}

impl BandSurface for RgpotProfileSurface {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for row in 0..positions.nrows() {
            let (energy, gradient) = PointSurface::eval(self, positions.row(row))?;
            energies[row] = energy;
            gradients.row_mut(row).assign(&gradient);
        }
        Ok(())
    }
}
