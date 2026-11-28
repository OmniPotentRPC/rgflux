//! Plane sampling against the mass-weighted surfaces of rgsaddle.

use ndarray::{Array1, Array2, ArrayView1, array};
use rgsaddle::{Hyperplane, PointSurface, SaddleError, mass_weighted_rigid_translations};

use crate::ohtst::sample::{LangevinConfig, sample_plane};

fn center_of_mass(position: ArrayView1<'_, f64>, masses: &[f64]) -> [f64; 3] {
    let mut com = [0.0; 3];
    let total_mass = masses.iter().sum::<f64>();
    for (atom, mass) in masses.iter().enumerate() {
        let weight = mass.sqrt();
        let base = 3 * atom;
        com[0] += weight * position[base];
        com[1] += weight * position[base + 1];
        com[2] += weight * position[base + 2];
    }
    for value in &mut com {
        *value /= total_mass;
    }
    com
}

fn rigid_translation_axis(normal: ArrayView1<'_, f64>, masses: &[f64]) -> [f64; 3] {
    let mut axis = [0.0; 3];
    for (atom, mass) in masses.iter().enumerate() {
        let weight = mass.sqrt();
        let base = 3 * atom;
        axis[0] += weight * normal[base];
        axis[1] += weight * normal[base + 1];
        axis[2] += weight * normal[base + 2];
    }
    let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    if norm > 0.0 {
        axis[0] /= norm;
        axis[1] /= norm;
        axis[2] /= norm;
    }
    axis
}

struct TranslationInvariant {
    translations: Array2<f64>,
}

impl PointSurface for TranslationInvariant {
    fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        Ok((0.0, Array1::zeros(q.len())))
    }

    fn excluded_modes(&self, _q: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        Ok(self.translations.clone())
    }
}

#[test]
fn fixed_plane_sampling_has_no_com_drift_along_the_normal() {
    let masses = [1.0, 3.0];
    let translations = mass_weighted_rigid_translations(&masses, masses.len()).unwrap();
    let surface = TranslationInvariant {
        translations: translations.clone(),
    };
    let point = array![0.1, -0.2, 0.3, 0.6, 0.5, -0.4];
    let raw_normal = array![0.7, 0.3, -0.2, 0.4, 0.1, 0.8];
    let axis = rigid_translation_axis(raw_normal.view(), &masses);
    let axis_norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    assert!(axis_norm > 0.0);
    let plane = Hyperplane::new(point.clone(), raw_normal)
        .unwrap()
        .orthogonal_to_mass_weighted_translations(&masses, masses.len())
        .unwrap();
    let config = LangevinConfig::default()
        .with_thermal(0.02)
        .with_time_step(0.3)
        .with_records(200)
        .with_blocks(4)
        .with_seed(11)
        .with_retain(4);
    let sample = sample_plane(&surface, &plane, None, &config).unwrap();
    let com0 = center_of_mass(point.view(), &masses);
    for q in sample.retained.iter().chain(std::iter::once(&sample.last)) {
        let com = center_of_mass(q.view(), &masses);
        let drift = (com[0] - com0[0]) * axis[0]
            + (com[1] - com0[1]) * axis[1]
            + (com[2] - com0[2]) * axis[2];
        assert!(drift.abs() < 1.0e-12, "COM drift along normal is {drift}");
    }
}
