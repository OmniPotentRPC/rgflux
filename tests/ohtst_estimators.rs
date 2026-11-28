//! The sampled force and torque on a hyperplane against finite differences
//! of its free energy: a closed form on a Gaussian, and quadrature of the
//! line free energy on the LEPS plus harmonic oscillator surface.

use ndarray::{Array1, Array2, ArrayView1, array, s};
use rgflux::ohtst::{HarmonicReference, Hyperplane, LangevinConfig, sample_plane};
use rgsaddle::{PointSurface, SaddleError};

include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/common/ohtst_surfaces.rs"
));

use ohtst_surfaces::{LepsHo, eigen2, hessian, line_free_energy, stationary};

/// `V = ½ qᵀ K q` with `K = Uᵀ diag(λ) U`.
struct Gaussian {
    k: Array2<f64>,
    k_inv: Array2<f64>,
}

impl Gaussian {
    fn new() -> Self {
        let rotate = |a: usize, b: usize, angle: f64| {
            let mut g = Array2::eye(3);
            g[(a, a)] = angle.cos();
            g[(b, b)] = angle.cos();
            g[(a, b)] = -angle.sin();
            g[(b, a)] = angle.sin();
            g
        };
        let u = rotate(0, 1, 0.4)
            .dot(&rotate(1, 2, -0.7))
            .dot(&rotate(0, 2, 0.25));
        let lambda = [2.0, 0.6, 1.3];
        let mut k = Array2::zeros((3, 3));
        let mut k_inv = Array2::zeros((3, 3));
        for i in 0..3 {
            for j in 0..3 {
                for m in 0..3 {
                    k[(i, j)] += u[(m, i)] * lambda[m] * u[(m, j)];
                    k_inv[(i, j)] += u[(m, i)] * u[(m, j)] / lambda[m];
                }
            }
        }
        Self { k, k_inv }
    }

    /// `A(p, ν) = (kT/2) ln(νᵀK⁻¹ν) + (ν · p)² / (2 νᵀK⁻¹ν)` up to a
    /// constant: the marginal of `ν · q` is Gaussian with variance
    /// `kT νᵀK⁻¹ν`.
    fn free_energy(&self, p: &Array1<f64>, nu: &Array1<f64>, kt: f64) -> f64 {
        let nu = nu / nu.dot(nu).sqrt();
        let s = nu.dot(&self.k_inv.dot(&nu));
        let a = nu.dot(p);
        0.5 * kt * s.ln() + a * a / (2.0 * s)
    }
}

impl PointSurface for Gaussian {
    fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let g = self.k.dot(&q);
        Ok((0.5 * q.dot(&g), g))
    }
}

/// Central difference of `A` along the packed direction `[δp; δν]`, the
/// normal renormalized.
fn packed_difference<F: Fn(&Array1<f64>, &Array1<f64>) -> f64>(
    free_energy: F,
    plane: &Hyperplane,
    direction: &Array1<f64>,
    h: f64,
) -> f64 {
    let n = plane.dim();
    let p = plane.point().to_owned();
    let nu = plane.normal().to_owned();
    let dp = direction.slice(s![..n]).to_owned();
    let dn = direction.slice(s![n..]).to_owned();
    let plus = free_energy(&(&p + &(&dp * h)), &(&nu + &(&dn * h)));
    let minus = free_energy(&(&p - &(&dp * h)), &(&nu - &(&dn * h)));
    (plus - minus) / (2.0 * h)
}

fn packed(dp: &Array1<f64>, dn: &Array1<f64>) -> Array1<f64> {
    let mut v = Array1::zeros(dp.len() + dn.len());
    v.slice_mut(s![..dp.len()]).assign(dp);
    v.slice_mut(s![dp.len()..]).assign(dn);
    v
}

/// Mean and standard error of `row · direction` over block rows.
fn blocked(rows: &Array2<f64>, direction: &Array1<f64>) -> (f64, f64) {
    let values: Vec<f64> = rows.rows().into_iter().map(|r| r.dot(direction)).collect();
    let m = values.len() as f64;
    let mean = values.iter().sum::<f64>() / m;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (m - 1.0);
    (mean, (variance / m).sqrt())
}

/// Unit vectors `w₁, w₂` completing `ν` to an orthonormal frame.
fn frame(nu: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
    let v = array![0.0, 1.0, 0.0];
    let v = &v - &(nu * nu.dot(&v));
    let w1 = &v / v.dot(&v).sqrt();
    let w2 = array![
        nu[1] * w1[2] - nu[2] * w1[1],
        nu[2] * w1[0] - nu[0] * w1[2],
        nu[0] * w1[1] - nu[1] * w1[0]
    ];
    (w1, w2)
}

#[test]
fn gaussian_force_and_torque_match_the_closed_form() {
    let surface = Gaussian::new();
    let kt = 0.5;
    let config = LangevinConfig::default()
        .with_thermal(kt)
        .with_time_step(0.4)
        .with_friction(0.5)
        .with_burn_in(200)
        .with_stride(4)
        .with_records(6000)
        .with_seed(11);
    let plane = Hyperplane::new(array![0.4, -0.2, 0.1], array![1.0, 0.5, -0.3]).unwrap();
    let sample = sample_plane(&surface, &plane, None, &config).unwrap();
    let nu = plane.normal().to_owned();
    let (w1, w2) = frame(&nu);
    let zero = Array1::zeros(3);
    let directions = [
        packed(&nu, &zero),
        packed(&zero, &w1),
        packed(&zero, &w2),
        packed(&(&w1 * 0.7 + &nu * 0.3), &(&w2 * 0.5)),
    ];
    for direction in &directions {
        let exact = packed_difference(
            |p, n| surface.free_energy(p, n, kt),
            &plane,
            direction,
            1.0e-4,
        );
        let (value, error) = sample.directional(direction.view()).unwrap();
        assert!(
            (value - exact).abs() < 4.0 * error + 1.0e-7,
            "derivative {value} ± {error}, closed form {exact}"
        );
        assert!(error < 0.03 * (1.0 + exact.abs()));
    }
    let slide = packed(&w1, &zero);
    let (value, error) = sample.directional(slide.view()).unwrap();
    assert!(value.abs() < 1.0e-12 && error < 1.0e-12);
}

#[test]
fn leps_force_and_torque_match_line_quadrature() {
    let saddle = stationary(LepsHo::energy, [2.02, -0.17]);
    let minimum = ohtst_surfaces::descend(LepsHo::energy, [0.75, 1.3]);
    let barrier = LepsHo::energy(saddle[0], saddle[1]).0 - LepsHo::energy(minimum[0], minimum[1]).0;
    let kt = 0.1 * barrier;
    let (_, _, mode) = eigen2(hessian(LepsHo::energy, saddle));
    let base = mode[1].atan2(mode[0]);
    let config = LangevinConfig::default()
        .with_thermal(kt)
        .with_time_step(0.15)
        .with_friction(1.0)
        .with_burn_in(500)
        .with_stride(5)
        .with_records(20000)
        .with_seed(3);
    for (offset, tilt) in [(0.1, 0.25), (-0.15, -0.1)] {
        let angle = base + tilt;
        let normal = [angle.cos(), angle.sin()];
        let point = [
            saddle[0] + offset * normal[0],
            saddle[1] + offset * normal[1],
        ];
        let plane =
            Hyperplane::new(array![point[0], point[1]], array![normal[0], normal[1]]).unwrap();
        let sample = sample_plane(&LepsHo, &plane, None, &config).unwrap();
        let free_energy = |p: &Array1<f64>, n: &Array1<f64>| {
            line_free_energy(LepsHo::energy, [p[0], p[1]], [n[0], n[1]], kt, 0.0, 6.0)
        };
        let nu = array![normal[0], normal[1]];
        let turn = array![-normal[1], normal[0]];
        let zero = Array1::zeros(2);
        for direction in [packed(&nu, &zero), packed(&zero, &turn)] {
            let exact = packed_difference(free_energy, &plane, &direction, 1.0e-4);
            let (value, error) = sample.directional(direction.view()).unwrap();
            assert!(
                (value - exact).abs() < 4.0 * error + 1.0e-6,
                "derivative {value} ± {error}, quadrature {exact}"
            );
            assert!(error < 0.05 * (0.1 + exact.abs()));
        }
    }
}

#[test]
fn gaussian_centroid_torque_matches_the_closed_form() {
    // The plane ν · q = σ cuts the Gaussian at the mean c = σ K⁻¹ν / νᵀK⁻¹ν,
    // and turning about c is the closed form with c held.
    let surface = Gaussian::new();
    let kt = 0.5;
    let config = LangevinConfig::default()
        .with_thermal(kt)
        .with_time_step(0.4)
        .with_friction(0.5)
        .with_burn_in(200)
        .with_stride(4)
        .with_records(6000)
        .with_seed(1);
    let normal: Array1<f64> = array![1.0, 0.5, -0.3];
    let nu = &normal / normal.dot(&normal).sqrt();
    let (w1, w2) = frame(&nu);
    let sigma = 0.35;
    let k_inv_nu = surface.k_inv.dot(&nu);
    let mean = &k_inv_nu * (sigma / nu.dot(&k_inv_nu));
    let about_mean = Hyperplane::new(mean.clone(), nu.clone()).unwrap();
    let far = Hyperplane::new(&mean + &(&w1 * 6.0) - &(&w2 * 4.0), nu.clone()).unwrap();
    let sample = sample_plane(&surface, &far, None, &config).unwrap();
    let zero = Array1::zeros(3);
    for w in [&w1, &w2] {
        let direction = packed(&zero, w);
        let exact = packed_difference(
            |p, n| surface.free_energy(p, n, kt),
            &about_mean,
            &direction,
            1.0e-4,
        );
        let (value, error) = blocked(&sample.centroid_block_gradients, &direction);
        assert!((value - sample.centroid_torque.dot(w)).abs() < 1e-12);
        assert!(
            (value - exact).abs() < 4.0 * error + 1.0e-7,
            "centroid torque {value} ± {error}, closed form {exact}"
        );
        // Over seeds 1 to 12 the ratio stayed within 0.08 to 0.41.
        let (_, pivot_error) = sample.directional(direction.view()).unwrap();
        assert!(
            error < 0.6 * pivot_error,
            "centroid error {error} against {pivot_error} about the far point"
        );
    }
    assert!(
        (sample.virial - 2.0 * kt).abs() < 4.0 * sample.virial_error,
        "virial {} ± {}",
        sample.virial,
        sample.virial_error
    );
}

/// With the reference Hessian equal to `K`, the control variates leave no
/// variance in the normal force, and on a plane through its in-plane mean
/// none in the centroid torque. The values are the closed forms
/// `⟨ν · ∇V⟩ = s / S` and `-Cov(ν · ∇V, q)_⊥ = (kT / S) P K⁻¹ ν`, with
/// `S = νᵀK⁻¹ν` and the plane at `ν · q = s`.
#[test]
fn control_variates_are_exact_on_a_gaussian() {
    let model = Gaussian::new();
    let kt = 0.4;
    let nu: Array1<f64> = array![0.6, -0.3, 0.74];
    let nu = &nu / nu.dot(&nu).sqrt();
    let k_inv_nu = model.k_inv.dot(&nu);
    let spread = nu.dot(&k_inv_nu);
    let offset = 0.3;
    let plane = Hyperplane::new(&k_inv_nu * (offset / spread), nu.clone()).unwrap();
    let reference = HarmonicReference::new(Array1::zeros(3)).with_tolerance(1.0e-12);
    let config = LangevinConfig::default()
        .with_thermal(kt)
        .with_time_step(0.1)
        .with_records(2000)
        .with_seed(8)
        .with_control(reference);
    let sample = sample_plane(&model, &plane, None, &config).unwrap();
    let raw = sample.uncontrolled.as_ref().unwrap();
    let force = -offset / spread;
    assert!((sample.normal_force - force).abs() < 1.0e-8);
    assert!(sample.normal_force_error < 1.0e-9);
    assert!((raw.normal_force - force).abs() < 4.0 * raw.normal_force_error);
    assert!(raw.normal_force_error > 1.0e-3);
    let mut torque = &k_inv_nu * (kt / spread);
    let along = torque.dot(&nu);
    torque.scaled_add(-along, &nu);
    for i in 0..3 {
        let (value, error) = (sample.centroid_torque[i], sample.centroid_torque_error[i]);
        assert!(
            (value - torque[i]).abs() < 1.0e-8,
            "{value} against {}",
            torque[i]
        );
        assert!(error < 1.0e-9, "{error}");
        let (value, error) = (raw.centroid_torque[i], raw.centroid_torque_error[i]);
        assert!((value - torque[i]).abs() < 4.0 * error + 1.0e-12);
    }
}

/// On the LEPS surface, control variates from the saddle Hessian stay
/// unbiased against line quadrature. They do not pay here: on these two
/// planes they changed the force error by factors of 1.27 and 0.85 and the
/// torque error by 1.74 and 1.18, since the in-plane fluctuations reach
/// well past the harmonic region of the saddle.
#[test]
fn control_variates_stay_unbiased_on_leps() {
    let saddle = stationary(LepsHo::energy, [2.02, -0.17]);
    let minimum = ohtst_surfaces::descend(LepsHo::energy, [0.75, 1.3]);
    let barrier = LepsHo::energy(saddle[0], saddle[1]).0 - LepsHo::energy(minimum[0], minimum[1]).0;
    let kt = 0.1 * barrier;
    let (_, _, mode) = eigen2(hessian(LepsHo::energy, saddle));
    let base = mode[1].atan2(mode[0]);
    let config = LangevinConfig::default()
        .with_thermal(kt)
        .with_time_step(0.1)
        .with_friction(1.0)
        .with_burn_in(500)
        .with_stride(5)
        .with_records(8000)
        .with_seed(3)
        .with_control(HarmonicReference::new(array![saddle[0], saddle[1]]));
    for (offset, tilt) in [(0.1, 0.25), (-0.15, -0.1)] {
        let angle = base + tilt;
        let normal = [angle.cos(), angle.sin()];
        let point = [
            saddle[0] + offset * normal[0],
            saddle[1] + offset * normal[1],
        ];
        let plane =
            Hyperplane::new(array![point[0], point[1]], array![normal[0], normal[1]]).unwrap();
        let sample = sample_plane(&LepsHo, &plane, None, &config).unwrap();
        let raw = sample.uncontrolled.as_ref().unwrap();
        let free_energy = |p: &Array1<f64>, n: &Array1<f64>| {
            line_free_energy(LepsHo::energy, [p[0], p[1]], [n[0], n[1]], kt, 0.0, 6.0)
        };
        let nu = array![normal[0], normal[1]];
        let turn = array![-normal[1], normal[0]];
        let zero = Array1::zeros(2);
        for direction in [packed(&nu, &zero), packed(&zero, &turn)] {
            let exact = packed_difference(free_energy, &plane, &direction, 1.0e-4);
            let (value, error) = sample.directional(direction.view()).unwrap();
            assert!(
                (value - exact).abs() < 4.0 * error + 1.0e-6,
                "derivative {value} ± {error}, quadrature {exact}"
            );
        }
        assert!(
            (sample.normal_force - raw.normal_force).abs()
                < 4.0 * sample.normal_force_error.max(raw.normal_force_error)
        );
    }
}
