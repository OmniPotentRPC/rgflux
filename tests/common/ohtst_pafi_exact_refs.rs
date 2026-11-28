// Exact quadrature and Fokker-Planck references for PAFI and OH-TST tests.

#[allow(dead_code)]
mod ohtst_pafi_exact_refs {
    use ndarray::{Array1, ArrayView1, array};
    use rgsaddle::{PointSurface, SaddleError};

    /// Symmetric double well: `V = (1 - x²)²`.
    pub struct DoubleWell1D;

    impl DoubleWell1D {
        pub fn energy(x: f64) -> (f64, f64) {
            let well = 1.0 - x * x;
            let energy = well * well;
            let gradient = -4.0 * x * well;
            (energy, gradient)
        }

        pub fn hessian(x: f64) -> f64 {
            12.0 * x * x - 4.0
        }
    }

    impl PointSurface for DoubleWell1D {
        fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let (energy, gx) = Self::energy(q[0]);
            Ok((energy, array![gx]))
        }
    }

    /// Tilted curved well with a curved minimum-energy valley
    /// `y = a x + b x²`.
    pub struct CurvedWell2D;

    pub const CURVED_K: f64 = 6.0;
pub const CURVED_TILT: f64 = 0.1;
pub const CURVED_BEND: f64 = 0.8;

    impl CurvedWell2D {
        pub fn valley(x: f64) -> f64 {
            CURVED_TILT * x + CURVED_BEND * x * x * x
        }

        pub fn energy(x: f64, y: f64) -> (f64, f64, f64) {
            let well = x * x - 1.0;
            let centre = Self::valley(x);
            let offset = y - centre;
            let slope = CURVED_TILT + 3.0 * CURVED_BEND * x * x;
            let energy = well * well + 0.5 * CURVED_K * offset * offset;
            let gx = 4.0 * x * well - CURVED_K * offset * slope;
            let gy = CURVED_K * offset;
            (energy, gx, gy)
        }

        pub fn hessian(x: f64, y: f64) -> [[f64; 2]; 2] {
            let centre = Self::valley(x);
            let offset = y - centre;
            let slope = CURVED_TILT + 3.0 * CURVED_BEND * x * x;
            let hxx = 12.0 * x * x - 4.0 + CURVED_K * slope * slope - 6.0 * CURVED_K * CURVED_BEND * x * offset;
            let hxy = -CURVED_K * slope;
            [[hxx, hxy], [hxy, CURVED_K]]
        }
    }

    impl PointSurface for CurvedWell2D {
        fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let (energy, gx, gy) = Self::energy(q[0], q[1]);
            Ok((energy, array![gx, gy]))
        }
    }

    /// Müller-Brown 2D surface.
    pub struct MullerBrown2D;

    const MB_A: [f64; 4] = [-200.0, -100.0, -170.0, 15.0];
    const MB_AX: [f64; 4] = [-1.0, -1.0, -6.5, 0.7];
    const MB_B: [f64; 4] = [0.0, 0.0, 11.0, 0.6];
    const MB_AY: [f64; 4] = [-10.0, -10.0, -6.5, 0.7];
    const MB_X0: [f64; 4] = [1.0, 0.0, -0.5, -1.0];
    const MB_Y0: [f64; 4] = [0.0, 0.5, 1.5, 1.0];

    impl MullerBrown2D {
        pub fn energy(x: f64, y: f64) -> (f64, f64, f64) {
            let mut energy = 0.0;
            let mut gx = 0.0;
            let mut gy = 0.0;
            for i in 0..4 {
                let dx = x - MB_X0[i];
                let dy = y - MB_Y0[i];
                let exponent = MB_AX[i] * dx * dx + MB_B[i] * dx * dy + MB_AY[i] * dy * dy;
                let term = MB_A[i] * exponent.exp();
                energy += term;
                gx += term * (2.0 * MB_AX[i] * dx + MB_B[i] * dy);
                gy += term * (MB_B[i] * dx + 2.0 * MB_AY[i] * dy);
            }
            (energy, gx, gy)
        }

        pub fn hessian(at: [f64; 2]) -> [[f64; 2]; 2] {
            let mut h = [[0.0; 2]; 2];
            for i in 0..4 {
                let dx = at[0] - MB_X0[i];
                let dy = at[1] - MB_Y0[i];
                let exponent = MB_AX[i] * dx * dx + MB_B[i] * dx * dy + MB_AY[i] * dy * dy;
                let e = MB_A[i] * exponent.exp();
                let ux = 2.0 * MB_AX[i] * dx + MB_B[i] * dy;
                let uy = MB_B[i] * dx + 2.0 * MB_AY[i] * dy;
                h[0][0] += e * (ux * ux + 2.0 * MB_AX[i]);
                h[0][1] += e * (ux * uy + MB_B[i]);
                h[1][1] += e * (uy * uy + 2.0 * MB_AY[i]);
            }
            h[1][0] = h[0][1];
            h
        }
    }

    impl PointSurface for MullerBrown2D {
        fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let (energy, gx, gy) = Self::energy(q[0], q[1]);
            Ok((energy, array![gx, gy]))
        }
    }

    /// Adaptive Simpson's rule on `[a, b]` to the absolute tolerance
    /// `tol`. The tolerance must sit above the rounding noise of the
    /// integrand, `ε · |f|` times the length, or the recursion runs to
    /// its depth limit on every panel. Use [`simpson_rel`] when only the
    /// relative accuracy is known.
    pub fn simpson<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64, tol: f64) -> f64 {
        #[allow(clippy::too_many_arguments)]
        fn recurse<F: Fn(f64) -> f64>(
            f: &F,
            a: f64,
            b: f64,
            fa: f64,
            fm: f64,
            fb: f64,
            whole: f64,
            tol: f64,
            depth: usize,
        ) -> f64 {
            let m = 0.5 * (a + b);
            let lmid = 0.5 * (a + m);
            let rmid = 0.5 * (m + b);
            let flm = f(lmid);
            let frm = f(rmid);
            let left = (m - a) / 6.0 * (fa + 4.0 * flm + fm);
            let right = (b - m) / 6.0 * (fm + 4.0 * frm + fb);
            let delta = left + right - whole;
            if depth == 0 || delta.abs() <= 15.0 * tol {
                return left + right + delta / 15.0;
            }
            recurse(f, a, m, fa, flm, fm, left, 0.5 * tol, depth - 1)
                + recurse(f, m, b, fm, frm, fb, right, 0.5 * tol, depth - 1)
        }

        let m = 0.5 * (a + b);
        let fa = f(a);
        let fm = f(m);
        let fb = f(b);
        let whole = (b - a) / 6.0 * (fa + 4.0 * fm + fb);
        recurse(f, a, b, fa, fm, fb, whole, tol, 40)
    }

    /// [`simpson`] with `rel_tol` taken relative to the scale of the
    /// integral, the largest `|f|` on a 1025-point grid times `b - a`.
    pub fn simpson_rel<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64, rel_tol: f64) -> f64 {
        let grid = 1024usize;
        let peak = (0..=grid)
            .map(|i| f(a + (b - a) * i as f64 / grid as f64).abs())
            .fold(0.0, f64::max);
        simpson(f, a, b, rel_tol * peak * (b - a).abs())
    }

    /// Smallest value of `potential` on `[a, b]`: a grid of 4001 points,
    /// then golden-section search in the two cells around its lowest.
    pub fn basin_minimum(potential: fn(f64) -> f64, a: f64, b: f64) -> f64 {
        let grid = 4000usize;
        let step = (b - a) / grid as f64;
        let value = |i: usize| potential(a + step * i as f64);
        let lowest = (0..=grid).fold(0, |best, i| if value(i) < value(best) { i } else { best });
        let mut low = a + step * lowest.saturating_sub(1) as f64;
        let mut high = a + step * (lowest + 1).min(grid) as f64;
        let ratio = 0.5 * (5.0_f64.sqrt() - 1.0);
        for _ in 0..80 {
            let c = high - ratio * (high - low);
            let d = low + ratio * (high - low);
            if potential(c) < potential(d) {
                high = d;
            } else {
                low = c;
            }
        }
        potential(0.5 * (low + high)).min(value(lowest))
    }

    fn bump_integral<F: Fn(f64) -> f64>(f: &F, extent: f64, rel_tol: f64) -> f64 {
        bump_integral_on(f, -extent, extent, rel_tol)
    }

    fn bump_integral_on<F: Fn(f64) -> f64>(f: &F, start: f64, end: f64, rel_tol: f64) -> f64 {
        let grid = 4000usize;
        let step = (end - start) / grid as f64;
        let values: Vec<f64> = (0..=grid).map(|i| f(start + step * i as f64)).collect();
        let peak = values.iter().copied().fold(0.0, f64::max);
        let floor = peak * 1.0e-18;
        let first = values.iter().position(|v| *v > floor).unwrap_or(0);
        let last = values.iter().rposition(|v| *v > floor).unwrap_or(grid);
        let low = start + step * first.saturating_sub(1) as f64;
        let high = start + step * (last + 1).min(grid) as f64;
        let panels = 64usize;
        let width = (high - low) / panels as f64;
        let scale = peak * (high - low);
        (0..panels)
            .map(|k| {
                let a = low + width * k as f64;
                simpson(f, a, a + width, rel_tol * scale / panels as f64)
            })
            .sum()
    }

    pub fn unit(theta: f64) -> [f64; 2] {
        [theta.cos(), theta.sin()]
    }

    /// `∫ exp(-(V - shift)/kT) dt` along the line through `point` normal to
    /// `normal`.
    pub fn line_integral(
        energy: fn(f64, f64) -> (f64, f64, f64),
        point: [f64; 2],
        normal: [f64; 2],
        kt: f64,
        shift: f64,
        extent: f64,
    ) -> f64 {
        let norm = normal[0].hypot(normal[1]);
        let nx = normal[0] / norm;
        let ny = normal[1] / norm;
        let tx = -ny;
        let ty = nx;
        let density = |t: f64| {
            let (e, _, _) = energy(point[0] + t * tx, point[1] + t * ty);
            (-(e - shift) / kt).exp()
        };
        bump_integral(&density, extent, 1.0e-11)
    }

    /// The span `[low, high]` of the line `point + t m` that drains to the
    /// in-line minimum nearest `t = 0`: down to that minimum, then up to
    /// the first local maximum of `V` on each side, or to `±extent`. This
    /// is the part of the line a walk confined to the basin of the path
    /// can reach. A line a few units long crosses other, deeper wells of
    /// the Müller-Brown surface, which no walk on the plane of the path
    /// visits.
    pub fn line_basin(
        energy: fn(f64, f64) -> (f64, f64, f64),
        point: [f64; 2],
        normal: [f64; 2],
        extent: f64,
    ) -> (f64, f64) {
        let norm = normal[0].hypot(normal[1]);
        let tx = -normal[1] / norm;
        let ty = normal[0] / norm;
        let grid = 4000usize;
        let step = 2.0 * extent / grid as f64;
        let values: Vec<f64> = (0..=grid)
            .map(|i| {
                let t = -extent + step * i as f64;
                energy(point[0] + t * tx, point[1] + t * ty).0
            })
            .collect();
        let mut bottom = grid / 2;
        loop {
            let down_left = bottom > 0 && values[bottom - 1] < values[bottom];
            let down_right = bottom < grid && values[bottom + 1] < values[bottom];
            match (down_left, down_right) {
                (true, true) if values[bottom - 1] < values[bottom + 1] => bottom -= 1,
                (_, true) => bottom += 1,
                (true, false) => bottom -= 1,
                (false, false) => break,
            }
        }
        let mut low = bottom;
        while low > 0 && values[low - 1] >= values[low] {
            low -= 1;
        }
        let mut high = bottom;
        while high < grid && values[high + 1] >= values[high] {
            high += 1;
        }
        (-extent + step * low as f64, -extent + step * high as f64)
    }

    /// [`line_free_energy`] over [`line_basin`] only.
    pub fn basin_line_free_energy(
        energy: fn(f64, f64) -> (f64, f64, f64),
        point: [f64; 2],
        normal: [f64; 2],
        kt: f64,
        shift: f64,
        extent: f64,
    ) -> f64 {
        let (low, high) = line_basin(energy, point, normal, extent);
        let norm = normal[0].hypot(normal[1]);
        let tx = -normal[1] / norm;
        let ty = normal[0] / norm;
        let density = |t: f64| {
            let (e, _, _) = energy(point[0] + t * tx, point[1] + t * ty);
            (-(e - shift) / kt).exp()
        };
        -kt * bump_integral_on(&density, low, high, 1.0e-11).ln()
    }

    pub fn line_free_energy(
        energy: fn(f64, f64) -> (f64, f64, f64),
        point: [f64; 2],
        normal: [f64; 2],
        kt: f64,
        shift: f64,
        extent: f64,
    ) -> f64 {
        -kt * line_integral(energy, point, normal, kt, shift, extent).ln()
    }

    /// `∫_{ξ<=0} exp(-(V-shift)/kT)` behind a line through `point` with
    /// normal `normal`.
    pub fn half_plane_integral(
        energy: fn(f64, f64) -> (f64, f64, f64),
        point: [f64; 2],
        normal: [f64; 2],
        kt: f64,
        shift: f64,
        extent: f64,
    ) -> f64 {
        let norm = normal[0].hypot(normal[1]);
        let nx = normal[0] / norm;
        let ny = normal[1] / norm;
        let line = |xi: f64| {
            line_integral(
                energy,
                [point[0] + xi * nx, point[1] + xi * ny],
                [nx, ny],
                kt,
                shift,
                extent,
            )
        };

        let grid = 400usize;
        let step = extent / grid as f64;
        let values: Vec<f64> = (0..=grid).map(|i| line(-step * i as f64)).collect();
        let peak = values.iter().copied().fold(0.0, f64::max);
        let deepest = values
            .iter()
            .rposition(|v| *v > peak * 1.0e-18)
            .unwrap_or(grid);
        let low = -step * (deepest + 1).min(grid) as f64;
        let panels = 32usize;
        let width = -low / panels as f64;
        let scale = peak * (-low);
        (0..panels)
            .map(|k| {
                let a = low + width * k as f64;
                simpson(&line, a, a + width, 1.0e-9 * scale / panels as f64)
            })
            .sum()
    }

    pub fn line_log_tst_rate(
        energy: fn(f64, f64) -> (f64, f64, f64),
        point: [f64; 2],
        theta: f64,
        kt: f64,
        shift: f64,
        extent: f64,
    ) -> f64 {
        let normal = unit(theta);
        let z_plane = line_integral(energy, point, normal, kt, shift, extent);
        let z_reactant = half_plane_integral(energy, point, normal, kt, shift, extent);
        0.5 * (kt / (2.0 * std::f64::consts::PI)).ln() + z_plane.ln() - z_reactant.ln()
    }

    pub fn optimize_line(
        energy: fn(f64, f64) -> (f64, f64, f64),
        saddle: [f64; 2],
        theta0: f64,
        kt: f64,
        shift: f64,
        extent: f64,
    ) -> (f64, f64, [[f64; 2]; 2]) {
        let free = |offset: f64, theta: f64| {
            let n = unit(theta);
            let point = [saddle[0] + offset * n[0], saddle[1] + offset * n[1]];
            line_free_energy(energy, point, n, kt, shift, extent) / kt
        };
        let mut offset = 0.0;
        let mut theta = theta0;
        let mut curvature = [[0.0; 2]; 2];
        let h = 1.0e-3;
        for _ in 0..8 {
            let f0 = free(offset, theta);
            let fdp = free(offset + h, theta);
            let fdm = free(offset - h, theta);
            let ftp = free(offset, theta + h);
            let ftm = free(offset, theta - h);
            let cross =
                (free(offset + h, theta + h) - free(offset + h, theta - h) - free(offset - h, theta + h)
                    + free(offset - h, theta - h))
                    / (4.0 * h * h);
            let gd = (fdp - fdm) / (2.0 * h);
            let gt = (ftp - ftm) / (2.0 * h);
            let hdd = (fdp - 2.0 * f0 + fdm) / (h * h);
            let htt = (ftp - 2.0 * f0 + ftm) / (h * h);
            let det = hdd * htt - cross * cross;
            offset -= (htt * gd - cross * gt) / det;
            theta -= (hdd * gt - cross * gd) / det;
            curvature = [[hdd, cross], [cross, htt]];
        }
        (offset, theta, curvature)
    }

    pub fn eigen2(h: [[f64; 2]; 2]) -> (f64, f64, [f64; 2]) {
        let mean = 0.5 * (h[0][0] + h[1][1]);
        let half = 0.5 * (h[0][0] - h[1][1]);
        let radius = half.hypot(h[0][1]);
        let low = mean - radius;
        let high = mean + radius;
        let vector = if h[0][1].abs() > 1.0e-300 {
            [h[0][1], low - h[0][0]]
        } else if h[0][0] <= h[1][1] {
            [1.0, 0.0]
        } else {
            [0.0, 1.0]
        };
        let norm = vector[0].hypot(vector[1]);
        (low, high, [vector[0] / norm, vector[1] / norm])
    }

    pub fn harmonic_log_rate_2d(
        reactant_hessian: [[f64; 2]; 2],
        saddle_hessian: [[f64; 2]; 2],
        barrier: f64,
        kt: f64,
    ) -> f64 {
        let (r1, r2, _) = eigen2(reactant_hessian);
        let (s1, s2, _) = eigen2(saddle_hessian);
        let stable = if s1 > 0.0 { s1 } else { s2 };
        ((r1 * r2).sqrt() / (2.0 * std::f64::consts::PI * stable.sqrt())).ln() - barrier / kt
    }

    /// Steepest-descent minimizer in 2D.
    pub fn descend(energy: fn(f64, f64) -> (f64, f64, f64), start: [f64; 2]) -> [f64; 2] {
        let mut point = start;
        let mut step = 1.0e-2;
        for _ in 0..200_000 {
            let (e, gx, gy) = energy(point[0], point[1]);
            if gx.hypot(gy) < 1.0e-12 {
                break;
            }
            loop {
                let trial = [point[0] - step * gx, point[1] - step * gy];
                if energy(trial[0], trial[1]).0 < e {
                    point = trial;
                    step *= 1.2;
                    break;
                }
                step *= 0.5;
                if step < 1.0e-16 {
                    return point;
                }
            }
        }
        point
    }

    /// Newton solve of `∇V = 0` from a start.
    pub fn stationary(energy: fn(f64, f64) -> (f64, f64, f64), start: [f64; 2]) -> [f64; 2] {
        let mut point = start;
        let h = 1.0e-6;
        for _ in 0..100 {
            let (_, gx, gy) = energy(point[0], point[1]);
            if gx.hypot(gy) < 1.0e-13 {
                break;
            }
            let (_, axp, ayp) = energy(point[0] + h, point[1]);
            let (_, axm, aym) = energy(point[0] - h, point[1]);
            let (_, bxp, byp) = energy(point[0], point[1] + h);
            let (_, bxm, bym) = energy(point[0], point[1] - h);
            let hxx = (axp - axm) / (2.0 * h);
            let hyx = (ayp - aym) / (2.0 * h);
            let hxy = (bxp - bxm) / (2.0 * h);
            let hyy = (byp - bym) / (2.0 * h);
            let det = hxx * hyy - hxy * hyx;
            point[0] -= (hyy * gx - hxy * gy) / det;
            point[1] -= (hxx * gy - hyx * gx) / det;
        }
        point
    }

    /// Points along steepest descent from a saddle launch direction.
    pub fn descent_path(
        energy: fn(f64, f64) -> (f64, f64, f64),
        start: [f64; 2],
        direction: [f64; 2],
        spacing: f64,
    ) -> Vec<[f64; 2]> {
        let flow = |p: [f64; 2]| {
            let (_, gx, gy) = energy(p[0], p[1]);
            let g = gx.hypot(gy);
            [-gx / g, -gy / g]
        };

        let h = spacing / 50.0;
        let mut points = vec![start];
        let mut p = [start[0] + h * direction[0], start[1] + h * direction[1]];
        let mut travelled = h;
        let mut previous = energy(p[0], p[1]).0;
        for _ in 0..1_000_000 {
            let k1 = flow(p);
            let k2 = flow([p[0] + 0.5 * h * k1[0], p[1] + 0.5 * h * k1[1]]);
            let k3 = flow([p[0] + 0.5 * h * k2[0], p[1] + 0.5 * h * k2[1]]);
            let k4 = flow([p[0] + h * k3[0], p[1] + h * k3[1]]);
            let next = [
                p[0] + h / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
                p[1] + h / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
            ];
            let e = energy(next[0], next[1]).0;
            let moved = (next[0] - p[0]).hypot(next[1] - p[1]);
            if e >= previous || moved < 0.25 * h {
                break;
            }
            previous = e;
            p = next;
            travelled += moved;
            if travelled >= spacing * points.len() as f64 {
                points.push(p);
            }
        }
        let minimum = descend(energy, p);
        let last = points[points.len() - 1];
        if (minimum[0] - last[0]).hypot(minimum[1] - last[1]) < 0.5 * spacing {
            points.pop();
        }
        points.push(minimum);
        points
    }

    /// Profile derivatives by one-sided/central finite differences.
    pub fn profile_gradient(arcs: &[f64], free_energy: &[f64]) -> Vec<f64> {
        let n = arcs.len();
        let mut derivative = vec![0.0; n];
        for i in 0..n {
            derivative[i] = if i == 0 {
                (free_energy[1] - free_energy[0]) / (arcs[1] - arcs[0])
            } else if i + 1 == n {
                (free_energy[n - 1] - free_energy[n - 2]) / (arcs[n - 1] - arcs[n - 2])
            } else {
                (free_energy[i + 1] - free_energy[i - 1]) / (arcs[i + 1] - arcs[i - 1])
            };
        }
        derivative
    }

    /// 1D TST rate with dividing point `x_div` and reactant region
    /// `x < x_div`. The Boltzmann factors are shifted by the lowest
    /// potential in the reactant region, so they run from 1 down to the
    /// barrier factor and the quadrature tolerance is relative to that.
    pub fn log_tst_rate_1d(
        potential: fn(f64) -> f64,
        kt: f64,
        x_div: f64,
        left_extent: f64,
    ) -> f64 {
        let shift = basin_minimum(potential, left_extent, x_div);
        let z_reactant = simpson_rel(
            &|x| (-(potential(x) - shift) / kt).exp(),
            left_extent,
            x_div,
            1.0e-10,
        );
        let z_plane = (-(potential(x_div) - shift) / kt).exp();
        0.5 * (kt / (2.0 * std::f64::consts::PI)).ln() + z_plane.ln() - z_reactant.ln()
    }

    /// Overdamped MFPT rate `1/τ` from `x_start` to absorbing `x_absorb`
    /// with reflecting boundary at `x_reflect` for
    /// `dx = -μ V'(x) dt + sqrt(2 μ kT) dW`.
    pub fn overdamped_mfpt_rate_1d(
        potential: fn(f64) -> f64,
        kt: f64,
        mobility: f64,
        x_reflect: f64,
        x_start: f64,
        x_absorb: f64,
    ) -> f64 {
        let points = 24_000usize;
        let step = (x_absorb - x_reflect) / points as f64;
        let shift = basin_minimum(potential, x_reflect, x_absorb);

        let mut coordinates = Vec::with_capacity(points + 1);
        let mut cumulative = vec![0.0; points + 1];
        for i in 0..=points {
            coordinates.push(x_reflect + step * i as f64);
        }
        let weight_minus = |x: f64| (-(potential(x) - shift) / kt).exp();
        let weight_plus = |x: f64| ((potential(x) - shift) / kt).exp();
        for i in 1..=points {
            cumulative[i] = cumulative[i - 1]
                + 0.5 * step * (weight_minus(coordinates[i - 1]) + weight_minus(coordinates[i]));
        }

        let inner = |x: f64| {
            let t = ((x - x_reflect) / step).clamp(0.0, points as f64 - 1.0);
            let i = t.floor() as usize;
            let frac = t - i as f64;
            cumulative[i] + frac * (cumulative[i + 1] - cumulative[i])
        };
        let outer = |x: f64| weight_plus(x) * inner(x);
        let tau = simpson_rel(&outer, x_start, x_absorb, 1.0e-10) / (mobility * kt);
        1.0 / tau
    }

    /// Exact independent draws from `exp(-V/kT)` on a line by inverting the
    /// trapezoid cumulative.
    pub fn line_draws(
        energy: fn(f64, f64) -> (f64, f64, f64),
        point: [f64; 2],
        normal: [f64; 2],
        kt: f64,
        extent: f64,
        count: usize,
        seed: u64,
    ) -> Vec<Array1<f64>> {
        let norm = normal[0].hypot(normal[1]);
        let tx = -normal[1] / norm;
        let ty = normal[0] / norm;
        let grid = 20_000usize;
        let step = 2.0 * extent / grid as f64;
        let energies: Vec<f64> = (0..=grid)
            .map(|i| {
                let s = -extent + step * i as f64;
                energy(point[0] + s * tx, point[1] + s * ty).0
            })
            .collect();
        let minimum = energies.iter().copied().fold(f64::INFINITY, f64::min);
        let mut cumulative = vec![0.0; grid + 1];
        for i in 1..=grid {
            let a = energies[i - 1] - minimum;
            let b = energies[i] - minimum;
            cumulative[i] = cumulative[i - 1] + 0.5 * step * ((-a / kt).exp() + (-b / kt).exp());
        }
        let total = cumulative[grid];
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                let u = (((z ^ (z >> 31)) >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
                let target = u * total;
                let i = cumulative.partition_point(|c| *c < target).clamp(1, grid);
                let within = (target - cumulative[i - 1]) / (cumulative[i] - cumulative[i - 1]);
                let s = -extent + step * ((i - 1) as f64 + within);
                array![point[0] + s * tx, point[1] + s * ty]
            })
            .collect()
    }
}
