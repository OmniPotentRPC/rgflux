//! Closed-form checks of the fixed-normal OH-TST rate and its standard error.
//!
//! The scripted engine in `ohtst_support` supplies an analytic double well and
//! a constraint-force script, so every quantity entering
//! `k = sqrt(κ / μ) / 2π * exp(-ΔV / kT)` is known and the sampler adds no
//! noise. The well is `V(u) = (u² - a²)² / (4 a²)`: minima at `u = ±a`, a
//! barrier of `a² / 4` at `u = 0` and `V''(±a) = 2`. The constraint force is the
//! harmonic mean force `⟨λ⟩ = κ (c - s_min)`, which takes the sign of `c - s_min`.

use rgflux::FluxError;
use rgflux::ohtst::OhtstConfig;
use rgflux::stats::StatsConfig;

mod ohtst_support;

use ohtst_support::*;

#[test]
fn rate_matches_harmonic_tst_on_both_sides_of_the_minimum() {
    let expected = harmonic_rate(CURVATURE, 1.0, BARRIER, THERMAL);
    // (start, normal component, offset the probe plane must take).
    let cases = [
        (-HALF_WIDTH, 1.0, -HALF_WIDTH + 0.25),
        (HALF_WIDTH, 1.0, HALF_WIDTH - 0.25),
        (-HALF_WIDTH, -1.0, HALF_WIDTH - 0.25),
    ];
    for (start, sign, probe_offset) in cases {
        let mut engine = Scripted::new(
            vec![start],
            vec![1.0],
            well(HALF_WIDTH),
            harmonic_force(CURVATURE, sign * start),
        );
        let seed = plane(&[sign], 0.0);
        let found = estimate(&mut engine, &seed, &quiet());
        assert_close(found.rate, expected, RELATIVE, "rate");
        assert_eq!(found.standard_error, 0.0, "a flat series has no error");
        assert_eq!(found.plane.offset, 0.0);
        assert_eq!(found.plane.normal, seed.normal);
        assert!(
            engine.log.step_offsets.iter().all(|&c| c == probe_offset),
            "start {start}, normal {sign}: probe plane at {:?}",
            engine.log.step_offsets.first()
        );
    }
}

#[test]
fn reactant_curvature_is_positive_below_the_minimum_coordinate() {
    // The dividing surface lies at a smaller `n . x` than the minimum, so the
    // constraint force at the probe is negative and only the signed offset
    // from the minimum keeps the curvature positive.
    let mut engine = well_engine(HALF_WIDTH);
    let found = estimate(&mut engine, &plane(&[1.0], 0.0), &quiet());
    let first = engine.log.step_offsets[0];
    assert!(first < HALF_WIDTH);
    assert!(CURVATURE * (first - HALF_WIDTH) < 0.0);
    assert_close(
        found.rate,
        harmonic_rate(CURVATURE, 1.0, BARRIER, THERMAL),
        RELATIVE,
        "rate",
    );
}

#[test]
fn probe_length_is_capped_by_the_translation_limit_and_half_the_separation() {
    // Separation 2: the cap 0.25 binds, then half the separation binds.
    for (max_translation, probe) in [(0.25, 0.25), (5.0, 1.0)] {
        let mut engine = well_engine(-HALF_WIDTH);
        let config = quiet().with_max_translation(max_translation);
        let found = estimate(&mut engine, &plane(&[1.0], 0.0), &config);
        let offset = -HALF_WIDTH + probe;
        assert!(engine.log.step_offsets.iter().all(|&c| c == offset));
        assert_eq!(engine.log.first_step_positions[0][0], offset);
        assert_close(
            found.rate,
            harmonic_rate(CURVATURE, 1.0, BARRIER, THERMAL),
            RELATIVE,
            "rate",
        );
    }
}

#[test]
fn probe_point_moves_the_minimum_along_the_normal() {
    let mut engine = Scripted::new(
        vec![-1.2, -1.6],
        vec![1.0, 1.0],
        oblique_well(),
        harmonic_force(CURVATURE, -HALF_WIDTH),
    );
    estimate(&mut engine, &plane(&[3.0, 4.0], 0.0), &quiet());
    let start = engine.log.first_step_positions[0].clone();
    let along = 0.25_f64;
    assert_close(start[0], -1.2 + 0.6 * along, RELATIVE, "x");
    assert_close(start[1], -1.6 + 0.8 * along, RELATIVE, "y");
}

#[test]
fn flux_mass_weights_the_normal_by_inverse_masses() {
    // `n = (0.6, 0.8)` and masses `(2, 8)` give `1 / mu = 0.36/2 + 0.64/8`.
    let mut engine = Scripted::new(
        vec![-1.2, -1.6],
        vec![2.0, 8.0],
        oblique_well(),
        harmonic_force(CURVATURE, -HALF_WIDTH),
    );
    let found = estimate(&mut engine, &plane(&[3.0, 4.0], 0.0), &quiet());
    let metric = 0.36 / 2.0 + 0.64 / 8.0;
    assert_close(metric, 0.26, RELATIVE, "hand value");
    let expected = harmonic_rate(CURVATURE, 1.0 / metric, BARRIER, THERMAL);
    assert_close(found.rate, expected, RELATIVE, "rate");
    let rate = found.rate;
    // The inverse-mass rate stays away from swapped masses and from mu = 1.
    let swapped = 0.64 / 2.0 + 0.36 / 8.0;
    let swapped_rate = harmonic_rate(CURVATURE, 1.0 / swapped, BARRIER, THERMAL);
    let euclidean_rate = harmonic_rate(CURVATURE, 1.0, BARRIER, THERMAL);
    let separation = RELATIVE * expected.abs().max(1.0);
    assert!(
        (rate - swapped_rate).abs() > separation,
        "swapped-mass rate {swapped_rate} is not separated from {rate}"
    );
    assert!(
        (rate - euclidean_rate).abs() > separation,
        "Euclidean rate {euclidean_rate} is not separated from {rate}"
    );
}

#[test]
fn flux_mass_rejects_inconsistent_or_unphysical_masses() {
    let build = |masses: Vec<f64>| {
        Scripted::new(
            vec![-1.2, -1.6],
            masses,
            oblique_well(),
            harmonic_force(CURVATURE, -HALF_WIDTH),
        )
    };
    let seed = plane(&[3.0, 4.0], 0.0);
    let err = error(&mut build(vec![1.0]), &seed, &quiet());
    assert_eq!(
        err,
        FluxError::Shape {
            expected: 1,
            got: 2,
            context: "OH-TST flux mass"
        }
    );
    for masses in [
        vec![-1.0, -1.0],
        vec![f64::INFINITY, f64::INFINITY],
        vec![f64::NAN, 1.0],
    ] {
        let err = error(&mut build(masses), &seed, &quiet());
        assert_eq!(
            err,
            invalid("OH-TST flux mass metric must be finite and positive")
        );
    }
}

#[test]
fn flux_mass_rejects_a_negative_component_with_a_positive_form() {
    // n = (0.6, 0.8) and masses (2, -8) give 0.36/2 + 0.64/(-8) = 0.1, so the
    // quadratic form is positive while one mass is not. The shared helper
    // rejects that mass before using the form.
    let mut engine = Scripted::new(
        vec![-1.2, -1.6],
        vec![2.0, -8.0],
        oblique_well(),
        harmonic_force(CURVATURE, -HALF_WIDTH),
    );
    let err = error(&mut engine, &plane(&[3.0, 4.0], 0.0), &quiet());
    assert_eq!(
        err,
        invalid("OH-TST flux mass metric must be finite and positive")
    );
}

fn two_batch_force(offset_by: f64) -> ConstraintForce {
    Box::new(move |plane, step| {
        let harmonic = CURVATURE * (plane.offset + HALF_WIDTH);
        harmonic + if step < 4 { -offset_by } else { offset_by }
    })
}

fn batched() -> OhtstConfig {
    quiet()
        .with_max_outer_steps(4)
        .with_sample_step_factor(2)
        .with_min_sample_steps(2)
        .with_burn_in_steps(0)
        .with_stats(StatsConfig::default().with_batch_size(4))
}

#[test]
fn standard_error_is_the_delta_method_image_of_the_batch_means_error() {
    // Two batches of four: means `0.5 -/+ 0.1`. Then
    // `sigma_BM^2 = (4 / 1) * (0.1^2 + 0.1^2) = 0.08`, `SE_lambda = sqrt(0.08 / 8) = 0.1`,
    // `sigma_ln = 0.5 * 0.1 / 0.5 = 0.1` and `SE = 0.1 * rate`.
    let mut engine = Scripted::new(
        vec![-HALF_WIDTH],
        vec![1.0],
        well(HALF_WIDTH),
        two_batch_force(0.1),
    );
    let found = estimate(&mut engine, &plane(&[1.0], 0.0), &batched());
    let rate = harmonic_rate(CURVATURE, 1.0, BARRIER, THERMAL);
    assert_close(found.rate, rate, RELATIVE, "rate");
    assert_close(found.standard_error, 0.1 * rate, RELATIVE, "standard error");
    assert_eq!(engine.log.step_sizes.len(), 8);
}

#[test]
fn standard_error_stays_positive_when_the_constraint_force_is_negative() {
    let mut engine = Scripted::new(
        vec![HALF_WIDTH],
        vec![1.0],
        well(HALF_WIDTH),
        Box::new(|plane, step| {
            let harmonic = CURVATURE * (plane.offset - HALF_WIDTH);
            harmonic + if step < 4 { -0.1 } else { 0.1 }
        }),
    );
    let found = estimate(&mut engine, &plane(&[1.0], 0.0), &batched());
    let rate = harmonic_rate(CURVATURE, 1.0, BARRIER, THERMAL);
    assert_close(found.rate, rate, RELATIVE, "rate");
    assert_close(found.standard_error, 0.1 * rate, RELATIVE, "standard error");
}

#[test]
fn curvature_sampling_follows_the_configuration() {
    let mut engine = well_engine(-HALF_WIDTH);
    let config = quiet()
        .with_seed(77)
        .with_max_outer_steps(5)
        .with_sample_step_factor(3)
        .with_min_sample_steps(4)
        .with_burn_in_steps(6)
        .with_retention_stride(5)
        .with_time_step(0.02);
    estimate(&mut engine, &plane(&[1.0], 0.0), &config);
    assert_eq!(engine.log.reseeds, vec![77]);
    assert_eq!(engine.log.shoots, vec![77]);
    // Six burn-in steps and 5 * 3 sampled steps.
    assert_eq!(engine.log.step_sizes.len(), 6 + 15);
    assert!(engine.log.step_sizes.iter().all(|&dt| dt == 0.02));
    // Positions are retained at sampled steps 0, 5 and 10.
    assert_eq!(engine.retained_reads.get(), 3);
}

#[test]
fn sample_length_takes_the_larger_of_the_floor_and_the_scaled_budget() {
    let cases = [
        // (max_outer_steps, factor, floor, expected sampled steps)
        (4, 2, 10, 10),
        (4, 2, 8, 8),
        (4, 2, 6, 8),
        (4, 0, 2, 2),
    ];
    for (outer, factor, floor, sampled) in cases {
        let mut engine = well_engine(-HALF_WIDTH);
        let config = quiet()
            .with_max_outer_steps(outer)
            .with_sample_step_factor(factor)
            .with_min_sample_steps(floor)
            .with_burn_in_steps(0);
        estimate(&mut engine, &plane(&[1.0], 0.0), &config);
        assert_eq!(engine.log.step_sizes.len(), sampled, "{cases:?}");
    }
}

#[test]
fn rate_scales_with_the_boltzmann_factor_of_the_thermal_energy() {
    for thermal in [0.125, 0.5] {
        let mut engine = well_engine(-HALF_WIDTH).with_temperature(thermal);
        let found = estimate(&mut engine, &plane(&[1.0], 0.0), &quiet());
        assert_close(
            found.rate,
            harmonic_rate(CURVATURE, 1.0, BARRIER, thermal),
            RELATIVE,
            "rate",
        );
    }
}

#[test]
fn probe_length_follows_the_signed_separation_of_an_off_center_surface() {
    // Minimum at -1.7 and stationary top at 0.3: separation 2.0 (not the
    // 1.4 of `0.3 + -1.7`), so the probe is half of it and lies above.
    let mut engine = Scripted::new(
        vec![-1.7],
        vec![1.0],
        shifted_well(0.3),
        harmonic_force(CURVATURE, -1.7),
    );
    let config = quiet().with_max_translation(5.0);
    let found = estimate(&mut engine, &plane(&[1.0], 0.3), &config);
    assert_eq!(found.plane.offset, 0.3);
    let probe_offset = -1.7 + 1.0;
    assert!(engine.log.step_offsets.iter().all(|&c| c == probe_offset));
    assert_close(
        found.rate,
        harmonic_rate(CURVATURE, 1.0, BARRIER, THERMAL),
        RELATIVE,
        "rate",
    );
}

#[test]
fn the_barrier_is_measured_from_the_minimum_energy() {
    for shift in [0.75, -3.0] {
        let inner = well(HALF_WIDTH);
        let potential: Potential = Box::new(move |x| {
            let (energy, force) = inner(x);
            (energy + shift, force)
        });
        let mut engine = Scripted::new(
            vec![-HALF_WIDTH],
            vec![1.0],
            potential,
            harmonic_force(CURVATURE, -HALF_WIDTH),
        );
        let found = estimate(&mut engine, &plane(&[1.0], 0.0), &quiet());
        assert_close(
            found.rate,
            harmonic_rate(CURVATURE, 1.0, BARRIER, THERMAL),
            RELATIVE,
            "rate",
        );
    }
}

#[test]
fn an_overflowing_rate_is_rejected() {
    // Curvature 4e300 over a flux mass of 1e-10 overflows `sqrt(κ / μ)`.
    let mut engine = Scripted::new(
        vec![-HALF_WIDTH],
        vec![1.0e-10],
        well(HALF_WIDTH),
        Box::new(|plane, _| 1.0e300 * (plane.offset + HALF_WIDTH)),
    );
    assert_eq!(
        error(&mut engine, &plane(&[1.0], 0.0), &quiet()),
        FluxError::NonFinite("OH-TST rate")
    );
}

#[test]
fn an_underflowing_rate_is_rejected() {
    // exp(-1 / 0.001) underflows to zero.
    let mut engine = well_engine(-HALF_WIDTH).with_temperature(0.001);
    assert_eq!(
        error(&mut engine, &plane(&[1.0], 0.0), &quiet()),
        FluxError::NonFinite("OH-TST rate")
    );
}
