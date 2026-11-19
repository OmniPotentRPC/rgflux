//! Equal-force-call PAFI quadrature benchmark on analytic callback potentials.

#[cfg(not(feature = "rgpot"))]
fn main() {
    eprintln!("enable the `rgpot` feature to run this benchmark");
}

#[cfg(feature = "rgpot")]
mod benchmark {
    use std::f64::consts::PI;
    use std::os::raw::c_void;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    use ndarray::{Array1, Array2};
    use rgflux::engine::native::{NativeConfig, NativeEngine};
    use rgflux::engine::rgpot::{RgpotConfig, RgpotSurface};
    use rgflux::engine::{DynamicsEngine, ThermostatConfig};
    use rgflux::pafi::{PafiConfig, PafiIntegration, pafi_profile};
    use rgflux::stats::StatsConfig;
    use rgpot_core::potential::PotentialImpl;
    use rgpot_core::status::rgpot_status_t;
    use rgpot_core::tensor::rgpot_tensor_cpu_f64_2d;
    use rgpot_core::types::{rgpot_force_input_t, rgpot_force_out_t};

    const REPLICATES: usize = 8;
    const MASSES: [f64; 3] = [2.0, 4.0, 3.0];

    #[derive(Clone, Copy, Debug)]
    enum AnalyticPotential {
        Quartic,
        Anharmonic,
        CoupledQuadratic,
    }

    impl AnalyticPotential {
        fn label(self) -> &'static str {
            match self {
                Self::Quartic => "quartic",
                Self::Anharmonic => "anharmonic-2d",
                Self::CoupledQuadratic => "coupled-quadratic",
            }
        }

        fn path(self, planes: usize) -> Array2<f64> {
            let (start, end) = match self {
                Self::Quartic | Self::Anharmonic => (-1.0, 0.0),
                Self::CoupledQuadratic => (0.0, 1.0),
            };
            let mut path = Array2::zeros((planes, 3));
            for index in 0..planes {
                path[[index, 0]] = start + (end - start) * index as f64 / (planes - 1) as f64;
            }
            path
        }

        fn free_energy(self, x: f64, kbt: f64) -> f64 {
            match self {
                Self::Quartic => (1.0 - x * x).powi(2),
                Self::Anharmonic => (1.0 - x * x).powi(2) + spectator_free_energy(x, kbt),
                Self::CoupledQuadratic => 0.5 * (3.0 - 0.8_f64.powi(2) / 2.0) * x * x,
            }
        }

        fn exact(self, kbt: f64) -> (f64, f64) {
            let (reactant, dividing, lower) = match self {
                Self::Quartic | Self::Anharmonic => (-1.0, 0.0, -2.0),
                Self::CoupledQuadratic => (0.0, 1.0, -2.0),
            };
            let reference = self.free_energy(reactant, kbt);
            let barrier = self.free_energy(dividing, kbt) - reference;
            let partition = simpson(
                |x| (-(self.free_energy(x, kbt) - reference) / kbt).exp(),
                lower,
                dividing,
                8_000,
            );
            let rate = (kbt / (2.0 * PI * MASSES[0])).sqrt() * (-barrier / kbt).exp() / partition;
            (barrier, rate.ln())
        }
    }

    struct CallbackState {
        potential: AnalyticPotential,
        forces: Vec<f64>,
        calls: Arc<AtomicU64>,
    }

    unsafe extern "C" fn drop_state(pointer: *mut c_void) {
        if !pointer.is_null() {
            let _ = unsafe { Box::from_raw(pointer as *mut CallbackState) };
        }
    }

    unsafe extern "C" fn callback(
        user_data: *mut c_void,
        input: *const rgpot_force_input_t,
        output: *mut rgpot_force_out_t,
    ) -> rgpot_status_t {
        if user_data.is_null() || input.is_null() || output.is_null() {
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        }
        let state = unsafe { &mut *(user_data as *mut CallbackState) };
        let inp = unsafe { &*input };
        let out = unsafe { &mut *output };
        if unsafe { inp.n_atoms() } != Some(1) {
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        }
        let tensor = unsafe { &(*inp.positions).dl_tensor };
        let position = unsafe { std::slice::from_raw_parts(tensor.data as *const f64, 3) };
        let (x, y, z) = (position[0], position[1], position[2]);
        let (energy, gradient) = match state.potential {
            AnalyticPotential::Quartic => (
                (1.0 - x * x).powi(2) + 0.75 * y * y + z * z,
                [-4.0 * x * (1.0 - x * x), 1.5 * y, 2.0 * z],
            ),
            AnalyticPotential::Anharmonic => {
                let stiffness = 1.5 + 0.5 * x;
                (
                    (1.0 - x * x).powi(2) + 0.5 * stiffness * y * y + 0.25 * y.powi(4) + z * z,
                    [
                        -4.0 * x * (1.0 - x * x) + 0.25 * y * y,
                        stiffness * y + y.powi(3),
                        2.0 * z,
                    ],
                )
            }
            AnalyticPotential::CoupledQuadratic => (
                0.5 * (3.0 * x * x + 1.6 * x * y + 2.0 * y * y + 2.0 * z * z),
                [3.0 * x + 0.8 * y, 0.8 * x + 2.0 * y, 2.0 * z],
            ),
        };
        out.energy = energy;
        out.variance = 0.0;
        for (force, component) in state.forces.iter_mut().zip(gradient) {
            *force = -component;
        }
        out.forces = unsafe { rgpot_tensor_cpu_f64_2d(state.forces.as_mut_ptr(), 1, 3) };
        state.calls.fetch_add(1, Ordering::Relaxed);
        rgpot_status_t::RGPOT_SUCCESS
    }

    fn simpson(function: impl Fn(f64) -> f64, lower: f64, upper: f64, intervals: usize) -> f64 {
        let width = (upper - lower) / intervals as f64;
        let mut sum = function(lower) + function(upper);
        for index in 1..intervals {
            let weight = if index % 2 == 0 { 2.0 } else { 4.0 };
            sum += weight * function(lower + index as f64 * width);
        }
        sum * width / 3.0
    }

    fn spectator_free_energy(x: f64, kbt: f64) -> f64 {
        let stiffness = 1.5 + 0.5 * x;
        let extent = 14.0 * (kbt / stiffness).sqrt();
        let partition = simpson(
            |y| (-(0.5 * stiffness * y * y + 0.25 * y.powi(4)) / kbt).exp(),
            -extent,
            extent,
            4_000,
        );
        -kbt * partition.ln()
    }

    fn engine(
        potential: AnalyticPotential,
        kbt: f64,
        seed: u64,
    ) -> (NativeEngine<RgpotSurface>, Arc<AtomicU64>) {
        let calls = Arc::new(AtomicU64::new(0));
        let state = Box::new(CallbackState {
            potential,
            forces: vec![0.0; 3],
            calls: Arc::clone(&calls),
        });
        let implementation =
            PotentialImpl::new(callback, Box::into_raw(state).cast(), Some(drop_state));
        let surface = RgpotSurface::new(
            implementation,
            RgpotConfig::default().with_atomic_numbers(vec![29]),
        )
        .expect("rgpot surface");
        let mut engine = NativeEngine::new(
            surface,
            NativeConfig::default()
                .with_masses(Array1::from_vec(MASSES.to_vec()))
                .with_seed(seed)
                .with_thermostat(
                    ThermostatConfig::default()
                        .with_gamma(2.0)
                        .with_temperature(kbt)
                        .with_boltzmann(1.0),
                ),
        )
        .expect("native engine");
        engine.shoot_velocities(seed).expect("velocities");
        (engine, calls)
    }

    #[derive(Clone, Copy)]
    struct Observation {
        barrier: f64,
        log_rate: f64,
        calls: u64,
    }

    fn observe(
        potential: AnalyticPotential,
        kbt: f64,
        integration: PafiIntegration,
        planes: usize,
        target_calls: usize,
        seed: u64,
    ) -> Observation {
        let burn_in = 8;
        let sample_steps = (target_calls / (2 * planes))
            .saturating_sub(burn_in)
            .max(16);
        let config = PafiConfig::default()
            .with_planes(planes)
            .with_integration(integration)
            .with_burn_in_steps(burn_in)
            .with_sample_steps(sample_steps)
            .with_time_step(0.01)
            .with_zero_temperature_tol(f64::MAX)
            .with_stats(StatsConfig::default().with_min_batches(2))
            .with_seed(seed);
        let path = potential.path(17);
        let (mut engine, calls) = engine(potential, kbt, seed);
        let profile = pafi_profile(&mut engine, path.view(), &config).expect("PAFI profile");
        Observation {
            barrier: profile.barrier,
            log_rate: profile.normalized_rate.ln(),
            calls: calls.load(Ordering::Relaxed),
        }
    }

    fn summarize(values: &[f64], exact: f64) -> (f64, f64, f64) {
        let count = values.len() as f64;
        let mean = values.iter().sum::<f64>() / count;
        let variance = values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (count - 1.0);
        let standard_error = (variance / count).sqrt();
        let bias = mean - exact;
        let statistic = if standard_error > 0.0 {
            bias / standard_error
        } else if bias == 0.0 {
            0.0
        } else {
            bias.signum() * f64::MAX
        };
        (bias, standard_error, statistic)
    }

    pub fn run() {
        let potentials = [
            AnalyticPotential::Quartic,
            AnalyticPotential::Anharmonic,
            AnalyticPotential::CoupledQuadratic,
        ];
        let integrations = [
            ("hermite", PafiIntegration::Hermite),
            ("gauss-legendre", PafiIntegration::GaussLegendre),
            ("gauss-jacobi", PafiIntegration::GaussJacobi),
            ("clenshaw-curtis", PafiIntegration::ClenshawCurtis),
            ("bayesian", PafiIntegration::Bayesian),
        ];
        let budgets = [(5, 2_048), (9, 4_096), (17, 8_192)];
        println!(
            "potential,kbt,rule,target_calls,mean_calls,df_bias,df_se,df_z,lnk_bias,lnk_se,lnk_z"
        );
        for potential in potentials {
            for kbt in [0.01, 0.005] {
                let (exact_barrier, exact_log_rate) = potential.exact(kbt);
                for (rule_name, integration) in integrations {
                    for (planes, target_calls) in budgets {
                        let observations: Vec<Observation> = std::thread::scope(|scope| {
                            let handles: Vec<_> = (0..REPLICATES)
                                .map(|replicate| {
                                    scope.spawn(move || {
                                        observe(
                                            potential,
                                            kbt,
                                            integration,
                                            planes,
                                            target_calls,
                                            10_000 * planes as u64 + 1_009 * replicate as u64,
                                        )
                                    })
                                })
                                .collect();
                            handles
                                .into_iter()
                                .map(|handle| handle.join().expect("replicate"))
                                .collect()
                        });
                        let barriers: Vec<f64> =
                            observations.iter().map(|value| value.barrier).collect();
                        let log_rates: Vec<f64> =
                            observations.iter().map(|value| value.log_rate).collect();
                        let calls = observations.iter().map(|value| value.calls).sum::<u64>()
                            as f64
                            / REPLICATES as f64;
                        let (barrier_bias, barrier_se, barrier_z) =
                            summarize(&barriers, exact_barrier);
                        let (rate_bias, rate_se, rate_z) = summarize(&log_rates, exact_log_rate);
                        println!(
                            "{},{kbt},{rule_name},{target_calls},{calls:.0},{barrier_bias:.9e},{barrier_se:.9e},{barrier_z:.6},{rate_bias:.9e},{rate_se:.9e},{rate_z:.6}",
                            potential.label()
                        );
                    }
                }
            }
        }
    }
}

#[cfg(feature = "rgpot")]
fn main() {
    benchmark::run();
}
