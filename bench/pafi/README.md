# PAFI benchmark

This benchmark compares the native projected-average-force integrator with the LAMMPS `fix pafi` implementation driven by the upstream PAFI package.

## Preregistered protocol

- Potential: finite Morse dimer, with `D=1`, `rho=6`, `r_e=1`, and cutoff 4. The native side calls the pinned `rgpot-pair` kernel. The comparator uses LAMMPS `morse/omp` with the same constants.
- Straight path: symmetric dimer stretch from `r=1.0` to `r=1.4`, with 9 and 17 planes.
- Curved path: symmetric rotation of the equilibrium dimer through a quarter circle, with 17 and 33 planes.
- Temperatures: 0.05, 0.1, and 0.2 in reduced units.
- Production counts: 64, 128, and 256 samples per plane after 128 burn-in steps.
- Matched setting: time step 0.002, friction 1.0, masses 1, identical seeds and path nodes, and zero-temperature tolerance 0.01.
- Thread cells: one thread on one fixed core and four threads on four fixed cores. The native scalar pair surface has one effective evaluation thread; the four-core placement is retained and reported.
- Replication: 12 independent runs per engine and cell. Engine order is reversed on alternating rounds. Failures and zero-temperature rejections remain in the data; there are no rescue runs.
- Statistics: medians and interquartile ranges, standard errors across independent runs, Mann-Whitney p-values, Fisher exact p-values for completion rates, and an A/A profile-noise band from adjacent independent native runs.

No sampled dimer configuration approaches a periodic image or the cutoff, so no neighbour-list ambiguity enters this coverage case. The native dimer is otherwise unconfined, while the comparator uses a box wider than twice the cutoff. Potential evaluation is delegated to the canonical rgpot kernel. The native integrator continues to use the pinned rgmin and rgsaddle stack. No local potential, neighbour-list, or minimum-image implementation is present in the harness.

## Exact constrained dimer reference

Write the relative coordinate as `q = x_2 - x_1` and preserve the centre of mass with `x_1 = -q/2` and `x_2 = q/2`. For unit Cartesian masses, the straight path has mass-weighted arc `ds = dr/sqrt(2)` and unit hyperplane normal

```text
n = (-e_x, +e_x)/sqrt(2).
```

The constraint `n dot x = r/sqrt(2)` therefore fixes `q_x = r`. The Morse energy is

```text
U(R) = D [exp(-2 rho (R/r_e - 1)) - 2 exp(-rho (R/r_e - 1))].
```

An unrestricted Morse dimer has infinite transverse configurational volume because `U(R)` tends to zero. The unconfined native system therefore has no normalized global constrained free energy, and the finite comparator box defines a different dissociated-state measure. A common finite exact reference consequently requires an explicit metastable basin. The exact reference uses the interacting basin `R <= R_c`, with `R_c=4`, matching the declared pair cutoff. Integrating the transverse polar coordinate gives the one-dimensional partition integral

```text
Z_straight(r,T) = 2 pi integral_r^R_c R exp[-U(R)/T] dR,
F(r,T) - F(r_0,T) = -T log[Z_straight(r,T)/Z_straight(r_0,T)].
```

The constant delta-function and centre-of-mass factors cancel in the ratio. Equivalently,

```text
dF/dr = T r exp[-U(r)/T] / integral_r^R_c R exp[-U(R)/T] dR.
```

For path anchor `a(s)`, unit plane normal `n(s)`, and constrained multiplier `lambda`, the PAFI projected sample is

```text
lambda [dc/ds - (x-a) dot dn/ds],  c(s) = n(s) dot a(s).
```

The second term is the moving-hyperplane Jacobian, or entropic curvature correction. It vanishes on the straight path because `dn/ds=0`. The conversion between `dF/dr` and the projected mean force is exactly cancelled by `ds=dr/sqrt(2)` when the profile is integrated. Thus no energy-unit, Boltzmann-constant, path-tangent, or plane-spacing convention enters this reference.

The actual LAMMPS comparator has a periodic cubic box of side `L=20`. Its transverse minimum-image square gives a different, also exact, one-dimensional partition because the noninteracting region must be retained:

```text
Z_periodic(r,T) =
  L^2 + 2 pi integral_r^R_c R [exp(-U(R)/T) - 1] dR.
```

Its exact straight barriers are approximately 0.62821, 0.30913, and 0.04230 at `T=0.05`, `0.1`, and `0.2`. These are the proper equilibrium references for LAMMPS. There is no corresponding finite global native reference: after removing centre-of-mass volume, the native transverse integral still diverges.

For the curved path, `q = r e_r(theta)` and `ds = r dtheta/sqrt(2)`. Its normal is proportional to `(-e_theta,+e_theta)`, so the hyperplane fixes `q dot e_theta = 0`. In the circular basin the remaining `(q dot e_r,q_z)` integral is

```text
Z_curved(theta,T) = 2 pi integral_0^R_c R exp[-U(R)/T] dR,
```

which is independent of `theta`. The exact curved free-energy profile is therefore zero. The changing-normal Jacobian in PAFI is precisely the term required to recover this rotational invariance.

For the periodic cube, the principal-cell plane section rotates with the path. Its exact curved partition is

```text
Z_periodic(theta,T) =
  L^2 / max(|cos(theta)|, |sin(theta)|)
  + 2 pi integral_0^R_c R [exp(-U(R)/T) - 1] dR.
```

The finite-box curved profile is therefore not globally zero, although the interacting-basin profile is. This boundary effect is negligible only while sampling remains local to the bound dimer.

The exact comparison shows that the preregistered 128-step burn-in does not equilibrate the transverse coordinate. The restricted-basin barriers are 0.722869, 0.516477, and 0.227950, while the finite-periodic LAMMPS barriers are approximately 0.62821, 0.30913, and 0.04230. Sampled barriers remain near the zero-temperature rise. The cause is insufficient transverse equilibration and, more fundamentally, different or undefined global ensembles—not `k_B`, units, quadrature spacing, the PAFI Jacobian formula, or tangent normalization. No setting is changed after this observation.

## Periodic vacancy coverage

The many-atom case is a nearest-neighbour vacancy hop in a periodic 3 by 4 by 4 conventional FCC cell. Removing one site leaves 191 atoms. The cell uses a 3.92 Angstrom lattice constant, 195.084 atomic mass units, and a shifted platinum-parameter Morse potential with `D_e=0.7102 eV`, `a=1.6047 1/Angstrom`, `r_e=2.897 Angstrom`, and a 5.5 Angstrom cutoff. The cutoff is less than half the shortest cell length.

The native surface is `rgpot.MorsePot`; its canonical neighbour machinery supplies the link-cell or vesin walk and minimum-image displacement. The comparator uses LAMMPS `morse/omp`, `pair_modify shift yes`, and exactly the same constants. The harness contains no local potential, neighbour-list, or minimum-image implementation.

The endpoints move one nearest-neighbour atom into the vacancy. A pinned rgsaddle NEB calculation generates the shared nine-image pathway from those endpoints using `BandConfig::default()`, a periodic `Cell`, and the documented default convergence criterion. The rgsaddle revision, iteration count, and terminal maximum force are stored with the path and copied into the result data.

The vacancy grid was declared before collection: 17 and 33 PAFI planes; `k_B T=0.025`, `0.05`, and `0.075 eV`; 256 production samples after 128 burn-in steps; 12 seeds per engine and cell; one and four threads on fixed core sets. The matched setting remains time step 0.002, friction 1.0, and zero-temperature tolerance 0.01. Engine order is reversed on alternating rounds. Rejections are results and are not rerun.

## Build

Build the native executable in release mode. Every source location is an explicit argument:

```sh
python3 bench/pafi/build_native.py \
  --cargo <cargo-executable> \
  --rgflux-source <rgflux-source> \
  --rgpot-source <rgpot-source> \
  --rgsaddle-source <rgsaddle-source> \
  --source bench/pafi/src/main.rs \
  --build-dir <native-build-dir>
```

Build a second release executable with `bench/pafi/src/vacancy.rs` as `--source`. Build the pinned rgpot Python package in release mode with Python bindings and the canonical fast neighbour stack enabled, while runtime ISA dispatch is disabled.

Configure LAMMPS in release mode with shared libraries, MPI, OpenMP, `EXTRA-FIX`, `OPENMP`, `MANYBODY`, `MOLECULE`, and `REPLICA`. Install its Python bindings and the PAFI driver from the recorded source revisions. Place the installed shared LAMMPS library beside the Python binding before running.

## Run

Choose CPU sets from the process affinity reported by the operating system. Run the complete preregistered grid:

```sh
python3 bench/pafi/run_campaign.py \
  --native-bin <native-build-dir>/target/release/rgflux-pafi-bench \
  --python <python-executable> \
  --reference-script bench/pafi/run_reference.py \
  --raw-dir <raw-results-dir> \
  --straight-planes 9,17 \
  --curved-planes 17,33 \
  --temperatures 0.05,0.1,0.2 \
  --sample-counts 64,128,256 \
  --burn-in 128 \
  --replicates 12 \
  --time-step 0.002 \
  --friction 1.0 \
  --zero-temperature-tolerance 0.01 \
  --high-threads 4 \
  --one-core-set <one-core> \
  --high-core-set <four-cores>
```

Raw logs and working files remain outside the repository. Aggregate the completed campaign:

```sh
python3 bench/pafi/analyze.py \
  --raw-dir <raw-results-dir> \
  --vacancy-raw-dir <vacancy-raw-results-dir> \
  --vacancy-path <vacancy-path-json> \
  --vacancy-static-reference <vacancy-static-json> \
  --metadata <metadata-json> \
  --output-json bench/pafi/results.json \
  --output-markdown bench/pafi/results.md
```

The retained campaign exposed a direct-exponential overflow in rgflux reactant-partition integration. Reproduce the exact rejected seed without changing any method setting:

```sh
python3 bench/pafi/repro_partition_overflow.py \
  --native-bin <native-build-dir>/target/release/rgflux-pafi-bench
```

Generate the vacancy path once from the pinned rgsaddle revision:

```sh
<vacancy-build-dir>/target/release/rgflux-pafi-bench generate \
  --site-packages <rgpot-site-packages> \
  --output <vacancy-path-json> \
  --rgsaddle-sha <rgsaddle-sha>
```

Run the complete vacancy campaign:

```sh
python3 bench/pafi/run_vacancy_campaign.py \
  --native-bin <vacancy-build-dir>/target/release/rgflux-pafi-bench \
  --site-packages <rgpot-site-packages> \
  --path <vacancy-path-json> \
  --python <python-executable> \
  --reference-script bench/pafi/run_vacancy_reference.py \
  --raw-dir <vacancy-raw-results-dir> \
  --planes 17,33 \
  --temperatures 0.025,0.05,0.075 \
  --samples 256 \
  --burn-in 128 \
  --replicates 12 \
  --time-step 0.002 \
  --friction 1.0 \
  --zero-temperature-tolerance 0.01 \
  --high-threads 4 \
  --one-core-set <one-core> \
  --high-core-set <four-cores>
```

Evaluate the shared NEB images independently with the release comparator:

```sh
python3 bench/pafi/compare_vacancy_static.py \
  --lammps <lammps-executable> \
  --path <vacancy-path-json> \
  --work-dir <static-work-dir> \
  --output <vacancy-static-json>
```
