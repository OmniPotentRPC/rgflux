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

The dimer is finite and its box is wider than twice the cutoff, so no periodic image or neighbour-list ambiguity enters this coverage case. Potential evaluation is delegated to the canonical rgpot kernel. The native integrator continues to use the pinned rgmin and rgsaddle stack. No local potential, neighbour-list, or minimum-image implementation is present in the harness.

## Build

Build the native executable in release mode. Every source location is an explicit argument:

```sh
python3 bench/pafi/build_native.py \
  --cargo <cargo-executable> \
  --rgflux-source <rgflux-source> \
  --rgpot-source <rgpot-source> \
  --source bench/pafi/src/main.rs \
  --build-dir <native-build-dir>
```

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
  --metadata <metadata-json> \
  --output-json bench/pafi/results.json \
  --output-markdown bench/pafi/results.md
```

The retained campaign exposed a direct-exponential overflow in rgflux reactant-partition integration. Reproduce the exact rejected seed without changing any method setting:

```sh
python3 bench/pafi/repro_partition_overflow.py \
  --native-bin <native-build-dir>/target/release/rgflux-pafi-bench
```
