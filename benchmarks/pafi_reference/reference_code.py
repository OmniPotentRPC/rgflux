#!/usr/bin/env python3
"""Swinburne and Marinica's PAFI code on its own W vacancy example.

``run`` samples ``examples/systems/EAM-VAC-W`` with the settings of the
reference ``examples/input_python.py`` (Langevin, 2000 thermalization and
2000 sampling steps per trajectory, the other parameters at their
defaults). Every worker samples every plane ``nRepeats`` times from the
seed ``GlobalSeed * (worker + 1)``, so a run is reproducible and each MPI
rank adds independent trajectories. The reference code writes ``dumps/``
in the working directory, so every run gets its own. ``collate`` averages
the valid trajectories on every plane and integrates the plane means the
two ways ``tests/pafi_reference.rs`` integrates rgsaddle's, and also
reports what the reference ``ResultsProcessor`` prints for the same dumps.

    python reference_code.py run SYSTEMS OUT 1000,2000 32 9
    mpirun -np 8 python reference_code.py run SYSTEMS OUT 1000,2000 32 9
    python reference_code.py collate SUMMARY.csv PLANES.csv OUT...

A process that does not join the ``mpirun`` world, as when mpi4py was
built against another MPI, is worker 0 of a world of its own: it samples
the whole set with the seeds of worker 0 and writes the same table under
its own suffix. ``run`` refuses to start then, and ``collate`` counts
repeated tables and rows once.

The reference code reads the per-worker variance, ``avePsi`` and
``dXTangent`` with ``extract_fix(..., ncol=i)``. LAMMPS reads that index
as ``nrow`` for a global vector, so with current LAMMPS those columns all
hold element 0 of ``fix pafi``. ``FreeEnergyGradient`` does not depend on
them, and the errors here come from the spread over workers alone.

``ResultsProcessor`` finds the rows of a plane by its coordinate rounded to
four decimals. With 33 planes the odd ``r = k/32`` then match no rows, and
the barrier it prints comes from the other 17; with nine planes every
plane is found.
"""

from __future__ import annotations

import hashlib
import os
import sys
from glob import glob

import numpy as np


def run(systems: str, out: str, temperatures: str, repeats: str, planes: str) -> None:
    from mpi4py import MPI
    from pafi import PAFIManager, PAFIParser

    system = os.path.join(os.path.abspath(systems), "EAM-VAC-W")
    world = MPI.COMM_WORLD
    launched = max(int(os.environ.get(name, "1")) for name in ("PMI_SIZE", "OMPI_COMM_WORLD_SIZE"))
    if world.Get_size() != launched:
        sys.exit(
            f"mpi4py sees a world of {world.Get_size()} of the {launched} processes launched; "
            "use the mpirun of the MPI mpi4py was built with, or one process"
        )
    if world.Get_rank() == 0:
        os.makedirs(out, exist_ok=True)
    world.Barrier()
    os.chdir(out)
    parameters = PAFIParser(rank=world.Get_rank())
    parameters.set_pathway("image_*.dat", directory=system)
    parameters.set_potential(os.path.join(system, "W.eam.fs"), pot_type="eam/fs")
    parameters.set_species("W")
    parameters.axes["Temperature"] = [float(t) for t in temperatures.split(",")]
    count = int(planes)
    parameters.axes["ReactionCoordinate"] = [k / (count - 1) for k in range(count)]
    parameters.set("nRepeats", int(repeats))
    parameters.set("OverDamped", 0)
    parameters.set("SampleSteps", 2000)
    parameters.set("ThermSteps", 2000)
    parameters.set("ThermWindow", 100)
    parameters.set("Verbose", 0)
    manager = PAFIManager(world, parameters=parameters)
    manager.run()
    manager.close()


def weights(arcs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linear maps from plane gradients to the cumulative integral.

    Rows are the trapezoid sum on the planes, and the trapezoid sum on
    ``10 n`` points of the not-a-knot cubic through them, as the reference
    ``ResultsProcessor.integrate`` does.
    """
    from scipy.integrate import cumulative_trapezoid
    from scipy.interpolate import interp1d

    count = arcs.size
    identity = np.eye(count)
    trapezoid = cumulative_trapezoid(identity, arcs, axis=0, initial=0)
    dense = np.linspace(arcs.min(), arcs.max(), 10 * count)
    cubic = interp1d(arcs, identity, axis=0, kind="cubic")(dense)
    remeshed = cumulative_trapezoid(cubic, dense, axis=0, initial=0)
    return trapezoid, remeshed


def integral(rows: np.ndarray, gradient: np.ndarray, error: np.ndarray) -> list[float]:
    values = rows @ gradient
    top = int(np.argmax(values))

    def spread(row: np.ndarray) -> float:
        return float(np.sqrt(np.sum((row * error) ** 2)))

    return [values[top], spread(rows[top]), values[-1], spread(rows[-1])]


def tables(path: str) -> list[str]:
    """The ``pafi_data_*.csv`` of a run, each distinct table once."""
    seen, files = set(), []
    for name in sorted(glob(os.path.join(path, "dumps", "pafi_data_*.csv"))):
        with open(name, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            files.append(name)
    return files


def processor(path: str) -> dict[float, tuple[float, float, float, float]]:
    """What the reference plotting snippet prints: ``y.max()`` and ``e.max()``.

    Reading the configuration makes its ``DumpFolder``, ``./dumps``, in the
    working directory, so this runs from that of the run.
    """
    from pafi import ResultsProcessor

    path = os.path.abspath(path)
    here = os.getcwd()
    os.chdir(path)
    try:
        results = ResultsProcessor(
            data_path=tables(path),
            xml_path=os.path.join(path, "dumps", "config_0.xml"),
            integrate=True,
        )
        data, _, y_key = results.plotting_data()
    finally:
        os.chdir(here)
    printed = {}
    for row in data:
        y, e = row[y_key], row[y_key + "_err"]
        printed[float(row["Temperature"])] = (y.max(), e.max(), y[-1], e[-1])
    return printed


def collate(summary: str, planes: str, paths: list[str]) -> None:
    import pandas as pd

    head = "system,temperature_k,planes,integration,barrier,barrier_error,end,end_error,trajectories\n"
    plane_head = "system,temperature_k,planes,r,gradient,gradient_error,valid,total,min_energy\n"
    summary_rows, plane_rows = [], []
    for path in paths:
        data = pd.concat([pd.read_csv(name) for name in tables(path)], ignore_index=True)
        data = data.drop_duplicates(subset=[c for c in data.columns if not c.startswith("Unnamed")])
        start = data[np.isclose(data["ReactionCoordinate"], 0.0)]["MinEnergy"].mean()
        printed = processor(path)
        for temperature, group in data.groupby("Temperature"):
            arcs = np.sort(group["ReactionCoordinate"].unique())
            gradient, error, valid = [], [], []
            for arc in arcs:
                plane = group[np.isclose(group["ReactionCoordinate"], arc)]
                kept = plane[plane["Valid"].astype(bool)]["FreeEnergyGradient"].to_numpy()
                gradient.append(kept.mean())
                error.append(np.sqrt(kept.var() / kept.size))
                valid.append(kept.size)
                plane_rows.append(
                    f"EAM-VAC-W,{temperature:g},{arcs.size},{arc:.6f},{kept.mean():.6f},"
                    f"{error[-1]:.6f},{kept.size},{len(plane)},"
                    f"{plane['MinEnergy'].mean() - start:.6f}\n"
                )
            gradient, error = np.array(gradient), np.array(error)
            for name, rows in zip(("trapezoid", "remeshed"), weights(arcs)):
                values = integral(rows, gradient, error)
                summary_rows.append(
                    f"EAM-VAC-W,{temperature:g},{arcs.size},{name},"
                    + ",".join(f"{value:.6f}" for value in values)
                    + f",{min(valid)}\n"
                )
            values = printed[float(temperature)]
            summary_rows.append(
                f"EAM-VAC-W,{temperature:g},{arcs.size},processor,"
                + ",".join(f"{value:.6f}" for value in values)
                + f",{min(valid)}\n"
            )
    with open(summary, "w") as handle:
        handle.write(head + "".join(summary_rows))
    with open(planes, "w") as handle:
        handle.write(plane_head + "".join(plane_rows))


if __name__ == "__main__":
    if len(sys.argv) == 7 and sys.argv[1] == "run":
        run(*sys.argv[2:])
    elif len(sys.argv) >= 5 and sys.argv[1] == "collate":
        collate(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        sys.exit(__doc__)
