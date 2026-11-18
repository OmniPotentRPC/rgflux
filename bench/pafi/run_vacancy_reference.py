#!/usr/bin/env python3
import argparse
import json
import pathlib
import time

import numpy as np
import pandas as pd
from mpi4py import MPI
from pafi import PAFIManager, PAFIParser


BOLTZMANN_EV_PER_KELVIN = 8.617333262e-5


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=pathlib.Path, required=True)
    parser.add_argument("--planes", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--burn-in", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--time-step", type=float, required=True)
    parser.add_argument("--friction", type=float, required=True)
    parser.add_argument("--work-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def write_data(path, cell, positions, mass):
    atom_count = len(positions) // 3
    lines = [
        "Periodic FCC vacancy PAFI path",
        "",
        f"{atom_count} atoms",
        "1 atom types",
        "",
        f"0.0 {cell[0]:.17g} xlo xhi",
        f"0.0 {cell[4]:.17g} ylo yhi",
        f"0.0 {cell[8]:.17g} zlo zhi",
        "",
        "Masses",
        "",
        f"1 {mass:.17g}",
        "",
        "Atoms # atomic",
        "",
    ]
    for index in range(atom_count):
        offset = 3 * index
        lines.append(
            f"{index + 1} 1 {positions[offset]:.17g} "
            f"{positions[offset + 1]:.17g} {positions[offset + 2]:.17g}"
        )
    path.write_text("\n".join(lines) + "\n")


def integrate_profile(reaction_coordinate, gradients):
    profile = [0.0]
    for index in range(1, len(reaction_coordinate)):
        width = reaction_coordinate[index] - reaction_coordinate[index - 1]
        profile.append(
            profile[-1] + 0.5 * width * (gradients[index - 1] + gradients[index])
        )
    return profile


def main():
    args = arguments()
    args.work_dir.mkdir(parents=True, exist_ok=False)
    pathway = json.loads(args.path.read_text())
    mass = 195.084
    image_paths = []
    for index, positions in enumerate(pathway["images"]):
        image = args.work_dir / f"image_{index:04d}.dat"
        write_data(image, pathway["cell"], positions, mass)
        image_paths.append(str(image))
    potential_marker = args.work_dir / "morse.parameters"
    potential_marker.write_text(
        "De=0.7102 a=1.6047 re=2.897 cutoff=5.5 shift=yes\n"
    )

    temperature_kelvin = args.temperature / BOLTZMANN_EV_PER_KELVIN
    parameters = PAFIParser(rank=MPI.COMM_WORLD.Get_rank())
    parameters.set("DumpFolder", str(args.work_dir))
    parameters.set_pathway(image_paths)
    parameters.set_potential(
        str(potential_marker), type="morse", species=["Pt"]
    )
    parameters.axes["Temperature"] = np.array([temperature_kelvin])
    parameters.axes["ReactionCoordinate"] = np.linspace(0.0, 1.0, args.planes)
    parameters.set("SampleSteps", args.samples)
    parameters.set("ThermSteps", args.burn_in)
    parameters.set("ThermWindow", min(args.burn_in, 100))
    parameters.set("GlobalSeed", args.seed)
    parameters.set("FreshSeed", 1)
    parameters.set("Friction", args.friction)
    parameters.set("PostDump", 0)
    parameters.set("nRepeats", 1)
    parameters.set_script(
        "Input",
        f"""
        units metal
        atom_style atomic
        atom_modify map array sort 0 0.0
        package omp {args.threads}
        read_data %FirstPathConfiguration%
        pair_style morse/omp 5.5
        pair_coeff * * 0.7102 1.6047 2.897
        pair_modify shift yes
        neighbor 1.1 bin
        neigh_modify check yes
        timestep {args.time_step:.17g}
        thermo 1000000
        run 0
        """,
    )

    started = time.perf_counter()
    manager = PAFIManager(MPI.COMM_WORLD, parameters=parameters)
    manager.run()
    manager.close()
    wall_seconds = time.perf_counter() - started

    if MPI.COMM_WORLD.Get_rank() != 0:
        return
    frame = pd.read_csv(parameters.csv_file)
    frame = frame.sort_values("ReactionCoordinate")
    coordinate = frame["ReactionCoordinate"].astype(float).tolist()
    gradients = frame["FreeEnergyGradient"].astype(float).tolist()
    profile = integrate_profile(coordinate, gradients)
    energies = frame["MinEnergy"].astype(float).tolist()
    result = {
        "engine": "lammps-pafi",
        "system": "fcc-vacancy",
        "status": "ok",
        "planes": args.planes,
        "temperature": args.temperature,
        "temperature_kelvin": temperature_kelvin,
        "samples": args.samples,
        "burn_in": args.burn_in,
        "seed": args.seed,
        "requested_threads": args.threads,
        "effective_threads": args.threads,
        "time_step": args.time_step,
        "friction": args.friction,
        "wall_seconds": wall_seconds,
        "potential_seconds": None,
        "overhead_seconds": None,
        "force_calls": args.planes * (args.burn_in + args.samples),
        "force_calls_kind": "derived_from_configured_integration_steps",
        "minimum_energy": min(energies),
        "saddle_energy": max(energies),
        "reaction_coordinate": coordinate,
        "free_energy_gradient": gradients,
        "free_energy": profile,
        "barrier": max(profile),
    }
    args.output.write_text(json.dumps(result, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
