#!/usr/bin/env python3
import argparse
import json
import math
import pathlib
import time

import numpy as np
import pandas as pd
from mpi4py import MPI
from pafi import PAFIManager, PAFIParser


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("straight", "curved"), required=True)
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


def coordinates(case_name, fraction):
    if case_name == "straight":
        distance = 1.0 + 0.4 * fraction
        angle = 0.0
    else:
        distance = 1.0
        angle = 0.5 * math.pi * fraction
    half_x = 0.5 * distance * math.cos(angle)
    half_y = 0.5 * distance * math.sin(angle)
    return ((-half_x, -half_y, 0.0), (half_x, half_y, 0.0))


def write_data(path, positions):
    path.write_text(
        "Morse dimer PAFI path\n\n"
        "2 atoms\n"
        "1 atom types\n\n"
        "-10.0 10.0 xlo xhi\n"
        "-10.0 10.0 ylo yhi\n"
        "-10.0 10.0 zlo zhi\n\n"
        "Atoms # atomic\n\n"
        f"1 1 {positions[0][0]:.17g} {positions[0][1]:.17g} {positions[0][2]:.17g}\n"
        f"2 1 {positions[1][0]:.17g} {positions[1][1]:.17g} {positions[1][2]:.17g}\n"
    )


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
    image_paths = []
    for index in range(args.planes):
        fraction = index / (args.planes - 1)
        image = args.work_dir / f"image_{index:04d}.dat"
        write_data(image, coordinates(args.case, fraction))
        image_paths.append(str(image))
    potential_marker = args.work_dir / "morse.parameters"
    potential_marker.write_text("D=1 rho=6 re=1 cutoff=4\n")

    parameters = PAFIParser(rank=MPI.COMM_WORLD.Get_rank())
    parameters.set("DumpFolder", str(args.work_dir))
    parameters.set_pathway(image_paths)
    parameters.set_potential(
        str(potential_marker), type="morse", species=["X"]
    )
    parameters.axes["Temperature"] = np.array([args.temperature])
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
        units lj
        atom_style atomic
        atom_modify map array sort 0 0.0
        package omp {args.threads}
        read_data %FirstPathConfiguration%
        mass 1 1.0
        pair_style morse/omp 4.0
        pair_coeff * * 1.0 6.0 1.0
        neighbor 0.3 bin
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
        "status": "ok",
        "case": args.case,
        "planes": args.planes,
        "temperature": args.temperature,
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
