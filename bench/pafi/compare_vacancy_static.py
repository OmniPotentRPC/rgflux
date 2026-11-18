#!/usr/bin/env python3
import argparse
import json
import pathlib
import subprocess

import numpy as np
import rgpot

from run_vacancy_reference import write_data


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lammps", type=pathlib.Path, required=True)
    parser.add_argument("--path", type=pathlib.Path, required=True)
    parser.add_argument("--work-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def main():
    args = arguments()
    args.work_dir.mkdir(parents=True, exist_ok=False)
    pathway = json.loads(args.path.read_text())
    commands = []
    for index, positions in enumerate(pathway["images"]):
        image = args.work_dir / f"image_{index:04d}.dat"
        write_data(image, pathway["cell"], positions, 195.084)
        commands.extend(
            [
                "clear",
                "units metal",
                "atom_style atomic",
                "atom_modify map array sort 0 0.0",
                f"read_data {image}",
                "pair_style morse 5.5",
                "pair_coeff * * 0.7102 1.6047 2.897",
                "pair_modify shift yes",
                "neighbor 1.1 bin",
                "run 0",
                f'print "STATIC {index} $(pe:%.17g)"',
            ]
        )
    input_file = args.work_dir / "static.in"
    input_file.write_text("\n".join(commands) + "\n")
    completed = subprocess.run(
        [str(args.lammps), "-in", str(input_file)],
        check=True,
        capture_output=True,
        text=True,
    )
    reference_energies = [None] * len(pathway["images"])
    for line in completed.stdout.splitlines():
        if line.startswith("STATIC "):
            _, index, energy = line.split()
            reference_energies[int(index)] = float(energy)
    if any(value is None for value in reference_energies):
        raise RuntimeError("LAMMPS did not report every static image energy")
    potential = rgpot.MorsePot(
        De=0.7102, a=1.6047, re=2.897, cutoff=5.5
    )
    atomic_numbers = np.asarray(pathway["atomic_numbers"], dtype=np.int32)
    cell = np.asarray(pathway["cell"], dtype=np.float64).reshape(3, 3)
    native_energies = [
        float(
            potential(
                np.asarray(positions, dtype=np.float64).reshape(-1, 3),
                atomic_numbers,
                cell,
            )[0]
        )
        for positions in pathway["images"]
    ]
    result = {
        "system": "fcc-vacancy",
        "rgflux": {
            "image_energies": native_energies,
            "minimum_energy": min(native_energies),
            "saddle_energy": max(native_energies),
        },
        "lammps": {
            "image_energies": reference_energies,
            "minimum_energy": min(reference_energies),
            "saddle_energy": max(reference_energies),
        },
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
