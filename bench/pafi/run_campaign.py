#!/usr/bin/env python3
import argparse
import itertools
import json
import pathlib
import subprocess


def csv_numbers(value, conversion):
    return [conversion(item) for item in value.split(",")]


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-bin", type=pathlib.Path, required=True)
    parser.add_argument("--python", type=pathlib.Path, required=True)
    parser.add_argument("--reference-script", type=pathlib.Path, required=True)
    parser.add_argument("--raw-dir", type=pathlib.Path, required=True)
    parser.add_argument("--straight-planes", required=True)
    parser.add_argument("--curved-planes", required=True)
    parser.add_argument("--temperatures", required=True)
    parser.add_argument("--sample-counts", required=True)
    parser.add_argument("--burn-in", type=int, required=True)
    parser.add_argument("--replicates", type=int, required=True)
    parser.add_argument("--time-step", type=float, required=True)
    parser.add_argument("--friction", type=float, required=True)
    parser.add_argument("--zero-temperature-tolerance", type=float, required=True)
    parser.add_argument("--high-threads", type=int, required=True)
    parser.add_argument("--one-core-set", required=True)
    parser.add_argument("--high-core-set", required=True)
    args = parser.parse_args()
    if args.replicates < 12:
        parser.error("--replicates must be at least 12")
    if args.high_threads <= 1:
        parser.error("--high-threads must exceed one")
    return args


def execute(command, log_path):
    with log_path.open("w") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    return completed.returncode


def native_run(args, cell, seed, threads, core_set, output, log):
    command = [
        "taskset",
        "--cpu-list",
        core_set,
        str(args.native_bin),
        "--case",
        cell["case"],
        "--planes",
        str(cell["planes"]),
        "--temperature",
        str(cell["temperature"]),
        "--samples",
        str(cell["samples"]),
        "--burn-in",
        str(args.burn_in),
        "--seed",
        str(seed),
        "--threads",
        str(threads),
        "--time-step",
        str(args.time_step),
        "--friction",
        str(args.friction),
        "--zero-temperature-tolerance",
        str(args.zero_temperature_tolerance),
    ]
    code = execute(command, log)
    if code == 0:
        output.write_text(log.read_text())
    else:
        output.write_text(
            json.dumps(
                {
                    "engine": "rgflux",
                    "status": "process-error",
                    "exit_code": code,
                },
                sort_keys=True,
            )
            + "\n"
        )


def reference_run(args, cell, seed, threads, core_set, output, log, work):
    command = [
        "taskset",
        "--cpu-list",
        core_set,
        str(args.python),
        str(args.reference_script),
        "--case",
        cell["case"],
        "--planes",
        str(cell["planes"]),
        "--temperature",
        str(cell["temperature"]),
        "--samples",
        str(cell["samples"]),
        "--burn-in",
        str(args.burn_in),
        "--seed",
        str(seed),
        "--threads",
        str(threads),
        "--time-step",
        str(args.time_step),
        "--friction",
        str(args.friction),
        "--work-dir",
        str(work),
        "--output",
        str(output),
    ]
    code = execute(command, log)
    if code != 0:
        output.write_text(
            json.dumps(
                {
                    "engine": "lammps-pafi",
                    "status": "process-error",
                    "exit_code": code,
                },
                sort_keys=True,
            )
            + "\n"
        )


def main():
    args = arguments()
    args.raw_dir.mkdir(parents=True, exist_ok=False)
    logs = args.raw_dir / "logs"
    work = args.raw_dir / "work"
    records = args.raw_dir / "records"
    logs.mkdir()
    work.mkdir()
    records.mkdir()

    plane_sets = {
        "straight": csv_numbers(args.straight_planes, int),
        "curved": csv_numbers(args.curved_planes, int),
    }
    temperatures = csv_numbers(args.temperatures, float)
    samples = csv_numbers(args.sample_counts, int)
    thread_cells = [
        (1, args.one_core_set),
        (args.high_threads, args.high_core_set),
    ]
    manifest = []
    cell_index = 0
    for case_name in ("straight", "curved"):
        cells = itertools.product(
            plane_sets[case_name], temperatures, samples, thread_cells
        )
        for planes, temperature, sample_count, thread_cell in cells:
            threads, core_set = thread_cell
            cell = {
                "case": case_name,
                "planes": planes,
                "temperature": temperature,
                "samples": sample_count,
                "threads": threads,
            }
            for replicate in range(args.replicates):
                seed = 1000003 + 1009 * cell_index + replicate
                order = ("rgflux", "lammps-pafi")
                if replicate % 2:
                    order = tuple(reversed(order))
                for engine in order:
                    stem = (
                        f"{case_name}-p{planes}-t{temperature:g}-s{sample_count}"
                        f"-n{threads}-r{replicate:02d}-{engine}"
                    )
                    output = records / f"{stem}.json"
                    log = logs / f"{stem}.log"
                    run_work = work / stem
                    if engine == "rgflux":
                        native_run(
                            args, cell, seed, threads, core_set, output, log
                        )
                    else:
                        reference_run(
                            args,
                            cell,
                            seed,
                            threads,
                            core_set,
                            output,
                            log,
                            run_work,
                        )
                    manifest.append(
                        {
                            **cell,
                            "replicate": replicate,
                            "seed": seed,
                            "engine": engine,
                            "record": str(output.relative_to(args.raw_dir)),
                            "log": str(log.relative_to(args.raw_dir)),
                        }
                    )
            cell_index += 1
    (args.raw_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
