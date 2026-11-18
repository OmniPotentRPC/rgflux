#!/usr/bin/env python3
import argparse
import json
import math
import pathlib
import statistics

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=pathlib.Path, required=True)
    parser.add_argument("--metadata", type=pathlib.Path, required=True)
    parser.add_argument("--output-json", type=pathlib.Path, required=True)
    parser.add_argument("--output-markdown", type=pathlib.Path, required=True)
    return parser.parse_args()


def distribution(values):
    clean = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if not len(clean):
        return {"n": 0, "median": None, "q1": None, "q3": None}
    return {
        "n": int(len(clean)),
        "median": float(np.median(clean)),
        "q1": float(np.percentile(clean, 25)),
        "q3": float(np.percentile(clean, 75)),
    }


def sem(values):
    clean = [value for value in values if math.isfinite(value)]
    if len(clean) < 2:
        return None
    return statistics.stdev(clean) / math.sqrt(len(clean))


def profile_sem(records):
    profiles = np.asarray([record["free_energy"] for record in records], dtype=float)
    if len(profiles) < 2:
        return [None] * profiles.shape[1]
    return (profiles.std(axis=0, ddof=1) / math.sqrt(len(profiles))).tolist()


def maximum_profile_difference(left, right):
    return max(abs(a - b) for a, b in zip(left["free_energy"], right["free_energy"]))


def format_distribution(summary, digits=4):
    if summary["median"] is None:
        return "n/a"
    return (
        f'{summary["median"]:.{digits}g} '
        f'[{summary["q1"]:.{digits}g}, {summary["q3"]:.{digits}g}]'
    )


def main():
    args = arguments()
    metadata = json.loads(args.metadata.read_text())
    manifest = json.loads((args.raw_dir / "manifest.json").read_text())
    entries = []
    for item in manifest:
        record = json.loads((args.raw_dir / item["record"]).read_text())
        entries.append((item, record))

    keys = sorted(
        {
            (
                item["case"],
                item["planes"],
                item["temperature"],
                item["samples"],
                item["threads"],
            )
            for item, _ in entries
        }
    )
    cells = []
    for key in keys:
        selected = [
            (item, record)
            for item, record in entries
            if (
                item["case"],
                item["planes"],
                item["temperature"],
                item["samples"],
                item["threads"],
            )
            == key
        ]
        by_engine = {
            engine: [
                (item, record)
                for item, record in selected
                if item["engine"] == engine and record.get("status") == "ok"
            ]
            for engine in ("rgflux", "lammps-pafi")
        }
        all_by_engine = {
            engine: [
                (item, record)
                for item, record in selected
                if item["engine"] == engine
            ]
            for engine in ("rgflux", "lammps-pafi")
        }
        native = [record for _, record in by_engine["rgflux"]]
        reference = [record for _, record in by_engine["lammps-pafi"]]
        native_by_seed = {item["seed"]: record for item, record in by_engine["rgflux"]}
        reference_by_seed = {
            item["seed"]: record for item, record in by_engine["lammps-pafi"]
        }
        paired = [
            (native_by_seed[seed], reference_by_seed[seed])
            for seed in sorted(set(native_by_seed) & set(reference_by_seed))
        ]
        profile_differences = [
            maximum_profile_difference(left, right) for left, right in paired
        ]
        barrier_differences = [
            abs(left["barrier"] - right["barrier"]) for left, right in paired
        ]
        native_profile_sem = profile_sem(native) if native else []
        reference_profile_sem = profile_sem(reference) if reference else []
        combined_profile_sem = [
            math.hypot(a, b)
            for a, b in zip(native_profile_sem, reference_profile_sem)
            if a is not None and b is not None
        ]
        aa_differences = []
        for index in range(0, len(native) - 1, 2):
            aa_differences.append(
                maximum_profile_difference(native[index], native[index + 1])
            )
        native_wall = [record["wall_seconds"] for record in native]
        reference_wall = [record["wall_seconds"] for record in reference]
        native_barriers = [record["barrier"] for record in native]
        reference_barriers = [record["barrier"] for record in reference]
        native_successes = len(by_engine["rgflux"])
        reference_successes = len(by_engine["lammps-pafi"])
        native_failures = len(all_by_engine["rgflux"]) - native_successes
        reference_failures = len(all_by_engine["lammps-pafi"]) - reference_successes
        hit_rate_p = fisher_exact(
            [
                [native_successes, native_failures],
                [reference_successes, reference_failures],
            ]
        ).pvalue
        wall_p = (
            float(mannwhitneyu(native_wall, reference_wall).pvalue)
            if native_wall and reference_wall
            else None
        )
        barrier_p = (
            float(mannwhitneyu(native_barriers, reference_barriers).pvalue)
            if native_barriers and reference_barriers
            else None
        )
        cell = {
            "case": key[0],
            "planes": key[1],
            "temperature": key[2],
            "samples": key[3],
            "threads": key[4],
            "successes": {
                "rgflux": native_successes,
                "lammps_pafi": reference_successes,
            },
            "failures": {
                "rgflux": native_failures,
                "lammps_pafi": reference_failures,
            },
            "hit_rate_fisher_p": float(hit_rate_p),
            "wall_seconds": {
                "rgflux": distribution(native_wall),
                "lammps_pafi": distribution(reference_wall),
                "mann_whitney_p": wall_p,
            },
            "barrier": {
                "rgflux": distribution(native_barriers),
                "rgflux_sem": sem(native_barriers),
                "lammps_pafi": distribution(reference_barriers),
                "lammps_pafi_sem": sem(reference_barriers),
                "absolute_difference": distribution(barrier_differences),
                "mann_whitney_p": barrier_p,
            },
            "max_absolute_profile_difference": distribution(profile_differences),
            "max_combined_profile_sem": max(combined_profile_sem)
            if combined_profile_sem
            else None,
            "aa_profile_noise": distribution(aa_differences),
            "force_calls_per_plane": {
                "rgflux": distribution(
                    [record["force_calls"] / key[1] for record in native]
                ),
                "lammps_pafi": distribution(
                    [record["force_calls"] / key[1] for record in reference]
                ),
            },
            "rgflux_overhead_seconds_per_plane": distribution(
                [record["overhead_seconds"] / key[1] for record in native]
            ),
            "static_energy_absolute_difference": {
                "minimum": distribution(
                    [
                        abs(left["minimum_energy"] - right["minimum_energy"])
                        for left, right in paired
                    ]
                ),
                "saddle": distribution(
                    [
                        abs(left["saddle_energy"] - right["saddle_energy"])
                        for left, right in paired
                    ]
                ),
            },
        }
        cells.append(cell)

    result = {
        "schema_version": 1,
        "metadata": metadata,
        "protocol": {
            "replicates_per_engine_cell": min(
                len([item for item, _ in entries if item["engine"] == engine])
                for engine in ("rgflux", "lammps-pafi")
            )
            // len(keys),
            "order": "interleaved and reversed on alternating replicates",
            "aa_noise": "absolute profile differences between adjacent independent rgflux replicates",
            "interval_summary": "median [Q1, Q3]",
            "sampling_error": "standard error across independent replicates",
        },
        "cells": cells,
    }
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    largest_samples = max(cell["samples"] for cell in cells)
    headline = [cell for cell in cells if cell["samples"] == largest_samples]
    lines = [
        "# Native PAFI benchmark",
        "",
        "## Verdict",
        "",
    ]
    complete = all(
        cell["successes"]["rgflux"] >= 12
        and cell["successes"]["lammps_pafi"] >= 12
        for cell in cells
    )
    lines.append(
        "All planned cells completed with at least 12 independent runs per engine."
        if complete
        else "At least one planned cell did not complete 12 independent runs per engine."
    )
    lines.extend(
        [
            "",
            "The table reports the largest sample count. Values are medians with interquartile ranges; profile and barrier differences are paired by seed.",
            "",
            "| case | planes | T | threads | max |dF| | barrier |dF| | combined profile SE | rgflux wall (s) | reference wall (s) | wall p |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in headline:
        lines.append(
            "| {case} | {planes} | {temperature:g} | {threads} | {profile} | "
            "{barrier} | {error} | {native} | {reference} | {p} |".format(
                case=cell["case"],
                planes=cell["planes"],
                temperature=cell["temperature"],
                threads=cell["threads"],
                profile=format_distribution(cell["max_absolute_profile_difference"]),
                barrier=format_distribution(cell["barrier"]["absolute_difference"]),
                error=(
                    f'{cell["max_combined_profile_sem"]:.4g}'
                    if cell["max_combined_profile_sem"] is not None
                    else "n/a"
                ),
                native=format_distribution(cell["wall_seconds"]["rgflux"]),
                reference=format_distribution(cell["wall_seconds"]["lammps_pafi"]),
                p=(
                    f'{cell["wall_seconds"]["mann_whitney_p"]:.3g}'
                    if cell["wall_seconds"]["mann_whitney_p"] is not None
                    else "n/a"
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Convergence and A/A noise",
            "",
            "| case | planes | T | threads | samples | max |dF| | A/A profile noise | barrier p |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in cells:
        lines.append(
            "| {case} | {planes} | {temperature:g} | {threads} | {samples} | "
            "{profile} | {noise} | {p} |".format(
                case=cell["case"],
                planes=cell["planes"],
                temperature=cell["temperature"],
                threads=cell["threads"],
                samples=cell["samples"],
                profile=format_distribution(cell["max_absolute_profile_difference"]),
                noise=format_distribution(cell["aa_profile_noise"]),
                p=(
                    f'{cell["barrier"]["mann_whitney_p"]:.3g}'
                    if cell["barrier"]["mann_whitney_p"] is not None
                    else "n/a"
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Comparator and protocol",
            "",
            f'- rgflux: `{metadata["rgflux_sha"]}`',
            f'- rgpot: `{metadata["rgpot_sha"]}`',
            f'- rgsaddle: `{metadata["rgsaddle_sha"]}`',
            f'- rgmin: `{metadata["rgmin_sha"]}`',
            f'- LAMMPS: `{metadata["lammps_tag"]}` (`{metadata["lammps_sha"]}`)',
            f'- PAFI driver: `{metadata["pafi_tag"]}` (`{metadata["pafi_sha"]}`), package {metadata["pafi_version"]}',
            f'- Potential: {metadata["potential"]}',
            f'- Build: {metadata["build"]}',
            "",
            "The straight case stretches a symmetric Morse dimer from r=1.0 to r=1.4. The curved case rotates the equilibrium dimer through a quarter circle while preserving its centre of mass. Both are finite two-atom systems in a box wider than twice the cutoff, so no periodic image enters the interaction. The rgpot pair kernel and LAMMPS `morse/omp` use D=1, rho=6, r_e=1, and cutoff=4.",
            "",
            "Plane counts are explicit in every cell: 9 and 17 for the straight path, 17 and 33 for the curved path. The zero-temperature tolerance remains 0.01. Rejections are retained in the hit-rate table and are never rerun.",
            "",
            "The matched setting was declared before collection: dt=0.002, friction=1.0, identical burn-in and production counts, and the same seeds, path nodes, masses, potential constants, and plane counts. Other settings use each implementation's documented defaults. Runs are release builds, fixed to the declared CPU sets, with one and four requested threads. rgflux currently evaluates this scalar pair surface on one effective thread; that limitation is recorded rather than hidden.",
            "",
            "The reported LAMMPS force-call count covers thermalization and production integration steps. Setup `run 0` evaluations are not exposed by the reference driver. rgflux reports measured potential calls and separates wall time accumulated inside the potential from integrator overhead.",
            "",
            "## Reproduction",
            "",
            "Build and run commands are documented in `bench/pafi/README.md`. Raw logs are intentionally excluded.",
        ]
    )
    args.output_markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
