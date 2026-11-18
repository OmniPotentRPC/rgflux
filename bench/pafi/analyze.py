#!/usr/bin/env python3
import argparse
import json
import math
import pathlib
import statistics

import numpy as np
from scipy.integrate import quad
from scipy.stats import fisher_exact, mannwhitneyu


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=pathlib.Path, required=True)
    parser.add_argument("--vacancy-raw-dir", type=pathlib.Path, required=True)
    parser.add_argument("--vacancy-path", type=pathlib.Path, required=True)
    parser.add_argument(
        "--vacancy-static-reference", type=pathlib.Path, required=True
    )
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


def profile_summary(records, target=None):
    if not records:
        return {"median": [], "standard_error": [], "median_error": []}
    profiles = np.asarray([record["free_energy"] for record in records], dtype=float)
    summary = {
        "median": np.median(profiles, axis=0).tolist(),
        "standard_error": profile_sem(records),
    }
    if target is not None:
        summary["median_error"] = (
            np.median(profiles - np.asarray(target), axis=0).tolist()
        )
    return summary


def maximum_profile_difference(left, right):
    return max(abs(a - b) for a, b in zip(left["free_energy"], right["free_energy"]))


def morse_energy(distance):
    exponential = math.exp(6.0 * (1.0 - distance))
    return exponential * (exponential - 2.0)


def exact_dimer_profile(case_name, planes, temperature):
    if case_name == "curved":
        return [0.0] * planes
    radii = np.linspace(1.0, 1.4, planes)
    partition = [
        quad(
            lambda radius: radius
            * math.exp(-morse_energy(radius) / temperature),
            lower,
            4.0,
            epsabs=1.0e-12,
            epsrel=1.0e-12,
        )[0]
        for lower in radii
    ]
    reference = partition[0]
    return [-temperature * math.log(value / reference) for value in partition]


def profile_error(record, exact):
    return max(abs(value - target) for value, target in zip(record["free_energy"], exact))


def vacancy_cells(raw_dir, vacancy_path, vacancy_static):
    manifest = json.loads((raw_dir / "manifest.json").read_text())
    entries = [
        (item, json.loads((raw_dir / item["record"]).read_text()))
        for item in manifest
    ]
    native_minimum = vacancy_static["rgflux"]["minimum_energy"]
    native_saddle = vacancy_static["rgflux"]["saddle_energy"]
    reference_minimum = vacancy_static["lammps"]["minimum_energy"]
    reference_saddle = vacancy_static["lammps"]["saddle_energy"]
    keys = sorted(
        {
            (
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
                item["planes"],
                item["temperature"],
                item["samples"],
                item["threads"],
            )
            == key
        ]
        all_by_engine = {
            engine: [
                (item, record)
                for item, record in selected
                if item["engine"] == engine
            ]
            for engine in ("rgflux", "lammps-pafi")
        }
        good = {
            engine: [
                (item, record)
                for item, record in all_by_engine[engine]
                if record.get("status") == "ok"
            ]
            for engine in ("rgflux", "lammps-pafi")
        }
        native = [record for _, record in good["rgflux"]]
        reference = [record for _, record in good["lammps-pafi"]]
        native_by_seed = {item["seed"]: record for item, record in good["rgflux"]}
        reference_by_seed = {
            item["seed"]: record for item, record in good["lammps-pafi"]
        }
        paired = [
            (native_by_seed[seed], reference_by_seed[seed])
            for seed in sorted(set(native_by_seed) & set(reference_by_seed))
        ]
        native_successes = len(native)
        reference_successes = len(reference)
        native_failures = len(all_by_engine["rgflux"]) - native_successes
        reference_failures = len(all_by_engine["lammps-pafi"]) - reference_successes
        reasons = {}
        for _, record in selected:
            if record.get("status") in ("rejected", "process-error"):
                reason = record.get("error", "process-error")
                reasons[reason] = reasons.get(reason, 0) + 1
        native_wall = [record["wall_seconds"] for record in native]
        reference_wall = [record["wall_seconds"] for record in reference]
        native_barriers = [record["barrier"] for record in native]
        reference_barriers = [record["barrier"] for record in reference]
        profile_differences = [
            maximum_profile_difference(left, right) for left, right in paired
        ]
        barrier_differences = [
            abs(left["barrier"] - right["barrier"]) for left, right in paired
        ]
        native_profile_sem = profile_sem(native) if native else []
        reference_profile_sem = profile_sem(reference) if reference else []
        combined_sem = [
            math.hypot(left, right)
            for left, right in zip(native_profile_sem, reference_profile_sem)
            if left is not None and right is not None
        ]
        native_aa = [
            maximum_profile_difference(native[index], native[index + 1])
            for index in range(0, len(native) - 1, 2)
        ]
        reference_aa = [
            maximum_profile_difference(reference[index], reference[index + 1])
            for index in range(0, len(reference) - 1, 2)
        ]
        cells.append(
            {
                "planes": key[0],
                "temperature": key[1],
                "samples": key[2],
                "threads": key[3],
                "successes": {
                    "rgflux": native_successes,
                    "lammps_pafi": reference_successes,
                },
                "failures": {
                    "rgflux": native_failures,
                    "lammps_pafi": reference_failures,
                },
                "rejection_reasons": reasons,
                "hit_rate_fisher_p": float(
                    fisher_exact(
                        [
                            [native_successes, native_failures],
                            [reference_successes, reference_failures],
                        ]
                    ).pvalue
                ),
                "max_absolute_profile_difference": distribution(profile_differences),
                "max_combined_profile_sem": max(combined_sem)
                if combined_sem
                else None,
                "reaction_fraction": np.linspace(0.0, 1.0, key[0]).tolist(),
                "sampled_profiles": {
                    "rgflux": profile_summary(native),
                    "lammps_pafi": profile_summary(reference),
                },
                "barrier": {
                    "rgflux": distribution(native_barriers),
                    "rgflux_sem": sem(native_barriers),
                    "lammps_pafi": distribution(reference_barriers),
                    "lammps_pafi_sem": sem(reference_barriers),
                    "absolute_difference": distribution(barrier_differences),
                    "mann_whitney_p": float(
                        mannwhitneyu(native_barriers, reference_barriers).pvalue
                    )
                    if native_barriers and reference_barriers
                    else None,
                },
                "wall_seconds": {
                    "rgflux": distribution(native_wall),
                    "lammps_pafi": distribution(reference_wall),
                    "mann_whitney_p": float(
                        mannwhitneyu(native_wall, reference_wall).pvalue
                    )
                    if native_wall and reference_wall
                    else None,
                },
                "force_calls_per_plane": {
                    "rgflux": distribution(
                        [record["force_calls"] / key[0] for record in native]
                    ),
                    "lammps_pafi": distribution(
                        [record["force_calls"] / key[0] for record in reference]
                    ),
                },
                "rgflux_potential_seconds_per_plane": distribution(
                    [record["potential_seconds"] / key[0] for record in native]
                ),
                "rgflux_overhead_seconds_per_plane": distribution(
                    [record["overhead_seconds"] / key[0] for record in native]
                ),
                "aa_profile_noise": {
                    "rgflux": distribution(native_aa),
                    "lammps_pafi": distribution(reference_aa),
                },
                "static_energy_absolute_difference": {
                    "minimum": distribution(
                        [abs(native_minimum - reference_minimum)]
                    ),
                    "saddle": distribution(
                        [abs(native_saddle - reference_saddle)]
                    ),
                },
            }
        )
    return entries, cells


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
    vacancy_path = json.loads(args.vacancy_path.read_text())
    vacancy_static = json.loads(args.vacancy_static_reference.read_text())
    vacancy_entries, vacancy = vacancy_cells(
        args.vacancy_raw_dir, vacancy_path, vacancy_static
    )
    vacancy_process_failures = sum(
        record.get("status") == "process-error" for _, record in vacancy_entries
    )
    vacancy_rejections = {}
    for _, record in vacancy_entries:
        if record.get("status") == "rejected":
            reason = record["error"]
            vacancy_rejections[reason] = vacancy_rejections.get(reason, 0) + 1
    manifest = json.loads((args.raw_dir / "manifest.json").read_text())
    entries = []
    for item in manifest:
        record = json.loads((args.raw_dir / item["record"]).read_text())
        entries.append((item, record))
    rejection_reasons = {}
    process_failures = 0
    for _, record in entries:
        if record.get("status") == "rejected":
            reason = record["error"]
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
        if record.get("status") == "process-error":
            process_failures += 1

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
        cell_rejections = {}
        for _, record in selected:
            if record.get("status") == "rejected":
                reason = record["error"]
                cell_rejections[reason] = cell_rejections.get(reason, 0) + 1
        exact_profile = exact_dimer_profile(key[0], key[1], key[2])
        exact_barrier = max(exact_profile)
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
            "rejection_reasons": cell_rejections,
            "hit_rate_fisher_p": float(hit_rate_p),
            "exact_reference": {
                "path_coordinate": (
                    np.linspace(1.0, 1.4, key[1]).tolist()
                    if key[0] == "straight"
                    else np.linspace(0.0, 0.5 * math.pi, key[1]).tolist()
                ),
                "profile": exact_profile,
                "barrier": exact_barrier,
                "sampled_profiles": {
                    "rgflux": profile_summary(native, exact_profile),
                    "lammps_pafi": profile_summary(reference, exact_profile),
                },
                "max_absolute_profile_error": {
                    "rgflux": distribution(
                        [profile_error(record, exact_profile) for record in native]
                    ),
                    "lammps_pafi": distribution(
                        [profile_error(record, exact_profile) for record in reference]
                    ),
                },
                "absolute_barrier_error": {
                    "rgflux": distribution(
                        [abs(record["barrier"] - exact_barrier) for record in native]
                    ),
                    "lammps_pafi": distribution(
                        [
                            abs(record["barrier"] - exact_barrier)
                            for record in reference
                        ]
                    ),
                },
            },
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
            "exact_dimer_reference": {
                "basin_cutoff": 4.0,
                "straight_partition": "2*pi*integral_r^4 R*exp(-U(R)/T)dR",
                "curved_profile": "zero by rotational invariance",
            },
        },
        "execution_summary": {
            "planned": len(entries) + len(vacancy_entries),
            "process_failures": {
                "dimer": process_failures,
                "fcc_vacancy": vacancy_process_failures,
            },
            "rejections": {
                "dimer": rejection_reasons,
                "fcc_vacancy": vacancy_rejections,
            },
        },
        "cells": cells,
        "fcc_vacancy": {
            "atom_count": len(vacancy_path["atomic_numbers"]),
            "neb": {
                "generator_sha": vacancy_path["generator_sha"],
                "images": len(vacancy_path["images"]),
                "iterations": vacancy_path["neb_iterations"],
                "max_force": vacancy_path["neb_max_force"],
            },
            "static_energies": {
                "rgflux": vacancy_static["rgflux"],
                "lammps": vacancy_static["lammps"],
            },
            "cells": vacancy,
        },
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
    lines.append(
        f'All {len(entries)} planned dimer executions completed with {process_failures} process failures. '
        f'rgflux retained {sum(rejection_reasons.values())} numerical or quadrature rejections; '
        "the reference accepted every execution."
    )
    lines.append(
        f'The vacancy campaign adds {len(vacancy_entries)} executions with '
        f'{vacancy_process_failures} process failures and '
        f'{sum(vacancy_rejections.values())} retained rgflux rejections.'
    )
    lines.extend(
        [
            "",
            "- Curved-path agreement improves strongly from 17 to 33 planes. At 33 planes and 256 samples, median maximum profile differences are below 3.7e-6 in every temperature and thread cell.",
            "- The accepted 17-plane straight profiles differ by 0.029 to 0.062 at 256 samples, exceeding the combined profile standard errors. This cell does not establish profile equivalence.",
            "- The exact interacting-basin dimer reference shows substantial absolute bias in both straight-path implementations. At T=0.1 and 0.2 rgflux is closer, so the temperature-growing pairwise discrepancy is carried by the LAMMPS side; it is not a units, Boltzmann-constant, Jacobian, plane-spacing, or tangent convention difference.",
            "- The 191-atom vacancy profiles do not agree: median maximum differences are 0.54 to 0.58 eV, far above combined standard errors and A/A noise. rgflux barriers are about 0.10 to 0.12 eV lower even though independent static minimum and saddle energies agree within 1.0e-11 eV. Increasing from 17 to 33 planes does not remove the discrepancy.",
            "- Every 9-plane straight run is rejected by the unchanged zero-temperature quadrature tolerance. Curved runs additionally expose a direct-exponential overflow in reactant-partition evaluation.",
            "- The scalar native surface has no four-thread speedup. Four OpenMP threads slow the two-atom reference case, as expected when scheduling dominates potential work.",
        ]
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
            "### Error against the exact constrained dimer reference",
            "",
            "| case | planes | T | threads | exact barrier | rgflux max error | reference max error | rgflux barrier error | reference barrier error |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in headline:
        exact = cell["exact_reference"]
        lines.append(
            "| {case} | {planes} | {temperature:g} | {threads} | {barrier:.6g} | "
            "{native_profile} | {reference_profile} | {native_barrier} | "
            "{reference_barrier} |".format(
                case=cell["case"],
                planes=cell["planes"],
                temperature=cell["temperature"],
                threads=cell["threads"],
                barrier=exact["barrier"],
                native_profile=format_distribution(
                    exact["max_absolute_profile_error"]["rgflux"]
                ),
                reference_profile=format_distribution(
                    exact["max_absolute_profile_error"]["lammps_pafi"]
                ),
                native_barrier=format_distribution(
                    exact["absolute_barrier_error"]["rgflux"]
                ),
                reference_barrier=format_distribution(
                    exact["absolute_barrier_error"]["lammps_pafi"]
                ),
            )
        )
    lines.extend(
        [
            "",
            "### Cost and force calls at the largest sample count",
            "",
            "| case | planes | T | threads | rgflux calls/plane | reference calls/plane | rgflux overhead/plane (s) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in headline:
        lines.append(
            "| {case} | {planes} | {temperature:g} | {threads} | {native} | "
            "{reference} | {overhead} |".format(
                case=cell["case"],
                planes=cell["planes"],
                temperature=cell["temperature"],
                threads=cell["threads"],
                native=format_distribution(cell["force_calls_per_plane"]["rgflux"]),
                reference=format_distribution(
                    cell["force_calls_per_plane"]["lammps_pafi"]
                ),
                overhead=format_distribution(
                    cell["rgflux_overhead_seconds_per_plane"]
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
    quadrature_reason = (
        "invalid configuration: PAFI zero-temperature path-force quadrature exceeds tolerance"
    )
    partition_reason = (
        "invalid configuration: PAFI reactant partition must be finite and positive"
    )
    lines.extend(
        [
            "",
            "## Per-cell acceptance",
            "",
            "| case | planes | T | threads | samples | rgflux hits | reference hits | quadrature rejects | partition rejects | Fisher p |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in cells:
        lines.append(
            "| {case} | {planes} | {temperature:g} | {threads} | {samples} | "
            "{native}/12 | {reference}/12 | {quadrature} | {partition} | {p:.3g} |".format(
                case=cell["case"],
                planes=cell["planes"],
                temperature=cell["temperature"],
                threads=cell["threads"],
                samples=cell["samples"],
                native=cell["successes"]["rgflux"],
                reference=cell["successes"]["lammps_pafi"],
                quadrature=cell["rejection_reasons"].get(quadrature_reason, 0),
                partition=cell["rejection_reasons"].get(partition_reason, 0),
                p=cell["hit_rate_fisher_p"],
            )
        )
    lines.extend(
        [
            "",
            "## Periodic FCC vacancy hop",
            "",
            f'The pathway contains {len(vacancy_path["atomic_numbers"])} atoms and '
            f'{len(vacancy_path["images"])} NEB images. The pinned rgsaddle run '
            f'converged in {vacancy_path["neb_iterations"]} iterations at maximum '
            f'force {vacancy_path["neb_max_force"]:.6g}.',
            "",
            "| planes | T (eV) | threads | hits rgflux/reference | max |dF| | combined profile SE | A/A rgflux/reference | barrier rgflux (eV) | barrier reference (eV) | barrier SE rgflux/reference | wall rgflux/reference (s) | calls/plane rgflux/reference | potential/overhead per plane (s) | Fisher p |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in vacancy:
        lines.append(
            "| {planes} | {temperature:g} | {threads} | {native_hits}/"
            "{reference_hits} | {profile} | {combined_se} | {native_aa}/"
            "{reference_aa} | {native_barrier} | "
            "{reference_barrier} | {native_se}/{reference_se} | "
            "{native_wall}/{reference_wall} | {native_calls}/{reference_calls} | "
            "{potential}/{overhead} | {fisher:.3g} |".format(
                planes=cell["planes"],
                temperature=cell["temperature"],
                threads=cell["threads"],
                native_hits=cell["successes"]["rgflux"],
                reference_hits=cell["successes"]["lammps_pafi"],
                profile=format_distribution(cell["max_absolute_profile_difference"]),
                combined_se=(
                    f'{cell["max_combined_profile_sem"]:.4g}'
                    if cell["max_combined_profile_sem"] is not None
                    else "n/a"
                ),
                native_aa=format_distribution(
                    cell["aa_profile_noise"]["rgflux"]
                ),
                reference_aa=format_distribution(
                    cell["aa_profile_noise"]["lammps_pafi"]
                ),
                native_barrier=format_distribution(cell["barrier"]["rgflux"]),
                reference_barrier=format_distribution(
                    cell["barrier"]["lammps_pafi"]
                ),
                native_se=(
                    f'{cell["barrier"]["rgflux_sem"]:.4g}'
                    if cell["barrier"]["rgflux_sem"] is not None
                    else "n/a"
                ),
                reference_se=(
                    f'{cell["barrier"]["lammps_pafi_sem"]:.4g}'
                    if cell["barrier"]["lammps_pafi_sem"] is not None
                    else "n/a"
                ),
                native_wall=format_distribution(cell["wall_seconds"]["rgflux"]),
                reference_wall=format_distribution(
                    cell["wall_seconds"]["lammps_pafi"]
                ),
                native_calls=format_distribution(
                    cell["force_calls_per_plane"]["rgflux"]
                ),
                reference_calls=format_distribution(
                    cell["force_calls_per_plane"]["lammps_pafi"]
                ),
                potential=format_distribution(
                    cell["rgflux_potential_seconds_per_plane"]
                ),
                overhead=format_distribution(
                    cell["rgflux_overhead_seconds_per_plane"]
                ),
                fisher=cell["hit_rate_fisher_p"],
            )
        )
    lines.extend(
        [
            "",
            "Static path energies agree to "
            f'{abs(vacancy_static["rgflux"]["minimum_energy"] - vacancy_static["lammps"]["minimum_energy"]):.4g} eV '
            "at the minimum and "
            f'{abs(vacancy_static["rgflux"]["saddle_energy"] - vacancy_static["lammps"]["saddle_energy"]):.4g} eV '
            "at the saddle. Per-cell rejection reasons and Mann-Whitney p-values are retained in `results.json`.",
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
            "The straight case stretches a symmetric Morse dimer from r=1.0 to r=1.4. The curved case rotates the equilibrium dimer through a quarter circle while preserving its centre of mass. No sampled configuration approaches the cutoff or a periodic image. The native dimer is unconfined, while the comparator uses a box wider than twice the cutoff; the common exact reference is therefore the explicitly declared interacting metastable basin. The rgpot pair kernel and LAMMPS `morse/omp` use D=1, rho=6, r_e=1, and cutoff=4.",
            "",
            "Plane counts are explicit in every cell: 9 and 17 for the straight path, 17 and 33 for the curved path. The zero-temperature tolerance remains 0.01. Rejections are retained in the hit-rate table and are never rerun.",
            "",
            "The matched setting was declared before collection: dt=0.002, friction=1.0, identical burn-in and production counts, and the same seeds, path nodes, masses, potential constants, and plane counts. Other settings use each implementation's documented defaults. Runs are release builds, fixed to the declared CPU sets, with one and four requested threads. rgflux currently evaluates this scalar pair surface on one effective thread; that limitation is recorded rather than hidden.",
            "",
            "The reported LAMMPS force-call count covers thermalization and production integration steps. Setup `run 0` evaluations are not exposed by the reference driver. rgflux reports measured potential calls and separates wall time accumulated inside the potential from integrator overhead.",
            "",
            "Static minimum energies agree exactly in accepted paired runs. The largest median saddle-energy discrepancy is below 3e-11. At 256 samples, rgflux reports 386 force calls per accepted plane and the reference reports 384 integration-step calls per plane; the two extra native calls cover path/setup evaluation.",
            "",
            "## Reproduction of the rgflux numerical rejection",
            "",
            "A fixed curved-path seed can make direct evaluation of `exp(-F/kBT)` overflow while integrating the reactant partition. `repro_partition_overflow.py` checks the retained `PAFI reactant partition must be finite and positive` rejection. It requires the release native executable through `--native-bin`. The benchmark does not alter rgflux or retry the rejected run.",
            "",
            "## Reproduction",
            "",
            "Build and run commands are documented in `bench/pafi/README.md`. Raw logs are intentionally excluded.",
        ]
    )
    args.output_markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
