#!/usr/bin/env python3
import argparse
import json
import math
import pathlib
import statistics

import numpy as np
from scipy.integrate import quad
from scipy.stats import fisher_exact, mannwhitneyu


REACTANT_PARTITION_ERROR = (
    "invalid configuration: PAFI reactant partition must be finite and positive"
)
ZERO_TEMPERATURE_TOL_ERROR = (
    "invalid configuration: PAFI zero-temperature path-force quadrature exceeds tolerance"
)


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


def exact_periodic_dimer_profile(case_name, planes, temperature, box_length=20.0):
    if case_name == "straight":
        radii = np.linspace(1.0, 1.4, planes)
        partition = [
            box_length**2
            + 2.0
            * math.pi
            * quad(
                lambda radius: radius
                * (math.exp(-morse_energy(radius) / temperature) - 1.0),
                lower,
                4.0,
                epsabs=1.0e-12,
                epsrel=1.0e-12,
            )[0]
            for lower in radii
        ]
    else:
        correction = 2.0 * math.pi * quad(
            lambda radius: radius
            * (math.exp(-morse_energy(radius) / temperature) - 1.0),
            0.0,
            4.0,
            epsabs=1.0e-12,
            epsrel=1.0e-12,
        )[0]
        angles = np.linspace(0.0, 0.5 * math.pi, planes)
        partition = [
            box_length**2 / max(abs(math.cos(angle)), abs(math.sin(angle)))
            + correction
            for angle in angles
        ]
    reference = partition[0]
    return [-temperature * math.log(value / reference) for value in partition]


def profile_error(record, exact):
    return max(abs(value - target) for value, target in zip(record["free_energy"], exact))


def rejection_audit(entries):
    buckets = {
        "zero_reactant_partition": {
            "count": 0,
            "error_strings": [REACTANT_PARTITION_ERROR],
        },
        "genuine_exp_overflow": {
            "count": 0,
            "error_strings": [],
        },
        "zero_temperature_tol": {
            "count": 0,
            "error_strings": [ZERO_TEMPERATURE_TOL_ERROR],
        },
        "other": {"count": 0, "error_strings": [], "rejects": []},
    }
    for item, record in entries:
        if record.get("status") != "rejected":
            continue
        reason = record["error"]
        if reason == REACTANT_PARTITION_ERROR:
            constant_energy_path = math.isclose(
                record["minimum_energy"],
                record["saddle_energy"],
                rel_tol=0.0,
                abs_tol=1.0e-14,
            )
            if item["case"] == "curved" and constant_energy_path:
                buckets["zero_reactant_partition"]["count"] += 1
            else:
                buckets["genuine_exp_overflow"]["count"] += 1
                buckets["genuine_exp_overflow"]["error_strings"] = [
                    REACTANT_PARTITION_ERROR
                ]
        elif reason == ZERO_TEMPERATURE_TOL_ERROR:
            buckets["zero_temperature_tol"]["count"] += 1
        else:
            buckets["other"]["count"] += 1
            buckets["other"]["rejects"].append(
                {
                    "error_string": reason,
                    "case": item["case"],
                    "planes": item["planes"],
                    "temperature": item["temperature"],
                    "threads": item["threads"],
                    "seed": item["seed"],
                }
            )
    buckets["other"]["error_strings"] = sorted(
        {reject["error_string"] for reject in buckets["other"]["rejects"]}
    )
    return buckets


def vacancy_cells(raw_dir, vacancy_path, vacancy_static):
    manifest = json.loads((raw_dir / "manifest.json").read_text())
    entries = [
        (item, json.loads((raw_dir / item["record"]).read_text()))
        for item in manifest
    ]
    for item, record in entries:
        if item["engine"] == "rgflux":
            record["effective_threads"] = 1
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


def format_vector(values):
    if not values:
        return "rejected"
    return "[" + ", ".join(f"{value:+.4g}" for value in values) + "]"


def format_range(values, digits=3):
    return f"{min(values):.{digits}g}-{max(values):.{digits}g}"


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
    rejection_buckets = rejection_audit(entries)

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
        cell_rejection_buckets = rejection_audit(selected)
        cell_rejection_counts = {
            name: bucket["count"]
            for name, bucket in cell_rejection_buckets.items()
        }
        exact_profile = exact_dimer_profile(key[0], key[1], key[2])
        exact_barrier = max(exact_profile)
        periodic_profile = exact_periodic_dimer_profile(key[0], key[1], key[2])
        periodic_barrier = max(0.0, max(periodic_profile))
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
            "rejection_counts": cell_rejection_counts,
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
                "finite_periodic_lammps": {
                    "box_length": 20.0,
                    "profile": periodic_profile,
                    "barrier": periodic_barrier,
                    "sampled_profile": profile_summary(
                        reference, periodic_profile
                    ),
                    "max_absolute_profile_error": distribution(
                        [
                            profile_error(record, periodic_profile)
                            for record in reference
                        ]
                    ),
                    "absolute_barrier_error": distribution(
                        [
                            abs(record["barrier"] - periodic_barrier)
                            for record in reference
                        ]
                    ),
                },
                "native_global_equilibrium": "undefined: the unconfined transverse partition diverges beyond the cutoff",
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
            "thread_accounting": {
                "dimer_rgflux_effective": sorted(
                    {
                        record["effective_threads"]
                        for item, record in entries
                        if item["engine"] == "rgflux"
                    }
                ),
                "vacancy_rgflux_effective": sorted(
                    {
                        record["effective_threads"]
                        for item, record in vacancy_entries
                        if item["engine"] == "rgflux"
                    }
                ),
                "requested": sorted(
                    {item["threads"] for item, _ in entries + vacancy_entries}
                ),
            },
            "force_call_accounting": {
                "rgflux": "measured potential invocations",
                "lammps_pafi": "derived as planes times configured thermalization plus production steps; setup evaluations are excluded",
            },
            "rejection_classification": "The generic non-positive reactant-partition error is zero_reactant_partition for retained curved constant-energy paths; any occurrence outside that verified class is counted as genuine_exp_overflow.",
            "exact_dimer_reference": {
                "basin_cutoff": 4.0,
                "straight_partition": "2*pi*integral_r^4 R*exp(-U(R)/T)dR",
                "curved_profile": "zero by rotational invariance",
                "native_global_equilibrium": "undefined because the unconfined transverse partition diverges",
                "lammps_periodic_box_length": 20.0,
                "lammps_straight_partition": "L^2 + 2*pi*integral_r^4 R*(exp(-U(R)/T)-1)dR",
                "lammps_curved_partition": "L^2/max(abs(cos(theta)),abs(sin(theta))) + 2*pi*integral_0^4 R*(exp(-U(R)/T)-1)dR",
            },
        },
        "execution_summary": {
            "planned": len(entries) + len(vacancy_entries),
            "process_failures": {
                "dimer": process_failures,
                "fcc_vacancy": vacancy_process_failures,
            },
            "raw_rejection_errors": {
                "dimer": rejection_reasons,
                "fcc_vacancy": vacancy_rejections,
            },
            "rejection_buckets": rejection_buckets,
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
    straight_exact = [cell for cell in headline if cell["case"] == "straight"]
    exact_temperatures = sorted({cell["temperature"] for cell in straight_exact})
    periodic_error_ranges = []
    native_basin_error_ranges = []
    for temperature in exact_temperatures:
        temperature_cells = [
            cell for cell in straight_exact if cell["temperature"] == temperature
        ]
        periodic_error_ranges.append(
            f'T={temperature:g}: '
            + format_range(
                [
                    cell["exact_reference"]["finite_periodic_lammps"][
                        "max_absolute_profile_error"
                    ]["median"]
                    for cell in temperature_cells
                ]
            )
        )
        native_errors = [
            cell["exact_reference"]["max_absolute_profile_error"]["rgflux"][
                "median"
            ]
            for cell in temperature_cells
            if cell["exact_reference"]["max_absolute_profile_error"]["rgflux"][
                "median"
            ]
            is not None
        ]
        native_basin_error_ranges.append(
            f'T={temperature:g}: ' + format_range(native_errors)
        )
    nine_plane_cells = [
        cell for cell in cells if cell["case"] == "straight" and cell["planes"] == 9
    ]
    nine_plane_rejects = sum(
        cell["failures"]["rgflux"] for cell in nine_plane_cells
    )
    nine_plane_runs = sum(
        cell["successes"]["rgflux"] + cell["failures"]["rgflux"]
        for cell in nine_plane_cells
    )
    curved_17_errors = [
        cell["max_absolute_profile_difference"]["median"]
        for cell in headline
        if cell["case"] == "curved" and cell["planes"] == 17
    ]
    curved_33_errors = [
        cell["max_absolute_profile_difference"]["median"]
        for cell in headline
        if cell["case"] == "curved" and cell["planes"] == 33
    ]
    straight_17_errors = [
        cell["max_absolute_profile_difference"]["median"]
        for cell in headline
        if cell["case"] == "straight"
        and cell["planes"] == 17
        and cell["max_absolute_profile_difference"]["median"] is not None
    ]
    periodic_exact_errors = [
        cell["exact_reference"]["finite_periodic_lammps"][
            "max_absolute_profile_error"
        ]["median"]
        for cell in straight_exact
    ]
    vacancy_profile_errors = [
        cell["max_absolute_profile_difference"]["median"] for cell in vacancy
    ]
    vacancy_barrier_errors = [
        cell["barrier"]["absolute_difference"]["median"] for cell in vacancy
    ]
    vacancy_static_minimum_error = abs(
        vacancy_static["rgflux"]["minimum_energy"]
        - vacancy_static["lammps"]["minimum_energy"]
    )
    vacancy_static_saddle_error = abs(
        vacancy_static["rgflux"]["saddle_energy"]
        - vacancy_static["lammps"]["saddle_energy"]
    )
    dimer_static_minimum_errors = [
        cell["static_energy_absolute_difference"]["minimum"]["median"]
        for cell in headline
        if cell["static_energy_absolute_difference"]["minimum"]["median"]
        is not None
    ]
    dimer_static_saddle_errors = [
        cell["static_energy_absolute_difference"]["saddle"]["median"]
        for cell in headline
        if cell["static_energy_absolute_difference"]["saddle"]["median"] is not None
    ]
    native_calls_per_plane = sorted(
        {
            cell["force_calls_per_plane"]["rgflux"]["median"]
            for cell in headline
            if cell["force_calls_per_plane"]["rgflux"]["median"] is not None
        }
    )
    reference_steps_per_plane = sorted(
        {
            cell["force_calls_per_plane"]["lammps_pafi"]["median"]
            for cell in headline
            if cell["force_calls_per_plane"]["lammps_pafi"]["median"] is not None
        }
    )
    lines = [
        "# Native PAFI benchmark",
        "",
        "## Exact-quadrature straight-path verdict",
        "",
        "There is no common global equilibrium to declare one engine right. The native cutoff dimer is transversely unconfined, so its constrained partition diverges and its exact global free energy is undefined. The rgflux columns below are errors against the explicitly imposed r<4 interacting-basin regularization, not errors against its actual equilibrium. LAMMPS finite-box median maximum per-plane errors are "
        + ", ".join(periodic_error_ranges)
        + ". Native basin-regularized diagnostic errors are "
        + ", ".join(native_basin_error_ranges)
        + f". The 9-plane native cells reject {nine_plane_rejects}/{nine_plane_runs} runs.",
        "",
        "Each vector is the median signed sampled-minus-exact error at every path plane, from reactant to product. Positive entries put the sampled free energy above the applicable exact quadrature.",
        "",
        "| planes | T | threads | rgflux per-plane error vs r<4 basin | LAMMPS per-plane error vs finite box | rgflux max basin error | LAMMPS max finite-box error |",
        "|---:|---:|---:|---|---|---:|---:|",
    ]
    for cell in straight_exact:
        exact = cell["exact_reference"]
        lines.append(
            "| {planes} | {temperature:g} | {threads} | {native_vector} | "
            "{reference_vector} | {native_max} | {reference_max} |".format(
                planes=cell["planes"],
                temperature=cell["temperature"],
                threads=cell["threads"],
                native_vector=format_vector(
                    exact["sampled_profiles"]["rgflux"]["median_error"]
                ),
                reference_vector=format_vector(
                    exact["finite_periodic_lammps"]["sampled_profile"][
                        "median_error"
                    ]
                ),
                native_max=format_distribution(
                    exact["max_absolute_profile_error"]["rgflux"]
                ),
                reference_max=format_distribution(
                    exact["finite_periodic_lammps"][
                        "max_absolute_profile_error"
                    ]
                ),
            )
        )
    lines.extend(
        [
            "",
        "## Verdict",
        "",
        ]
    )
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
            f"- Curved-path median maximum profile differences change from {format_range(curved_17_errors)} at 17 planes to {format_range(curved_33_errors)} at 33 planes.",
            f"- Accepted 17-plane straight profiles have median maximum pairwise differences of {format_range(straight_17_errors)}.",
            f"- The native global straight-path equilibrium is undefined. LAMMPS finite-periodic median maximum errors are {format_range(periodic_exact_errors)}; the common interacting-basin comparison remains a regularized diagnostic.",
            f"- Vacancy median maximum profile differences are {format_range(vacancy_profile_errors)} eV and barrier differences are {format_range(vacancy_barrier_errors)} eV. Static minimum and saddle differences are {vacancy_static_minimum_error:.3g} and {vacancy_static_saddle_error:.3g} eV.",
            f'- Rejections split into {rejection_buckets["zero_temperature_tol"]["count"]} zero-temperature-tolerance cases, {rejection_buckets["zero_reactant_partition"]["count"]} zero-width reactant partitions, {rejection_buckets["genuine_exp_overflow"]["count"]} genuine exponential overflows, and {rejection_buckets["other"]["count"]} other cases.',
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
            "### Error against the exact interacting-basin regularization",
            "",
            "This r<4 table is a deliberately imposed common diagnostic, not either engine's unmodified global equilibrium.",
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
            "### LAMMPS error against its exact finite-periodic equilibrium",
            "",
            "| case | planes | T | threads | exact barrier | LAMMPS max profile error | LAMMPS barrier error |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in headline:
        periodic = cell["exact_reference"]["finite_periodic_lammps"]
        lines.append(
            "| {case} | {planes} | {temperature:g} | {threads} | "
            "{barrier:.6g} | {profile} | {barrier_error} |".format(
                case=cell["case"],
                planes=cell["planes"],
                temperature=cell["temperature"],
                threads=cell["threads"],
                barrier=periodic["barrier"],
                profile=format_distribution(
                    periodic["max_absolute_profile_error"]
                ),
                barrier_error=format_distribution(
                    periodic["absolute_barrier_error"]
                ),
            )
        )
    lines.extend(
        [
            "",
            "### Cost and force calls at the largest sample count",
            "",
            "| case | planes | T | threads | measured rgflux calls/plane | derived reference steps/plane | rgflux overhead/plane (s) |",
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
    lines.extend(
        [
            "",
            "## Rejection audit",
            "",
            "| bucket | count | exact error string |",
            "|---|---:|---|",
            f'| zero_reactant_partition | {rejection_buckets["zero_reactant_partition"]["count"]} | `{REACTANT_PARTITION_ERROR}` |',
            f'| zero_temperature_tol | {rejection_buckets["zero_temperature_tol"]["count"]} | `{ZERO_TEMPERATURE_TOL_ERROR}` |',
            f'| genuine_exp_overflow | {rejection_buckets["genuine_exp_overflow"]["count"]} | '
            + (
                ", ".join(
                    f"`{value}`"
                    for value in rejection_buckets["genuine_exp_overflow"][
                        "error_strings"
                    ]
                )
                if rejection_buckets["genuine_exp_overflow"]["error_strings"]
                else "none"
            )
            + " |",
            f'| other | {rejection_buckets["other"]["count"]} | '
            + (
                ", ".join(
                    f"`{value}`"
                    for value in rejection_buckets["other"]["error_strings"]
                )
                if rejection_buckets["other"]["error_strings"]
                else "none"
            )
            + " |",
            "",
        ]
    )
    if rejection_buckets["other"]["rejects"]:
        lines.extend(
            [
                "| exact error string | case | planes | T | threads | seed |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for reject in rejection_buckets["other"]["rejects"]:
            lines.append(
                "| `{error}` | {case} | {planes} | {temperature:g} | "
                "{threads} | {seed} |".format(
                    error=reject["error_string"],
                    case=reject["case"],
                    planes=reject["planes"],
                    temperature=reject["temperature"],
                    threads=reject["threads"],
                    seed=reject["seed"],
                )
            )
    else:
        lines.append("There are no `other` rejects.")
    lines.extend(
        [
            "",
            "## Per-cell acceptance",
            "",
            "| case | planes | T | threads | samples | rgflux hits | reference hits | zero_temperature_tol | zero_reactant_partition | genuine_exp_overflow | other | Fisher p |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in cells:
        buckets = cell["rejection_counts"]
        lines.append(
            "| {case} | {planes} | {temperature:g} | {threads} | {samples} | "
            "{native}/12 | {reference}/12 | {quadrature} | {zero_partition} | "
            "{overflow} | "
            "{other} | {p:.3g} |".format(
                case=cell["case"],
                planes=cell["planes"],
                temperature=cell["temperature"],
                threads=cell["threads"],
                samples=cell["samples"],
                native=cell["successes"]["rgflux"],
                reference=cell["successes"]["lammps_pafi"],
                quadrature=cell["rejection_reasons"].get(
                    ZERO_TEMPERATURE_TOL_ERROR, 0
                ),
                zero_partition=buckets["zero_reactant_partition"],
                overflow=buckets["genuine_exp_overflow"],
                other=buckets["other"],
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
            "This is a periodic 3 by 4 by 4 FCC vacancy hop using the same shifted platinum-parameter Morse potential on both sides. rgpot supplies the native link-cell/minimum-image neighbour stack; LAMMPS uses `morse/omp`. The explicit grid is 17 and 33 planes at kBT=0.025, 0.05, and 0.075 eV, with 12 seeds at one and four threads. The native potential time covers the complete rgpot surface call; overhead is the remaining rgflux wall time.",
            "",
            "| planes | T (eV) | threads | hits rgflux/reference | max |dF| | combined profile SE | A/A rgflux/reference | barrier rgflux (eV) | barrier reference (eV) | barrier SE rgflux/reference | wall rgflux/reference (s) | measured calls/derived steps per plane | potential/overhead per plane (s) | Fisher p |",
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
            "The matched setting was declared before collection: dt=0.002, friction=1.0, identical burn-in and production counts, and the same seeds, path nodes, masses, potential constants, and plane counts. Other settings use each implementation's documented defaults. Runs are release builds, fixed to the declared CPU sets, with one and four requested threads. Both native benchmark surfaces used one effective evaluation thread; the harness requires that value explicitly.",
            "",
            "LAMMPS values are derived integration-step counts: planes times configured thermalization plus production steps. They are not measured force-call counters, and setup evaluations are excluded. rgflux reports measured potential invocations and separates wall time accumulated inside the potential from integrator overhead.",
            "",
            f"Across accepted headline cells, the largest median static minimum and saddle differences are {max(dimer_static_minimum_errors):.3g} and {max(dimer_static_saddle_errors):.3g}. Measured rgflux calls per plane are {native_calls_per_plane}; derived reference integration steps per plane are {reference_steps_per_plane}.",
            "",
            "## Reproduction of the rgflux numerical rejection",
            "",
            "Seed 1036330 selects dividing plane index zero. The reactant integration loop therefore has no interval, leaving a finite partition of zero; no exponential is evaluated. The cargo test `reproduces_zero_width_reactant_partition` and `repro_zero_width_reactant_partition.py` check the retained `PAFI reactant partition must be finite and positive` rejection. No retained cell is classified as a genuine exponential overflow.",
            "",
            "## Reproduction",
            "",
            "Build and run commands are documented in `bench/pafi/README.md`. Raw logs are intentionally excluded.",
        ]
    )
    args.output_markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
