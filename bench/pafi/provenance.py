#!/usr/bin/env python3
"""Record commits, versions, seeds, host, and commands for a PAFI run."""

import importlib.metadata
import json
import pathlib
import platform
import re
import subprocess
import tomllib


ROOT = pathlib.Path(__file__).resolve().parent
RAW_ROOT = ROOT / "raw"
DIMER_RAW = RAW_ROOT / "dimer"
VACANCY_RAW = RAW_ROOT / "vacancy"
METADATA_PATH = RAW_ROOT / "metadata.json"
VACANCY_PATH = RAW_ROOT / "vacancy-path.json"
VACANCY_STATIC = RAW_ROOT / "vacancy-static.json"
LOCKFILE = ROOT / "Cargo.lock"

NATIVE_CRATES = {
    "ndarray": "0.17.2",
    "pyo3": "0.27.2",
    "serde": "1.0.229",
    "serde_json": "1.0.154",
}

PYTHON_PACKAGES = ("mpi4py", "numpy", "pandas", "pafi", "rgpot", "scipy")

POTENTIAL = (
    "Dimer: Morse D=1, rho=6, r_e=1, cutoff=4. "
    "Vacancy: shifted Morse D_e=0.7102 eV, a=1.6047 1/Angstrom, "
    "r_e=2.897 Angstrom, cutoff=5.5 Angstrom; "
    "rgpot 3.7.0 and LAMMPS morse/omp"
)


def add_checkout_arguments(parser):
    parser.add_argument("--rgflux-source", type=pathlib.Path, required=True)
    parser.add_argument("--rgpot-source", type=pathlib.Path, required=True)
    parser.add_argument("--rgsaddle-source", type=pathlib.Path, required=True)
    parser.add_argument("--lammps-source", type=pathlib.Path)
    parser.add_argument("--pafi-source", type=pathlib.Path)


def add_source_arguments(parser):
    add_checkout_arguments(parser)
    parser.add_argument("--metadata", type=pathlib.Path, default=METADATA_PATH)


def sources_from_args(args):
    return {
        "rgflux": args.rgflux_source,
        "rgpot": args.rgpot_source,
        "rgsaddle": args.rgsaddle_source,
        "lammps": args.lammps_source,
        "pafi": args.pafi_source,
    }


def git_commit(path):
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def git_exact_tag(path):
    completed = subprocess.run(
        ["git", "-C", str(path), "describe", "--tags", "--exact-match"],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    text = completed.stdout.strip()
    return text or None


def rgmin_commit(rgflux_source):
    manifest = pathlib.Path(rgflux_source) / "Cargo.toml"
    if not manifest.is_file():
        raise RuntimeError(f"rgflux Cargo.toml is missing: {manifest}")
    parsed = tomllib.loads(manifest.read_text())
    dependency = parsed.get("dependencies", {}).get("rgmin")
    if not isinstance(dependency, dict) or "rev" not in dependency:
        raise RuntimeError("rgflux Cargo.toml does not pin rgmin with a rev")
    rev = dependency["rev"]
    if re.fullmatch(r"[0-9a-f]{40}", rev) is None:
        raise RuntimeError("rgmin rev in rgflux Cargo.toml is not 40 hex digits")
    return rev


def installed_packages():
    found = {}
    for name in PYTHON_PACKAGES:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return found


def tool_line(command):
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        return None
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    return lines[0]


def build_description():
    parts = []
    for command in (["rustc", "--version"], ["cargo", "--version"]):
        line = tool_line(command)
        if line:
            parts.append(line)
    if not parts:
        return "native binary supplied without rustc or cargo on PATH"
    return "; ".join(parts)


def command_text(command):
    return [str(part) for part in command]


def extend_source_command(command, args):
    command.extend(
        [
            "--rgflux-source",
            str(args.rgflux_source),
            "--rgpot-source",
            str(args.rgpot_source),
            "--rgsaddle-source",
            str(args.rgsaddle_source),
        ]
    )
    if args.lammps_source is not None:
        command.extend(["--lammps-source", str(args.lammps_source)])
    if args.pafi_source is not None:
        command.extend(["--pafi-source", str(args.pafi_source)])
    return command


def source_block(sources):
    commits = {
        "rgflux": git_commit(sources["rgflux"]),
        "rgpot": git_commit(sources["rgpot"]),
        "rgsaddle": git_commit(sources["rgsaddle"]),
    }
    packages = installed_packages()
    block = {
        "build": build_description(),
        "host": {
            "hostname": platform.node(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "native_harness_crates": dict(NATIVE_CRATES),
        "packages": packages,
        "potential": POTENTIAL,
        "python": platform.python_version(),
        "rgflux_sha": commits["rgflux"],
        "rgmin_sha": rgmin_commit(sources["rgflux"]),
        "rgpot_sha": commits["rgpot"],
        "rgsaddle_sha": commits["rgsaddle"],
        "source_commits": commits,
        "versions": {
            "native_harness_crates": dict(NATIVE_CRATES),
            "packages": packages,
            "python": platform.python_version(),
        },
    }
    if sources.get("lammps") is not None:
        commits["lammps"] = git_commit(sources["lammps"])
        block["lammps_sha"] = commits["lammps"]
        tag = git_exact_tag(sources["lammps"])
        if tag is not None:
            block["lammps_tag"] = tag
    if sources.get("pafi") is not None:
        commits["pafi"] = git_commit(sources["pafi"])
        block["pafi_sha"] = commits["pafi"]
        tag = git_exact_tag(sources["pafi"])
        if tag is not None:
            block["pafi_tag"] = tag
    if "pafi" in packages:
        block["pafi_version"] = packages["pafi"]
    return block


def write_json(path, payload):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def cpu_affinity(one_core_set, high_threads, high_core_set):
    recorded = {
        "high_thread_cores": high_core_set,
        "high_threads": high_threads,
        "one_thread": one_core_set,
    }
    if high_threads == 4:
        recorded["four_threads"] = high_core_set
    return recorded


def merge_campaign_metadata(path, campaign, seeds, command_lines, sources, affinity):
    path = pathlib.Path(path)
    current = {}
    if path.is_file():
        current = json.loads(path.read_text())
        if not isinstance(current, dict):
            raise RuntimeError(f"metadata.json must be an object: {path}")
    current.update(source_block(sources))
    current["cpu_affinity"] = affinity
    stored_seeds = current.get("seeds")
    if not isinstance(stored_seeds, dict):
        stored_seeds = {}
    stored_seeds[campaign] = list(seeds)
    current["seeds"] = stored_seeds
    stored_commands = current.get("command_lines")
    if not isinstance(stored_commands, dict):
        stored_commands = {}
    stored_commands[campaign] = [command_text(command) for command in command_lines]
    current["command_lines"] = stored_commands
    write_json(path, current)
    return current


def write_run_metadata(path, seed, command, sources):
    block = source_block(sources)
    block["command_lines"] = [command_text(command)]
    block["seeds"] = [seed]
    write_json(path, block)
