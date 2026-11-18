#!/usr/bin/env python3
"""Fail on a git dependency not pinned by full rev, or a second copy of our crates."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from urllib.parse import parse_qsl

ROOT = Path(__file__).resolve().parents[1]
OWN = frozenset(
    {
        "dlpk",
        "eindir-core",
        "linkcell",
        "minimage",
        "readcon-core",
        "readcon-db",
        "rgflux",
        "rgmin",
        "rgpot-core",
        "rgsaddle",
    }
)
SHA = re.compile(r"[0-9a-f]{40}")
LOCKED_GIT = re.compile(
    r"git\+(?P<url>[^?#]+)\?rev=(?P<rev>[0-9a-f]{40})#(?P<precise>[0-9a-f]{40})"
)
TREE_LINE = re.compile(r"(?P<name>\S+) v(?P<version>\S+)(?P<rest>.*)")
TREE_ARGS = [
    "tree",
    "-d",
    "--all-features",
    "--target",
    "all",
    "-e",
    "normal,build,dev",
]


def cargo(args: list[str], cwd: Path) -> str:
    proc = subprocess.run(["cargo", *args], cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(
            f"cargo {' '.join(args)} failed in {cwd}:\n{proc.stderr}"
        )
    return proc.stdout


def lock_packages(where: Path) -> list[dict]:
    with open(where / "Cargo.lock", "rb") as fh:
        return tomllib.load(fh).get("package", [])


def check_lock(where: Path, label: str) -> list[str]:
    errors = []
    for pkg in lock_packages(where):
        source = pkg.get("source", "")
        if not source.startswith("git+"):
            continue
        m = LOCKED_GIT.fullmatch(source)
        if m is None or m["rev"] != m["precise"]:
            errors.append(
                f"{label}/Cargo.lock: {pkg['name']} {pkg['version']} is locked from "
                f"{source}, not a full rev"
            )
    return errors


def check_manifests(where: Path, label: str) -> list[str]:
    errors = []
    meta = json.loads(cargo(["metadata", "--no-deps", "--format-version", "1"], where))
    locked = {
        (pkg["name"], pkg["source"].split("#", 1)[0]): pkg["version"]
        for pkg in lock_packages(where)
        if pkg.get("source", "").startswith("git+")
    }
    seen: dict[str, set[tuple[str, str]]] = {}
    for member in meta["packages"]:
        manifest = Path(member["manifest_path"]).relative_to(ROOT)
        for dep in member["dependencies"]:
            source = dep.get("source") or ""
            if not source.startswith("git+"):
                continue
            name = dep["name"]
            what = f"{manifest}: {name}"
            url, _, query = source.partition("?")
            reference = dict(parse_qsl(query))
            if set(reference) != {"rev"} or not SHA.fullmatch(reference["rev"]):
                shown = query or "the default branch"
                errors.append(f"{what} follows {shown}; pin rev to a full 40-hex commit")
            req = dep["req"]
            version = locked.get((name, source))
            if version is None:
                errors.append(f"{what} from {source} is not in {label}/Cargo.lock")
            elif req != f"={version}":
                errors.append(
                    f"{what} asks for {req!r}, but the crate at that rev is {version}; "
                    f"use '={version}'"
                )
            seen.setdefault(name, set()).add((url, reference.get("rev", "")))
    for name, sources in sorted(seen.items()):
        if len(sources) > 1:
            listed = ", ".join(f"{url}@{rev or '?'}" for url, rev in sorted(sources))
            errors.append(f"{label}: {name} is taken from more than one git source: {listed}")
    return errors


def tree_copies(where: Path, extra: list[str]) -> dict[str, set[tuple[str, str]]]:
    out = cargo([*TREE_ARGS, *extra, "--depth", "0", "--prefix", "none", "--format", "{p}"], where)
    copies: dict[str, set[tuple[str, str]]] = {}
    for line in out.splitlines():
        m = TREE_LINE.fullmatch(line.strip())
        if m is None:
            continue
        rest = m["rest"].replace("(*)", "").replace("(proc-macro)", "").strip()
        copies.setdefault(m["name"], set()).add((m["version"], rest.strip("()") or "crates.io"))
    return copies


def ours(name: str, sources: set[tuple[str, str]]) -> bool:
    return name in OWN or any("://" in source for _, source in sources)


def check_duplicates(where: Path, label: str, extra: list[str]) -> list[str]:
    copies = tree_copies(where, extra)
    bad = {name for name, sources in copies.items() if len(sources) > 1 and ours(name, sources)}
    if not bad:
        return []
    full = cargo([*TREE_ARGS, *extra, "--no-dedupe"], where)
    blocks = [b for b in full.split("\n\n") if b.split(" ", 1)[0] in bad]
    detail = "\n\n".join(blocks)
    return [
        f"{label}: cargo tree -d lists more than one copy of {', '.join(sorted(bad))}:\n{detail}"
    ]


def main() -> int:
    errors: list[str] = []
    cargo(["metadata", "--locked", "--format-version", "1"], ROOT)
    errors += check_lock(ROOT, ".")
    errors += check_manifests(ROOT, ".")
    errors += check_duplicates(ROOT, ".", ["--locked"])

    if errors:
        print("dependency pins:\n" + "\n".join(errors), file=sys.stderr)
        return 1
    pins = sorted(
        f"{pkg['name']} {pkg['version']} @ {LOCKED_GIT.fullmatch(pkg['source'])['rev']}"
        for pkg in lock_packages(ROOT)
        if pkg.get("source", "").startswith("git+")
    )
    print("dependency pins ok: " + "; ".join(pins))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
