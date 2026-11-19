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
# `lammpc` links liblammpc. The duplicate check runs in the test job, which
# does not install that library, so the tree stays on the library features.
TREE_ARGS = [
    "tree",
    "-d",
    "--features",
    "rgpot,rgpot-session",
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


def patch_git_source(url: str, rev: str) -> str:
    return f"git+{url}?rev={rev}"


def check_patch_crates_io(
    patches: dict,
    locked: dict[tuple[str, str], str],
    reqs: dict[str, str],
    manifest: str,
) -> tuple[list[str], dict[str, set[tuple[str, str]]]]:
    errors: list[str] = []
    seen: dict[str, set[tuple[str, str]]] = {}
    for name, spec in sorted(patches.items()):
        if not isinstance(spec, dict):
            continue
        git = spec.get("git")
        if git is None:
            continue
        what = f"{manifest}: {name} [patch.crates-io]"
        rev = spec.get("rev")
        if rev is None or not SHA.fullmatch(rev):
            bits = []
            for key in ("branch", "tag"):
                if key in spec:
                    bits.append(f"{key}={spec[key]}")
            if rev is not None:
                bits.append(f"rev={rev}")
            shown = ", ".join(bits) if bits else "the default branch"
            errors.append(f"{what} follows {shown}; pin rev to a full 40-hex commit")
            continue
        if "branch" in spec or "tag" in spec:
            bits = [f"{key}={spec[key]}" for key in ("branch", "tag") if key in spec]
            errors.append(
                f"{what} sets {', '.join(bits)} with rev; "
                "patch overrides must name only rev"
            )
        source = patch_git_source(git, rev)
        req = reqs.get(name)
        version = locked.get((name, source))
        if version is None:
            errors.append(f"{what} from {source} is not in Cargo.lock")
        elif req is not None and req != f"={version}":
            errors.append(
                f"{what} replaces a crate versioned {req!r}, but the commit is "
                f"{version}; use '={version}' in [dependencies]"
            )
        seen.setdefault(name, set()).add((git, rev))
    return errors, seen


def check_manifests(where: Path, label: str) -> list[str]:
    errors = []
    meta = json.loads(cargo(["metadata", "--no-deps", "--format-version", "1"], where))
    locked = {
        (pkg["name"], pkg["source"].split("#", 1)[0]): pkg["version"]
        for pkg in lock_packages(where)
        if pkg.get("source", "").startswith("git+")
    }
    reqs: dict[str, str] = {}
    for member in meta["packages"]:
        for dep in member["dependencies"]:
            reqs.setdefault(dep["name"], dep["req"])
    seen: dict[str, set[tuple[str, str]]] = {}
    for member in meta["packages"]:
        manifest_path = Path(member["manifest_path"])
        rel_manifest = manifest_path.relative_to(ROOT)
        with open(manifest_path, "rb") as fh:
            patch_table = tomllib.load(fh).get("patch", {}).get("crates-io") or {}
        if patch_table:
            patch_errors, patch_seen = check_patch_crates_io(
                patch_table, locked, reqs, str(rel_manifest)
            )
            errors.extend(patch_errors)
            for crate, sources in patch_seen.items():
                seen.setdefault(crate, set()).update(sources)
        manifest = rel_manifest
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
