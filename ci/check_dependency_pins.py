#!/usr/bin/env python3
"""Fail on a git source, a patch override, a non-caret requirement, or a second copy of our crates.

The published manifest asks crates.io for caret requirements, so downstream
patch releases unify. `Cargo.lock` must resolve every package from crates.io,
and the manifest must carry no git dependency and no `[patch]` table, so a
local build resolves what `cargo publish` verifies. Each locked version must
satisfy the requirement that asks for it.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

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
VERSION = re.compile(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?")
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


def parse_version(text: str) -> tuple[int, int, int]:
    m = VERSION.fullmatch(text.strip())
    if m is None:
        raise ValueError(f"unsupported version {text!r}")
    return tuple(int(part or 0) for part in m.groups())  # type: ignore[return-value]


def caret_upper(low: tuple[int, int, int], given: int) -> tuple[int, int, int]:
    major, minor, patch = low
    if major > 0 or given == 1:
        return (major + 1, 0, 0)
    if minor > 0 or given == 2:
        return (0, minor + 1, 0)
    return (0, 0, patch + 1)


def satisfies(version: str, req: str) -> bool:
    """Whether `version` meets a Cargo requirement made of comparators."""
    have = parse_version(version)
    for comparator in req.split(","):
        comparator = comparator.strip()
        if comparator in ("", "*"):
            continue
        m = re.fullmatch(r"(\^|~|=|>=|<=|>|<)?\s*(\S+)", comparator)
        if m is None:
            raise ValueError(f"unsupported requirement {req!r}")
        op, text = m[1] or "^", m[2]
        want = parse_version(text)
        given = len(text.split("."))
        if op == "=":
            ok = have == want if given == 3 else have[:given] == want[:given]
        elif op == "^":
            ok = want <= have < caret_upper(want, given)
        elif op == "~":
            upper = (want[0], want[1] + 1, 0) if given > 1 else (want[0] + 1, 0, 0)
            ok = want <= have < upper
        elif op == ">=":
            ok = have >= want
        elif op == ">":
            ok = have > want
        elif op == "<=":
            ok = have <= want
        else:
            ok = have < want
        if not ok:
            return False
    return True


def lock_packages(where: Path) -> list[dict]:
    with open(where / "Cargo.lock", "rb") as fh:
        return tomllib.load(fh).get("package", [])


def check_lock(where: Path, label: str) -> list[str]:
    errors = []
    for pkg in lock_packages(where):
        source = pkg.get("source", "")
        if source.startswith("git+"):
            errors.append(
                f"{label}/Cargo.lock: {pkg['name']} {pkg['version']} is locked from "
                f"{source}; resolve it from crates.io"
            )
    return errors


def check_manifests(where: Path, label: str) -> list[str]:
    errors = []
    with open(where / "Cargo.toml", "rb") as fh:
        manifest = tomllib.load(fh)
    if "patch" in manifest:
        errors.append(
            f"{label}/Cargo.toml: remove the [patch] table; dependencies resolve from crates.io"
        )
    meta = json.loads(cargo(["metadata", "--no-deps", "--format-version", "1"], where))
    locked: dict[str, set[str]] = {}
    for pkg in lock_packages(where):
        locked.setdefault(pkg["name"], set()).add(pkg["version"])
    for member in meta["packages"]:
        manifest_path = Path(member["manifest_path"]).relative_to(ROOT)
        for dep in member["dependencies"]:
            name = dep["name"]
            what = f"{manifest_path}: {name}"
            source = dep.get("source") or ""
            if source.startswith("git+"):
                errors.append(f"{what} is a git dependency; use a crates.io version")
                continue
            if dep.get("path"):
                continue
            req = dep["req"]
            if re.search(r"(^|,)\s*=", req):
                errors.append(
                    f"{what} asks for {req!r}; use a caret requirement so downstream "
                    f"patch releases unify"
                )
            versions = locked.get(name)
            if versions is None:
                if not dep.get("optional"):
                    errors.append(f"{what} is not in {label}/Cargo.lock")
            elif not any(satisfies(version, req) for version in versions):
                errors.append(
                    f"{what} asks for {req!r}, but {label}/Cargo.lock has "
                    f"{', '.join(sorted(versions))}"
                )
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
    shown = sorted(
        f"{name} {version}"
        for name, version in (
            (pkg["name"], pkg["version"])
            for pkg in lock_packages(ROOT)
            if pkg["name"] in OWN and pkg["name"] != "rgflux"
        )
    )
    print("dependency pins ok: " + "; ".join(shown) + " (all from crates.io)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
