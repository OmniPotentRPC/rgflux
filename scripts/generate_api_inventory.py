#!/usr/bin/env python3
"""Generate the canonical public Rust API inventory in the Org reference."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "orgmode" / "reference.org"
BEGIN = "# BEGIN GENERATED PUBLIC API"
END = "# END GENERATED PUBLIC API"
PATH_RE = re.compile(r"rgflux(?:::[A-Za-z_][A-Za-z0-9_]*(?:<[^>]+>)?)+")


def public_items() -> list[str]:
    """Return stable qualified item names from cargo-public-api."""
    empty_libs = ROOT / "target" / "empty-libs"
    empty_libs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LAMMPC_LIB_DIR"] = str(empty_libs)
    command = [
        "cargo",
        "public-api",
        "-sss",
        "--all-features",
        "--color=never",
    ]
    output = subprocess.check_output(command, cwd=ROOT, env=env, text=True)
    items: set[str] = set()
    for line in output.splitlines():
        stripped = line.removeprefix("#[non_exhaustive] ").strip()
        if not stripped.startswith("pub "):
            continue
        match = PATH_RE.search(stripped)
        if match:
            items.add(re.sub(r"<[^>]+>", "", match.group(0)))
    return sorted(items)


def render(items: list[str]) -> str:
    """Render an Org inventory grouped by top-level module."""
    groups: dict[str, list[str]] = {}
    for item in items:
        parts = item.split("::")
        group = parts[1] if len(parts) > 1 else "crate"
        groups.setdefault(group, []).append(item)
    lines = [BEGIN]
    for group, group_items in sorted(groups.items()):
        lines.extend([f"*** ={group}=", ""])
        lines.extend(f"- ={item}=" for item in group_items)
        lines.append("")
    lines.append(END)
    return "\n".join(lines)


def updated_text() -> tuple[str, int]:
    """Return reference text with the generated block replaced."""
    items = public_items()
    current = REFERENCE.read_text(encoding="utf-8")
    if BEGIN not in current or END not in current:
        raise RuntimeError("reference.org has no generated API markers")
    before, remainder = current.split(BEGIN, 1)
    _, after = remainder.split(END, 1)
    return before + render(items) + after, len(items)


def main() -> int:
    expected, count = updated_text()
    current = REFERENCE.read_text(encoding="utf-8")
    if "--check" in sys.argv:
        if current != expected:
            print("docs/orgmode/reference.org public API inventory is stale", file=sys.stderr)
            return 1
        print(f"{count} public Rust API items documented")
        return 0
    REFERENCE.write_text(expected, encoding="utf-8")
    print(f"documented {count} public Rust API items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
