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
# A path is a whole identifier. Do not match the rgflux suffix of a longer name.
PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])rgflux(?:::[A-Za-z_][A-Za-z0-9_]*(?:<[^>]+>)?)+(?![A-Za-z0-9_])"
)
# A count of API items or paths, which drifts unless the generator writes it.
COUNT_RE = re.compile(
    r"\b\d+\s+of\s+\d+\b"
    r"|\b\d+\s+(?:qualified\s+)?(?:public|API)\s+(?:Rust\s+)?(?:API\s+)?(?:item\s+)?"
    r"(?:items|paths)\b"
)
COUNTED_DOCS = (
    "docs/orgmode/index.org",
    "docs/orgmode/reference.org",
    "docs/orgmode/explanation.org",
    "docs/orgmode/howto.org",
    "docs/orgmode/getting-started.org",
    "README.md",
)


def item_path(line: str) -> str | None:
    """Return the qualified rgflux path on one cargo-public-api line."""
    stripped = line.removeprefix("#[non_exhaustive] ").strip()
    if not stripped.startswith("pub "):
        return None
    match = PATH_RE.search(stripped)
    if match is None:
        return None
    return re.sub(r"<[^>]+>", "", match.group(0))


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
        path = item_path(line)
        if path:
            items.add(path)
    return sorted(items)


def render(items: list[str]) -> str:
    """Render an Org inventory grouped by top-level module."""
    groups: dict[str, list[str]] = {}
    for item in items:
        parts = item.split("::")
        group = parts[1] if len(parts) > 1 else "crate"
        groups.setdefault(group, []).append(item)
    lines = [
        BEGIN,
        f"This inventory lists {len(items)} qualified public item paths.",
        "",
    ]
    for group, group_items in sorted(groups.items()):
        lines.extend([f"*** ={group}=", ""])
        lines.extend(f"- ={item}=" for item in group_items)
        lines.append("")
    lines.append(END)
    return "\n".join(lines)


def stray_counts(text: str) -> list[str]:
    """Return hand-written API counts in `text` outside the generated block."""
    if BEGIN in text and END in text:
        before, remainder = text.split(BEGIN, 1)
        _, after = remainder.split(END, 1)
        text = before + after
    return [match.group(0) for match in COUNT_RE.finditer(text)]


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
        stale = False
        if current != expected:
            print("docs/orgmode/reference.org public API inventory is stale", file=sys.stderr)
            stale = True
        for path in COUNTED_DOCS:
            for count_text in stray_counts((ROOT / path).read_text(encoding="utf-8")):
                print(
                    f"{path}: hand-written API count {count_text!r}; "
                    "only the generated inventory states the count",
                    file=sys.stderr,
                )
                stale = True
        if stale:
            return 1
        print(f"{count} public Rust API items documented")
        return 0
    REFERENCE.write_text(expected, encoding="utf-8")
    print(f"documented {count} public Rust API items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
