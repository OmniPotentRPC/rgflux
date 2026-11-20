#!/usr/bin/env python3
"""Generate llms.txt and llms-full.txt from the canonical Org sources."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORG_ROOT = ROOT / "docs" / "orgmode"
SHORT_PATH = ROOT / "llms.txt"
FULL_PATH = ROOT / "llms-full.txt"

TITLE_RE = re.compile(r"^#\+TITLE:\s*(.+)$", re.IGNORECASE)
HEADING_RE = re.compile(r"^(\*+)\s+(.+)$")
LINK_RE = re.compile(r"\[\[([^\]]+)\]\[([^\]]+)\]\]")
# Verbatim spans may contain " = ". Open only at a boundary so HTML
# attributes (attr="value") and math equals are left untouched.
CODE_RE = re.compile(
    r"(?:^|(?<=[^A-Za-z0-9_]))=(?!\s)"
    r"((?:[^=\n]|=(?=\s))*?[^\s=])=(?![A-Za-z0-9_])"
)
VERBATIM_RE = re.compile(r"~([^~\n]+)~")
SRC_BEGIN_RE = re.compile(r"^#\+begin_src\s+(\S+)", re.IGNORECASE)
SRC_END_RE = re.compile(r"^#\+end_src", re.IGNORECASE)
CITATION_RE = re.compile(r"@@rst::cite:`[^`]+`@@")


def clean_inline(text: str) -> str:
    """Convert compact Org inline markup used in index descriptions."""
    text = CITATION_RE.sub("", text)
    text = CODE_RE.sub(r"`\1`", text)
    text = VERBATIM_RE.sub(r"`\1`", text)
    text = text.replace(r"\(", "").replace(r"\)", "")
    text = " ".join(text.split())
    return re.sub(r"\s+([,.;:])", r"\1", text)


def org_files() -> list[Path]:
    """Return canonical pages in navigation order, then any unlisted pages."""
    index = ORG_ROOT / "index.org"
    text = index.read_text(encoding="utf-8")
    ordered: list[Path] = [index]
    for raw in text.splitlines():
        rel = raw.strip()
        if not rel or rel.startswith(("#", "*", ":")):
            continue
        candidate = ORG_ROOT / f"{rel}.org"
        if candidate.is_file() and candidate not in ordered:
            ordered.append(candidate)
    for candidate in sorted(ORG_ROOT.rglob("*.org")):
        if candidate not in ordered:
            ordered.append(candidate)
    return ordered


def title(path: Path) -> str:
    """Read an Org title."""
    for line in path.read_text(encoding="utf-8").splitlines():
        match = TITLE_RE.match(line)
        if match:
            return match.group(1).strip()
    return path.stem.replace("-", " ").title()


def summary(path: Path) -> str:
    """Return the first prose paragraph as a compact page summary."""
    paragraphs: list[str] = []
    in_export = False
    in_source = False
    in_list = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if SRC_BEGIN_RE.match(stripped):
            in_source = True
            in_list = False
            continue
        if SRC_END_RE.match(stripped):
            in_source = False
            continue
        if in_source:
            continue
        if stripped.lower().startswith("#+begin_export"):
            in_export = True
            in_list = False
            continue
        if stripped.lower().startswith("#+end_export"):
            in_export = False
            continue
        if in_export:
            continue
        if not stripped:
            if paragraphs:
                break
            in_list = False
            continue
        if stripped.startswith(("#", "*", "|", ":", "-", "..")):
            in_list = stripped.startswith("-")
            continue
        # Org wraps a bullet by indenting the following lines.
        if in_list and (line.startswith(" ") or line.startswith("\t")):
            continue
        in_list = False
        paragraphs.append(stripped)
    prose = clean_inline(" ".join(paragraphs))
    if not prose:
        return f"{title(path)} documentation."
    if len(prose) > 240:
        prose = prose[:237].rstrip() + "..."
    return prose


def org_to_markdown(path: Path) -> str:
    """Convert the Org subset used by this repository to readable Markdown."""
    output: list[str] = []
    in_source = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if TITLE_RE.match(line):
            continue
        source = SRC_BEGIN_RE.match(line)
        if source:
            output.append(f"```{source.group(1)}")
            in_source = True
            continue
        if SRC_END_RE.match(line):
            output.append("```")
            in_source = False
            continue
        if not in_source and line.lower().startswith("#+"):
            continue
        heading = HEADING_RE.match(line)
        if heading and not in_source:
            level = min(len(heading.group(1)) + 1, 6)
            line = f"{'#' * level} {heading.group(2)}"
        if not in_source:
            line = LINK_RE.sub(r"[\2](\1)", line)
            line = CODE_RE.sub(r"`\1`", line)
            line = line.replace("@@rst::", "").replace("@@", "")
        output.append(line.rstrip())
    return "\n".join(output).strip()


def generated() -> tuple[str, str]:
    """Render both llms files."""
    pages = org_files()
    short = [
        "# rgflux",
        "",
        "> Rare event kinetics over rgsaddle and rgmin.",
        "",
        "The Rust API separates rate estimators and trajectory sampling from",
        "potential energy surfaces and molecular dynamics engines.",
        "",
        "## Documentation",
        "",
    ]
    for page in pages:
        rel = page.relative_to(ORG_ROOT).with_suffix(".html").as_posix()
        short.append(f"- [{title(page)}](docs/{rel}): {summary(page)}")
    short.extend(
        [
            "",
            "## Source references",
            "",
            "- [Rust crate API](https://docs.rs/rgflux): Generated Rust API reference.",
            "- [Repository](https://github.com/OmniPotentRPC/rgflux): Source and examples.",
            "",
        ]
    )

    full = short[:6] + ["", "## Complete documentation", ""]
    for page in pages:
        full.extend([f"## {title(page)}", "", org_to_markdown(page), ""])
    return "\n".join(short), "\n".join(full).rstrip() + "\n"


def main() -> int:
    short, full = generated()
    if "--check" in sys.argv:
        errors = []
        for path, expected in ((SHORT_PATH, short), (FULL_PATH, full)):
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                errors.append(f"{path.relative_to(ROOT)} is stale")
        if errors:
            print("\n".join(errors), file=sys.stderr)
            return 1
        print("llms files are current")
        return 0
    SHORT_PATH.write_text(short, encoding="utf-8")
    FULL_PATH.write_text(full, encoding="utf-8")
    print("generated llms.txt and llms-full.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
