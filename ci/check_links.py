#!/usr/bin/env python3
"""Verify Org toctree coverage, absence of TODO stubs, and DOI Handle resolution."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORG_ROOT = ROOT / "docs" / "orgmode"
BIB_PATH = ROOT / "docs" / "source" / "references.bib"

TODO_RE = re.compile(r"\b(?:TODO|FIXME|XXX)\b")
BIB_DOI_RE = re.compile(r"doi\s*=\s*\{([^}]+)\}")
URL_DOI_RE = re.compile(r"https://doi\.org/(10\.[^\s\]>\"'`]+)")


def tracked_files() -> list[str]:
    out = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
    return [line for line in out.splitlines() if line]


def check_toctree() -> list[str]:
    index_text = (ORG_ROOT / "index.org").read_text(encoding="utf-8")
    errors: list[str] = []
    for org_file in sorted(ORG_ROOT.rglob("*.org")):
        rel = org_file.relative_to(ORG_ROOT).with_suffix("").as_posix()
        if rel == "index":
            continue
        pattern = re.compile(rf"^\s*{re.escape(rel)}\s*$", re.MULTILINE)
        if not pattern.search(index_text):
            errors.append(f"docs/orgmode/{rel}.org is missing from docs/orgmode/index.org toctree")
    return errors


def check_todo_stubs() -> list[str]:
    errors: list[str] = []
    for rel in tracked_files():
        if rel == "ci/check_links.py":
            continue
        path = ROOT / rel
        if not path.is_file() or path.suffix in {".png", ".pdf", ".woff", ".woff2", ".svg"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if TODO_RE.search(line):
                errors.append(f"{rel}:{lineno}: contains TODO/FIXME/XXX stub")
    return errors


def collect_dois() -> dict[str, list[str]]:
    sources: dict[str, list[str]] = {}
    bib_text = BIB_PATH.read_text(encoding="utf-8")
    for m in BIB_DOI_RE.finditer(bib_text):
        doi = m.group(1).strip()
        sources.setdefault(doi, []).append("docs/source/references.bib")

    for rel in tracked_files():
        if rel == "ci/check_links.py":
            continue
        path = ROOT / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for m in URL_DOI_RE.finditer(text):
            doi = m.group(1).rstrip(".,;:")
            while doi.endswith(")") and doi.count(")") > doi.count("("):
                doi = doi[:-1]
            sources.setdefault(doi, []).append(rel)
    return sources


def verify_dois(dois: dict[str, list[str]]) -> list[str]:
    errors: list[str] = []
    for doi, locs in sorted(dois.items()):
        encoded = urllib.parse.quote(doi, safe="/")
        url = f"https://doi.org/api/handles/{encoded}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "rgflux-ci-linkcheck/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            errors.append(f"DOI {doi} ({locs[0]}): lookup failed: {exc}")
            continue
        if payload.get("responseCode") != 1:
            errors.append(f"DOI {doi} ({locs[0]}): invalid handle response {payload}")
            continue
        urls = [
            entry.get("data", {}).get("value")
            for entry in payload.get("values", [])
            if entry.get("type") == "URL"
        ]
        if not urls:
            errors.append(f"DOI {doi} ({locs[0]}): handle has no URL target")
    return errors


def main() -> int:
    errors: list[str] = []
    errors.extend(check_toctree())
    errors.extend(check_todo_stubs())
    dois = collect_dois()
    errors.extend(verify_dois(dois))
    if errors:
        print("link/doc checks failed:\n" + "\n".join(errors), file=sys.stderr)
        return 1
    print(f"docs and {len(dois)} unique DOIs ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
