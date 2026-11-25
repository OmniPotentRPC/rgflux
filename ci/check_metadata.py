#!/usr/bin/env python3
"""Verify the citation metadata files.

codemeta.json is expanded as JSON-LD against the CodeMeta 2.0 context, which
is passed to pyld explicitly through a document loader so that the check does
not depend on how the context URL in the file resolves. The check fails when

* a term, type or key is not defined by the context (JSON-LD drops unknown
  terms silently on expansion),
* an IRI in the expanded document is relative, because a bare string given
  for an ``@id``-typed term or for ``@type`` expands against the document
  base instead of a vocabulary.

.zenodo.json is checked against the legacy Zenodo deposit schema, which
rejects a ``version`` key.

Requires ``pyld`` (``python3 -m pip install pyld``).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CODEMETA_PATH = ROOT / "codemeta.json"
ZENODO_PATH = ROOT / ".zenodo.json"

CODEMETA_2_CONTEXT_URL = "https://raw.githubusercontent.com/codemeta/codemeta/2.0/codemeta.jsonld"
CONTEXT_ALIASES = frozenset(
    {
        CODEMETA_2_CONTEXT_URL,
        "https://doi.org/10.5063/schema/codemeta-2.0",
        "https://w3id.org/codemeta/2.0",
    }
)
# Any IRI that starts with this base came from a relative reference.
SENTINEL_BASE = "https://relative-iri.invalid/"
ABSOLUTE_IRI_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*:")


def fetch_context(url: str = CODEMETA_2_CONTEXT_URL) -> dict[str, Any]:
    """Download the CodeMeta 2.0 context document."""
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
        return json.load(response)


def make_loader(context_document: dict[str, Any]):
    """Return a pyld document loader that serves only the CodeMeta context."""
    from pyld import jsonld

    def loader(url: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        if url not in CONTEXT_ALIASES:
            raise jsonld.JsonLdError(
                f"refusing to load {url}: only the CodeMeta 2.0 context is allowed",
                "jsonld.LoadDocumentError",
                code="loading document failed",
            )
        return {
            "contentType": "application/ld+json",
            "contextUrl": None,
            "documentUrl": url,
            "document": context_document,
        }

    return loader


def input_terms(node: Any) -> set[str]:
    """Collect every property name used in a JSON-LD document."""
    terms: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if not key.startswith("@"):
                terms.add(key)
            terms |= input_terms(value)
    elif isinstance(node, list):
        for item in node:
            terms |= input_terms(item)
    return terms


def input_types(node: Any) -> set[str]:
    """Collect every ``@type`` value used in a JSON-LD document."""
    types: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "@type":
                types |= {value} if isinstance(value, str) else set(value)
            elif key != "@context":
                types |= input_types(value)
    elif isinstance(node, list):
        for item in node:
            types |= input_types(item)
    return types


def relative_iris(node: Any, path: str = "$") -> list[str]:
    """Locate relative IRIs in an expanded JSON-LD document."""
    problems: list[str] = []

    def check(iri: str, where: str) -> None:
        if iri.startswith(SENTINEL_BASE) or not (
            ABSOLUTE_IRI_RE.match(iri) or iri.startswith("_:")
        ):
            problems.append(f"{where}: relative IRI {iri.removeprefix(SENTINEL_BASE)!r}")

    if isinstance(node, dict):
        for key, value in node.items():
            where = f"{path}.{key}"
            if key == "@id" and isinstance(value, str):
                check(value, where)
            elif key == "@type":
                for item in [value] if isinstance(value, str) else value:
                    check(item, where)
            elif key.startswith("@"):
                problems += relative_iris(value, where)
            else:
                check(key, where)
                problems += relative_iris(value, where)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            problems += relative_iris(item, f"{path}[{index}]")
    return problems


def check_codemeta(document: dict[str, Any], context_document: dict[str, Any]) -> list[str]:
    """Expand ``document`` against the context and report every problem."""
    from pyld import jsonld

    defined = set(context_document["@context"])
    errors: list[str] = []
    for name in sorted(input_terms(document) | input_types(document)):
        if name not in defined and not ABSOLUTE_IRI_RE.match(name):
            errors.append(f"term {name!r} is not defined by the CodeMeta 2.0 context")
        elif ":" in name and name.split(":", 1)[0] not in defined and not name.startswith(
            ("http://", "https://")
        ):
            errors.append(f"prefix of {name!r} is not defined by the CodeMeta 2.0 context")

    expanded = jsonld.expand(
        document,
        {"documentLoader": make_loader(context_document), "base": SENTINEL_BASE},
    )
    if not expanded:
        errors.append("expansion produced an empty document")
    errors += relative_iris(expanded)
    return errors


def check_zenodo(document: dict[str, Any]) -> list[str]:
    """Reject keys that the legacy Zenodo deposit schema does not accept."""
    if "version" in document:
        return ["'version' is not accepted by the legacy Zenodo deposit schema"]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--context-file",
        type=Path,
        help="read the CodeMeta 2.0 context from a file instead of downloading it",
    )
    args = parser.parse_args(argv)

    if args.context_file:
        context_document = json.loads(args.context_file.read_text(encoding="utf-8"))
    else:
        context_document = fetch_context()

    codemeta = json.loads(CODEMETA_PATH.read_text(encoding="utf-8"))
    zenodo = json.loads(ZENODO_PATH.read_text(encoding="utf-8"))
    errors = [f"codemeta.json: {e}" for e in check_codemeta(codemeta, context_document)]
    errors += [f".zenodo.json: {e}" for e in check_zenodo(zenodo)]
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        return 1
    print("codemeta.json expands against CodeMeta 2.0 with absolute IRIs; .zenodo.json ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
