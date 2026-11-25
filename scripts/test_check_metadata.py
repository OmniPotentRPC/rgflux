#!/usr/bin/env python3
"""Tests for the codemeta.json expansion check in ci/check_metadata.py."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ci"))

try:
    import pyld  # noqa: F401
except ImportError:  # pragma: no cover - CI installs pyld before running the tests
    pyld = None

import check_metadata  # noqa: E402

# The shape of the CodeMeta 2.0 context: schema.org terms behind a prefix,
# with applicationCategory and codeRepository typed as IRIs. ScholarlyArticle
# and Periodical are deliberately not bare terms, as in the real context.
CONTEXT = {
    "@context": {
        "schema": "http://schema.org/",
        "codemeta": "https://codemeta.github.io/terms/",
        "SoftwareSourceCode": "schema:SoftwareSourceCode",
        "Person": "schema:Person",
        "name": "schema:name",
        "citation": {"@id": "schema:citation"},
        "applicationCategory": {"@id": "schema:applicationCategory", "@type": "@id"},
        "codeRepository": {"@id": "schema:codeRepository", "@type": "@id"},
        "isPartOf": {"@id": "schema:isPartOf"},
    }
}


def document(**extra: object) -> dict[str, object]:
    base: dict[str, object] = {
        "@context": "https://doi.org/10.5063/schema/codemeta-2.0",
        "@type": "SoftwareSourceCode",
        "name": "demo",
        "codeRepository": "https://example.org/demo",
    }
    base.update(extra)
    return base


@unittest.skipIf(pyld is None, "pyld is not installed")
class ExpansionTests(unittest.TestCase):
    def check(self, doc: dict[str, object]) -> list[str]:
        return check_metadata.check_codemeta(doc, CONTEXT)

    def test_prefixed_types_and_iri_values_pass(self) -> None:
        doc = document(
            applicationCategory="https://www.wikidata.org/wiki/Q336",
            citation=[
                {
                    "@type": "schema:ScholarlyArticle",
                    "name": "paper",
                    "isPartOf": {"@type": "schema:Periodical", "name": "journal"},
                }
            ],
        )
        self.assertEqual(self.check(doc), [])

    def test_bare_type_is_reported_as_undefined_and_relative(self) -> None:
        doc = document(citation=[{"@type": "ScholarlyArticle", "name": "paper"}])
        errors = self.check(doc)
        self.assertTrue(any("'ScholarlyArticle' is not defined" in e for e in errors), errors)
        self.assertTrue(any("relative IRI 'ScholarlyArticle'" in e for e in errors), errors)

    def test_bare_string_for_id_typed_term_is_relative(self) -> None:
        errors = self.check(document(applicationCategory="science"))
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("relative IRI 'science'", errors[0])

    def test_unknown_term_is_reported_instead_of_dropped(self) -> None:
        errors = self.check(document(unmappedProperty="x"))
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("'unmappedProperty' is not defined", errors[0])

    def test_loader_refuses_other_contexts(self) -> None:
        from pyld import jsonld

        loader = check_metadata.make_loader(CONTEXT)
        with self.assertRaises(jsonld.JsonLdError):
            loader("https://example.org/other-context")


class ZenodoTests(unittest.TestCase):
    VERSIONS = {"Cargo.toml": "1.2.3", "CITATION.cff": "1.2.3", "codemeta.json": "1.2.3"}

    def test_matching_version_passes(self) -> None:
        self.assertEqual(check_metadata.check_zenodo({"title": "x", "version": "1.2.3"}, self.VERSIONS), [])

    def test_mismatched_version_fails(self) -> None:
        errors = check_metadata.check_zenodo({"title": "x", "version": "1.2.2"}, self.VERSIONS)
        self.assertEqual(len(errors), 3, errors)
        self.assertTrue(all("'1.2.2'" in e for e in errors), errors)

    def test_missing_version_passes(self) -> None:
        self.assertEqual(check_metadata.check_zenodo({"title": "x"}, self.VERSIONS), [])

    def test_repository_file_matches_release_versions(self) -> None:
        import json

        zenodo = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
        self.assertEqual(check_metadata.check_zenodo(zenodo, check_metadata.release_versions()), [])


if __name__ == "__main__":
    unittest.main()
