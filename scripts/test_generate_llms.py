#!/usr/bin/env python3
"""Tests for Org summaries and verbatim spans in generate_llms."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_llms  # noqa: E402


def _summary(text: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "page.org"
        path.write_text(text, encoding="utf-8")
        return generate_llms.summary(path)


class SummaryTests(unittest.TestCase):
    def test_wrapped_changelog_bullet_is_not_the_summary(self) -> None:
        # docs/orgmode/changelog.org wraps long bullets onto indented lines.
        text = """#+TITLE: Changelog
#+OPTIONS: toc:nil num:nil

* 0.1.0 (2026-11-19)

** Fixed

- Compute the PAFI reactant partition in log space with shifted
  log-sum-exp trapezoid quadrature so deep free-energy wells no longer
  overflow the partition integral. Propagate =ln Z_R= through normalized
  rate evaluation and uncertainty propagation while retaining legacy
  linear-rate and partition fields for compatibility.
- Correct PAFI mass-metric projection.
"""
        prose = _summary(text)
        self.assertEqual(prose, "Changelog documentation.")
        self.assertNotIn("log-sum-exp", prose)

    def test_prose_after_a_wrapped_list_is_the_summary(self) -> None:
        text = """#+TITLE: Notes

- First bullet
  that continues on the next line

The page introduces the estimator.
"""
        self.assertEqual(_summary(text), "The page introduces the estimator.")

    def test_published_changelog_summary_ignores_the_list(self) -> None:
        prose = generate_llms.summary(ROOT / "docs" / "orgmode" / "changelog.org")
        self.assertEqual(prose, "Changelog documentation.")
        self.assertNotIn("log-sum-exp", prose)


class CodeSpanTests(unittest.TestCase):
    def test_internal_equals_stays_inside_one_span(self) -> None:
        rendered = generate_llms.clean_inline(
            "builds with =opt-level = 2= through =[profile.test]="
        )
        self.assertEqual(
            rendered,
            "builds with `opt-level = 2` through `[profile.test]`",
        )

    def test_contributing_index_keeps_version_requirement(self) -> None:
        line = (
            (ROOT / "docs" / "orgmode" / "contributing" / "index.org")
            .read_text(encoding="utf-8")
            .splitlines()[36]
        )
        rendered = generate_llms.clean_inline(line)
        self.assertIn('`version = "0.5.0"`', rendered)
        self.assertNotIn("`version `", rendered)

    def test_index_html_attributes_are_not_split(self) -> None:
        rendered = generate_llms.org_to_markdown(ROOT / "docs" / "orgmode" / "index.org")
        self.assertIn('class="vi-hero-mark"', rendered)
        self.assertIn('href="getting-started.html"', rendered)
        self.assertIn("\n   howto\n", rendered)
        self.assertNotIn('class`"', rendered)
        self.assertNotIn('src`"', rendered)
        self.assertNotIn('href`"', rendered)

    def test_math_equals_are_left_alone(self) -> None:
        rendered = generate_llms.clean_inline(r"\(\gamma = 0\) and \(\|\mathbf{n}\|_2 = 1\)")
        self.assertEqual(rendered, r"\gamma = 0 and \|\mathbf{n}\|_2 = 1")
        self.assertNotIn("`", rendered)

    def test_adjacent_spans_stay_separate(self) -> None:
        rendered = generate_llms.clean_inline("(=openmm=, =nblib=, =lammps=)")
        self.assertEqual(rendered, "(`openmm`, `nblib`, `lammps`)")


if __name__ == "__main__":
    unittest.main()
