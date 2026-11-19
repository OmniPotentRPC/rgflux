#!/usr/bin/env python3
"""Tests for anchored public API paths in generate_api_inventory."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_api_inventory  # noqa: E402


class ItemPathTests(unittest.TestCase):
    def test_does_not_match_rgflux_inside_a_longer_identifier(self) -> None:
        self.assertIsNone(generate_api_inventory.item_path("pub struct not_rgflux::Hidden"))
        self.assertIsNone(
            generate_api_inventory.item_path("pub use wrapper_rgflux::engine::NativeEngine")
        )

    def test_keeps_the_item_path_ahead_of_later_paths(self) -> None:
        line = (
            "pub fn rgflux::stats::batch_means(&[f64]) -> "
            "core::result::Result<rgflux::stats::BatchEstimate, rgflux::error::FluxError>"
        )
        self.assertEqual(
            generate_api_inventory.item_path(line),
            "rgflux::stats::batch_means",
        )

    def test_field_path_stops_at_the_type(self) -> None:
        self.assertEqual(
            generate_api_inventory.item_path("pub rgflux::stats::BatchEstimate::n_eff: f64"),
            "rgflux::stats::BatchEstimate::n_eff",
        )

    def test_strips_generics_and_non_exhaustive(self) -> None:
        self.assertEqual(
            generate_api_inventory.item_path("#[non_exhaustive] pub struct rgflux::pafi::PafiConfig<T>"),
            "rgflux::pafi::PafiConfig",
        )

    def test_does_not_truncate_a_segment(self) -> None:
        self.assertEqual(
            generate_api_inventory.item_path("pub fn rgflux::stats::batch_means2()"),
            "rgflux::stats::batch_means2",
        )

    def test_alias_is_not_replaced_by_a_suffix_in_the_target(self) -> None:
        line = "pub type rgflux::foo::Alias = not_rgflux::Item"
        self.assertEqual(generate_api_inventory.item_path(line), "rgflux::foo::Alias")


if __name__ == "__main__":
    unittest.main()
