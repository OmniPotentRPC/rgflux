#!/usr/bin/env python3
from __future__ import annotations

import unittest
from pathlib import Path

from check_dependency_pins import ROOT, check_manifests, check_patch_crates_io

LOCKED = {
    (
        "rgsaddle",
        "git+https://github.com/OmniPotentRPC/rgsaddle?rev=92a74493d16364a1ba8542a63598ce8bcee5387c",
    ): "0.5.0",
}
REQS = {"rgsaddle": "=0.5.0"}


class PatchCratesIoTest(unittest.TestCase):
    def test_unpinned_branch_override_reports_error(self) -> None:
        patches = {
            "rgsaddle": {
                "git": "https://github.com/OmniPotentRPC/rgsaddle",
                "branch": "develop",
            }
        }
        errors, _seen = check_patch_crates_io(
            patches, LOCKED, REQS, "Cargo.toml"
        )
        self.assertTrue(
            errors,
            "unpinned patch.crates-io git overrides must be reported",
        )
        self.assertTrue(
            any("[patch.crates-io]" in line and "40-hex" in line for line in errors),
            errors,
        )

    def test_current_manifest_patch_table_is_clean(self) -> None:
        errors = check_manifests(ROOT, ".")
        patch_lines = [line for line in errors if "[patch.crates-io]" in line]
        self.assertEqual(patch_lines, [], patch_lines)


if __name__ == "__main__":
    unittest.main()