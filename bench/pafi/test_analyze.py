#!/usr/bin/env python3
import copy
import importlib.util
import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("pafi_analyze", ROOT / "analyze.py")
ANALYZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZE)


class AnalysisFailureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json.loads((ROOT / "results.json").read_text())

    def test_committed_result_passes(self):
        self.assertEqual(
            ANALYZE.validate_result(self.result)["status"],
            "passed",
        )

    def test_nan_profile_fails(self):
        result = copy.deepcopy(self.result)
        result["cells"][0]["exact_reference"]["profile"][0] = float("nan")
        with self.assertRaises(ANALYZE.AnalysisError):
            ANALYZE.validate_result(result)

    def test_truncated_profile_fails(self):
        result = copy.deepcopy(self.result)
        sampled = result["cells"][0]["exact_reference"]["sampled_profiles"]
        sampled["lammps_pafi"]["median"].pop()
        with self.assertRaises(ANALYZE.AnalysisError):
            ANALYZE.validate_result(result)

    def test_tenfold_profile_error_fails(self):
        result = copy.deepcopy(self.result)
        cell = result["fcc_vacancy"]["cells"][0]
        cell["max_absolute_profile_difference"]["median"] *= 10.0
        with self.assertRaises(ANALYZE.AnalysisError):
            ANALYZE.validate_result(result)


if __name__ == "__main__":
    unittest.main()
