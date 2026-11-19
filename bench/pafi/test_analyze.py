#!/usr/bin/env python3
import ast
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


class StrictPairingTests(unittest.TestCase):
    def test_every_zip_in_the_benchmark_scripts_is_strict(self):
        offenders = []
        for path in sorted(ROOT.glob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "zip"
                ):
                    continue
                strict = [
                    keyword
                    for keyword in node.keywords
                    if keyword.arg == "strict"
                ]
                if not (
                    len(strict) == 1
                    and isinstance(strict[0].value, ast.Constant)
                    and strict[0].value.value is True
                ):
                    offenders.append(f"{path.name}:{node.lineno}")
        self.assertEqual(offenders, [])

    def test_combined_sem_pairs_every_plane(self):
        combined = ANALYZE.combined_profile_sem([3.0, None, 6.0], [4.0, 1.0, 8.0])
        self.assertEqual(combined, [5.0, 10.0])

    def test_combined_sem_is_empty_when_one_engine_has_no_samples(self):
        self.assertEqual(ANALYZE.combined_profile_sem([], [1.0, 2.0]), [])
        self.assertEqual(ANALYZE.combined_profile_sem([1.0, 2.0], []), [])

    def test_combined_sem_rejects_length_mismatch(self):
        with self.assertRaises(ValueError):
            ANALYZE.combined_profile_sem([1.0, 2.0, 3.0], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
