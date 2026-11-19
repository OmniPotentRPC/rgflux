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
        validation = ANALYZE.validate_result(self.result)
        self.assertEqual(validation["status"], "passed")
        self.assertEqual(validation["nonfinite_samples"], 0)
        self.assertEqual(
            validation,
            self.result["execution_summary"]["analysis_validation"],
        )

    def test_nan_profile_fails(self):
        result = copy.deepcopy(self.result)
        result["cells"][0]["exact_reference"]["profile"][0] = float("nan")
        with self.assertRaisesRegex(ANALYZE.AnalysisError, "non-finite value"):
            ANALYZE.validate_result(result)

    def test_truncated_profile_fails(self):
        result = copy.deepcopy(self.result)
        sampled = result["cells"][0]["exact_reference"]["sampled_profiles"]
        sampled["lammps_pafi"]["median"].pop()
        with self.assertRaisesRegex(
            ANALYZE.AnalysisError, r"lammps_pafi\.median: expected \d+ values"
        ):
            ANALYZE.validate_result(result)

    def test_tenfold_profile_error_fails(self):
        result = copy.deepcopy(self.result)
        cell = result["fcc_vacancy"]["cells"][0]
        cell["max_absolute_profile_difference"]["median"] *= 10.0
        with self.assertRaisesRegex(
            ANALYZE.AnalysisError, "profile disagreement exceeds energy scale"
        ):
            ANALYZE.validate_result(result)


class VerdictProseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json.loads((ROOT / "results.json").read_text())
        cls.markdown = (ROOT / "results.md").read_text()
        cls.statistics = ANALYZE.verdict_statistics(cls.result)

    def test_committed_verdict_matches_the_data(self):
        summary = self.result["execution_summary"]
        statistics = self.statistics
        self.assertEqual(statistics["dimer_reference_failures"], 0)
        self.assertEqual(
            summary["process_failures"], {"dimer": 0, "fcc_vacancy": 0}
        )
        self.assertEqual(statistics["vacancy_rejections"], 0)
        buckets = summary["rejection_buckets"]
        self.assertEqual(buckets["genuine_exp_overflow"]["count"], 0)
        self.assertEqual(buckets["other"]["count"], 0)
        self.assertEqual(
            statistics["dimer_rejections"],
            buckets["zero_temperature_tol"]["count"]
            + buckets["zero_reactant_partition"]["count"],
        )
        self.assertEqual(
            statistics["dimer_executions"] + statistics["vacancy_executions"],
            summary["planned"],
        )

    def test_results_markdown_is_exactly_the_generated_output(self):
        generated = ANALYZE.render_markdown(self.result)
        self.assertTrue(generated)
        self.assertEqual(generated, self.markdown)

    def test_appended_stale_claim_is_not_the_generated_output(self):
        generated = ANALYZE.render_markdown(self.result)
        stale = self.markdown + "\nEvery retained cell converged.\n"
        self.assertNotEqual(generated, stale)

    def test_contradictory_verdict_claim_is_not_the_generated_output(self):
        generated = ANALYZE.render_markdown(self.result)
        claim = "the reference accepted every execution."
        self.assertIn(claim, generated)
        contradicted = self.markdown.replace(
            claim, "the reference retained 1 failed executions."
        )
        self.assertNotEqual(generated, contradicted)

    def test_markdown_follows_every_change_in_the_result(self):
        result = copy.deepcopy(self.result)
        result["execution_summary"]["analysis_validation"]["gates_checked"] += 1
        self.assertNotEqual(ANALYZE.render_markdown(result), self.markdown)

    def test_verdict_prose_follows_a_changed_result(self):
        result = copy.deepcopy(self.result)
        result["cells"][0]["failures"]["lammps_pafi"] += 1
        result["execution_summary"]["planned"] += 1
        statistics = ANALYZE.verdict_statistics(result)
        section = "\n".join(ANALYZE.verdict_section(statistics, result))
        self.assertIn("the reference retained 1 failed executions.", section)
        self.assertNotIn("the reference accepted every execution.", section)
        self.assertNotIn(section, self.markdown)

    def test_inconsistent_execution_count_fails(self):
        result = copy.deepcopy(self.result)
        result["execution_summary"]["planned"] += 1
        with self.assertRaisesRegex(
            ANALYZE.AnalysisError, "planned executions"
        ):
            ANALYZE.verdict_statistics(result)


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
