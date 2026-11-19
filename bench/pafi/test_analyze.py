#!/usr/bin/env python3
import ast
import copy
import importlib.util
import io
import json
import pathlib
import subprocess
import sys
import tarfile
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parent
PACK_SPEC = importlib.util.spec_from_file_location("pafi_pack", ROOT / "pack_raw.py")
PACK = importlib.util.module_from_spec(PACK_SPEC)
PACK_SPEC.loader.exec_module(PACK)
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
        self.assertEqual(buckets["other"]["count"], 0)
        self.assertEqual(
            statistics["dimer_rejections"],
            buckets["zero_temperature_tol"]["count"]
            + buckets["reproduced_reactant_partition"]["count"]
            + buckets["unclassified_reactant_partition"]["count"],
        )
        self.assertEqual(buckets["reproduced_reactant_partition"]["count"], 1)
        self.assertEqual(
            summary["raw_rejection_errors"]["dimer"][
                ANALYZE.REACTANT_PARTITION_ERROR
            ],
            buckets["reproduced_reactant_partition"]["count"]
            + buckets["unclassified_reactant_partition"]["count"],
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


def write_campaign(directory, count):
    (directory / "records").mkdir(parents=True)
    manifest = []
    for index in range(count):
        name = f"records/run-{index:02d}.json"
        manifest.append({"record": name, "seed": index})
        (directory / name).write_text(json.dumps({"status": "ok", "n": index}))
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return manifest


class RawArchiveTests(unittest.TestCase):
    def test_archive_yields_the_directory_entries(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = pathlib.Path(scratch)
            write_campaign(root / "live", 3)
            PACK.pack(root / "live", root / "archived.tar.xz")
            from_directory = ANALYZE.load_campaign_entries(root / "live")
            from_archive = ANALYZE.load_campaign_entries(root / "archived")
        self.assertEqual(len(from_archive), 3)
        self.assertEqual(from_archive, from_directory)

    def test_packing_twice_gives_the_same_bytes(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = pathlib.Path(scratch)
            write_campaign(root / "campaign", 4)
            PACK.pack(root / "campaign", root / "one.tar.xz")
            PACK.pack(root / "campaign", root / "two.tar.xz")
            self.assertEqual(
                (root / "one.tar.xz").read_bytes(),
                (root / "two.tar.xz").read_bytes(),
            )

    def test_record_absent_from_the_archive_is_reported(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = pathlib.Path(scratch)
            manifest = json.dumps([{"record": "records/run-00.json"}]).encode()
            with tarfile.open(root / "campaign.tar.xz", "w:xz") as bundle:
                member = tarfile.TarInfo("manifest.json")
                member.size = len(manifest)
                bundle.addfile(member, io.BytesIO(manifest))
            with self.assertRaisesRegex(
                ANALYZE.AnalysisError, "raw record is missing"
            ):
                ANALYZE.load_campaign_entries(root / "campaign")

    def test_missing_campaign_is_reported(self):
        with tempfile.TemporaryDirectory() as scratch:
            with self.assertRaisesRegex(
                ANALYZE.AnalysisError, "campaign manifest is missing"
            ):
                ANALYZE.load_campaign_entries(pathlib.Path(scratch) / "absent")


class RegenerationTests(unittest.TestCase):
    def test_committed_raw_records_regenerate_the_committed_results(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = pathlib.Path(scratch)
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "analyze.py"),
                    "--output-json",
                    str(root / "results.json"),
                    "--output-markdown",
                    str(root / "results.md"),
                ],
                check=True,
                capture_output=True,
            )
            for name in ("results.json", "results.md"):
                with self.subTest(name=name):
                    self.assertEqual(
                        (root / name).read_text(), (ROOT / name).read_text()
                    )


def reactant_partition_entry(item_changes=None, record_changes=None):
    configuration = ANALYZE.REPRODUCED_REJECTION
    item = {
        "engine": "rgflux",
        "case": configuration["case"],
        "planes": configuration["planes"],
        "temperature": configuration["temperature"],
        "samples": configuration["samples"],
        "threads": configuration["threads"],
        "seed": configuration["seed"],
    }
    record = {
        "status": "rejected",
        "error": ANALYZE.REACTANT_PARTITION_ERROR,
        "burn_in": configuration["burn_in"],
        "effective_threads": configuration["effective_threads"],
        "time_step": configuration["time_step"],
        "friction": configuration["friction"],
        "zero_temperature_tolerance": configuration["zero_temperature_tolerance"],
        "minimum_energy": -1.0,
        "saddle_energy": -1.0,
    }
    item.update(item_changes or {})
    record.update(record_changes or {})
    return item, record


class RejectionClassificationTests(unittest.TestCase):
    def counts(self, *entries):
        buckets = ANALYZE.rejection_audit(list(entries))
        return (
            buckets["reproduced_reactant_partition"]["count"],
            buckets["unclassified_reactant_partition"]["count"],
        )

    def test_the_reproduced_configuration_is_classified(self):
        self.assertEqual(self.counts(reactant_partition_entry()), (1, 0))

    def test_other_seed_with_constant_energy_is_unclassified(self):
        entry = reactant_partition_entry({"seed": 1036331})
        self.assertEqual(entry[1]["minimum_energy"], entry[1]["saddle_energy"])
        self.assertEqual(self.counts(entry), (0, 1))

    def test_any_changed_setting_is_unclassified(self):
        changes = (
            ({"case": "straight"}, {}),
            ({"planes": 33}, {}),
            ({"temperature": 0.1}, {}),
            ({"samples": 128}, {}),
            ({"threads": 4}, {}),
            ({}, {"burn_in": 64}),
            ({}, {"time_step": 0.001}),
            ({}, {"friction": 2.0}),
            ({}, {"zero_temperature_tolerance": 0.02}),
            ({"engine": "lammps-pafi"}, {}),
        )
        for item_changes, record_changes in changes:
            with self.subTest(item_changes=item_changes, record=record_changes):
                entry = reactant_partition_entry(item_changes, record_changes)
                self.assertEqual(self.counts(entry), (0, 1))

    def test_other_errors_are_not_reactant_partition_rejections(self):
        item, record = reactant_partition_entry()
        record["error"] = ANALYZE.ZERO_TEMPERATURE_TOL_ERROR
        buckets = ANALYZE.rejection_audit([(item, record)])
        self.assertEqual(buckets["zero_temperature_tol"]["count"], 1)
        self.assertEqual(self.counts((item, record)), (0, 0))

    def test_reproducer_runs_the_classified_configuration(self):
        spec = importlib.util.spec_from_file_location(
            "pafi_repro", ROOT / "repro_zero_width_reactant_partition.py"
        )
        reproducer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reproducer)
        arguments = reproducer.command("native")
        self.assertEqual(arguments[0], "native")
        flags = dict(zip(arguments[1::2], arguments[2::2], strict=True))
        configuration = ANALYZE.REPRODUCED_REJECTION
        self.assertEqual(len(flags), len(configuration))
        for key, flag in reproducer.FLAGS.items():
            self.assertEqual(flags[flag], str(configuration[key]))


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
