#!/usr/bin/env python3
import json
import os
import pathlib
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = pathlib.Path(__file__).resolve().parent
RAW = ROOT / "raw"


def run(command, **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def git_repository(path, filename, text):
    path.mkdir()
    run(["git", "init", "-q", path])
    run(["git", "-C", path, "config", "user.email", "rgoswami@ieee.org"])
    run(["git", "-C", path, "config", "user.name", "Rohit Goswami"])
    run(["git", "-C", path, "config", "commit.gpgsign", "false"])
    target = path / filename
    target.write_text(text)
    run(["git", "-C", path, "add", filename])
    run(
        [
            "git",
            "-C",
            path,
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-q",
            "-m",
            "init",
        ]
    )
    return path


def write_script(path, body):
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


class RetainedInputTests(unittest.TestCase):
    def test_retained_static_reference_matches_published_results(self):
        result = json.loads((ROOT / "results.json").read_text())
        static = json.loads((RAW / "vacancy-static.json").read_text())
        published = result["fcc_vacancy"]["static_energies"]
        self.assertEqual(static["system"], "fcc-vacancy")
        self.assertEqual(static["rgflux"], published["rgflux"])
        self.assertEqual(static["lammps"], published["lammps"])

    def test_retained_vacancy_path_matches_published_neb(self):
        result = json.loads((ROOT / "results.json").read_text())
        path = json.loads((RAW / "vacancy-path.json").read_text())
        neb = result["fcc_vacancy"]["neb"]
        self.assertEqual(len(path["images"]), neb["images"])
        self.assertEqual(
            len(path["atomic_numbers"]), result["fcc_vacancy"]["atom_count"]
        )
        self.assertEqual(path["neb_iterations"], neb["iterations"])
        self.assertEqual(path["neb_max_force"], neb["max_force"])
        self.assertEqual(len(path["images"][0]), 3 * len(path["atomic_numbers"]))


class MissingInputTests(unittest.TestCase):
    def analyze(self, root):
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "analyze.py"),
                "--metadata",
                str(root / "metadata.json"),
                "--raw-dir",
                str(root / "dimer"),
                "--vacancy-raw-dir",
                str(root / "vacancy"),
                "--vacancy-path",
                str(root / "vacancy-path.json"),
                "--vacancy-static-reference",
                str(root / "vacancy-static.json"),
                "--output-json",
                str(root / "out.json"),
                "--output-markdown",
                str(root / "out.md"),
            ],
            capture_output=True,
            text=True,
        )

    def test_missing_metadata_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            completed = self.analyze(pathlib.Path(tmp))
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("metadata.json is missing", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)

    def test_missing_raw_record_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "metadata.json").write_text("{}\n")
            (root / "vacancy-path.json").write_text(
                json.dumps(
                    {
                        "atomic_numbers": [],
                        "generator_sha": "0" * 40,
                        "images": [],
                        "neb_iterations": 0,
                        "neb_max_force": 0.0,
                    }
                )
                + "\n"
            )
            (root / "vacancy-static.json").write_text(
                json.dumps(
                    {
                        "lammps": {
                            "image_energies": [],
                            "minimum_energy": 0.0,
                            "saddle_energy": 0.0,
                        },
                        "rgflux": {
                            "image_energies": [],
                            "minimum_energy": 0.0,
                            "saddle_energy": 0.0,
                        },
                    }
                )
                + "\n"
            )
            vacancy = root / "vacancy"
            vacancy.mkdir()
            (vacancy / "manifest.json").write_text("[]\n")
            dimer = root / "dimer"
            (dimer / "records").mkdir(parents=True)
            (dimer / "manifest.json").write_text(
                json.dumps(
                    [
                        {
                            "record": "records/absent.json",
                            "engine": "rgflux",
                        }
                    ]
                )
                + "\n"
            )
            completed = self.analyze(root)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("raw record is missing", completed.stderr)
        self.assertIn("records/absent.json", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)

    def test_analyze_defaults_to_the_stable_results_directory(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "analyze.py"), "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
        collapsed = "".join(completed.stdout.split())
        for path in (
            RAW / "metadata.json",
            RAW / "dimer",
            RAW / "vacancy",
            RAW / "vacancy-path.json",
            RAW / "vacancy-static.json",
        ):
            self.assertIn(str(path), collapsed)


class LockedBuildTests(unittest.TestCase):
    def test_native_build_pins_versions_and_uses_the_lockfile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            log = root / "cargo.log"
            cargo = root / "cargo"
            write_script(
                cargo,
                "#!/bin/sh\nprintf '%s\\n' \"$@\" >> \"$CARGO_LOG\"\nexit 0\n",
            )
            rgflux = git_repository(root / "rgflux", "README", "rgflux\n")
            rgpot = git_repository(root / "rgpot", "README", "rgpot\n")
            (rgpot / "rgpot-pair").mkdir()
            rgsaddle = git_repository(root / "rgsaddle", "README", "rgsaddle\n")
            build = root / "build"
            env = os.environ.copy()
            env["CARGO_LOG"] = str(log)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "build_native.py"),
                    "--cargo",
                    str(cargo),
                    "--rgflux-source",
                    str(rgflux),
                    "--rgpot-source",
                    str(rgpot),
                    "--rgsaddle-source",
                    str(rgsaddle),
                    "--source",
                    str(ROOT / "src" / "main.rs"),
                    "--build-dir",
                    str(build),
                ],
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            manifest = (build / "Cargo.toml").read_text()
            self.assertIn('ndarray = "=0.17.2"', manifest)
            self.assertIn('version = "=0.27.2"', manifest)
            self.assertIn('version = "=1.0.229"', manifest)
            self.assertIn('serde_json = "=1.0.154"', manifest)
            self.assertNotIn('ndarray = "0.17"', manifest)
            self.assertNotIn('version = "0.27"', manifest)
            self.assertNotIn('version = "1"', manifest)
            invoked = log.read_text()
            self.assertNotIn("generate-lockfile", invoked)
            self.assertIn("build", invoked)
            self.assertIn("--locked", invoked)
            self.assertIn("--release", invoked)
            self.assertEqual(
                (build / "Cargo.lock").read_bytes(),
                (ROOT / "Cargo.lock").read_bytes(),
            )
            recorded = json.loads((build / "source-commits.json").read_text())
            for name, path in (
                ("rgflux", rgflux),
                ("rgpot", rgpot),
                ("rgsaddle", rgsaddle),
            ):
                head = subprocess.check_output(
                    ["git", "-C", str(path), "rev-parse", "HEAD"],
                    text=True,
                ).strip()
                self.assertEqual(recorded[name], head)


class CampaignRecordTests(unittest.TestCase):
    def test_campaign_and_reference_write_metadata_and_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            rgflux = git_repository(
                root / "rgflux",
                "Cargo.toml",
                textwrap.dedent(
                    """\
                    [dependencies]
                    rgmin = { version = "=0.6.0", git = "https://example.invalid/rgmin", rev = "0123456789abcdef0123456789abcdef01234567" }
                    """
                ),
            )
            rgpot = git_repository(root / "rgpot", "README", "rgpot\n")
            rgsaddle = git_repository(root / "rgsaddle", "README", "rgsaddle\n")
            lammps = git_repository(root / "lammps", "README", "lammps\n")
            pafi = git_repository(root / "pafi", "README", "pafi\n")
            native = root / "native"
            write_script(
                native,
                textwrap.dedent(
                    """\
                    #!/usr/bin/env python3
                    import json
                    print(json.dumps({"engine": "rgflux", "status": "ok"}))
                    """
                ),
            )
            raw = root / "raw-dimer"
            metadata = root / "metadata.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "run_campaign.py"),
                    "--native-bin",
                    str(native),
                    "--python",
                    sys.executable,
                    "--reference-script",
                    str(ROOT / "run_reference.py"),
                    "--raw-dir",
                    str(raw),
                    "--metadata",
                    str(metadata),
                    "--rgflux-source",
                    str(rgflux),
                    "--rgpot-source",
                    str(rgpot),
                    "--rgsaddle-source",
                    str(rgsaddle),
                    "--lammps-source",
                    str(lammps),
                    "--pafi-source",
                    str(pafi),
                    "--straight-planes",
                    "9",
                    "--curved-planes",
                    "17",
                    "--temperatures",
                    "0.05",
                    "--sample-counts",
                    "64",
                    "--burn-in",
                    "128",
                    "--replicates",
                    "12",
                    "--time-step",
                    "0.002",
                    "--friction",
                    "1.0",
                    "--zero-temperature-tolerance",
                    "0.01",
                    "--high-threads",
                    "4",
                    "--native-effective-threads",
                    "1",
                    "--one-core-set",
                    "0",
                    "--high-core-set",
                    "0",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(metadata.is_file())
            recorded = json.loads(metadata.read_text())
            self.assertEqual(
                recorded["source_commits"]["rgflux"],
                subprocess.check_output(
                    ["git", "-C", str(rgflux), "rev-parse", "HEAD"], text=True
                ).strip(),
            )
            self.assertEqual(
                recorded["rgmin_sha"],
                "0123456789abcdef0123456789abcdef01234567",
            )
            self.assertIn("host", recorded)
            self.assertIn("hostname", recorded["host"])
            self.assertEqual(len(recorded["seeds"]["dimer"]), 48)
            self.assertEqual(len(recorded["command_lines"]["dimer"]), 97)
            self.assertEqual(
                recorded["versions"]["native_harness_crates"]["ndarray"],
                "0.17.2",
            )
            lammps_head = subprocess.check_output(
                ["git", "-C", str(lammps), "rev-parse", "HEAD"], text=True
            ).strip()
            pafi_head = subprocess.check_output(
                ["git", "-C", str(pafi), "rev-parse", "HEAD"], text=True
            ).strip()
            self.assertEqual(recorded["lammps_sha"], lammps_head)
            self.assertEqual(recorded["pafi_sha"], pafi_head)
            self.assertEqual(recorded["source_commits"]["lammps"], lammps_head)
            self.assertEqual(recorded["source_commits"]["pafi"], pafi_head)
            manifest = json.loads((raw / "manifest.json").read_text())
            self.assertEqual(len(manifest), 96)
            for item in manifest:
                record_path = raw / item["record"]
                self.assertTrue(record_path.is_file(), item["record"])
                self.assertTrue(record_path.read_text().strip())
            reference_metadata = list((raw / "work").glob("*/metadata.json"))
            self.assertEqual(len(reference_metadata), 48)
            sample = json.loads(reference_metadata[0].read_text())
            self.assertEqual(len(sample["seeds"]), 1)
            self.assertIn("command_lines", sample)
            self.assertIn("host", sample)
            self.assertEqual(sample["source_commits"]["rgpot"], recorded["rgpot_sha"])

            vacancy_raw = root / "vacancy"
            vacancy_path = root / "vacancy-path.json"
            vacancy_path.write_text('{"images": [], "cell": [1, 0, 0, 0, 1, 0, 0, 0, 1]}\n')
            vacancy_run = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "run_vacancy_campaign.py"),
                    "--native-bin",
                    str(native),
                    "--site-packages",
                    str(root / "site-packages"),
                    "--path",
                    str(vacancy_path),
                    "--python",
                    sys.executable,
                    "--reference-script",
                    str(ROOT / "run_vacancy_reference.py"),
                    "--raw-dir",
                    str(vacancy_raw),
                    "--metadata",
                    str(metadata),
                    "--rgflux-source",
                    str(rgflux),
                    "--rgpot-source",
                    str(rgpot),
                    "--rgsaddle-source",
                    str(rgsaddle),
                    "--lammps-source",
                    str(lammps),
                    "--pafi-source",
                    str(pafi),
                    "--planes",
                    "17",
                    "--temperatures",
                    "0.025",
                    "--samples",
                    "4",
                    "--burn-in",
                    "1",
                    "--replicates",
                    "12",
                    "--time-step",
                    "0.002",
                    "--friction",
                    "1.0",
                    "--zero-temperature-tolerance",
                    "0.01",
                    "--high-threads",
                    "4",
                    "--native-effective-threads",
                    "1",
                    "--one-core-set",
                    "0",
                    "--high-core-set",
                    "0",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(vacancy_run.returncode, 0, vacancy_run.stderr)
            merged = json.loads(metadata.read_text())
            self.assertEqual(len(merged["seeds"]["dimer"]), 48)
            self.assertEqual(len(merged["seeds"]["vacancy"]), 24)
            self.assertEqual(len(merged["command_lines"]["vacancy"]), 49)
            vacancy_manifest = json.loads((vacancy_raw / "manifest.json").read_text())
            self.assertEqual(len(vacancy_manifest), 48)
            for item in vacancy_manifest:
                self.assertTrue((vacancy_raw / item["record"]).is_file())
            self.assertEqual(
                len(list((vacancy_raw / "work").glob("*/metadata.json"))), 24
            )

            absent = raw / manifest[0]["record"]
            absent.unlink()
            analysis = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "analyze.py"),
                    "--metadata",
                    str(metadata),
                    "--raw-dir",
                    str(raw),
                    "--vacancy-raw-dir",
                    str(root / "vacancy"),
                    "--vacancy-path",
                    str(root / "vacancy-path.json"),
                    "--vacancy-static-reference",
                    str(root / "vacancy-static.json"),
                    "--output-json",
                    str(root / "out.json"),
                    "--output-markdown",
                    str(root / "out.md"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(analysis.returncode, 0)
            self.assertIn("raw record is missing", analysis.stderr)
            self.assertNotIn("Traceback", analysis.stderr)


if __name__ == "__main__":
    unittest.main()
