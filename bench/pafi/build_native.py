#!/usr/bin/env python3
import argparse
import pathlib
import shutil
import subprocess
import sys


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from provenance import LOCKFILE, NATIVE_CRATES, git_commit, write_json


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cargo", type=pathlib.Path, required=True)
    parser.add_argument("--rgflux-source", type=pathlib.Path, required=True)
    parser.add_argument("--rgpot-source", type=pathlib.Path, required=True)
    parser.add_argument("--rgsaddle-source", type=pathlib.Path, required=True)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--build-dir", type=pathlib.Path, required=True)
    return parser.parse_args()


def toml_path(path):
    return str(path.resolve()).replace("\\", "\\\\").replace('"', '\\"')


def render_manifest(args):
    return f"""[package]
name = "rgflux-pafi-bench"
version = "0.1.0"
edition = "2024"
publish = false

[dependencies]
ndarray = "={NATIVE_CRATES["ndarray"]}"
pyo3 = {{ version = "={NATIVE_CRATES["pyo3"]}", features = ["auto-initialize"] }}
rgflux = {{ path = "{toml_path(args.rgflux_source)}" }}
rgpot-pair = {{ path = "{toml_path(args.rgpot_source / "rgpot-pair")}" }}
rgsaddle = {{ path = "{toml_path(args.rgsaddle_source)}" }}
serde = {{ version = "={NATIVE_CRATES["serde"]}", features = ["derive"] }}
serde_json = "={NATIVE_CRATES["serde_json"]}"
"""


def main():
    args = arguments()
    if not LOCKFILE.is_file():
        raise SystemExit(f"committed Cargo.lock is missing: {LOCKFILE}")
    args.build_dir.mkdir(parents=True, exist_ok=False)
    source_dir = args.build_dir / "src"
    source_dir.mkdir()
    shutil.copyfile(args.source, source_dir / "main.rs")
    (args.build_dir / "Cargo.toml").write_text(render_manifest(args))
    shutil.copyfile(LOCKFILE, args.build_dir / "Cargo.lock")
    write_json(
        args.build_dir / "source-commits.json",
        {
            "rgflux": git_commit(args.rgflux_source),
            "rgpot": git_commit(args.rgpot_source),
            "rgsaddle": git_commit(args.rgsaddle_source),
        },
    )
    subprocess.run(
        [
            str(args.cargo),
            "build",
            "--release",
            "--locked",
            "--manifest-path",
            str(args.build_dir / "Cargo.toml"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
