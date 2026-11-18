#!/usr/bin/env python3
import argparse
import pathlib
import shutil
import subprocess


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


def main():
    args = arguments()
    args.build_dir.mkdir(parents=True, exist_ok=False)
    source_dir = args.build_dir / "src"
    source_dir.mkdir()
    shutil.copyfile(args.source, source_dir / "main.rs")
    manifest = f"""[package]
name = "rgflux-pafi-bench"
version = "0.1.0"
edition = "2024"
publish = false

[dependencies]
ndarray = "0.17"
pyo3 = {{ version = "0.27", features = ["auto-initialize"] }}
rgflux = {{ path = "{toml_path(args.rgflux_source)}" }}
rgpot-pair = {{ path = "{toml_path(args.rgpot_source / "rgpot-pair")}" }}
rgsaddle = {{ path = "{toml_path(args.rgsaddle_source)}" }}
serde = {{ version = "1", features = ["derive"] }}
serde_json = "1"
"""
    (args.build_dir / "Cargo.toml").write_text(manifest)
    subprocess.run(
        [
            str(args.cargo),
            "generate-lockfile",
            "--manifest-path",
            str(args.build_dir / "Cargo.toml"),
        ],
        check=True,
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
