#!/usr/bin/env python3
import argparse
import json
import pathlib
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-bin", type=pathlib.Path, required=True)
    args = parser.parse_args()
    completed = subprocess.run(
        [
            str(args.native_bin),
            "--case",
            "curved",
            "--planes",
            "17",
            "--temperature",
            "0.05",
            "--samples",
            "64",
            "--burn-in",
            "128",
            "--seed",
            "1036330",
            "--threads",
            "1",
            "--time-step",
            "0.002",
            "--friction",
            "1.0",
            "--zero-temperature-tolerance",
            "0.01",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    record = json.loads(completed.stdout)
    expected = "invalid configuration: PAFI reactant partition must be finite and positive"
    if record.get("status") != "rejected" or record.get("error") != expected:
        raise SystemExit(
            "zero-width reactant partition at dividing plane index 0 "
            f"was not reproduced: {record}"
        )
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
