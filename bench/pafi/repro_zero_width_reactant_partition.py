#!/usr/bin/env python3
import argparse
import importlib.util
import json
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("pafi_analyze", ROOT / "analyze.py")
ANALYZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZE)

FLAGS = {
    "case": "--case",
    "planes": "--planes",
    "temperature": "--temperature",
    "samples": "--samples",
    "burn_in": "--burn-in",
    "seed": "--seed",
    "threads": "--threads",
    "effective_threads": "--effective-threads",
    "time_step": "--time-step",
    "friction": "--friction",
    "zero_temperature_tolerance": "--zero-temperature-tolerance",
}


def command(native_bin):
    configuration = ANALYZE.REPRODUCED_REJECTION
    assert set(configuration) == set(FLAGS)
    arguments = [str(native_bin)]
    for key, flag in FLAGS.items():
        arguments.extend([flag, str(configuration[key])])
    return arguments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-bin", type=pathlib.Path, required=True)
    args = parser.parse_args()
    completed = subprocess.run(
        command(args.native_bin),
        check=True,
        capture_output=True,
        text=True,
    )
    record = json.loads(completed.stdout)
    if (
        record.get("status") != "rejected"
        or record.get("error") != ANALYZE.REACTANT_PARTITION_ERROR
    ):
        raise SystemExit(
            f"the retained reactant partition rejection was not reproduced: {record}"
        )
    echoed = {
        key: record.get("requested_threads" if key == "threads" else key)
        for key in ANALYZE.REPRODUCED_REJECTION
    }
    if echoed != ANALYZE.REPRODUCED_REJECTION:
        raise SystemExit(
            f"the rejected run is not the classified configuration: {record}"
        )
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
