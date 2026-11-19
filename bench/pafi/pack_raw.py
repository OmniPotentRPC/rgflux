#!/usr/bin/env python3
"""Pack a campaign's manifest and raw records into one xz archive.

The archive is deterministic: members are sorted and carry no owner,
mode or time information, so packing the same records twice gives the
same bytes. Logs and per-run work directories are not packed.
"""

import argparse
import io
import json
import pathlib
import tarfile


def member_names(raw_dir):
    manifest = json.loads((raw_dir / "manifest.json").read_text())
    return ["manifest.json"] + sorted(item["record"] for item in manifest)


def pack(raw_dir, output):
    raw_dir = pathlib.Path(raw_dir)
    output = pathlib.Path(output)
    names = member_names(raw_dir)
    if len(set(names)) != len(names):
        raise SystemExit("manifest names a record more than once")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:xz", preset=9) as bundle:
        for name in names:
            path = raw_dir / name
            if not path.is_file():
                raise SystemExit(f"raw record is missing: {path}")
            member = tarfile.TarInfo(name)
            data = path.read_bytes()
            member.size = len(data)
            member.mode = 0o644
            member.mtime = 0
            member.uid = member.gid = 0
            member.uname = member.gname = ""
            bundle.addfile(member, fileobj=io.BytesIO(data))
    return len(names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    count = pack(args.raw_dir, args.output)
    print(f"packed {count} files into {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
