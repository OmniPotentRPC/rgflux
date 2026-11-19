#!/usr/bin/env python3
"""Fail when the public Rust API inventory in the Org reference is stale."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    command = [sys.executable, "scripts/generate_api_inventory.py", "--check"]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
