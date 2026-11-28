#!/usr/bin/env bash
# Print an rgpot tree that contains CppCore/rgpot/EAMTable/EAMTablePot.cc.
# Honour RGPOT_DIR when it is already set. Otherwise clone the pinned
# tree into the workspace target directory and replace a checkout whose
# HEAD is not that revision. The sparse set is CppCore plus include, so
# a revision that has the plugin header fetches it.
set -euo pipefail

if [[ -n "${RGPOT_DIR:-}" ]]; then
  printf '%s\n' "$RGPOT_DIR"
  exit 0
fi

pin=ae6a2cfec686b06cb5921f027b6b56763c67bd91
root="$(cd "$(dirname "$0")/.." && pwd)/target/rgpot-cpp"
marker="$root/CppCore/rgpot/EAMTable/EAMTablePot.cc"
current=""
if [[ -d "$root/.git" || -f "$root/.git" ]]; then
  current="$(git -C "$root" rev-parse HEAD 2>/dev/null || true)"
fi
if [[ "$current" != "$pin" || ! -f "$marker" ]]; then
  rm -rf "$root"
  git clone --filter=blob:none --sparse https://github.com/OmniPotentRPC/rgpot.git "$root"
  git -C "$root" sparse-checkout set CppCore include
  git -C "$root" checkout "$pin"
fi
test "$(git -C "$root" rev-parse HEAD)" = "$pin"
test -f "$marker"
printf '%s\n' "$root"
