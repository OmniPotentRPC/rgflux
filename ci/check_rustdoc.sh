#!/usr/bin/env bash
# Build rustdoc with every warning denied across feature sets, then with
# private items documented so broken intra-doc links in private items fail.
set -euo pipefail

export RUSTDOCFLAGS="-D warnings"

feature_sets=("" "rgpot" "rgpot-session" "rgpot,rgpot-session")

for features in "${feature_sets[@]}"; do
  echo "== rustdoc features: ${features:-default}"
  if [[ -n "$features" ]]; then
    cargo doc --locked --no-deps --features "$features"
  else
    cargo doc --locked --no-deps
  fi
done

echo "== rustdoc private items, library features"
# `lammpc` links liblammpc. The lammpc CI job documents that feature.
cargo doc --locked --no-deps --document-private-items --features rgpot,rgpot-session
