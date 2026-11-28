#!/usr/bin/env bash
# The examples of Swinburne and Marinica's reference PAFI code, rerun with
# rgsaddle's PAFI.
#
# Fetches EAM-VAC-W and EAM-SIA-Fe from github.com/tomswinburne/pafi at a
# pinned commit, checks them, and runs the ignored tests in
# tests/pafi_reference.rs: one writes benchmarks/pafi_reference.csv and
# benchmarks/pafi_reference_planes.csv, and one samples six W planes at two
# time steps and prints them. With PAFI_REFERENCE_CODE=1 it also
# runs the reference code on the W example through
# benchmarks/pafi_reference/reference_code.py, which needs the pafi package,
# the LAMMPS Python module and mpi4py, and writes
# benchmarks/pafi_reference_code.csv and
# benchmarks/pafi_reference_code_planes.csv. PAFI_RANKS sets the reference
# workers (1, the committed tables; more run under mpirun, each adding its
# own trajectories) and PYTHON the interpreter (python3).
set -euo pipefail

commit=1661a1d61b2ccbbd33ec4c9656d60947b8b74cc1
source_url="https://raw.githubusercontent.com/tomswinburne/pafi/${commit}/examples/systems"
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
work=$(mktemp -d)
cleanup() {
  rm -r -- "${work:?}"
}
trap cleanup EXIT

fetch() {
  local system=$1 file=$2 sum=$3
  mkdir -p "${work}/systems/${system}"
  curl -fsSL --retry 4 "${source_url}/${system}/${file}" -o "${work}/systems/${system}/${file}"
  echo "${sum}  ${work}/systems/${system}/${file}" | sha256sum -c --quiet -
}

while read -r system file sum; do
  fetch "${system}" "${file}" "${sum}"
done <<'EOF'
EAM-VAC-W W.eam.fs 88a64fa43fde6617b3bd66797c6295bcd036b242ba60f8bd591fc5a64f12197e
EAM-VAC-W image_0.dat b241002228887c8ce802ede6d3199a266d0116022781c58fcc3cbd1e631b542e
EAM-VAC-W image_1.dat 277acfd1af606b303f8ecf6636d2aeca6771fc3f4c62d5d84250811e9eba626d
EAM-VAC-W image_2.dat a52dc9a6aa91db355c0885e1dc1e8fab957bf8c0159133ebff7c42022c4949fe
EAM-VAC-W image_3.dat a6b06bea25bb03b0019210a206a02345eedeb94c40bbde345c70d66252f1a3ed
EAM-VAC-W image_4.dat 92eecf044746d4dea7b47dcc089e1ffa2f45a9c8b147d1a407fd910634fca782
EAM-VAC-W image_5.dat 1e7399ee975d0f289829ea460f14bbc7ac8a55cd9a22d9b8850e061bb7088d78
EAM-VAC-W image_6.dat 51c100776597a2c58313af1e64537abecd3bdc4346557c9ddb954c9659943bbd
EAM-VAC-W image_7.dat 31074852e52d398c6ffb2edda6ccc124e1a8d2489b829b1ccc201661986f4829
EAM-VAC-W image_8.dat e006e4ec2c8962544d60eb4eb598f13bf6643d7e1a3b5f1f8481dc176bc0355c
EAM-SIA-Fe Fe.eam.fs bbc37a6417e461f832bf679f32b3c1b6871f1f11bc4339c85e9451df57b74ae4
EAM-SIA-Fe image_1.dat 6a3e254c8d882b2a997c93ebc6dfd3de533b589484905de11a3d8dcb6d015566
EAM-SIA-Fe image_2.dat 1e5d74972adb8ea32f8927e9af9730975cb6e5883b09ff95dff7016c04a03337
EAM-SIA-Fe image_3.dat 531a91bfb307dd6eceaacbe41bff0beb8ebf8dc1b05858ff79cd19ef38d70a76
EAM-SIA-Fe image_4.dat d3b9cf4e267623917763eccf45a3d71a9144ee47cb6e0096581ad1a2cc253d8a
EAM-SIA-Fe image_5.dat b29f5e2d4c2d88fa0e6e352ca74b83943b990b8ba06cc98b8a2cc322eddf42d7
EAM-SIA-Fe image_6.dat cd783e608c1c4ecd8f1a5399103801c3c08db95376c18d87fa1b6dd46813ac22
EAM-SIA-Fe image_7.dat fc6778ef7ee64ad53dce75ebf8926b6dd73bd1760ea9804d9b5183cf1c0346be
EAM-SIA-Fe image_8.dat a5d11574274213a50be862423cde463cd28f19198f95cf28ab1ad20e2f10034b
EAM-SIA-Fe image_9.dat 7c37c6379059e5a8cacd6e320f96733a54594fcf771c762891a7ef94315a6c50
EOF

cd "${root}"
RGSADDLE_PAFI_REFERENCE="${work}/systems" cargo test --release --test pafi_reference \
  reference_examples_reproduce -- --ignored --nocapture
RGSADDLE_PAFI_REFERENCE="${work}/systems" cargo test --release --test pafi_reference \
  reference_step_size -- --ignored --nocapture

if [ "${PAFI_REFERENCE_CODE:-0}" = 1 ]; then
  python=${PYTHON:-python3}
  ranks=${PAFI_RANKS:-1}
  launch=()
  if [ "${ranks}" -gt 1 ]; then
    launch=(mpirun -np "${ranks}")
  fi
  script="${root}/benchmarks/pafi_reference/reference_code.py"
  while read -r name temperatures repeats planes; do
    ${launch[@]+"${launch[@]}"} "${python}" "${script}" run "${work}/systems" \
      "${work}/runs/${name}" "${temperatures}" "${repeats}" "${planes}" </dev/null
  done <<'EOF'
zero_9 0 1 9
zero_33 0 1 33
hot_9 1000,2000 32 9
hot_33 1000,2000 16 33
EOF
  "${python}" "${script}" collate "${root}/benchmarks/pafi_reference_code.csv" \
    "${root}/benchmarks/pafi_reference_code_planes.csv" \
    "${work}/runs/zero_9" "${work}/runs/zero_33" "${work}/runs/hot_9" "${work}/runs/hot_33"
fi
