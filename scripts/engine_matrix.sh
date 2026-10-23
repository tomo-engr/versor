#!/bin/bash
# Matches against Edax and Egaroucid at fixed depth: versor at depth 8
# and 10 against three levels of each engine, 1000 games per cell,
# versor switching to the exact solver at 20 empties. Each cell has a
# fixed seed:
#   edax      depth 8 -> seeds 51/52/53, depth 10 -> 54/55/56
#   egaroucid depth 8 -> seeds 81/82/83, depth 10 -> 84/85/86
#
# Needs `cargo build --release` and the engine binaries named in
# config/engines.json. Runs one cell at a time with 2 versor threads.
# Each cell writes its own file; results.txt is the running summary.
# A cell whose file already has a result is skipped, so the script can
# be restarted.

set -u
cd "$(dirname "$0")/.."
BIN=./target/release/versor-cli
W=models/pattern_v5.bin
OUT=data/engine_matrix
mkdir -p "$OUT"

run_cell() {
    local engine=$1 level=$2 depth=$3 seed=$4 solve=$5
    local tag="${engine}-L${level}-d${depth}-solve${solve}"
    local f="$OUT/$tag.txt"
    if [ -s "$f" ] && grep -q '^elo ' "$f"; then
        echo "skip $tag (already done)"
        return
    fi
    echo "start $tag $(date -Is)" >> "$OUT/results.txt"
    $BIN enginematch --engine "$engine" --level "$level" --weights "$W" \
        --games 1000 --depth "$depth" --seed "$seed" --threads 2 \
        --tt 32 --solve-at "$solve" > "$f" 2>&1
    echo "done  $tag $(date -Is) rc=$?" >> "$OUT/results.txt"
    cat "$f" >> "$OUT/results.txt"
    echo >> "$OUT/results.txt"
}

# Most informative cells first: the levels where versor is near even.
run_cell edax      8 8  53 20
run_cell egaroucid 7 8  83 20
run_cell edax      5 8  52 20
run_cell egaroucid 5 8  82 20
run_cell edax      3 8  51 20
run_cell egaroucid 3 8  81 20

run_cell edax      8 10 56 20
run_cell egaroucid 7 10 86 20
run_cell edax      5 10 55 20
run_cell egaroucid 5 10 85 20
run_cell edax      3 10 54 20
run_cell egaroucid 3 10 84 20

# Depth 10 again with the solver switched off (--solve-at 0), same
# weights and seeds: the paired control for the effect of the solver.
run_cell edax      8 10 56 0
run_cell egaroucid 7 10 86 0
run_cell edax      5 10 55 0
run_cell egaroucid 5 10 85 0
run_cell edax      3 10 54 0
run_cell egaroucid 3 10 84 0

echo "all cells finished $(date -Is)" >> "$OUT/results.txt"
