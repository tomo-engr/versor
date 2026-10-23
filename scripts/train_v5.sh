#!/bin/sh
# The recipe of models/pattern_v5.bin: prepare the training data with
# the 15-table pattern set, then train on all of it (seed 42, at most 24
# epochs). Sequential; <= 4 threads. Needs `cargo build --release` and
# the data fetched by scripts/fetch_train_data.sh into $VERSOR_DATA,
# where the prepared data and the weights are written too. The Python
# interpreter is $PYTHON (default python3) with numpy installed.
set -eu
D="${VERSOR_DATA:?set VERSOR_DATA to the directory used by scripts/fetch_train_data.sh}"
D="$(cd "$D" && pwd)"
cd "$(dirname "$0")/.."
echo "[v5] dataprep start $(date '+%F %T')"
./target/release/dataprep \
    --wthor "$D/wthor/extracted" \
    --egaroucid-v2 "$D/egaroucid/extracted_Egaroucid_Train_Data_v0002_0" \
    --egaroucid-v2 "$D/egaroucid/extracted_Egaroucid_Train_Data_v0002_1" \
    --transcript "$D/egaroucid/extracted_Egaroucid_Transcript" \
    --out "$D/prep5" --threads 4 --shards 32 --dump-boards 300
echo "[v5] dataprep done $(date '+%F %T')"
"${PYTHON:-python3}" python/train.py --data "$D/prep5" --fraction 1.0 \
    --out "$D/models/w5_f100.bin" --log "$D/models/log_v5_f100.json" \
    --testvec "$D/models/testvec_v5_f100.txt" --seed 42 --max-epochs 24
echo "[v5] train done $(date '+%F %T')"
