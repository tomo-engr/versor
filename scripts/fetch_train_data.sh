#!/bin/sh
# Download the training data into $VERSOR_DATA, a directory of your
# choice outside the repository (data files must never be committed).
# Sources:
#   - WThor database, French Othello Federation (ffothello.org)
#   - Egaroucid Free Training Data (Yamana, Takuto), redistribution
#     prohibited; downloaded zips and extracted files stay local.
# Zips are kept after extraction.

set -eu

base="${VERSOR_DATA:?set VERSOR_DATA to the directory for the training data}"
mkdir -p "$base/wthor" "$base/egaroucid"
base="$(cd "$base" && pwd)"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# WThor: all year files plus player/tournament tables.
cd "$base/wthor"
for f in WTH_7789.ZIP $(seq 1990 2025 | sed 's/^/WTH_/;s/$/.ZIP/'); do
    if [ ! -f "$f" ]; then
        log "fetching $f"
        curl -sfL -o "$f" "https://www.ffothello.org/wthor/base_zip/$f"
    fi
done
for f in WTHOR.JOU WTHOR.TRN; do
    [ -f "$f" ] || curl -sfL -o "$f" "https://www.ffothello.org/wthor/base/$f"
done
mkdir -p extracted
for z in WTH_*.ZIP; do
    unzip -o -q "$z" -d extracted
done
log "wthor done: $(ls extracted | wc -l) files"

# Egaroucid Free Training Data: v0002 transcript series and the 2023
# transcript archive.
cd "$base/egaroucid"
for url in \
    "https://github.com/Nyanyan/Egaroucid/releases/download/training_data_v0002/Egaroucid_Train_Data_v0002_0.zip" \
    "https://github.com/Nyanyan/Egaroucid/releases/download/training_data_v0002/Egaroucid_Train_Data_v0002_1.zip" \
    "https://github.com/Nyanyan/Egaroucid/releases/download/transcript/Egaroucid_Transcript.zip"; do
    f="$(basename "$url")"
    if [ ! -f "$f" ]; then
        log "fetching $f"
        curl -sfL -o "$f" "$url"
    fi
    d="extracted_${f%.zip}"
    if [ ! -d "$d" ]; then
        log "extracting $f"
        mkdir -p "$d"
        unzip -o -q "$f" -d "$d"
    fi
done
log "egaroucid done"
du -sh "$base"/* 2>/dev/null
log "all done"
