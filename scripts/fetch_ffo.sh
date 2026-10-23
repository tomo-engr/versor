#!/bin/sh
# Fetch the FFO endgame test suite (problems 40-59) from radagast.se
# (Gunnar Andersson's Zebra site; the GPL engine repositories are not
# used as a source) and convert it to the versor-cli ffo input format:
#   id;board64;stm;score
# Expected scores (for the side to move) are from the Zebra results
# table on http://radagast.se/othello/ffotest.html.
# Problems 1-39 are FForum problems with no known source outside the
# GPL engine repositories, so they are not fetched.

set -eu

dir="$(dirname "$0")/../data/ffo"
mkdir -p "$dir/raw"
cd "$dir"

if [ ! -f raw/end40.pos ]; then
    curl -s -m 120 -o raw/ffotest.zip http://radagast.se/othello/ffotest.zip
    unzip -o -q raw/ffotest.zip -d raw
fi

# Expected exact scores for the side to move, problems 40-59.
scores="40:+38 41:+0 42:+6 43:-12 44:-14 45:+6 46:-8 47:+4 48:+28 49:+16
50:+10 51:+6 52:+0 53:-2 54:-2 55:+0 56:+2 57:-10 58:+4 59:+64"

out=ffo_40_59.txt
{
    echo "# FFO endgame test suite problems 40-59"
    echo "# source: http://radagast.se/othello/ffotest.zip"
    echo "# scores: Zebra results table on http://radagast.se/othello/ffotest.html"
    echo "# format: id;board64(a1..h8, X=black, O=white, -=empty);side to move;score for side to move"
    for entry in $scores; do
        id="${entry%%:*}"
        score="${entry#*:}"
        board="$(head -1 "raw/end${id}.pos" | tr -d '\r')"
        side_line="$(sed -n 2p "raw/end${id}.pos" | tr -d '\r')"
        case "$side_line" in
            Black*) stm=X ;;
            White*) stm=O ;;
            *) echo "unexpected side line in end${id}.pos: $side_line" >&2; exit 1 ;;
        esac
        echo "${id};${board};${stm};${score}"
    done
} > "$out"

echo "wrote $dir/$out"
