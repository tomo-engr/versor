# versor

versor is a reversi engine written in Rust. It represents the board with bitboards, searches with PVS, a
transposition table and Multi-ProbCut, solves endgames exactly, and evaluates positions with a pattern evaluation
trained on game records. Besides the command line interface, it has a browser UI for playing and analysing games.

versor is released under the MIT License.

## Strength

Measured with this release against Edax 4.6 and Egaroucid for Console 7.8.0, 1,000 games per row: 500 openings of 8
random moves, each played once with each colour, no opening books, one thread per engine. versor used
`models/pattern_v5.bin`, searched to a fixed depth without Multi-ProbCut and switched to the exact endgame solver at 20
empty squares; the engines were set to the levels shown. Under these conditions **versor at depth 8 is about even with
Edax level 9, and versor at depth 10 lies between Edax levels 10 and 11**. Against Egaroucid, depth 8 lies between
levels 8 and 9, and depth 10 is about even with level 10.

| versor | Opponent | Wins-draws-losses | Score | Elo difference (95% CI) | Seed |
|---|---|---|---|---|---|
| depth 8 | Edax level 8 | 538-56-406 | 0.5660 | +46.1 (+31.0 .. +61.4) | 53 |
| depth 8 | Edax level 9 | 458-54-488 | 0.4850 | -10.4 (-24.6 .. +3.7) | 109 |
| depth 8 | Edax level 10 | 372-44-584 | 0.3940 | -74.8 (-90.8 .. -59.1) | 110 |
| depth 10 | Edax level 10 | 493-60-447 | 0.5230 | +16.0 (+3.8 .. +28.3) | 210 |
| depth 10 | Edax level 11 | 417-66-517 | 0.4500 | -34.9 (-48.9 .. -20.9) | 211 |
| depth 10 | Edax level 12 | 377-47-576 | 0.4005 | -70.1 (-84.6 .. -55.8) | 212 |
| depth 8 | Egaroucid level 8 | 501-57-442 | 0.5295 | +20.5 (+6.2 .. +34.9) | 308 |
| depth 8 | Egaroucid level 9 | 433-49-518 | 0.4575 | -29.6 (-43.3 .. -16.0) | 309 |
| depth 10 | Egaroucid level 9 | 553-44-403 | 0.5750 | +52.5 (+37.5 .. +67.7) | 409 |
| depth 10 | Egaroucid level 10 | 460-53-487 | 0.4865 | -9.4 (-22.8 .. +4.0) | 410 |
| depth 10 | Egaroucid level 11 | 390-54-556 | 0.4170 | -58.2 (-71.6 .. -45.0) | 411 |

The score counts a draw as half a win; the confidence interval of the Elo difference is computed over the pairs of
games. A level is each engine's own setting (an Edax level is a search depth; an Egaroucid level also sets where the
exact endgame search starts and the Multi-ProbCut probability), so the levels of the two engines do not correspond,
and the table says nothing about versor at other settings. Each row is reproduced exactly by

```sh
./target/release/versor-cli enginematch --engine edax --level 9 --depth 8 --seed 109 --games 1000 \
    --weights models/pattern_v5.bin --tt 32 --solve-at 20 --threads 4
```

with the engine, level, depth and seed of that row (the games do not depend on `--threads`; see "Usage" for setting up
the engines).

With 30 minutes per player per game instead (100 games each, versor with Multi-ProbCut), versor lost to
Edax level 20 (score 0.4600, Elo -27.9, 95% CI -50.4 .. -5.5), to Edax level 60 (score 0.4150) and to
Egaroucid (score 0.4000; Egaroucid ignores the level when it is given a clock). The comparison depends on the
conditions, so a level quoted without them means little.

## Layout

| Directory | Contents |
|---|---|
| `versor-core/` | Board representation (bitboards), rules, move generation and flips, stable discs, perft |
| `versor-search/` | Search (PVS, transposition table, move ordering, Multi-ProbCut, exact pruning), endgame solver, time allocation |
| `versor-eval/` | Pattern evaluation, weight files, training-data preprocessing (`dataprep`) |
| `versor-cli/` | Command line interface: self-play, matches against external engines, a resident server |
| `python/` | Training the evaluation (`train.py`), fitting Multi-ProbCut (`mpc_fit.py`), the browser UI (`webui/`) |
| `models/` | Trained weights (`pattern_v1`, `v2`, `v3`, `v5`) and the Multi-ProbCut coefficients |
| `scripts/` | Fetching the training data and the FFO test, training v5, running batches of matches against external engines |

## Building and testing

You need stable Rust (tested with 1.92).

```sh
cargo build --release
cargo test --release
cargo run --release -p versor-cli -- help
```

The slow checks are skipped by default. They include a 100,000-game cross-check of the rules, deep perft, a
10,000-game check of the incremental evaluation and a comparison of the endgame solver with a full-width search; the
last one takes about 30 minutes on one core. To run them:

```sh
VERSOR_WEIGHTS=$PWD/models/pattern_v5.bin cargo test --release -- --ignored
```

Give the weight file as an absolute path, because each test runs in its own crate's directory. One of these checks,
the parity test between the Python and Rust evaluations (`versor-eval/tests/parity.rs`), also needs `VERSOR_TESTVEC`
to point at the test positions and values that `python/train.py --testvec` writes. They are generated from the
training data, so they are not in the repository (the steps are at the top of the test file). Without them, skip
that test:

```sh
VERSOR_WEIGHTS=$PWD/models/pattern_v5.bin cargo test --release -- --ignored --skip python_rust_inference_parity
```

## Python environment

The Python parts (training, the Multi-ProbCut fit, the browser UI and its tests) need Python 3.11 or later and the
packages in `requirements.txt`, which are pinned to the tested versions (numpy for training; fastapi, pydantic and
uvicorn for the UI; pytest and httpx for the UI tests). Install them into an environment of their own:

```sh
python3 -m venv .venv                  # on Debian and Ubuntu this needs the python3-venv package
. .venv/bin/activate
pip install -r requirements.txt
```

With conda instead: `conda env create -f environment.yml`, then `conda activate versor`. Tested with Python 3.11 and
3.13.

## Not included

The repository holds the code, the trained weights and the scripts. These have to be obtained separately:

| What | Needed by | How to get it |
|---|---|---|
| Edax and Egaroucid executables (GPL) | `enginematch`, `scripts/engine_matrix.sh` | From their own distributions; set `binary` in `config/engines.json` |
| FFO endgame problems #40 to #59 | `ffo` | `scripts/fetch_ffo.sh` writes `data/ffo/ffo_40_59.txt` |
| Training data: WTHOR database, Egaroucid Free Training Data | `dataprep`, `python/train.py`, `scripts/train_v5.sh` | `VERSOR_DATA=/path/to/data scripts/fetch_train_data.sh` |
| XOT openings (optional) | `--openings` of `enginematch` | By hand, from the XOT site (see below) |
| Test positions for the Python/Rust parity test | `versor-eval/tests/parity.rs` (skipped by default) | `python/train.py --testvec`, from the training data |

Without them, these commands stop with a message that names the missing file or setting.

## Usage

```sh
./target/release/versor-cli perft 9                                          # check the rules (3,005,320 at depth 9)
./target/release/versor-cli search 8 f5d6c3 --weights models/pattern_v5.bin  # analyse a position at depth 8
./target/release/versor-cli arena --weights models/pattern_v5.bin --games 10 --depth 4    # self-play, 10 games
./target/release/versor-cli help                                             # list the commands
```

A position is given as the moves played from the initial position (for example `f5d6c3`); `-` reads a position in
text format from standard input instead. The main commands are listed below; `help` shows their arguments.

| Command | What it does |
|---|---|
| `show` / `play` | Print a board; replay a sequence of moves |
| `perft` | Check the rules implementation |
| `bench` | Speed of move generation and flipping |
| `search` | Analyse a position to a given depth (with 20 or fewer empty squares, the endgame solver takes over) |
| `solve` | Solve a position to the end of the game |
| `ffo` | Run the FFO endgame test |
| `arena` | Self-play: the pattern evaluation against the hand-written one, or two weight files with `--weights-b`; at a fixed depth, or at a fixed node count with `--nodes` |
| `enginematch` | Matches against Edax and Egaroucid |
| `prune` | Check the exact pruning (same values, fewer nodes; nothing is timed) |
| `serve` | Resident server (the back end of the browser UI) |

- When a command takes weights, name the file itself, as in `--weights models/pattern_v5.bin`, so that the record of
  a run shows which weights it used. `models/pattern_best.bin` is a link to the current best weights (v5 at present)
  and is what the browser UI loads by default.
- `scripts/fetch_ffo.sh` downloads FFO problems #40 to #59 into `data/ffo/ffo_40_59.txt`. Solve them with
  `versor-cli ffo data/ffo/ffo_40_59.txt --weights models/pattern_v5.bin`.
- The opponents for `enginematch`, Edax and Egaroucid (both GPL), are not included. Get or build them from their own
  distributions (Edax: https://github.com/abulmo/edax-reversi/releases, version 4.6 was used; Egaroucid:
  https://github.com/Nyanyan/Egaroucid, Egaroucid for Console 7.8.0 was built from source) and set `binary` in
  `config/engines.json` to the executables (the `/path/to/...` entries are placeholders). If an engine cannot be
  started, `enginematch` says which file and why, and exits with status 1.
  `scripts/engine_matrix.sh` runs a batch of fixed-depth matches against several levels of each engine. To start the
  games from the XOT openings, add `--openings <file>`. Fixed-depth and fixed-node matches (`arena` and
  `enginematch`) give the same games on every run, however many games run at once (`--threads`, up to 8): every game
  starts from a fresh search state, and `enginematch` starts a new engine process for every game. `--log <file>`
  writes one line per game (pair, colour, disc difference, moves, nodes).
- The examples (`examples/`) print node counts and statistics. For instance,
  `cargo run --release -p versor-eval --example effects -- models/pattern_v5.bin` gives the node counts with each
  part of the search switched off, and `mpc_stats` collects the statistics used to fit Multi-ProbCut.

## Browser UI

A browser UI for playing and analysing games (FastAPI, with plain HTML, CSS and JavaScript).

```sh
cargo build --release                  # the UI runs the release build
. .venv/bin/activate                   # the Python environment above
cd python/webui
python -m uvicorn app:app --host 127.0.0.1 --port 8103
```

Open http://127.0.0.1:8103/. No reverse proxy is needed. The engine (`versor-cli serve`) starts with the first
request. Set `VERSOR_WEIGHTS` to use other weights and `VERSOR_BIN` to use another engine binary. By default the UI
loads `models/pattern_best.bin`, a symbolic link to the current best weights; where git does not create symbolic links
(on Windows without symbolic link support), set `VERSOR_WEIGHTS` to the absolute path of `models/pattern_v5.bin`.
Behind a reverse proxy, add `--root-path /<prefix>`. The interface is in English; Japanese can be chosen from the
language menu. Run the tests with `python -m pytest python/webui/tests`.

## Training data and training

The training data is not included. `scripts/fetch_train_data.sh` downloads the WTHOR database and the Egaroucid Free
Training Data into the directory named by `VERSOR_DATA` (sources and terms are listed under "License, third-party code
and data" below).

`scripts/train_v5.sh` preprocesses the data with `dataprep` and trains with `python/train.py`. Given the same
`VERSOR_DATA`, it reads the data from there and writes the preprocessed data and the trained weights below it. The
preprocessed data takes about 180 GB. Training runs on the CPU only (numpy). Where each set of weights came from is
described in `models/README.md`.

```sh
VERSOR_DATA=/path/to/data scripts/fetch_train_data.sh
VERSOR_DATA=/path/to/data scripts/train_v5.sh
```

## Acknowledgments

The evaluation weights were trained almost entirely on data published on the Egaroucid website: 54,000,000 of the
54,137,135 games in the training data are the Egaroucid Free Training Data
(https://www.egaroucid.nyanyan.dev/en/technology/train-data/), and the other 137,135 games are from the WTHOR database.
Thanks to Takuto Yamana, the author of Egaroucid (website: https://www.egaroucid.nyanyan.dev/en/, GitHub:
https://github.com/Nyanyan/Egaroucid), for publishing the training data and the technical explanations, and to the
Federation Francaise d'Othello for publishing the WTHOR database.

## License, third-party code and data

versor is released under the MIT License (`LICENSE`).

- The only dependencies are `serde_json` (MIT or Apache-2.0) and, for tests only, `magpie` (MIT). Cargo fetches both;
  neither is included here. The repository contains no code from other projects.
- The opponents, Edax and Egaroucid (both GPL-3.0), are not included. versor only runs them as separate processes
  (their paths are set in `config/engines.json`). No source code of the GPL engines was consulted in writing versor.
- No data is included, and none is redistributed: the scripts download it from the original sites. These are the
  sources and terms of the data that the scripts download or that you fetch by hand:
  - WTHOR database (Federation Francaise d'Othello, https://www.ffothello.org/informatique/la-base-wthor/):
    distributed free of charge.
  - Egaroucid Free Training Data (data published on the Egaroucid website by Takuto Yamana,
    https://www.egaroucid.nyanyan.dev/en/technology/train-data/): free to use, for example to build evaluation
    functions. **Redistributing the data is prohibited.**
  - FFO endgame test, problems #40 to #59 (Gunnar Andersson's Zebra site, http://radagast.se/othello/ffotest.html):
    downloaded by `scripts/fetch_ffo.sh`.
  - XOT openings (https://berg.earthlingz.de/xot/download.php): no terms of use could be found.
- The weights in `models/` are model parameters learned from the WTHOR database and the Egaroucid Free Training Data;
  they are not the data itself. Their provenance is described in `models/README.md`.
