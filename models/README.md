# versor: trained evaluation weights

## pattern_best.bin (a symbolic link to the current best)

It currently points to `pattern_v5.bin`. The browser UI (`python/webui/engine.py`) loads this link by default. When
new weights are adopted, repoint the link:

```sh
ln -sfn pattern_vN.bin models/pattern_best.bin
```

The UI's engine reads the weights when it starts, so restart the UI after repointing the link. The link exists so
that the weight file name is not written into more places. A checkout without symbolic links (for example on Windows
without symbolic link support) has a small text file here instead of the link; then point `VERSOR_WEIGHTS` at
`pattern_v5.bin` (the engine reports "not a versor weight file" for the text file).

For training, evaluation and match measurements, **name the weight file itself, not the link**
(`--weights models/pattern_v5.bin`), so that the record of each run shows which weights it used. The link is only
for the browser UI.

## Common to all versions

- Format: versor-eval weight files (`versor-eval/src/weights.rs`). Versions differ in the number of pattern tables
  and in other details.
- Training: all of the data, seed 42. The recipe for v5 is `scripts/train_v5.sh` (`dataprep` and `python/train.py`).
  The current `python/train.py` writes the v5 format; v1 to v3 were trained with the versions of it that wrote their
  formats.
- Training data (the same for every version):
  - Egaroucid Free Training Data, data published on the Egaroucid website by Takuto Yamana,
    https://www.egaroucid.nyanyan.dev/en/technology/train-data/
    (the Egaroucid_Train_Data_v0002 series and the Egaroucid_Transcript series)
  - WThor database, Federation Francaise d'Othello (ffothello.org)
- These weights are model parameters learned from the data above. They are not a redistribution of the data.
- New weights were adopted when, in a `versor-cli arena` match against the previous weights at fixed depth 8 over
  1,024 games, the lower end of the 95% confidence interval of the Elo difference was above zero. The match results
  below were measured again with this release, so the commands reproduce them exactly (`--tt 32 --solve-at 20`, any
  `--threads`).

## pattern_v1.bin

- Format v1: 10 pattern tables and a bias, 15 phases.
- Trained: 2026-07-17.

## pattern_v2.bin

- Format v2: v1 plus three scalar features per phase (mobility of the side to move, mobility of the opponent, and
  parity), with interpolation between neighbouring phases in steps of one eighth.
- Trained: 2026-07-17.
- Validation MAE: 5.54 -> 5.30.
- Match against v1 (`--weights models/pattern_v2.bin --weights-b models/pattern_v1.bin --depth 8 --games 1024
  --seed 31`): Elo +21.7 (95% CI +6.1 .. +37.4).

## pattern_v3.bin

- Format v3: adds a corner 2x5 block to the patterns (11 tables).
- Trained: 2026-07-19.
- Validation MAE: 5.30 -> 4.91.
- Match against v2 (`--weights models/pattern_v3.bin --weights-b models/pattern_v2.bin --depth 8 --games 1024
  --seed 62`): Elo +49.2 (95% CI +34.2 .. +64.4).

## pattern_v5.bin (current best)

- Format v5: 15 pattern tables (200,097 weights per phase), 3 scalars and a bias, 15 phases.
- Trained: 2026-07-27.
- Changes from v3: four more tables, center2x4, inner3x3, diag4 and diag3, which cover the centre of the board, the
  area between the corners and the centre, and the short diagonals. Validation MAE: 4.905 -> 4.854.
- Match against v3 (`--weights models/pattern_v5.bin --weights-b models/pattern_v3.bin --depth 8 --games 1024
  --seed 64`): Elo +24.1 (95% CI +10.4 .. +37.9). Per node it is about 3% slower than v3, but it searches 3-8% fewer
  nodes, so a search to the same depth takes about as long.

v4 (the v3 tables split into 30 phases) was not adopted and is not included.

## mpc_params.txt

- The Multi-ProbCut regression coefficients (36 lines of `phase d dprime a b sigma`), fitted by `python/mpc_fit.py`
  to search statistics collected with pattern_v5:
  `cargo run --release -p versor-eval --example mpc_stats -- models/pattern_v5.bin stats.csv 20000 4 77`
  (phases 2 to 13, 20,000 positions each, depths 2/4/6/8, seed 77; the output does not depend on the thread count),
  then `python python/mpc_fit.py stats.csv models/mpc_params.txt`. Fitted: 2026-10-12. The slopes `a` are 0.986 to
  1.007 and `sigma` is 1.45 to 3.86. They are model parameters, not training data.
- With these coefficients, pattern_v5 with Multi-ProbCut against pattern_v5 without it (`arena --weights
  models/pattern_v5.bin --weights-b models/pattern_v5.bin --mpc models/mpc_params.txt --mpc-t 2.0 --nodes 1000000
  --games 1024 --seed 33 --tt 32 --solve-at 20`): Elo +87.3 (95% CI +73.2 .. +101.7). At a fixed depth (the same
  with `--depth 8` instead of `--nodes`) it is -0.3 (-6.2 .. +5.5): the cut pays off only when the saved nodes can be
  spent on searching deeper.
- The browser UI uses them with `--mpc models/mpc_params.txt --mpc-t 2.0`. `serve` takes the same options; in
  `arena` they apply to the side given by `--weights`.
