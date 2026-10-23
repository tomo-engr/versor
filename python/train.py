# Train the versor pattern evaluation (weight file format v5: 15 pattern
# tables, three scalar features and phase smoothing). Reads the dataprep
# output (132-byte records, see versor-eval/src/bin/dataprep.rs), trains
# the per-phase linear model with Adam on MSE, and writes the Rust weight
# file plus a JSON training log and Python/Rust parity test vectors.
#
# The feature layout must match versor-eval exactly: the table sizes and
# instance counts below, then the three scalar weights and one bias
# entry at the end of each phase stride. Any mismatch is caught by the
# parity test (versor-eval/tests/parity.rs compares the quantized integer
# predictions).
#
# Thread limit: 4 by default, set through environment variables before
# numpy is imported.

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse
import glob
import json
import struct
import sys
import time

import numpy as np

# Weight file v5: 15 pattern tables (v3's 11 plus center2x4, inner3x3,
# diag4, diag3) + 3 scalars + bias, 15 phases (4-disc buckets). Matches
# versor-eval's superset.
N_INSTANCES = 60
TABLE_SIZES = [6561, 6561, 6561, 6561, 6561, 2187, 729, 243, 19683, 59049, 59049,
               6561, 19683, 81, 27]
TABLE_CELLS = [8, 8, 8, 8, 8, 7, 6, 5, 9, 10, 10, 8, 9, 4, 3, 200, 0]  # 200 = scalar sentinel
INSTANCES_PER_TABLE = [4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 8, 2, 4, 4, 4]
TOTAL_WEIGHTS = sum(TABLE_SIZES)  # 200097
N_PHASES = 15
BUCKET_DISCS = 4  # disc counts per phase bucket
N_SCALARS = 3  # own mobility, opponent mobility, parity
STRIDE = TOTAL_WEIGHTS + N_SCALARS + 1  # + bias
SCALE = 256
VERSION = 5

RECORD_DTYPE = np.dtype(
    [
        ("idx", "<u2", (N_INSTANCES,)),
        ("score", "<i2"),
        ("phase", "u1"),
        ("src", "u1"),
        ("ghash", "<u4"),
        ("mob_own", "u1"),
        ("mob_opp", "u1"),
        ("parity", "u1"),
        ("pad", "u1"),
    ]
)
assert RECORD_DTYPE.itemsize == 132

NONZERO_DIGITS = None

def nonzero_digit_lut():
    global NONZERO_DIGITS
    if NONZERO_DIGITS is None:
        lut = np.zeros(6561, dtype=np.int64)
        for idx in range(6561):
            x, c = idx, 0
            while x:
                if x % 3:
                    c += 1
                x //= 3
            lut[idx] = c
        NONZERO_DIGITS = lut
    return NONZERO_DIGITS

def discs_of(batch):
    # The 16 line instances (tables 0-3) cover every square exactly
    # twice (one row line, one column line), and canonicalization only
    # permutes digits, so the disc count is exactly recoverable from
    # the stored indices.
    lut = nonzero_digit_lut()
    return lut[batch["idx"][:, :16].astype(np.int64)].sum(axis=1) // 2

def phase_of_batch(batch):
    # Recompute phase from disc count with the current bucket width
    # (the stored phase byte is not used).
    discs = discs_of(batch)
    return np.clip((discs - 4) // BUCKET_DISCS, 0, N_PHASES - 1)

def blend_coeffs(batch):
    # Phase smoothing coefficients, mirroring versor-eval exactly.
    # 4-disc buckets: u 0/1 -> phase-1 (w8 3/1), u 2/3 -> phase+1 (1/3).
    # 2-disc buckets: u 0 -> phase-1, u 1 -> phase+1, each w8 = 2.
    discs = discs_of(batch)
    p = phase_of_batch(batch)
    u = (discs - 4) % BUCKET_DISCS
    if BUCKET_DISCS == 4:
        q = np.where(u <= 1, p - 1, p + 1)
        w8 = np.where((u == 0) | (u == 3), 3.0, 1.0)
    else:
        q = np.where(u == 0, p - 1, p + 1)
        w8 = np.full(p.shape, 2.0)
    valid = (q >= 0) & (q < N_PHASES)
    w8 = np.where(valid, w8, 0.0)
    q = np.clip(q, 0, N_PHASES - 1)
    return p, q, (8.0 - w8) / 8.0, w8 / 8.0

def instance_offsets():
    offs = []
    table_off = 0
    for t, n_inst in enumerate(INSTANCES_PER_TABLE):
        offs.extend([table_off] * n_inst)
        table_off += TABLE_SIZES[t]
    assert len(offs) == N_INSTANCES
    return np.asarray(offs, dtype=np.int64)

INST_OFF = instance_offsets()
SCALAR_OFF = TOTAL_WEIGHTS  # own_mob, opp_mob, parity slots

def load_shard(path, fraction):
    # Sequential read; random access via memmap would thrash the disk.
    a = np.fromfile(path, dtype=RECORD_DTYPE)
    if fraction >= 1.0:
        return a
    keep = a["ghash"] % 10000 < int(round(fraction * 10000))
    return a[keep]

def scalar_values(batch):
    return (
        batch["mob_own"].astype(np.float64),
        batch["mob_opp"].astype(np.float64),
        (batch["parity"] & 1).astype(np.float64),
    )

def phase_acc(w, batch, ph):
    # Accumulator of one phase vector for every record in the batch.
    base = ph * STRIDE
    flat = base[:, None] + INST_OFF[None, :] + batch["idx"].astype(np.int64)
    acc = w[flat].sum(axis=1)
    om, pm, par = scalar_values(batch)
    acc += w[base + SCALAR_OFF] * om
    acc += w[base + SCALAR_OFF + 1] * pm
    acc += w[base + SCALAR_OFF + 2] * par
    acc += w[base + STRIDE - 1]
    return acc, flat

def predict(w, batch):
    p, q, cp, cq = blend_coeffs(batch)
    acc_p, _ = phase_acc(w, batch, p)
    acc_q, _ = phase_acc(w, batch, q)
    return cp * acc_p + cq * acc_q

def evaluate(w, val):
    mse_num = np.zeros(N_PHASES)
    mae_num = np.zeros(N_PHASES)
    cnt = np.zeros(N_PHASES)
    bs = 1 << 20
    for i in range(0, len(val), bs):
        b = val[i : i + bs]
        err = predict(w, b) - b["score"].astype(np.float64)
        ph = phase_of_batch(b)
        mse_num += np.bincount(ph, weights=err * err, minlength=N_PHASES)
        mae_num += np.bincount(ph, weights=np.abs(err), minlength=N_PHASES)
        cnt += np.bincount(ph, minlength=N_PHASES)
    cnt = np.maximum(cnt, 1)
    return mse_num / cnt, mae_num / cnt, cnt

def accumulate_grad(g, batch, ph, coef, err, flat):
    # Gradient contribution of one phase side: coef * err at every
    # touched slot (patterns, scalars, bias).
    ce = coef * err
    g += np.bincount(flat.ravel(), weights=np.repeat(ce, N_INSTANCES), minlength=len(g))
    base = ph * STRIDE
    om, pm, par = scalar_values(batch)
    g += np.bincount(base + SCALAR_OFF, weights=ce * om, minlength=len(g))
    g += np.bincount(base + SCALAR_OFF + 1, weights=ce * pm, minlength=len(g))
    g += np.bincount(base + SCALAR_OFF + 2, weights=ce * par, minlength=len(g))
    g += np.bincount(base + STRIDE - 1, weights=ce, minlength=len(g))

def adam_epoch(w, m, v, shards, rng, lr, batch_size, adam_t):
    order = rng.permutation(len(shards))
    total_loss = 0.0
    total_n = 0
    for si in order:
        data = shards[si]
        if isinstance(data, str):
            data = np.fromfile(data, dtype=RECORD_DTYPE)
        perm = rng.permutation(len(data))
        for i in range(0, len(data), batch_size):
            b = data[perm[i : i + batch_size]]
            p, q, cp, cq = blend_coeffs(b)
            acc_p, flat_p = phase_acc(w, b, p)
            acc_q, flat_q = phase_acc(w, b, q)
            err = cp * acc_p + cq * acc_q - b["score"].astype(np.float64)
            total_loss += float(err @ err)
            total_n += len(b)
            g = np.zeros_like(w)
            accumulate_grad(g, b, p, cp, err, flat_p)
            accumulate_grad(g, b, q, cq, err, flat_q)
            g *= 2.0 / len(b)
            adam_t[0] += 1
            t = adam_t[0]
            m *= 0.9
            m += 0.1 * g
            v *= 0.999
            v += 0.001 * g * g
            mh = m / (1 - 0.9**t)
            vh = v / (1 - 0.999**t)
            w -= lr * mh / (np.sqrt(vh) + 1e-8)
    return total_loss / max(total_n, 1)

def write_weights(path, w):
    wq = np.clip(np.rint(w * SCALE), -32767, 32767).astype(np.int16)
    with open(path, "wb") as f:
        f.write(b"VEVL")
        f.write(struct.pack("<IIII", VERSION, SCALE, N_PHASES, len(TABLE_CELLS)))
        for c in TABLE_CELLS:
            f.write(struct.pack("<I", c))
        f.write(wq.tobytes())
    return wq

def emulate_rust_eval(wq, discs, idx_row, om, pm, par):
    # Integer path identical to versor-eval: per-phase accumulator,
    # eighth blending with (+4 >> 3), round half away from zero, clamp.
    def acc_at(ph):
        base = ph * STRIDE
        a = int(wq[base + STRIDE - 1])
        for i in range(N_INSTANCES):
            a += int(wq[base + int(INST_OFF[i]) + int(idx_row[i])])
        a += int(wq[base + SCALAR_OFF]) * om
        a += int(wq[base + SCALAR_OFF + 1]) * pm
        a += int(wq[base + SCALAR_OFF + 2]) * par
        return a
    p = min((discs - 4) // BUCKET_DISCS, N_PHASES - 1)
    acc = acc_at(p)
    u = (discs - 4) % BUCKET_DISCS
    if BUCKET_DISCS == 4:
        q = p - 1 if u <= 1 else p + 1
        w8 = 3 if u in (0, 3) else 1
    else:
        q = p - 1 if u == 0 else p + 1
        w8 = 2
    if 0 <= q < N_PHASES:
        acc = ((8 - w8) * acc + w8 * acc_at(q) + 4) >> 3
    if acc >= 0:
        r = (acc + 128) >> 8
    else:
        r = -((-acc + 128) >> 8)
    return max(-63, min(63, r))

def write_testvec(boards_path, out_path, wq):
    lines_out = []
    with open(boards_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            board, idxs, scalars = parts[0], parts[2], parts[3]
            idx_row = [int(x) for x in idxs.split(",")]
            om, pm, par = (int(x) for x in scalars.split(","))
            discs = sum(1 for c in board[:64] if c != "-")
            val = emulate_rust_eval(wq, discs, idx_row, om, pm, par)
            lines_out.append(f"{board}\t{val}\n")
    with open(out_path, "w") as f:
        f.writelines(lines_out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=1 << 20)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--min-improve", type=float, default=1e-3)
    ap.add_argument("--log", default=None)
    ap.add_argument("--testvec", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    shard_paths = sorted(glob.glob(os.path.join(args.data, "train_*.bin")))
    if not shard_paths:
        sys.exit(f"no training shards (train_*.bin) in {args.data}; "
                 "make them with dataprep (see scripts/train_v5.sh)")
    print(f"preparing {len(shard_paths)} shards (fraction {args.fraction})", flush=True)
    if args.fraction < 1.0:
        # Filter once into RAM.
        shards = [load_shard(p, args.fraction) for p in shard_paths]
        shards = [s for s in shards if len(s)]
        n_train = sum(len(s) for s in shards)
    else:
        # Stream one shard at a time per epoch.
        shards = shard_paths
        n_train = sum(
            os.path.getsize(p) // RECORD_DTYPE.itemsize for p in shard_paths
        )
    val = np.fromfile(os.path.join(args.data, "val.bin"), dtype=RECORD_DTYPE)
    print(f"train positions: {n_train}, val positions: {len(val)}", flush=True)

    w = np.zeros(N_PHASES * STRIDE, dtype=np.float64)
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    adam_t = [0]

    log = {
        "fraction": args.fraction,
        "seed": args.seed,
        "lr": args.lr,
        "batch": args.batch,
        "n_train": int(n_train),
        "n_val": int(len(val)),
        "curve": [],
    }
    best_val = float("inf")
    best_w = None
    bad = 0
    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()
        train_mse = adam_epoch(w, m, v, shards, rng, args.lr, args.batch, adam_t)
        val_mse_ph, val_mae_ph, cnt = evaluate(w, val)
        val_mse = float((val_mse_ph * cnt).sum() / cnt.sum())
        val_mae = float((val_mae_ph * cnt).sum() / cnt.sum())
        dt = time.time() - t0
        log["curve"].append(
            {
                "epoch": epoch,
                "train_mse": train_mse,
                "val_mse": val_mse,
                "val_mae": val_mae,
                "val_mae_per_phase": [round(float(x), 4) for x in val_mae_ph],
                "seconds": round(dt, 1),
            }
        )
        print(
            f"epoch {epoch}: train_mse {train_mse:.3f} val_mse {val_mse:.3f} "
            f"val_mae {val_mae:.3f} ({dt:.0f}s)",
            flush=True,
        )
        if val_mse < best_val - args.min_improve:
            best_val = val_mse
            best_w = w.copy()
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"val mse saturated (patience {args.patience})", flush=True)
                break
    if best_w is None:
        best_w = w

    wq = write_weights(args.out, best_w)
    print(f"wrote {args.out}", flush=True)
    if args.log:
        with open(args.log, "w") as f:
            json.dump(log, f, indent=1)
    if args.testvec:
        boards = os.path.join(args.data, "boards_val.txt")
        write_testvec(boards, args.testvec, wq.astype(np.int64))
        print(f"wrote {args.testvec}", flush=True)

if __name__ == "__main__":
    main()
