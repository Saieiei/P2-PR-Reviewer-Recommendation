#!/usr/bin/env python3
"""
Parallel wrapper – unchanged except diff-0 writer is now deduplicated
just like in process_data_pr.py.
"""

import os, random, argparse
from multiprocessing import Pool
from tqdm import tqdm
import ijson
import process_data_pr as core

def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--chunk-size", type=int, default=1000)
    p.add_argument("--file", type=str, default=core.DATA_FILE)
    return p.parse_args()

def stream_chunks(fp, size):
    items, chunk = ijson.items(fp, "item"), []
    for rec in items:
        chunk.append(rec)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

def do_chunk(chunk):
    out = []
    for rec in chunk:
        out.extend(core.make_positives(rec))
    return out

# ───────── main ─────────
if __name__ == "__main__":
    args     = cli()
    workers  = max(1, min(args.workers, os.cpu_count() or 1))

    with open(args.file, "rb") as fp, Pool(workers) as pool:
        positives = []
        for batch in tqdm(pool.imap_unordered(
                do_chunk, stream_chunks(fp, args.chunk_size)),
                desc="Building positives"):
            positives.extend(batch)

    negatives = core.sample_negatives(positives)

    # pretty NEG debug – dedup
    seen0 = set()
    for idx, (_, prn, _, diffw, commw) in enumerate(negatives, start=1):
        try:
            _, rest = diffw.split("[FILE]", 1)
            fp, r   = rest.split(" ", 1)
            lbl_tag, win = r.split(" ", 1)
            lbls = lbl_tag.replace("[LABEL]", "")
        except ValueError:
            fp, lbls, win = "(unknown)", "", diffw
        key = (prn, commw.strip())
        if key not in seen0:
            core.pretty(core._diff0_fp, "NEG", prn, idx, fp, lbls, win, commw)
            seen0.add(key)

    # TSV shuffle / split
    all_rows = positives + negatives
    random.shuffle(all_rows)
    n = len(all_rows)
    tr, dv = int(n * core.TRAIN_RATIO), int(n * core.DEV_RATIO)
    core.write_tsv(all_rows[:tr],        "train.tsv")
    core.write_tsv(all_rows[tr:tr+dv],   "dev.tsv")
    core.write_tsv(all_rows[tr+dv:],     "test.tsv")

    print("✔ process_data_pr_parallel.py complete")
