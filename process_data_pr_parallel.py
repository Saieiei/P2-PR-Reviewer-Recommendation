#!/usr/bin/env python3
# ------------------------------------------------------------------
# Streaming + parallel wrapper for process_data_pr.py
# Parses the JSON file record-by-record, yielding chunks directly
# into your parallel workers—no upfront orjson.loads().
# ------------------------------------------------------------------

import os
import re
import random
import argparse
from multiprocessing import Pool
from tqdm import tqdm

import ijson                         # pip install ijson
import orjson
from fastcache import clru_cache as lru_cache
from rapidfuzz import fuzz, process

# Import your helpers (normalize, is_trivial, is_generic, sliding_windows, etc.)
import process_data_pr as orig

def parse_args():
    p = argparse.ArgumentParser(
        description="Stream and parallelize PR data prep"
    )
    p.add_argument("--workers",   type=int, default=os.cpu_count() or 1,
                   help="Number of worker processes")
    p.add_argument("--max-workers", type=int,
                   default=getattr(orig, 'max_workers', 128),
                   help="Cap on workers")
    p.add_argument("--chunk-size", type=int, default=1000,
                   help="Number of records per chunk (streamed)")
    p.add_argument("--file",      type=str,
                   default=getattr(orig, 'DATA_FILE', "preprocessed_data.json"),
                   help="Path to your JSON array file")
    return p.parse_args()

def chunked_record_stream(fp, chunk_size):
    """
    Stream JSON records from a top-level array using ijson, 
    yielding lists of `chunk_size` records.
    """
    parser = ijson.items(fp, "item")
    chunk = []
    for rec in parser:
        chunk.append(rec)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

def process_chunk(chunk):
    out = []
    for rec in chunk:
        out.extend(orig.make_positives(rec))
    return out

def sample_negatives(positives):
    # same as your orig.sample_negatives
    return orig.sample_negatives(positives)

def write_tsv(examples, filename):
    odir = getattr(orig, 'OUTPUT_DIR', '.')
    split = getattr(orig, 'CODESPLIT', '\t')
    path = os.path.join(odir, filename)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(split.join(map(str, ex)).replace('\n',' ') + "\n")

if __name__ == "__main__":
    args = parse_args()
    workers = min(args.workers, args.max_workers)

    # Open file once for streaming
    with open(args.file, 'rb') as f:
        # Build positives in parallel as we parse
        pool = Pool(processes=workers, maxtasksperchild=1)
        builder = pool.imap_unordered(
            process_chunk,
            chunked_record_stream(f, args.chunk_size)
        )
        positives = []
        for batch in tqdm(builder, desc="Building positives"):
            positives.extend(batch)
        pool.close()
        pool.join()

    print(f"> Created {len(positives)} positive examples")

    # Negative sampling (single‐threaded; it’s fast)
    negatives = sample_negatives(positives)
    print(f"> Created {len(negatives)} negative examples")

    # Shuffle & split
    all_data = positives + negatives
    random.shuffle(all_data)
    n  = len(all_data)
    t0 = int(n * getattr(orig, 'TRAIN_RATIO', 0.8))
    d0 = t0 + int(n * getattr(orig, 'DEV_RATIO',   0.1))
    train_set = all_data[:t0]
    dev_set   = all_data[t0:d0]
    test_set  = all_data[d0:]
    print(f"Train/Dev/Test counts: {len(train_set)}/{len(dev_set)}/{len(test_set)}")

    # Write TSVs
    write_tsv(train_set, 'train.tsv')
    write_tsv(dev_set,   'dev.tsv')
    write_tsv(test_set,  'test.tsv')

    print("✔ process_data_pr_parallel_stream.py complete — run_classifier ready")
