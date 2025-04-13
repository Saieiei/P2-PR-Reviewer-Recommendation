#!/usr/bin/env python
import argparse
import json
import os
import sys
import re
import time
import mmap
import platform
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import ijson
import cache_db

# --------------------------------------------------------------------------
# 1. Parse command-line arguments: support --rebuild and --disable-ast flags.
# --------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Unified BERT Retrieval Demo with OS-specific configuration"
)
parser.add_argument("--rebuild", action="store_true", help="Force rebuilding the embedding index")
parser.add_argument("--disable-ast", action="store_true", help="Disable AST parsing to reduce memory usage")
args = parser.parse_args()

# --------------------------------------------------------------------------
# 2. OS-specific configuration for BERT model and AST library file paths.
# --------------------------------------------------------------------------
if platform.system() == "Windows":
    MODEL_NAME = r"C:\sai\HPE\projects\project 2\cloned repo\P2-PR-Reviewer-Recommendation\bert-base-uncased"
    clang_library_path = r"C:\Program Files\LLVM\bin\libclang.dll"
else:
    MODEL_NAME = "bert-base-uncased"
    clang_library_path = "/ptmp2/nshashwa/llvm-project/build/lib/libclang.so"

DATA_FILE = "preprocessed_data.json"

# --------------------------------------------------------------------------
# 3. Configure AST Parsing (if not disabled).
# --------------------------------------------------------------------------
CLANG_AVAILABLE = False
if not args.disable_ast:
    try:
        import clang.cindex
        clang.cindex.Config.set_library_file(clang_library_path)
        CLANG_AVAILABLE = True
        print("INFO: AST parser enabled. Using clang library at", clang_library_path)
    except Exception as e:
        CLANG_AVAILABLE = False
        print("WARNING: AST parser not available:", e)
        print("INFO: Proceeding with BERT model only, without AST parsing.")
else:
    print("INFO: AST parsing disabled as per command-line argument.")

# --------------------------------------------------------------------------
# 4. Helper Functions for JSON Processing
# --------------------------------------------------------------------------
def clean_invalid_control_chars(s):
    return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', s)

def salvage_json_data_mmap(file_path):
    """Fallback salvage for truncated JSON files using mmap."""
    try:
        with open(file_path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            last_brace = mm.rfind(b"}")
            if last_brace == -1:
                print("No closing brace found; cannot salvage the file.")
                sys.exit(1)
            salvaged_bytes = mm[:last_brace + 1]
            mm.close()
            salvaged_text = salvaged_bytes.decode("utf-8", errors="ignore").rstrip()
            salvaged_text = clean_invalid_control_chars(salvaged_text)
            if not salvaged_text.lstrip().startswith('['):
                print("File does not appear to be a JSON array; cannot salvage with mmap method.")
                sys.exit(1)
            if not salvaged_text.rstrip().endswith("]"):
                salvaged_text += "\n]"
            return salvaged_text
    except Exception as e:
        print("Exception in salvage_json_data_mmap:", e)
        sys.exit(1)

def stream_json_data(file_path):
    """Stream JSON data using ijson assuming a top-level JSON array."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for record in ijson.items(f, "item"):
                yield record
    except Exception as e:
        print("Error streaming JSON data:", e)
        sys.exit(1)

# --------------------------------------------------------------------------
# 5. AST Parsing for C/C++ code using clang (if enabled)
# --------------------------------------------------------------------------
def parse_cpp_ast(code_text: str) -> str:
    if not CLANG_AVAILABLE:
        return ""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".cpp", delete=False) as tmp:
        tmp.write(code_text.encode("utf-8"))
        tmp_path = tmp.name
    try:
        index = clang.cindex.Index.create()
        tu = index.parse(tmp_path, args=["-std=c++11"])
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return ""
    ast_info = []
    def visitor(node):
        try:
            if node.kind.is_declaration():
                name = node.spelling or ""
                kind = str(node.kind)
                ast_info.append(f"({kind}:{name})")
        except ValueError:
            return
        for child in node.get_children():
            visitor(child)
    try:
        visitor(tu.cursor)
    except ValueError:
        pass
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    return " ".join(ast_info)

# --------------------------------------------------------------------------
# 6. BERT Embedder Class with Conditional GPU (autocast) Support.
# --------------------------------------------------------------------------
class BERTEmbedder:
    def __init__(self, model_path=MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def embed_text(self, text: str, max_length=512) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)
        cls_vec = outputs.last_hidden_state[:, 0, :]
        return cls_vec.cpu().numpy().squeeze(0)

    def embed_text_batch(self, texts: list, max_length=512) -> np.ndarray:
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)
        cls_vecs = outputs.last_hidden_state[:, 0, :]
        return cls_vecs.cpu().numpy()

# Instantiate the embedder.
embedder = BERTEmbedder()

# --------------------------------------------------------------------------
# 7. Build the Embedding Index with Caching and Batch Processing.
# --------------------------------------------------------------------------
def build_index():
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: {DATA_FILE} not found.")
        sys.exit(1)
    
    items = []            # All records from the JSON file.
    embeddings_list = []  # Computed embeddings corresponding to each record.
    indices_to_compute = []  # List indices that require new embedding computation.
    texts_to_compute = []    # Combined text for each record (for embedding).
    keys_to_compute = []     # Cache keys for the combined text.

    cache_conn = cache_db.init_cache_db()

    print("Building index from scratch (streaming records)...")
    start_time = time.time()
    for i, entry in enumerate(stream_json_data(DATA_FILE)):
        items.append(entry)
        if i % 50 == 0:
            print(f"  Processing record {i}")
        diff_text = entry.get("diff_text", "")
        comment_text = entry.get("comment_text", "")
        filename = entry.get("filename", "")
        ast_str = ""
        if CLANG_AVAILABLE and filename.lower().endswith((".c", ".cpp", ".cc", ".h", ".hpp")):
            ast_str = parse_cpp_ast(diff_text)
        combined_str = diff_text + "\n" + ast_str + "\n" + comment_text
        key = cache_db.compute_checksum(combined_str)
        embedding = cache_db.get_cached_embedding(cache_conn, key)
        if embedding is None or args.rebuild:
            indices_to_compute.append(i)
            texts_to_compute.append(combined_str)
            keys_to_compute.append(key)
            embeddings_list.append(None)  # Placeholder for later replacement.
        else:
            embeddings_list.append(embedding)
    end_stream = time.time()
    print(f"Streaming and initial cache lookup completed in {end_stream - start_time:.2f} seconds.")

    # Batch process records that need embeddings.
    if texts_to_compute:
        batch_size = 8  # Adjust based on available memory.
        total_batches = (len(texts_to_compute) + batch_size - 1) // batch_size
        print(f"Processing {len(texts_to_compute)} records in {total_batches} batches.")
        for batch_idx, start in enumerate(range(0, len(texts_to_compute), batch_size), start=1):
            batch_texts = texts_to_compute[start:start + batch_size]
            print(f"--- Batch {batch_idx}/{total_batches}: Processing records {start} to {start + len(batch_texts) - 1} ---")
            t0 = time.time()
            batch_embeddings = embedder.embed_text_batch(batch_texts)
            t1 = time.time()
            print(f"    Embedding computation for batch {batch_idx} took {t1 - t0:.2f} seconds.")
            for j, emb in enumerate(batch_embeddings):
                idx = indices_to_compute[start + j]
                embeddings_list[idx] = emb
                t_db0 = time.time()
                cache_db.save_cached_embedding(cache_conn, keys_to_compute[start + j], emb, batch_texts[j])
                t_db1 = time.time()
                print(f"    Saved embedding for record {idx} in {t_db1 - t_db0:.2f} seconds.")
    else:
        print("No new embeddings to compute; cache was up-to-date.")

    cache_conn.close()
    EMBEDDINGS = np.vstack(embeddings_list)
    total_time = time.time() - start_time
    print(f"Index built using cache database. Total records: {len(items)}. Total processing time: {total_time:.2f} seconds.")
    return EMBEDDINGS, items

# Build the index once (using the cache and streaming approach).
EMBEDDINGS, ITEMS = build_index()
print(f"Index is ready. Total records: {len(ITEMS)}\n")

# --------------------------------------------------------------------------
# 8. Retrieval Functions: Cosine Similarity-based Matching.
# --------------------------------------------------------------------------
def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-10
    return vec / norm

def retrieve_similar(user_diff: str, top_k=3) -> list:
    user_vec = embedder.embed_text(user_diff).reshape(1, -1)
    user_norm = l2_normalize(user_vec)
    stored_norm = l2_normalize(EMBEDDINGS)
    scores = stored_norm.dot(user_norm[0])
    idx_sorted = np.argsort(-scores)
    top_indices = idx_sorted[:top_k]
    
    results = []
    for idx in top_indices:
        score = float(scores[idx])
        results.append((score, ITEMS[idx]))
    return results

# --------------------------------------------------------------------------
# 9. Interactive Retrieval Loop.
# --------------------------------------------------------------------------
def main():
    print("Paste a diff snippet (or code). Then press Ctrl+D (Linux/Mac) or Ctrl+Z + Enter (Windows).")
    user_input = sys.stdin.read().strip()
    if not user_input:
        print("No input provided. Exiting.")
        return
    top_matches = retrieve_similar(user_input, top_k=3)
    print("\n=== Top Matches ===")
    for rank, (score, item) in enumerate(top_matches, start=1):
        snippet = item.get("diff_text", "")[:150].replace("\n", " ")
        comment = item.get("comment_text", "")
        filename = item.get("filename", "")
        print(f"Rank #{rank} | Score={score:.4f} | File={filename}")
        print(f"  Matched Diff Snippet: {snippet}...")
        print(f"  Reviewer Comment: {comment}")
        print("-" * 50)

if __name__ == "__main__":
    main()
