import argparse
import json
import os
import sys
import mmap
import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# Import cache_db functions for caching
import cache_db

# --------------------------------------------------------------------------
# 0. Parse command-line args for optional --rebuild
# --------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="AST-based diff retrieval with SQLite caching"
)
parser.add_argument("--rebuild", action="store_true", help="Force rebuilding the embedding index")
args = parser.parse_args()

# --------------------------------------------------------------------------
# 1. Configuration
# --------------------------------------------------------------------------
DATA_FILE = "preprocessed_data.json"
MODEL_NAME = "bert-base-uncased"

# --------------------------------------------------------------------------
# Helper function to clean invalid control characters.
# This function removes all control characters except tab (\x09), newline (\x0A),
# and carriage return (\x0D) because JSON allows these whitespace characters.
# --------------------------------------------------------------------------
def clean_invalid_control_chars(s):
    return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', s)

# --------------------------------------------------------------------------
# Fallback salvage function using mmap for fast recovery from truncated JSON
# --------------------------------------------------------------------------
def salvage_json_data_mmap(file_path):
    """
    Uses memory mapping to quickly salvage a truncated JSON file.
    This function assumes the file is a JSON array. It finds the last occurrence
    of the byte corresponding to '}' and returns the file content up to that point,
    with a trailing ']' appended if necessary. The text is then cleaned from invalid
    control characters before being returned.
    """
    try:
        with open(file_path, "rb") as f:
            # Memory-map the file, mapping the entire file
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            # Find the last occurrence of the closing brace (b"}")
            last_brace = mm.rfind(b"}")
            if last_brace == -1:
                print("No closing brace found; cannot salvage the file.")
                sys.exit(1)
            salvaged_bytes = mm[:last_brace+1]
            mm.close()
            salvaged_text = salvaged_bytes.decode("utf-8", errors="ignore").rstrip()
            # Clean invalid control characters
            salvaged_text = clean_invalid_control_chars(salvaged_text)
            # Ensure the file is intended to be a JSON array.
            if not salvaged_text.lstrip().startswith('['):
                print("File does not appear to be a JSON array; cannot salvage with mmap method.")
                sys.exit(1)
            # Append a closing bracket if missing.
            if not salvaged_text.rstrip().endswith("]"):
                salvaged_text += "\n]"
            return salvaged_text
    except Exception as e:
        print("Exception in salvage_json_data_mmap:", e)
        sys.exit(1)

# --------------------------------------------------------------------------
# Helper function: load (or salvage) JSON data
# --------------------------------------------------------------------------
def load_json_data(file_path):
    """
    Attempts to load JSON data from the given file.
    If json.load() fails due to a truncation or control character issue,
    uses salvage_json_data_mmap() to recover the complete portion of the file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        print(f"Initial json.load() failed: {e}")
        print("Attempting to salvage the JSON file using mmap...")
        salvaged_text = salvage_json_data_mmap(file_path)
        try:
            data = json.loads(salvaged_text)
            print(f"Successfully salvaged {len(data)} records from the JSON file.")
            recovered_file = "preprocessed_data_recovered.json"
            with open(recovered_file, "w", encoding="utf-8") as fw:
                json.dump(data, fw, indent=2, ensure_ascii=False)
            print(f"Recovered JSON written to: {recovered_file}")
            return data
        except json.JSONDecodeError as e2:
            print("Failed to parse salvaged JSON:", e2)
            sys.exit(1)

# --------------------------------------------------------------------------
# 2. Optional clang-based AST parsing
# --------------------------------------------------------------------------
try:
    import clang.cindex
    # Adjust the library path as needed for your system.
    clang.cindex.Config.set_library_file(r"C:\Program Files\LLVM\bin\libclang.dll")
    CLANG_AVAILABLE = True
except ImportError:
    CLANG_AVAILABLE = False

def parse_cpp_ast(code_text: str) -> str:
    """Parse code_text using clang and return a textual summary of AST declarations."""
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
# 3. Embedder using a Transformer Model
# --------------------------------------------------------------------------
class BERTEmbedder:
    def __init__(self, model_path=MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()
        # Uncomment below if you have GPU support:
        # self.model.cuda()

    def embed_text(self, text: str, max_length=512) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use the [CLS] token representation as the embedding.
        cls_vec = outputs.last_hidden_state[:, 0, :]
        return cls_vec.cpu().numpy().squeeze(0)

embedder = BERTEmbedder()

# --------------------------------------------------------------------------
# 4. Building the Embedding Index with Cache DB
# --------------------------------------------------------------------------
def build_index():
    """
    Reads preprocessed_data.json (or the salvaged version), computes (or loads cached)
    embeddings for each record, and returns the embeddings list along with the associated metadata.
    """
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: {DATA_FILE} not found.")
        sys.exit(1)
    
    # Load JSON data using the salvage helper function.
    data = load_json_data(DATA_FILE)
    
    # Initialize the cache DB.
    cache_conn = cache_db.init_cache_db()
    
    items = []
    embedding_list = []
    
    print(f"Building index from scratch (processing {len(data)} records)...")
    for i, entry in enumerate(data):
        if i % 50 == 0:
            print(f"  Processing record {i}/{len(data)}")
        
        diff_text = entry.get("diff_text", "")
        comment_text = entry.get("comment_text", "")
        filename = entry.get("filename", "")
        
        # For C/C++ files, get an AST summary if available.
        ast_str = ""
        if CLANG_AVAILABLE and filename.endswith((".c", ".cpp", ".cc", ".h", ".hpp")):
            ast_str = parse_cpp_ast(diff_text)
        
        # Combine text fields for embedding.
        combined_str = diff_text + "\n" + ast_str + "\n" + comment_text
        # Compute a unique key (checksum) for this combined text.
        key = cache_db.compute_checksum(combined_str)
        
        # Attempt to load a cached embedding.
        embedding = cache_db.get_cached_embedding(cache_conn, key)
        if embedding is None or args.rebuild:
            # Compute embedding if not found in cache or forced rebuild.
            embedding = embedder.embed_text(combined_str)
            cache_db.save_cached_embedding(cache_conn, key, embedding, combined_str)
        
        embedding_list.append(embedding)
        items.append(entry)
    
    # Convert the list of embeddings to a NumPy array.
    embeddings = np.vstack(embedding_list)
    cache_conn.close()
    print("Index built using cache database.")
    return embeddings, items

# Build the index.
EMBEDDINGS, ITEMS = build_index()
print(f"Index is ready. Total records: {len(ITEMS)}\n")

# --------------------------------------------------------------------------
# 5. Retrieval: Cosine Similarity-based Matching
# --------------------------------------------------------------------------
def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-10
    return vec / norm

def retrieve_similar(user_diff: str, top_k=3) -> list:
    """
    Given a user-provided diff snippet, compute its embedding and retrieve the top_k most similar records
    based on cosine similarity.
    """
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
# 6. Interactive Retrieval Loop
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
        snippet = item["diff_text"][:150].replace("\n", " ")
        comment = item["comment_text"]
        filename = item.get("filename", "")
        print(f"Rank #{rank} | Score={score:.4f} | File={filename}")
        print(f"  Matched Diff Snippet: {snippet}...")
        print(f"  Reviewer Comment: {comment}")
        print("-" * 50)

if __name__ == "__main__":
    main()
