import argparse
import json
import os
import sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# Import cache_db functions for caching
import cache_db

# --------------------------------------------------------------------------
# 0. Parse command-line args for optional --rebuild
# --------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="AST-based diff retrieval with SQLite caching")
parser.add_argument("--rebuild", action="store_true", help="Force rebuilding the embedding index")
args = parser.parse_args()

# --------------------------------------------------------------------------
# 1. Configuration
# --------------------------------------------------------------------------
DATA_FILE = "preprocessed_data.json"
MODEL_NAME = r"C:\sai\HPE\projects\project 2\cloned repo\P2-PR-Reviewer-Recommendation\bert-base-uncased"

# --------------------------------------------------------------------------
# 2. Optional clang-based AST parsing
# --------------------------------------------------------------------------
try:
    import clang.cindex
    clang.cindex.Config.set_library_file(r"C:\Program Files\LLVM\bin\libclang.dll")
    CLANG_AVAILABLE = True
except ImportError:
    CLANG_AVAILABLE = False

def parse_cpp_ast(code_text: str) -> str:
    """Parse code_text with clang and return a textual summary of AST declarations."""
    if not CLANG_AVAILABLE:
        return ""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".cpp", delete=False) as tmp:
        tmp.write(code_text.encode("utf-8"))
        tmp_path = tmp.name
    try:
        index = clang.cindex.Index.create()
        tu = index.parse(tmp_path, args=['-std=c++11'])
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
        # Uncomment below if using a GPU:
        # self.model.cuda()

    def embed_text(self, text: str, max_length=512) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
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
    Reads preprocessed_data.json, computes (or loads cached) embeddings for each record,
    and returns the embeddings list along with the associated metadata.
    """
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: {DATA_FILE} not found.")
        sys.exit(1)
    
    # Initialize the cache DB
    cache_conn = cache_db.init_cache_db()
    
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    items = []
    embedding_list = []
    
    print(f"Building index from scratch (processing {len(data)} records)...")
    for i, entry in enumerate(data):
        if i % 50 == 0:
            print(f"  Processing record {i}/{len(data)}")
        
        diff_text = entry.get("diff_text", "")
        comment_text = entry.get("comment_text", "")
        filename = entry.get("filename", "")
        
        # For C/C++ files, get AST summary.
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
            # Not in cache (or rebuild forced): compute embedding.
            embedding = embedder.embed_text(combined_str)
            # Save embedding to cache DB.
            cache_db.save_cached_embedding(cache_conn, key, embedding, combined_str)
        
        embedding_list.append(embedding)
        items.append(entry)
    
    # Convert list of embeddings to a NumPy array.
    embeddings = np.vstack(embedding_list)
    cache_conn.close()
    print("Index built using cache database.")
    return embeddings, items

# For this design, we always build the index by checking the DB cache.
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
    Given a user-provided diff snippet, compute its embedding
    and retrieve the top_k most similar records based on cosine similarity.
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
