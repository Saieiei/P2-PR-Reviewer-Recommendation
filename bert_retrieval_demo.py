import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------
# 0. Parse command-line args for optional --rebuild
# --------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="BERT-based diff retrieval with optional AST & caching")
parser.add_argument("--rebuild", action="store_true", help="force rebuilding the embedding index")
args = parser.parse_args()

# --------------------------------------------------------------------------
# 1. Configuration
# --------------------------------------------------------------------------
DATA_FILE = "preprocessed_data.json"
CACHE_EMB = "cached_embeddings.npy"  # stores embeddings as NumPy array
CACHE_ITEMS = "cached_items.pkl"     # stores the associated metadata list
MODEL_NAME = r"C:\sai\HPE\projects\project 2\cloned repo\P2-PR-Reviewer-Recommendation\bert-base-uncased"

# --------------------------------------------------------------------------
# 2. Optional clang-based AST parsing
# --------------------------------------------------------------------------
try:
    import clang.cindex
    # If you installed LLVM in this location, uncomment or adjust as needed:
    clang.cindex.Config.set_library_file(r"C:\Program Files\LLVM\bin\libclang.dll")
    CLANG_AVAILABLE = True
except ImportError:
    CLANG_AVAILABLE = False

def parse_cpp_ast(code_text: str) -> str:
    """Attempt to parse code_text with clang and return a textual summary of AST declarations."""
    if not CLANG_AVAILABLE:
        return ""

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".cpp", delete=False) as tmp:
        tmp.write(code_text.encode("utf-8"))
        tmp_path = tmp.name

    try:
        index = clang.cindex.Index.create()
        tu = index.parse(tmp_path, args=['-std=c++11'])
    except Exception as e:
        # Could fail if code_text is incomplete or clang doesn't like it
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
            # e.g. unknown node kind
            return
        for c in node.get_children():
            visitor(c)

    try:
        visitor(tu.cursor)
    except ValueError:
        pass

    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    return " ".join(ast_info)

# --------------------------------------------------------------------------
# 3. BERT embedder
# --------------------------------------------------------------------------
class BERTEmbedder:
    def __init__(self, model_path=MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()
        # If you have a GPU, uncomment:
        # self.model.cuda()

    def embed_text(self, text: str, max_length=512) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
        # If on GPU:
        # inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Take the [CLS] token embedding
        cls_vec = outputs.last_hidden_state[:, 0, :]
        return cls_vec.cpu().numpy().squeeze(0)

embedder = BERTEmbedder()

# --------------------------------------------------------------------------
# 4. Building or Loading the Index
# --------------------------------------------------------------------------
def build_index():
    """Reads preprocessed_data.json, does AST parsing + BERT embeddings for each record, caches the result."""
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: {DATA_FILE} not found.")
        sys.exit(1)

    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    items = []
    embedding_list = []

    print(f"Building the index from scratch (found {len(data)} records). This may take a while...")

    for i, entry in enumerate(data):
        if i % 50 == 0:
            print(f"  -> Processing record {i}/{len(data)}")

        diff_text = entry.get("diff_text", "")
        comment_text = entry.get("comment_text", "")
        filename = entry.get("filename", "")

        # If it's a C/C++ file, parse AST
        ast_str = ""
        if CLANG_AVAILABLE and filename.endswith((".c", ".cpp", ".cc", ".h", ".hpp")):
            ast_str = parse_cpp_ast(diff_text)

        # Combine diff + AST + comment
        combined_str = diff_text + "\n" + ast_str + "\n" + comment_text
        emb = embedder.embed_text(combined_str)
        embedding_list.append(emb)
        items.append(entry)

    # Convert to numpy array
    embeddings = np.vstack(embedding_list)

    # Cache results
    np.save(CACHE_EMB, embeddings)
    with open(CACHE_ITEMS, 'wb') as f:
        pickle.dump(items, f)
    print("Index built and cached to disk.")
    return embeddings, items

def load_index():
    """Loads the cached embeddings and metadata if they exist."""
    if not (os.path.exists(CACHE_EMB) and os.path.exists(CACHE_ITEMS)):
        return None, None
    embeddings = np.load(CACHE_EMB)
    with open(CACHE_ITEMS, 'rb') as f:
        items = pickle.load(f)
    return embeddings, items

# Decide whether to build or load
if args.rebuild:
    EMBEDDINGS, ITEMS = build_index()
else:
    EMBEDDINGS, ITEMS = load_index()
    if EMBEDDINGS is None or ITEMS is None:
        EMBEDDINGS, ITEMS = build_index()

print(f"Index is ready. We have {len(ITEMS)} diffs loaded.\n")

# --------------------------------------------------------------------------
# 5. Retrieval
# --------------------------------------------------------------------------
def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-10
    return vec / norm

def retrieve_similar(user_diff: str, top_k=3) -> list:
    """Embed the user's new snippet, do cosine similarity vs. all embeddings, return top_k results."""
    # Create user embedding
    user_vec = embedder.embed_text(user_diff).reshape(1, -1)

    # Cosine similarity
    # We'll L2 normalize everything first
    user_norm = l2_normalize(user_vec)
    stored_norm = l2_normalize(EMBEDDINGS)
    scores = stored_norm.dot(user_norm[0])  # shape (N,)

    # Sort by descending similarity
    idx_sorted = np.argsort(-scores)
    top_indices = idx_sorted[:top_k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        results.append((score, ITEMS[idx]))
    return results

# --------------------------------------------------------------------------
# 6. Main Interactive Loop
# --------------------------------------------------------------------------
def main():
    print("Paste a diff snippet (or code). Then press Ctrl+D (Linux/Mac) or Ctrl+Z + Enter (Windows).")
    user_input = sys.stdin.read().strip()
    if not user_input:
        print("No input. Exiting.")
        return

    top_matches = retrieve_similar(user_input, top_k=3)

    print("\n=== Top Matches ===")
    for rank, (score, item) in enumerate(top_matches, start=1):
        snippet  = item["diff_text"][:150].replace("\n", " ")
        comment  = item["comment_text"]
        filename = item.get("filename", "")
        print(f"Rank #{rank} | Score={score:.4f} | File={filename}")
        print(f"  Matched Diff Snippet: {snippet}...")
        print(f"  Original Reviewer Comment: {comment}")
        print("-" * 50)

if __name__ == "__main__":
    main()
