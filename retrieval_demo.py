"""
retrieval_demo.py

Allows the user to paste a diff snippet (or any text) at runtime via stdin.
Then retrieves the top-3 most similar diffs from preprocessed_data.json,
showing their associated comments.

Usage:
  1. Make sure preprocessed_data.json is in the same folder.
  2. Install needed libs: pip install scikit-learn
  3. Run: python retrieval_demo.py
  4. When prompted, paste your diff. End input with Ctrl+D (Linux/Mac) or
     Ctrl+Z followed by Enter (Windows). Then see the top matches.
"""

import json
import numpy as np
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

DATA_FILE = "preprocessed_data.json"

def main():
    # 1. Load the preprocessed data
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    diff_texts = [item["diff_text"] for item in data]
    comment_texts = [item["comment_text"] for item in data]

    # 2. Build TF-IDF vectorizer on all diff_text
    vectorizer = TfidfVectorizer(
        stop_words="english", 
        max_features=50000,
        ngram_range=(1, 2)
    )
    diff_matrix = vectorizer.fit_transform(diff_texts)
    print(f"Indexed {diff_matrix.shape[0]} diffs with TF-IDF size {diff_matrix.shape[1]}.")

    def find_similar_diffs(new_diff, top_k=3):
        """
        Given a new diff string, return the top_k most similar diffs from the dataset,
        along with original comment_text and a similarity score.
        """
        query_vec = vectorizer.transform([new_diff])
        similarities = linear_kernel(query_vec, diff_matrix).flatten()  # shape: (N,)

        best_indices = np.argpartition(similarities, -top_k)[-top_k:]
        best_indices = best_indices[np.argsort(-similarities[best_indices])]

        results = []
        for idx in best_indices:
            score = similarities[idx]
            results.append({
                "diff_text": diff_texts[idx],
                "comment_text": comment_texts[idx],
                "similarity_score": float(score)
            })
        return results

    # 3. Prompt user for a diff snippet
    print("================================================")
    print("Please paste/enter your diff snippet (or any text).")
    print("End input with Ctrl+D (Linux/Mac) or Ctrl+Z + Enter (Windows).")
    print("================================================")

    # Read entire input from stdin
    user_diff = sys.stdin.read().strip()

    if not user_diff:
        print("No input received. Exiting.")
        return

    # 4. Retrieve top-3 matches
    top_matches = find_similar_diffs(user_diff, top_k=3)

    print("\n=== Top Matches ===")
    for i, match in enumerate(top_matches, 1):
        score = match["similarity_score"]
        snippet = match["diff_text"][:200].replace("\n", " ")
        print(f"Rank #{i} (Cosine Score={score:.4f})")
        print(f"  Matched Diff Snippet: {snippet}...")
        print(f"  Matched Comment: {match['comment_text']}")
        print("-" * 50)

if __name__ == "__main__":
    main()
