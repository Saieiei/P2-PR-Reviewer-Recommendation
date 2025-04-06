"""
prep_data.py

Purpose:
  1. Reads from the pull_requests.db database, which contains tables:
     - PullRequests (metadata)
     - PRFiles (diff_text per file)
     - PRComments (comments from reviewers)
  2. Joins these tables to gather (diff, comment) pairs.
  3. Filters out trivial or empty data.
  4. Saves cleaned examples to preprocessed_data.json (for retrieval-based usage).

Usage:
  python prep_data.py
"""

import sqlite3
import json
import os

DB_NAME = "pull_requests.db"
OUTPUT_JSON = "preprocessed_data.json"

def main():
    # Ensure the DB file exists
    if not os.path.exists(DB_NAME):
        print(f"Database file '{DB_NAME}' not found.")
        return

    # 1. Connect to the database
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # 2. Query the joined data
    #    - We join PullRequests.id = PRFiles.pr_id = PRComments.pr_id
    #    - That means we get a row for each combination of file + comment in the same PR.
    query = """
    SELECT
        PullRequests.pr_number,
        PullRequests.title,
        PullRequests.description,
        PRFiles.filename,
        PRFiles.diff_text,
        PRComments.commenter,
        PRComments.comment_text,
        PRComments.file_path,
        PRComments.line_number,
        PRComments.created_at
    FROM PullRequests
    JOIN PRFiles ON PullRequests.id = PRFiles.pr_id
    JOIN PRComments ON PullRequests.id = PRComments.pr_id
    """
    rows = cursor.execute(query).fetchall()

    # 3. Build and filter the dataset
    data_examples = []
    for row in rows:
        # Unpack
        pr_number       = row[0]
        pr_title        = row[1] or ""
        pr_description  = row[2] or ""
        filename        = row[3] or ""
        diff_text       = row[4] or ""
        commenter       = row[5] or ""
        comment_text    = row[6] or ""
        comment_file    = row[7]  # might be None
        comment_line    = row[8]  # might be None
        comment_created = row[9]  # string, e.g. "2025-03-30 12:34:56"

        # Basic cleaning
        comment_text = comment_text.strip()
        diff_text = diff_text.strip()

        # Skip empty diffs
        if not diff_text:
            continue

        # Skip short comments (e.g., < 5 chars is trivial like "ok", "LGTM", etc.)
        if len(comment_text) < 5:
            continue

        # (Optional) skip if not a C/C++ file? e.g. if you only want C/C++:
        # if not filename.endswith((".c", ".cpp", ".h", ".hpp", ".cc")):
        #     continue

        # Build the cleaned example
        example = {
            "pr_number": pr_number,
            "title": pr_title,
            "description": pr_description,
            "filename": filename,
            "diff_text": diff_text,
            "commenter": commenter,
            "comment_text": comment_text,
            "comment_file_path": comment_file,
            "comment_line_num": comment_line,
            "comment_created_at": comment_created
        }
        data_examples.append(example)

    conn.close()

    # 4. Save the cleaned data to JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data_examples, f, indent=2, ensure_ascii=False)

    print(f"Done! Wrote {len(data_examples)} examples to {OUTPUT_JSON}.")

if __name__ == "__main__":
    main()
