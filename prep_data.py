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
BATCH_SIZE = 1000  # How many rows to process before printing a progress update

def main():
    # Ensure the DB file exists
    if not os.path.exists(DB_NAME):
        print(f"Database file '{DB_NAME}' not found.")
        return

    print("Connecting to the database...")
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Prepare the SQL query (joining the three tables)
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
    
    print("Executing SQL query...")
    rows = cursor.execute(query)

    data_examples = []
    total_rows = 0
    kept_examples = 0

    # Process rows one by one to provide live updates
    for row in rows:
        total_rows += 1

        # Print progress update at each batch interval
        if total_rows % BATCH_SIZE == 0:
            print(f"Processed {total_rows} rows so far; kept {kept_examples} examples.")

        # Unpack row data
        pr_number       = row[0]
        pr_title        = row[1] or ""
        pr_description  = row[2] or ""
        filename        = row[3] or ""
        diff_text       = row[4] or ""
        commenter       = row[5] or ""
        comment_text    = row[6] or ""
        comment_file    = row[7]  # could be None
        comment_line    = row[8]  # could be None
        comment_created = row[9]  # expected as string e.g. "2025-03-30 12:34:56"

        # Clean text data
        diff_text = diff_text.strip()
        comment_text = comment_text.strip()

        # Skip empty diffs
        if not diff_text:
            continue

        # Skip trivial comments (e.g., less than 5 characters)
        if len(comment_text) < 5:
            continue

        # Build the cleaned example dictionary
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
        kept_examples += 1

    conn.close()

    # Write the cleaned data to JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data_examples, f, indent=2, ensure_ascii=False)

    print(f"Finished processing. Total rows processed: {total_rows}.")
    print(f"Done! Wrote {kept_examples} examples to {OUTPUT_JSON}.")

if __name__ == "__main__":
    main()
