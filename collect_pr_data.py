"""
collect_pr_data.py (Descending Order)

1. Reads your GitHub config (token, owner, repo) and date range (start_date, end_date) from config.ini.
2. Connects to GitHub via PyGithub.
3. Creates/opens an SQLite DB and sets up tables if needed.
4. Fetches PRs in descending order by created date, only processing those within [start_date, end_date].
5. Stores each PR's metadata, file diffs, and comments.
To run:
    python collect_pr_data.py
"""

import configparser
import sqlite3
import time
from datetime import datetime
from github import Github

DB_NAME = "pull_requests.db"

def main():
    # -------------------------------------------------------------
    # 1. Parse the config.ini
    # -------------------------------------------------------------
    config = configparser.ConfigParser()
    config.read("config.ini")

    github_token = config["github"]["token"]
    owner = config["github"]["owner"]
    repo_name = config["github"]["repo"]

    start_str = config["range"]["start_date"]  # e.g. "2025-03-25"
    end_str = config["range"]["end_date"]      # e.g. "2025-04-01"

    # Convert strings to naive datetime objects (no timezone)
    start_date = datetime.strptime(start_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_str, "%Y-%m-%d")

    print("Starting PR collection ...")
    print(f"Processing PRs created between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}", flush=True)

    # -------------------------------------------------------------
    # 2. Connect to GitHub
    # -------------------------------------------------------------
    g = Github(github_token)
    full_repo_name = f"{owner}/{repo_name}"
    repo = g.get_repo(full_repo_name)

    # -------------------------------------------------------------
    # 3. Create/Connect to SQLite DB & Create Tables if needed
    # -------------------------------------------------------------
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS PullRequests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pr_number INTEGER,
        title TEXT,
        description TEXT,
        author TEXT,
        created_at TEXT,
        merged_at TEXT,
        state TEXT,
        labels TEXT
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS PRFiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pr_id INTEGER,
        filename TEXT,
        diff_text TEXT,
        FOREIGN KEY (pr_id) REFERENCES PullRequests(id)
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS PRComments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pr_id INTEGER,
        commenter TEXT,
        comment_text TEXT,
        file_path TEXT,
        line_number INTEGER,
        created_at TEXT,
        FOREIGN KEY (pr_id) REFERENCES PullRequests(id)
    );
    """)

    conn.commit()

    # -------------------------------------------------------------
    # 4. Fetch Pull Requests in DESC order by created date
    # -------------------------------------------------------------
    pulls = repo.get_pulls(state="all", sort="created", direction="desc")
    pr_count = 0

    print("Fetching PRs from GitHub ...", flush=True)

    for pr in pulls:
        created_utc = pr.created_at
        # Convert offset-aware datetime to naive
        created_utc_naive = created_utc.replace(tzinfo=None)

        # If this PR is newer than end_date, skip it
        if created_utc_naive > end_date:
            continue

        # If this PR is older than start_date, then all subsequent PRs
        # will be older (since we're going in descending order), so break.
        if created_utc_naive < start_date:
            break

        # Now we know pr is in the desired date range.
        pr_number = pr.number
        title = pr.title or ""
        body = pr.body or ""
        author = pr.user.login if pr.user else "unknown"
        created_at_str = str(pr.created_at)
        merged_at_str = str(pr.merged_at) if pr.merged_at else None
        state = pr.state
        labels_str = ",".join([label.name for label in pr.labels])

        # Print progress for each PR
        print(f"Processing PR #{pr_number} created on {created_at_str}", flush=True)

        # Insert into PullRequests
        cursor.execute("""
            INSERT INTO PullRequests
            (pr_number, title, description, author, created_at, merged_at, state, labels)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (pr_number, title, body, author, created_at_str, merged_at_str, state, labels_str))

        new_pr_id = cursor.lastrowid

        # 4a. Fetch Files (Diffs)
        files = pr.get_files()
        for f in files:
            filename = f.filename
            diff_text = f.patch or ""
            cursor.execute("""
                INSERT INTO PRFiles (pr_id, filename, diff_text)
                VALUES (?, ?, ?)
            """, (new_pr_id, filename, diff_text))

        # 4b. Fetch Comments
        comments = pr.get_comments()
        for c in comments:
            commenter = c.user.login if c.user else "unknown"
            comment_text = (c.body or "").strip()
            created_date_str = str(c.created_at)
            file_path = getattr(c, 'path', None)
            line_number = getattr(c, 'original_line', None)

            # Optionally skip very short comments
            if len(comment_text) < 5:
                continue

            cursor.execute("""
                INSERT INTO PRComments
                (pr_id, commenter, comment_text, file_path, line_number, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (new_pr_id, commenter, comment_text, file_path, line_number, created_date_str))

        conn.commit()
        pr_count += 1

        # Print an update every 10 PRs processed
        if pr_count % 10 == 0:
            print(f"Total PRs processed so far: {pr_count}", flush=True)

        # Optionally reduce or remove the sleep if you want maximum speed
        #time.sleep(0.2)

    # -------------------------------------------------------------
    # 5. Done
    # -------------------------------------------------------------
    conn.close()
    print(f"Done! Inserted {pr_count} PRs into {DB_NAME} for the range "
          f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.", flush=True)


if __name__ == "__main__":
    main()
