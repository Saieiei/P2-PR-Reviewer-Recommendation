"""
cache_db.py

This module implements a SQLite-based cache for storing embeddings.
Each cached record is identified by a unique key generated as a SHA-256 checksum of the combined text.
The cache is stored in 'cache.db'.
"""

import sqlite3
import hashlib
import pickle

CACHE_DB_PATH = "cache.db"

def init_cache_db():
    """
    Initializes the cache database and creates the CachedEmbeddings table if it doesn't exist.
    Returns a connection object with a 30-second timeout, busy_timeout set, and an attempt to enable WAL mode.
    """
    # Open the connection in autocommit mode (isolation_level=None) with a 30-second timeout.
    conn = sqlite3.connect(CACHE_DB_PATH, timeout=30, isolation_level=None)
    # Set busy timeout so SQLite waits longer for locks to clear.
    conn.execute("PRAGMA busy_timeout = 30000;")
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
    except sqlite3.OperationalError as e:
        print("Warning: Could not set WAL mode due to:", e)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS CachedEmbeddings (
            key TEXT PRIMARY KEY,
            embedding BLOB,
            data TEXT
        )
    """)
    conn.commit()
    return conn

def compute_checksum(text: str) -> str:
    """
    Computes and returns a SHA-256 checksum for the given text.
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def get_cached_embedding(conn, key: str):
    """
    Checks whether an embedding exists in the cache DB for the given key.
    Returns the embedding if found; otherwise, returns None.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT embedding FROM CachedEmbeddings WHERE key = ?", (key,))
    row = cursor.fetchone()
    if row:
        return pickle.loads(row[0])
    return None

def save_cached_embedding(conn, key: str, embedding, data: str):
    """
    Saves the embedding and its associated data (the combined text) into the cache DB.
    """
    cursor = conn.cursor()
    embedding_blob = pickle.dumps(embedding)
    cursor.execute("INSERT OR REPLACE INTO CachedEmbeddings (key, embedding, data) VALUES (?, ?, ?)",
                   (key, embedding_blob, data))
    conn.commit()
