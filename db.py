"""PostgreSQL connection layer: connection pool, schema setup, helpers."""
import os
import time
from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/studyhelper"
)

# Inside a Docker container "localhost" is the API container itself, which has
# no database. If we are in Docker and the URL points at localhost, use the
# compose service name "db" on its internal port 5432 instead.
if os.path.exists("/.dockerenv"):
    DATABASE_URL = "postgresql://postgres:postgres@db:5432/studyhelper"
print(f"[db] using database host: {DATABASE_URL.split('@')[-1]}")

SCHEMA = [
    "CREATE EXTENSION IF NOT EXISTS vector",
    """CREATE TABLE IF NOT EXISTS chunks (
        id          SERIAL PRIMARY KEY,
        title       TEXT NOT NULL,
        chunk_index INT  NOT NULL,
        content     TEXT NOT NULL,
        embedding   vector(384) NOT NULL
    )""",
    """CREATE INDEX IF NOT EXISTS chunks_embedding_idx
       ON chunks USING hnsw (embedding vector_cosine_ops)""",
    """CREATE TABLE IF NOT EXISTS history (
        id         SERIAL PRIMARY KEY,
        endpoint   TEXT NOT NULL,
        input      TEXT NOT NULL,
        output     TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )""",
]

# Pool is created closed; opened in init_db() after the schema exists,
# because register_vector needs the "vector" extension to be present.
pool = ConnectionPool(
    DATABASE_URL,
    min_size=1,
    max_size=10,
    open=False,
    kwargs={"autocommit": True},
    configure=register_vector,
)


def init_db(retries: int = 15, delay: float = 2.0) -> None:
    """Wait for Postgres (it may still be starting in Docker), then create tables."""
    for attempt in range(1, retries + 1):
        try:
            with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
                for stmt in SCHEMA:
                    conn.execute(stmt)
            break
        except psycopg.OperationalError as e:
            if attempt == retries:
                raise RuntimeError(f"Could not connect to PostgreSQL: {e}")
            print(f"[db] waiting for PostgreSQL ({attempt}/{retries})...")
            time.sleep(delay)
    pool.open()
    pool.wait()
    print("[db] connected to PostgreSQL")


def close_db() -> None:
    pool.close()


@contextmanager
def get_conn():
    with pool.connection() as conn:
        yield conn


def ping() -> bool:
    with get_conn() as conn:
        return conn.execute("SELECT 1").fetchone()[0] == 1


def log_history(endpoint: str, input_text: str, output_text: str) -> None:
    """Save an AI request/response. Never breaks the API if the DB write fails."""
    try:
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO history (endpoint, input, output) VALUES (%s, %s, %s)",
                (endpoint, input_text, output_text),
            )
    except Exception as e:
        print(f"[db] history write failed: {e}")