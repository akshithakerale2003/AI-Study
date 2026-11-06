# AI Study Helper (FastAPI + Generative AI + pgvector + Docker)

A mini project exposing generative AI features as REST APIs, now with
**RAG** (chat with your own documents) using **Postgres + pgvector**.

## Features
| Endpoint | What it does |
|---|---|
| `POST /summarize` | Summarize text (bullets / paragraph / ELI5) |
| `POST /rewrite` | Rewrite text in a chosen tone |
| `POST /quiz` | Generate MCQs as JSON from any text |
| `POST /chat/stream` | Streaming tutor chat |
| `POST /rag/ingest` | Chunk + embed text and store it in pgvector |
| `POST /rag/upload` | Same, for `.txt` / `.md` / `.pdf` files |
| `GET /rag/documents` | List stored documents |
| `POST /rag/ask` | Ask a question; answer is grounded in retrieved chunks with sources |

| `GET /health` | Checks the backend ↔ PostgreSQL connection (shows Postgres + pgvector versions) |
| `GET /history` | Last AI requests saved in PostgreSQL |

## How the backend connects to PostgreSQL
```
 Browser / curl ──► api container (FastAPI :8000) ──► db container (Postgres + pgvector :5432)
                         db.py: connection pool           volume: pgdata (data survives restarts)
```
- `docker-compose.yml` starts the `db` service and the `api` service on one network. Inside it the API reaches the database by the hostname **`db`**, using `DATABASE_URL=postgresql://postgres:postgres@db:5432/studyhelper`.
- `db.py` keeps a shared connection pool. On startup it retries until Postgres is ready, then runs `init.sql` (creates the `vector` extension and the `chunks` and `history` tables).
- Running the API outside Docker? Start only the DB (`docker compose up db -d`) and use `localhost` in `.env` (see `.env.example`).
- Verify it: `curl http://localhost:8000/health` should return `"status": "ok"`.

## How RAG works here
1. Text is split into ~200-word overlapping chunks.
2. Each chunk is embedded locally (`BAAI/bge-small-en-v1.5`, 384-dim) with `fastembed`.
3. Vectors are stored in the `chunks` table (`vector(384)` + HNSW cosine index).
4. On `/rag/ask`, the question is embedded, the nearest chunks are fetched with `ORDER BY embedding <=> query`, and Claude answers using only that context.

## Run with Docker (recommended)
```bash
cp .env.example .env        # add your ANTHROPIC_API_KEY
docker compose up --build
```
Open http://localhost:8000/docs. The first `/rag` call downloads the embedding model (cached in a Docker volume).

## Try it
```bash
curl -X POST http://localhost:8000/rag/ingest \
  -H "Content-Type: application/json" \
  -d '{"title": "fastapi-notes", "text": "FastAPI is a modern Python web framework built on Starlette and Pydantic. It uses type hints for validation and generates OpenAPI docs automatically. It supports async endpoints and dependency injection."}'

curl -X POST http://localhost:8000/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What does FastAPI generate automatically?"}'

curl -X POST http://localhost:8000/rag/upload -F "file=@notes.pdf"
```

## Run without Docker for the API
Start only the database, then run the API locally:
```bash
docker compose up db -d
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

## Useful commands
```bash
docker compose exec db psql -U postgres -d studyhelper -c "SELECT title, COUNT(*) FROM chunks GROUP BY title;"
docker compose down -v      # stop and delete all data
```

## Ideas to extend
- Add a simple HTML/JS chat frontend
- Add document deletion and per-user collections
- Hybrid search (pgvector + Postgres full-text search)
- API-key auth and rate limiting