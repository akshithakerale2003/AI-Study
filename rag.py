"""RAG module: store document chunks as embeddings in Postgres (pgvector) and
answer questions with Claude using the most relevant chunks."""
import os
from functools import lru_cache

from anthropic import Anthropic
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from db import get_conn, log_history

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # 384 dimensions, runs locally on CPU

router = APIRouter(prefix="/rag", tags=["RAG (pgvector)"])
client = Anthropic()


# ---------- helpers ----------
@lru_cache(maxsize=1)
def get_embedder():
    from fastembed import TextEmbedding  # lazy import: model loads on first use

    return TextEmbedding(model_name=EMBED_MODEL)


def embed(texts: list[str]):
    return list(get_embedder().embed(texts))


def chunk_text(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    words = text.split()
    step = size - overlap
    return [
        " ".join(words[i : i + size])
        for i in range(0, len(words), step)
        if words[i : i + size]
    ]


def store_document(title: str, text: str) -> int:
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text found to ingest.")
    vectors = embed(chunks)
    rows = [(title, i, c, v) for i, (c, v) in enumerate(zip(chunks, vectors))]
    with get_conn() as conn, conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (title, chunk_index, content, embedding) "
            "VALUES (%s, %s, %s, %s)",
            rows,
        )
    return len(chunks)


# ---------- schemas ----------
class IngestIn(BaseModel):
    title: str = Field(..., min_length=1)
    text: str = Field(..., min_length=20)


class AskIn(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(4, ge=1, le=10)


# ---------- routes ----------
@router.post("/ingest")
def ingest(body: IngestIn):
    n = store_document(body.title, body.text)
    return {"title": body.title, "chunks_stored": n}


@router.post("/upload")
def upload(file: UploadFile = File(...)):
    """Upload a .txt, .md or .pdf file and store it as embeddings."""
    name = file.filename or "upload"
    if name.lower().endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(file.file)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = file.file.read().decode("utf-8", errors="ignore")
    n = store_document(name, text)
    return {"title": name, "chunks_stored": n}


@router.get("/documents")
def documents():
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT title, COUNT(*) FROM chunks GROUP BY title ORDER BY title"
        )
        return [{"title": t, "chunks": c} for t, c in cur.fetchall()]


@router.post("/ask")
def ask(body: AskIn):
    q_vec = embed([body.question])[0]
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT title, chunk_index, content, 1 - (embedding <=> %s) AS score "
            "FROM chunks ORDER BY embedding <=> %s LIMIT %s",
            (q_vec, q_vec, body.top_k),
        )
        hits = cur.fetchall()

    if not hits:
        raise HTTPException(status_code=404, detail="No documents ingested yet.")

    context = "\n\n".join(
        f"[{i + 1}] ({title}) {content}" for i, (title, _, content, _) in enumerate(hits)
    )
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            system=(
                "Answer the question using ONLY the numbered context. "
                "Cite sources like [1]. If the answer is not in the context, say so."
            ),
            messages=[
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {body.question}",
                }
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI error: {e}")

    answer = resp.content[0].text
    log_history("rag/ask", body.question, answer)
    return {
        "answer": answer,
        "sources": [
            {"title": t, "chunk": ci, "score": round(float(s), 4), "preview": c[:150]}
            for t, ci, c, s in hits
        ],
    }