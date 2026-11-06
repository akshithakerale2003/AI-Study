"""AI Study Helper API - a FastAPI mini project using generative AI (Claude)."""
"""AI Study Helper API - a FastAPI mini project using generative AI (Claude)."""
import json
import os
from contextlib import asynccontextmanager
from typing import Literal

from anthropic import Anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()

import db  # noqa: E402
from rag import router as rag_router  # noqa: E402

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment

@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()  # waits for PostgreSQL, creates tables, opens the pool
    yield
    db.close_db()


app = FastAPI(
    lifespan=lifespan,
    title="AI Study Helper",
    description="Summarize, rewrite, quiz and chat - powered by generative AI.",
    version="2.0.0",
)
app.include_router(rag_router)


# ---------- helpers ----------
def ask_claude(system: str, user: str, max_tokens: int = 1000) -> str:
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text
    except Exception as e:  # keep the mini project simple
        raise HTTPException(status_code=502, detail=f"AI error: {e}")


# ---------- schemas ----------
class TextIn(BaseModel):
    text: str = Field(..., min_length=20, description="Text to process")


class SummaryIn(TextIn):
    style: Literal["bullets", "paragraph", "eli5"] = "bullets"


class RewriteIn(TextIn):
    tone: Literal["formal", "casual", "friendly", "professional"] = "professional"


class QuizIn(TextIn):
    num_questions: int = Field(5, ge=1, le=10)


class ChatIn(BaseModel):
    message: str = Field(..., min_length=1)


# ---------- routes ----------
@app.get("/")
def root():
    return {"app": "AI Study Helper", "docs": "/docs"}


@app.get("/health")
def health():
    """Checks that the API can reach PostgreSQL."""
    try:
        return {"status": "ok", "database": "connected" if db.ping() else "error"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e}")


@app.get("/history")
def history(limit: int = 20):
    """Latest saved AI requests (summarize / rewrite / quiz)."""
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT id, endpoint, input, output, created_at FROM history "
            "ORDER BY id DESC LIMIT %s",
            (min(limit, 100),),
        ).fetchall()
    return [
        {"id": r[0], "endpoint": r[1], "input": r[2][:200], "output": r[3], "created_at": r[4]}
        for r in rows
    ]


@app.post("/summarize")
def summarize(body: SummaryIn):
    styles = {
        "bullets": "5 concise bullet points",
        "paragraph": "one short paragraph",
        "eli5": "a simple explanation a 10-year-old would understand",
    }
    summary = ask_claude(
        "You are a precise study assistant.",
        f"Summarize the text below as {styles[body.style]}.\n\n{body.text}",
    )
    db.log_history("summarize", body.text, summary)
    return {"style": body.style, "summary": summary}


@app.post("/rewrite")
def rewrite(body: RewriteIn):
    result = ask_claude(
        "You are an expert editor. Return only the rewritten text.",
        f"Rewrite the text below in a {body.tone} tone, keeping the meaning.\n\n{body.text}",
    )
    db.log_history("rewrite", body.text, result)
    return {"tone": body.tone, "rewritten": result}


@app.post("/quiz")
def quiz(body: QuizIn):
    prompt = (
        f"Create {body.num_questions} multiple-choice questions from the text below. "
        "Respond with ONLY valid JSON: a list of objects with keys "
        '"question", "options" (list of 4 strings), "answer" (one of the options).\n\n'
        f"{body.text}"
    )
    raw = ask_claude("You generate quizzes as strict JSON.", prompt, max_tokens=2000)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        questions = json.loads(raw)
        db.log_history("quiz", body.text, raw)
        return {"questions": questions}
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="Model returned invalid JSON, try again.")


@app.post("/chat/stream")
def chat_stream(body: ChatIn):
    """Streams the answer token-by-token (Server-Sent style plain text)."""

    def token_gen():
        with client.messages.stream(
            model=MODEL,
            max_tokens=1000,
            system="You are a friendly tutor. Keep answers clear and short.",
            messages=[{"role": "user", "content": body.message}],
        ) as stream:
            for text in stream.text_stream:
                yield text

    return StreamingResponse(token_gen(), media_type="text/plain")