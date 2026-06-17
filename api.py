"""
api.py
------
FastAPI backend for the ADHD RAG pipeline.

Endpoints:
  GET  /                           - health check
  GET  /status                     - pipeline status
  POST /chat                       - single question, stateless
  POST /chat/session               - multi-turn with session memory
  GET  /chat/session/{id}/history  - get conversation history
  DELETE /chat/session/{id}        - clear a session

Run with:
  uvicorn api:app --reload --port 8000
"""

import os
import sys
import uuid
import time
from typing import Optional
from datetime import datetime
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from pipeline.embedder import load_index, EMBEDDING_MODEL
from pipeline.retriever import retrieve
from pipeline.generator import (
    generate,
    OLLAMA_MODEL,
    MAX_CHUNKS,
    MAX_CHUNK_CHARS,
    SYSTEM_PROMPT,
    _format_citations,
)
from sentence_transformers import SentenceTransformer

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ADHD RAG Assistant API",
    description="Healthcare RAG pipeline for ADHD information. All answers grounded in verified sources.",
    version="1.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pipeline ──────────────────────────────────────────────────────────────────
index       = None
chunks      = None
embed_model = None

# ── Session store ─────────────────────────────────────────────────────────────
sessions: dict[str, list[dict]] = {}
MAX_HISTORY_TURNS = 6

# ── Rate limiting — simple in-memory per IP ───────────────────────────────────
RATE_LIMIT_REQUESTS = 20    # max requests
RATE_LIMIT_WINDOW   = 60    # per 60 seconds
rate_store: dict[str, list[float]] = defaultdict(list)

# ── Config ────────────────────────────────────────────────────────────────────
MAX_QUESTION_LENGTH = 500
DISCLAIMER = (
    "This information is for educational purposes only. "
    "Please consult a qualified healthcare professional for personal medical advice."
)


# ── Request / Response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    category_filter: Optional[str] = None

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty")
        if len(v) > MAX_QUESTION_LENGTH:
            raise ValueError(f"Question too long (max {MAX_QUESTION_LENGTH} chars)")
        return v

    @field_validator("top_k")
    @classmethod
    def top_k_must_be_valid(cls, v):
        if v is not None and (v < 1 or v > 10):
            raise ValueError("top_k must be between 1 and 10")
        return v


class SessionChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    top_k: Optional[int] = 5

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty")
        if len(v) > MAX_QUESTION_LENGTH:
            raise ValueError(f"Question too long (max {MAX_QUESTION_LENGTH} chars)")
        return v


class ChatResponse(BaseModel):
    answer: str
    citations: str
    chunks_used: int
    model: str
    disclaimer: str


class SessionChatResponse(BaseModel):
    answer: str
    citations: str
    chunks_used: int
    model: str
    session_id: str
    turn_number: int
    disclaimer: str


class StatusResponse(BaseModel):
    status: str
    chunks_indexed: int
    model: str
    sessions_active: int
    timestamp: str


# ── Rate limiting middleware ───────────────────────────────────────────────────

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Only rate limit POST endpoints
    if request.method == "POST":
        client_ip = request.client.host
        now       = time.time()
        window    = rate_store[client_ip]

        # Remove timestamps outside the window
        rate_store[client_ip] = [t for t in window if now - t < RATE_LIMIT_WINDOW]

        if len(rate_store[client_ip]) >= RATE_LIMIT_REQUESTS:
            return JSONResponse(
                status_code=429,
                content={"detail": f"Rate limit exceeded. Max {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW}s."}
            )

        rate_store[client_ip].append(now)

    return await call_next(request)


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    global index, chunks, embed_model
    print("Loading RAG pipeline...")
    try:
        index, chunks = load_index()
        embed_model   = SentenceTransformer(EMBEDDING_MODEL)
        print(f"Pipeline ready. {index.ntotal} chunks indexed.")
    except Exception as e:
        print(f"ERROR loading pipeline: {e}")
        raise


# ── Helper: check pipeline loaded ─────────────────────────────────────────────

def _require_pipeline():
    if index is None or embed_model is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded. Try again in a moment.")


# ── Helper: build session response ────────────────────────────────────────────

def _generate_session_response(
    question: str,
    session_id: str,
    history: list[dict],
    top_k: int,
) -> dict:
    """Retrieve + generate with session history context."""
    import ollama

    # Enrich query — only use last user question for context
    retrieval_query = question
    user_msgs = [
        m["content"].replace("USER QUESTION:", "").strip()
        for m in history
        if m["role"] == "user"
    ]
    if user_msgs:
        last = user_msgs[-1][:80]
        retrieval_query = f"{last} {question}"

    # Retrieve
    retrieved = retrieve(
        retrieval_query,
        index,
        chunks,
        embed_model,
        top_k=top_k,
    )

    # Build context block
    context_parts = []
    for i, chunk in enumerate(retrieved[:MAX_CHUNKS], 1):
        text = chunk["text"]
        if len(text) > MAX_CHUNK_CHARS:
            text = text[:MAX_CHUNK_CHARS] + "..."
        context_parts.append(
            f"CONTEXT SECTION {i}: {chunk['heading']} "
            f"(source: {chunk['source_file']})\n{text}"
        )
    context_block = "\n\n".join(context_parts)

    user_message = f"""Below are relevant sections from a verified ADHD knowledge base.
Use ONLY this information to answer the question.

{context_block}

USER QUESTION: {question}

Please provide a helpful, accurate answer based strictly on the context above."""

    # Build messages with history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-MAX_HISTORY_TURNS:])
    messages.append({"role": "user", "content": user_message})

    # Generate
    try:
        response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
        answer   = response["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {str(e)}. Is Ollama running?")

    # Update history
    history.append({"role": "user",      "content": f"USER QUESTION: {question}"})
    history.append({"role": "assistant", "content": answer})

    # Trim
    if len(history) > MAX_HISTORY_TURNS * 2:
        sessions[session_id] = history[-(MAX_HISTORY_TURNS * 2):]

    turn_number = len([m for m in sessions[session_id] if m["role"] == "user"])
    citations   = _format_citations(retrieved)

    return {
        "answer":      answer,
        "citations":   citations,
        "chunks_used": min(len(retrieved), MAX_CHUNKS),
        "model":       OLLAMA_MODEL,
        "session_id":  session_id,
        "turn_number": turn_number,
        "disclaimer":  DISCLAIMER,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
async def health_check():
    return {"status": "ok", "service": "ADHD RAG Assistant", "version": "1.0.0"}


@app.get("/status", response_model=StatusResponse)
async def get_status():
    _require_pipeline()
    return StatusResponse(
        status          = "ok",
        chunks_indexed  = index.ntotal,
        model           = EMBEDDING_MODEL,
        sessions_active = len(sessions),
        timestamp       = datetime.utcnow().isoformat(),
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Single-turn stateless chat."""
    _require_pipeline()

    retrieved = retrieve(
        request.question,
        index,
        chunks,
        embed_model,
        top_k=request.top_k,
        category_filter=request.category_filter,
    )

    try:
        result = generate(request.question, retrieved)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {str(e)}. Is Ollama running?")

    return ChatResponse(
        answer      = result["answer"],
        citations   = result["citations"],
        chunks_used = result["chunks_used"],
        model       = result["model"],
        disclaimer  = DISCLAIMER,
    )


@app.post("/chat/session", response_model=SessionChatResponse)
async def chat_with_session(request: SessionChatRequest):
    """Multi-turn chat with session memory."""
    _require_pipeline()

    # Create or retrieve session
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = []

    result = _generate_session_response(
        question   = request.question,
        session_id = session_id,
        history    = sessions[session_id],
        top_k      = request.top_k,
    )

    return SessionChatResponse(**result)


@app.get("/chat/session/{session_id}/history")
async def get_session_history(session_id: str):
    """Get conversation history for a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    history = sessions[session_id]
    cleaned = []
    for msg in history:
        content = msg["content"]
        if "USER QUESTION:" in content:
            content = content.split("USER QUESTION:")[-1].strip()
        cleaned.append({"role": msg["role"], "content": content[:300]})

    return {
        "session_id": session_id,
        "turns":      len([m for m in cleaned if m["role"] == "user"]),
        "history":    cleaned,
    }


@app.delete("/chat/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del sessions[session_id]
    return {"status": "cleared", "session_id": session_id}