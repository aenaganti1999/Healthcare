"""
generator.py
------------
Takes a user query + retrieved chunks from retriever.py.
Builds a safe medical prompt.
Calls local Ollama LLM (llama3.2:3b).
Returns a grounded answer with source citations.

This completes the full RAG loop:
  Query → Retrieve → Augment Prompt → Generate → Answer
"""

import os
import sys

import ollama

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# ── Model config ──────────────────────────────────────────────────────────────
OLLAMA_MODEL   = "llama3.2:3b"
MAX_CHUNKS     = 5       # max chunks to include in prompt
MAX_CHUNK_CHARS = 800    # truncate individual chunks to keep prompt manageable

# ── System prompt ─────────────────────────────────────────────────────────────
# This is the most important part of a healthcare RAG system.
# It tells the LLM exactly how to behave: grounded, safe, no hallucinations.
SYSTEM_PROMPT = """You are a helpful ADHD information assistant for a healthcare application.

Your role is to provide accurate, grounded information about ADHD based ONLY on the context provided to you.

STRICT RULES you must always follow:
1. Answer ONLY using information from the provided context sections.
2. If the context does not contain enough information to answer, say: "I don't have enough information in my knowledge base to answer that question fully."
3. Never invent facts, medications, dosages, statistics, or medical claims not present in the context.
4. Always end your response with a disclaimer: "This information is for educational purposes only. Please consult a qualified healthcare professional for personal medical advice."
5. If the question involves a crisis, self-harm, or emergency, respond with: "Please contact a healthcare professional or emergency services immediately."
6. Keep answers clear, compassionate, and factual.
7. Do not diagnose. Do not recommend specific medications or dosages.
"""


def _build_prompt(query: str, chunks: list[dict]) -> str:
    """
    Build the augmented prompt from query + retrieved chunks.

    Structure:
      CONTEXT SECTION 1: [chunk heading] (source file)
      [chunk text]

      CONTEXT SECTION 2: ...

      USER QUESTION: [query]
    """
    context_parts = []

    for i, chunk in enumerate(chunks[:MAX_CHUNKS], 1):
        # Truncate chunk text if too long
        text = chunk["text"]
        if len(text) > MAX_CHUNK_CHARS:
            text = text[:MAX_CHUNK_CHARS] + "..."

        context_parts.append(
            f"CONTEXT SECTION {i}: {chunk['heading']} "
            f"(source: {chunk['source_file']})\n{text}"
        )

    context_block = "\n\n".join(context_parts)

    prompt = f"""Below are relevant sections from a verified ADHD knowledge base.
Use ONLY this information to answer the question.

{context_block}

USER QUESTION: {query}

Please provide a helpful, accurate answer based strictly on the context above."""

    return prompt


def _format_citations(chunks: list[dict]) -> str:
    """Format source citations from retrieved chunks."""
    seen = set()
    citations = []

    for chunk in chunks[:MAX_CHUNKS]:
        source = chunk["source_file"]
        if source not in seen:
            seen.add(source)
            # Convert path to readable name
            # e.g. '05_medications/stimulant_medications.md' → 'Stimulant Medications'
            name = (
                source
                .split("/")[-1]
                .replace(".md", "")
                .replace("_", " ")
                .title()
            )
            citations.append(f"  - {name} ({source})")

    return "\n".join(citations)


def generate(
    query: str,
    chunks: list[dict],
    model: str = OLLAMA_MODEL,
    verbose: bool = False,
) -> dict:
    """
    Generate a grounded answer from query + retrieved chunks.

    Args:
        query   : the user's question
        chunks  : retrieved chunks from retriever.py
        model   : ollama model name
        verbose : if True, print the full prompt before generating

    Returns dict:
        {
            answer     : the LLM's response text
            citations  : formatted source citations
            query      : original query
            chunks_used: number of chunks included in prompt
            model      : model used
        }
    """
    if not chunks:
        return {
            "answer": (
                "I don't have enough information in my knowledge base "
                "to answer that question. "
                "This information is for educational purposes only. "
                "Please consult a qualified healthcare professional "
                "for personal medical advice."
            ),
            "citations":   "",
            "query":       query,
            "chunks_used": 0,
            "model":       model,
        }

    # ── Build prompt ──────────────────────────────────────────────────────────
    prompt = _build_prompt(query, chunks)

    if verbose:
        print("── Prompt sent to LLM ───────────────────────────────────────────")
        print(prompt)
        print("─" * 70)

    # ── Call Ollama ───────────────────────────────────────────────────────────
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )

    answer     = response["message"]["content"]
    citations  = _format_citations(chunks)

    return {
        "answer":      answer,
        "citations":   citations,
        "query":       query,
        "chunks_used": min(len(chunks), MAX_CHUNKS),
        "model":       model,
    }


def format_response(result: dict) -> str:
    """Pretty-print the full RAG response for terminal output."""
    lines = [
        f"Query : {result['query']}",
        f"Model : {result['model']} | Chunks used: {result['chunks_used']}",
        "",
        "── Answer ───────────────────────────────────────────────────────────",
        result["answer"],
    ]

    if result["citations"]:
        lines += [
            "",
            "── Sources ──────────────────────────────────────────────────────────",
            result["citations"],
        ]

    return "\n".join(lines)


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from pipeline.embedder import load_index, EMBEDDING_MODEL
    from pipeline.retriever import retrieve
    from sentence_transformers import SentenceTransformer

    print("Loading pipeline...")
    index, chunks = load_index()
    model_embed   = SentenceTransformer(EMBEDDING_MODEL)

    test_queries = [
        "What are the symptoms of ADHD?",
        "How do stimulant medications work for ADHD?",
        "What sleep strategies help with ADHD?",
    ]

    for query in test_queries:
        print(f"\n{'=' * 70}")
        retrieved = retrieve(query, index, chunks, model_embed, top_k=5)
        result    = generate(query, retrieved)
        print(format_response(result))

    print(f"\n{'=' * 70}")
    print("Smoke test complete.")