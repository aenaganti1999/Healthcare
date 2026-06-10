"""
retriever.py
------------
Given a query string, finds the most relevant chunks
from the FAISS index using cosine similarity.

This is the core of the RAG pipeline — the R in RAG.
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer

# import from sibling module
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from pipeline.embedder import load_index, EMBEDDING_MODEL

# ── How many chunks to return per query ──────────────────────────────────────
DEFAULT_TOP_K = 5


def load_retriever() -> tuple:
    """
    Load the FAISS index, chunk metadata, and embedding model.
    Call once at startup — reuse the returned objects for all queries.

    Returns:
        index   : faiss.Index
        chunks  : list[dict]
        model   : SentenceTransformer
    """
    print("Loading retriever...")
    index, chunks = load_index()
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"Retriever ready. Index contains {index.ntotal} vectors.\n")
    return index, chunks, model


def retrieve(
    query: str,
    index,
    chunks: list[dict],
    model: SentenceTransformer,
    top_k: int = DEFAULT_TOP_K,
    category_filter: str = None,
) -> list[dict]:
    """
    Retrieve the top_k most relevant chunks for a query.

    Args:
        query           : the user's question
        index           : loaded FAISS index
        chunks          : loaded chunk metadata list
        model           : loaded SentenceTransformer
        top_k           : number of chunks to return
        category_filter : optional — only return chunks from this category
                          e.g. 'safety', 'medications', 'symptoms'

    Returns:
        list of chunk dicts, each with an added 'score' field (cosine similarity)
        sorted by score descending
    """
    # ── Embed the query ───────────────────────────────────────────────────────
    query_vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    # ── Search FAISS ──────────────────────────────────────────────────────────
    # Search more than top_k if filtering, so we have enough after filter
    search_k = top_k * 4 if category_filter else top_k

    scores, indices = index.search(query_vector, search_k)

    # ── Build results ─────────────────────────────────────────────────────────
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:           # FAISS returns -1 for empty slots
            continue

        chunk = chunks[idx].copy()
        chunk["score"] = round(float(score), 4)

        # Apply category filter if requested
        if category_filter:
            if chunk["category"] != category_filter.lower():
                continue

        results.append(chunk)

        if len(results) >= top_k:
            break

    return results


def format_results(results: list[dict], show_text: bool = False) -> str:
    """Pretty-print retrieval results for inspection."""
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] score={r['score']:.4f} | {r['source_file']} | {r['heading']}")
        lines.append(f"    category={r['category']} | topic={r['topic']}")
        if show_text:
            preview = r["text"][:200].replace("\n", " ")
            lines.append(f"    text: {preview}...")
        lines.append("")
    return "\n".join(lines)


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    index, chunks, model = load_retriever()

    # Test queries — covering different categories
    test_queries = [
        ("What are the symptoms of inattention in ADHD?",       None),
        ("How do stimulant medications work for ADHD?",         None),
        ("What is emotional dysregulation?",                    None),
        ("How can adults with ADHD manage time?",               None),
        ("What school accommodations help children with ADHD?", None),
        ("What are side effects of ADHD medication?",           "medications"),  # filtered
    ]

    for query, category_filter in test_queries:
        print(f"Query : {query}")
        if category_filter:
            print(f"Filter: category={category_filter}")

        results = retrieve(
            query,
            index,
            chunks,
            model,
            top_k=3,
            category_filter=category_filter,
        )

        print(format_results(results, show_text=True))
        print("─" * 70)
        print()