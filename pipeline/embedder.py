"""
embedder.py
-----------
Takes chunks from chunker.py.
Embeds each chunk using sentence-transformers (all-MiniLM-L6-v2).
Stores vectors in a FAISS index.
Saves index + chunk metadata to disk so we never re-embed unless docs change.

Output files:
  data/faiss_index.bin  ← FAISS vector index
  data/chunks.json      ← chunk metadata (parallel to index rows)
"""

import os
import json
import time
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_ROOT     = os.path.join(BASE_DIR, "adhd_kb")
DATA_DIR    = os.path.join(BASE_DIR, "data")
INDEX_PATH  = os.path.join(DATA_DIR, "faiss_index.bin")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.json")

# ── Model ─────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384   # fixed output size for this model


def load_model() -> SentenceTransformer:
    """Load the embedding model. Downloads once (~90MB), cached after."""
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Model loaded.\n")
    return model


def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> np.ndarray:
    """
    Embed the text of every chunk.
    Returns a float32 numpy array of shape (n_chunks, 384).
    """
    texts = [c["text"] for c in chunks]

    print(f"Embedding {len(texts)} chunks...")
    start = time.time()

    vectors = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    elapsed = time.time() - start
    print(f"Embedding complete in {elapsed:.1f}s")
    print(f"Vector matrix shape: {vectors.shape}\n")

    return vectors.astype(np.float32)


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    """
    Build a FAISS flat index using inner product.
    Because vectors are L2-normalized, inner product == cosine similarity.
    """
    n, dim = vectors.shape
    print(f"Building FAISS index: {n} vectors × {dim} dims")

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    print(f"FAISS index built. Total vectors: {index.ntotal}\n")
    return index


def save_index(index: faiss.Index, chunks: list[dict]) -> None:
    """Save FAISS index and chunk metadata to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)

    faiss.write_index(index, INDEX_PATH)
    print(f"FAISS index saved  → {INDEX_PATH}")

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Chunk metadata saved → {CHUNKS_PATH}")


def load_index() -> tuple[faiss.Index, list[dict]]:
    """
    Load FAISS index and chunk metadata from disk.
    Call this at query time instead of re-embedding.
    """
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"No index found at {INDEX_PATH}. Run embedder.py first."
        )
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            f"No chunks found at {CHUNKS_PATH}. Run embedder.py first."
        )

    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return index, chunks


def build_and_save(kb_root: str = KB_ROOT) -> None:
    """
    Full pipeline: chunk → embed → index → save.
    Run this once to build the index, then use load_index() at query time.
    """
    import sys
    sys.path.insert(0, BASE_DIR)
    from pipeline.chunker import load_all_chunks

    # ── Load chunks ───────────────────────────────────────────────────────────
    chunks = load_all_chunks(kb_root)
    print(f"Loaded {len(chunks)} chunks from {kb_root}")

    # ── Deduplicate by text content ───────────────────────────────────────────
    seen_texts = set()
    unique_chunks = []
    for chunk in chunks:
        normalized = " ".join(chunk["text"].split())
        if normalized not in seen_texts:
            seen_texts.add(normalized)
            unique_chunks.append(chunk)

    removed = len(chunks) - len(unique_chunks)
    print(f"Removed {removed} duplicate chunks → {len(unique_chunks)} unique chunks\n")
    chunks = unique_chunks

    # ── Embed, index, save ────────────────────────────────────────────────────
    model   = load_model()
    vectors = embed_chunks(chunks, model)
    index   = build_faiss_index(vectors)
    save_index(index, chunks)

    print("\n── Build complete ───────────────────────────────────────────────")
    print(f"  Chunks indexed : {index.ntotal}")
    print(f"  Index file     : {INDEX_PATH}")
    print(f"  Chunks file    : {CHUNKS_PATH}")


if __name__ == "__main__":
    build_and_save()
    build_and_save()