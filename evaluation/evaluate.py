"""
evaluate.py
-----------
Measures retrieval quality against the golden test set.

Metrics calculated per query and overall:
  Precision@k  =  relevant chunks retrieved / k chunks retrieved
  Recall@k     =  relevant chunks retrieved / total relevant chunks that exist
  Hit@k        =  1 if at least 1 relevant chunk in top-k, else 0

Usage:
  python evaluation/evaluate.py
  python evaluation/evaluate.py --k 3
  python evaluation/evaluate.py --k 5 --verbose
"""

import os
import sys
import csv
import json
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from pipeline.embedder import load_index, EMBEDDING_MODEL
from pipeline.retriever import retrieve
from sentence_transformers import SentenceTransformer

GOLDEN_SET_PATH = os.path.join(BASE_DIR, "evaluation", "golden_set.csv")


def load_golden_set(path: str) -> list[dict]:
    """
    Load the golden test set from CSV.
    Returns list of {query, relevant_ids} dicts.
    """
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            relevant_ids = [
                cid.strip()
                for cid in row["relevant_chunk_ids"].split(",")
                if cid.strip()
            ]
            rows.append({
                "query":        row["query"].strip('"'),
                "relevant_ids": relevant_ids,
            })
    return rows


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Of the top-k retrieved, what fraction are relevant?"""
    top_k = retrieved_ids[:k]
    hits  = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Of all relevant chunks, what fraction did we retrieve in top-k?"""
    top_k = retrieved_ids[:k]
    hits  = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(relevant_ids) if relevant_ids else 0.0


def hit_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> int:
    """Did we retrieve at least 1 relevant chunk in top-k?"""
    top_k = retrieved_ids[:k]
    return int(any(rid in relevant_ids for rid in top_k))


def run_evaluation(k: int = 5, verbose: bool = False) -> dict:
    """
    Run full evaluation against the golden test set.
    Returns summary metrics dict.
    """
    # ── Load pipeline ─────────────────────────────────────────────────────────
    print("Loading retrieval pipeline...")
    index, chunks, model = load_index()[0], load_index()[1], SentenceTransformer(EMBEDDING_MODEL)
    print(f"Index loaded: {index.ntotal} vectors")

    # ── Load golden set ───────────────────────────────────────────────────────
    golden = load_golden_set(GOLDEN_SET_PATH)
    print(f"Golden set loaded: {len(golden)} queries")
    print(f"Evaluating at k={k}\n")
    print("=" * 70)

    # ── Per-query results ─────────────────────────────────────────────────────
    all_precision = []
    all_recall    = []
    all_hits      = []
    per_query     = []

    for row in golden:
        query        = row["query"]
        relevant_ids = set(row["relevant_ids"])

        # retrieve top-k chunks
        results      = retrieve(query, index, chunks, model, top_k=k)
        retrieved_ids = [r["chunk_id"] for r in results]

        p = precision_at_k(retrieved_ids, relevant_ids, k)
        r = recall_at_k(retrieved_ids, relevant_ids, k)
        h = hit_at_k(retrieved_ids, relevant_ids, k)

        all_precision.append(p)
        all_recall.append(r)
        all_hits.append(h)

        per_query.append({
            "query":        query,
            "precision":    round(p, 3),
            "recall":       round(r, 3),
            "hit":          h,
            "retrieved":    retrieved_ids,
            "relevant":     list(relevant_ids),
        })

        if verbose:
            status = "✓" if h else "✗"
            print(f"{status} P={p:.2f} R={r:.2f} | {query[:60]}")
            # show which relevant chunks were found vs missed
            for rid in relevant_ids:
                found = rid in retrieved_ids
                mark  = "  FOUND  " if found else "  MISSED "
                print(f"    {mark} {rid}")
            print()

    # ── Summary ───────────────────────────────────────────────────────────────
    avg_precision = sum(all_precision) / len(all_precision)
    avg_recall    = sum(all_recall)    / len(all_recall)
    hit_rate      = sum(all_hits)      / len(all_hits)

    print("=" * 70)
    print(f"\n── Evaluation Results @ k={k} ───────────────────────────────────────")
    print(f"  Queries evaluated  : {len(golden)}")
    print(f"  Mean Precision@{k}  : {avg_precision:.3f}  ({avg_precision*100:.1f}%)")
    print(f"  Mean Recall@{k}     : {avg_recall:.3f}  ({avg_recall*100:.1f}%)")
    print(f"  Hit Rate@{k}        : {hit_rate:.3f}  ({hit_rate*100:.1f}%)")

    # ── Per-category breakdown ────────────────────────────────────────────────
    print(f"\n── Per-query breakdown ──────────────────────────────────────────────")
    print(f"  {'Query':<52} {'P':>5} {'R':>5} {'Hit':>4}")
    print(f"  {'-'*52} {'-'*5} {'-'*5} {'-'*4}")
    for pq in per_query:
        q_short = pq["query"][:50] + ".." if len(pq["query"]) > 50 else pq["query"]
        hit_str = " ✓" if pq["hit"] else " ✗"
        print(f"  {q_short:<52} {pq['precision']:>5.2f} {pq['recall']:>5.2f} {hit_str:>4}")

    # ── Failures ─────────────────────────────────────────────────────────────
    failures = [pq for pq in per_query if pq["hit"] == 0]
    if failures:
        print(f"\n── Queries with zero relevant chunks retrieved ({len(failures)}) ──────────")
        for f in failures:
            print(f"  ✗ {f['query']}")
            print(f"    Expected : {f['relevant']}")
            print(f"    Got      : {f['retrieved']}")

    print()

    return {
        "k":             k,
        "n_queries":     len(golden),
        "mean_precision": round(avg_precision, 3),
        "mean_recall":    round(avg_recall, 3),
        "hit_rate":       round(hit_rate, 3),
        "per_query":      per_query,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality")
    parser.add_argument("--k",       type=int,  default=5,     help="top-k chunks to retrieve")
    parser.add_argument("--verbose", action="store_true",      help="show per-chunk found/missed details")
    args = parser.parse_args()

    results = run_evaluation(k=args.k, verbose=args.verbose)