"""
test_retriever.py
-----------------
Integration tests for pipeline/retriever.py

Tests cover:
  - index loads correctly from disk
  - retrieve returns correct number of results
  - scores are valid cosine similarity values
  - results are sorted by score descending
  - category filter works correctly
  - known high-confidence queries return expected source files
  - retrieve handles edge cases gracefully
"""

import os
import sys
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from pipeline.embedder import load_index, EMBEDDING_MODEL
from pipeline.retriever import retrieve
from sentence_transformers import SentenceTransformer


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pipeline():
    """Load index, chunks, and model once for all tests in this module."""
    data_dir   = os.path.join(BASE_DIR, "data")
    index_path = os.path.join(data_dir, "faiss_index.bin")
    if not os.path.exists(index_path):
        pytest.skip("FAISS index not built — run: python main.py build")

    index, chunks = load_index()
    model         = SentenceTransformer(EMBEDDING_MODEL)
    return index, chunks, model


# ── Index loading ─────────────────────────────────────────────────────────────

class TestIndexLoading:

    def test_index_loads_without_error(self, pipeline):
        index, chunks, model = pipeline
        assert index is not None

    def test_index_contains_expected_vector_count(self, pipeline):
        index, chunks, model = pipeline
        assert index.ntotal >= 300, f"Expected 300+ vectors, got {index.ntotal}"

    def test_chunks_count_matches_index(self, pipeline):
        index, chunks, model = pipeline
        assert len(chunks) == index.ntotal, (
            f"Chunk count {len(chunks)} doesn't match index size {index.ntotal}"
        )

    def test_every_chunk_has_chunk_id(self, pipeline):
        index, chunks, model = pipeline
        for chunk in chunks:
            assert "chunk_id" in chunk, "chunk missing chunk_id field"
            assert chunk["chunk_id"], "chunk_id is empty"


# ── Retrieve basic behavior ───────────────────────────────────────────────────

class TestRetrieveBasic:

    def test_returns_list(self, pipeline):
        index, chunks, model = pipeline
        results = retrieve("What is ADHD?", index, chunks, model, top_k=5)
        assert isinstance(results, list)

    def test_returns_correct_count(self, pipeline):
        index, chunks, model = pipeline
        for k in [1, 3, 5]:
            results = retrieve("What is ADHD?", index, chunks, model, top_k=k)
            assert len(results) == k, f"Expected {k} results, got {len(results)}"

    def test_each_result_has_score(self, pipeline):
        index, chunks, model = pipeline
        results = retrieve("What is ADHD?", index, chunks, model, top_k=3)
        for r in results:
            assert "score" in r, "Result missing score field"

    def test_scores_are_valid_cosine_similarity(self, pipeline):
        index, chunks, model = pipeline
        results = retrieve("What is ADHD?", index, chunks, model, top_k=5)
        for r in results:
            assert -1.0 <= r["score"] <= 1.0, (
                f"Score out of valid range: {r['score']}"
            )

    def test_results_sorted_by_score_descending(self, pipeline):
        index, chunks, model = pipeline
        results = retrieve("What is ADHD?", index, chunks, model, top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), (
            "Results not sorted by score descending"
        )

    def test_each_result_has_required_fields(self, pipeline):
        index, chunks, model = pipeline
        results = retrieve("What is ADHD?", index, chunks, model, top_k=3)
        required = {"chunk_id", "text", "source_file", "category", "topic", "heading", "score"}
        for r in results:
            assert required.issubset(r.keys()), f"Missing fields in result: {r['chunk_id']}"


# ── Category filter ───────────────────────────────────────────────────────────

class TestCategoryFilter:

    def test_filter_returns_only_matching_category(self, pipeline):
        index, chunks, model = pipeline
        results = retrieve(
            "What are ADHD medications?",
            index, chunks, model,
            top_k=5,
            category_filter="medications"
        )
        for r in results:
            assert r["category"] == "medications", (
                f"Category filter failed: got {r['category']}"
            )

    def test_filter_returns_only_symptoms_category(self, pipeline):
        index, chunks, model = pipeline
        results = retrieve(
            "What are ADHD symptoms?",
            index, chunks, model,
            top_k=3,
            category_filter="symptoms"
        )
        for r in results:
            assert r["category"] == "symptoms"

    def test_filter_is_case_insensitive(self, pipeline):
        index, chunks, model = pipeline
        results_lower = retrieve(
            "ADHD medications",
            index, chunks, model,
            top_k=3,
            category_filter="medications"
        )
        results_upper = retrieve(
            "ADHD medications",
            index, chunks, model,
            top_k=3,
            category_filter="MEDICATIONS"
        )
        ids_lower = [r["chunk_id"] for r in results_lower]
        ids_upper = [r["chunk_id"] for r in results_upper]
        assert ids_lower == ids_upper, "Filter should be case-insensitive"

    def test_nonexistent_category_returns_empty(self, pipeline):
        index, chunks, model = pipeline
        results = retrieve(
            "What is ADHD?",
            index, chunks, model,
            top_k=5,
            category_filter="nonexistent_category_xyz"
        )
        assert results == [], (
            f"Expected empty list for nonexistent category, got {len(results)} results"
        )


# ── Known query relevance ─────────────────────────────────────────────────────

class TestKnownQueryRelevance:
    """
    High-confidence tests: these queries should always return
    chunks from the correct source files at top-k=5.
    If these fail, retrieval quality has regressed.
    """

    def test_stimulant_query_hits_medications_file(self, pipeline):
        index, chunks, model = pipeline
        results = retrieve(
            "How do stimulant medications work for ADHD?",
            index, chunks, model,
            top_k=5
        )
        source_files = [r["source_file"] for r in results]
        assert any("stimulant" in f for f in source_files), (
            f"Expected stimulant file in results, got: {source_files}"
        )

    def test_sleep_query_hits_sleep_file(self, pipeline):
        index, chunks, model = pipeline
        results = retrieve(
            "How does sleep affect ADHD symptoms?",
            index, chunks, model,
            top_k=5
        )
        source_files = [r["source_file"] for r in results]
        assert any("sleep" in f for f in source_files), (
            f"Expected sleep file in results, got: {source_files}"
        )

    def test_diagnosis_query_hits_diagnosis_file(self, pipeline):
        index, chunks, model = pipeline
        results = retrieve(
            "How is ADHD diagnosed?",
            index, chunks, model,
            top_k=5
        )
        source_files = [r["source_file"] for r in results]
        assert any("diagnosis" in f for f in source_files), (
            f"Expected diagnosis file in results, got: {source_files}"
        )

    def test_top_result_score_above_threshold(self, pipeline):
        index, chunks, model = pipeline
        results = retrieve(
            "What are symptoms of inattention in ADHD?",
            index, chunks, model,
            top_k=1
        )
        assert results[0]["score"] >= 0.5, (
            f"Top result score too low: {results[0]['score']:.3f} — retrieval may have degraded"
        )


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_very_short_query(self, pipeline):
        index, chunks, model = pipeline
        results = retrieve("ADHD", index, chunks, model, top_k=3)
        assert len(results) == 3

    def test_long_query(self, pipeline):
        index, chunks, model = pipeline
        long_query = (
            "What are the most effective evidence-based treatment approaches "
            "for adults with ADHD who have been recently diagnosed and are "
            "struggling with time management and emotional regulation at work?"
        )
        results = retrieve(long_query, index, chunks, model, top_k=5)
        assert len(results) == 5

    def test_unrelated_query_still_returns_results(self, pipeline):
        index, chunks, model = pipeline
        results = retrieve(
            "What is the weather like today?",
            index, chunks, model,
            top_k=3
        )
        assert len(results) == 3

    def test_k_1_returns_single_best_result(self, pipeline):
        index, chunks, model = pipeline
        results = retrieve("ADHD diagnosis", index, chunks, model, top_k=1)
        assert len(results) == 1