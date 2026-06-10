"""
test_chunker.py
---------------
Unit tests for pipeline/chunker.py

Tests cover:
  - frontmatter parsing
  - category and topic inference
  - section splitting by ## headings
  - minimum chunk size filtering
  - duplicate handling
  - full KB load produces expected output
"""

import os
import sys
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from pipeline.chunker import (
    _parse_frontmatter,
    _infer_category_from_path,
    _infer_topic_from_filename,
    chunk_file,
    load_all_chunks,
    MIN_CHUNK_CHARS,
)

KB_ROOT = os.path.join(BASE_DIR, "adhd_kb")


# ── Frontmatter parsing ───────────────────────────────────────────────────────

class TestParseFrontmatter:

    def test_parses_valid_frontmatter(self):
        text = "---\ntopic: inattention\ncategory: symptoms\n---\n## Definition\nSome text."
        meta, body = _parse_frontmatter(text)
        assert meta["topic"] == "inattention"
        assert meta["category"] == "symptoms"
        assert "## Definition" in body

    def test_returns_empty_dict_when_no_frontmatter(self):
        text = "## Definition\nSome text without frontmatter."
        meta, body = _parse_frontmatter(text)
        assert meta == {}
        assert body == text

    def test_body_does_not_contain_frontmatter(self):
        text = "---\ntopic: test\n---\n## Section\nContent here."
        meta, body = _parse_frontmatter(text)
        assert "topic: test" not in body

    def test_handles_missing_closing_delimiter(self):
        text = "---\ntopic: broken\n## Section\nContent."
        meta, body = _parse_frontmatter(text)
        assert meta == {}
        assert body == text


# ── Category and topic inference ──────────────────────────────────────────────

class TestInferCategory:

    def test_strips_numeric_prefix(self):
        result = _infer_category_from_path("02_symptoms/inattention.md")
        assert result == "symptoms"

    def test_returns_lowercase(self):
        result = _infer_category_from_path("09_FAQ/adult_faq.md")
        assert result == result.lower()

    def test_returns_general_for_root_file(self):
        result = _infer_category_from_path("source_mapping.md")
        assert result == "general"

    def test_handles_children_teen_folder(self):
        result = _infer_category_from_path("07_children_teen/adhd_in_children.md")
        assert result == "children_teen"


class TestInferTopic:

    def test_replaces_underscores_with_spaces(self):
        result = _infer_topic_from_filename("adhd_diagnosis_overview.md")
        assert "_" not in result

    def test_removes_md_extension(self):
        result = _infer_topic_from_filename("inattention.md")
        assert ".md" not in result

    def test_returns_correct_topic(self):
        result = _infer_topic_from_filename("stimulant_medications.md")
        assert result == "stimulant medications"


# ── Chunk file ────────────────────────────────────────────────────────────────

class TestChunkFile:

    def test_returns_list(self):
        filepath = os.path.join(KB_ROOT, "01_overview", "what_is_adhd.md")
        if not os.path.exists(filepath):
            pytest.skip("KB file not available")
        chunks = chunk_file(filepath, KB_ROOT)
        assert isinstance(chunks, list)

    def test_chunks_are_non_empty(self):
        filepath = os.path.join(KB_ROOT, "01_overview", "what_is_adhd.md")
        if not os.path.exists(filepath):
            pytest.skip("KB file not available")
        chunks = chunk_file(filepath, KB_ROOT)
        assert len(chunks) > 0

    def test_chunk_has_required_fields(self):
        filepath = os.path.join(KB_ROOT, "01_overview", "what_is_adhd.md")
        if not os.path.exists(filepath):
            pytest.skip("KB file not available")
        chunks = chunk_file(filepath, KB_ROOT)
        required = {"chunk_id", "text", "source_file", "category", "topic", "heading", "char_count"}
        for chunk in chunks:
            assert required.issubset(chunk.keys()), f"Missing fields in chunk: {chunk['chunk_id']}"

    def test_all_chunks_meet_minimum_size(self):
        filepath = os.path.join(KB_ROOT, "01_overview", "what_is_adhd.md")
        if not os.path.exists(filepath):
            pytest.skip("KB file not available")
        chunks = chunk_file(filepath, KB_ROOT)
        for chunk in chunks:
            assert chunk["char_count"] >= MIN_CHUNK_CHARS, (
                f"Chunk too small: {chunk['chunk_id']} ({chunk['char_count']} chars)"
            )

    def test_chunk_ids_are_unique_within_file(self):
        filepath = os.path.join(KB_ROOT, "01_overview", "what_is_adhd.md")
        if not os.path.exists(filepath):
            pytest.skip("KB file not available")
        chunks = chunk_file(filepath, KB_ROOT)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found in same file"

    def test_chunk_text_contains_heading(self):
        filepath = os.path.join(KB_ROOT, "01_overview", "what_is_adhd.md")
        if not os.path.exists(filepath):
            pytest.skip("KB file not available")
        chunks = chunk_file(filepath, KB_ROOT)
        for chunk in chunks:
            if chunk["heading"] != "overview":
                assert chunk["heading"] in chunk["text"], (
                    f"Heading not found in text for chunk: {chunk['chunk_id']}"
                )

    def test_source_file_is_relative_path(self):
        filepath = os.path.join(KB_ROOT, "01_overview", "what_is_adhd.md")
        if not os.path.exists(filepath):
            pytest.skip("KB file not available")
        chunks = chunk_file(filepath, KB_ROOT)
        for chunk in chunks:
            assert not os.path.isabs(chunk["source_file"]), (
                f"source_file should be relative, got: {chunk['source_file']}"
            )

    def test_category_is_lowercase(self):
        filepath = os.path.join(KB_ROOT, "01_overview", "what_is_adhd.md")
        if not os.path.exists(filepath):
            pytest.skip("KB file not available")
        chunks = chunk_file(filepath, KB_ROOT)
        for chunk in chunks:
            assert chunk["category"] == chunk["category"].lower(), (
                f"Category not lowercase: {chunk['category']}"
            )


# ── Load all chunks ───────────────────────────────────────────────────────────

class TestLoadAllChunks:

    @pytest.fixture(scope="class")
    def all_chunks(self):
        if not os.path.exists(KB_ROOT):
            pytest.skip("adhd_kb not available")
        return load_all_chunks(KB_ROOT)

    def test_returns_expected_count(self, all_chunks):
        assert len(all_chunks) >= 300, f"Expected 300+ chunks, got {len(all_chunks)}"

    def test_no_reference_folder_chunks(self, all_chunks):
        for chunk in all_chunks:
            assert "10_reference" not in chunk["source_file"], (
                f"Reference folder chunk found: {chunk['source_file']}"
            )

    def test_no_source_mapping_chunks(self, all_chunks):
        for chunk in all_chunks:
            assert "source_mapping" not in chunk["source_file"], (
                f"source_mapping.md chunk found: {chunk['source_file']}"
            )

    def test_all_categories_present(self, all_chunks):
        categories = {c["category"] for c in all_chunks}
        expected = {"overview", "symptoms", "diagnosis", "treatment",
                    "medications", "adults", "children_teen", "safety", "faq"}
        missing = expected - categories
        assert not missing, f"Missing categories: {missing}"

    def test_no_empty_texts(self, all_chunks):
        for chunk in all_chunks:
            assert chunk["text"].strip(), f"Empty text in chunk: {chunk['chunk_id']}"

    def test_no_duplicate_chunk_ids(self, all_chunks):
        ids = [c["chunk_id"] for c in all_chunks]
        duplicates = [id for id in set(ids) if ids.count(id) > 1]
        assert not duplicates, f"Duplicate chunk IDs: {duplicates}"

    def test_char_count_matches_text_length(self, all_chunks):
        for chunk in all_chunks:
            assert chunk["char_count"] == len(chunk["text"]), (
                f"char_count mismatch in {chunk['chunk_id']}"
            )