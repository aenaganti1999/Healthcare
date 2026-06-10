"""
chunker.py
----------
Reads all .md files from the knowledge base directory.
Splits each file into sections by ## headings.
Attaches metadata to every chunk.
Returns a list of dicts ready for embedding.
"""

import os
import re
from pathlib import Path


# ── Folders to skip entirely ──────────────────────────────────────────────────
SKIP_FOLDERS = {"10_reference", "metadata"}

# ── Files to skip entirely ────────────────────────────────────────────────────
SKIP_FILES = {"source_mapping.md"}

# ── Headings to skip entirely (no clinical value) ─────────────────────────────
SKIP_HEADINGS = {"References", "references"}

# ── Minimum characters a chunk must have to be kept ──────────────────────────
MIN_CHUNK_CHARS = 200

# ── Maximum characters before a chunk gets split ─────────────────────────────
MAX_CHUNK_CHARS = 1000


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """
    If the file starts with YAML frontmatter (--- ... ---), parse it.
    Returns (metadata_dict, remaining_text).
    If no frontmatter, returns ({}, original_text).
    """
    metadata = {}
    if not text.startswith("---"):
        return metadata, text

    end = text.find("---", 3)
    if end == -1:
        return metadata, text

    frontmatter = text[3:end].strip()
    body = text[end + 3:].strip()

    for line in frontmatter.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            metadata[key.strip()] = value.strip()

    return metadata, body


def _infer_category_from_path(filepath: str) -> str:
    """
    Infer category from the folder name.
    e.g. '02_symptoms/inattention.md' → 'symptoms'
    """
    parts = Path(filepath).parts
    if len(parts) >= 2:
        folder = parts[-2]                     # e.g. '02_symptoms'
        # strip leading digits and underscore
        category = re.sub(r"^\d+_", "", folder).lower()
        return category
    return "general"


def _infer_topic_from_filename(filepath: str) -> str:
    """
    Infer topic from filename without extension.
    e.g. 'inattention.md' → 'inattention'
    """
    return Path(filepath).stem.replace("_", " ")


def _split_large_section(text: str, heading: str, max_chars: int) -> list[str]:
    """
    Split a section that exceeds max_chars into smaller sub-chunks.
    Splits at paragraph boundaries (blank lines) to preserve meaning.
    Each sub-chunk keeps the heading for context.

    Returns a list of text strings, each under max_chars.
    """
    if len(text) <= max_chars:
        return [text]

    # Split into paragraphs by blank lines
    paragraphs = re.split(r"\n\n+", text)

    sub_chunks = []
    current    = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If adding this paragraph exceeds limit, save current and start new
        if current and len(current) + len(para) + 2 > max_chars:
            sub_chunks.append(current.strip())
            # Start new sub-chunk with heading for context
            current = f"## {heading}\n\n{para}"
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        sub_chunks.append(current.strip())

    return sub_chunks if sub_chunks else [text]


def chunk_file(filepath: str, kb_root: str) -> list[dict]:
    """
    Split a single .md file into section-level chunks.

    Each chunk is a dict:
    {
        chunk_id    : unique string id
        text        : heading + section body (what gets embedded)
        source_file : relative path from kb_root
        category    : inferred from folder name
        topic       : inferred from filename
        heading     : the ## heading text
        char_count  : length of text
    }
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    frontmatter, body = _parse_frontmatter(raw)

    # relative path for display e.g. '02_symptoms/inattention.md'
    rel_path = os.path.relpath(filepath, kb_root)

    category = frontmatter.get("category") or _infer_category_from_path(rel_path)
    topic    = frontmatter.get("topic")    or _infer_topic_from_filename(filepath)

    # ── Split by ## headings ──────────────────────────────────────────────────
    sections = re.split(r"\n(?=## )", body)

    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Extract heading from first line
        lines        = section.splitlines()
        heading_line = lines[0].strip()

        if heading_line.startswith("##"):
            heading = heading_line.lstrip("#").strip()
        else:
            heading = "overview"

        # Skip reference sections — no clinical value
        if heading in SKIP_HEADINGS:
            continue

        # Skip if too short
        if len(section) < MIN_CHUNK_CHARS:
            continue

        # ── Split large sections at paragraph boundaries ──────────────────────
        sub_texts = _split_large_section(section, heading, MAX_CHUNK_CHARS)

        safe_rel  = rel_path.replace("/", "_").replace(".md", "")
        safe_head = re.sub(r"[^a-z0-9]+", "_", heading.lower()).strip("_")

        for i, sub_text in enumerate(sub_texts):
            if len(sub_text) < MIN_CHUNK_CHARS:
                continue

            # Add part suffix only when section was split into multiple parts
            if len(sub_texts) > 1:
                chunk_id = f"{safe_rel}__{safe_head}__part{i+1}"
            else:
                chunk_id = f"{safe_rel}__{safe_head}"

            chunks.append({
                "chunk_id":    chunk_id,
                "text":        sub_text,
                "source_file": rel_path,
                "category":    category,
                "topic":       topic,
                "heading":     heading,
                "char_count":  len(sub_text),
            })

    return chunks


def load_all_chunks(kb_root: str) -> list[dict]:
    """
    Walk the entire knowledge base directory.
    Skip reference folders and files defined in SKIP_FOLDERS / SKIP_FILES.
    Return a flat list of all chunks from all files.
    """
    kb_path    = Path(kb_root)
    all_chunks = []

    md_files = sorted(kb_path.rglob("*.md"))

    for filepath in md_files:
        # Skip unwanted folders
        parts = filepath.parts
        if any(skip in parts for skip in SKIP_FOLDERS):
            continue

        # Skip unwanted files
        if filepath.name in SKIP_FILES:
            continue

        file_chunks = chunk_file(str(filepath), kb_root)
        all_chunks.extend(file_chunks)

    return all_chunks


if __name__ == "__main__":
    import json

    KB_ROOT = os.path.join(os.path.dirname(__file__), "..", "adhd_kb")
    KB_ROOT = os.path.abspath(KB_ROOT)

    print(f"Loading chunks from: {KB_ROOT}\n")
    chunks = load_all_chunks(KB_ROOT)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"Total chunks produced : {len(chunks)}")
    print(f"Min chunk size        : {min(c['char_count'] for c in chunks)} chars")
    print(f"Max chunk size        : {max(c['char_count'] for c in chunks)} chars")
    print(f"Avg chunk size        : {int(sum(c['char_count'] for c in chunks) / len(chunks))} chars")

    # ── Size distribution ─────────────────────────────────────────────────────
    sizes  = sorted(c["char_count"] for c in chunks)
    ranges = [(0,200),(200,400),(400,600),(600,800),(800,1000),(1000,1500),(1500,9999)]
    print("\nSize distribution:")
    for lo, hi in ranges:
        count = sum(1 for s in sizes if lo <= s < hi)
        bar   = "█" * (count // 2)
        print(f"  {lo:>5}-{hi:<5} chars : {count:>3} chunks  {bar}")

    # ── Category breakdown ────────────────────────────────────────────────────
    from collections import Counter
    cats = Counter(c["category"] for c in chunks)
    print("\nChunks per category:")
    for cat, count in sorted(cats.items()):
        print(f"  {cat:<25} {count}")

    # ── Sample: first 3 chunks ────────────────────────────────────────────────
    print("\n── First 3 chunks (preview) ──────────────────────────────────────")
    for chunk in chunks[:3]:
        print(json.dumps({
            "chunk_id":     chunk["chunk_id"],
            "source_file":  chunk["source_file"],
            "category":     chunk["category"],
            "heading":      chunk["heading"],
            "char_count":   chunk["char_count"],
            "text_preview": chunk["text"][:120] + "...",
        }, indent=2))
        print()