"""
main.py
-------
Entry point for the ADHD RAG pipeline.

Commands:
  python main.py build              - chunk, embed, and save index
  python main.py retrieve "query"   - retrieve top-k chunks for a query
  python main.py evaluate           - run precision/recall evaluation
  python main.py evaluate --k 3     - evaluate at k=3
  python main.py chat               - start interactive multi-turn chat
  python main.py status             - show index and data status
"""

import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


def cmd_build(args):
    """Chunk all .md files, embed, and save FAISS index to disk."""
    from pipeline.embedder import build_and_save
    print("── Building RAG index ───────────────────────────────────────────────")
    build_and_save()


def cmd_retrieve(args):
    """Retrieve top-k chunks for a query and print results."""
    from pipeline.embedder import load_index, EMBEDDING_MODEL
    from pipeline.retriever import retrieve, format_results
    from sentence_transformers import SentenceTransformer

    query = args.query
    k     = args.k

    print(f"── Retrieving top-{k} chunks ─────────────────────────────────────────")
    print(f"Query: {query}\n")

    index, chunks = load_index()
    model         = SentenceTransformer(EMBEDDING_MODEL)
    results       = retrieve(query, index, chunks, model, top_k=k)

    print(format_results(results, show_text=True))


def cmd_evaluate(args):
    """Run precision/recall evaluation against the golden test set."""
    from evaluation.evaluate import run_evaluation
    print("── Running evaluation ───────────────────────────────────────────────")
    run_evaluation(k=args.k, verbose=args.verbose)


def cmd_chat(args):
    """Start interactive multi-turn conversational chat."""
    from pipeline.conversation import ConversationSession
    session = ConversationSession()
    session.run()


def cmd_status(args):
    """Show current status of the index and data files."""
    import json

    data_dir    = os.path.join(BASE_DIR, "data")
    index_path  = os.path.join(data_dir, "faiss_index.bin")
    chunks_path = os.path.join(data_dir, "chunks.json")
    golden_path = os.path.join(BASE_DIR, "evaluation", "golden_set.csv")
    kb_path     = os.path.join(BASE_DIR, "adhd_kb")

    print("── RAG Pipeline Status ──────────────────────────────────────────────")

    if os.path.exists(kb_path):
        md_files = list(__import__('pathlib').Path(kb_path).rglob("*.md"))
        print(f"  Knowledge base : {len(md_files)} .md files found")
    else:
        print(f"  Knowledge base : NOT FOUND at {kb_path}")

    if os.path.exists(index_path):
        size_kb = os.path.getsize(index_path) // 1024
        print(f"  FAISS index    : exists ({size_kb} KB)")
    else:
        print(f"  FAISS index    : NOT BUILT — run: python main.py build")

    if os.path.exists(chunks_path):
        with open(chunks_path) as f:
            chunks = json.load(f)
        from collections import Counter
        cats = Counter(c["category"] for c in chunks)
        print(f"  Chunks stored  : {len(chunks)} total")
        for cat, count in sorted(cats.items()):
            print(f"    {cat:<25} {count}")
    else:
        print(f"  Chunks file    : NOT FOUND — run: python main.py build")

    if os.path.exists(golden_path):
        with open(golden_path) as f:
            lines = f.readlines()
        print(f"  Golden set     : {len(lines) - 1} queries labeled")
    else:
        print(f"  Golden set     : NOT FOUND at {golden_path}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="ADHD RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py build
  python main.py retrieve "What are symptoms of ADHD?"
  python main.py retrieve "How do stimulants work?" --k 3
  python main.py evaluate
  python main.py evaluate --k 3 --verbose
  python main.py chat
  python main.py status
        """
    )

    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    # build
    subparsers.add_parser("build", help="Chunk, embed, and save index to disk")

    # retrieve
    p_retrieve = subparsers.add_parser("retrieve", help="Retrieve chunks for a query")
    p_retrieve.add_argument("query", type=str, help="Query string in quotes")
    p_retrieve.add_argument("--k", type=int, default=5, help="Number of chunks to retrieve")

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Run precision/recall evaluation")
    p_eval.add_argument("--k",       type=int,  default=5,  help="top-k to evaluate at")
    p_eval.add_argument("--verbose", action="store_true",   help="show per-chunk details")

    # chat — no arguments needed, interactive loop handles everything
    subparsers.add_parser("chat", help="Start interactive multi-turn chat")

    # status
    subparsers.add_parser("status", help="Show pipeline status")

    args = parser.parse_args()

    commands = {
        "build":    cmd_build,
        "retrieve": cmd_retrieve,
        "evaluate": cmd_evaluate,
        "chat":     cmd_chat,
        "status":   cmd_status,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()