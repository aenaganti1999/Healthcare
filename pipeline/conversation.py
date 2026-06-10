"""
conversation.py
---------------
Multi-turn conversational interface for the ADHD RAG pipeline.

Maintains conversation history across turns so the LLM can:
  - Understand pronouns ("it", "that condition", "those medications")
  - Build on previous answers
  - Provide contextually aware follow-up responses

Usage:
  from pipeline.conversation import ConversationSession
  session = ConversationSession()
  session.run()
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import ollama
from pipeline.embedder import load_index, EMBEDDING_MODEL
from pipeline.retriever import retrieve
from pipeline.generator import OLLAMA_MODEL, MAX_CHUNKS, MAX_CHUNK_CHARS, SYSTEM_PROMPT, _format_citations
from sentence_transformers import SentenceTransformer

# ── Conversation config ───────────────────────────────────────────────────────
MAX_HISTORY_TURNS  = 6    # keep last 6 turns (3 user + 3 bot) in context
TOP_K_CHUNKS       = 5    # chunks to retrieve per turn
CONTEXT_WINDOW     = 3    # how many prior turns to include in retrieval context

# ── Exit commands ─────────────────────────────────────────────────────────────
EXIT_COMMANDS = {"quit", "exit", "bye", "q", "stop"}

def _build_retrieval_query(current_query: str, history: list[dict]) -> str:
    """
    Enrich current query with only the immediately previous user question.
    Keeps enrichment focused to avoid noisy retrieval.
    """
    if not history:
        return current_query

    user_msgs = [
        msg["content"].replace("USER QUESTION:", "").strip()
        for msg in history
        if msg["role"] == "user"
    ]

    if not user_msgs:
        return current_query

    last_question = user_msgs[-1][:100]
    enriched = f"{last_question} {current_query}"
    return enriched


def _build_messages(
    current_query: str,
    chunks: list[dict],
    history: list[dict],
) -> list[dict]:
    """
    Build the full message list to send to the LLM.

    Structure:
      [system prompt]
      [prior conversation turns]
      [current user message with retrieved context]
    """
    # Build context block from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(chunks[:MAX_CHUNKS], 1):
        text = chunk["text"]
        if len(text) > MAX_CHUNK_CHARS:
            text = text[:MAX_CHUNK_CHARS] + "..."
        context_parts.append(
            f"CONTEXT SECTION {i}: {chunk['heading']} "
            f"(source: {chunk['source_file']})\n{text}"
        )
    context_block = "\n\n".join(context_parts)

    # Current user message with context injected
    user_message = f"""Below are relevant sections from a verified ADHD knowledge base.
Use ONLY this information to answer the question.

{context_block}

USER QUESTION: {current_query}

Please provide a helpful, accurate answer based strictly on the context above."""

    # Full message list: system + history + current
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-MAX_HISTORY_TURNS:])
    messages.append({"role": "user", "content": user_message})

    return messages


def _print_separator():
    print("\n" + "─" * 60)


def _print_welcome():
    print("\n" + "=" * 60)
    print("  ADHD RAG Assistant — Multi-turn Chat")
    print("=" * 60)
    print("  Ask questions about ADHD.")
    print("  Type 'quit' to exit.")
    print("  Type 'clear' to start a new conversation.")
    print("  Type 'history' to see conversation so far.")
    print("=" * 60 + "\n")


class ConversationSession:
    """
    Manages a multi-turn conversation with the RAG pipeline.

    Maintains:
      - conversation history (user + assistant turns)
      - loaded index, chunks, and models (loaded once at startup)
    """

    def __init__(self):
        self.history: list[dict] = []
        self.turn_count: int     = 0
        self.index               = None
        self.chunks              = None
        self.embed_model         = None

    def _load_pipeline(self):
        """Load FAISS index, chunks, and embedding model once."""
        print("Loading pipeline...")
        self.index, self.chunks = load_index()
        self.embed_model        = SentenceTransformer(EMBEDDING_MODEL)
        print(f"Ready. {self.index.ntotal} chunks indexed.\n")

    def _handle_command(self, user_input: str) -> bool:
        """
        Handle special commands.
        Returns True if command was handled (skip normal RAG flow).
        """
        cmd = user_input.strip().lower()

        if cmd in EXIT_COMMANDS:
            print("\nGoodbye. Take care.\n")
            sys.exit(0)

        if cmd == "clear":
            self.history    = []
            self.turn_count = 0
            print("\nConversation cleared. Starting fresh.\n")
            return True

        if cmd == "history":
            if not self.history:
                print("\nNo conversation history yet.\n")
            else:
                print("\n── Conversation History ─────────────────────────────")
                for msg in self.history:
                    role   = "You" if msg["role"] == "user" else "Bot"
                    # show truncated version of user messages
                    # (bot messages include full context, too long to display)
                    if msg["role"] == "user":
                        # strip the context block, show only the question
                        content = msg["content"]
                        if "USER QUESTION:" in content:
                            content = content.split("USER QUESTION:")[-1].strip()
                        print(f"  {role}: {content[:100]}")
                    else:
                        print(f"  {role}: {msg['content'][:100]}...")
                print()
            return True

        if cmd == "":
            return True

        return False

    def _ask(self, query: str) -> dict:
        """
        Process one turn: retrieve chunks + generate answer.
        Returns result dict with answer and citations.
        """
        # Enrich query with conversation context for better retrieval
        retrieval_query = _build_retrieval_query(query, self.history)

        # Retrieve relevant chunks
        retrieved = retrieve(
            retrieval_query,
            self.index,
            self.chunks,
            self.embed_model,
            top_k=TOP_K_CHUNKS,
        )

        # Build messages with history
        messages = _build_messages(query, retrieved, self.history)

        # Generate response
        response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
        answer   = response["message"]["content"]

        # Update history with actual query text (not enriched version)
        # Store simplified user message in history (not the full context block)
        self.history.append({"role": "user",      "content": f"USER QUESTION: {query}"})
        self.history.append({"role": "assistant",  "content": answer})

        # Trim history to avoid context window overflow
        if len(self.history) > MAX_HISTORY_TURNS * 2:
            self.history = self.history[-(MAX_HISTORY_TURNS * 2):]

        citations = _format_citations(retrieved)
        self.turn_count += 1

        return {
            "answer":      answer,
            "citations":   citations,
            "chunks_used": min(len(retrieved), MAX_CHUNKS),
        }

    def run(self):
        """Start the interactive conversation loop."""
        self._load_pipeline()
        _print_welcome()

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nGoodbye. Take care.\n")
                break

            # Handle special commands
            if self._handle_command(user_input):
                continue

            # Process query
            print("\nBot: ", end="", flush=True)

            try:
                result = self._ask(user_input)

                print(result["answer"])

                if result["citations"]:
                    print(f"\nSources:")
                    print(result["citations"])

                _print_separator()
                print()

            except Exception as e:
                print(f"Error: {e}")
                print("Please try again.\n")