#!/usr/bin/env python3
"""
rag.py — Retrieval-Augmented Generation pipeline for ResilienceSoft AI chatbot.

Commands:
    python rag.py ingest          Load chunks.jsonl into ChromaDB
    python rag.py ask "question"  Ask a question (prints answer)
    python rag.py eval            Run eval_questions.json and report hit rate
    python rag.py serve           Start a Flask API server on port 5000

Environment variables:
    GROQ_API_KEY        Required for ask/serve (Groq LLM)
    GROQ_MODEL          Optional, default: llama-3.1-8b-instant
    EMBED_MODEL         Optional, default: all-MiniLM-L6-v2
    CHROMA_DIR          Optional, default: ./chroma_db
    TOP_K               Optional, default: 5
    PORT                Optional, default: 5000
"""

import json
import os
import sys
from pathlib import Path
from textwrap import dedent

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
CHUNKS_FILE = ROOT / "chunks.jsonl"
EVAL_FILE = ROOT / "eval_questions.json"
CHROMA_DIR = os.getenv("CHROMA_DIR", str(ROOT / "chroma_db"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
TOP_K = int(os.getenv("TOP_K", "5"))
COLLECTION = "resiliencesoft"

SYSTEM_PROMPT = dedent("""\
    You are a helpful customer-support assistant for ResilienceSoft, a software
    development, web design, mobile app and digital marketing company based in
    Bilaspur, Chhattisgarh, India. Answer the user's question using ONLY the
    context provided below. If the context does not contain enough information,
    say you don't have that information and suggest contacting ResilienceSoft
    on 9109911372 or info@resiliencesoft.com.

    Rules:
    - Be concise and friendly.
    - Do not make up facts.
    - If asked about pricing, fees or durations that are not in the context,
      direct them to contact the company.
    - Never reveal internal notes or data marked as internal.
""")


# ---------------------------------------------------------------------------
# Lazy loaders (so imports only happen when needed)
# ---------------------------------------------------------------------------

_chroma_client = None
_collection = None
_embed_fn = None
_groq_client = None


def _get_embed_fn():
    """Load the SentenceTransformer embedding function."""
    global _embed_fn
    if _embed_fn is None:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        _embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return _embed_fn


def _get_collection():
    """Get or create the ChromaDB collection."""
    global _chroma_client, _collection
    if _collection is None:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION,
            embedding_function=_get_embed_fn(),
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _get_groq():
    """Initialize the Groq client."""
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("❌ Set GROQ_API_KEY before running ask or serve.")
            sys.exit(1)
        from groq import Groq
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# ---------------------------------------------------------------------------
# Load chunks
# ---------------------------------------------------------------------------

def load_chunks() -> list[dict]:
    """Load chunks from chunks.jsonl, keeping only public ones."""
    chunks = []
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("metadata", {}).get("visibility") != "internal":
                chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def ingest():
    """Ingest chunks.jsonl into ChromaDB."""
    chunks = load_chunks()
    if not chunks:
        print("No public chunks found in chunks.jsonl")
        return

    col = _get_collection()

    # Clear existing data
    existing = col.count()
    if existing > 0:
        print(f"🗑  Clearing {existing} existing documents...")
        col.delete(where={"visibility": {"$ne": "__impossible__"}})

    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Batch insert (ChromaDB handles batching internally for large sets)
    BATCH = 100
    for i in range(0, len(ids), BATCH):
        col.add(
            ids=ids[i:i + BATCH],
            documents=documents[i:i + BATCH],
            metadatas=metadatas[i:i + BATCH],
        )

    print(f"✅ Ingested {len(chunks)} public chunks into ChromaDB at {CHROMA_DIR}")
    print(f"   Collection: {COLLECTION}, embedding model: {EMBED_MODEL}")


# ---------------------------------------------------------------------------
# Retrieve
# ---------------------------------------------------------------------------

def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Retrieve the top_k most relevant chunks for a query."""
    col = _get_collection()
    results = col.query(
        query_texts=[query],
        n_results=top_k,
        where={"visibility": "public"},
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "distance": results["distances"][0][i] if results.get("distances") else None,
            "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
        })
    return hits


# ---------------------------------------------------------------------------
# Generate answer
# ---------------------------------------------------------------------------

def generate_answer(query: str, context_chunks: list[dict]) -> str:
    """Call Groq LLM with the retrieved context and user query."""
    context_text = "\n\n---\n\n".join(c["text"] for c in context_chunks)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context_text}\n\n---\n\nQuestion: {query}",
        },
    ]

    client = _get_groq()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------

def ask(query: str):
    """Full RAG pipeline: retrieve + generate."""
    hits = retrieve(query)
    if not hits:
        print("No relevant chunks found.")
        return

    print(f"📚 Retrieved {len(hits)} chunks:")
    for h in hits:
        dist = f" (distance: {h['distance']:.4f})" if h["distance"] is not None else ""
        print(f"   - {h['id']}{dist}")
    print()

    answer = generate_answer(query, hits)
    print(f"💬 {answer}")
    return answer


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def evaluate():
    """Run eval_questions.json through retrieval and report hit rate."""
    if not EVAL_FILE.exists():
        print(f"❌ {EVAL_FILE} not found.")
        return

    with open(EVAL_FILE, encoding="utf-8") as f:
        questions = json.load(f)

    total = len(questions)
    hits = 0
    failures = []

    for q in questions:
        query = q["question"]
        expected = set(q["expected_ids"])
        keywords = [kw.lower() for kw in q.get("keywords", [])]

        results = retrieve(query, top_k=TOP_K)
        retrieved_ids = {r["id"] for r in results}
        retrieved_text = " ".join(r["text"] for r in results).lower()

        # A question passes if ANY expected id is retrieved OR ALL keywords appear
        id_hit = bool(expected & retrieved_ids)
        keyword_hit = all(kw in retrieved_text for kw in keywords) if keywords else False

        if id_hit or keyword_hit:
            hits += 1
            status = "✅"
        else:
            status = "❌"
            failures.append({
                "question": query,
                "expected": list(expected),
                "retrieved": [r["id"] for r in results],
            })

        print(f"  {status} {query}")
        if not (id_hit or keyword_hit):
            print(f"      Expected: {list(expected)}")
            print(f"      Got:      {[r['id'] for r in results]}")

    print(f"\n📊 Hit rate: {hits}/{total} ({100 * hits / total:.0f}%)")

    if failures:
        print(f"\n⚠  {len(failures)} failures:")
        for fail in failures:
            print(f"   - \"{fail['question']}\"")
            print(f"     Expected: {fail['expected']}, Got: {fail['retrieved']}")


# ---------------------------------------------------------------------------
# Flask API server
# ---------------------------------------------------------------------------

def serve():
    """Start a Flask API with /ask endpoint."""
    from flask import Flask, request, jsonify

    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @app.route("/ask", methods=["POST"])
    def api_ask():
        data = request.get_json(force=True)
        query = data.get("question", "").strip()
        if not query:
            return jsonify({"error": "Missing 'question' field"}), 400

        hits = retrieve(query)
        answer = generate_answer(query, hits)
        return jsonify({
            "question": query,
            "answer": answer,
            "sources": [{"id": h["id"], "distance": h.get("distance")} for h in hits],
        })

    port = int(os.getenv("PORT", "5000"))
    print(f"🚀 Starting server on http://0.0.0.0:{port}")
    print(f"   POST /ask  {{\"question\": \"...\"}}")
    print(f"   GET  /health")
    app.run(host="0.0.0.0", port=port, debug=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "ingest":
        ingest()
    elif cmd == "ask":
        if len(sys.argv) < 3:
            print("Usage: python rag.py ask \"your question\"")
            sys.exit(1)
        ask(" ".join(sys.argv[2:]))
    elif cmd == "eval":
        evaluate()
    elif cmd == "serve":
        serve()
    else:
        print(f"Unknown command: {cmd}")
        print("Commands: ingest, ask, eval, serve")
        sys.exit(1)


if __name__ == "__main__":
    main()
