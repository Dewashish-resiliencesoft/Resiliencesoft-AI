#!/usr/bin/env python3
"""
rag.py — Retrieval-Augmented Generation pipeline for ResilienceSoft AI chatbot.

Commands:
    python rag.py ingest          Load chunks.jsonl into ChromaDB
    python rag.py ask "question"  Ask a question (prints answer)
    python rag.py eval            Run eval_questions.json and report hit rate
    python rag.py serve           Start a Flask API server on port 5000

Environment variables (also loadable from .env):
    GROQ_API_KEY        Required for ask/serve when using Groq
    GROQ_MODEL          Optional, default: llama-3.1-8b-instant
    EMBED_MODEL         Optional, default: all-MiniLM-L6-v2
    CHROMA_DIR          Optional, default: ./chroma_db
    TOP_K               Optional, default: 5
    PORT                Optional, default: 5000
    LLM_BACKEND         Optional, "groq" (default) or "local"
    LOCAL_MODEL_PATH    Path to a GGUF file for local inference
    API_KEY             Optional, secret key to protect /ask endpoint
"""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from textwrap import dedent

# ---------------------------------------------------------------------------
# Load .env if python-dotenv is installed (optional convenience)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # dotenv not installed; rely on shell env vars

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
LLM_BACKEND = os.getenv("LLM_BACKEND", "groq").lower()  # "groq" or "local"
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "")
API_KEY = os.getenv("API_KEY", "")  # optional auth for /ask

# Rate limiting: max requests per IP per minute
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "30"))

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
_local_llm = None


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


def _get_local_llm():
    """Load a local GGUF model via llama-cpp-python."""
    global _local_llm
    if _local_llm is None:
        if not LOCAL_MODEL_PATH:
            print("❌ Set LOCAL_MODEL_PATH to a .gguf file.")
            sys.exit(1)
        if not Path(LOCAL_MODEL_PATH).exists():
            print(f"❌ Model file not found: {LOCAL_MODEL_PATH}")
            sys.exit(1)
        try:
            from llama_cpp import Llama
        except ImportError:
            print("❌ Install llama-cpp-python:  pip install llama-cpp-python")
            sys.exit(1)
        print(f"⏳ Loading local model: {LOCAL_MODEL_PATH}")
        _local_llm = Llama(
            model_path=LOCAL_MODEL_PATH,
            n_ctx=2048,
            n_threads=os.cpu_count() or 4,
            verbose=False,
        )
        print("✅ Local model loaded.")
    return _local_llm


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
    """Generate an answer using Groq or a local GGUF model."""
    context_text = "\n\n---\n\n".join(c["text"] for c in context_chunks)

    if LLM_BACKEND == "local":
        return _generate_local(query, context_text)
    return _generate_groq(query, context_text)


def _generate_groq(query: str, context_text: str) -> str:
    """Call Groq LLM with the retrieved context and user query."""
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


def _generate_local(query: str, context_text: str) -> str:
    """Run inference on a local GGUF model via llama-cpp-python."""
    llm = _get_local_llm()
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nContext:\n{context_text}\n\n---\n\n"
        f"Question: {query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    output = llm(prompt, max_tokens=512, temperature=0.3, stop=["<|im_end|>"])
    return output["choices"][0]["text"].strip()


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
# Eval — keyword-only (no brittle hard-coded IDs)
# ---------------------------------------------------------------------------

def evaluate():
    """Run eval_questions.json through retrieval and report hit rate.

    A question passes when ALL its keywords appear somewhere in the
    concatenated text of the top-k retrieved chunks.  This is stable
    across chunk-ID renumbering after knowledge base edits.
    """
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
        keywords = [kw.lower() for kw in q.get("keywords", [])]

        results = retrieve(query, top_k=TOP_K)
        retrieved_text = " ".join(r["text"] for r in results).lower()

        # Pass when every keyword appears in the retrieved context
        passed = all(kw in retrieved_text for kw in keywords) if keywords else False

        if passed:
            hits += 1
            print(f"  ✅ {query}")
        else:
            missing = [kw for kw in keywords if kw not in retrieved_text]
            failures.append({"question": query, "missing_keywords": missing,
                             "retrieved": [r["id"] for r in results]})
            print(f"  ❌ {query}")
            print(f"      Missing keywords: {missing}")
            print(f"      Retrieved: {[r['id'] for r in results]}")

    print(f"\n📊 Hit rate: {hits}/{total} ({100 * hits / total:.0f}%)")

    if failures:
        print(f"\n⚠  {len(failures)} failures:")
        for fail in failures:
            print(f"   - \"{fail['question']}\"")
            print(f"     Missing: {fail['missing_keywords']}")


# ---------------------------------------------------------------------------
# Flask API server
# ---------------------------------------------------------------------------

def serve():
    """Start a Flask API with /ask endpoint."""
    from flask import Flask, request, jsonify

    app = Flask(__name__)

    # ---- simple in-memory rate limiter ----
    _rate_log: dict[str, list[float]] = defaultdict(list)

    def _check_rate_limit(ip: str) -> bool:
        """Return True if the IP is within the rate limit."""
        now = time.time()
        window = 60.0  # 1 minute
        timestamps = _rate_log[ip]
        # Prune old entries
        _rate_log[ip] = [t for t in timestamps if now - t < window]
        if len(_rate_log[ip]) >= RATE_LIMIT_RPM:
            return False
        _rate_log[ip].append(now)
        return True

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @app.route("/ask", methods=["POST"])
    def api_ask():
        # --- API key check ---
        if API_KEY:
            auth = request.headers.get("Authorization", "")
            if auth != f"Bearer {API_KEY}":
                return jsonify({"error": "Unauthorized"}), 401

        # --- rate limit ---
        client_ip = request.remote_addr or "unknown"
        if not _check_rate_limit(client_ip):
            return jsonify({"error": "Rate limit exceeded. Try again shortly."}), 429

        # --- input validation ---
        data = request.get_json(silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Request body must be JSON with a 'question' field"}), 400

        query = data.get("question", "")
        if not isinstance(query, str):
            return jsonify({"error": "'question' must be a string"}), 400
        query = query.strip()
        if not query:
            return jsonify({"error": "Missing or empty 'question' field"}), 400
        if len(query) > 1000:
            return jsonify({"error": "Question too long (max 1000 characters)"}), 400

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
    if API_KEY:
        print(f"   🔒 API key required (Authorization: Bearer <key>)")
    print(f"   ⏱  Rate limit: {RATE_LIMIT_RPM} req/min per IP")
    print(f"   🤖 LLM backend: {LLM_BACKEND}")
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
