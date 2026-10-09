# ResilienceSoft AI — RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for **ResilienceSoft**, built with ChromaDB, Sentence Transformers (`all-MiniLM-L6-v2`), and Groq or local self-hosted models (via `llama-cpp-python`).

## Architecture

```
User question
    │
    ▼
┌──────────────┐    top-k chunks    ┌────────────────────────────────────────┐
│  ChromaDB    │ ──────────────────▶│  LLM Engine                            │
│  (vectors)   │                    │  • Groq (default: llama-3.1-8b-instant)│
└──────────────┘                    │  • Local GGUF (e.g. Qwen2.5 1.5B/3B)   │
    ▲                               └───────────────────┬────────────────────┘
    │ embed + index                                     │
    │                                                   ▼
chunks.jsonl                                         Answer
(70 chunks: 27 KB docs + 43 Q&A pairs)
```

## Files

| File | Purpose |
|---|---|
| `chunks.jsonl` | Pre-built retrieval chunks (70 total: 69 public, 1 internal) |
| `build_chunks.py` | Rebuilds `chunks.jsonl` from `knowledge_base.md` + `qa_pairs.json`, splitting long sections at ~300 words |
| `knowledge_base.md` | Core Markdown knowledge base extracted from public site crawl |
| `qa_pairs.json` | 43 curated Q&A pairs for precision retrieval |
| `eval_questions.json` | 28 test questions with strict keywords for retrieval validation |
| `rag.py` | Complete RAG pipeline: `ingest`, `ask`, `eval`, `serve` |
| `requirements.txt` | Python dependencies |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt

# (Optional) If you plan to run local GGUF models:
# pip install llama-cpp-python
```

### 2. Configure Environment

Create a `.env` file (loaded automatically by `rag.py`):

```bash
# For Groq cloud inference:
GROQ_API_KEY="your-groq-api-key"
GROQ_MODEL="llama-3.1-8b-instant"

# Or for local self-hosted inference:
# LLM_BACKEND="local"
# LOCAL_MODEL_PATH="/path/to/qwen2.5-1.5b-instruct.Q4_K_M.gguf"

# Optional security:
# API_KEY="your-secret-api-token"
# RATE_LIMIT_RPM=30
```

### 3. Build Chunks & Ingest

```bash
# Build chunks (splits at 300 words, filters internal sections)
python build_chunks.py

# Ingest into ChromaDB vector store
python rag.py ingest
```

### 4. Evaluate Retrieval

```bash
python rag.py eval
```
Retrieval checks that all required keywords appear in the retrieved context, independent of chunk ID shifts.

### 5. Ask Questions

```bash
python rag.py ask "Do you offer Flutter training?"
```

### 6. Start the API Server

```bash
python rag.py serve
```

---

## API Reference

### `POST /ask`

Includes input validation, query length checks, and per-IP rate limiting (default 30 requests/min).

```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <API_KEY>" \
  -d '{"question": "Where is your office located?"}'
```

**Success Response (200):**
```json
{
  "question": "Where is your office located?",
  "answer": "ResilienceSoft is located at 2nd Floor, Emerald Plaza, Telephone Exchange Road, Opposite CG Plaza, Bilaspur, Chhattisgarh 495001.",
  "sources": [
    {"id": "qa-003", "distance": 0.12},
    {"id": "kb-009", "distance": 0.19}
  ]
}
```

**Error Responses:**
- `400 Bad Request`: Empty query, non-string input, or exceeds 1,000 characters.
- `401 Unauthorized`: When `API_KEY` is configured and header is missing or incorrect.
- `429 Too Many Requests`: When client exceeds the configured rate limit.

### `GET /health`

Returns `{"status": "ok"}`.

---

## Configuration Reference

| Env Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required for Groq)* | Groq API authentication key |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq chat model identifier |
| `LLM_BACKEND` | `groq` | Choose `groq` or `local` |
| `LOCAL_MODEL_PATH` | `""` | Path to `.gguf` file if `LLM_BACKEND=local` |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformers embedding model |
| `CHROMA_DIR` | `./chroma_db` | Directory for persistent ChromaDB storage |
| `TOP_K` | `5` | Number of context chunks retrieved per query |
| `PORT` | `5000` | Flask server port |
| `API_KEY` | `""` | Optional Bearer token protection for `/ask` |
| `RATE_LIMIT_RPM` | `30` | Max requests per IP per minute |

---

## Open Decisions (Team Confirmation)

- [ ] **Address Confirmation**: Reconcile footer address ("Opposite CG Plaza") vs About page ("Opposite Vinayak Netralaya, Link Road").
- [ ] **Client Count**: Reconcile homepage stat ("100+ clients") vs About page stat ("250+ customers").
- [ ] **Team Roles**: Confirm titles for Deepak Verma and Shreya Gupta across different pages.
- [ ] **Internal Products**: Add documentation for ResilioHub, Navigator AI, Android TV signage, etc., tagged with `visibility: internal`.
