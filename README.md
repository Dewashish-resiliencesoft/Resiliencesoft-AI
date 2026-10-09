# ResilienceSoft AI — RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for **ResilienceSoft**, built with ChromaDB, Sentence Transformers, and Groq.

## Architecture

```
User question
    │
    ▼
┌──────────────┐    top-k chunks    ┌────────────────┐
│  ChromaDB    │ ──────────────────▶│  Groq LLM      │
│  (vectors)   │                    │  (llama-3.1)   │
└──────────────┘                    └────────┬───────┘
    ▲                                        │
    │ embed + index                          ▼
    │                                   Answer
chunks.jsonl
(70 chunks: 27 KB docs + 43 Q&A pairs)
```

## Files

| File | Purpose |
|---|---|
| `chunks.jsonl` | Pre-built retrieval chunks (70 lines) |
| `build_chunks.py` | Rebuilds chunks.jsonl from `knowledge_base.md` + `qa_pairs.json` |
| `qa_pairs.json` | 43 question-answer pairs for retrieval |
| `eval_questions.json` | 28 test questions with expected chunk IDs and keywords |
| `rag.py` | Full RAG pipeline: ingest, ask, eval, serve |
| `requirements.txt` | Python dependencies |

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build chunks (if you've edited knowledge_base.md or qa_pairs.json)
python build_chunks.py

# 3. Ingest chunks into ChromaDB
python rag.py ingest

# 4. Run evaluation (check retrieval hit rate)
python rag.py eval

# 5. Ask a question (requires GROQ_API_KEY)
export GROQ_API_KEY="your-key-here"
python rag.py ask "Do you offer Flutter training?"

# 6. Start the API server
python rag.py serve
```

## API

### `POST /ask`

```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What services do you offer?"}'
```

Response:
```json
{
  "question": "What services do you offer?",
  "answer": "ResilienceSoft offers web solutions...",
  "sources": [{"id": "qa-008", "distance": 0.23}, ...]
}
```

### `GET /health`

Returns `{"status": "ok"}`.

## Configuration

| Env variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Groq API key |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model name |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence Transformers model |
| `CHROMA_DIR` | `./chroma_db` | ChromaDB persistence directory |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `PORT` | `5000` | Flask server port |

## Evaluation

The eval set has 28 questions. A TF-IDF stand-in retriever scored 25/28. The three that failed use different wording than the data:

| Question | Problem | Fix |
|---|---|---|
| "Where is your office?" | "office" vs "located" | Real embeddings should handle this |
| "Have you built anything for hospitals?" | "hospitals" vs "healthcare" | Real embeddings should handle this |
| "Do you work with clients outside India?" | different phrasing | Real embeddings should handle this |

If these still fail with real embeddings, add Q&A pairs worded like those questions.

## Adding your own material

1. Add content to `knowledge_base.md` (or create it if it doesn't exist)
2. Mark non-customer content with `visibility: internal` 
3. Run `python build_chunks.py` → `python rag.py ingest` → `python rag.py eval`
4. Topics to add: ResilioHub, Navigator AI, TV app, product features/pricing

## Open items to confirm

- [ ] Two addresses: footer ("Opposite CG Plaza") vs About page ("Opposite Vinayak Netralaya, Link Road")
- [ ] Client count: About page says 250+ customers, homepage says "more than 100 clients"
- [ ] Team role inconsistencies across pages
- [ ] Product detail pages (features, pricing, tech stack)
