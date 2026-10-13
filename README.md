# Resiliencesoft AI — Central AI Knowledge Engine
One Knowledge Engine. Every Product. Smarter Answers.

FastAPI + MongoDB Atlas Vector Search + LLM Gateway (Groq → NVIDIA NIM → Gemini). One API serves ResilioHub and all Resiliencesoft sites.

## Modules
- `app/rag_engine` — fastembed (bge-small-en-v1.5, 384d) + hybrid Atlas retrieval
- `app/llm_gateway` — provider fallback with per-provider timeout and overall deadline
- `app/security` — site_key→site/scope map, Origin allowlist, rate limit
- `ingestion` — Knowledge Ingestion Pipeline (admin CLI only)
- `eval` — retrieval keyword eval

## Run
```bash
cp .env.example .env   # fill MONGODB_URI, keys, SITES_JSON
uv venv && uv pip install -r requirements.txt
python -m ingestion.build_chunks
python -m ingestion.ingest --indexes      # once; wait for Atlas to build indexes
python -m ingestion.ingest
python -m eval.run_eval
uvicorn app.main:app --reload
```

## API
- `POST /api/v1/ai/chat` `{site_key, message, page_context:{url, product_slug}, history}` → `{answer, sources, request_id}`
- `GET /api/v1/ai/products/{slug}?site_key=...`
- `GET /health`

NIM is off by default (`NIM_ENABLED=false`): hosted NIM API is prototype-only without production licensing.
