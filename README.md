# Resiliencesoft AI — Central AI Knowledge Engine

One knowledge engine for ResilienceSoft and its products. Public sites ask a question. The API retrieves only that site's approved knowledge and answers from those chunks. It does not invent prices or features.

Live service: `https://resiliencesoft-ai-rag.onrender.com` (Render name `resiliencesoft-ai-rag`, `render.yaml`).

```mermaid
flowchart LR
  Visitor([Visitor on resiliohub.com])
  Widget["Landing chatbot<br/>whatsappcrm · landing-chatbot.tsx"]
  API["This API<br/>Render · /api/v1/ai/chat"]
  RH[("Atlas<br/>products.resiliohub")]
  LLM["Groq, then Gemini"]

  Visitor --> Widget
  Widget -->|"site_key resiliohub-public<br/>product_slug resiliohub"| API
  API --> RH
  RH --> LLM
  LLM --> Widget
```

Sharp Quiz and the company site are the same API with a different key. They are drawn under [Where things live](#where-things-live).

## Where things live

This repo is the engine and the knowledge. The chat widget is not here.

| What | Where |
|---|---|
| This API, retrieval, providers, security | `app/` |
| Company copy (ResilienceSoft) | `data/knowledge_base.md`, `data/qa_pairs.json`, `data/chunks.jsonl` → Atlas `resiliencesoft_ai.chunks` |
| ResilioHub copy and pricing record | `data/resiliohub/` → Atlas `products.resiliohub` |
| Sharp Quiz copy and pricing record | `data/sharpquiz/` → Atlas `products.sharpquiz` |
| Which site key may call, and from which Origin | `SITES_JSON` in `.env` (shape in `.env.example`) |
| Deploy | `render.yaml`. Start runs `python -m ingestion.sync`, then Uvicorn. |
| ResilioHub landing chatbot (only wired client) | `workprojects/whatsappcrm/client/web/components/landing/landing-chatbot.tsx`, opened from `whatsapp-float-button.tsx` |

```mermaid
flowchart TB
  subgraph crm ["Repo: workprojects/whatsappcrm"]
    Widget["client/web/components/landing/landing-chatbot.tsx"]
    Float["whatsapp-float-button.tsx"]
    Float --> Widget
  end

  subgraph engine ["Repo: Resiliencesoft AI"]
    App["app/ · FastAPI"]
    Env["SITES_JSON"]
    Sync["ingestion/sync.py"]
    D1["data/ · company"]
    D2["data/resiliohub/"]
    D3["data/sharpquiz/"]
  end

  subgraph atlas ["MongoDB Atlas"]
    C1[("resiliencesoft_ai.chunks")]
    C2[("products.resiliohub")]
    C3[("products.sharpquiz")]
  end

  Widget -->|"browser POST<br/>no proxy"| App
  Env --> App
  D1 --> Sync --> C1
  D2 --> Sync --> C2
  D3 --> Sync --> C3
  App -->|"resiliohub-public"| C2
  App -.->|"sharpquiz-public"| C3
  App -.->|"resiliencesoft-public"| C1
```

Solid line is the only client wired today. Dashed lines are the same API, unused by the landing widget.

The landing widget is a static Next export. It calls Render from the browser. It cannot proxy.

What that widget sends:

- `site_key`: `resiliohub-public`
- `message`: the visitor's question only, so search stays on the question
- `page_context`: `{ url: location.href, product_slug: "resiliohub" }`
- `history`: a fixed ResilioBot scope pair, then the last few turns

That scope pair says: visitor is on the ResilioHub landing page, ResilioHub is a ResilienceSoft product, answer from ResilioHub, mention another product only if they name it, mention the company only if they ask. **This API does not pass `history` to the model**, so that pair does not change the answer today. Separation is already in the data: a `resiliohub-public` call never opens the Sharp Quiz collection or the company collection. Other products need `sharpquiz-public`. Company facts need `resiliencesoft-public`.

Browser Origins that work for ResilioHub right now: `https://resiliohub.com` and `https://www.resiliohub.com`. `https://staging.resiliohub.com` and `http://localhost:3000` get `403 Origin not allowed` until they are added to that site's `origins` list.

## What we are doing

Each public site gets its own `site_key`. The key picks a site, an Origin allowlist, and a `public` scope.

```mermaid
flowchart TD
  Q["POST /api/v1/ai/chat"] --> Key{site_key and Origin}
  Key -->|no| Deny["403"]
  Key -->|yes| Guard{injection or secret}
  Guard -->|yes| Contact["Contact line, no sources"]
  Guard -->|no| Price{pricing words}
  Price -->|yes| Record["Structured product record only"]
  Record -->|unpublished| Contact
  Record -->|published| Done["answer + sources"]
  Price -->|no| Search["Embed message · vector search · keyword re-rank"]
  Search -->|no hits| Contact
  Search -->|hits| Model["LLM sees chunks only, not history"]
  Model -->|NO_ANSWER or all providers fail| Contact
  Model -->|grounded text| Done
```

A chat call then:

1. Rejects unknown keys and Origins that are not on that site's list.
2. Rate-limits by IP (`RATE_LIMIT_RPM`, default 30).
3. Drops prompt-injection attempts and any chunk or answer that looks like a secret.
4. Answers pricing questions from the structured product record only. If pricing is unpublished, it tells the visitor to contact the company. The model never guesses a price.
5. For everything else, embeds the visitor message and searches that site's chunks (vector search plus a keyword re-rank). `page_context.product_slug` keeps hits on that product and on `general` chunks.
6. Sends the retrieved text to the LLM with a fixed system prompt: answer only from `<context>`, otherwise `NO_ANSWER`.
7. If retrieval is empty, every provider fails, or the model says `NO_ANSWER`, the visitor gets a contact line instead of a made-up answer.

`history` is accepted and scanned for injection. It is not sent to the model. The answer is grounded in the current message plus retrieved chunks.

Sites in `SITES_JSON` today:

| site_key | Knowledge | Typical page |
|---|---|---|
| `resiliencesoft-public` | Company site | resiliencesoft.com |
| `resiliohub-public` | ResilioHub | resiliohub.com |
| `sharpquiz-public` | Sharp Quiz | sharpquiz.com |

A ResilioHub call does not search Sharp Quiz or the company corpus. Those live in other collections and need their own `site_key`. Pass `product_slug` (for example `resiliohub`) so retrieval stays on that product.

## What we are using

| Piece | Choice |
|---|---|
| API | FastAPI, Uvicorn, Pydantic |
| Database | MongoDB Atlas. Company chunks in `resiliencesoft_ai.chunks`. One collection per product in the `products` database (`resiliohub`, `sharpquiz`). |
| Embeddings | fastembed, `BAAI/bge-small-en-v1.5`, 384 dimensions, cosine |
| Retrieval | Atlas Vector Search (`chunks_vector`) plus an in-collection keyword scan, merged with reciprocal rank. Top 5, minimum vector score 0.55. |
| LLM | Groq (`openai/gpt-oss-20b`), then Gemini (`gemini-3.1-flash-lite`). NVIDIA NIM (`nvidia/nemotron-3-super-120b-a12b`) is in the chain but off (`NIM_ENABLED=false`) until it has a production license. |
| Fallback | Per-provider timeout 15s, one retry on 429/5xx, overall deadline 30s. Order: `groq,nim,gemini`. |

```mermaid
flowchart LR
  Hits["Retrieved chunks"] --> Groq["Groq<br/>gpt-oss-20b"]
  Groq -->|fail| NIM["NVIDIA NIM<br/>off"]
  NIM -->|skipped| Gemini["Gemini<br/>3.1 flash lite"]
  Groq -->|ok| Answer["Answer"]
  Gemini -->|ok| Answer
  Gemini -->|fail| Contact["Contact line"]
```
| Host | Render free web service, Python 3.12. Start command syncs knowledge, then serves. |

## Knowledge in the repo

```mermaid
flowchart LR
  subgraph folder ["Each product folder"]
    MD["knowledge_base.md"]
    QA["qa_pairs.json"]
    CH["chunks.jsonl"]
    PR["products/slug.json"]
  end
  MD --> Build["build_chunks.py"]
  QA --> Build
  Build --> CH
  CH --> Atlas[("Atlas collection")]
  PR --> Atlas
```

```
data/                      ResilienceSoft company site
data/resiliohub/           ResilioHub
data/sharpquiz/            Sharp Quiz
  knowledge_base.md        source copy
  qa_pairs.json            one chunk per pair
  chunks.jsonl             retrieval chunks
  products/<slug>.json     structured record (pricing, contact)
```

`python -m ingestion.sync` loads those three folders. Unchanged chunks are skipped, so a code-only deploy does not re-embed. `ingestion/crawl_site.py` is an optional same-site crawler (`robots.txt`, 0.5s delay). It needs BeautifulSoup and is not part of the API runtime.

## Run

```bash
cp .env.example .env   # fill MONGODB_URI, provider keys, SITES_JSON
uv venv && uv pip install -r requirements.txt
python -m ingestion.build_chunks
python -m ingestion.ingest --indexes      # once; wait for Atlas to build indexes
python -m ingestion.sync
python -m eval.run_eval
uvicorn app.main:app --reload
```

## API

`POST /api/v1/ai/chat`

```json
{
  "site_key": "resiliohub-public",
  "message": "How do I connect a WhatsApp number?",
  "page_context": { "url": "https://resiliohub.com/", "product_slug": "resiliohub" },
  "history": []
}
```

Response: `{ "answer", "sources": [{ "title", "url" }], "request_id" }`.

The browser `Origin` must be on that site's allowlist. `message` max 1000 characters. Each history item max 2000 characters, role `user` or `assistant`. The ResilioHub landing widget is the client that sends this body. See [Where things live](#where-things-live).

- `GET /api/v1/ai/products/{slug}?site_key=...` — structured product record for that site
- `GET /health`

## Modules

- `app/rag_engine` — embed, hybrid retrieval, grounded prompt
- `app/llm_gateway` — provider order, timeout, retry
- `app/security` — site key, Origin allowlist, rate limit, secret and injection checks
- `ingestion` — chunk build, Atlas ingest, deploy sync
- `eval` — retrieval keyword checks (`eval/run_eval.py`) and chat checks (`eval/run_chat_eval.py`)
