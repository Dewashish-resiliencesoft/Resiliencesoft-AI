from app.config import get_settings
from app.db import get_db
from app.rag_engine.embed import embed_query


def retrieve(query: str, site_id: str, scopes: list[str], product_id: str | None = None,
             top_k: int | None = None) -> list[dict]:
    """Hybrid retrieval: Atlas vector search + Atlas text search, merged by reciprocal rank."""
    s = get_settings()
    k = top_k or s.top_k
    coll = get_db()["chunks"]

    flt: dict = {"site_id": site_id, "scope": {"$in": scopes}}
    if product_id:
        flt["product_id"] = {"$in": [product_id, "general"]}

    project = {"_id": 0, "text": 1, "title": 1, "source_url": 1, "product_id": 1}

    vec = list(coll.aggregate([
        {"$vectorSearch": {"index": s.vector_index, "path": "embedding",
                           "queryVector": embed_query(query), "numCandidates": k * 20,
                           "limit": k, "filter": flt}},
        {"$project": {**project, "score": {"$meta": "vectorSearchScore"}}},
    ]))
    vec = [h for h in vec if h["score"] >= s.min_score]

    txt_filter = [{"equals": {"path": "site_id", "value": site_id}},
                  {"in": {"path": "scope", "value": scopes}}]
    txt = list(coll.aggregate([
        {"$search": {"index": s.text_index,
                     "compound": {"must": [{"text": {"query": query, "path": ["text", "title"]}}],
                                  "filter": txt_filter}}},
        {"$limit": k},
        {"$project": {**project, "score": {"$meta": "searchScore"}}},
    ]))
    if product_id:
        txt = [h for h in txt if h.get("product_id") in (product_id, "general")]

    # only keep keyword hits when vector side found something relevant (avoids junk answers)
    if not vec:
        return []
    fused: dict[str, dict] = {}
    for ranked in (vec, txt):
        for rank, h in enumerate(ranked):
            key = h["text"]
            entry = fused.setdefault(key, {**h, "rrf": 0.0})
            entry["rrf"] += 1.0 / (60 + rank)
    return sorted(fused.values(), key=lambda h: h["rrf"], reverse=True)[:k]
