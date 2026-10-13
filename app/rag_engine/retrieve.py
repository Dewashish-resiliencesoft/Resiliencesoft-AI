import re

from app.config import get_settings
from app.db import COMPANY_SITE, knowledge_collection
from app.rag_engine.embed import embed_query


def _keyword_hits(coll, query: str, flt: dict, k: int, project: dict) -> list[dict]:
    # ponytail: scans the collection. Fine while a product is a few hundred chunks.
    # Upgrade: Atlas Search text index once the cluster is past the M0 3-index cap.
    words = [w for w in re.findall(r"[a-z0-9]{3,}", query.lower())][:6]
    if not words:
        return []
    ors = [{"text": {"$regex": re.escape(w), "$options": "i"}} for w in words]
    scored = []
    for h in coll.find({**flt, "$or": ors}, project):
        blob = f"{h.get('title', '')} {h.get('text', '')}".lower()
        n = sum(w in blob for w in words)
        if n:
            scored.append({**h, "score": n})
    scored.sort(key=lambda h: h["score"], reverse=True)
    return scored[:k]


def retrieve(query: str, site_id: str, scopes: list[str], product_id: str | None = None,
             top_k: int | None = None) -> list[dict]:
    """Hybrid retrieval: Atlas vector search + keyword scan, merged by reciprocal rank."""
    s = get_settings()
    k = top_k or s.top_k
    coll = knowledge_collection(site_id)
    if coll is None:
        return []

    flt: dict = {"scope": {"$in": scopes}}
    if site_id == COMPANY_SITE:
        flt["site_id"] = site_id
    else:
        flt["doc_type"] = "chunk"
    if product_id:
        flt["product_id"] = {"$in": [product_id, "general"]}

    project = {"_id": 0, "text": 1, "title": 1, "source_url": 1, "product_id": 1}
    pool = max(k * 3, 15)

    vec = list(coll.aggregate([
        {"$vectorSearch": {"index": s.vector_index, "path": "embedding",
                           "queryVector": embed_query(query), "numCandidates": max(pool * 4, 60),
                           "limit": pool, "filter": flt}},
        {"$project": {**project, "score": {"$meta": "vectorSearchScore"}}},
    ]))
    vec = [h for h in vec if h["score"] >= s.min_score]
    # keyword only re-ranks vector hits; it must not push a relevant chunk out of the pool
    in_vec = {h["text"] for h in vec}
    txt = [h for h in _keyword_hits(coll, query, flt, pool, project) if h["text"] in in_vec]

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
