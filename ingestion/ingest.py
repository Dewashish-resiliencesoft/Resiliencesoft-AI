"""Knowledge Ingestion Pipeline: chunks.jsonl (+ data/products/*.json) -> MongoDB Atlas.

Admin-only CLI. Usage:
    python -m ingestion.build_chunks
    python -m ingestion.ingest [--site-id resiliencesoft] [--product-id general] [--source-url URL] [--dir data/resiliohub]
    python -m ingestion.ingest --indexes     # create Atlas search indexes
"""
import argparse
import hashlib
import json
from pathlib import Path

from app.config import get_settings
from app.db import COMPANY_SITE, get_db, knowledge_collection
from app.rag_engine.embed import embed_documents

DATA = Path(__file__).resolve().parent.parent / "data"


def ensure_vector_index(coll):
    """One Atlas vector index per collection. M0 allows 3 on the whole cluster."""
    from pymongo.operations import SearchIndexModel
    s = get_settings()
    coll.create_index([("chunk_key", 1)], unique=True)
    names = {i.get("name") for i in coll.list_search_indexes()}
    if s.vector_index in names:
        print(f"{coll.full_name}: {s.vector_index} already exists")
        return
    coll.create_search_index(SearchIndexModel(name=s.vector_index, type="vectorSearch", definition={
        "fields": [
            {"type": "vector", "path": "embedding", "numDimensions": s.embed_dim, "similarity": "cosine"},
            {"type": "filter", "path": "site_id"}, {"type": "filter", "path": "product_id"},
            {"type": "filter", "path": "scope"}, {"type": "filter", "path": "language"},
            {"type": "filter", "path": "doc_type"}]}))
    print(f"{coll.full_name}: vector index requested")


def ensure_indexes():
    ensure_vector_index(get_db()["chunks"])


def ingest(site_id: str, product_id: str, source_url: str, folder: Path = DATA):
    coll = knowledge_collection(site_id, create=True)
    product_here = site_id != COMPANY_SITE
    chunks = [json.loads(l) for l in (folder / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    docs = []
    for c in chunks:
        m = c["metadata"]
        docs.append({
            "chunk_key": c["id"], "site_id": site_id, "product_id": product_id,
            "scope": "internal" if m.get("visibility") == "internal" else "public",
            "language": "en", "title": m.get("section", ""), "text": c["text"],
            "source_url": m.get("source_url", source_url), "source": m.get("source", ""),
            "content_hash": hashlib.sha256(c["text"].encode()).hexdigest(),
            **({"doc_type": "chunk"} if product_here else {}),
        })
    owned = {"doc_type": "chunk"} if product_here else {"site_id": site_id}
    existing = {d["chunk_key"]: d["content_hash"] for d in coll.find(owned, {"chunk_key": 1, "content_hash": 1})}
    changed = [d for d in docs if existing.get(d["chunk_key"]) != d["content_hash"]]
    if changed:
        for d, v in zip(changed, embed_documents([d["text"] for d in changed])):
            d["embedding"] = v
            d["document_version"] = 1
            coll.replace_one({**owned, "chunk_key": d["chunk_key"]}, d, upsert=True)
    stale = set(existing) - {d["chunk_key"] for d in docs}
    if stale:
        coll.delete_many({**owned, "chunk_key": {"$in": list(stale)}})
    print(f"upserted {len(changed)}, unchanged {len(docs) - len(changed)}, removed {len(stale)}")

    store = coll if product_here else get_db()["products"]
    for p in (folder / "products").glob("*.json"):
        rec = json.loads(p.read_text(encoding="utf-8"))
        key = {"site_id": rec["site_id"], "product_id": rec["product_id"]}
        if product_here:
            rec["doc_type"] = "product"
            key = {"doc_type": "product", "product_id": rec["product_id"]}
        store.replace_one(key, rec, upsert=True)
        print(f"product {rec['product_id']} upserted")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--site-id", default="resiliencesoft")
    ap.add_argument("--product-id", default="general")
    ap.add_argument("--source-url", default="https://www.resiliencesoft.com")
    ap.add_argument("--dir", default=str(DATA))
    ap.add_argument("--indexes", action="store_true")
    a = ap.parse_args()
    ensure_indexes() if a.indexes else ingest(a.site_id, a.product_id, a.source_url, Path(a.dir))
