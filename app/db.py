from functools import lru_cache

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from app.config import get_settings

# Atlas tree: database `products`, one collection per product (resiliohub, sharpquiz).
# The company site stays in resiliencesoft_ai.chunks. Mongo has no subfolders.
PRODUCTS_DB = "products"
COMPANY_SITE = "resiliencesoft"


@lru_cache
def get_client() -> MongoClient:
    s = get_settings()
    return MongoClient(s.mongodb_uri, serverSelectionTimeoutMS=20000)


@lru_cache
def get_db() -> Database:
    return get_client()[get_settings().mongodb_db]


def knowledge_collection(site_id: str, *, create: bool = False) -> Collection | None:
    if site_id == COMPANY_SITE:
        return get_db()["chunks"]
    db = get_client()[PRODUCTS_DB]
    if not create and site_id not in db.list_collection_names():
        return None
    return db[site_id]


def find_product(site_id: str, product_id: str | None, scopes: list[str]) -> dict | None:
    if not product_id:
        return None
    proj = {"_id": 0, "embedding": 0, "doc_type": 0}
    if site_id == COMPANY_SITE:
        return get_db()["products"].find_one(
            {"site_id": site_id, "product_id": product_id, "scope": {"$in": scopes}}, proj)
    return knowledge_collection(site_id).find_one(
        {"doc_type": "product", "product_id": product_id, "scope": {"$in": scopes}}, proj)
