from functools import lru_cache

from pymongo import MongoClient
from pymongo.database import Database

from app.config import get_settings


@lru_cache
def get_db() -> Database:
    s = get_settings()
    return MongoClient(s.mongodb_uri, serverSelectionTimeoutMS=20000)[s.mongodb_db]
