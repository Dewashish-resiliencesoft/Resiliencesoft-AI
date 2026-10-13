from fastapi import APIRouter, HTTPException, Request

from app.db import get_db
from app.security.guards import check_rate_limit, resolve_site

router = APIRouter()


@router.get("/products/{slug}")
def get_product(slug: str, site_key: str, request: Request):
    check_rate_limit(request.client.host if request.client else "unknown")
    site = resolve_site(site_key, request)
    rec = get_db()["products"].find_one(
        {"site_id": site.site_id, "product_id": slug, "scope": {"$in": site.scopes}}, {"_id": 0})
    if rec is None:
        raise HTTPException(404, "Product not found")
    return rec
