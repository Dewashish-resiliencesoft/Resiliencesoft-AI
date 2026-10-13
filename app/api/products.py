from fastapi import APIRouter, HTTPException, Request

from app.db import find_product
from app.security.guards import check_rate_limit, resolve_site

router = APIRouter()


@router.get("/products/{slug}")
def get_product(slug: str, site_key: str, request: Request):
    check_rate_limit(request.client.host if request.client else "unknown")
    site = resolve_site(site_key, request)
    rec = find_product(site.site_id, slug, site.scopes)
    if rec is None:
        raise HTTPException(404, "Product not found")
    return rec
