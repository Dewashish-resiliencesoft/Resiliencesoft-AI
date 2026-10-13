import time
from collections import defaultdict

from fastapi import HTTPException, Request

from app.config import SiteConfig, get_settings

_rate_log: dict[str, list[float]] = defaultdict(list)


def check_rate_limit(ip: str) -> None:
    now = time.time()
    recent = [t for t in _rate_log[ip] if now - t < 60]
    if len(recent) >= get_settings().rate_limit_rpm:
        _rate_log[ip] = recent
        raise HTTPException(429, "Rate limit exceeded. Try again shortly.")
    recent.append(now)
    _rate_log[ip] = recent


def resolve_site(site_key: str, request: Request) -> SiteConfig:
    """site_key is an identifier, not a secret: also enforce Origin allowlist."""
    site = get_settings().sites.get(site_key)
    if site is None:
        raise HTTPException(403, "Invalid site key")
    origin = request.headers.get("origin")
    if site.origins and origin not in site.origins:
        raise HTTPException(403, "Origin not allowed")
    return site
