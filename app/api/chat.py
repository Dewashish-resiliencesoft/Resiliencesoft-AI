import re
import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.config import get_settings
from app.db import COMPANY_SITE, find_product
from app.llm_gateway.manager import generate_answer
from app.rag_engine.prompt import NO_ANSWER, social_reply, system_prompt, user_prompt
from app.rag_engine.retrieve import retrieve
from app.security.guards import (
    check_rate_limit, contains_secret, looks_like_injection, resolve_site, scrub_untrusted,
)

router = APIRouter()

PRICE_RE = re.compile(r"\b(price|pricing|cost|plan|plans|fee|fees|charge|quote|package)s?\b", re.I)


class PageContext(BaseModel):
    url: str = Field("", max_length=500)
    product_slug: str | None = Field(None, max_length=100)


class HistoryItem(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=2000)


class ChatRequest(BaseModel):
    site_key: str = Field(max_length=100)
    message: str = Field(min_length=1)
    page_context: PageContext = PageContext()
    history: list[HistoryItem] = []


def _contact_line(site_id: str) -> str:
    s = get_settings()
    rec = find_product(site_id, site_id, ["public"]) if site_id != COMPANY_SITE else None
    contact = (rec or {}).get("contact") or {}
    phone = contact.get("phone") or s.contact_phone
    email = contact.get("email") or contact.get("developer_email") or s.contact_email
    who = (rec or {}).get("name") or "Resiliencesoft"
    return f"contact {who} on {phone} or {email}"


def _published_price(rec: dict | None) -> tuple[str, str] | None:
    """Return (answer, source_url) from an approved product record. None if unpublished."""
    if not rec:
        return None
    pricing = rec.get("pricing") or {}
    plans = pricing.get("plans") or []
    if not plans or pricing.get("status") == "requires_verification":
        return None
    parts = []
    for plan in plans:
        price = plan.get("price")
        label = plan.get("name") or "Plan"
        bit = f"{label} ₹{price:,}" if isinstance(price, int) else f"{label} {price}"
        if plan.get("days"):
            bit += f" / {plan['days']} days"
        parts.append(bit)
    trial = pricing.get("free_trial_days")
    extra = f" {trial}-day free trial." if trial else ""
    url = pricing.get("source") or rec.get("source_url") or ""
    return f"{rec.get('name', 'Plans')}: {'; '.join(parts)}.{extra}", url


def _dialog(history: list[HistoryItem], limit: int, char_budget: int) -> list[HistoryItem]:
    # Each request brings its own history. This process stores none, so two visitors cannot share a thread.
    chosen: list[HistoryItem] = []
    used = 0
    for item in reversed(history[-limit:]):
        room = char_budget - used
        if room <= 0:
            break
        text = item.content.strip()[:room]
        if not text:
            continue
        used += len(text)
        chosen.append(HistoryItem(role=item.role, content=text))
    chosen.reverse()
    return chosen


def _pack(answer: str, sources: list[dict], rid: str) -> dict:
    return {"answer": answer, "sources": sources, "request_id": rid}


def _finish(answer: str | None, sources: list[dict], site_id: str, rid: str) -> dict:
    if not answer or NO_ANSWER in answer or contains_secret(answer):
        return _pack(f"I don't have that information. Please {_contact_line(site_id)}.", [], rid)
    return _pack(answer, sources, rid)


@router.post("/chat")
def chat(req: ChatRequest, request: Request):
    s = get_settings()
    rid = "req_" + uuid.uuid4().hex[:12]
    check_rate_limit(request.client.host if request.client else "unknown")
    site = resolve_site(req.site_key, request)

    message = req.message.strip()
    if not message or len(message) > s.max_message_chars:
        raise HTTPException(400, f"Message must be 1-{s.max_message_chars} characters")
    if looks_like_injection(message) or any(looks_like_injection(h.content) for h in req.history):
        return {"answer": f"I don't have that information. Please {_contact_line(site.site_id)}.",
                "sources": [], "request_id": rid}

    product = req.page_context.product_slug or (None if site.site_id == COMPANY_SITE else site.site_id)
    sources: list[dict] = []
    greeted = social_reply(message)
    if greeted:
        return _pack(greeted, [], rid)

    # exact-value questions: structured record only, never LLM guess
    if PRICE_RE.search(message):
        rec = find_product(site.site_id, product, site.scopes)
        published = _published_price(rec)
        if published is None:
            return _pack(
                f"I don't have approved pricing information for that. Please {_contact_line(site.site_id)}.",
                [], rid,
            )
        answer, url = published
        sources = [{"title": rec.get("name") or "Pricing", "url": url}] if url else []
        return _pack(answer, sources, rid)

    hits = retrieve(message, site.site_id, site.scopes, product)
    safe_hits = []
    seen = set()
    for h in hits:
        if contains_secret(h.get("text", "")) or contains_secret(h.get("title", "")):
            continue
        h = {**h, "text": scrub_untrusted(h.get("text", ""))}
        safe_hits.append(h)
        if h.get("source_url") and h["source_url"] not in seen:
            seen.add(h["source_url"])
            sources.append({"title": h.get("title", ""), "url": h["source_url"]})
    if not safe_hits:
        sources = []
    history = _dialog(req.history, s.max_history, s.max_history_chars)
    answer = generate_answer(system_prompt(), user_prompt(message, safe_hits, history), rid)
    if answer is None:
        return _pack(
            f"Sorry, I'm having trouble answering right now. Please {_contact_line(site.site_id)}.",
            [], rid,
        )
    return _finish(answer, sources, site.site_id, rid)
