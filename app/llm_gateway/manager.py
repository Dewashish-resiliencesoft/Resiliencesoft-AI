import logging
import time

from app.config import Settings, get_settings
from app.llm_gateway import providers

log = logging.getLogger("llm_gateway")

_REGISTRY = {"groq": providers.groq, "nim": providers.nim, "gemini": providers.gemini}


def _enabled(name: str, s: Settings) -> bool:
    if name == "groq":
        return bool(s.groq_api_key)
    if name == "nim":
        return s.nim_enabled and bool(s.nim_api_key)
    if name == "gemini":
        return bool(s.gemini_api_key)
    return False


def generate_answer(system: str, user: str, request_id: str = "") -> str | None:
    """Try providers in order; return None if all fail (caller returns contact fallback)."""
    s = get_settings()
    deadline = time.monotonic() + s.overall_deadline
    for name in (p.strip() for p in s.provider_order.split(",")):
        fn = _REGISTRY.get(name)
        if fn is None or not _enabled(name, s):
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 1:
            break
        for attempt in (1, 2):
            try:
                return fn(s, system, user, min(s.provider_timeout, remaining))
            except providers.ProviderError as e:
                log.warning("req=%s provider=%s attempt=%d failed: %s", request_id, name, attempt, e)
                # retry once, only on transient errors (rate limit / 5xx)
                if attempt == 2 or not str(e).startswith(("HTTP 429", "HTTP 5")):
                    break
                time.sleep(1)
                remaining = deadline - time.monotonic()
                if remaining <= 1:
                    break
    return None


def generate_deltas(system: str, user: str, request_id: str = ""):
    """Yield text chunks. Groq streams. A failed stream with no chunks falls back to one full reply."""
    s = get_settings()
    if _enabled("groq", s):
        yielded = False
        try:
            for delta in providers.groq_deltas(s, system, user, s.provider_timeout):
                yielded = True
                yield delta
        except providers.ProviderError as e:
            log.warning("req=%s provider=groq stream failed: %s", request_id, e)
        if yielded:
            return
    text = generate_answer(system, user, request_id)
    if text:
        yield text
