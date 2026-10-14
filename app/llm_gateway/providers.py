import json

import httpx

from app.config import Settings


class ProviderError(Exception):
    """Temporary failure; gateway should try next provider."""


def _openai_compat(url: str, key: str, model: str, system: str, user: str, timeout: float) -> str:
    try:
        r = httpx.post(
            url,
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "temperature": 0.3, "max_tokens": 512,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}]},
            timeout=timeout,
        )
    except httpx.HTTPError as e:
        raise ProviderError(str(e)) from e
    if r.status_code != 200:
        raise ProviderError(f"HTTP {r.status_code}")
    try:
        text = (r.json()["choices"][0]["message"].get("content") or "").strip()
    except (KeyError, IndexError) as e:
        raise ProviderError("malformed response") from e
    if not text:
        raise ProviderError("empty response")
    return text


def groq(s: Settings, system: str, user: str, timeout: float) -> str:
    return _openai_compat("https://api.groq.com/openai/v1/chat/completions",
                          s.groq_api_key, s.groq_model, system, user, timeout)


def groq_deltas(s: Settings, system: str, user: str, timeout: float):
    """Yield Groq text chunks. Raises ProviderError so the gateway can fall back."""
    try:
        with httpx.Client(timeout=timeout) as client:
            with client.stream(
                "POST",
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {s.groq_api_key}"},
                json={"model": s.groq_model, "temperature": 0.3, "max_tokens": 512, "stream": True,
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user", "content": user}]},
            ) as r:
                if r.status_code != 200:
                    raise ProviderError(f"HTTP {r.status_code}")
                saw = False
                for line in r.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        delta = json.loads(data)["choices"][0]["delta"].get("content") or ""
                    except (KeyError, IndexError, json.JSONDecodeError) as e:
                        raise ProviderError("malformed response") from e
                    if delta:
                        saw = True
                        yield delta
                if not saw:
                    raise ProviderError("empty response")
    except ProviderError:
        raise
    except httpx.HTTPError as e:
        raise ProviderError(str(e)) from e


def nim(s: Settings, system: str, user: str, timeout: float) -> str:
    return _openai_compat("https://integrate.api.nvidia.com/v1/chat/completions",
                          s.nim_api_key, s.nim_model, system, user, timeout)


def gemini(s: Settings, system: str, user: str, timeout: float) -> str:
    try:
        r = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{s.gemini_model}:generateContent",
            headers={"x-goog-api-key": s.gemini_api_key},
            json={"systemInstruction": {"parts": [{"text": system}]},
                  "contents": [{"role": "user", "parts": [{"text": user}]}],
                  "generationConfig": {"temperature": 0.3, "maxOutputTokens": 512}},
            timeout=timeout,
        )
    except httpx.HTTPError as e:
        raise ProviderError(str(e)) from e
    if r.status_code != 200:
        raise ProviderError(f"HTTP {r.status_code}")
    try:
        parts = r.json()["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError) as e:
        raise ProviderError("empty response") from e
    text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
    if not text:
        raise ProviderError("empty response")
    return text
