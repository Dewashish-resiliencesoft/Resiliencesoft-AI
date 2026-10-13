import json
from functools import lru_cache

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class SiteConfig(BaseModel):
    site_id: str
    origins: list[str] = []
    scopes: list[str] = ["public"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", "/etc/secrets/.env"), extra="ignore")

    mongodb_uri: str = ""
    mongodb_db: str = "resiliencesoft_ai"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    vector_index: str = "chunks_vector"
    text_index: str = "chunks_text"

    top_k: int = 5
    min_score: float = 0.55  # vector similarity floor; tune with eval

    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    nim_api_key: str = ""
    nim_model: str = "nvidia/nemotron-3-super-120b-a12b"
    nim_enabled: bool = False
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.1-flash-lite"
    provider_order: str = "groq,nim,gemini"
    provider_timeout: float = 15.0
    overall_deadline: float = 30.0

    sites_json: str = "{}"
    rate_limit_rpm: int = 30
    max_message_chars: int = 1000
    max_history: int = 6
    contact_phone: str = "9109911372"
    contact_email: str = "info@resiliencesoft.com"

    @property
    def sites(self) -> dict[str, SiteConfig]:
        return {k: SiteConfig(**v) for k, v in json.loads(self.sites_json).items()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
