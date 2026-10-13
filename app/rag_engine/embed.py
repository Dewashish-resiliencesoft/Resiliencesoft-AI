from functools import lru_cache

from app.config import get_settings


@lru_cache
def _model():
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=get_settings().embed_model)


def embed_documents(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in _model().embed(texts)]


def embed_query(text: str) -> list[float]:
    # bge models: query instruction improves retrieval
    q = "Represent this sentence for searching relevant passages: " + text
    return next(iter(_model().embed([q]))).tolist()
