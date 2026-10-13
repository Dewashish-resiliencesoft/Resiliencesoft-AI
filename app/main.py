import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, products
from app.config import get_settings



@asynccontextmanager
async def lifespan(_: FastAPI):
    # warm the embedding model off the request path so the first chat isn't slow
    from app.rag_engine.embed import embed_query
    threading.Thread(target=embed_query, args=("warmup",), daemon=True).start()
    yield


app = FastAPI(title="Resiliencesoft AI", description="Central AI Knowledge Engine", lifespan=lifespan)

origins = sorted({o for site in get_settings().sites.values() for o in site.origins})
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type"])

app.include_router(chat.router, prefix="/api/v1/ai")
app.include_router(products.router, prefix="/api/v1/ai")


@app.get("/health")
def health():
    return {"status": "ok"}
