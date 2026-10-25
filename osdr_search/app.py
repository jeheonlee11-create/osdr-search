"""FastAPI app: a web UI plus a JSON API over the search pipeline.

Settings, all optional environment variables:
  OSDR_MAX_QUERY_CHARS     longest accepted question (default 300)
  OSDR_RATE_LIMIT_PER_MIN  /api/search requests per client IP per minute (default 30; 0 disables)
  OSDR_CORS_ORIGINS        comma-separated origins allowed to call the API from a browser;
                           unset means no CORS headers, i.e. same-origin only
"""

import logging
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

STATIC = Path(__file__).resolve().parent / "static"
MAX_QUERY_CHARS = int(os.environ.get("OSDR_MAX_QUERY_CHARS", "300"))
RATE_LIMIT_PER_MIN = int(os.environ.get("OSDR_RATE_LIMIT_PER_MIN", "30"))

log = logging.getLogger(__name__)


class RateLimiter:
    """Sliding-window limit per key, in memory (one process; resets on restart)."""

    def __init__(self, limit, window_s=60.0, now=time.monotonic):
        self.limit, self.window_s, self.now = limit, window_s, now
        self.hits = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key):
        if self.limit <= 0:
            return True
        t = self.now()
        with self.lock:
            q = self.hits[key]
            while q and q[0] <= t - self.window_s:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(t)
            if len(self.hits) > 10_000:  # forget idle clients so memory stays bounded
                for k in [k for k, v in self.hits.items() if not v or v[-1] <= t - self.window_s]:
                    del self.hits[k]
            return True


def _default_engine():
    from .search import SearchEngine  # imported here so tests can build the app without data/ files

    return SearchEngine()


def create_app(engine_factory=_default_engine, rate_limit=RATE_LIMIT_PER_MIN, cors_origins=None):
    @asynccontextmanager
    async def lifespan(app):
        engine = engine_factory()
        warmup = getattr(engine, "warmup", None)
        if warmup:
            warmup()
        app.state.engine = engine
        yield

    app = FastAPI(title="OSDR plain-English search", lifespan=lifespan)
    limiter = RateLimiter(rate_limit)

    if cors_origins is None:
        cors_origins = [o.strip() for o in os.environ.get("OSDR_CORS_ORIGINS", "").split(",") if o.strip()]
    if cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_methods=["GET"])

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.error("unhandled error on %s", request.url.path, exc_info=exc)
        return JSONResponse({"detail": "internal server error"}, status_code=500)

    @app.get("/")
    def home():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/info")
    def info(request: Request):
        engine = request.app.state.engine
        return {
            "datasets": len(engine.datasets),
            "samples": engine.schema.sample_count,
            "llm_enabled": engine.llm_enabled,
            "max_query_chars": MAX_QUERY_CHARS,
            "vocab_sizes": {f: len(v) for f, v in engine.schema.vocab.items()},
        }

    @app.get("/api/search")
    def search(request: Request,
               q: str = Query(..., min_length=2, max_length=MAX_QUERY_CHARS),
               k: int = Query(10, ge=1, le=50),
               samples: bool = True):
        client = request.client.host if request.client else "unknown"
        if not limiter.allow(client):
            raise HTTPException(429, "too many requests; try again in a minute", headers={"Retry-After": "60"})
        if not q.strip():
            raise HTTPException(400, "empty query")
        return request.app.state.engine.search(q.strip(), k=k, fetch_samples=samples)

    @app.get("/api/schema")
    def schema(request: Request):
        return request.app.state.engine.schema.vocab

    return app


app = create_app()
