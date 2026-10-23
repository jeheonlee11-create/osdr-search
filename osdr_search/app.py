"""FastAPI app: a web UI plus a JSON API over the search pipeline."""

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from .search import SearchEngine

STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="OSDR plain-English search")
engine: SearchEngine | None = None


@app.on_event("startup")
def _load():
    global engine
    engine = SearchEngine()


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    return {
        "datasets": len(engine.datasets),
        "samples": engine.schema.sample_count,
        "llm_enabled": engine.llm_enabled,
        "vocab_sizes": {f: len(v) for f, v in engine.schema.vocab.items()},
    }


@app.get("/api/search")
def search(q: str = Query(..., min_length=2), k: int = Query(10, ge=1, le=50), samples: bool = True):
    if not q.strip():
        raise HTTPException(400, "empty query")
    return engine.search(q.strip(), k=k, fetch_samples=samples)


@app.get("/api/schema")
def schema():
    return engine.schema.vocab
