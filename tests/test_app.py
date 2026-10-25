"""HTTP layer: query length, rate limit, CORS default, lifespan and error sanitising. Uses a fake engine."""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from osdr_search.app import MAX_QUERY_CHARS, RateLimiter, create_app


class FakeEngine:
    def __init__(self, fail=False):
        self.fail, self.warmed, self.queries = fail, False, []
        self.datasets = {"OSD-1": {}}
        self.schema = SimpleNamespace(sample_count=3, vocab={"organism": ["Mus musculus"]})
        self.llm_enabled = False

    def warmup(self):
        self.warmed = True

    def search(self, q, k=10, fetch_samples=True):
        if self.fail:
            raise RuntimeError("database password is hunter2")
        self.queries.append(q)
        return {"question": q, "datasets": []}


def client(engine=None, **kw):
    engine = engine or FakeEngine()
    return TestClient(create_app(lambda: engine, **kw), raise_server_exceptions=False), engine


def test_lifespan_builds_and_warms_the_engine():
    c, engine = client()
    with c:
        assert c.get("/api/info").json()["datasets"] == 1
    assert engine.warmed


def test_query_length_is_limited():
    c, engine = client()
    with c:
        assert c.get("/api/search", params={"q": "x" * MAX_QUERY_CHARS}).status_code == 200
        assert c.get("/api/search", params={"q": "x" * (MAX_QUERY_CHARS + 1)}).status_code == 422
        assert c.get("/api/search", params={"q": "x"}).status_code == 422
    assert len(engine.queries) == 1


def test_rate_limit_per_client():
    c, _ = client(rate_limit=2)
    with c:
        codes = [c.get("/api/search", params={"q": "mouse liver"}).status_code for _ in range(3)]
        assert codes == [200, 200, 429]


def test_rate_limiter_window_slides():
    t = [0.0]
    rl = RateLimiter(1, window_s=60, now=lambda: t[0])
    assert rl.allow("a") and not rl.allow("a") and rl.allow("b")
    t[0] = 61.0
    assert rl.allow("a")


def test_no_cors_headers_by_default():
    c, _ = client()
    with c:
        r = c.get("/api/info", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in r.headers


def test_cors_only_for_configured_origins():
    c, _ = client(cors_origins=["https://ok.example"])
    with c:
        ok = c.get("/api/info", headers={"Origin": "https://ok.example"})
        bad = c.get("/api/info", headers={"Origin": "https://evil.example"})
    assert ok.headers.get("access-control-allow-origin") == "https://ok.example"
    assert "access-control-allow-origin" not in bad.headers


def test_unhandled_errors_return_a_generic_500():
    c, _ = client(FakeEngine(fail=True))
    with c:
        r = c.get("/api/search", params={"q": "mouse liver"})
    assert r.status_code == 500
    assert r.json() == {"detail": "internal server error"}
    assert "hunter2" not in r.text
