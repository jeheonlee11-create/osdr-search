"""Live sample query client, with a fake HTTP layer: byte cap, timeout settings, TTL cache."""

import json

import pytest

from osdr_search import osdr_api


class FakeResponse:
    def __init__(self, body, chunk=10):
        self.body, self.chunk = body, chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        for i in range(0, len(self.body), self.chunk):
            yield self.body[i:i + self.chunk]


class FakeGet:
    def __init__(self, rows):
        self.body = json.dumps(rows).encode()
        self.calls = []

    def __call__(self, url, timeout=None, stream=False):
        self.calls.append({"url": url, "timeout": timeout, "stream": stream})
        return FakeResponse(self.body)


ROWS = [{"id.accession": "OSD-1", osdr_api.QUERY_FIELDS["organism"]: "Mus musculus"}]


@pytest.fixture(autouse=True)
def _fresh_cache():
    osdr_api.clear_cache()
    yield
    osdr_api.clear_cache()


def test_run_streams_with_short_timeouts_and_renames_columns():
    get = FakeGet(ROWS)
    out = osdr_api.run("u1", get=get)
    assert out == {"total": 1, "rows": [{"accession": "OSD-1", "organism": "Mus musculus"}], "truncated": False}
    assert get.calls[0]["stream"] is True
    assert get.calls[0]["timeout"] == osdr_api.TIMEOUT_S and max(osdr_api.TIMEOUT_S) <= 30


def test_oversized_response_is_cut_off(monkeypatch):
    monkeypatch.setattr(osdr_api, "MAX_BYTES", 50)
    with pytest.raises(osdr_api.SampleQueryTooLarge):
        osdr_api.run("u2", get=FakeGet(ROWS * 20))


def test_results_are_cached_until_the_ttl_expires():
    get, t = FakeGet(ROWS), [0.0]
    osdr_api.run("u3", get=get, now=lambda: t[0])
    first = osdr_api.run("u3", get=get, now=lambda: t[0])
    first["rows"].clear()  # callers mutate their copy; the cache must not change
    assert len(get.calls) == 1
    assert osdr_api.run("u3", get=get, now=lambda: t[0])["total"] == 1
    t[0] = osdr_api.CACHE_TTL_S + 1
    osdr_api.run("u3", get=get, now=lambda: t[0])
    assert len(get.calls) == 2
