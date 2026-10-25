"""Build and run sample-level queries against the live OSDR biodata API.

The response is streamed with a byte cap and an overall deadline, and
successful results are kept in a small in-memory TTL cache, so repeated
questions do not hit NASA's API again. For scale: 10 datasets return about
0.2 MB (540 sample rows), 50 datasets about 1.3 MB (measured 2026-10-24).
"""

import json
import threading
import time
from collections import OrderedDict
from urllib.parse import quote

import requests

from .fetch import BASE
from .schema import QUERY_FIELDS

MAX_ROWS = 500                 # rows returned to the browser
MAX_BYTES = 5_000_000          # stop reading a response larger than this
TIMEOUT_S = (5, 15)            # (connect, read between bytes) for requests
DEADLINE_S = 25                # wall-clock limit for the whole download
CACHE_TTL_S = 600
CACHE_SIZE = 128

_cache = OrderedDict()         # url -> (expires_at, result)
_cache_lock = threading.Lock()


class SampleQueryTooLarge(RuntimeError):
    """The response exceeded MAX_BYTES; the caller should narrow the query."""


def build_url(clean, accessions=None, fmt="json.records"):
    """clean: validated {field: [values]}; accessions: restrict to these datasets."""
    params = []
    if accessions:
        params.append("id.accession=" + "|".join(accessions))
    for field, vals in clean.items():
        column = QUERY_FIELDS[field]
        params.append(quote(column) + "=" + "|".join(quote(v, safe="") for v in vals))
    # always return these columns so the result table is readable
    for field in ("organism", "material_type", "spaceflight", "tissue", "assay_technology"):
        if field not in clean:
            params.append(quote(QUERY_FIELDS[field]))
    params.append(f"format={fmt}")
    return f"{BASE}/query/samples/?" + "&".join(params)


def _download(url, get):
    deadline = time.monotonic() + DEADLINE_S
    with get(url, timeout=TIMEOUT_S, stream=True) as r:
        r.raise_for_status()
        body = bytearray()
        for chunk in r.iter_content(chunk_size=64 * 1024):
            body += chunk
            if len(body) > MAX_BYTES:
                raise SampleQueryTooLarge(f"response over {MAX_BYTES} bytes")
            if time.monotonic() > deadline:
                raise TimeoutError(f"download took over {DEADLINE_S}s")
    return json.loads(body)


def _shape(rows):
    short = {column: field for field, column in QUERY_FIELDS.items()}
    out = []
    for row in rows[:MAX_ROWS]:
        out.append({short.get(k, k.replace("id.", "")): v for k, v in row.items()})
    return {"total": len(rows), "rows": out, "truncated": len(rows) > MAX_ROWS}


def run(url, get=requests.get, now=time.monotonic):
    """Fetch and reshape the sample rows for `url`, from the TTL cache when possible."""
    with _cache_lock:
        hit = _cache.get(url)
        if hit and hit[0] > now():
            _cache.move_to_end(url)
            return _copy(hit[1])
    result = _shape(_download(url, get))
    with _cache_lock:
        _cache[url] = (now() + CACHE_TTL_S, result)
        _cache.move_to_end(url)
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    return _copy(result)


def _copy(result):
    # callers re-sort rows in place; keep the cached copy intact
    return {**result, "rows": list(result["rows"])}


def clear_cache():
    with _cache_lock:
        _cache.clear()
