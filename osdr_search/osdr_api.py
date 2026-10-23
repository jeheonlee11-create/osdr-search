"""Build and run sample-level queries against the live OSDR biodata API."""

from urllib.parse import quote

import requests

from .fetch import BASE
from .schema import QUERY_FIELDS

MAX_ROWS = 500


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


def run(url):
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    rows = r.json()
    short = {column: field for field, column in QUERY_FIELDS.items()}
    out = []
    for row in rows[:MAX_ROWS]:
        o = {}
        for k, v in row.items():
            key = short.get(k, k.replace("id.", ""))
            o[key] = v
        out.append(o)
    return {"total": len(rows), "rows": out, "truncated": len(rows) > MAX_ROWS}
