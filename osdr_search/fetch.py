"""Download and cache OSDR dataset metadata from the public biodata API.

Produces data/datasets.json: a list of dicts, one per OSD dataset, holding the
fields used for embedding and for schema validation.
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

BASE = "https://visualization.osdr.nasa.gov/biodata/api/v2"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASETS_PATH = DATA_DIR / "datasets.json"
RAW_DIR = DATA_DIR / "raw"


def _get(url, retries=4):
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    raise RuntimeError(f"failed after {retries} tries: {url}")


def list_accessions():
    return list(_get(f"{BASE}/datasets/").json().keys())


def fetch_one(acc):
    cache = RAW_DIR / f"{acc}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    r = _get(f"{BASE}/dataset/{acc}/")
    if r is None:
        return None
    payload = r.json()
    cache.write_text(json.dumps(payload))
    return payload


def _as_list(v):
    if v is None or v == "":
        return []
    if isinstance(v, list):
        return [str(x) for x in v if x not in (None, "")]
    return [str(v)]


def _as_str(v):
    return "; ".join(_as_list(v))


def _mission_name(v):
    if isinstance(v, dict):
        return _as_str(v.get("name"))
    if isinstance(v, list):
        return "; ".join(_mission_name(x) for x in v if _mission_name(x))
    return _as_str(v)


def flatten(acc, payload):
    """Pull the fields we care about out of the nested dataset payload."""
    m = payload[acc]["metadata"]
    assays = list((payload[acc].get("assays") or {}).keys())
    assays = [a for a in assays if a != "REST_URL"]
    return {
        "accession": acc,
        "title": _as_str(m.get("study title")),
        "description": _as_str(m.get("study description")),
        "organism": _as_list(m.get("organism")),
        "material_type": _as_list(m.get("material type")),
        "project_type": _as_str(m.get("project type")),
        "flight_program": _as_str(m.get("flight program")),
        "mission": _mission_name(m.get("mission")),
        "factors": _as_list(m.get("study factor name")),
        "assay_technology": _as_list(m.get("study assay technology type")),
        "assay_measurement": _as_list(m.get("study assay measurement type")),
        "platform": _as_list(m.get("study assay technology platform")),
        "assays": assays,
        "release_date": _as_str(m.get("study public release date")),
        "managing_center": _as_str(m.get("managing nasa center")),
    }


# Sample-level columns pulled from /query/samples/. The keys are the canonical
# OSDR field paths; they double as the filterable fields of the structured query.
SAMPLE_COLUMNS = [
    "study.characteristics.organism",
    "study.characteristics.material type",
    "study.factor value.spaceflight",
    "study.factor value.tissue",
    "investigation.study assays.study assay technology type",
    "investigation.study assays.study assay measurement type",
    "investigation.study.comment.mission name",
    "investigation.study.comment.project type",
]
SAMPLES_PATH = DATA_DIR / "samples.json"


def fetch_samples():
    """One query returning every sample with the SAMPLE_COLUMNS as columns."""
    params = "&".join(requests.utils.quote(c) for c in SAMPLE_COLUMNS)
    url = f"{BASE}/query/samples/?{params}&format=json.records"
    rows = _get(url).json()
    SAMPLES_PATH.write_text(json.dumps(rows))
    print(f"wrote {len(rows)} samples to {SAMPLES_PATH}", file=sys.stderr)


def main():
    DATA_DIR.mkdir(exist_ok=True)
    RAW_DIR.mkdir(exist_ok=True)
    accs = list_accessions()
    print(f"{len(accs)} datasets listed", file=sys.stderr)
    rows, failed = [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(fetch_one, a): a for a in accs}
        for i, f in enumerate(as_completed(futs), 1):
            acc = futs[f]
            try:
                payload = f.result()
                if payload and acc in payload:
                    rows.append(flatten(acc, payload))
                else:
                    failed.append(acc)
            except Exception as e:  # noqa: BLE001
                failed.append(acc)
                print(f"{acc}: {e}", file=sys.stderr)
            if i % 50 == 0:
                print(f"{i}/{len(accs)}", file=sys.stderr)
    rows.sort(key=lambda r: int(r["accession"].split("-")[1]))
    DATASETS_PATH.write_text(json.dumps(rows, indent=1))
    print(f"wrote {len(rows)} datasets to {DATASETS_PATH}; {len(failed)} failed: {failed[:10]}", file=sys.stderr)


if __name__ == "__main__":
    if sys.argv[1:] == ["samples"]:
        DATA_DIR.mkdir(exist_ok=True)
        fetch_samples()
    else:
        main()
