"""LLM step: model configuration, rewrite validation, and what reaches the browser on failure.

No API key and no network: the Anthropic client is replaced by a fake.
"""

import logging
from types import SimpleNamespace

import pytest
from conftest import dataset, make_engine, sample_row

from osdr_search import search as search_module
from osdr_search.llm import DEFAULT_MODEL, QueryTranslator, StructuredQuery, accept_search_text
from osdr_search.schema import Schema

SCHEMA = Schema(rows=[sample_row("OSD-1", organism="Mus musculus", material_type="Liver")])


class FakeMessages:
    def __init__(self, parsed=None, error=None):
        self.parsed, self.error, self.calls = parsed, error, []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(stop_reason="end_turn", parsed_output=self.parsed)


def fake_client(**kw):
    return SimpleNamespace(messages=FakeMessages(**kw))


def test_default_model_is_small_and_overridable(monkeypatch):
    monkeypatch.delenv("OSDR_LLM_MODEL", raising=False)
    assert DEFAULT_MODEL == "claude-haiku-5-5"
    assert QueryTranslator(SCHEMA, client=fake_client()).model == "claude-haiku-5-5"
    monkeypatch.setenv("OSDR_LLM_MODEL", "claude-sonnet-5-5")
    assert QueryTranslator(SCHEMA, client=fake_client()).model == "claude-sonnet-5-5"


def test_translate_sends_configured_model_and_low_effort():
    client = fake_client(parsed=StructuredQuery(search_text="x", rationale=""))
    QueryTranslator(SCHEMA, model="claude-haiku-5-5", client=client).translate("mouse liver")
    call = client.messages.calls[0]
    assert call["model"] == "claude-haiku-5-5"
    assert call["output_config"] == {"effort": "low"}
    assert call["output_format"] is StructuredQuery


def test_real_client_gets_a_timeout(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    monkeypatch.setenv("OSDR_LLM_TIMEOUT_S", "7")
    t = QueryTranslator(SCHEMA)  # constructing the client makes no request
    assert t.enabled and t.client.timeout == 7.0 and t.client.max_retries == 1


def test_disabled_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    t = QueryTranslator(SCHEMA)
    assert not t.enabled and t.translate("anything") is None


def test_accept_search_text_keeps_on_topic_rewrites():
    q = "how does spaceflight affect mouse liver gene expression?"
    assert accept_search_text(q, "  Mus musculus liver   transcriptome spaceflight ") == \
        "Mus musculus liver transcriptome spaceflight"


def test_accept_search_text_rejects_drift_empty_and_overlong():
    q = "how does spaceflight affect mouse liver gene expression?"
    assert accept_search_text(q, "Arabidopsis root growth under LED light") is None
    assert accept_search_text(q, "") is None
    assert accept_search_text(q, "liver " * 100) is None


def engine_with(translator, datasets=None):
    datasets = datasets or [dataset("OSD-1", "mouse liver"), dataset("OSD-2", "rat soleus")]
    rows = [sample_row("OSD-1", organism="Mus musculus", material_type="Liver"),
            sample_row("OSD-2", organism="Rattus norvegicus", material_type="Soleus")]
    return make_engine(datasets, rows, {"OSD-1": 0.8, "OSD-2": 0.7}, retriever="hybrid", translator=translator)


def test_drifted_rewrite_is_ignored_and_question_still_ranked():
    sq = StructuredQuery(search_text="rat soleus muscle atrophy", rationale="")
    hits, _, _ = engine_with(None).rank("mouse liver", sq, k=2)
    assert hits[0]["accession"] == "OSD-1"
    assert set(hits[0]["ranks"]) == {"semantic", "bm25"}  # no *_rewrite rankers were added


def test_accepted_rewrite_adds_votes_next_to_the_question():
    sq = StructuredQuery(search_text="mouse liver transcriptome", rationale="")
    hits, _, _ = engine_with(None).rank("mouse liver", sq, k=2)
    assert set(hits[0]["ranks"]) == {"semantic", "bm25", "semantic_rewrite", "bm25_rewrite"}


def test_llm_exception_text_never_reaches_the_response(caplog):
    secret = "AuthenticationError: invalid x-api-key sk-ant-SECRET at https://internal.example/v1"
    translator = QueryTranslator(SCHEMA, client=fake_client(error=RuntimeError(secret)))
    with caplog.at_level(logging.ERROR):
        result = engine_with(translator).search("mouse liver", fetch_samples=False)
    assert result["llm_error"] == search_module.LLM_ERROR
    assert "SECRET" not in str(result)
    assert "SECRET" in caplog.text  # still logged server-side
    assert [h["accession"] for h in result["datasets"]][0] == "OSD-1"  # degraded, not failed


def test_sample_query_exception_text_never_reaches_the_response(monkeypatch, caplog):
    def boom(url):
        raise ConnectionError("HTTPSConnectionPool(host='10.0.0.5', port=443): token=abc123")

    monkeypatch.setattr(search_module, "run", boom)
    with caplog.at_level(logging.ERROR):
        result = engine_with(None).search("mouse liver", fetch_samples=True)
    assert result["samples"]["error"] == search_module.SAMPLES_ERROR
    assert "abc123" not in str(result["samples"])
    assert "abc123" in caplog.text


def test_rewrite_votes_count_half():
    sq = StructuredQuery(search_text="mouse liver transcriptome", rationale="")
    plain, _, _ = engine_with(None).rank("mouse liver", None, k=1)
    rewritten, _, _ = engine_with(None).rank("mouse liver", sq, k=1)
    # OSD-1 is first in all four lists: 1/61 per question ranker, 0.5/61 per rewrite ranker
    assert rewritten[0]["score"] - plain[0]["score"] == pytest.approx(1 / 61, abs=2e-5)
