"""Natural language -> structured OSDR query, via Claude with structured outputs.

The model only sees the real vocabulary for each field, and whatever it
returns is validated again in schema.py before anything reaches the API.

The task is a constrained lookup (copy values from a list), so the default is
a small, fast model at low effort. Configure with environment variables:
OSDR_LLM_MODEL (default claude-haiku-5-5) and OSDR_LLM_TIMEOUT_S (default 15).
"""

import os
from pathlib import Path
from typing import List, Optional

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .bm25 import tokenize
from .schema import QUERY_FIELDS

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DEFAULT_MODEL = "claude-haiku-5-5"
DEFAULT_TIMEOUT_S = 15.0
MAX_SEARCH_TEXT = 300   # characters
MIN_OVERLAP = 1 / 3     # share of the question's content words the rewrite must keep


class StructuredQuery(BaseModel):
    organism: List[str] = Field(default_factory=list, description="Scientific names, e.g. 'Mus musculus'.")
    material_type: List[str] = Field(default_factory=list, description="Tissue / sample material, e.g. 'Liver'.")
    tissue: List[str] = Field(default_factory=list, description="Only when the tissue vocabulary has a match.")
    spaceflight: List[str] = Field(
        default_factory=list, description="Flight condition, e.g. 'Space Flight', 'Ground Control'.")
    assay_technology: List[str] = Field(default_factory=list)
    assay_measurement: List[str] = Field(default_factory=list)
    mission: List[str] = Field(default_factory=list)
    project_type: List[str] = Field(default_factory=list)
    search_text: str = Field(description="A short, specific restatement of the question for semantic search.")
    rationale: str = Field(description="One sentence on why these filters were chosen.")

    def filters(self):
        """Just the filter fields, for schema.validate (search_text/rationale are not filters)."""
        return self.model_dump(include=set(QUERY_FIELDS))


SYSTEM_HEAD = """You translate plain-English questions about NASA space biology data into filters for the
OSDR (Open Science Data Repository) sample metadata API.

Rules:
- Every filter value MUST be copied verbatim from the vocabulary below. If the question's concept
  has no vocabulary entry, leave that field empty - never invent a value.
- Pick the most specific matching values. Use several values in one field when the question is
  ambiguous between them (for example 'Liver' and 'liver' spelt differently, or both 'Space Flight'
  and 'in-flight' for spaceflight samples).
- 'gene expression' / 'transcriptome' -> assay_measurement 'transcription profiling'; RNA-seq ->
  assay_technology 'RNA Sequencing (RNA-Seq)'; microarray -> 'DNA microarray'.
- Common names map to scientific names: mouse -> Mus musculus, rat -> Rattus norvegicus,
  human -> Homo sapiens, fruit fly -> Drosophila melanogaster, thale cress / Arabidopsis ->
  Arabidopsis thaliana, yeast -> Saccharomyces cerevisiae, worm -> Caenorhabditis elegans.
- Only add a spaceflight filter when the question is specifically about flown vs control samples;
  a question like 'how does spaceflight affect X' usually wants the whole study, so leave it empty.
- Leave fields empty rather than guessing. An empty filter set is a valid answer.
"""


def _vocab_block(schema):
    lines = []
    for field in QUERY_FIELDS:
        vals = schema.vocab[field]
        lines.append(f"\n## {field} ({len(vals)} values)\n" + " | ".join(vals))
    return "\n".join(lines)


def accept_search_text(question, search_text):
    """The LLM's rewrite of the question, or None when it is empty, too long or has drifted off topic.

    Drift check: at least a third of the question's content words must survive in
    the rewrite. Synonyms are fine ("mouse liver" -> "Mus musculus liver
    transcriptome" keeps "liver"); an unrelated rewrite is not. The original
    question is always ranked as well, so a rejected rewrite only loses its
    extra vote.
    """
    text = " ".join(str(search_text or "").split())
    if not text or len(text) > MAX_SEARCH_TEXT:
        return None
    q = set(tokenize(question))
    if q and len(q & set(tokenize(text))) / len(q) < MIN_OVERLAP:
        return None
    return text


class QueryTranslator:
    def __init__(self, schema, model=None, timeout=None, client=None):
        self.schema = schema
        self.model = model or os.environ.get("OSDR_LLM_MODEL") or DEFAULT_MODEL
        self.timeout = float(timeout or os.environ.get("OSDR_LLM_TIMEOUT_S") or DEFAULT_TIMEOUT_S)
        self.enabled = client is not None or bool(os.environ.get("ANTHROPIC_API_KEY"))
        if client is None and self.enabled:
            # one retry at most: a search request should fail fast and fall back to retrieval only
            client = anthropic.Anthropic(timeout=self.timeout, max_retries=1)
        self.client = client
        self.system = [
            {"type": "text", "text": SYSTEM_HEAD},
            # the vocabulary is large and identical on every call, so cache it
            {"type": "text", "text": _vocab_block(schema), "cache_control": {"type": "ephemeral"}},
        ]

    def translate(self, question) -> Optional[StructuredQuery]:
        if not self.enabled:
            return None
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=4096,
            system=self.system,
            messages=[{"role": "user", "content": question}],
            output_config={"effort": "low"},
            output_format=StructuredQuery,
        )
        if response.stop_reason == "refusal" or response.parsed_output is None:
            return None
        return response.parsed_output
