"""Pydantic models shared by the pipeline, the app and the evaluation harness."""
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """A verbatim quote from a context chunk."""

    chunk_id: str
    quote: str


class Claim(BaseModel):
    """An atomic statement backed by one or more quotes."""

    text: str
    evidence: List[Evidence] = Field(default_factory=list)


class Highlights(BaseModel):
    """Graph elements to highlight for an answer."""

    nodes: List[str] = Field(default_factory=list)
    edges: List[str] = Field(default_factory=list)
    communities: List[str] = Field(default_factory=list)


class Source(BaseModel):
    """A source chunk with its document name and an excerpt."""

    doc: str
    chunk_id: str
    text: str


class Response(BaseModel):
    """Full pipeline output for one question."""

    route: Literal["global", "local", "abstain", "basic"]
    answer: str
    abstained: bool = False
    abstain_reason: Optional[str] = None
    claims: List[Claim] = Field(default_factory=list)
    highlights: Highlights = Field(default_factory=Highlights)
    sources: List[Source] = Field(default_factory=list)
    timings_ms: Dict[str, int] = Field(default_factory=dict)
    router: Dict[str, object] = Field(default_factory=dict)
    search_answer: str = ""
    stats: Dict[str, int] = Field(default_factory=dict)
