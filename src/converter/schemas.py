from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class StyleInstructions(BaseModel):
    language: str | None = Field(default=None, description="Natural language for comments or docs.")
    naming: str | None = Field(default=None, description="Naming style, e.g., snake_case or camelCase.")
    patterns: list[str] = Field(default_factory=list, description="Architectural or coding patterns.")


class ConversionRequest(BaseModel):
    raw_code: str = Field(min_length=1)
    target_language: str = Field(min_length=1)
    source_language: str | None = None
    company_name: str | None = None
    rag_enabled: bool = True
    style: StyleInstructions = Field(default_factory=StyleInstructions)
    include_diff: bool = True


class ChunkResult(BaseModel):
    id: str
    original: str
    converted: str


class ValidationReport(BaseModel):
    syntax_ok: bool
    warnings: list[str] = Field(default_factory=list)


class ConversionResponse(BaseModel):
    source_language: str
    target_language: str
    converted_code: str
    diff: str | None = None
    report: ValidationReport
    chunks: list[ChunkResult]
    rag_context_used: bool = False
    rag_sources: list[str] = Field(default_factory=list)
    mode: Literal["llm"]
