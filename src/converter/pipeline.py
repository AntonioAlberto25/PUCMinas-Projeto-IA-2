from __future__ import annotations

import ast
import difflib
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Protocol

from .llm import LLMClient
from .schemas import ChunkResult, ConversionRequest, ConversionResponse, ValidationReport


class ConversionQualityError(Exception):
    """Raised when converted output does not appear meaningfully transformed."""


class RAGRetriever(Protocol):
    def __call__(self, *, query: str, company_name: str | None) -> tuple[str, list[str]]:
        ...


@dataclass
class CodeChunk:
    id: str
    content: str


class ConversionPipeline:
    def __init__(
        self,
        llm_client: LLMClient,
        mode: str = "llm",
        rag_retriever: RAGRetriever | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.mode = mode
        self.rag_retriever = rag_retriever

    def run(self, request: ConversionRequest) -> ConversionResponse:
        source_language = request.source_language or detect_language(request.raw_code)
        normalized = normalize_code(request.raw_code)
        chunks = chunk_code(normalized, source_language)
        rag_context = ""
        rag_sources: list[str] = []

        if request.rag_enabled and self.rag_retriever is not None:
            query = build_rag_query(
                raw_code=normalized,
                source_language=source_language,
                target_language=request.target_language,
                style=request.style.model_dump(),
            )
            rag_context, rag_sources = self.rag_retriever(
                query=query,
                company_name=request.company_name,
            )

        converted_chunks: list[ChunkResult] = []
        converted_context_parts: list[str] = []

        for chunk in chunks:
            raw_converted = self.llm_client.convert_chunk(
                source_language=source_language,
                target_language=request.target_language,
                chunk_id=chunk.id,
                chunk_content=chunk.content,
                accumulated_context="\n\n".join(converted_context_parts),
                enterprise_context=rag_context,
                style_instructions=request.style.model_dump(),
            )
            converted = extract_code_only(raw_converted)
            converted_chunks.append(
                ChunkResult(id=chunk.id, original=chunk.content, converted=converted)
            )
            converted_context_parts.append(converted)

        converted_code = "\n\n".join(item.converted for item in converted_chunks)
        converted_code = fix_import_duplicates(converted_code)

        if source_language.lower() != request.target_language.lower() and not is_meaningfully_changed(
            original=normalized,
            converted=normalize_code(converted_code),
        ):
            raise ConversionQualityError(
                "Conversion rejected: output is effectively identical to input. "
                "Check key/model and try again."
            )

        report = validate_code(converted_code, request.target_language)
        unified_diff = None
        if request.include_diff:
            unified_diff = build_diff(request.raw_code, converted_code)

        return ConversionResponse(
            source_language=source_language,
            target_language=request.target_language,
            converted_code=converted_code,
            diff=unified_diff,
            report=report,
            chunks=converted_chunks,
            rag_context_used=bool(rag_context.strip()),
            rag_sources=rag_sources,
            mode=self.mode,
        )


def build_rag_query(
    *,
    raw_code: str,
    source_language: str,
    target_language: str,
    style: dict[str, object],
) -> str:
    return (
        f"source={source_language}; target={target_language}; "
        f"style={style}; code={raw_code[:3500]}"
    )


def detect_language(code: str) -> str:
    lower_code = code.lower()
    if "def " in lower_code and ":" in code:
        return "python"
    if "public class" in lower_code or "system.out.println" in lower_code:
        return "java"
    if "function " in lower_code or "console.log" in lower_code:
        return "javascript"
    if "#include" in lower_code or "std::" in lower_code:
        return "cpp"
    return "unknown"


def normalize_code(code: str) -> str:
    # Normalize line endings and trim noisy trailing spaces while preserving indentation.
    return "\n".join(line.rstrip() for line in code.replace("\r\n", "\n").split("\n")).strip()


def chunk_code(code: str, source_language: str) -> list[CodeChunk]:
    if source_language == "python":
        return chunk_python_code(code)
    return chunk_generic_code(code)


def chunk_python_code(code: str) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []

    try:
        module = ast.parse(code)
    except SyntaxError:
        return [CodeChunk(id="chunk-1", content=code)]

    lines = code.splitlines()
    for index, node in enumerate(module.body, start=1):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and hasattr(node, "end_lineno"):
            start = node.lineno - 1
            end = node.end_lineno
            snippet = "\n".join(lines[start:end]).strip()
            if snippet:
                chunks.append(CodeChunk(id=f"chunk-{index}", content=snippet))

    if not chunks:
        chunks.append(CodeChunk(id="chunk-1", content=code))

    return chunks


def chunk_generic_code(code: str) -> list[CodeChunk]:
    pattern = re.compile(r"(?=\b(class|function|def|public|private|protected)\b)")
    split_parts = [part.strip() for part in pattern.split(code) if part.strip()]

    if not split_parts:
        return [CodeChunk(id="chunk-1", content=code)]

    chunks: list[CodeChunk] = []
    buffer = ""
    chunk_index = 1

    for part in split_parts:
        candidate = (buffer + "\n" + part).strip() if buffer else part
        if len(candidate) > 1800:
            chunks.append(CodeChunk(id=f"chunk-{chunk_index}", content=buffer.strip()))
            chunk_index += 1
            buffer = part
        else:
            buffer = candidate

    if buffer.strip():
        chunks.append(CodeChunk(id=f"chunk-{chunk_index}", content=buffer.strip()))

    return chunks or [CodeChunk(id="chunk-1", content=code)]


def fix_import_duplicates(code: str) -> str:
    lines = code.splitlines()
    seen: set[str] = set()
    imports: list[str] = []
    rest: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            if stripped not in seen:
                imports.append(line)
                seen.add(stripped)
        else:
            rest.append(line)

    if not imports:
        return code
    return "\n".join(imports + [""] + rest).strip()


def validate_code(code: str, target_language: str) -> ValidationReport:
    warnings: list[str] = []
    syntax_ok = True

    if target_language.lower() == "python":
        try:
            ast.parse(code)
        except SyntaxError as exc:
            syntax_ok = False
            warnings.append(f"Python syntax error at line {exc.lineno}: {exc.msg}")
    else:
        warnings.append("No language-specific syntax validator configured for this target.")

    if len(code.strip()) == 0:
        syntax_ok = False
        warnings.append("Converted output is empty.")

    return ValidationReport(syntax_ok=syntax_ok, warnings=warnings)


def build_diff(original: str, converted: str) -> str:
    explanation = build_diff_explanation(original=original, converted=converted)
    diff_lines = difflib.unified_diff(
        original.splitlines(),
        converted.splitlines(),
        fromfile="original",
        tofile="converted",
        lineterm="",
    )
    diff_text = "\n".join(diff_lines)
    if not diff_text.strip():
        diff_text = "(sem alteracoes textuais detectadas)"
    return explanation + "\n\n" + diff_text


def is_meaningfully_changed(*, original: str, converted: str) -> bool:
    if original != converted:
        return True

    # Ignore whitespace-only differences when deciding if conversion happened.
    compact_original = "".join(original.split())
    compact_converted = "".join(converted.split())
    return compact_original != compact_converted


def extract_code_only(text: str) -> str:
    """Remove wrappers/explanations and keep only likely code content."""
    cleaned = text.strip()
    if not cleaned:
        return cleaned

    fence_match = re.search(r"```(?:[a-zA-Z0-9_+-]*)?\n([\s\S]*?)```", cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    lines = cleaned.splitlines()
    if not lines:
        return cleaned

    first_code_idx = 0
    for idx, line in enumerate(lines):
        if looks_like_code_line(line):
            first_code_idx = idx
            break
    sliced = lines[first_code_idx:]

    last_code_idx = len(sliced) - 1
    for idx in range(len(sliced) - 1, -1, -1):
        if looks_like_code_line(sliced[idx]):
            last_code_idx = idx
            break

    result = "\n".join(sliced[: last_code_idx + 1]).strip()
    return result or cleaned


def looks_like_code_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False

    code_markers = (
        "def ",
        "class ",
        "function ",
        "return ",
        "import ",
        "from ",
        "if ",
        "for ",
        "while ",
        "try",
        "catch",
        "public ",
        "private ",
        "protected ",
        "const ",
        "let ",
        "var ",
        "#include",
    )
    if stripped.startswith(code_markers):
        return True

    if any(token in stripped for token in ("{", "}", ";", "=>", "::", "(", ")", "=")):
        return True

    return False


def build_diff_explanation(*, original: str, converted: str) -> str:
    original_lines = original.splitlines()
    converted_lines = converted.splitlines()

    ratio = SequenceMatcher(a=original, b=converted).ratio()
    similarity_pct = round(ratio * 100, 2)
    changed_line_count = sum(
        1
        for line in difflib.ndiff(original_lines, converted_lines)
        if line.startswith("+ ") or line.startswith("- ")
    )

    summary_lines = [
        "Explicacao do diff:",
        f"- Linhas no original: {len(original_lines)}",
        f"- Linhas no convertido: {len(converted_lines)}",
        f"- Similaridade textual aproximada: {similarity_pct}%",
        f"- Linhas alteradas/adicionadas/removidas: {changed_line_count}",
        "- Abaixo, o diff unificado compara diretamente original e convertido.",
    ]
    return "\n".join(summary_lines)
