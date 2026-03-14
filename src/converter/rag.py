from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from io import BytesIO
from pathlib import Path
from threading import Lock

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RagChunk:
    id: str
    source_file: str
    text: str


class PDFRAGIndex:
    def __init__(self, storage_dir: Path) -> None:
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_dir / "chunks.json"
        self._chunks: list[RagChunk] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._lock = Lock()
        self._load()

    def status(self) -> dict[str, int]:
        files = {chunk.source_file for chunk in self._chunks}
        return {
            "documents": len(files),
            "chunks": len(self._chunks),
        }

    def add_pdf_bytes(self, filename: str, content: bytes) -> dict[str, int]:
        text = self._extract_pdf_text(content)
        normalized = normalize_text(text)
        chunks = chunk_text(normalized)
        if not chunks:
            return {"chunks_added": 0}

        with self._lock:
            # Remove previous chunks from same source to keep latest upload version.
            self._chunks = [chunk for chunk in self._chunks if chunk.source_file != filename]

            base = len(self._chunks)
            for idx, item in enumerate(chunks, start=1):
                self._chunks.append(
                    RagChunk(
                        id=f"chunk-{base + idx}",
                        source_file=filename,
                        text=item,
                    )
                )

            self._rebuild_index()
            self._save()

        return {"chunks_added": len(chunks)}

    def build_context(self, *, query: str, company_name: str | None, top_k: int = 4) -> tuple[str, list[str]]:
        hits = self.search(query=query, company_name=company_name, top_k=top_k)
        if not hits:
            return "", []

        context_parts: list[str] = []
        sources: list[str] = []
        for hit in hits:
            source_label = hit["source_file"]
            sources.append(source_label)
            context_parts.append(f"[Fonte: {source_label}]\n{hit['text']}")

        unique_sources = list(dict.fromkeys(sources))
        context = "\n\n".join(context_parts)
        return context[:12000], unique_sources

    def search(self, *, query: str, company_name: str | None, top_k: int = 4) -> list[dict[str, object]]:
        if not query.strip() or not self._chunks or self._vectorizer is None or self._matrix is None:
            return []

        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix).flatten()

        candidate_rows: list[tuple[float, RagChunk]] = []
        company = (company_name or "").strip().lower()
        for score, chunk in zip(scores, self._chunks):
            boosted_score = float(score)
            if company and company in chunk.text.lower():
                boosted_score += 0.2
            candidate_rows.append((boosted_score, chunk))

        candidate_rows.sort(key=lambda row: row[0], reverse=True)

        hits: list[dict[str, object]] = []
        for score, chunk in candidate_rows[:top_k]:
            if score <= 0:
                continue
            hits.append(
                {
                    "id": chunk.id,
                    "source_file": chunk.source_file,
                    "text": chunk.text,
                    "score": round(score, 5),
                }
            )
        return hits

    def _extract_pdf_text(self, content: bytes) -> str:
        reader = PdfReader(BytesIO(content))
        pages: list[str] = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)

    def _rebuild_index(self) -> None:
        if not self._chunks:
            self._vectorizer = None
            self._matrix = None
            return

        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        corpus = [chunk.text for chunk in self._chunks]
        self._matrix = self._vectorizer.fit_transform(corpus)

    def _save(self) -> None:
        data = [asdict(chunk) for chunk in self._chunks]
        self.index_file.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if not self.index_file.exists():
            return

        raw = json.loads(self.index_file.read_text(encoding="utf-8"))
        self._chunks = [RagChunk(**item) for item in raw]
        self._rebuild_index()


def normalize_text(text: str) -> str:
    without_noise = re.sub(r"\s+", " ", text).strip()
    return without_noise


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 220) -> list[str]:
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + max_chars, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(end - overlap, start + 1)

    return chunks
