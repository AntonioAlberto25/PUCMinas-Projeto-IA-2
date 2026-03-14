from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from .config import Settings
from .llm import LLMAuthenticationError, LLMRequestError, OpenAICompatibleLLMClient
from .pipeline import ConversionPipeline, ConversionQualityError
from .rag import PDFRAGIndex
from .schemas import ConversionRequest, ConversionResponse


def build_pipeline(
    settings: Settings,
    *,
    rag_index: PDFRAGIndex,
) -> ConversionPipeline | None:
    if not settings.llm_api_key:
        return None

    def rag_retriever(*, query: str, company_name: str | None) -> tuple[str, list[str]]:
        return rag_index.build_context(query=query, company_name=company_name, top_k=4)

    return ConversionPipeline(
        llm_client=OpenAICompatibleLLMClient(
            api_url=settings.llm_api_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        ),
        mode="llm",
        rag_retriever=rag_retriever,
    )


def create_app() -> FastAPI:
    load_dotenv()
    settings = Settings.from_env()
    app = FastAPI(title="Code Converter API", version="0.1.0")
    project_root = Path(__file__).resolve().parents[2]
    rag_index = PDFRAGIndex(storage_dir=project_root / "data" / "rag")
    pipeline = build_pipeline(settings, rag_index=rag_index)
    static_dir = Path(__file__).parent / "static"

    if static_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(static_dir)), name="assets")

    @app.get("/")
    def home() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "mode": "llm" if pipeline else "disabled",
            "message": "Set LLM_API_KEY to enable conversion." if pipeline is None else "LLM configured.",
        }

    @app.get("/rag/status")
    def rag_status() -> dict[str, int | str]:
        stats = rag_index.status()
        return {
            "status": "ok",
            "documents": stats["documents"],
            "chunks": stats["chunks"],
        }

    @app.post("/rag/upload-pdfs")
    async def rag_upload_pdfs(files: list[UploadFile] = File(...)) -> dict[str, int | list[str]]:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded.")

        uploaded_files: list[str] = []
        total_chunks = 0
        for file in files:
            if not file.filename.lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail=f"Only PDF is supported: {file.filename}")
            content = await file.read()
            stats = rag_index.add_pdf_bytes(filename=file.filename, content=content)
            uploaded_files.append(file.filename)
            total_chunks += int(stats["chunks_added"])

        return {
            "uploaded": len(uploaded_files),
            "chunks_added": total_chunks,
            "files": uploaded_files,
        }

    @app.post("/convert", response_model=ConversionResponse)
    def convert(request: ConversionRequest) -> ConversionResponse:
        if pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="LLM is disabled. Configure a valid LLM_API_KEY and restart the API.",
            )

        try:
            return pipeline.run(request)
        except LLMAuthenticationError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        except LLMRequestError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ConversionQualityError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


app = create_app()
