"""
FastAPI application exposing the RAG-HPO pipeline over HTTP.
"""

from __future__ import annotations

import logging

from anyio import to_thread
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .deps import Settings, get_settings
from .schemas import AnalyzeRequest, AnalyzeResponse
from .services import run_case

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG-HPO Web API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow local frontend development by default.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> dict:
    settings: Settings = get_settings()
    return {
        "status": "ok",
        "meta_path": settings.meta_path,
        "vec_path": settings.vec_path,
        "llm_model": settings.llm_model_name,
        "use_sbert": settings.use_sbert,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    settings: Settings = get_settings()
    try:
        result = await to_thread.run_sync(run_case, request.text, settings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to run analysis")
        raise HTTPException(status_code=500, detail="Internal server error") from exc
    return AnalyzeResponse(**result)
