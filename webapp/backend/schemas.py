"""
Pydantic models for request and response payloads.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Clinical note content to analyse.")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "患者女性，17岁，原发性闭经，自诉嗅觉减退……",
            }
        }


class PhenotypeResult(BaseModel):
    phrase: str
    category: str
    hpo_id: str
    translation: Optional[str] = None
    keep: bool = True


class AnalyzeResponse(BaseModel):
    patient_id: int = 1
    phenotypes: List[PhenotypeResult]
    runtime_seconds: float
    raw_entries: Optional[list] = None
