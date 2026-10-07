"""Request/response models for the agent HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class DatasetUploadResponse(BaseModel):
    dataset_version_id: str
    n_rows: int
    n_columns: int
    columns: list[str]


class StartRunRequest(BaseModel):
    dataset_version_id: str
    objective: str
    model: str | None = None


class AnswerRequest(BaseModel):
    answer: str


class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    tool_calls: list[dict[str, Any]]
    pending_question: dict[str, Any] | None = None
    final_text: str | None = None
    experiments: list[dict[str, Any]]
