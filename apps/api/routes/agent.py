"""Endpoints that drive agent/llm_agent.py's loop over HTTP.

In-memory only (no DB yet — see docs/spec.md section 24): the dataset and
experiment registries, and the run sessions, live in module-level state for
the lifetime of the process. Restarting the API loses everything — same
caveat as agent/dataset_registry.py, just not yet backed by Postgres.
"""

from __future__ import annotations

import uuid
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile
from openai import OpenAI, OpenAIError

from agent.dataset_registry import DatasetRegistry
from agent.llm_agent import AgentTurnResult, build_resume_messages, run_agent
from agent.state import AgentState
from apps.api.dependencies import Settings, get_settings
from apps.api.schemas import AnswerRequest, DatasetUploadResponse, RunStatusResponse, StartRunRequest
from experiments.registry import ExperimentRegistry
from tools.data.profile import inspect_dataset

router = APIRouter()

_dataset_registry = DatasetRegistry()
_experiment_registry = ExperimentRegistry()
_runs: dict[str, dict[str, Any]] = {}


def _build_client(settings: Settings) -> OpenAI:
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.litellm_base_url,
    )


def _resolve_model(request: StartRunRequest, settings: Settings) -> str:
    return request.model or settings.agent_model


def _run_agent_safely(*args: Any, **kwargs: Any) -> AgentTurnResult:
    """run_agent(), but provider errors (bad/missing credentials, rate limits,
    network) become a clean 502 instead of an opaque 500."""
    try:
        return run_agent(*args, **kwargs)
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"LLM provider error: {exc}") from exc


def _status_response(run_id: str, result: AgentTurnResult) -> RunStatusResponse:
    return RunStatusResponse(
        run_id=run_id,
        status=result.status,
        tool_calls=result.tool_calls,
        pending_question=result.pending_question.model_dump() if result.pending_question else None,
        final_text=result.final_text,
        experiments=[e.model_dump() for e in _experiment_registry.all()],
    )


@router.post("/datasets", response_model=DatasetUploadResponse)
async def upload_dataset(file: UploadFile) -> DatasetUploadResponse:
    df = pd.read_csv(file.file)
    dataset_version_id = _dataset_registry.register(df)
    inspection = inspect_dataset(df)
    return DatasetUploadResponse(
        dataset_version_id=dataset_version_id,
        n_rows=inspection.n_rows,
        n_columns=inspection.n_columns,
        columns=inspection.columns,
    )


@router.post("/runs", response_model=RunStatusResponse)
def start_run(request: StartRunRequest) -> RunStatusResponse:
    try:
        _dataset_registry.get(request.dataset_version_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    settings = get_settings()
    model = _resolve_model(request, settings)
    client = _build_client(settings)
    state = AgentState(project_id="api-run", dataset_version_id=request.dataset_version_id, objective=request.objective)

    result = _run_agent_safely(client, state, _dataset_registry, _experiment_registry, model=model)

    run_id = f"run_{uuid.uuid4().hex[:12]}"
    _runs[run_id] = {"client": client, "state": state, "result": result, "model": model}
    return _status_response(run_id, result)


@router.post("/runs/{run_id}/answer", response_model=RunStatusResponse)
def answer_run(run_id: str, request: AnswerRequest) -> RunStatusResponse:
    session = _runs.get(run_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"unknown run_id '{run_id}'")
    if session["result"].status != "awaiting_user_input":
        raise HTTPException(
            status_code=400, detail=f"run '{run_id}' is not awaiting input (status={session['result'].status})"
        )

    messages = build_resume_messages(session["result"], request.answer)
    result = _run_agent_safely(
        session["client"],
        session["state"],
        _dataset_registry,
        _experiment_registry,
        messages=messages,
        model=session["model"],
    )
    session["result"] = result
    return _status_response(run_id, result)


@router.get("/runs/{run_id}", response_model=RunStatusResponse)
def get_run(run_id: str) -> RunStatusResponse:
    session = _runs.get(run_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"unknown run_id '{run_id}'")
    return _status_response(run_id, session["result"])
