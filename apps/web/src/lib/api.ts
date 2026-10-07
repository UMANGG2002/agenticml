import type { DatasetUploadResponse, RunStatusResponse } from "./types";

const API_BASE = "/api";

export async function uploadDataset(
  file: File
): Promise<DatasetUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE}/datasets`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Upload failed");
  }
  return res.json();
}

export async function startRun(
  datasetVersionId: string,
  objective: string,
  model?: string
): Promise<RunStatusResponse> {
  const res = await fetch(`${API_BASE}/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      dataset_version_id: datasetVersionId,
      objective,
      ...(model ? { model } : {}),
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Failed to start run");
  }
  return res.json();
}

export async function answerQuestion(
  runId: string,
  answer: string
): Promise<RunStatusResponse> {
  const res = await fetch(`${API_BASE}/runs/${runId}/answer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ answer }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Failed to send answer");
  }
  return res.json();
}

export async function getRunStatus(
  runId: string
): Promise<RunStatusResponse> {
  const res = await fetch(`${API_BASE}/runs/${runId}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Failed to get run status");
  }
  return res.json();
}
