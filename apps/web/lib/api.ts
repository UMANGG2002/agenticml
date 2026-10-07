export type Provider = "anthropic" | "gemini";

export interface DatasetUploadResponse {
  dataset_version_id: string;
  n_rows: number;
  n_columns: number;
  columns: string[];
}

export interface ToolCall {
  name: string;
  input: Record<string, unknown>;
}

export interface PendingQuestion {
  reason: string;
  question: string;
  context: Record<string, unknown>;
}

export interface ExperimentRecord {
  experiment_id: string;
  model: string;
  metric: string;
  primary_score: number;
  hypothesis: string;
  conclusion: string | null;
  parent_experiment_id: string | null;
  [key: string]: unknown;
}

export type RunStatus = "done" | "awaiting_user_input" | "max_turns_reached";

export interface RunStatusResponse {
  run_id: string;
  status: RunStatus;
  tool_calls: ToolCall[];
  pending_question: PendingQuestion | null;
  final_text: string | null;
  experiments: ExperimentRecord[];
}

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function handle<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail = typeof body?.detail === "string" ? body.detail : `request failed with status ${response.status}`;
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export async function uploadDataset(file: File): Promise<DatasetUploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`${API_BASE}/api/datasets`, { method: "POST", body: form });
  return handle<DatasetUploadResponse>(response);
}

export async function startRun(params: {
  dataset_version_id: string;
  objective: string;
  provider: Provider;
}): Promise<RunStatusResponse> {
  const response = await fetch(`${API_BASE}/api/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return handle<RunStatusResponse>(response);
}

export async function answerRun(runId: string, answer: string): Promise<RunStatusResponse> {
  const response = await fetch(`${API_BASE}/api/runs/${runId}/answer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ answer }),
  });
  return handle<RunStatusResponse>(response);
}
