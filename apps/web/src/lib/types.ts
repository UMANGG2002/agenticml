export interface DatasetUploadResponse {
  dataset_version_id: string;
  n_rows: number;
  n_columns: number;
  columns: string[];
}

export interface ToolCall {
  tool_name: string;
  tool_input: Record<string, unknown>;
  tool_result?: unknown;
  status?: "running" | "success" | "error";
}

export interface Experiment {
  experiment_id: string;
  hypothesis: string;
  model_type: string;
  features: string[];
  metric_name: string;
  metric_value: number;
  timestamp?: string;
  [key: string]: unknown;
}

export interface RunStatusResponse {
  run_id: string;
  status: string;
  tool_calls: ToolCall[];
  pending_question: { question: string; [key: string]: unknown } | null;
  final_text: string | null;
  experiments: Experiment[];
}

export interface ChatMessage {
  id: string;
  role: "user" | "agent" | "system";
  content: string;
  timestamp: Date;
  toolCalls?: ToolCall[];
}

export interface AppState {
  dataset: DatasetUploadResponse | null;
  objective: string;
  runId: string | null;
  status: "idle" | "running" | "awaiting_user_input" | "completed" | "error";
  messages: ChatMessage[];
  experiments: Experiment[];
  currentToolCalls: ToolCall[];
  pendingQuestion: string | null;
  finalText: string | null;
}
