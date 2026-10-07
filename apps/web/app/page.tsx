"use client";

import { useState } from "react";
import {
  answerRun,
  startRun,
  uploadDataset,
  type ExperimentRecord,
  type Provider,
  type RunStatusResponse,
  type ToolCall,
} from "@/lib/api";

type Phase = "idle" | "uploading" | "running" | "awaiting_answer" | "done" | "error";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [objective, setObjective] = useState("");
  const [provider, setProvider] = useState<Provider>("anthropic");

  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [datasetInfo, setDatasetInfo] = useState<string | null>(null);
  const [toolLog, setToolLog] = useState<ToolCall[]>([]);
  const [pendingQuestion, setPendingQuestion] = useState<RunStatusResponse["pending_question"]>(null);
  const [finalText, setFinalText] = useState<string | null>(null);
  const [experiments, setExperiments] = useState<ExperimentRecord[]>([]);
  const [answer, setAnswer] = useState("");

  function applyStatus(result: RunStatusResponse) {
    setRunId(result.run_id);
    setToolLog((prev) => [...prev, ...result.tool_calls]);
    setExperiments(result.experiments);
    setPendingQuestion(result.pending_question);
    setFinalText(result.final_text);

    if (result.status === "awaiting_user_input") {
      setPhase("awaiting_answer");
    } else if (result.status === "done") {
      setPhase("done");
    } else {
      setPhase("error");
      setError(`Agent stopped without finishing (status: ${result.status}).`);
    }
  }

  async function handleStart() {
    if (!file || !objective.trim()) return;
    setError(null);
    setToolLog([]);
    setExperiments([]);
    setPendingQuestion(null);
    setFinalText(null);
    setPhase("uploading");
    try {
      const uploaded = await uploadDataset(file);
      setDatasetInfo(`${uploaded.n_rows} rows x ${uploaded.n_columns} columns`);
      setPhase("running");
      const result = await startRun({
        dataset_version_id: uploaded.dataset_version_id,
        objective,
        provider,
      });
      applyStatus(result);
    } catch (err) {
      setPhase("error");
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleAnswer() {
    if (!runId || !answer.trim()) return;
    setError(null);
    setPhase("running");
    try {
      const result = await answerRun(runId, answer);
      setAnswer("");
      applyStatus(result);
    } catch (err) {
      setPhase("error");
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  const busy = phase === "uploading" || phase === "running";

  return (
    <main className="mx-auto flex min-h-screen max-w-3xl flex-col gap-8 p-8">
      <header>
        <h1 className="text-2xl font-semibold">Agentic ML Engineer</h1>
        <p className="mt-1 text-sm text-gray-500">
          Upload a dataset, describe the goal, and watch the agent audit it, ask questions when it
          needs to, and run experiments.
        </p>
      </header>

      <section className="flex flex-col gap-3 rounded-lg border border-gray-200 p-4">
        <label className="flex flex-col gap-1 text-sm font-medium">
          Dataset (CSV)
          <input
            type="file"
            accept=".csv"
            disabled={busy}
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            className="rounded border border-gray-300 p-2 text-sm"
          />
        </label>

        <label className="flex flex-col gap-1 text-sm font-medium">
          Objective
          <input
            type="text"
            disabled={busy}
            value={objective}
            onChange={(e) => setObjective(e.target.value)}
            placeholder="predict customer churn"
            className="rounded border border-gray-300 p-2 text-sm"
          />
        </label>

        <label className="flex flex-col gap-1 text-sm font-medium">
          Model provider
          <select
            disabled={busy}
            value={provider}
            onChange={(e) => setProvider(e.target.value as Provider)}
            className="rounded border border-gray-300 p-2 text-sm"
          >
            <option value="anthropic">Anthropic (Claude)</option>
            <option value="gemini">Gemini (via LiteLLM)</option>
          </select>
        </label>

        <button
          onClick={handleStart}
          disabled={busy || !file || !objective.trim()}
          className="mt-2 rounded bg-gray-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
        >
          {phase === "uploading" ? "Uploading…" : phase === "running" ? "Running…" : "Start"}
        </button>

        {datasetInfo && <p className="text-xs text-gray-500">{datasetInfo}</p>}
      </section>

      {error && (
        <div className="rounded border border-red-300 bg-red-50 p-3 text-sm text-red-800">{error}</div>
      )}

      {toolLog.length > 0 && (
        <section className="flex flex-col gap-2">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-gray-500">Agent activity</h2>
          <ul className="flex flex-col gap-1 rounded-lg border border-gray-200 p-3 font-mono text-xs">
            {toolLog.map((call, i) => (
              <li key={i}>
                <span className="font-semibold">{call.name}</span>({JSON.stringify(call.input)})
              </li>
            ))}
          </ul>
        </section>
      )}

      {pendingQuestion && phase === "awaiting_answer" && (
        <section className="flex flex-col gap-2 rounded-lg border border-amber-300 bg-amber-50 p-4">
          <p className="text-sm font-medium text-amber-900">{pendingQuestion.question}</p>
          <p className="text-xs text-amber-700">{pendingQuestion.reason}</p>
          {Object.keys(pendingQuestion.context).length > 0 && (
            <pre className="overflow-x-auto rounded bg-white p-2 text-xs">
              {JSON.stringify(pendingQuestion.context, null, 2)}
            </pre>
          )}
          <div className="flex gap-2">
            <input
              type="text"
              value={answer}
              onChange={(e) => setAnswer(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleAnswer()}
              placeholder="your answer"
              className="flex-1 rounded border border-gray-300 p-2 text-sm"
            />
            <button
              onClick={handleAnswer}
              disabled={!answer.trim()}
              className="rounded bg-gray-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
            >
              Answer
            </button>
          </div>
        </section>
      )}

      {finalText && phase === "done" && (
        <section className="rounded-lg border border-green-300 bg-green-50 p-4 text-sm text-green-900">
          {finalText}
        </section>
      )}

      {experiments.length > 0 && (
        <section className="flex flex-col gap-2">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-gray-500">
            Experiments ({experiments.length})
          </h2>
          <div className="overflow-x-auto rounded-lg border border-gray-200">
            <table className="w-full text-left text-xs">
              <thead className="bg-gray-50">
                <tr>
                  <th className="p-2">id</th>
                  <th className="p-2">model</th>
                  <th className="p-2">hypothesis</th>
                  <th className="p-2">metric</th>
                  <th className="p-2">score</th>
                  <th className="p-2">conclusion</th>
                </tr>
              </thead>
              <tbody>
                {experiments.map((exp) => (
                  <tr key={exp.experiment_id} className="border-t border-gray-100">
                    <td className="p-2 font-mono">{exp.experiment_id}</td>
                    <td className="p-2">{exp.model}</td>
                    <td className="p-2">{exp.hypothesis}</td>
                    <td className="p-2">{exp.metric}</td>
                    <td className="p-2">{exp.primary_score.toFixed(4)}</td>
                    <td className="p-2">{exp.conclusion ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </main>
  );
}
