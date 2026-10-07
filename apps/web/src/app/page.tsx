"use client";

import { useState, useCallback } from "react";
import { motion } from "framer-motion";
import type {
  DatasetUploadResponse,
  ChatMessage,
  ToolCall,
  Experiment,
} from "@/lib/types";
import { uploadDataset, startRun, answerQuestion } from "@/lib/api";
import Sidebar from "@/components/Sidebar";
import ChatPanel from "@/components/ChatPanel";
import ObservationsPanel from "@/components/ObservationsPanel";

function makeId() {
  return Math.random().toString(36).slice(2, 10);
}

export default function Home() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [dataset, setDataset] = useState<DatasetUploadResponse | null>(null);
  const [uploading, setUploading] = useState(false);
  const [objective, setObjective] = useState("");
  const [runId, setRunId] = useState<string | null>(null);
  const [status, setStatus] = useState<
    "idle" | "running" | "awaiting_user_input" | "completed" | "error"
  >("idle");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [toolCalls, setToolCalls] = useState<ToolCall[]>([]);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  const [finalText, setFinalText] = useState<string | null>(null);
  const [answerInput, setAnswerInput] = useState("");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const addMessage = useCallback(
    (role: ChatMessage["role"], content: string) => {
      setMessages((prev) => [
        ...prev,
        {
          id: makeId(),
          role,
          content,
          timestamp: new Date(),
        },
      ]);
    },
    []
  );

  const handleProcessResponse = useCallback(
    (data: {
      run_id: string;
      status: string;
      tool_calls: ToolCall[];
      pending_question: { question: string } | null;
      final_text: string | null;
      experiments: Experiment[];
    }) => {
      setRunId(data.run_id);
      setToolCalls(data.tool_calls || []);
      setExperiments(data.experiments || []);

      if (data.final_text) {
        setFinalText(data.final_text);
        addMessage("agent", data.final_text);
      }

      if (data.status === "awaiting_user_input" && data.pending_question) {
        setStatus("awaiting_user_input");
        setPendingQuestion(data.pending_question.question);
        addMessage("agent", data.pending_question.question);
      } else if (
        data.status === "completed" ||
        data.status === "done"
      ) {
        setStatus("completed");
        if (!data.final_text) {
          addMessage("system", "Agent run completed.");
        }
      } else if (data.status === "error") {
        setStatus("error");
        addMessage("system", "An error occurred during the run.");
      } else {
        setStatus("completed");
        if (!data.final_text) {
          addMessage("system", "Agent finished processing.");
        }
      }
    },
    [addMessage]
  );

  const handleUpload = useCallback(
    async (file: File) => {
      setUploading(true);
      setErrorMsg(null);
      try {
        const result = await uploadDataset(file);
        setDataset(result);
        addMessage(
          "system",
          `Dataset loaded: ${result.n_rows.toLocaleString()} rows, ${result.n_columns} columns`
        );
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Upload failed";
        setErrorMsg(msg);
        addMessage("system", `Upload error: ${msg}`);
      } finally {
        setUploading(false);
      }
    },
    [addMessage]
  );

  const handleStart = useCallback(async () => {
    if (!dataset || !objective.trim()) return;
    setStatus("running");
    setErrorMsg(null);
    setFinalText(null);
    setToolCalls([]);
    setExperiments([]);
    addMessage("user", objective);
    addMessage("system", "Starting agent run...");

    try {
      const data = await startRun(dataset.dataset_version_id, objective);
      handleProcessResponse(data);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to start run";
      setStatus("error");
      setErrorMsg(msg);
      addMessage("system", `Error: ${msg}`);
    }
  }, [dataset, objective, addMessage, handleProcessResponse]);

  const handleAnswer = useCallback(async () => {
    if (!runId || !answerInput.trim()) return;
    const answer = answerInput.trim();
    setAnswerInput("");
    setStatus("running");
    setPendingQuestion(null);
    addMessage("user", answer);

    try {
      const data = await answerQuestion(runId, answer);
      handleProcessResponse(data);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to send answer";
      setStatus("error");
      setErrorMsg(msg);
      addMessage("system", `Error: ${msg}`);
    }
  }, [runId, answerInput, addMessage, handleProcessResponse]);

  const canStart =
    !!dataset && objective.trim().length > 0 && status !== "running";

  const statusConfig: Record<
    string,
    { color: string; glow: string; label: string }
  > = {
    idle: { color: "bg-text-muted", glow: "", label: "Idle" },
    running: {
      color: "bg-accent",
      glow: "glow-cyan",
      label: "Agent Working...",
    },
    awaiting_user_input: {
      color: "bg-warning",
      glow: "",
      label: "Waiting for your input",
    },
    completed: { color: "bg-success", glow: "glow-success", label: "Completed" },
    error: { color: "bg-error", glow: "glow-error", label: "Error" },
  };
  const currentStatus = statusConfig[status] || statusConfig.idle;

  return (
    <div className="h-screen flex overflow-hidden">
      {/* Sidebar */}
      <Sidebar
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
        dataset={dataset}
        onUpload={handleUpload}
        uploading={uploading}
        objective={objective}
        onObjectiveChange={setObjective}
        onStart={handleStart}
        canStart={canStart}
        status={status}
      />

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Main panels */}
        <div className="flex-1 flex min-h-0">
          {/* Chat Panel - 60% */}
          <div className="flex-[3] min-w-0 flex flex-col">
            <ChatPanel
              messages={messages}
              toolCalls={toolCalls}
              status={status}
            />
          </div>

          {/* Observations Panel - 40% */}
          <div className="flex-[2] min-w-0 hidden lg:flex">
            <ObservationsPanel
              experiments={experiments}
              toolCalls={toolCalls}
              status={status}
              finalText={finalText}
            />
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="border-t border-white/5 bg-dark-bg/60 backdrop-blur-xl px-5 py-3">
          <div className="flex items-center gap-4">
            {/* Status */}
            <div className="flex items-center gap-2 flex-shrink-0">
              <div
                className={`w-2 h-2 rounded-full ${currentStatus.color} ${currentStatus.glow} ${
                  status === "running" ? "animate-pulse" : ""
                }`}
              />
              <span className="text-xs text-text-secondary">
                {currentStatus.label}
              </span>
            </div>

            {/* Error display */}
            {errorMsg && (
              <div className="text-xs text-error truncate max-w-xs">
                {errorMsg}
              </div>
            )}

            {/* Answer input */}
            {status === "awaiting_user_input" && pendingQuestion && (
              <motion.div
                className="flex-1 flex items-center gap-2"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                <div className="flex-1 relative">
                  <input
                    type="text"
                    value={answerInput}
                    onChange={(e) => setAnswerInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        handleAnswer();
                      }
                    }}
                    placeholder="Type your answer..."
                    className="w-full px-4 py-2 rounded-xl bg-white/5 border border-warning/30 text-sm text-text-primary placeholder-text-muted focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/20 transition-all"
                    autoFocus
                  />
                </div>
                <motion.button
                  onClick={handleAnswer}
                  disabled={!answerInput.trim()}
                  className="px-4 py-2 rounded-xl bg-gradient-to-r from-primary to-secondary text-white text-sm font-medium disabled:opacity-40 cursor-pointer hover:shadow-lg hover:shadow-primary/25 transition-shadow"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  Send
                </motion.button>
              </motion.div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
