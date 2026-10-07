"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import type { ToolCall } from "@/lib/types";

const TOOL_META: Record<
  string,
  { icon: string; label: string; color: string }
> = {
  profile_columns: {
    icon: "🔍",
    label: "Profiling dataset...",
    color: "text-accent",
  },
  detect_target: {
    icon: "🎯",
    label: "Detecting target column...",
    color: "text-primary-light",
  },
  detect_leakage: {
    icon: "⚠️",
    label: "Checking for data leakage...",
    color: "text-warning",
  },
  run_experiment: {
    icon: "🧪",
    label: "Running experiment...",
    color: "text-success",
  },
  analyze_errors: {
    icon: "📊",
    label: "Analyzing errors...",
    color: "text-secondary-light",
  },
  run_robustness_checks: {
    icon: "🛡️",
    label: "Running robustness checks...",
    color: "text-accent-light",
  },
  run_critic: {
    icon: "🔬",
    label: "Running critic review...",
    color: "text-primary",
  },
  ask_user: {
    icon: "❓",
    label: "Asking you a question...",
    color: "text-warning-light",
  },
};

interface ToolCallCardProps {
  toolCall: ToolCall;
  index: number;
}

export default function ToolCallCard({ toolCall, index }: ToolCallCardProps) {
  const [expanded, setExpanded] = useState(false);
  const meta = TOOL_META[toolCall.tool_name] || {
    icon: "⚙️",
    label: toolCall.tool_name,
    color: "text-text-secondary",
  };

  const status = toolCall.status || (toolCall.tool_result ? "success" : "running");

  const hypothesis =
    toolCall.tool_name === "run_experiment" && toolCall.tool_input?.hypothesis
      ? String(toolCall.tool_input.hypothesis)
      : null;

  return (
    <motion.div
      className="glass-card glass-card-hover mb-2 overflow-hidden"
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.4, delay: index * 0.05, ease: "easeOut" }}
    >
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-4 py-3 flex items-center gap-3 text-left cursor-pointer"
      >
        {/* Status indicator */}
        <div className="flex-shrink-0">
          {status === "running" ? (
            <div className="w-2.5 h-2.5 rounded-full bg-accent animate-pulse" />
          ) : status === "error" ? (
            <div className="w-2.5 h-2.5 rounded-full bg-error glow-error" />
          ) : (
            <div className="w-2.5 h-2.5 rounded-full bg-success glow-success" />
          )}
        </div>

        {/* Icon */}
        <span className="text-lg flex-shrink-0">{meta.icon}</span>

        {/* Label */}
        <div className="flex-1 min-w-0">
          <span className={`text-sm font-medium ${meta.color}`}>
            {meta.label}
          </span>
          {hypothesis && (
            <p className="text-xs text-text-muted mt-0.5 truncate">
              Hypothesis: {hypothesis}
            </p>
          )}
        </div>

        {/* Expand indicator */}
        {toolCall.tool_result != null && (
          <motion.span
            className="text-text-muted text-xs flex-shrink-0"
            animate={{ rotate: expanded ? 180 : 0 }}
            transition={{ duration: 0.2 }}
          >
            &#9660;
          </motion.span>
        )}
      </button>

      <AnimatePresence>
        {expanded && toolCall.tool_result != null && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-3 border-t border-white/5">
              <pre className="text-xs text-text-secondary mt-2 overflow-x-auto max-h-48 overflow-y-auto whitespace-pre-wrap break-words">
                {typeof toolCall.tool_result === "string"
                  ? toolCall.tool_result
                  : JSON.stringify(toolCall.tool_result, null, 2)}
              </pre>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
