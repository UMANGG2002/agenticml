"use client";

import { motion, AnimatePresence } from "framer-motion";
import type { Experiment, ToolCall } from "@/lib/types";
import ExperimentChart from "./ExperimentChart";
import BudgetMeter from "./BudgetMeter";

interface ObservationsPanelProps {
  experiments: Experiment[];
  toolCalls: ToolCall[];
  status: string;
  finalText: string | null;
}

function getQualityInsights(toolCalls: ToolCall[]) {
  const insights: {
    leakage: "pending" | "clean" | "warning" | "unknown";
    robustness: "pending" | "pass" | "fail" | "unknown";
    critic: "pending" | "pass" | "fail" | "unknown";
    currentAction: string | null;
  } = {
    leakage: "unknown",
    robustness: "unknown",
    critic: "unknown",
    currentAction: null,
  };

  for (const tc of toolCalls) {
    if (tc.tool_name === "detect_leakage") {
      if (!tc.tool_result) {
        insights.leakage = "pending";
      } else {
        const res =
          typeof tc.tool_result === "string"
            ? tc.tool_result
            : JSON.stringify(tc.tool_result);
        insights.leakage =
          res.toLowerCase().includes("no leakage") ||
          res.toLowerCase().includes("no data leakage")
            ? "clean"
            : "warning";
      }
    }
    if (tc.tool_name === "run_robustness_checks") {
      if (!tc.tool_result) {
        insights.robustness = "pending";
      } else {
        insights.robustness = "pass";
      }
    }
    if (tc.tool_name === "run_critic") {
      if (!tc.tool_result) {
        insights.critic = "pending";
      } else {
        insights.critic = "pass";
      }
    }

    // Track current action
    if (!tc.tool_result) {
      const toolLabels: Record<string, string> = {
        profile_columns: "Profiling columns",
        detect_target: "Detecting target",
        detect_leakage: "Checking leakage",
        run_experiment: "Running experiment",
        analyze_errors: "Analyzing errors",
        run_robustness_checks: "Checking robustness",
        run_critic: "Running critic",
        ask_user: "Waiting for input",
      };
      insights.currentAction =
        toolLabels[tc.tool_name] || tc.tool_name;
    }
  }

  return insights;
}

const statusColors: Record<string, { bg: string; text: string; label: string }> = {
  unknown: { bg: "bg-white/5", text: "text-text-muted", label: "Pending" },
  pending: { bg: "bg-accent/10", text: "text-accent", label: "Running..." },
  clean: { bg: "bg-success/10", text: "text-success", label: "Clean" },
  pass: { bg: "bg-success/10", text: "text-success", label: "Passed" },
  warning: { bg: "bg-warning/10", text: "text-warning", label: "Warning" },
  fail: { bg: "bg-error/10", text: "text-error", label: "Failed" },
};

export default function ObservationsPanel({
  experiments,
  toolCalls,
  status,
  finalText,
}: ObservationsPanelProps) {
  const insights = getQualityInsights(toolCalls);
  const bestExperiment =
    experiments.length > 0
      ? experiments.reduce((best, e) =>
          e.metric_value > best.metric_value ? e : best
        )
      : null;

  const experimentBudget = 10; // default max

  return (
    <div className="w-full h-full flex flex-col border-l border-white/5">
      {/* Header */}
      <div className="px-5 py-3 border-b border-white/5 flex items-center gap-2">
        <svg
          width="18"
          height="18"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          className="text-accent"
        >
          <path d="M2 12h5l3-9 4 18 3-9h5" />
        </svg>
        <h2 className="text-sm font-semibold text-text-primary">
          Observations
        </h2>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Current Action */}
        <AnimatePresence mode="wait">
          {status === "running" && insights.currentAction && (
            <motion.div
              key={insights.currentAction}
              className="glass-card p-3 border-l-2 border-accent"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
            >
              <p className="text-xs text-text-muted mb-0.5">Currently</p>
              <p className="text-sm font-medium text-accent">
                {insights.currentAction}
              </p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Experiment Stats */}
        <motion.div
          className="glass-card p-4"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <h3 className="text-sm font-medium text-text-secondary mb-3">
            Experiments
          </h3>
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-white/5 rounded-lg p-3 text-center">
              <motion.p
                className="text-2xl font-bold gradient-text"
                key={experiments.length}
                initial={{ scale: 1.2 }}
                animate={{ scale: 1 }}
                transition={{ type: "spring", stiffness: 300 }}
              >
                {experiments.length}
              </motion.p>
              <p className="text-[10px] text-text-muted mt-0.5">Total Runs</p>
            </div>
            <div className="bg-white/5 rounded-lg p-3 text-center">
              <motion.p
                className="text-2xl font-bold text-success"
                key={bestExperiment?.metric_value || 0}
                initial={{ scale: 1.2 }}
                animate={{ scale: 1 }}
                transition={{ type: "spring", stiffness: 300 }}
              >
                {bestExperiment
                  ? bestExperiment.metric_value.toFixed(3)
                  : "---"}
              </motion.p>
              <p className="text-[10px] text-text-muted mt-0.5">Best Score</p>
            </div>
          </div>
          {bestExperiment && (
            <div className="mt-3 p-2 rounded-lg bg-success/5 border border-success/10">
              <p className="text-[10px] text-text-muted">Best Model</p>
              <p className="text-xs font-medium text-success">
                {bestExperiment.model_type}
              </p>
              {bestExperiment.hypothesis && (
                <p className="text-[10px] text-text-muted mt-1 line-clamp-2">
                  {bestExperiment.hypothesis}
                </p>
              )}
            </div>
          )}
        </motion.div>

        {/* Experiment Chart */}
        <ExperimentChart experiments={experiments} />

        {/* Budget Meter */}
        <BudgetMeter used={experiments.length} max={experimentBudget} />

        {/* Quality Dashboard */}
        <motion.div
          className="glass-card p-4"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.1 }}
        >
          <h3 className="text-sm font-medium text-text-secondary mb-3">
            Quality Checks
          </h3>
          <div className="space-y-2">
            {[
              { label: "Data Leakage", status: insights.leakage, icon: "⚠️" },
              {
                label: "Robustness",
                status: insights.robustness,
                icon: "🛡️",
              },
              {
                label: "Critic Review",
                status: insights.critic,
                icon: "🔬",
              },
            ].map((item) => {
              const colors = statusColors[item.status] || statusColors.unknown;
              return (
                <div
                  key={item.label}
                  className="flex items-center justify-between p-2 rounded-lg bg-white/[0.02]"
                >
                  <span className="text-xs text-text-secondary flex items-center gap-2">
                    <span>{item.icon}</span>
                    {item.label}
                  </span>
                  <span
                    className={`text-[10px] font-medium px-2 py-0.5 rounded-full ${colors.bg} ${colors.text}`}
                  >
                    {item.status === "pending" ? (
                      <span className="flex items-center gap-1">
                        <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse" />
                        {colors.label}
                      </span>
                    ) : (
                      colors.label
                    )}
                  </span>
                </div>
              );
            })}
          </div>
        </motion.div>

        {/* Final Summary */}
        <AnimatePresence>
          {finalText && (
            <motion.div
              className="glass-card p-4 border-l-2 border-success"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <h3 className="text-sm font-medium text-success mb-2 flex items-center gap-2">
                <svg
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  className="text-success"
                >
                  <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                  <polyline points="22 4 12 14.01 9 11.01" />
                </svg>
                Agent Summary
              </h3>
              <p className="text-xs text-text-secondary leading-relaxed whitespace-pre-wrap">
                {finalText}
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
