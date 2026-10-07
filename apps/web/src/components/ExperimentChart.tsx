"use client";

import { motion } from "framer-motion";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import type { Experiment } from "@/lib/types";

interface ExperimentChartProps {
  experiments: Experiment[];
}

export default function ExperimentChart({ experiments }: ExperimentChartProps) {
  if (experiments.length === 0) {
    return (
      <div className="glass-card p-4">
        <h3 className="text-sm font-medium text-text-secondary mb-3">
          Experiment Scores
        </h3>
        <div className="flex items-center justify-center h-32 text-text-muted text-sm">
          No experiments yet
        </div>
      </div>
    );
  }

  const data = experiments.map((exp, i) => ({
    name: `#${i + 1}`,
    score: typeof exp.metric_value === "number" ? Number(exp.metric_value.toFixed(4)) : 0,
    model: exp.model_type || "unknown",
    hypothesis: exp.hypothesis || "",
  }));

  const bestScore = Math.max(...data.map((d) => d.score));
  const bestIdx = data.findIndex((d) => d.score === bestScore);

  return (
    <motion.div
      className="glass-card p-4"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-text-secondary">
          Experiment Scores
        </h3>
        <div className="flex items-center gap-2">
          <span className="text-xs text-text-muted">Best:</span>
          <motion.span
            className="text-sm font-bold text-success"
            key={bestScore}
            initial={{ scale: 1.3 }}
            animate={{ scale: 1 }}
            transition={{ type: "spring", stiffness: 300 }}
          >
            {bestScore.toFixed(4)}
          </motion.span>
        </div>
      </div>
      <div className="h-40">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 5, right: 5, bottom: 5, left: -10 }}>
            <XAxis
              dataKey="name"
              tick={{ fill: "#94A3B8", fontSize: 11 }}
              axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
              tickLine={false}
            />
            <YAxis
              tick={{ fill: "#94A3B8", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              domain={[0, 1]}
            />
            <Tooltip
              contentStyle={{
                background: "rgba(15, 23, 42, 0.95)",
                border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: "8px",
                color: "#fff",
                fontSize: "12px",
              }}
              formatter={(value: unknown) => [
                `${Number(value).toFixed(4)}`,
                "Score",
              ]}
            />
            <Bar dataKey="score" radius={[4, 4, 0, 0]}>
              {data.map((_entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={
                    index === bestIdx
                      ? "#10B981"
                      : index === data.length - 1
                        ? "#6366F1"
                        : "rgba(99, 102, 241, 0.4)"
                  }
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
}
