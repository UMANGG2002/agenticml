"use client";

import { motion } from "framer-motion";

interface BudgetMeterProps {
  used: number;
  max: number;
}

export default function BudgetMeter({ used, max }: BudgetMeterProps) {
  const percent = max > 0 ? Math.min((used / max) * 100, 100) : 0;
  const isHigh = percent > 75;
  const isMid = percent > 50;

  const barColor = isHigh
    ? "from-warning to-error"
    : isMid
      ? "from-warning-light to-warning"
      : "from-primary to-accent";

  const glowClass = isHigh ? "glow-error" : "";

  return (
    <div className="glass-card p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-text-secondary">
          Experiment Budget
        </span>
        <span className="text-sm font-bold text-text-primary">
          {used} / {max}
        </span>
      </div>
      <div
        className={`h-2.5 rounded-full bg-white/5 overflow-hidden ${glowClass}`}
      >
        <motion.div
          className={`h-full rounded-full bg-gradient-to-r ${barColor}`}
          initial={{ width: 0 }}
          animate={{ width: `${percent}%` }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        />
      </div>
      {isHigh && (
        <motion.p
          className="text-xs text-warning mt-1.5"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          Budget running low
        </motion.p>
      )}
    </div>
  );
}
