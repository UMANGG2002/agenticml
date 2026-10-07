"use client";

import { useState, useCallback, type DragEvent, type ChangeEvent } from "react";
import { motion, AnimatePresence } from "framer-motion";
import type { DatasetUploadResponse } from "@/lib/types";

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
  dataset: DatasetUploadResponse | null;
  onUpload: (file: File) => void;
  uploading: boolean;
  objective: string;
  onObjectiveChange: (val: string) => void;
  onStart: () => void;
  canStart: boolean;
  status: string;
}

export default function Sidebar({
  collapsed,
  onToggle,
  dataset,
  onUpload,
  uploading,
  objective,
  onObjectiveChange,
  onStart,
  canStart,
  status,
}: SidebarProps) {
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files?.[0];
      if (file && file.name.endsWith(".csv")) {
        onUpload(file);
      }
    },
    [onUpload]
  );

  const handleFileInput = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        onUpload(file);
        e.target.value = "";
      }
    },
    [onUpload]
  );

  return (
    <motion.aside
      className="h-screen flex-shrink-0 border-r border-white/5 bg-dark-bg/80 backdrop-blur-xl flex flex-col overflow-hidden"
      animate={{ width: collapsed ? 60 : 280 }}
      transition={{ duration: 0.3, ease: "easeInOut" }}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-white/5">
        <AnimatePresence>
          {!collapsed && (
            <motion.div
              className="flex items-center gap-2.5 overflow-hidden"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
            >
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary to-secondary flex items-center justify-center flex-shrink-0">
                <svg
                  width="18"
                  height="18"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="text-white"
                >
                  <path d="M12 2a4 4 0 0 1 4 4c0 1.5-.8 2.8-2 3.5v1a1.5 1.5 0 0 0 3 0" />
                  <path d="M12 2a4 4 0 0 0-4 4c0 1.5.8 2.8 2 3.5v1a1.5 1.5 0 0 1-3 0" />
                  <circle cx="12" cy="16" r="2" />
                  <path d="M12 18v4" />
                  <path d="M7 22h10" />
                  <circle cx="7" cy="8" r="1.5" fill="currentColor" />
                  <circle cx="17" cy="8" r="1.5" fill="currentColor" />
                  <path d="M7 8c-2 0-3.5 1-4 3" />
                  <path d="M17 8c2 0 3.5 1 4 3" />
                </svg>
              </div>
              <div>
                <h1 className="text-base font-bold gradient-text whitespace-nowrap">
                  Agentic ML
                </h1>
                <p className="text-[10px] text-text-muted">
                  AI-Powered ML Engineer
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
        <button
          onClick={onToggle}
          className="w-8 h-8 rounded-lg bg-white/5 hover:bg-white/10 flex items-center justify-center text-text-secondary hover:text-text-primary transition-colors flex-shrink-0 cursor-pointer"
        >
          <motion.svg
            width="16"
            height="16"
            viewBox="0 0 16 16"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            animate={{ rotate: collapsed ? 180 : 0 }}
            transition={{ duration: 0.3 }}
          >
            <path d="M10 3L5 8L10 13" />
          </motion.svg>
        </button>
      </div>

      {/* Content (hidden when collapsed) */}
      <AnimatePresence>
        {!collapsed && (
          <motion.div
            className="flex-1 overflow-y-auto p-4 space-y-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            {/* Upload Area */}
            {!dataset ? (
              <div
                className={`drop-zone p-6 text-center cursor-pointer transition-all ${
                  dragOver ? "drag-over" : ""
                }`}
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragOver(true);
                }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
                onClick={() =>
                  document.getElementById("file-input")?.click()
                }
              >
                <input
                  id="file-input"
                  type="file"
                  accept=".csv"
                  className="hidden"
                  onChange={handleFileInput}
                />
                {uploading ? (
                  <div className="space-y-2">
                    <div className="w-8 h-8 mx-auto border-2 border-primary border-t-transparent rounded-full animate-spin" />
                    <p className="text-xs text-text-secondary">
                      Uploading...
                    </p>
                  </div>
                ) : (
                  <>
                    <div className="w-12 h-12 mx-auto mb-3 rounded-xl bg-gradient-to-br from-primary/20 to-secondary/20 flex items-center justify-center">
                      <svg
                        width="24"
                        height="24"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="1.5"
                        className="text-primary-light"
                      >
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                        <polyline points="17 8 12 3 7 8" />
                        <line x1="12" y1="3" x2="12" y2="15" />
                      </svg>
                    </div>
                    <p className="text-sm font-medium text-text-primary mb-1">
                      Drop CSV here
                    </p>
                    <p className="text-xs text-text-muted">
                      or click to browse
                    </p>
                  </>
                )}
              </div>
            ) : (
              /* Dataset Info */
              <motion.div
                className="glass-card p-4"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.3 }}
              >
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-medium text-text-primary flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-success" />
                    Dataset Loaded
                  </h3>
                  <button
                    onClick={() => window.location.reload()}
                    className="text-xs text-text-muted hover:text-text-secondary transition-colors cursor-pointer"
                  >
                    Reset
                  </button>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <span className="text-text-muted">Rows</span>
                    <span className="text-text-primary font-mono">
                      {dataset.n_rows.toLocaleString()}
                    </span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-text-muted">Columns</span>
                    <span className="text-text-primary font-mono">
                      {dataset.n_columns}
                    </span>
                  </div>
                  <div className="pt-2 border-t border-white/5">
                    <p className="text-xs text-text-muted mb-1.5">Columns:</p>
                    <div className="flex flex-wrap gap-1">
                      {dataset.columns.map((col) => (
                        <span
                          key={col}
                          className="px-2 py-0.5 text-[10px] rounded-full bg-primary/10 text-primary-light border border-primary/20"
                        >
                          {col}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            {/* Objective Input */}
            <div className="space-y-2">
              <label className="text-xs font-medium text-text-secondary block">
                Objective
              </label>
              <textarea
                className="w-full h-24 px-3 py-2.5 rounded-xl bg-white/5 border border-white/10 text-sm text-text-primary placeholder-text-muted resize-none focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/20 transition-all"
                placeholder="e.g., Predict house prices using the given features with the best possible accuracy..."
                value={objective}
                onChange={(e) => onObjectiveChange(e.target.value)}
                disabled={status === "running"}
              />
            </div>

            {/* Start Button */}
            <motion.button
              onClick={onStart}
              disabled={!canStart}
              className={`w-full py-3 px-4 rounded-xl text-sm font-semibold transition-all cursor-pointer ${
                canStart
                  ? "bg-gradient-to-r from-primary to-secondary text-white hover:shadow-lg hover:shadow-primary/25 pulse-glow"
                  : "bg-white/5 text-text-muted cursor-not-allowed"
              }`}
              whileHover={canStart ? { scale: 1.02 } : {}}
              whileTap={canStart ? { scale: 0.98 } : {}}
            >
              {status === "running" ? (
                <span className="flex items-center justify-center gap-2">
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Agent Working...
                </span>
              ) : (
                <span className="flex items-center justify-center gap-2">
                  <svg
                    width="16"
                    height="16"
                    viewBox="0 0 24 24"
                    fill="currentColor"
                  >
                    <polygon points="5 3 19 12 5 21 5 3" />
                  </svg>
                  Start Agent
                </span>
              )}
            </motion.button>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.aside>
  );
}
