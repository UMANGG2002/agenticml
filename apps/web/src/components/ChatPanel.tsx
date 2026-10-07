"use client";

import { useEffect, useRef } from "react";
import { AnimatePresence } from "framer-motion";
import type { ChatMessage, ToolCall } from "@/lib/types";
import MessageBubble from "./MessageBubble";
import ToolCallCard from "./ToolCallCard";

interface ChatPanelProps {
  messages: ChatMessage[];
  toolCalls: ToolCall[];
  status: string;
}

export default function ChatPanel({
  messages,
  toolCalls,
  status,
}: ChatPanelProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, toolCalls]);

  return (
    <div className="flex-1 flex flex-col h-full">
      {/* Header */}
      <div className="px-5 py-3 border-b border-white/5 flex items-center gap-3">
        <div className="flex items-center gap-2">
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            className="text-primary-light"
          >
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
          </svg>
          <h2 className="text-sm font-semibold text-text-primary">
            Agent Chat
          </h2>
        </div>
        {status === "running" && (
          <div className="flex items-center gap-1.5 ml-auto">
            <div className="flex gap-0.5">
              <div
                className="w-1.5 h-1.5 rounded-full bg-primary animate-bounce"
                style={{ animationDelay: "0ms" }}
              />
              <div
                className="w-1.5 h-1.5 rounded-full bg-secondary animate-bounce"
                style={{ animationDelay: "150ms" }}
              />
              <div
                className="w-1.5 h-1.5 rounded-full bg-accent animate-bounce"
                style={{ animationDelay: "300ms" }}
              />
            </div>
            <span className="text-xs text-text-muted">Processing</span>
          </div>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-5 py-4 space-y-1">
        {messages.length === 0 && toolCalls.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary/20 to-secondary/20 flex items-center justify-center mb-4">
              <svg
                width="32"
                height="32"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
                className="text-primary-light"
              >
                <path d="M12 2a4 4 0 0 1 4 4c0 1.5-.8 2.8-2 3.5v1" />
                <path d="M12 2a4 4 0 0 0-4 4c0 1.5.8 2.8 2 3.5v1" />
                <circle cx="12" cy="16" r="2" />
                <path d="M12 18v4" />
                <path d="M7 22h10" />
              </svg>
            </div>
            <h3 className="text-lg font-semibold gradient-text mb-2">
              Ready to Go
            </h3>
            <p className="text-sm text-text-muted max-w-xs">
              Upload a CSV dataset, describe your objective, and let the AI
              agent build the best model for you.
            </p>
          </div>
        )}

        <AnimatePresence mode="popLayout">
          {messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}
        </AnimatePresence>

        {/* Tool calls */}
        {toolCalls.length > 0 && (
          <div className="mt-3 space-y-1">
            <p className="text-xs text-text-muted mb-2 flex items-center gap-1.5">
              <svg
                width="12"
                height="12"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                className="text-accent"
              >
                <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" />
              </svg>
              Tool Calls ({toolCalls.length})
            </p>
            {toolCalls.map((tc, i) => (
              <ToolCallCard
                key={`${tc.tool_name}-${i}`}
                toolCall={tc}
                index={i}
              />
            ))}
          </div>
        )}

        <div ref={bottomRef} />
      </div>
    </div>
  );
}
