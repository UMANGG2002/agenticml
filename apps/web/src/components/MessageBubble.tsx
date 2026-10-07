"use client";

import { motion } from "framer-motion";
import type { ChatMessage } from "@/lib/types";

interface MessageBubbleProps {
  message: ChatMessage;
}

export default function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";
  const isSystem = message.role === "system";

  if (isSystem) {
    return (
      <motion.div
        className="flex justify-center my-2"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.3 }}
      >
        <div className="px-4 py-1.5 rounded-full bg-white/5 border border-white/10 text-text-muted text-xs">
          {message.content}
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      className={`flex ${isUser ? "justify-end" : "justify-start"} mb-3`}
      initial={{ opacity: 0, y: 15, scale: 0.97 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
    >
      <div
        className={`max-w-[85%] ${
          isUser
            ? "bg-gradient-to-br from-primary to-secondary text-white rounded-2xl rounded-br-sm"
            : "glass-card text-text-primary rounded-2xl rounded-bl-sm"
        } px-4 py-3`}
      >
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs font-medium opacity-70">
            {isUser ? "You" : "Agent"}
          </span>
          <span className="text-xs opacity-40">
            {message.timestamp.toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            })}
          </span>
        </div>
        <p className="text-sm leading-relaxed whitespace-pre-wrap">
          {message.content}
        </p>
      </div>
    </motion.div>
  );
}
