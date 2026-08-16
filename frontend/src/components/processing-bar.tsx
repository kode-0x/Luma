"use client";

import { motion, AnimatePresence } from "motion/react";
import type { ProcessingPhase } from "@/hooks/use-document-processor";

interface ProcessingBarProps {
  phase: ProcessingPhase;
  filename: string | null;
  error: string | null;
  chunkCount?: number;
  onRetry?: () => void;
}

const statusLabels: Record<ProcessingPhase, string> = {
  idle: "",
  uploading: "Uploading document...",
  processing: "Processing and indexing...",
  ready: "Ready",
  error: "Processing failed",
};

export function ProcessingBar({ phase, filename, error, chunkCount, onRetry }: ProcessingBarProps) {
  if (phase === "idle") return null;

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={phase}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -8 }}
        transition={{ duration: 0.3, ease: "easeOut" }}
        className="w-full rounded-xl border border-border bg-surface-elevated p-5"
      >
        <div className="flex items-center justify-between">
          <div className="min-w-0 flex-1">
            {filename && (
              <p className="truncate text-sm font-medium text-text-primary">{filename}</p>
            )}
            <div className="mt-1 flex items-center gap-2">
              {(phase === "uploading" || phase === "processing") && (
                <motion.span
                  animate={{ opacity: [0.5, 1, 0.5] }}
                  transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
                  className="inline-block h-1.5 w-1.5 rounded-full bg-text-secondary"
                />
              )}
              {phase === "ready" && (
                <motion.span
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ duration: 0.3, type: "spring", stiffness: 300 }}
                  className="inline-block h-1.5 w-1.5 rounded-full bg-text-primary"
                />
              )}
              {phase === "error" && (
                <span className="inline-block h-1.5 w-1.5 rounded-full bg-text-muted" />
              )}
              <motion.p
                key={statusLabels[phase]}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="text-xs text-text-secondary"
              >
                {phase === "ready" && chunkCount
                  ? `${chunkCount} chunks indexed`
                  : statusLabels[phase]}
              </motion.p>
            </div>
          </div>
          {phase === "error" && onRetry && (
            <motion.button
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={onRetry}
              className="rounded-lg border border-border px-3 py-1.5 text-xs font-medium text-text-primary transition-colors hover:bg-border-subtle"
            >
              Retry
            </motion.button>
          )}
        </div>

        <div className="mt-3 h-1 w-full overflow-hidden rounded-full bg-border-subtle">
          {phase === "uploading" && (
            <motion.div
              initial={{ width: "0%" }}
              animate={{ width: "30%" }}
              transition={{ duration: 0.8, ease: "easeOut" }}
              className="h-full rounded-full bg-text-muted"
            />
          )}
          {phase === "processing" && (
            <motion.div
              initial={{ width: "30%" }}
              animate={{ width: "85%" }}
              transition={{ duration: 8, ease: "linear" }}
              className="h-full rounded-full bg-text-secondary"
            />
          )}
          {phase === "ready" && (
            <motion.div
              initial={{ width: "85%" }}
              animate={{ width: "100%" }}
              transition={{ duration: 0.4, ease: "easeOut" }}
              className="h-full rounded-full bg-text-primary"
            />
          )}
          {phase === "error" && (
            <motion.div
              initial={{ width: "50%" }}
              animate={{ width: "50%", opacity: 0.5 }}
              className="h-full rounded-full bg-text-muted"
            />
          )}
        </div>

        {phase === "error" && error && (
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.15 }}
            className="mt-2 text-xs text-text-muted"
          >
            {error}
          </motion.p>
        )}
      </motion.div>
    </AnimatePresence>
  );
}
