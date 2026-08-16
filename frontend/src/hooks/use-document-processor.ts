"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { getDocument, uploadDocument, type DocumentSummary } from "@/lib/api";

export type ProcessingPhase = "idle" | "uploading" | "processing" | "ready" | "error";

interface ProcessorState {
  phase: ProcessingPhase;
  filename: string | null;
  documentId: string | null;
  document: DocumentSummary | null;
  error: string | null;
}

export function useDocumentProcessor() {
  const [state, setState] = useState<ProcessorState>({
    phase: "idle",
    filename: null,
    documentId: null,
    document: null,
    error: null,
  });

  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
  }, []);

  const pollStatus = useCallback(
    (documentId: string) => {
      stopPolling();
      pollingRef.current = setInterval(async () => {
        try {
          const doc = await getDocument(documentId);
          if (doc.status === "completed") {
            stopPolling();
            setState((prev) => ({
              ...prev,
              phase: "ready",
              document: doc,
            }));
          } else if (doc.status === "failed") {
            stopPolling();
            setState((prev) => ({
              ...prev,
              phase: "error",
              document: doc,
              error: "Document processing failed. Please try again.",
            }));
          }
        } catch {
          stopPolling();
          setState((prev) => ({
            ...prev,
            phase: "error",
            error: "Lost connection while checking status.",
          }));
        }
      }, 2000);
    },
    [stopPolling]
  );

  const upload = useCallback(
    async (file: File) => {
      setState({
        phase: "uploading",
        filename: file.name,
        documentId: null,
        document: null,
        error: null,
      });

      try {
        const result = await uploadDocument(file);
        setState((prev) => ({
          ...prev,
          phase: "processing",
          documentId: result.document_id,
        }));
        pollStatus(result.document_id);
      } catch (err) {
        setState((prev) => ({
          ...prev,
          phase: "error",
          error: err instanceof Error ? err.message : "Upload failed",
        }));
      }
    },
    [pollStatus]
  );

  const reset = useCallback(() => {
    stopPolling();
    setState({
      phase: "idle",
      filename: null,
      documentId: null,
      document: null,
      error: null,
    });
  }, [stopPolling]);

  useEffect(() => {
    return () => stopPolling();
  }, [stopPolling]);

  return {
    ...state,
    upload,
    reset,
    isLocked: state.phase === "uploading" || state.phase === "processing",
  };
}
