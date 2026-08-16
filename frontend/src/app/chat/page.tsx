"use client";

import { Button } from "@/components/button";
import { Card } from "@/components/card";
import { FadeIn } from "@/components/fade-in";
import { listDocuments, listModels, sendQuery, type ChatResponse, type DocumentSummary, type OpenRouterModel } from "@/lib/api";
import { motion, AnimatePresence } from "motion/react";
import { useCallback, useEffect, useState } from "react";

export default function ChatPage() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState<ChatResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [docsLoading, setDocsLoading] = useState(true);
  const [models, setModels] = useState<OpenRouterModel[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [modelsOpen, setModelsOpen] = useState(false);

  const hasReady = documents.some((d) => d.status === "completed");
  const hasProcessing = documents.some((d) => d.status === "processing");
  const isLocked = !hasReady || docsLoading;

  const fetchDocuments = useCallback(async () => {
    try {
      const data = await listDocuments();
      setDocuments(data.documents);
    } catch {
      /* API unavailable */
    } finally {
      setDocsLoading(false);
    }
  }, []);

  const fetchModels = useCallback(async () => {
    try {
      const data = await listModels();
      setModels(data);
      if (!selectedModel && data.length > 0) {
        const free = data.find((m) => m.is_free);
        setSelectedModel(free?.id || data[0].id);
      }
    } catch {
      /* Models unavailable */
    }
  }, [selectedModel]);

  useEffect(() => {
    fetchDocuments();
    fetchModels();
  }, [fetchDocuments, fetchModels]);

  useEffect(() => {
    if (!hasProcessing) return;
    const interval = setInterval(fetchDocuments, 3000);
    return () => clearInterval(interval);
  }, [hasProcessing, fetchDocuments]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || loading || isLocked) return;

    setLoading(true);
    setError("");
    setResponse(null);

    try {
      const result = await sendQuery(query.trim(), undefined, selectedModel || undefined);
      setResponse(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const formatPrice = (price: number): string => {
    if (price === 0) return "Free";
    if (price < 0.000001) return "<$0.001/M";
    return `$${(price * 1_000_000).toFixed(2)}/M`;
  };

  const currentModel = models.find((m) => m.id === selectedModel);

  return (
    <div>
      <FadeIn>
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Chat</h1>
            <p className="mt-1 text-sm text-text-secondary">
              Ask questions about your documents and get evidence-backed answers.
            </p>
          </div>
          <div className="relative">
            <button
              onClick={() => setModelsOpen(!modelsOpen)}
              className="flex items-center gap-1.5 rounded-lg border border-border px-3 py-1.5 text-xs font-medium text-text-secondary transition-colors hover:border-text-muted hover:text-text-primary"
            >
              <span className="max-w-[140px] truncate">
                {currentModel?.name || "Select model"}
              </span>
              {currentModel?.is_free && (
                <span className="rounded bg-border-subtle px-1 py-0.5 text-[10px] font-medium text-text-muted">
                  free
                </span>
              )}
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="6 9 12 15 18 9" />
              </svg>
            </button>

            <AnimatePresence>
              {modelsOpen && (
                <>
                  <div className="fixed inset-0 z-40" onClick={() => setModelsOpen(false)} />
                  <motion.div
                    initial={{ opacity: 0, y: -4, scale: 0.98 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -4, scale: 0.98 }}
                    transition={{ duration: 0.15 }}
                    className="absolute right-0 top-full z-50 mt-1 max-h-80 w-80 overflow-y-auto rounded-xl border border-border bg-surface-elevated shadow-sm"
                  >
                    {models.length === 0 ? (
                      <p className="px-4 py-3 text-xs text-text-muted">Loading models...</p>
                    ) : (
                      models.map((model) => (
                        <button
                          key={model.id}
                          onClick={() => {
                            setSelectedModel(model.id);
                            setModelsOpen(false);
                          }}
                          className={`flex w-full items-center justify-between px-4 py-2.5 text-left transition-colors hover:bg-border-subtle ${
                            model.id === selectedModel ? "bg-border-subtle" : ""
                          }`}
                        >
                          <span className="min-w-0 flex-1">
                            <span className="block truncate text-xs font-medium text-text-primary">
                              {model.name}
                            </span>
                            <span className="block truncate text-[10px] text-text-muted">
                              {model.id}
                            </span>
                          </span>
                          <span className={`ml-2 shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium ${
                            model.is_free
                              ? "bg-border-subtle text-text-primary"
                              : "text-text-muted"
                          }`}>
                            {model.is_free ? "free" : formatPrice(model.prompt_price)}
                          </span>
                        </button>
                      ))
                    )}
                  </motion.div>
                </>
              )}
            </AnimatePresence>
          </div>
        </div>
      </FadeIn>

      <AnimatePresence>
        {hasProcessing && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
            className="mt-4 overflow-hidden"
          >
            <div className="flex items-center gap-2 rounded-lg border border-border bg-surface-elevated px-4 py-3">
              <motion.span
                animate={{ opacity: [0.4, 1, 0.4] }}
                transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
                className="inline-block h-1.5 w-1.5 rounded-full bg-text-secondary"
              />
              <p className="text-xs text-text-secondary">
                Documents are being processed. Chat will unlock when ready.
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <FadeIn delay={0.1} className="mt-8">
        <form onSubmit={handleSubmit} className="flex gap-3">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={isLocked ? "Waiting for documents..." : "Ask a question about your documents..."}
            disabled={isLocked}
            className="flex-1 rounded-lg border border-border bg-surface-elevated px-4 py-2.5 text-sm text-text-primary placeholder:text-text-muted transition-colors duration-200 focus:border-text-secondary focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed"
          />
          <Button type="submit" disabled={loading || !query.trim() || isLocked}>
            {loading ? "Thinking..." : "Ask"}
          </Button>
        </form>
      </FadeIn>

      <AnimatePresence>
        {error && (
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="mt-4 text-sm text-text-muted"
          >
            {error}
          </motion.p>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {response && (
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.4, ease: "easeOut" }}
            className="mt-8 space-y-4"
          >
            <Card>
              <p className="text-xs font-medium uppercase tracking-wider text-text-muted">Answer</p>
              <p className="mt-3 text-sm leading-relaxed text-text-primary whitespace-pre-wrap">
                {response.answer}
              </p>
            </Card>

            {response.citations.length > 0 && (() => {
              const seen = new Set<string>();
              const unique = response.citations.filter((c) => {
                const key = `${c.document_id}:${c.page_number}:${c.content}`;
                if (seen.has(key)) return false;
                seen.add(key);
                return true;
              });
              return (
                <div className="space-y-2">
                  <p className="text-xs font-medium uppercase tracking-wider text-text-muted">
                    Sources ({unique.length})
                  </p>
                  {unique.map((citation, index) => (
                    <motion.div
                      key={`${citation.document_id}-${citation.page_number}-${index}`}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.3, delay: index * 0.05 }}
                    >
                      <Card className="!p-4">
                        <div className="flex items-center gap-2">
                          <span className="flex h-5 w-5 items-center justify-center rounded bg-text-primary text-xs text-surface-elevated font-medium">
                            {index + 1}
                          </span>
                          <span className="text-xs font-medium text-text-primary">
                            {citation.filename}
                            {citation.page_number && ` — Page ${citation.page_number}`}
                          </span>
                        </div>
                        <p className="mt-2 text-xs leading-relaxed text-text-secondary line-clamp-3">
                          {citation.content}
                        </p>
                      </Card>
                    </motion.div>
                  ))}
                </div>
              );
            })()}
          </motion.div>
        )}
      </AnimatePresence>

      {!response && !loading && !error && !isLocked && (
        <FadeIn delay={0.2} className="mt-16 text-center">
          <p className="text-sm text-text-muted">
            Your documents are ready. Ask a question to get started.
          </p>
        </FadeIn>
      )}

      {!response && !loading && !error && isLocked && !hasProcessing && !docsLoading && (
        <FadeIn delay={0.2} className="mt-16 text-center">
          <p className="text-sm text-text-muted">
            Upload a document first to start chatting.
          </p>
        </FadeIn>
      )}
    </div>
  );
}
