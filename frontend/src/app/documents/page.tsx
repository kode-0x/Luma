"use client";

import { Card } from "@/components/card";
import { FadeIn } from "@/components/fade-in";
import { Button } from "@/components/button";
import { deleteDocument, listDocuments, type DocumentSummary } from "@/lib/api";
import { motion } from "motion/react";
import { useCallback, useEffect, useState } from "react";

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchDocuments = useCallback(async () => {
    try {
      const data = await listDocuments();
      setDocuments(data.documents);
    } catch {
      // API not available
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  const handleDelete = async (id: string) => {
    try {
      await deleteDocument(id);
      setDocuments((prev) => prev.filter((d) => d.id !== id));
    } catch {
      // Silently fail
    }
  };

  const statusBadge = (status: string) => {
    const styles: Record<string, string> = {
      completed: "bg-border-subtle text-text-primary",
      processing: "bg-border-subtle text-text-secondary",
      failed: "bg-border-subtle text-text-muted",
      pending: "bg-border-subtle text-text-muted",
    };
    return (
      <span className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ${styles[status] || styles.pending}`}>
        {status}
      </span>
    );
  };

  return (
    <div>
      <FadeIn>
        <h1 className="text-2xl font-semibold tracking-tight">Documents</h1>
        <p className="mt-1 text-sm text-text-secondary">
          Your uploaded documents and their processing status.
        </p>
      </FadeIn>

      <div className="mt-8 space-y-3">
        {loading ? (
          <FadeIn delay={0.1}>
            <p className="text-sm text-text-muted">Loading...</p>
          </FadeIn>
        ) : documents.length === 0 ? (
          <FadeIn delay={0.1}>
            <div className="flex flex-col items-center py-16 text-center">
              <p className="text-sm text-text-muted">No documents uploaded yet.</p>
              <p className="mt-1 text-xs text-text-muted">
                Upload a document from the home page to get started.
              </p>
            </div>
          </FadeIn>
        ) : (
          documents.map((doc, index) => (
            <motion.div
              key={doc.id}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.35, delay: index * 0.05, ease: "easeOut" }}
            >
              <Card className="flex items-center justify-between">
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-3">
                    <p className="truncate text-sm font-medium text-text-primary">{doc.filename}</p>
                    {statusBadge(doc.status)}
                  </div>
                  <p className="mt-1 text-xs text-text-muted">
                    {doc.file_type.toUpperCase()} &middot; {doc.chunk_count} chunks &middot;{" "}
                    {new Date(doc.created_at).toLocaleDateString()}
                  </p>
                </div>
                <Button variant="secondary" size="sm" onClick={() => handleDelete(doc.id)}>
                  Delete
                </Button>
              </Card>
            </motion.div>
          ))
        )}
      </div>
    </div>
  );
}
