"use client";

import { FadeIn } from "@/components/fade-in";
import { FileUpload } from "@/components/file-upload";
import { ProcessingBar } from "@/components/processing-bar";
import { useDocumentProcessor } from "@/hooks/use-document-processor";
import { AnimatePresence, motion } from "motion/react";
import { useRouter } from "next/navigation";
import { useEffect } from "react";

export default function HomePage() {
  const router = useRouter();
  const processor = useDocumentProcessor();

  useEffect(() => {
    if (processor.phase === "ready") {
      const timeout = setTimeout(() => router.push("/chat"), 1200);
      return () => clearTimeout(timeout);
    }
  }, [processor.phase, router]);

  return (
    <div className="flex flex-col items-center pt-16 md:pt-24">
      <FadeIn>
        <h1 className="text-center text-4xl font-semibold tracking-tight text-text-primary md:text-5xl">
          Evidence-based AI
          <br />
          for your documents.
        </h1>
      </FadeIn>

      <FadeIn delay={0.15}>
        <p className="mt-4 max-w-md text-center text-text-secondary">
          Upload documents, ask questions, and get answers grounded in the original sources with clear citations.
        </p>
      </FadeIn>

      <div className="mt-12 w-full max-w-lg">
        <AnimatePresence mode="wait">
          {processor.phase === "idle" || processor.phase === "error" ? (
            <motion.div
              key="upload"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              transition={{ duration: 0.3 }}
            >
              <FileUpload
                onUpload={processor.upload}
                isUploading={false}
              />
            </motion.div>
          ) : null}
        </AnimatePresence>

        <AnimatePresence>
          {processor.phase !== "idle" && (
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.35 }}
              className={processor.phase === "error" ? "mt-4" : ""}
            >
              <ProcessingBar
                phase={processor.phase}
                filename={processor.filename}
                error={processor.error}
                chunkCount={processor.document?.chunk_count}
                onRetry={processor.reset}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
