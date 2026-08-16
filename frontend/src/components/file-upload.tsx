"use client";

import { motion } from "motion/react";
import { useCallback, useRef, useState } from "react";

interface FileUploadProps {
  onUpload: (file: File) => void;
  isUploading?: boolean;
  disabled?: boolean;
  accept?: string;
}

export function FileUpload({
  onUpload,
  isUploading = false,
  disabled = false,
  accept = ".pdf,.docx,.txt,.md,.csv",
}: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const locked = isUploading || disabled;

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      if (locked) return;
      const file = e.dataTransfer.files[0];
      if (file) {
        setSelectedFile(file);
        onUpload(file);
      }
    },
    [onUpload, locked]
  );

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (locked) return;
      const file = e.target.files?.[0];
      if (file) {
        setSelectedFile(file);
        onUpload(file);
      }
    },
    [onUpload, locked]
  );

  const handleClick = () => {
    if (!locked) {
      inputRef.current?.click();
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.1 }}
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      onClick={handleClick}
      className={`relative flex flex-col items-center justify-center rounded-xl border-2 border-dashed px-8 py-12 transition-colors duration-200 ${
        locked
          ? "border-border opacity-50 cursor-not-allowed"
          : isDragging
            ? "border-text-secondary bg-border-subtle cursor-pointer"
            : "border-border hover:border-text-muted cursor-pointer"
      }`}
    >
      <div className="mb-4 text-text-muted">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="17 8 12 3 7 8" />
          <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
      </div>
      <p className="mb-1 text-sm font-medium text-text-primary">
        {isUploading ? "Uploading..." : "Drop a document here"}
      </p>
      <p className="mb-4 text-xs text-text-muted">PDF, DOCX, TXT, Markdown, or CSV</p>
      <span className="inline-flex items-center justify-center rounded-lg border border-border px-3 py-1.5 text-xs font-medium text-text-primary transition-colors duration-200 hover:bg-border-subtle">
        Choose file
      </span>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        onChange={handleFileChange}
        className="hidden"
        disabled={locked}
      />
      {selectedFile && !isUploading && (
        <p className="mt-3 text-xs text-text-secondary">{selectedFile.name}</p>
      )}
    </motion.div>
  );
}
