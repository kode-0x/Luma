const API_BASE = "/api/v1";

export interface DocumentSummary {
  id: string;
  filename: string;
  file_type: string;
  status: string;
  chunk_count: number;
  created_at: string;
}

export interface DocumentListResponse {
  documents: DocumentSummary[];
  total: number;
}

export interface DocumentUploadResponse {
  document_id: string;
  filename: string;
  status: string;
  message: string;
}

export interface Citation {
  document_id: string;
  filename: string;
  page_number: number | null;
  content: string;
  score: number;
}

export interface ChatResponse {
  answer: string;
  citations: Citation[];
  query: string;
  timestamp: string;
}

export async function uploadDocument(file: File): Promise<DocumentUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE}/documents`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Upload failed");
  }

  return response.json();
}

export async function getDocument(id: string): Promise<DocumentSummary> {
  const response = await fetch(`${API_BASE}/documents/${id}`);
  if (!response.ok) throw new Error("Failed to fetch document");
  return response.json();
}

export async function listDocuments(): Promise<DocumentListResponse> {
  const response = await fetch(`${API_BASE}/documents`);
  if (!response.ok) throw new Error("Failed to fetch documents");
  return response.json();
}

export async function deleteDocument(id: string): Promise<void> {
  const response = await fetch(`${API_BASE}/documents/${id}`, {
    method: "DELETE",
  });
  if (!response.ok) throw new Error("Failed to delete document");
}

export interface OpenRouterModel {
  id: string;
  name: string;
  prompt_price: number;
  completion_price: number;
  is_free: boolean;
}

export async function listModels(): Promise<OpenRouterModel[]> {
  const response = await fetch(`${API_BASE}/chat/models`);
  if (!response.ok) throw new Error("Failed to fetch models");
  return response.json();
}

export async function sendQuery(
  query: string,
  documentIds?: string[],
  model?: string
): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, document_ids: documentIds || [], model: model || null }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Query failed");
  }

  return response.json();
}
