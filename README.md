# Luma

Evidence-based AI for your documents.

Luma is a document intelligence platform that lets you upload documents, ask natural language questions, and get answers grounded in the original sources with inline citations. It uses a full RAG (Retrieval-Augmented Generation) pipeline with hybrid search, cross-encoder reranking, and configurable LLM generation via OpenRouter.

---

## How It Works

![How-It-Works](assets\How-It-Works.png)

1. Documents are parsed, split into overlapping chunks, embedded, and stored in a vector database.
2. Queries run through hybrid retrieval (semantic + BM25), fused with Reciprocal Rank Fusion, and reranked with a cross-encoder.
3. The top evidence is passed to an LLM which generates an answer with citations pointing to the source document and page.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Python 3.11+, Pydantic |
| Frontend | Next.js 15, TypeScript, Tailwind CSS 4, Motion |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Reranking | Cross-encoder (ms-marco-MiniLM-L-6-v2) |
| Vector DB | Qdrant |
| Lexical Search | BM25 (rank-bm25) |
| LLM | OpenRouter (Any Model) |
| Logging | structlog |

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- Qdrant instance (local or cloud)
- OpenRouter API key ([get one free](https://openrouter.ai/keys))

### Setup

```bash
# Clone
git clone https://github.com/<your-username>/luma.git
cd luma

# Backend
uv sync              # or: pip install -e ".[dev]"
cp .env.example .env # Edit with your keys

# Frontend
cd frontend
npm install
```

### Environment Variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=sk-or-v1-...
QDRANT_URL=http://localhost:6333
```

All other variables have sensible defaults. See `backend/core/config.py` for the full list.

### Run

```bash
# Terminal 1: Backend
uv run uvicorn backend.main:app --reload

# Terminal 2: Frontend
cd frontend
npm run dev
```

The frontend runs on `http://localhost:3000` and proxies API requests to the backend at `http://localhost:8000`.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/documents` | Upload a document |
| `GET` | `/api/v1/documents` | List all documents |
| `GET` | `/api/v1/documents/{id}` | Get document details |
| `DELETE` | `/api/v1/documents/{id}` | Delete a document |
| `POST` | `/api/v1/chat` | Query documents (returns answer + citations) |
| `POST` | `/api/v1/chat/stream` | Query with SSE streaming |
| `GET` | `/api/v1/chat/models` | List available OpenRouter models |

---

## Project Structure

```
├── backend/
│   ├── main.py              # Uvicorn entry point
│   ├── api/                 # FastAPI app, routers, dependencies
│   ├── core/                # Config, exceptions, logging, DI container
│   ├── generation/          # LLM client, RAG pipeline, prompts
│   ├── ingestion/           # Parser, chunker, embedder
│   ├── models/              # Pydantic schemas
│   ├── repository/          # Qdrant vector store, document storage
│   ├── retrieval/           # BM25, hybrid fusion, reranker
│   └── services/            # Document and chat orchestration
├── frontend/
│   └── src/
│       ├── app/             # Next.js pages (home, chat, documents)
│       ├── components/      # UI components
│       ├── hooks/           # Document processing state hook
│       └── lib/             # API client
└── pyproject.toml           # Python dependencies and tooling
```

---

## Supported File Types

- PDF
- DOCX
- TXT
- Markdown
- CSV

---

## Model Selection

The chat interface lets you choose from any model available on OpenRouter before sending a query. Free models are listed first. The selected model persists for the session.

---

## Development

```bash
# Lint
uv run ruff check backend/

# Format
uv run ruff format backend/

# Frontend build check
cd frontend && npm run build
```

---

## License

MIT
