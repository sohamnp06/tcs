# AI Interview Risk Assessment System

> **Production-quality AI assistant that helps interviewers detect resume inconsistencies, skill mismatches, knowledge depth, and suspicious interview behavior.**

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Application                      │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  Upload  │  │  Parse   │  │ Embeddings│  │  Assess  │  │
│  │  API     │  │  API     │  │   API    │  │   API    │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
│       │              │              │              │         │
│  ┌────▼─────────────────────────────────────────▼──────┐  │
│  │                Resume Intelligence Engine             │  │
│  │  PDF Extractor → Resume Parser → Knowledge Profile  │  │
│  │  Semantic Chunker → Embeddings → ChromaDB Storage   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Web Framework | FastAPI + Uvicorn |
| Data Validation | Pydantic v2 |
| PDF Extraction | PyMuPDF |
| LLM | Google Gemini 1.5 Flash |
| Embeddings | static-retrieval-mrl-en-v1 |
| Vector Store | ChromaDB |
| Configuration | python-dotenv |

## Project Structure

```
app/
├── api/           # Route handlers (no business logic)
├── core/          # Config, logger, exceptions
├── models/        # Pydantic schemas
├── parser/        # PDF extraction + resume parsing
├── knowledge/     # Knowledge profile generation
├── embeddings/    # Embedding model + semantic chunking
├── rag/           # Retriever + context builder
├── llm/           # Gemini client (isolated)
├── database/      # ChromaDB client
├── uploads/       # Uploaded PDF storage
└── utils/         # Pure utility functions
```

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd ai-interview-risk-assessment
```

### 2. Create virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env and set your GOOGLE_API_KEY
```

### 5. Run the application

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Verify health

```bash
curl http://localhost:8000/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "environment": "development"
}
```

## API Documentation

When `DEBUG=true`, interactive API docs are available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Development Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Complete | Project initialization, FastAPI app, health endpoint |
| 2 | 🔲 Pending | Resume Upload API |
| 3 | 🔲 Pending | PyMuPDF PDF text extraction |
| 4 | 🔲 Pending | Gemini resume structured parsing |
| 5 | 🔲 Pending | Knowledge profile generation |
| 6 | 🔲 Pending | Semantic chunking |
| 7 | 🔲 Pending | Embeddings + ChromaDB storage |
| 8 | 🔲 Pending | Semantic retriever |
| 9 | 🔲 Pending | Context builder |

## Environment Variables

See [.env.example](.env.example) for all configuration options.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_API_KEY` | ✅ Yes | — | Google Gemini API key |
| `GEMINI_MODEL` | No | `gemini-1.5-flash` | Model identifier |
| `DEBUG` | No | `false` | Enable Swagger docs |
| `LOG_LEVEL` | No | `INFO` | Logging verbosity |
| `MAX_UPLOAD_SIZE_MB` | No | `10` | PDF size limit |

## Error Handling

All errors return a consistent JSON shape:

```json
{
  "success": false,
  "error": "INVALID_FILE_TYPE",
  "message": "Only PDF files are accepted.",
  "detail": "Received: .docx"
}
```

## License

Internal use — TCS AI Hackathon 2025.
