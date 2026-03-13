# Local Vector Store (RAG Development)

Test RAG (Retrieval-Augmented Generation) locally without setting up an external
vector database. Documents are embedded and stored in-memory for fast iteration.

## How Local RAG Works

```
┌─────────────────────────────────────────────────────────────────┐
│                  LOCAL RAG PIPELINE                               │
│                                                                  │
│  STEP 1: INDEX (one-time)          STEP 2: RETRIEVE (per query) │
│  ─────────────────────────         ──────────────────────────── │
│                                                                  │
│  "The Matrix is a 1999..."         "What's a good sci-fi film?" │
│         │                                   │                    │
│         ▼                                   ▼                    │
│  ┌──────────────┐                  ┌──────────────┐             │
│  │  Embedder    │                  │  Embedder    │             │
│  │  (Vertex AI) │                  │  (Vertex AI) │             │
│  └──────┬───────┘                  └──────┬───────┘             │
│         │                                  │                     │
│         ▼                                  ▼                     │
│  ┌──────────────┐                  ┌──────────────┐             │
│  │ Local Vector │◄─── compare ────│ Query Vector │             │
│  │ Store        │                  └──────────────┘             │
│  │ (in-memory)  │────────────────────────────────►              │
│  └──────────────┘                  "The Matrix" (0.95 match)    │
│                                    "Inception"  (0.89 match)    │
└─────────────────────────────────────────────────────────────────┘
```

## Features Demonstrated

| Feature | Flow / API | Description |
|---------|-----------|-------------|
| Local Vector Store | `define_dev_local_vector_store()` | In-memory store, no external DB needed |
| Document Indexing | `index_documents` | Embed and store film descriptions |
| Similarity Retrieval | `retrieve_documents` | Find semantically similar documents |
| Document Creation | `Document.from_text()` | Convert plain text to Genkit documents |
| Vertex AI Embeddings | `gemini-embedding-001` | Cloud-based embedding model |

## ELI5: Key Concepts

| Concept | ELI5 |
|---------|------|
| **RAG** | AI looks up your docs before answering — fewer hallucinations! |
| **Vector Store** | A database that finds "similar" items by meaning — "happy" finds docs about "joyful" too |
| **Local Store** | Runs on your computer, no cloud DB needed — perfect for testing before production |
| **Indexing** | Adding documents to the store — like a librarian cataloging new books |
| **Retrieval** | Finding documents that match a query — "sci-fi films" returns The Matrix, Inception |

## Quick Start

```bash
export GCLOUD_PROJECT=your-project-id
gcloud auth application-default login
./run.sh
```

Then open the Dev UI at http://localhost:4000.

## Setup

### 1. Authenticate with Google Cloud

Vertex AI embeddings require GCP authentication:

```bash
gcloud auth application-default login
```

### 2. Set Your Project ID

```bash
export GCLOUD_PROJECT=your-project-id
```

Or the demo will prompt you interactively.

### 3. Run the Sample

```bash
./run.sh
```

Or manually:

```bash
genkit start -- uv run src/main.py
```

## Testing This Demo

1. **Open DevUI** at http://localhost:4000

2. **Index documents first**:
   - [ ] Run `index_documents` — indexes 10 classic film descriptions

3. **Test retrieval**:
   - [ ] Run `retrieve_documents` — queries for "sci-fi film"
   - [ ] Verify The Matrix, Inception, Star Wars appear as top matches

4. **Expected behavior**:
   - Documents are embedded using Vertex AI (`gemini-embedding-001`)
   - Vector store is in-memory (no external database)
   - Retrieval returns semantically similar documents (top 3)
   - No external vector database required

## Sample Documents

The sample indexes 10 classic films including The Godfather, The Dark Knight,
Pulp Fiction, Inception, The Matrix, Star Wars, and more.

## Development

The `run.sh` script uses `watchmedo` for hot reloading on file changes.
