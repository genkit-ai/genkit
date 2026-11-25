# Multimodal RAG

Demonstrates multimodal Retrieval-Augmented Generation (RAG) with PDFs and
videos: extract text and images, embed them with a multimodal embedder into a
local [vectra](https://www.npmjs.com/package/vectra) index (see
`src/vector-store.ts`), and query with natural language.

## Features Demonstrated

| Feature | Flow | Description |
|---------|------|-------------|
| PDF Indexing | `indexMultimodalPdf` | Extract text + images from a PDF and index them |
| PDF QA | `multimodalPdfQuestions` | RAG over indexed PDF content (text + images) |
| Video Indexing (Local) | `localIndexVideo` | Extract video segments and index locally |
| Video QA (Local) | `localVideoQuestions` | RAG over locally indexed video content |

## Setup

### Prerequisites

- **Node.js** (v18 or higher)
- **pnpm** package manager

### API Keys

This sample uses Vertex AI (models and `multimodalembedding@001`):

```bash
gcloud auth application-default login
export GCLOUD_PROJECT='<your-project-id>'
```

### Build and Install

From the repo root:

```bash
pnpm install
pnpm run setup
```

## Run the Sample

```bash
pnpm run genkit:dev
```

## Testing This Demo

1. **Open DevUI** at http://localhost:4000

2. **Index a PDF first**:
   ```bash
   genkit flow:run indexMultimodalPdf '"./docs/BirthdayPets.pdf"'
   ```

3. **Query the indexed PDF**:
   - [ ] `multimodalPdfQuestions`: input `"What pets are shown?"`

4. **Index a video**:
   ```bash
   genkit flow:run localIndexVideo '"gs://cloud-samples-data/generative-ai/video/pixel8.mp4"'
   ```

5. **Query the indexed video**:
   - [ ] `localVideoQuestions`: input `"describe the video"`

6. **Expected behavior**:
   - PDF text and images are extracted and indexed
   - Video segments are extracted with timestamps
   - QA flows retrieve relevant context and generate answers
   - Streaming shows incremental output
