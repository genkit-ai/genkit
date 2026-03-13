# Firestore Vector Retriever

Demonstrates using Firestore as a vector store for RAG applications.

## Quick Start

```bash
export GOOGLE_CLOUD_PROJECT=your-project-id
./run.sh
```

The script will:

1. ✓ Prompt for your project ID if not set
2. ✓ Check gcloud authentication (and help you authenticate if needed)
3. ✓ Enable Firestore and Vertex AI APIs (with your permission)
4. ✓ Install dependencies
5. ✓ Remind you to create the vector index (see below)
6. ✓ Start the demo and open your browser

## Required: Create Firestore Vector Index

Before using vector search, you **must** create a composite index:

```bash
gcloud firestore indexes composite create \
  --project=$GOOGLE_CLOUD_PROJECT \
  --collection-group=films \
  --query-scope=COLLECTION \
  --field-config=vector-config='{"dimension":"768","flat": {}}',field-path=embedding
```

> **Note:** Index creation may take a few minutes. The demo will show an error until the index is ready.

## Manual Setup (if needed)

If you prefer manual setup or the automatic setup fails:

### 1. Install gcloud CLI

Download from: https://cloud.google.com/sdk/docs/install

### 2. Authentication

```bash
gcloud auth application-default login
```

### 3. Enable Required APIs

```bash
# Firestore API
gcloud services enable firestore.googleapis.com --project=$GOOGLE_CLOUD_PROJECT

# Vertex AI API (for embeddings)
gcloud services enable aiplatform.googleapis.com --project=$GOOGLE_CLOUD_PROJECT
```

### 4. Create Vector Index

See the command above in "Required: Create Firestore Vector Index"

### 5. Run the Demo

```bash
./run.sh
```

Or manually:

```bash
genkit start -- uv run src/main.py
```

Then open the Dev UI at http://localhost:4000

## Testing the Demo

1. Open DevUI at http://localhost:4000
2. Run `index_documents` to populate the vector store
3. Run `retrieve_documents` with a query to test similarity search
4. Verify retrieved documents match query semantics

## Expected Behavior

- Documents are embedded using Vertex AI and stored in Firestore
- Vector similarity search returns semantically relevant documents
- Firebase telemetry captures traces and metrics

## Development

The `run.sh` script uses `watchmedo` to monitor changes in:
- `src/` (Python logic)
- `../../packages` (Genkit core)
- `../../plugins` (Genkit plugins)
- File patterns: `*.py`, `*.prompt`, `*.json`

Changes will automatically trigger a restart of the sample.
