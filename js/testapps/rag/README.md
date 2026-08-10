# RAG (Retrieval-Augmented Generation)

Demonstrates RAG workflows with multiple vector store backends — Pinecone,
ChromaDB, and the local dev vector store. Includes PDF-based RAG, simple
fact-based RAG, and Firebase-backed RAG.

## Features Demonstrated

| Feature | Flow | Description |
|---------|------|-------------|
| Cat Facts QA | `askQuestionsAboutCats` | RAG using Pinecone retriever |
| Dog Facts QA | `askQuestionsAboutDogs` | RAG using ChromaDB retriever |
| Cat Facts Indexing | `indexCatFactsDocuments` | Index text documents into Pinecone |
| Dog Facts Indexing | `indexDogFactsDocuments` | Index text documents into ChromaDB |
| PDF RAG | `pdfQA` | RAG over PDF documents with text chunking |
| PDF RAG (Firebase) | `pdfQAFirebase` | RAG with Firebase-backed vector store |
| Simple RAG (Local) | `askQuestionsAboutDogs` (local) | RAG using local dev vector store |

## Setup

### Prerequisites

- **Node.js** (v18 or higher)
- **pnpm** package manager

### API Keys

```bash
export GEMINI_API_KEY='<your-api-key>'
```

For Pinecone:

```bash
export PINECONE_API_KEY='<your-pinecone-key>'
```

For Firebase:

```bash
export GOOGLE_APPLICATION_CREDENTIALS='/path/to/service-account-key.json'
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

2. **Index documents first**:
   ```bash
   genkit flow:run indexCatFactsDocuments '["Cats sleep 12-16 hours a day.", "Cats have 230 bones."]'
   genkit flow:run indexDogFactsDocuments '["There are over 400 distinct dog breeds."]'
   ```

3. **Query indexed documents**:
   - [ ] `askQuestionsAboutCats` — Input: `"How much do cats sleep?"`
   - [ ] `askQuestionsAboutDogs` — Input: `"How many dog breeds are there?"`

4. **Expected behavior**:
   - Indexing flows store documents in the respective vector stores
   - QA flows retrieve relevant context and generate answers
   - Answers are grounded in the indexed facts
