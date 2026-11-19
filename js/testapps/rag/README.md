# RAG (Retrieval-Augmented Generation)

Demonstrates RAG workflows written as plain flows: retrieval is ordinary code
that returns `Document`s, which are then passed to the model. Includes simple
fact-based RAG and PDF-based RAG using
[vectra](https://www.npmjs.com/package/vectra) as a local vector store (see
`src/vector-store.ts`), and Firestore-backed RAG using Firestore vector search
(`findNearest`).

## Features Demonstrated

| Feature | Flow | Description |
|---------|------|-------------|
| Cat Facts Indexing | `indexCatFactsDocuments` | Index text documents into a local vector store |
| Cat Facts QA | `askQuestionsAboutCats` | RAG over the indexed cat facts |
| Dog Facts Indexing | `indexDogFactsDocuments` | Index text documents into a local vector store |
| Dog Facts QA | `askQuestionsAboutDogs` | RAG over the indexed dog facts |
| PDF Indexing | `indexPdf` | Chunk a PDF and index it into a local vector store |
| PDF RAG | `pdfQA` | RAG over PDF chunks |
| PDF Indexing (Firestore) | `indexPdfFirestore` | Chunk a PDF and store chunks with embeddings in Firestore |
| PDF RAG (Firestore) | `pdfQAFirebase` | RAG using Firestore vector search |

## Setup

### Prerequisites

- **Node.js** (v18 or higher)
- **pnpm** package manager

### API Keys

```bash
export GEMINI_API_KEY='<your-api-key>'
```

For Firestore:

```bash
export GOOGLE_APPLICATION_CREDENTIALS='/path/to/service-account-key.json'
```

The Firestore flows need a vector index on the `embedding` field of the
`pdf-qa` collection (see the
[Firestore vector search docs](https://firebase.google.com/docs/firestore/vector-search)).

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
   - [ ] `askQuestionsAboutCats`: input `"How much do cats sleep?"`
   - [ ] `askQuestionsAboutDogs`: input `"How many dog breeds are there?"`

4. **Expected behavior**:
   - Indexing flows store documents in local vectra indexes (`.vectra/` folder)
   - QA flows retrieve relevant context and generate answers
   - Answers are grounded in the indexed facts
