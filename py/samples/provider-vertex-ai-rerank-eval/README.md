# Vertex AI Rerankers and Evaluators Demo

Demonstrates using Vertex AI rerankers for RAG quality improvement and evaluators
for assessing model outputs.

## Features

### Rerankers

Semantic document reranking improves RAG quality by re-ordering retrieved documents
based on their semantic relevance to a query.

* **`rerank_documents`** - Basic document reranking
* **`rag_with_reranking`** - Full RAG pipeline (retrieve → rerank → generate)

### Evaluators

Vertex AI evaluators assess model outputs using various quality metrics:

* **`evaluate_fluency`** - Text fluency (1-5 scale)
* **`evaluate_safety`** - Content safety assessment
* **`evaluate_groundedness`** - Hallucination detection (is output grounded in context?)
* **`evaluate_bleu`** - BLEU score for translation quality
* **`evaluate_summarization`** - Summarization quality assessment

## Quick Start

```bash
export GOOGLE_CLOUD_PROJECT=your-project-id
./run.sh
```

That's it! The script will:

1. ✓ Prompt for your project ID if not set
2. ✓ Check gcloud authentication (and help you authenticate if needed)
3. ✓ Enable required APIs (with your permission)
4. ✓ Install dependencies
5. ✓ Start the demo and open your browser

## Manual Setup (if needed)

If you prefer manual setup or the automatic setup fails:

### 1. Authentication

```bash
gcloud auth application-default login
```

### 2. Enable Required APIs

```bash
# Vertex AI API (for models and evaluators)
gcloud services enable aiplatform.googleapis.com

# Discovery Engine API (for rerankers)
gcloud services enable discoveryengine.googleapis.com
```

### 3. Run the Demo

```bash
./run.sh
```

Or manually:

```bash
genkit start -- uv run src/main.py
```

Then open the Dev UI at http://localhost:4000

## Testing the Demo

### Reranker Flows

1. **`rerank_documents`**
   * Input: A query string (default: "How do neural networks learn?")
   * Output: Documents sorted by relevance score
   * The sample includes irrelevant documents to show how reranking filters them

2. **`rag_with_reranking`**
   * Input: A question (default: "What is machine learning?")
   * Output: Generated answer using top-ranked documents as context
   * Demonstrates the two-stage retrieval pattern

### Evaluator Flows

1. **`evaluate_fluency`**
   * Tests text fluency with samples including intentionally poor grammar
   * Scores: 1 (poor) to 5 (excellent)

2. **`evaluate_safety`**
   * Tests content safety
   * Higher scores = safer content

3. **`evaluate_groundedness`**
   * Tests if outputs are grounded in provided context
   * Includes a hallucination example (claims population when not in context)

4. **`evaluate_bleu`**
   * Tests translation quality against reference translations
   * Scores: 0 to 1 (higher = closer to reference)

5. **`evaluate_summarization`**
   * Tests summarization quality

## Supported Reranker Models

| Model | Description |
|-------|-------------|
| `semantic-ranker-default@latest` | Latest default semantic ranker |
| `semantic-ranker-default-004` | Semantic ranker version 004 |
| `semantic-ranker-fast-004` | Fast variant (lower latency) |
| `semantic-ranker-default-003` | Semantic ranker version 003 |
| `semantic-ranker-default-002` | Semantic ranker version 002 |

## Supported Evaluation Metrics

| Metric | Description |
|--------|-------------|
| BLEU | Translation quality (compare to reference) |
| ROUGE | Summarization quality (compare to reference) |
| FLUENCY | Language mastery and readability |
| SAFETY | Harmful/inappropriate content check |
| GROUNDEDNESS | Factual grounding in context |
| SUMMARIZATION\_QUALITY | Overall summarization ability |
| SUMMARIZATION\_HELPFULNESS | Usefulness as a summary |
| SUMMARIZATION\_VERBOSITY | Conciseness of summary |

## Troubleshooting

### "Discovery Engine API not enabled"

The script should enable this automatically, but if it fails:

```bash
gcloud services enable discoveryengine.googleapis.com
```

### "Permission denied"

Ensure your account has the required IAM roles:

* `roles/discoveryengine.admin` (for rerankers)
* `roles/aiplatform.user` (for evaluators)

### "Project not found"

Verify `GOOGLE_CLOUD_PROJECT` is set correctly:

```bash
echo $GOOGLE_CLOUD_PROJECT
```

### "gcloud not found"

Install the Google Cloud SDK from:
https://cloud.google.com/sdk/docs/install
