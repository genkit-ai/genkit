# Evaluation in Genkit

This sample demonstrates the different evaluation features using Genkit Python SDK.

Note: This sample focuses on evaluation features in Genkit, by utilizing the official Genkit Evaluators plugin. If you are interested in writing your own custom evaluator, please check the `custom/test_evaluator` defined in `src/main.py`.

### How to Get Your Gemini API Key

To use the Google GenAI plugin, you need a Gemini API key.

1.  **Visit AI Studio**: Go to [Google AI Studio](https://aistudio.google.com/).
2.  **Create API Key**: Click on "Get API key" and create a key in a new or existing Google Cloud project.

For more details, check out the [official documentation](https://ai.google.dev/gemini-api/docs/api-key).

### Monitoring and Running

For an enhanced development experience, use the provided `run.sh` script to start the sample with automatic reloading:

```bash
./run.sh
```

This script uses `watchmedo` to monitor changes in:
- `src/` (Python logic)
- `docs/` (PDF documents)
- `../../packages` (Genkit core)
- `../../plugins` (Genkit plugins)
- File patterns: `*.py`, `*.prompt`, `*.json`, `*.pdf`

Changes to Python, prompt, or JSON files will trigger an automatic restart. **Changes to PDF files in `docs/` will also trigger an automatic re-indexing** of the documents by deleting the internal marker file `__db_pdf_qa.json` before restarting.

You can also pass command-line arguments directly to the script, e.g., `./run.sh --some-flag`.

## Setup and start the sample

```bash
export GEMINI_API_KEY=<Your api key>
```

Choose any flow of interest and run in the Dev UI.
## Available Flows

- **setup**: Indexes the default PDF document (`docs/cat-handbook.pdf`) into the vector store
- **pdf_qa**: RAG flow that answers questions based on indexed PDF documents. It requires `setup` flow run first.
- **index_pdf**: Indexes a specified PDF file (defaults to `docs/cat-wiki.pdf`)
- **simple_structured**: Simple flow with structured input/output
- **simple_echo**: Simple echo flow
- **dog_facts_eval**: Programmatic evaluation flow using the faithfulness metric on a dog facts dataset. **Note:** This flow can take several minutes to complete.

### Running Evaluations in Dev UI

1. Navigate to the **Evaluations** tab in the Dev UI
2. Click **"Run Evaluation"** or **"New Evaluation"**
3. Configure:
   - **Flow**: Select the flow to evaluate (e.g., `pdf_qa`)
   - **Dataset**: Upload or select a JSON file (e.g., `data/cat_adoption_questions.json`)
   - **Evaluators**: Select one or more evaluators:
     - `custom/test_evaluator` - Random evaluator for testing (fast, no LLM calls)
     - `genkitEval/faithfulness` - Checks if output is faithful to context
     - `genkitEval/maliciousness` - Detects harmful content
     - `genkitEval/answer_relevancy` - Checks if answer is relevant to question
4. Click **"Run"**
5. View results in the Evaluations tab

## Reference

For more details on using Genkit evaluations, please refer to the official [Genkit documentation](https://firebase.google.com/docs/genkit/evaluation).

## Testing This Demo

1. **Open DevUI** at http://localhost:4000

2. **Test the flows**:
   - [ ] Go to the Evaluators tab in DevUI
   - [ ] Find `custom/test_evaluator` (Random Eval)
   - [ ] Run evaluation on sample data
   - [ ] Check that scores are generated (0.0-1.0 range)
   - [ ] Verify ~10% of evaluations fail (simulated errors)

3. **Test the PDF RAG flow**:
   - [ ] Run `setup` flow first to index documents
   - [ ] Test the `pdf_qa` flow
   - [ ] Check evaluation metrics for relevance

4. **Expected behavior**:
   - Evaluators appear in DevUI Evaluators tab
   - Random evaluator produces varied scores
   - PASS/FAIL status based on score threshold (0.5)
