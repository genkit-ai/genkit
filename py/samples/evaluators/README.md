# Evaluators Sample

This sample demonstrates how to work with configurable evaluators in Genkit, including both built-in plugins and custom LLM-based scoring. Each evaluator runs against a dataset of test cases and produces structured evaluation results.

## Included Evaluators

- **`genkitEval/regex`**  
  Simple regex match evaluator.
  - No LLM or API keys required.
  - Compares output to a reference regex pattern defined in the test data.

- **`byo/maliciousness`**  
  LLM-powered; checks if the output intends to deceive, harm, or exploit.
  - Requires access to an LLM (Google Gemini; set `GEMINI_API_KEY`).
  - Uses a scoring rubric to rate maliciousness.

- **`byo/answer_accuracy`**  
  LLM-powered; rates the quality of the output versus a reference.
  - Scoring: 0 (no match), 2 (partial match), 4 (full match).

## Quickstart

1. **Set up dependencies and API keys (if required):**
   ```bash
   export GEMINI_API_KEY=your-api-key   # Only needed for byo/* LLM evaluators
   uv sync
   uv run src/main.py
   ```

2. **Run evaluation from the command line:**  
   (Requires `genkit` CLI; replace dataset filenames as needed)

   - **Regex evaluator (no LLM needed):**
     ```bash
     genkit eval:run datasets/genkit_eval_dataset.json --evaluators=genkitEval/regex
     ```

   - **Maliciousness (requires LLM):**
     ```bash
     genkit eval:run datasets/maliciousness_dataset.json --evaluators=byo/maliciousness
     ```

   - **Answer accuracy (requires LLM):**
     ```bash
     genkit eval:run datasets/answer_accuracy_dataset.json --evaluators=byo/answer_accuracy
     ```

## Developer Notes

- Each evaluator function is defined in `src/main.py`.
- Datasets are expected to be JSON files located in the `datasets/` directory.
- Enable more evaluators or customize logic by editing the corresponding Python source.
- For LLM evaluators, ensure required API keys are available in your environment.

See `src/main.py` for entry points, and modify as needed for your use case.
