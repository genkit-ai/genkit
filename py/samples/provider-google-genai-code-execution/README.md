# Google GenAI Code Execution

Demonstrates Gemini's code execution capability — the model writes and runs
Python code on Google's servers to solve computational problems.

## Features Demonstrated

| Feature | Flow / API | Description |
|---------|-----------|-------------|
| Code Execution | `code_execution_config` | Model writes and executes Python code |
| Executable Code Part | `ExecutableCodePart` | Generated code in the response |
| Code Execution Result | `CodeExecutionResultPart` | Execution output in the response |
| CustomPart Handling | `CustomPart` | Non-standard response parts |

## ELI5: Key Concepts

| Concept | ELI5 |
|---------|------|
| **Code Execution** | AI writes Python code and runs it on Google's servers — like giving the AI a calculator |
| **ExecutableCodePart** | The code the AI wrote — you can see exactly what it ran |
| **CodeExecutionResultPart** | The output from running the code — the actual answer |

## Setup environment

### How to Get Your Gemini API Key

To use the Google GenAI plugin, you need a Gemini API key.

1.  **Visit AI Studio**: Go to [Google AI Studio](https://aistudio.google.com/).
2.  **Create API Key**: Click on "Get API key" and create a key in a new or existing Google Cloud project.

For more details, check out the [official documentation](https://ai.google.dev/gemini-api/docs/api-key).

Export the API key as env variable `GEMINI_API_KEY` in your shell configuration.

```bash
export GEMINI_API_KEY='<Your api key>'
```

### Monitoring and Running

For an enhanced development experience, use the provided `run.sh` script to start the sample with automatic reloading:

```bash
./run.sh
```

This script uses `watchmedo` to monitor changes in:
- `src/` (Python logic)
- `../../packages` (Genkit core)
- `../../plugins` (Genkit plugins)
- File patterns: `*.py`, `*.prompt`, `*.json`

Changes will automatically trigger a restart of the sample. You can also pass command-line arguments directly to the script, e.g., `./run.sh --some-flag`.

## Run the sample

```bash
genkit start -- uv run src/main.py
```

## Testing This Demo

1. **Prerequisites**:
   ```bash
   export GEMINI_API_KEY=your_api_key
   ```
   Or the demo will prompt for the key interactively.

2. **Run the demo**:
   ```bash
   cd py/samples/provider-google-genai-code-execution
   ./run.sh
   ```

3. **Open DevUI** at http://localhost:4000

4. **Test code execution flows**:
   - [ ] Run a flow that requests computation
   - [ ] Verify the model writes Python code
   - [ ] Check that code execution results appear
   - [ ] Test math problems, data analysis queries

5. **Example prompts to try**:
   - "Calculate the factorial of 10"
   - "Generate the first 20 Fibonacci numbers"
   - "What is the sum of prime numbers under 100?"

6. **Expected behavior**:
   - Model generates Python code
   - Code executes on Gemini's servers
   - Results include both code and output
   - CustomPart contains execution details
