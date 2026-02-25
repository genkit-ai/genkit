# Short-n-long

An example demonstrating running flows as both a short-lived application and a
server.

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

## Run the sample

To start the short-lived application normally.

```bash
uv run src/main.py
```

To start the short-lived application in dev mode:

```bash
genkit start -- uv run src/main.py
```

To start as a server normally:

```bash
uv run src/main.py --server
```

To start as a server in dev mode:

```bash
genkit start -- uv run src/main.py --server
```

## Running with a specific version of Python

```bash
genkit start -- uv run --python python3.10 src/main.py
```

## Testing This Demo

1. **Prerequisites**:
   ```bash
   export GEMINI_API_KEY=your_api_key
   ```

2. **Run the server** (two modes):
   ```bash
   cd py/samples/web-short-n-long

   # Short mode (development with DevUI)
   ./run.sh

   # Long mode (production server)
   uv run python src/main.py --mode=long
   ```

3. **Test the API directly**:
   ```bash
   # Call a flow via HTTP
   curl -X POST http://localhost:8000/say_hi \\
     -H "Content-Type: application/json" \\
     -d '{"name": "World"}'
   ```

4. **Open DevUI** (short mode) at http://localhost:4000

5. **Test the flows**:
   - [ ] `say_hi` - Simple generation
   - [ ] `say_hi_stream` - Streaming response
   - [ ] `simple_generate_with_tools_flow` - Tool calling
   - [ ] `generate_character` - Structured output

6. **Expected behavior**:
   - Server starts and accepts HTTP requests
   - Lifecycle hooks run on startup/shutdown
   - All flows work via HTTP API
   - Proper graceful shutdown on SIGTERM
