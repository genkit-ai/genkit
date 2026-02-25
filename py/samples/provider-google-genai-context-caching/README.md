# Context Caching

Cache large documents (books, codebases) so subsequent queries are faster and
cheaper — the model doesn't re-process the same content each time.

## Features Demonstrated

| Feature | Flow / API | Description |
|---------|-----------|-------------|
| Context Caching | `text_context_flow` | Cache a book for repeated queries |
| URL Document Loading | `httpx.get(url)` | Download text from a URL |
| Cache TTL | `ttl_seconds=300` | Cache expires after 5 minutes |
| Cached Content Config | `cached_content` | Pass cached content to generation |

## ELI5: Key Concepts

| Concept | ELI5 |
|---------|------|
| **Context Caching** | Pre-load a big document so the AI remembers it — like giving a student a textbook before the exam |
| **TTL** | How long the cache lasts — after 5 minutes, the AI "forgets" and needs to re-read |
| **Cached Content** | The pre-processed document — saves time and money on repeated queries |

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
- `../../packages` (Genkit core)
- `../../plugins` (Genkit plugins)
- File patterns: `*.py`, `*.prompt`, `*.json`

Changes will automatically trigger a restart of the sample. You can also pass command-line arguments directly to the script, e.g., `./run.sh --some-flag`.

## Setup

```bash
export GEMINI_API_KEY=<Your api key>
```

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
   cd py/samples/provider-google-genai-context-caching
   ./run.sh
   ```

3. **Open DevUI** at http://localhost:4000

4. **Test context caching flow**:
   - [ ] Run `text_context_flow` with default inputs (Tom Sawyer)
   - [ ] Note the first response time (slower - caching context)
   - [ ] Run again with a different query - should be faster
   - [ ] Try custom book URL and query

5. **Default test values**:
   - Book: Tom Sawyer from Project Gutenberg
   - Query: "What are Huck Finn's views on society?"

6. **Expected behavior**:
   - First call: Slower (downloads book, caches context)
   - Subsequent calls: Much faster (uses cached context)
   - Accurate answers about the book's content
   - Cache expires after 5 minutes (configurable)
