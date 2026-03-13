# Format Demo

This sample demonstrates the different output formats supported by Genkit Python.

## Features Demonstrated

| Feature | Flow | Description |
|---------|------|-------------|
| Text Format | `generate_haiku_text` | Raw text output (haiku about a topic) |
| JSON Format | `get_country_info_json` | Structured JSON object output |
| Array Format | `recommend_books_array` | JSON array of items |
| Enum Format | `classify_sentiment_enum` | Single value from a predefined list |
| JSONL Format | `create_story_characters_jsonl` | Newline-delimited JSON (great for streaming) |

## ELI5: Key Concepts

| Concept | ELI5 |
|---------|------|
| **Output Format** | Tell the AI what shape the answer should be — text, JSON, list, etc. |
| **JSON** | Structured data with named fields — like a form with "name", "age", etc. |
| **JSONL** | One JSON object per line — like a spreadsheet where each row is a record |
| **Enum** | Pick one from a list — like a multiple-choice question |

## Running

1.  Set your `GEMINI_API_KEY` environment variable.
2.  Run the sample:

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

```bash
export GEMINI_API_KEY=your-key
./run.sh
```

Then use the Genkit Developer UI to test the flows.

## Testing This Demo

1. **Open DevUI** at http://localhost:4000

2. **Test each format** (inputs pre-populated with defaults):
   - [ ] `generate_haiku_text` - Text format (topic: "coding")
   - [ ] `get_country_info_json` - JSON format (country: "Japan")
   - [ ] `recommend_books_array` - Array format (genre: "Fantasy")
   - [ ] `classify_sentiment_enum` - Enum format (review text)
   - [ ] `create_story_characters_jsonl` - JSONL format (theme: "Space Opera")

3. **Expected behavior**:
   - Text: Returns plain haiku text
   - JSON: Returns structured country info object
   - Array: Returns list of book recommendations
   - Enum: Returns single sentiment classification
   - JSONL: Returns newline-delimited character objects

4. **Verify output structure**:
   - JSON outputs should be valid JSON
   - Array outputs should be valid JSON arrays
   - Enum outputs should match predefined values
