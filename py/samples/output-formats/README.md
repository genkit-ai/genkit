# Output Formats

Constrain model output to text, enums, JSON objects, arrays, or JSONL.

`stream_country_info_json` shows `generate_stream(..., output_schema=CountryInfo)`:
chunks are a partial of that type (fields may still be `None` or a prefix);
`(await sr.response).output` is the finished object.

```bash
export GEMINI_API_KEY=your-api-key
uv sync
uv run src/main.py
```

To inspect the flows in Dev UI instead:

```bash
genkit start -- uv run src/main.py
```
