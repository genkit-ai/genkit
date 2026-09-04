# Typed generate()

`output_schema` is the Pydantic model you get back on `response.output`.

`generate_stream(..., output_schema=Country)` streams typed chunks:
chunks are a `Country` (fields may still be `None` or a prefix);
`(await sr.response).output` is the finished, validated object.

```bash
export GEMINI_API_KEY=your-api-key
uv sync
uv run src/main.py
```
