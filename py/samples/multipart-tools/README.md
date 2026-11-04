# Multipart tools

A tool either returns a plain value (wrapped as output-only), returns `MultipartToolResponse(...)` with optional media parts, or streams live progress updates via `ToolContext[ChunkT].send_chunk(...)`.

```bash
export GEMINI_API_KEY=your-api-key
uv sync
uv run src/main.py
```

In Dev UI:

```bash
genkit start -- uv run src/main.py
```

Then open [http://localhost:4000](http://localhost:4000) and run `ask_about_the_lab` or stream `prep_with_progress`.
