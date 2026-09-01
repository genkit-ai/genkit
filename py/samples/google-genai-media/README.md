# Speech, image, video

Same `generate()` for speech and images. Video is
`generate_operation` plus `check_operation`.

```bash
export GEMINI_API_KEY=your-api-key
uv sync
uv run src/main.py
```

That does voice and a poster. The Veo poll is commented at the bottom
of `src/main.py` — it takes a couple of minutes.
