# Google Code Execution

Gemini runs Python server-side to solve problems (math, data analysis, etc.).

```bash
export GEMINI_API_KEY=your-api-key
uv sync
uv run src/main.py
```

To run it from Dev UI instead:

```bash
genkit start -- uv run src/main.py
```
