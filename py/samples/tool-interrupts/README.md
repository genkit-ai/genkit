# Tool Interrupts

Human-in-the-loop: `ctx.interrupt()` and `tool_response()` let the model pause for user input, then continue.

```bash
export GEMINI_API_KEY=your-api-key
uv sync
uv run src/main.py
```

This launches a small interactive CLI trivia session.

To inspect the same flow in Dev UI instead:

```bash
genkit start -- uv run src/main.py
```

Try `play_trivia`.
