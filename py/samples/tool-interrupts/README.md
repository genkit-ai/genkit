# Tool interrupts

A tool can stop and ask a human before it finishes. You see the reason,
then `restart_tool` re-runs it or `respond_to_interrupt` injects a
result without running the tool.

```bash
export GEMINI_API_KEY=your-api-key
uv sync
uv run src/main.py
```
