# Context

The user and tenant from your request land on `generate()` and tools.
They are not prompt arguments, so the model cannot hop tenants.

```bash
export GEMINI_API_KEY=your-api-key
uv sync
uv run src/main.py
```
