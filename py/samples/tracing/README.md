# Tracing

Run this, open http://localhost:4000, and the ticket triage steps show
up as spans.

```bash
export GEMINI_API_KEY=your-api-key
uv sync
genkit start -- uv run src/main.py
```

In Dev UI, run `triage_ticket` and open the Traces tab. Or skip the UI:

```bash
uv run src/main.py
```

To send the same spans to Cloud Trace later:

```python
from genkit_google_cloud import enable_google_cloud_telemetry

enable_google_cloud_telemetry(project_id='my-project')
```
