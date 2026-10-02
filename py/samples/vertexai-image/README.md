# Vertex AI Image Generation

Vertex image generation through the same `generate()`. Uses Application Default
Credentials, not a Gemini API key.

```bash
export GOOGLE_CLOUD_PROJECT=your-project-id
gcloud auth application-default login
uv sync
uv run src/main.py
```
