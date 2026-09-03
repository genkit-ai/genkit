# Speech, image, video

Same `generate()` for speech and images. Video is
`generate_operation` plus `check_operation`. Speech and image use
Google AI Studio; video uses Vertex (`VertexAI.veo_model`).

```bash
export GEMINI_API_KEY=your-api-key
gcloud auth application-default login
uv sync
uv run src/main.py
```
