# Google Media

Three focused Google media examples: text-to-speech, Imagen image generation, and Veo video generation.

```python
from genkit_google_genai import GeminiTtsConfig, ImagenConfig, VeoConfig, gemini_tts_model, imagen_model, veo_model

# TTS
await ai.generate(
    model=gemini_tts_model('gemini-2.5-flash-preview-tts'),
    config=GeminiTtsConfig.model_validate({'speech_config': {'voice_config': {'prebuilt_voice_config': {'voice_name': 'Kore'}}}}),
    prompt='Hello!',
)

# Imagen
await ai.generate(
    model=imagen_model('imagen-3.0-generate-002'),
    config=ImagenConfig(number_of_images=1),
    prompt='A watercolor postcard',
)

# Veo (background model — poll with check_operation)
await ai.generate(
    model=veo_model('veo-3.1-generate-preview'),
    config=VeoConfig(aspect_ratio='16:9', duration_seconds=5),
    prompt='A paper airplane gliding through a classroom',
)
```
```bash
export GEMINI_API_KEY=your-api-key
uv sync
uv run src/main.py
```

To explore the flows in Dev UI instead:

```bash
genkit start -- uv run src/main.py
```

Flows: `generate_speech`, `generate_image`, `generate_video`.

`generate_video` supports testing Veo models by setting `model` in flow input, for example:

- `googleai/veo-3.1-generate-preview`
- `googleai/veo-3.1-fast-generate-preview`
- `googleai/veo-3.0-generate-001`
- `googleai/veo-3.0-fast-generate-001`
- `googleai/veo-3.1-generate-001`
- `googleai/veo-3.1-fast-generate-001`
- `googleai/veo-2.0-generate-001`

The flow input includes Veo config fields such as `aspect_ratio`, `duration_seconds`, `resolution`, and `seed`.
