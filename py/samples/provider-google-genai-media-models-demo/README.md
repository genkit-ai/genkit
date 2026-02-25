# Media Generation Models Demo

This sample demonstrates all media generation capabilities in the Google GenAI plugin:

| Model | Output | Pattern | Latency |
|-------|--------|---------|---------|
| **TTS** | Audio (WAV) | Standard | ~1-5 seconds |
| **Imagen** | Image (PNG) | Standard | ~5-15 seconds |
| **Gemini Image** | Image (PNG) | Standard | ~5-15 seconds |
| **Lyria** | Audio (WAV) | Standard | ~5-30 seconds |
| **Veo** | Video (MP4) | Background | 30s - 5 minutes |

## Quick Start

```bash
# Run with simulated models (no API key needed)
./run.sh

# Open DevUI at http://localhost:4000
```

## Testing with Real Models

### Google AI Models (TTS, Image, Veo)

```bash
# Set your Google AI API key
export GEMINI_API_KEY=your_api_key_here

# Run the demo
./run.sh
```

Get an API key from [Google AI Studio](https://aistudio.google.com/apikey).

### Vertex AI Models (Lyria)

Lyria is only available through Vertex AI:

```bash
# Authenticate with Google Cloud
gcloud auth application-default login

# Set your project
export GOOGLE_CLOUD_PROJECT=your_project_id

# Run the demo
./run.sh
```

## Available Flows

### 1. TTS Speech Generator (`tts_speech_generator`)

Converts text to natural-sounding speech.

```python
# In DevUI or code:
result = await tts_speech_generator_flow(
    text="Hello! Welcome to Genkit.",
    voice="Kore"  # Options: Zephyr, Puck, Charon, Kore, Fenrir, Leda, etc.
)
```

**Voices Available:**
- Zephyr, Puck, Charon, Kore, Fenrir, Leda
- Orus, Aoede, Callirrhoe, Autonoe, Enceladus, Iapetus

### 2. Imagen Image Generator (`imagen_image_generator`)

Generates photorealistic images from text descriptions using the Imagen predict API.
Imagen models are discovered dynamically by the GoogleAI plugin under the `googleai/` prefix.

```python
result = await imagen_image_generator_flow(
    prompt="A photorealistic image of a cat astronaut in space",
    number_of_images=1  # 1-4 images
)
```

### 3. Gemini Image Generator (`gemini_image_generator`)

Generates images from text descriptions using Gemini's native generateContent API.

```python
result = await gemini_image_generator_flow(
    prompt="A serene Japanese garden with cherry blossoms",
    aspect_ratio="16:9"  # Options: 1:1, 16:9, 9:16, 4:3, 3:4
)
```

### 4. Lyria Audio Generator (`lyria_audio_generator`)

Generates music and audio from text descriptions. Requires Vertex AI.

```python
result = await lyria_audio_generator_flow(
    prompt="A peaceful piano melody with gentle rain",
    negative_prompt="loud, aggressive"
)
```

### 5. Veo Video Generator (`veo_video_generator`)

Generates videos using the background model pattern (long-running operation).

```python
result = await veo_video_generator_flow(
    prompt="A cat playing piano in a jazz club",
    aspect_ratio="16:9",
    duration_seconds=5
)
```

**How Veo Works:**
```
┌─────────────────────────────────────────────────────────────┐
│                  Veo Background Model Flow                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. start()        2. poll every 3-5s      3. complete       │
│  ┌────────┐        ┌────────────────┐      ┌────────────┐   │
│  │ Prompt │───────►│   Operation    │─────►│ Video URL  │   │
│  └────────┘        │   (job ID)     │      └────────────┘   │
│                    │   10%...50%... │                        │
│                    └────────────────┘                        │
│                                                              │
│  Total time: 30 seconds to 5 minutes                         │
└─────────────────────────────────────────────────────────────┘
```

### 6. Media Models Overview (`media_models_overview`)

Returns information about all available models and their status.

```python
result = await media_models_overview_flow()
# Returns which models are available based on environment
```

## Testing Checklist

### Without API Keys (Simulated)

- [ ] Run `./run.sh` without any API keys set
- [ ] Open DevUI at http://localhost:4000
- [ ] Test `tts_speech_generator` - should return fake audio in ~1s
- [ ] Test `imagen_image_generator` - should return fake image in ~2s
- [ ] Test `gemini_image_generator` - should return fake image in ~2s
- [ ] Test `lyria_audio_generator` - should return fake audio in ~3s
- [ ] Test `veo_video_generator` - should show progress updates over ~10s
- [ ] Test `media_models_overview` - should show all models as simulated

### With GEMINI_API_KEY

- [ ] Set `GEMINI_API_KEY` environment variable
- [ ] Run `./run.sh`
- [ ] Test `tts_speech_generator` with different voices
- [ ] Test `imagen_image_generator` with different prompts
- [ ] Test `gemini_image_generator` with different prompts
- [ ] Test `veo_video_generator` (may take 30s-5min)
- [ ] Verify returned URLs/data are real media content

### With GOOGLE_CLOUD_PROJECT (Vertex AI)

- [ ] Authenticate with `gcloud auth application-default login`
- [ ] Set `GOOGLE_CLOUD_PROJECT` environment variable
- [ ] Run `./run.sh`
- [ ] Test `lyria_audio_generator` - should generate real audio

## Model Configuration

### TTS Config

```python
config = {
    'speech_config': {
        'voice_config': {
            'prebuilt_voice_config': {'voice_name': 'Kore'}
        },
        # Multi-speaker (optional):
        'multi_speaker_voice_config': {
            'speaker_voice_configs': [
                {'speaker': 'Alice', 'voice_config': {...}},
                {'speaker': 'Bob', 'voice_config': {...}},
            ]
        }
    }
}
```

### Gemini Image Config

```python
config = {
    'image_config': {
        'aspect_ratio': '16:9',  # or 1:1, 9:16, 4:3, 3:4, etc.
        'image_size': '2K',       # 1K, 2K, or 4K
    }
}
```

### Imagen Config

```python
# Imagen uses the predict API with simpler config:
config = {
    'number_of_images': 1,  # 1-4 images
}
```

### Veo Config

```python
config = {
    'aspect_ratio': '16:9',      # or 9:16
    'duration_seconds': 5,        # 5-8 seconds
    'enhance_prompt': True,       # AI-enhanced prompts
    'negative_prompt': 'blurry',  # What to avoid
    'person_generation': 'allow', # or 'block'
}
```

### Lyria Config

```python
config = {
    'negative_prompt': 'noise, distortion',
    'seed': 12345,        # For reproducibility
    'sample_count': 1,    # Number of samples
    'location': 'global', # Required for Lyria
}
```

## Troubleshooting

### "Model not found" error

Ensure the API key is set correctly:
```bash
echo $GEMINI_API_KEY  # Should show your key
```

### Lyria not working

Lyria requires Vertex AI. Ensure:
1. `GOOGLE_CLOUD_PROJECT` is set
2. You've run `gcloud auth application-default login`
3. The project has Vertex AI API enabled

### Veo timeout

Video generation can take up to 5 minutes. The demo has a 5-minute timeout.
For longer videos or complex prompts, consider increasing the timeout.

## See Also

- [Veo Documentation](https://ai.google.dev/gemini-api/docs/video)
- [TTS Documentation](https://ai.google.dev/gemini-api/docs/speech)
- [Imagen Documentation](https://ai.google.dev/gemini-api/docs/imagen)
- [Gemini Image](https://ai.google.dev/gemini-api/docs/image-generation)
- [Lyria (Vertex AI)](https://cloud.google.com/vertex-ai/docs/generative-ai/audio)
