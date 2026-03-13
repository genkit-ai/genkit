# OpenAI Sample

## Features Demonstrated

| Feature | Flow | Description |
|---------|------|-------------|
| Simple Generation | `say_hi` | Basic text generation with GPT |
| Streaming | `say_hi_stream` | Token-by-token streaming response |
| Constrained Output | `say_hi_constrained` | Constrained generation |
| Tool Calling | `calculate_gablorken` | Function calling demo |
| Structured Output | `generate_character` | RPG character as JSON (Pydantic) |
| Vision | `describe_image` | Image description with GPT-4o |
| Reasoning (CoT) | `reasoning_flow` | Chain-of-thought with o4-mini |
| Image Generation | `generate_image` | DALL-E image generation |
| Text-to-Speech | `text_to_speech` | TTS with voice selection |
| Round-trip TTS/STT | `round_trip_tts_stt` | Text → Speech → Text demo |

## Setup environment

### How to Get Your OpenAI API Key

To use the OpenAI plugin, you need an OpenAI API key.

1.  **Visit OpenAI Platform**: Go to [OpenAI API Keys](https://platform.openai.com/api-keys) and sign in.
2.  **Create API Key**: Click on "Create new secret key".
3.  **Add Credits**: You may need to add credits to your account.

For more details, check out the [official documentation](https://platform.openai.com/docs/quickstart).

Export the API key as env variable `OPENAI_API_KEY`:

```bash
export OPENAI_API_KEY=<Your api key>
```

```bash
uv venv
source .venv/bin/activate
```

### Monitoring and Running

For an enhanced development experience, use the provided `run.sh` script to start the sample with automatic reloading:

```bash
./run.sh
```

This script uses `watchmedo` to monitor changes in:
- `src/` (Python logic)
- `../../packages` (Genkit core)
- `../../plugins` (Genkit plugins)
- File patterns: `*.py`, `*.prompt`, `*.json`

Changes will automatically trigger a restart of the sample. You can also pass command-line arguments directly to the script, e.g., `./run.sh --some-flag`.

## Run the sample

```bash
genkit start -- uv run src/main.py
```

## Testing This Demo

1. **Prerequisites**:
   ```bash
   export OPENAI_API_KEY=your_api_key
   ```
   Get your API key from https://platform.openai.com/api-keys
   Or the demo will prompt for the key interactively.

2. **Run the demo**:
   ```bash
   cd py/samples/provider-compat-oai-hello
   ./run.sh
   ```

3. **Open DevUI** at http://localhost:4000

4. **Test basic flows**:
   - [ ] `say_hi` - Simple text generation
   - [ ] `say_hi_stream` - Streaming response
   - [ ] `say_hi_constrained` - Constrained output

5. **Test tools**:
   - [ ] `calculate_gablorken` - Tool calling demo

6. **Test structured output**:
   - [ ] `generate_character` - RPG character generation

7. **Test vision**:
   - [ ] `describe_image` - Image description using GPT-4o vision

8. **Test reasoning**:
   - [ ] `reasoning_flow` - Chain-of-thought reasoning with o4-mini

9. **Test multimodal**:
   - [ ] `generate_image` - DALL-E image generation (returns base64 data URI)
   - [ ] `text_to_speech` - TTS with voice selection (alloy, echo, nova, etc.)
   - [ ] `round_trip_tts_stt` - Text → Speech → Text round-trip demo

10. **Expected behavior**:
    - GPT models respond appropriately
    - Streaming shows incremental text
    - Tools are invoked and responses processed
    - Structured output matches Pydantic schema
    - Vision describes the kitten image accurately
    - Reasoning shows chain-of-thought explanation
    - DALL-E returns a base64 image data URI
    - TTS returns a base64 audio data URI
    - Round-trip returns transcribed text matching the original input
