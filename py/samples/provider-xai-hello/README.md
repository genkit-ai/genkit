# xAI Hello Sample

Simple sample demonstrating the xAI (Grok) plugin for Genkit.

## How to Get Your xAI API Key

To use the xAI plugin, you need an API key from xAI.

1.  **Visit the xAI Console**: Go to the [xAI Console](https://console.x.ai/) and sign in.
2.  **Create an API Key**: Navigate to the API Keys section and create a new key.
3.  **Add Credits**: You may need to add credits to your account to use the API.

For a more detailed guide, check out the [official tutorial](https://docs.x.ai/docs/tutorial).

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

## Setup


```bash
export XAI_API_KEY=your_api_key_here
```

## Run

```bash
cd py/samples/provider-xai-hello
uv run src/main.py
```

## Features Demonstrated

| Feature | Flow | Description |
|---------|------|-------------|
| Simple Generation | `say_hi` | Basic text generation with Grok |
| Streaming | `say_hi_stream` | Token-by-token streaming response |
| Generation Config | `say_hi_with_config` | Custom temperature, max_output_tokens |
| Tool Calling | `weather_flow` | Weather tool with function calling |
| Math Tool | `calculate` | Math calculation tool |
| Vision | `describe_image` | Image description using Grok Vision |
| Reasoning (CoT) | `reasoning_flow` | Chain-of-thought with Grok 4 |
| xAI Parameters | `reasoning_effort` | xAI-specific config options |

## Testing This Demo

1. **Open DevUI** at http://localhost:4000

2. **Test basic flows**:
   - [ ] `say_hi` - Simple text generation
   - [ ] `say_hi_stream` - Streaming response
   - [ ] `say_hi_with_config` - Custom temperature

3. **Test tools**:
   - [ ] `weather_flow` - Weather tool calling
   - [ ] `calculate` - Math calculation tool

4. **Test vision**:
   - [ ] `describe_image` - Image description using Grok Vision

5. **Test reasoning**:
   - [ ] `reasoning_flow` - Chain-of-thought reasoning with Grok 4

6. **Expected behavior**:
   - Grok responds with characteristic wit
   - Streaming shows incremental output
   - Tools are invoked correctly
   - Vision describes the image accurately
   - Reasoning shows chain-of-thought explanation
