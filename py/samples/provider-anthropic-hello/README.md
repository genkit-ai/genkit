## Anthropic Sample

### Features Demonstrated

| Feature | Flow | Description |
|---------|------|-------------|
| Simple Generation | `say_hi` | Basic text generation with Claude |
| Streaming | `say_hi_stream` | Token-by-token streaming response |
| Generation Config | `say_hi_with_config` | Custom temperature settings |
| Tool Calling | `weather_flow` | Weather tool with function calling |
| Currency Tool | `currency_exchange` | Currency conversion tool |
| Thinking Mode (CoT) | `thinking_demo` | Chain-of-thought reasoning |
| Multimodal (Image) | `describe_image` | Image description with Claude |
| Prompt Caching | `cached_generation` | Cache breakpoint metadata |
| PDF Input | `analyze_pdf` | PDF document analysis |

### How to Get Your Anthropic API Key

To use the Anthropic plugin, you need an API key from Anthropic.

1.  **Visit the Anthropic Console**: Go to the [Anthropic Console](https://console.anthropic.com/settings/keys) and sign in.
2.  **Create an API Key**: Click on "Create Key" to generate a new API key.
3.  **Add Credits**: You may need to add credits to your account to use the API.

For more details, check out the [official API overview](https://platform.claude.com/docs/en/api/overview).

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

### Usage

1. Setup environment and install dependencies:
```bash
uv venv
source .venv/bin/activate

uv sync
```

2. Set Anthropic API key:
```bash
export ANTHROPIC_API_KEY=your-api-key
```

3. Run the sample:
```bash
genkit start -- uv run src/main.py
```

## Testing This Demo

1. **Open DevUI** at http://localhost:4000

2. **Test basic flows**:
   - [ ] `say_hi` - Simple greeting generation
   - [ ] `say_hi_stream` - Streaming response
   - [ ] `say_hi_with_config` - Custom temperature

3. **Test tools**:
   - [ ] `weather_flow` - Weather tool calling
   - [ ] `currency_exchange` - Currency conversion tool

4. **Test advanced features**:
   - [ ] `thinking_demo` - Chain-of-thought reasoning
   - [ ] `describe_image` - Image description (multimodal)
   - [ ] `cached_generation` - Prompt caching demo
   - [ ] `analyze_pdf` - PDF document input demo

5. **Expected behavior**:
   - Claude responds appropriately to prompts
   - Tools are invoked and responses integrated
   - Thinking mode shows reasoning process
   - Image descriptions are accurate
   - Cache breakpoint metadata is applied (check traces)
   - PDF content is analyzed and described correctly
