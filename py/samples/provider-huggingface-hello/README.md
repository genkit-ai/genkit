# Hugging Face Sample

## Features Demonstrated

| Feature | Flow | Description |
|---------|------|-------------|
| Simple Generation | `say_hi` | Basic text generation with Llama 3.1 |
| Streaming | `streaming_flow` | Token-by-token streaming response |
| Generation Config | `custom_config_flow` | Custom temperature and config |
| Multi-model | `llama_flow` / `qwen_flow` / `gemma_flow` | Different model providers |
| Multi-turn Chat | `chat_flow` | Context-preserving conversations |
| Inference Providers | `provider='auto'` | Auto-select best provider per model |

## How to Get Your Hugging Face Token

A Hugging Face token is required to access the Inference API.

**Steps:**
1. **Sign Up/Login**: Go to [huggingface.co](https://huggingface.co/) and create an account
2. **Navigate to Settings**: Click your profile icon → Settings → Access Tokens
3. **Create Token**: Click "New token", select "Read" access (or "Write" if needed)
4. **Copy Token**: Copy the generated token

## Monitoring and Running

For an enhanced development experience, use the provided `run.sh` script:

```bash
./run.sh
```

This script uses `watchmedo` to monitor changes in:
- `src/` (Python logic)
- `../../packages` (Genkit core)
- `../../plugins` (Genkit plugins)
- File patterns: `*.py`, `*.prompt`, `*.json`

Changes will automatically trigger a restart of the sample.

## Usage

1. Setup environment and install dependencies:
```bash
uv venv
source .venv/bin/activate
uv sync
```

2. Set Hugging Face token:
```bash
export HF_TOKEN=your-token
```

3. Run the sample:
```bash
genkit start -- uv run src/main.py
```

## Testing This Demo

1. **Open DevUI** at http://localhost:4000

2. **Test basic flows**:
   - [ ] `say_hi` - Simple generation with Llama 3.1
   - [ ] `streaming_flow` - Streaming response
   - [ ] `custom_config_flow` - Custom temperature/config

3. **Test different models**:
   - [ ] `llama_flow` - Meta's Llama model
   - [ ] `qwen_flow` - Alibaba's Qwen model
   - [ ] `gemma_flow` - Google's Gemma model

4. **Test chat**:
   - [ ] `chat_flow` - Multi-turn conversation

5. **Expected behavior**:
   - All models should generate coherent responses
   - Streaming should show text appearing gradually
   - Chat should maintain context across turns

## Popular Models

You can use ANY model from huggingface.co! Here are some popular ones:

| Model ID | Description |
|----------|-------------|
| `meta-llama/Llama-3.3-70B-Instruct` | Meta's latest Llama |
| `meta-llama/Llama-3.1-8B-Instruct` | Smaller, faster Llama |
| `Qwen/Qwen2.5-72B-Instruct` | Alibaba's powerful model |
| `google/gemma-2-27b-it` | Google's open Gemma |
| `deepseek-ai/DeepSeek-R1` | DeepSeek reasoning model |

## Using Inference Providers

HuggingFace routes model requests through third-party inference providers.
**A provider is required for most models** — without one, you may get a
`400 Bad Request: not a chat model` error.

```python
ai = Genkit(
    plugins=[HuggingFace(provider='auto')],  # Auto-select best provider per model
    model='huggingface/meta-llama/Llama-3.1-8B-Instruct',
)
```

Using `provider='auto'` lets HuggingFace's routing infrastructure automatically
select a compatible provider for each model. You can also pin a specific provider
(e.g., `'novita'`, `'cerebras'`, `'groq'`, `'together'`, `'fireworks-ai'`), but
note that not all models are available on every provider.

## Rate Limits

The free tier has rate limits. For higher limits:
- Upgrade to HF Pro ($9/month)
- Use Inference Providers (separate billing)
- Deploy your own Inference Endpoint
