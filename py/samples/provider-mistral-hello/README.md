# Mistral AI Sample

## Features Demonstrated

| Feature | Flow | Description |
|---------|------|-------------|
| Simple Generation | `say_hi` | Basic text generation with mistral-small |
| Streaming | `streaming_flow` | Token-by-token streaming response |
| Generation Config | `custom_config_flow` | Custom temperature and config |
| Code Generation | `generate_code` | Code generation with Codestral |
| Large Model | `large_model_flow` | Complex reasoning with mistral-large |
| Multi-turn Chat | `chat_flow` | Context-preserving conversations |
| Embeddings | `embed_flow` | Text embeddings with mistral-embed |

## How to Get Your Mistral API Key

A Mistral API key is a secret token for accessing Mistral AI's powerful models.

**Steps:**
1. **Sign Up/Login**: Go to [console.mistral.ai](https://console.mistral.ai/) and create an account
2. **Navigate to API Keys**: Find the "API Keys" section in your dashboard
3. **Create Key**: Click "Create new key" and give it a name
4. **Copy Key**: Copy the generated key immediately as it's shown only once
5. **Add Credits (if needed)**: You may need to add a payment method for usage

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

2. Set Mistral API key:
```bash
export MISTRAL_API_KEY=your-api-key
```

3. Run the sample:
```bash
genkit start -- uv run src/main.py
```

## Testing This Demo

1. **Open DevUI** at http://localhost:4000

2. **Test basic flows**:
   - [ ] `say_hi` - Simple generation with mistral-small
   - [ ] `streaming_flow` - Streaming response
   - [ ] `custom_config_flow` - Custom temperature/config

3. **Test code generation**:
   - [ ] `generate_code` - Code generation with Codestral

4. **Test large model**:
   - [ ] `large_model_flow` - Complex reasoning with mistral-large

5. **Test chat**:
   - [ ] `chat_flow` - Multi-turn conversation

6. **Test embeddings**:
   - [ ] `embed_flow` - Generate embeddings using mistral-embed

7. **Expected behavior**:
   - mistral-small: Fast, capable responses
   - mistral-large: More detailed, nuanced responses
   - codestral: High-quality code generation

## Available Models

| Model | Best For |
|-------|----------|
| `mistral-small-latest` | Everyday tasks, fast responses |
| `mistral-large-latest` | Complex reasoning, nuanced tasks |
| `codestral-latest` | Code generation and explanation |
| `pixtral-large-latest` | Vision tasks (image understanding) |
| `ministral-8b-latest` | Edge deployment, resource-constrained |
| `mistral-embed` | Text embeddings for search and RAG |
