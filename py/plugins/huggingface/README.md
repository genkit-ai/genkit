# Genkit Hugging Face Plugin (Community)

> **Community Plugin** — This plugin is community-maintained and is not an
> official Google or Hugging Face product. It is provided on an "as-is" basis.
>
> **Preview** — This plugin is in preview and may have API changes in future releases.

This Genkit plugin provides integration with Hugging Face's Inference API and
Inference Providers, giving access to 1,000,000+ models through a unified interface.

## Installation

```bash
pip install genkit-plugin-huggingface
```

Or with uv:

```bash
uv add genkit-plugin-huggingface
```

## Setup

Set your Hugging Face token:

```bash
export HF_TOKEN=your-token
```

Get your token from: https://huggingface.co/settings/tokens

## Usage

```python
from genkit import Genkit
from genkit.plugins.huggingface import HuggingFace

ai = Genkit(
    plugins=[HuggingFace()],
    model='huggingface/meta-llama/Llama-3.3-70B-Instruct',
)

response = await ai.generate(prompt='Hello, Hugging Face!')
print(response.text)
```

## Popular Models

| Model | Description |
|-------|-------------|
| `huggingface/meta-llama/Llama-3.3-70B-Instruct` | Meta's latest Llama model |
| `huggingface/mistralai/Mistral-Small-24B-Instruct-2501` | Mistral's efficient model |
| `huggingface/Qwen/Qwen2.5-72B-Instruct` | Alibaba's multilingual model |
| `huggingface/deepseek-ai/DeepSeek-R1` | DeepSeek's reasoning model |
| `huggingface/google/gemma-2-27b-it` | Google's open Gemma model |
| `huggingface/microsoft/phi-4` | Microsoft's compact Phi model |

## Features

- **Text Generation**: Standard chat completions
- **Streaming**: Real-time token streaming
- **Inference Providers**: Access 17+ providers (Cerebras, Groq, Together, etc.)
- **Any Model**: Use any of 1M+ models on Hugging Face Hub

## Configuration

```python
from genkit.plugins.huggingface import HuggingFace

# With explicit token
ai = Genkit(plugins=[HuggingFace(token='your-token')])

# With specific inference provider
ai = Genkit(plugins=[HuggingFace(provider='cerebras')])
```

## Inference Providers

Hugging Face supports routing requests to different inference providers for
better performance or cost optimization:

- Cerebras
- Groq
- Together
- Fireworks
- Replicate
- And more...

```python
# Use a specific provider
ai = Genkit(plugins=[HuggingFace(provider='groq')])
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `HF_TOKEN` | Your Hugging Face API token | Yes |

## Notes

- Free tier has rate limits; consider [HF Pro](https://huggingface.co/pricing) for higher limits
- Some models require accepting terms on huggingface.co first
- Model IDs use the format `owner/model-name` (e.g., `meta-llama/Llama-3.3-70B-Instruct`)

## Disclaimer

This is a **community-maintained** plugin and is not officially supported by
Google or Hugging Face. Use of Hugging Face's API is subject to
[Hugging Face's Terms of Service](https://huggingface.co/terms-of-service) and
[Privacy Policy](https://huggingface.co/privacy). You are responsible for
complying with all applicable terms when using this plugin.

- **API Token Security** — Never commit your Hugging Face token to version
  control. Use environment variables or a secrets manager.
- **Model Licensing** — Individual models on Hugging Face Hub have their own
  licenses. Review model cards before use.
- **Usage Limits** — Free tier has rate limits; consider
  [HF Pro](https://huggingface.co/pricing) for higher limits.

## License

Apache-2.0
