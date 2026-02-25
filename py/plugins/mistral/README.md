# Genkit Mistral AI Plugin (Community)

> **Community Plugin** — This plugin is community-maintained and is not an
> official Google or Mistral AI product. It is provided on an "as-is" basis.
>
> **Preview** — This plugin is in preview and may have API changes in future releases.

This Genkit plugin provides integration with Mistral AI's models, including
Mistral Large, Mistral Small, Codestral, and Pixtral vision models.

## Installation

```bash
pip install genkit-plugin-mistral
```

Or with uv:

```bash
uv add genkit-plugin-mistral
```

## Setup

Set your Mistral API key:

```bash
export MISTRAL_API_KEY=your-api-key
```

Get your API key from: https://console.mistral.ai/api-keys/

## Usage

```python
from genkit import Genkit
from genkit.plugins.mistral import Mistral

ai = Genkit(plugins=[Mistral()], model='mistral/mistral-large-latest')

response = await ai.generate(prompt='Hello, Mistral!')
print(response.text)
```

## Supported Models

| Model | Description |
|-------|-------------|
| `mistral/mistral-large-latest` | Most capable model for complex tasks |
| `mistral/mistral-small-latest` | Fast and efficient for everyday use |
| `mistral/codestral-latest` | Specialized for code generation |
| `mistral/pixtral-large-latest` | Vision-language model |
| `mistral/ministral-8b-latest` | Compact model for edge deployment |
| `mistral/ministral-3b-latest` | Smallest model for resource-constrained environments |

## Features

- **Text Generation**: Standard chat completions
- **Streaming**: Real-time token streaming
- **Vision**: Image understanding with Pixtral models
- **Code Generation**: Specialized coding with Codestral
- **Function Calling**: Tool use support

## Configuration

```python
from genkit.plugins.mistral import Mistral

# With explicit API key
ai = Genkit(plugins=[Mistral(api_key='your-key')])
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `MISTRAL_API_KEY` | Your Mistral AI API key | Yes |

## Disclaimer

This is a **community-maintained** plugin and is not officially supported by
Google or Mistral AI. Use of Mistral's API is subject to
[Mistral AI's Terms of Service](https://mistral.ai/terms/) and
[Privacy Policy](https://mistral.ai/terms/#privacy-policy). You are responsible
for complying with all applicable terms when using this plugin.

- **API Key Security** — Never commit your Mistral API key to version control.
  Use environment variables or a secrets manager.
- **Usage Limits** — Be aware of your Mistral plan's rate limits and token
  quotas. See [Mistral Pricing](https://mistral.ai/products/la-plateforme/).
- **Data Handling** — Review Mistral's data processing practices before
  sending sensitive or personally identifiable information.

## License

Apache-2.0
