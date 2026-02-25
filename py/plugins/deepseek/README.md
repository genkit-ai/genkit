# Genkit DeepSeek Plugin (Community)

> **Community Plugin** — This plugin is community-maintained and is not an
> official Google or DeepSeek product. It is provided on an "as-is" basis.
>
> **Preview** — This plugin is in preview and may have API changes in future releases.

This Genkit plugin provides integration with DeepSeek's AI models, including
their powerful reasoning model (R1) and general-purpose chat model (V3).

## Installation

```bash
pip install genkit-plugin-deepseek
```

Or with uv:

```bash
uv add genkit-plugin-deepseek
```

## Setup

Set your DeepSeek API key:

```bash
export DEEPSEEK_API_KEY=your-api-key
```

Get your API key from: https://platform.deepseek.com/api_keys

## Usage

```python
from genkit import Genkit
from genkit.plugins.deepseek import DeepSeek

ai = Genkit(plugins=[DeepSeek()], model='deepseek/deepseek-chat')

response = await ai.generate(prompt='Hello, DeepSeek!')
print(response.text)
```

## Supported Models

| Model | Description |
|-------|-------------|
| `deepseek/deepseek-chat` | General-purpose chat model (V3). Fast and capable. |
| `deepseek/deepseek-reasoner` | Reasoning model (R1). Shows chain-of-thought. |

## Features

- **Text Generation**: Standard chat completions
- **Streaming**: Real-time token streaming
- **Chain-of-Thought**: R1 model shows reasoning steps
- **OpenAI-Compatible**: Uses familiar API format

## Configuration

```python
from genkit.plugins.deepseek import DeepSeek

# With explicit API key
ai = Genkit(plugins=[DeepSeek(api_key='your-key')])

# With custom API URL (for proxies)
ai = Genkit(plugins=[DeepSeek(api_url='https://your-proxy.com')])
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DEEPSEEK_API_KEY` | Your DeepSeek API key | Yes |

## Disclaimer

This is a **community-maintained** plugin and is not officially supported by
Google or DeepSeek. Use of DeepSeek's API is subject to
[DeepSeek's Terms of Use](https://chat.deepseek.com/downloads/DeepSeek%20Terms%20of%20Use.html)
and [Privacy Policy](https://chat.deepseek.com/downloads/DeepSeek%20Privacy%20Policy.html).
You are responsible for complying with all applicable terms when using this plugin.

- **API Key Security** — Never commit your DeepSeek API key to version control.
  Use environment variables or a secrets manager.
- **Usage Limits** — Be aware of your DeepSeek plan's rate limits and token
  quotas. See [DeepSeek Pricing](https://api-docs.deepseek.com/quick_start/pricing).
- **Data Handling** — Review DeepSeek's data processing practices before
  sending sensitive or personally identifiable information.

## License

Apache-2.0
