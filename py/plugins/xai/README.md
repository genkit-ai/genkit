# Genkit xAI Plugin (Community)

> **Community Plugin** — This plugin is community-maintained and is not an
> official Google or xAI product. It is provided on an "as-is" basis.
>
> **Preview** — This plugin is in preview and may have API changes in future releases.

This Genkit plugin provides integration with xAI's Grok models for
text generation, streaming, and tool calling.

## Installation

```bash
pip install genkit-plugin-xai
```

## Setup

Set your xAI API key:

```bash
export XAI_API_KEY=your-api-key
```

Get your API key from: https://console.x.ai/

## Usage

```python
from genkit import Genkit
from genkit.plugins.xai import XAI

ai = Genkit(plugins=[XAI()], model='xai/grok-3')

response = await ai.generate(prompt='Hello, Grok!')
print(response.text)
```

## Disclaimer

This is a **community-maintained** plugin and is not officially supported by
Google or xAI. Use of xAI's API is subject to
[xAI's Terms of Service](https://x.ai/legal/terms-of-service) and
[Privacy Policy](https://x.ai/legal/privacy-policy). You are responsible for
complying with all applicable terms when using this plugin.

- **API Key Security** — Never commit your xAI API key to version control.
  Use environment variables or a secrets manager.
- **Usage Limits** — Be aware of your xAI plan's rate limits and token
  quotas. See [xAI Pricing](https://x.ai/api#pricing).
- **Data Handling** — Review xAI's data processing practices before
  sending sensitive or personally identifiable information.

## License

Apache-2.0
