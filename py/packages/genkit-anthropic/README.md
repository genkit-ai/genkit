# Genkit Anthropic Plugin

Anthropic Claude model provider for Genkit.

## Installation

```bash
uv add genkit-anthropic
```

## Usage

```python
from genkit import Genkit
from genkit_anthropic import Anthropic

ai = Genkit(plugins=[Anthropic()])

res = await ai.generate(
    model=Anthropic.claude_model('claude-sonnet-4-6'),
    prompt='Explain recursion in 10 words.',
)
print(res.text)
```

Set `ANTHROPIC_API_KEY` in the environment, or pass `api_key=` to `Anthropic()`.

## Disclaimer

Use of Anthropic's API is subject to
[Anthropic's Terms of Service](https://www.anthropic.com/terms) and
[Privacy Policy](https://www.anthropic.com/privacy). You are responsible for
complying with all applicable terms when using this plugin.

- **API Key Security** — Never commit your Anthropic API key to version control.
  Use environment variables or a secrets manager.
- **Usage Limits** — Be aware of your Anthropic plan's rate limits and token
  quotas. See [Anthropic Pricing](https://www.anthropic.com/pricing).
- **Data Handling** — Review Anthropic's data processing practices before
  sending sensitive or personally identifiable information.

## License

Apache-2.0
