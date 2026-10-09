# Genkit Anthropic Plugin

Anthropic Claude model provider for Genkit.

> **Building with a coding agent? Install the Genkit Python skill first.**
>
> ```bash
> npx skills add genkit-ai/skills --skill developing-genkit-python
> ```
>
> It teaches your agent the current Genkit Python APIs and common gotchas.
> Source, manual install and skills for other languages:
> [genkit-ai/skills](https://github.com/genkit-ai/skills).

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

## Per-request API key

To bill a call to someone else's Anthropic account, pass their key in
`context.secrets`. It's used for that call only:

```python
res = await ai.generate(
    model='anthropic/claude-sonnet-4-6',
    prompt='hi',
    context={'secrets': {'api_key': tenant_key}},
)
```

- `apiKey` works as well as `api_key`. Other entries in `context.secrets`
  are left alone, so a call whose secrets hold no key runs on the plugin's
  key. A key that is blank or not a string raises `INVALID_ARGUMENT`.
- A key in `config` or `config.extra` raises `INVALID_ARGUMENT`; config is
  recorded in traces.
- The plugin doesn't need a key of its own: `Anthropic()` with no
  `ANTHROPIC_API_KEY` serves every call that brings a key in
  `context.secrets`, and a call without one fails with `FAILED_PRECONDITION`.
- A client whose credential can't be swapped refuses a tenant key with
  `FAILED_PRECONDITION` rather than billing its own account: an
  `Anthropic(auth_token=...)` client, one whose `default_headers` pin
  `x-api-key` or `Authorization`, and Claude on Vertex AI Model Garden, which
  authenticates with Google Cloud credentials.

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
