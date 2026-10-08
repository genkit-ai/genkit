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

## Typed config

`AnthropicConfig` is flat. Choices are exported Literals (`ThinkingMode`,
`ThinkingDisplay`, `Effort`), so pyright, pyrefly and ty check every key and
value. The plugin builds Anthropic's nested request objects from the flat
fields:

- `thinking`, `thinking_budget`, `thinking_display` → `thinking`
- `effort`, `task_budget` → `output_config`
- `user_id` → `metadata`

```python
from genkit_anthropic import AnthropicConfig

# 1. Build the config from flat fields
config = AnthropicConfig(
    max_output_tokens=4096,
    thinking='adaptive',
    thinking_display='summarized',
    effort='high',
    task_budget=20000,
    user_id='diner-42',
)

# 2. Generate with it
res = await ai.generate(model='anthropic/claude-opus-4-6', prompt='Plan a tasting menu.', config=config)
# Sent as:
#   thinking={'display': 'summarized', 'type': 'adaptive'}
#   output_config={'effort': 'high', 'task_budget': {'type': 'tokens', 'total': 20000}}
#   metadata={'user_id': 'diner-42'}
```

- A `thinking_budget` alone means `thinking='enabled'`. `'enabled'` requires
  a budget, `'disabled'` rejects one, and `'adaptive'` ignores it.
- `task_budget` is a beta field, so it sends the call to the beta API.
- To send a shape the flat fields don't cover, put Anthropic's object in
  `extra`, e.g. `extra={'thinking': {...}}`. It replaces the built object.

## Tool choice

Tool choice is Genkit's `tool_choice` option, not a config field. The plugin
sends it as Anthropic's `tool_choice` object: `'auto'` → `{'type': 'auto'}`,
`'required'` → `{'type': 'any'}`, `'none'` → `{'type': 'none'}`. Anthropic
carries `disable_parallel_tool_use` inside that object, so the config takes it
as a flat flag:

```python
# 1. Require one tool call, and only one
res = await ai.generate(
    model='anthropic/claude-sonnet-4-6',
    prompt='Is the pho in stock?',
    tools=['lookup_menu'],
    tool_choice='required',
    config=AnthropicConfig(disable_parallel_tool_use=True),
)
# Sent as: tool_choice={'type': 'any', 'disable_parallel_tool_use': True}
```

To pin one named tool, send Anthropic's object as
`extra={'tool_choice': {'type': 'tool', 'name': 'lookup_menu'}}`; it replaces
the translated value.

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
