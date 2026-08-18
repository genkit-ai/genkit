# Amazon Bedrock

Run text generation, structured output, tool calling, and reasoning through
Genkit with Amazon Bedrock's Converse API.

You need an AWS account with Amazon Bedrock model access granted for the four
models the sample uses:

- `us.amazon.nova-lite-v1:0`
- `us.meta.llama3-3-70b-instruct-v1:0`
- `us.deepseek.r1-v1:0`
- `us.anthropic.claude-sonnet-4-5-20250929-v1:0`

The Anthropic model additionally needs the account's one-time use-case
agreement (Bedrock console, Model access, Anthropic use case details); the
`thinking` flow fails with `ResourceNotFoundException` until it is granted.

Credentials come from the standard AWS chain; environment variables, an
`AWS_PROFILE` (including SSO profiles after `aws sso login`), or instance
credentials. A region is required, from `AWS_REGION`, `AWS_DEFAULT_REGION`, or
the active profile:

```bash
export AWS_PROFILE=my-profile
export AWS_REGION=us-east-1
```

Run the quick smoke test:

```bash
uv sync
uv run src/main.py
```

To explore all flows in Dev UI instead:

```bash
genkit start -- uv run src/main.py
```

Then open [http://localhost:4000](http://localhost:4000) and try:

- `haiku`
- `cat_profile`
- `weather_report`
- `reasoning`
- `thinking`

The plugin resolves any Bedrock model ID, inference profile, or ARN on demand,
so the Dev UI model runner also works with models beyond the four declared
ones.

Bedrock has no constrained-decoding mode, so structured output is carried by
prompt instructions: pass `output_instructions=True` alongside `output_format`
and `output_schema`, as `cat_profile` does. Without it the schema never reaches
the model and it answers in prose.
