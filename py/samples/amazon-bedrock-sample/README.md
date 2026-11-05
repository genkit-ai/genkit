# Bedrock

Same `generate()` as Gemini, on the AWS profile you already have.

```bash
export AWS_PROFILE=my-profile
export AWS_REGION=us-west-2
uv sync
uv run src/main.py
```

Grant model access for Nova Lite and Titan Embed in the Bedrock console
first.
