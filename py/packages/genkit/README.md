# genkit

[![PyPI version](https://img.shields.io/pypi/v/genkit)](https://pypi.org/project/genkit/) [![Python versions](https://img.shields.io/pypi/pyversions/genkit)](https://pypi.org/project/genkit/) [![Downloads](https://img.shields.io/pypi/dm/genkit)](https://pypi.org/project/genkit/) [![License](https://img.shields.io/pypi/l/genkit)](https://github.com/genkit-ai/genkit/blob/main/LICENSE)

Genkit is a Python SDK from Google. One API for generate, tools, structured output, and agents, plus a local Developer UI.

Vertex AI, Cloud Trace, and Firestore are there if you want them. So are OpenAI, Anthropic, Ollama, and Bedrock.

## Install

```bash
uv add genkit genkit-google-genai
```

```python
from pydantic import BaseModel, Field
from genkit import Genkit
from genkit_google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI()], model=GoogleAI.gemini_model('gemini-flash-latest'))


class Issue(BaseModel):
    title: str = Field(description='Short title')
    severity: str = Field(description='critical, warning, or info')
    suggestion: str = Field(description='How to fix it')


@ai.flow()
async def review(code: str) -> Issue:
    result = await ai.generate(
        prompt=f'Review this code:\n{code}',
        output_schema=Issue,
    )
    return result.output


async def main() -> None:
    print((await review('eval(user_input)')).model_dump_json(indent=2))


if __name__ == '__main__':
    ai.run_main(main())
```

See https://python.api.genkit.dev for more details.
