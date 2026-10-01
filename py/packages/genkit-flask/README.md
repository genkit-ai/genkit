# Genkit Flask Plugin

Expose Genkit flows as HTTP endpoints in a Flask application.

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
uv add genkit-flask genkit-google-genai
```

## Usage

```python
from flask import Flask
from genkit import Genkit
from genkit_flask import genkit_flask_handler
from genkit_google_genai import GoogleAI

app = Flask(__name__)
ai = Genkit(plugins=[GoogleAI()], model=GoogleAI.gemini_model('gemini-flash-latest'))


@app.post('/api/greet')
@genkit_flask_handler(ai)
@ai.flow()
async def greet_user(name: str) -> str:
    res = await ai.generate(prompt=f'Say hello to {name} in one sentence.')
    return res.text
```

Requires Flask 3.1+.
