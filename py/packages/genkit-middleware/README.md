# Genkit Middleware Plugin

A collection of middleware implementations for Genkit Python.

## Overview

This plugin provides six concrete middleware implementations for common use cases:

- **Retry**: Retries model API calls on transient errors with exponential backoff
- **Fallback**: Falls back to alternative models when the primary model fails
- **ToolApproval**: Requires explicit approval before executing tool calls
- **Skills**: Exposes a library of skills as system prompts and tools
- **Filesystem**: Provides sandboxed filesystem operations
- **Artifacts**: Session artifact listing plus read/write artifact tools

## Quick start

Import the middleware classes you need and pass instances directly into `use=[]`:

```python
from genkit import Genkit
from genkit_google_genai import GoogleAI
from genkit_middleware import Retry, Fallback, Middleware

ai = Genkit(plugins=[GoogleAI(), Middleware()])

response = await ai.generate(
    model=GoogleAI.gemini_model('gemini-flash-latest'),
    prompt='Hello!',
    use=[
        Retry(max_retries=5),
        Fallback(models=['googleai/gemini-2.5-pro']),
    ],
)
```

These middlewares appear in the Dev UI by default.

## Installation

```bash
uv add genkit-middleware genkit-google-genai
```

## Usage

### Retry

Automatically retries model calls on transient failures with configurable exponential backoff:

```python
from genkit_google_genai import GoogleAI
from genkit_middleware import Retry

retry = Retry(
    max_retries=3,
    statuses=['UNAVAILABLE', 'DEADLINE_EXCEEDED', 'RESOURCE_EXHAUSTED'],
    initial_delay_ms=1000,
    max_delay_ms=60000,
    backoff_factor=2.0,
    no_jitter=False,  # set True for deterministic backoff (tests)
)

response = await ai.generate(
    model=GoogleAI.gemini_model('gemini-flash-latest'),
    prompt='Hello!',
    use=[retry],
)
```

### Fallback

Falls back to alternative models on retryable errors:

```python
from genkit_google_genai import GoogleAI
from genkit_middleware import Fallback

fallback = Fallback(
    models=[
        'googleai/gemini-2.5-pro',
        'googleai/gemini-flash-latest',
    ],
    statuses=['UNAVAILABLE', 'DEADLINE_EXCEEDED'],
)

response = await ai.generate(
    model=GoogleAI.gemini_model('gemini-pro-latest'),
    prompt='Hello!',
    use=[fallback],
)
```

### ToolApproval

Requires approval before executing tools (useful for sensitive operations):

```python
from pydantic import BaseModel, Field

from genkit import restart_tool
from genkit_google_genai import GoogleAI
from genkit_middleware import ToolApproval


class DeleteInput(BaseModel):
    name: str = Field(description='Database name to delete')


@ai.tool()
async def delete_database(input: DeleteInput) -> str:
    return f'Deleted {input.name}'


approval = ToolApproval(
    allowed_tools=['get_weather', 'search'],  # These tools run without approval
)

first = await ai.generate(
    model=GoogleAI.gemini_model('gemini-flash-latest'),
    prompt='Delete the database',
    tools=['delete_database'],
    use=[approval],
)
```

When a non-allowed tool is called, execution is interrupted. Approve and re-run the
tool by restarting it with ``resumed_metadata`` that includes ``tool_approved``:

```python
response = await ai.generate(
    model=GoogleAI.gemini_model('gemini-flash-latest'),
    prompt='Delete the database',
    messages=list(first.messages),
    tools=['delete_database'],
    use=[approval],
    resume_restart=restart_tool(
        interrupt=first.interrupts[0],
        resumed_metadata={'tool_approved': True},
    ),
)
```

### Skills

Scans directories for SKILL.md files and exposes them as loadable instructions:

```python
from genkit_google_genai import GoogleAI
from genkit_middleware import Skills

skills = Skills(
    skill_paths=['skills', 'prompts/skills'],
)

response = await ai.generate(
    model=GoogleAI.gemini_model('gemini-flash-latest'),
    prompt='Help me with Python',
    use=[skills],
)
```

Skills are discovered by scanning for directories containing `SKILL.md` files. Each `SKILL.md` can have optional YAML frontmatter:

```markdown
---
name: python-expert
description: Expert Python programming assistance
---

You are an expert Python programmer...
```

### Filesystem

Provides sandboxed file operations confined to a root directory:

```python
from genkit_google_genai import GoogleAI
from genkit_middleware import Filesystem

fs = Filesystem(
    root_dir='./workspace',
    allow_write_access=True,
    tool_name_prefix='',
)

response = await ai.generate(
    model=GoogleAI.gemini_model('gemini-flash-latest'),
    prompt='List files in the current directory',
    use=[fs],
)
```

Provides four tools:
- `list_files`: List files in a directory
- `read_file`: Read file content
- `write_file`: Write to a file (requires `allow_write_access=True`)
- `edit_file`: Edit file with string replacements (requires `allow_write_access=True`)

### Artifacts

Exposes `read_artifact` / `write_artifact` tools and lists session artifacts in the
system prompt. Intended for agent sessions:

```python
from genkit_middleware import Artifacts, Middleware

from genkit import Genkit
from genkit.agent import InMemorySessionStore
from genkit_google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI(), Middleware()])

agent = ai.define_agent(
    name='workspaceAgent',
    model=GoogleAI.gemini_model('gemini-flash-latest'),
    use=[Artifacts()],
    store=InMemorySessionStore(),
)

chat = agent.chat()
await chat.send('Write poem.txt with a short poem about Python agents.')
# chat.artifacts now includes poem.txt
```
