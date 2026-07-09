# Genkit Middleware Plugin

A collection of middleware implementations for Firebase Genkit Python.

## Overview

This plugin provides five concrete middleware implementations for common use cases:

- **Retry**: Retries model API calls on transient errors with exponential backoff
- **Fallback**: Falls back to alternative models when the primary model fails
- **ToolApproval**: Requires explicit approval before executing tool calls
- **Skills**: Exposes a library of skills as system prompts and tools
- **Filesystem**: Provides sandboxed filesystem operations

## Quick start

Import the middleware classes you need and pass instances directly into `use=[]`:

```python
from genkit import Genkit
from genkit_middleware import Retry, Fallback, Middleware

ai = Genkit(plugins=[Middleware()])

response = await ai.generate(
    model='googleai/gemini-flash-latest',
    prompt='Hello!',
    use=[
        Retry(max_retries=5),
        Fallback(models=['googleai/gemini-2.5-pro']),
    ],
)
```

These pre-packaged middlewares will be available to play with in the Dev UI by default.

## Installation

```bash
pip install genkit-plugin-middleware
```

## Usage

### Retry

Automatically retries model calls on transient failures with configurable exponential backoff:

```python
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
    model='googleai/gemini-flash-latest',
    prompt='Hello!',
    use=[retry],
)
```

### Fallback

Falls back to alternative models on retryable errors:

```python
from genkit_middleware import Fallback

fallback = Fallback(
    models=['googleai/gemini-2.5-pro', 'googleai/gemini-flash-latest'],
    statuses=['UNAVAILABLE', 'DEADLINE_EXCEEDED'],
)

response = await ai.generate(
    model='googleai/gemini-2.5-ultra',
    prompt='Hello!',
    use=[fallback],
)
```

### ToolApproval

Requires approval before executing tools (useful for sensitive operations):

```python
from genkit_middleware import ToolApproval

approval = ToolApproval(
    allowed_tools=['get_weather', 'search'],  # These tools run without approval
)

response = await ai.generate(
    model='googleai/gemini-flash-latest',
    prompt='Delete the database',
    tools=[delete_database_tool],
    use=[approval],
)
```

When a non-allowed tool is called, execution is interrupted. Approve and re-run the
tool by restarting it with ``resumed_metadata`` that includes ``toolApproved``
(the middleware only treats explicit dict metadata as approval):

```python
first = await ai.generate(
    model='googleai/gemini-flash-latest',
    prompt='Delete the database',
    tools=[delete_database_tool],
    use=[approval],
)

from genkit import restart_tool

response = await ai.generate(
    model='googleai/gemini-flash-latest',
    prompt='Delete the database',
    messages=list(first.messages),
    tools=[delete_database_tool],
    use=[approval],
    resume_restart=restart_tool(
        interrupt=first.interrupts[0],
        resumed_metadata={'toolApproved': True},
    ),
)
```

### Skills

Scans directories for SKILL.md files and exposes them as loadable instructions:

```python
from genkit_middleware import Skills

skills = Skills(
    skill_paths=['skills', 'prompts/skills'],
)

response = await ai.generate(
    model='googleai/gemini-flash-latest',
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
from genkit_middleware import Filesystem

fs = Filesystem(
    root_dir='./workspace',
    allow_write_access=True,
    tool_name_prefix='',
)

response = await ai.generate(
    model='googleai/gemini-flash-latest',
    prompt='List files in the current directory',
    use=[fs],
)
```

Provides four tools:
- `list_files`: List files in a directory
- `read_file`: Read file content
- `write_file`: Write to a file (requires `allow_write_access=True`)
- `edit_file`: Edit file with string replacements (requires `allow_write_access=True`)

