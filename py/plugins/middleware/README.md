# Genkit Middleware Plugin

A collection of middleware implementations for Firebase Genkit Python.

## Overview

This plugin provides five concrete middleware implementations for common use cases:

- **Retry**: Retries model API calls on transient errors with exponential backoff
- **Fallback**: Falls back to alternative models when the primary model fails
- **ToolApproval**: Requires explicit approval before executing tool calls
- **Skills**: Exposes a library of skills as system prompts and tools
- **Filesystem**: Provides sandboxed filesystem operations

## Installation

```bash
pip install genkit-plugin-middleware
```

## Usage

### Retry

Automatically retries model calls on transient failures with configurable exponential backoff:

```python
from genkit.plugins.middleware import Retry

retry = Retry(
    max_retries=3,
    statuses=['UNAVAILABLE', 'DEADLINE_EXCEEDED', 'RESOURCE_EXHAUSTED'],
    initial_delay_ms=1000,
    max_delay_ms=60000,
    backoff_factor=2.0,
    no_jitter=False,
)

response = await ai.generate(
    model='googleai/gemini-2.5-flash',
    prompt='Hello!',
    use=[retry],
)
```

### Fallback

Falls back to alternative models on retryable errors:

```python
from genkit.plugins.middleware import Fallback

fallback = Fallback(
    models=['googleai/gemini-2.5-pro', 'googleai/gemini-2.5-flash'],
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
from genkit.plugins.middleware import ToolApproval

approval = ToolApproval(
    allowed_tools=['get_weather', 'search'],  # These tools run without approval
)

response = await ai.generate(
    model='googleai/gemini-2.5-flash',
    prompt='Delete the database',
    tools=[delete_database_tool],
    use=[approval],
)
```

When a non-allowed tool is called, execution is interrupted. Resume with:

```python
response = await ai.generate(
    # ... same params ...
    resume=Resume(
        interrupted_tool_metadata={'resumed': {'toolApproved': True}}
    ),
)
```

### Skills

Scans directories for SKILL.md files and exposes them as loadable instructions:

```python
from genkit.plugins.middleware import Skills

skills = Skills(
    skill_paths=['skills', 'prompts/skills'],
)

response = await ai.generate(
    model='googleai/gemini-2.5-flash',
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
from genkit.plugins.middleware import Filesystem

fs = Filesystem(
    root_dir='./workspace',
    allow_write_access=True,
    tool_name_prefix='',
)

response = await ai.generate(
    model='googleai/gemini-2.5-flash',
    prompt='List files in the current directory',
    use=[fs],
)
```

Provides four tools:
- `list_files`: List files in a directory
- `read_file`: Read file content
- `write_file`: Write to a file (requires `allow_write_access=True`)
- `edit_file`: Edit file with string replacements (requires `allow_write_access=True`)

## Development

```bash
cd py/plugins/middleware
pip install -e ".[dev]"
pytest tests/
```

## License

Apache 2.0
