# Multi-Runtime Support

The `conform` tool tests Genkit plugins across multiple runtimes
(Python, JavaScript, Go) simultaneously.

## How it works

Each runtime has its own `[tool.conform.runtimes.<name>]` section in
`pyproject.toml` defining how to locate specs, plugins, and execute
entry points:

| Key | Purpose |
|-----|---------|
| `specs-dir` | Directory containing conformance specs |
| `plugins-dir` | Directory containing plugin source trees |
| `entry-command` | Command prefix to execute an entry point |
| `cwd` | Working directory for subprocess execution (optional) |
| `entry-filename` | Entry point filename (e.g. `conformance_entry.py`) |
| `model-marker` | File pattern to identify model plugins (e.g. `model_info.py`) |

## Default behavior

When `--runtime` is omitted, **all configured runtimes** are used.
Results are displayed in a unified table with a Runtime column.

```bash
# Runs across all configured runtimes (python, js, go)
conform check-model

# Auto-discovers which runtimes have entry points for google-genai
conform check-model google-genai
```

## Restricting to a single runtime

```bash
# Python only
conform --runtime python check-model

# JavaScript only
conform --runtime js check-model google-genai
```

## Runtime configuration

All three runtimes are fully configured:

```toml
[tool.conform.runtimes.python]
specs-dir = "py/tests/conform"
plugins-dir = "py/plugins"
entry-command = ["uv", "run", "--project", "py", "--active"]

[tool.conform.runtimes.js]
specs-dir = "py/tests/conform"
plugins-dir = "js/plugins"
entry-command = ["npx", "tsx"]
cwd = "js"

[tool.conform.runtimes.go]
specs-dir = "py/tests/conform"
plugins-dir = "go/plugins"
entry-command = ["go", "run"]
cwd = "go"
```

## The Runtime Protocol

The runner depends on the `Runtime` Protocol, not any concrete class:

```python
@runtime_checkable
class Runtime(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def specs_dir(self) -> Path: ...

    @property
    def plugins_dir(self) -> Path: ...

    @property
    def entry_command(self) -> list[str]: ...

    @property
    def cwd(self) -> Path | None: ...

    @property
    def entry_filename(self) -> str: ...

    @property
    def model_marker(self) -> str: ...
```

This means you can also create custom runtime implementations
programmatically without using TOML configuration.

## Adding a new runtime

1. Add a `[tool.conform.runtimes.<name>]` section to `pyproject.toml`:

    ```toml
    [tool.conform.runtimes.dart]
    specs-dir = "dart/tests/conform"
    plugins-dir = "dart/plugins"
    entry-command = ["dart", "run"]
    entry-filename = "conformance_entry.dart"
    model-marker = "model_info.dart"
    ```

2. Create the spec directory structure with entry points:

    ```
    py/tests/conform/
    ├── <plugin>/
    │   ├── model-conformance.yaml
    │   └── conformance_entry.dart
    ```

3. Run tests:

    ```bash
    conform --runtime dart check-model
    ```

## Cross-runtime entry points

Each plugin can have entry points for multiple runtimes in the same
spec directory.  The tool discovers which runtimes have entry points
for each plugin:

```
py/tests/conform/google-genai/
├── model-conformance.yaml       ← shared spec
├── conformance_entry.py         ← Python entry point
├── conformance_entry.ts         ← JS entry point
└── conformance_entry.go         ← Go entry point
```

The `conform list` command shows which runtimes are available per plugin.
