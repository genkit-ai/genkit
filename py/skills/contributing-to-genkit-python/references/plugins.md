# Plugin packages

A plugin is the veneer between the user and a provider SDK. The user should see a few names and flat arguments. The provider's nested request shapes stay inside.

## One public module

The package root `__init__.py` is the only file that exports anything, and its `__all__` is the whole API. If the plugin has experimental pieces, they go in `exp/__init__.py` under the same rule. Everything else lives in underscore modules (`_plugin.py`, `_models.py`, `_config.py`). Private subpackages get no `__init__.py`, because a re-exporting `__init__.py` makes an internal package look like an API; import from the defining module instead.

```text
genkit_mcp/
  __init__.py   # McpClient, McpHost, define_mcp_client, define_mcp_host
  _client.py
  _config.py    # McpStdioServerConfig: not exported
  _connection.py
  _host.py
  _tools.py
```

## Every exported name needs a reason

A name goes in `__all__` only if a user constructs it, calls it, or writes it in an annotation, and there's no simpler way to do that. Leave it out if:

- it only names a shape a parameter already accepts. A `TypedDict` for a dict argument or an alias for a callback type adds an import and nothing else.
- it belongs to a request or response the user never makes directly.
- it declares no fields of its own. A config class that accepts anything does nothing a dict doesn't.
- it's a nested config class the user would have to build by hand. See the next section.

If you can't say in one sentence what the user does with a name, it isn't public.

## Flat at the surface, nested in the action

The plugin constructor, `define_*` functions and model refs take flat keyword arguments. Infer every field you can from what the user already passed, then build the provider's nested config inside the plugin or model action. Don't export a nested config class and ask the user to assemble it.

```python
# 1. Bedrock infers the route from the model ID
Bedrock(models=['stability.sd3-5-large-v1:0'])
# instead of
Bedrock(models=[ModelDefinition(name='stability.sd3-5-large-v1:0', type='image')])

# 2. MCP takes the server's launch fields as keywords and builds its config internally
client = define_mcp_client(ai, 'git', command='uvx', args=['mcp-server-git'])
# => McpClient
```

- **Derive it**: if a value follows from something already passed (model ID to route, model family to config class), compute it. Don't ask for it.
- **Drop it**: if only one value is valid, remove the argument. A keyword can come back later without breaking anyone.
- **Flatten it**: a group of related settings becomes keywords on the outer call, not a new class the user imports.
- **Model config stays one class, kept flat**: each model family gets one `GenkitConfig` (`GeminiConfig`, `OllamaConfig`), since the Dev UI renders it as a form and a dict validates into it. Keep its fields flat and translate to the provider's nested request inside the model action.
