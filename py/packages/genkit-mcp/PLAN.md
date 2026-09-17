# Plan v2: genkit-ai/genkit#6194 - py MCP stdio client, tool.v2 via DAP

Revised after adversarial review against JS (`js/plugins/mcp`), Go (`go/plugins/mcp`) and py core.
Worktree `.claude/worktrees/py-mcp-stdio-6194`, branch `worktree-py-mcp-stdio-6194`, base origin/main 48845db87.

## Blockers found in v1 (all verified in code)

1. **DAP wildcard binds by `action.metadata['name']`, not `action.name`.** `_core/_dap.py:116-123`
   filters on `m.get('name')`; `_ai/_generate.py:417-423` skips any metadata without it. `tool()` never
   sets it (`_core/_action.py:655-675`). v1 would have expanded `mcp:tool/*` to zero tools, silently.
2. **Event-loop affinity.** Dev UI reflection server runs `asyncio.run()` on its own daemon thread
   (`_ai/_aio.py:925-929`). A single background task holding the MCP session is loop-bound. Core ships
   `loop_local_client` (`_core/_loop_cache.py:14-40`, exported from `genkit.plugin_api`) for exactly this.
3. **`Registry.list_actions` does not expand DAP children** (`_core/_registry.py:302-345`, docstring says
   so outright). `list_action_metadata_by_key` has no production caller. JS does the opposite
   (`js/core/src/registry.ts:365,401`). So the Dev UI has no row to click, though *running* by key works
   (`_core/_registry.py:639-680`).
4. **`<server>/<tool>` is a guaranteed 400 on Gemini and OpenAI.** `genkit-google-genai` sends
   `name=tool.name` verbatim in the declaration (`models/gemini.py:1194`) and only mangles `/`->`__` on
   the part path (`models/utils.py:168,231,357,381`). `genkit-openai` is verbatim too
   (`genkit_openai/models/model.py:277`). Go already picked `_` (`go/plugins/mcp/common.go:30-32`).

## Decisions taken

| # | Decision |
|---|---|
| D1 | Function API `define_mcp_client(ai, ...)`, not a `Plugin` subclass. `Plugin.init()` renames actions to `<plugin>/<name>` and DAP provider names cannot contain `/`. |
| D2 | Separator is `_`: tool name `<server>_<tool>`, matching Go. Selector `everything:tool/everything_echo`. |
| D3 | `server_name` is a separate optional arg defaulting to the server's advertised `serverInfo.name`, as JS does (`client/client.ts:98-99,177-183`). Not the DAP name. |
| D4 | Connection is per-event-loop via `loop_local_client`. One subprocess per loop. |
| D5 | `isError` -> `output={'error': <concatenated text>}`, metadata merged not replaced. No raise. |
| D6 | `resource_link` -> `ResourcePart`, matching JS (`util/tools.ts:81-88`). The issue excludes the MCP *resources* API, not this content block. |
| D7 | Dev UI listing needs a core change. Split into its own commit, and its own follow-up issue if it grows. |

## Package layout

`py/packages/genkit-mcp`, import `genkit_mcp`:

```
pyproject.toml  README.md  LICENSE  CHANGELOG.md
src/genkit_mcp/__init__.py      define_mcp_client, McpClient, McpStdioServerConfig
src/genkit_mcp/_config.py       McpStdioServerConfig(command, args, env, cwd, disabled)
src/genkit_mcp/_connection.py   loop-local session lifecycle
src/genkit_mcp/_tools.py        MCP Tool -> Action; CallToolResult -> MultipartToolResponse
src/genkit_mcp/_client.py       McpClient + define_mcp_client
src/genkit_mcp/py.typed
tests/fake_server.py  tests/mcp_tools_test.py  tests/mcp_client_test.py  tests/mcp_loop_test.py
```

## Result mapping (JS `util/tools.ts:56-121` is the spec)

- `isError` truthy: `output={'error': <all text concatenated>}`, `metadata={**(result.meta or {}), 'mcp': {'isError': True}}`, return early, drop media. Py SDK `isError` is non-optional `bool` so unset and false are indistinguishable.
- `structuredContent` set: that is `output`.
- Else: concatenate every text block with no separator. JSON-parse only if the trimmed string starts with `{` or `[`. Never coerce `42` / `true` / `null`.
- image, audio: `MediaPart` with `data:<mimeType>;base64,<data>`.
- Embedded resource: `blob` -> `MediaPart`; `text` -> folded into output text.
- `resource_link` -> `ResourcePart(resource={'uri': ...})`.
- `result.meta` merged into envelope metadata.
- `call_tool` raises `RuntimeError` when the server declares `outputSchema` and the result fails it
  (mcp 1.26 `client/session.py:394-423`). Catch it and fold into the `isError` envelope.

## Tool registration

DAP fn: `list_tools` paginated with `params=PaginatedRequestParams(cursor=...)` (not the deprecated
positional form). Per tool:

```python
action = genkit.tool(
    handler, name=f'{server_name}_{t.name}', description=t.description or '', input_schema=t.inputSchema
).action()
action.metadata['name'] = action.name  # required by expand_wildcard_tools
action.metadata['mcp'] = {'server': server_name, '_meta': t.meta}
```

Handler is 2-arg `(input, ctx)` so it can forward `ctx.context.get('mcp', {}).get('_meta')` as
`call_tool(..., meta=...)`, matching JS (`util/tools.ts:171-180`). On a closed-session error the handler
calls `dap.invalidate_cache()` before raising, so the next `generate` re-lists.
`disabled=True` makes the DAP return `{'tool': []}`.

## Steps

1. **Scaffold + wiring.** New package files. Root `py/pyproject.toml`: `dependencies`,
   `[tool.uv.sources]`, pyright/ty/pyrefly `src` lists. Plus, missed in v1:
   `py/packages/genkit/pyproject.toml` `[project.optional-dependencies] mcp` and its own
   `[tool.uv.sources]`; `.github/workflows/publish_python.yml` `PACKAGES`; `py/tests/smoke/`.
   Promote `mcp` from the dev group. `uv lock`, `just py lint`, check liccheck on the new transitive tree.
2. **`_tools.py` + pure mapping tests.** Hand-built `CallToolResult` per row above. Plus a fixture tool
   with an awkward inputSchema (no `type`, a `$ref`) to see how far it travels.
3. **`_connection.py` + loop tests.** Spawned fake server: connect, list, call, close, process reaped.
   A test that drives a fetch from a second `asyncio.run()` on another thread. A test for close during an
   in-flight `call_tool`.
4. **`_client.py` + `define_mcp_client`.** Test `expand_wildcard_tools(registry, ['fake:tool/*'])` returns
   non-empty (this is the regression test for blocker 1). Test `resolve_action_by_key` then `run` returns
   the envelope. Generate round-trip with `define_programmable_model`.
5. **Dev UI listing (core).** Wire `list_action_metadata_by_key` into `Registry.list_actions` for
   registered DAPs, mirroring JS. Separate commit. If it turns out to need design, drop it and file a
   follow-up, and say in the PR that the Dev UI criterion is only half met.
6. **Sample** `py/samples/mcp-hello` (FastMCP server + main). Add to `py/pyproject.toml` workspace and to
   `.github/workflows/python-samples.yml` if that matrix is live. Manual Dev UI check with a real Gemini
   model to confirm D2 naming.

## Known gaps to state in the PR

- No auto-reconnect. Cache invalidation on a dead session only.
- No `notifications/tools/list_changed` subscription. Neither JS nor Go has one either. TTL is the story.
- No client-side required-argument validation. Go has it (`go/plugins/mcp/tools.go:177-203`); py core does
  none for a dict input schema (`_core/_action.py:527-536,725-727`).
- Windows untested. py CI is ubuntu-only and `stdio_client` has a win32-specific path.
