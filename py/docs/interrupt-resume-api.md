# Python Tool Interrupt/Resume API

## Proposed API

### Triggering an Interrupt

```python
from genkit import Interrupt

@ai.tool()
async def transfer_money(input: Transfer, ctx: ToolRunContext) -> TransferResult:
    if input.amount > 1000:
        raise Interrupt({'reason': 'confirm_large', 'amount': input.amount})
    return TransferResult(status='success')
```

### Handling Interrupts

```python
response = await ai.generate(prompt='Transfer $5000 to Alice', tools=[transfer_money])

if response.interrupts:
    interrupt = response.interrupts[0]  # Interrupt object, not raw ToolRequestPart

    if user_confirms():
        tr = interrupt.restart()  # re-run the tool
    else:
        tr = interrupt.respond(TransferResult(status='cancelled'))

    response = await ai.generate(
        messages=response.messages,
        resume={'respond': [tr]},
    )
```

### Convenience: Interrupt-Only Tools

```python
ask_user = ai.define_interrupt(
    name='ask_user',
    input_schema=Question,
    output_schema=Answer,
)
```

---

## Deviations from JS

| | JS | Python | Why |
|---|---|---|---|
| **Trigger** | `interrupt(data)` (function on ctx) | `raise Interrupt(data)` | More Pythonic. Explicit exception vs hidden throw. Makes `ctx` optional for tools that only interrupt. |
| **Respond** | `tool.respond(interrupt, output)` | `interrupt.respond(output)` | Don't need tool reference. Works for plugin tools, string-referenced tools. You always have the interrupt. |
| **Restart** | `tool.restart(interrupt, meta)` | `interrupt.restart(metadata=...)` | Same reasoning. |
| **Resume param** | `resume: { respond: [...] }` | `resume={'respond': [...]}` | Same. (Currently Python uses `tool_responses=[...]` - we're aligning.) |

### Why `interrupt.respond()` over `tool.respond()`

JS requires the tool reference for schema validation:
```typescript
// JS - need tool in scope
const tr = myTool.respond(interrupt, output);
```

Python validates via registry lookup inside the `Interrupt` wrapper:
```python
# Python - interrupt has hidden registry reference
tr = interrupt.respond(output)  # looks up tool internally, validates
```

User never sees the registry. `response.interrupts` returns pre-wrapped `Interrupt` objects with validation built in.

---

## What We're Changing

### Before (current Python)

```python
from genkit._ai._tools import tool_response  # buried import

@ai.tool()
async def my_tool(input: Input, ctx: ToolRunContext) -> Output:
    ctx.interrupt(data)  # hidden throw

# Later
tr = tool_response(interrupt, output)  # no validation
await ai.generate(tool_responses=[tr])  # inconsistent param name
```

### After (proposed)

```python
from genkit import Interrupt  # top-level

@ai.tool()
async def my_tool(input: Input, ctx: ToolRunContext) -> Output:
    raise Interrupt(data)  # explicit

# Later
tr = interrupt.respond(output)  # validates against tool schema
await ai.generate(resume={'respond': [tr]})  # consistent with JS
```

---

## Open Questions

1. **Should `Interrupt` be the exception class name, or something else?**
   - `Interrupt`, `ToolInterrupt`, `InterruptExecution`?
   - JS uses `ToolInterruptError` but we're not wrapping it in a function call anymore.
