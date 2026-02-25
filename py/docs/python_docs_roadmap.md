# Python Documentation Roadmap for genkit.dev

> **Source:** [genkit-ai/docsite](https://github.com/genkit-ai/docsite)
> **Docs path:** `src/content/docs/docs/`
> **Generated:** 2026-02-07
> **Updated:** 2026-02-07
>
> **Scope Exclusions:**
> - **Chat/Session API** — Deprecated, skip
> - **Agents / Multi-Agent** — Not yet in Python SDK, skip
> - **MCP** — Will come later, skip for now
>
> This roadmap tracks every genkit.dev documentation page and whether Python
> examples/tabs need to be added, updated, or are already complete. It also
> identifies features demonstrated in JS examples that should be covered by
> Python samples in the `firebase/genkit` repo.

---

## Summary

| Status | Count | Description |
|--------|-------|-------------|
| ✅ Complete | 5 | Python tab exists with full parity |
| 🔶 Partial | 6 | Python tab exists but is incomplete or stale |
| ❌ Missing | 8 | No Python tab at all (JS/Go only or JS-only) |
| ➖ N/A | 5 | Language-agnostic or meta pages |

---

## 1. Core Documentation Pages

### ✅ Python Tab Complete (verify accuracy)

| Page | File | Languages | Notes |
|------|------|-----------|-------|
| **Models** | `models.mdx` | js, go, python | Python examples for: `generate()`, system prompts, model parameters, structured output, streaming, multimodal input. **Missing:** Generating Media, Middleware (retry/fallback). |
| **Tool Calling** | `tool-calling.mdx` | js, go, python | Python examples for: defining tools, using tools, interrupts (link), explicitly handling tool calls. **Missing:** `maxTurns`, dynamically defining tools at runtime. |
| **Flows** | `flows.mdx` | js, go, python | Python examples for: defining flows, input/output schemas, calling flows, streaming flows, deploying (Flask). **Missing:** Flow steps (`ai.run()`), durable streaming. |
| **Get Started** | `get-started.mdx` | js, go, python | Complete walkthrough for Python. |
| **Interrupts** | `interrupts.mdx` | js, go, python | Python examples for interrupt definition and resumption. |

### 🔶 Python Tab Exists but Incomplete

| Page | File | Languages | What's Missing |
|------|------|-----------|----------------|
| **Models** | `models.mdx` | js, go, python | **Generating Media** section (Python SDK supports TTS, image gen via google-genai). **Middleware** section (retry/fallback — Python has `use=` support but no docs). |
| **Tool Calling** | `tool-calling.mdx` | js, go, python | **`maxTurns`** — Python supports `max_turns=`. **Dynamically defining tools at runtime** — needs investigation. **Streaming + Tool calling** — needs docs. |
| **Flows** | `flows.mdx` | js, go, python | **Flow steps** (`genkit.Run()` equivalent in Python). **Durable streaming** — needs investigation. |
| **RAG** | `rag.mdx` | js, go, python | Python tab may be stale. Verify: indexers, embedders, retrievers, simple retrievers, custom retrievers, rerankers sections. |
| **Context** | `context.mdx` | js only (rendered) | The rendered page shows JS-only. Python supports `context=` on `generate()` and flows. Needs Python examples for context in actions, context at runtime, context propagation. |
| **Dotprompt** | `dotprompt.mdx` | js, go | Python SDK has dotprompt support (`genkit.core.prompt`). Needs full Python tab with: creating prompt files, running prompts, model configuration, schemas, tool calling in prompts, multi-message prompts, partials, prompt variants, defining prompts in code. |

### ❌ Python Tab Missing (needs to be added)

| Page | File | Current Languages | Priority | What to Add |
|------|------|-------------------|----------|-------------|
| **Chat (Sessions)** | `chat.mdx` | js, go | **P1** | Python SDK has `ai.chat()` and `Session` with `session.chat()`. Need: session basics, stateful sessions with tools, multi-thread sessions. Session persistence is experimental — may tag as such. |
| **Agentic Patterns** | `agentic-patterns.mdx` | js, go | **P1** | Python supports all required primitives (flows, tools, interrupts). Need: sequential workflow, conditional routing, parallel execution, tool calling, iterative refinement, autonomous agent, stateful interactions. |
| **Multi-Agent** | `multi-agent.mdx` | js (likely) | **P2** | Need to verify Python support for agent-to-agent delegation. |
| **Durable Streaming** | `durable-streaming.mdx` | js (likely) | **P3** | Need to verify Python support. |
| **Client SDK** | `client.mdx` | js | **P3** | Client-side integration. May not apply to Python backend SDK directly. |
| **MCP Server** | `mcp-server.mdx` | js (likely) | **P2** | Python has MCP support via `genkit.plugins.mcp`. Needs Python examples. |
| **Model Context Protocol** | `model-context-protocol.mdx` | js (likely) | **P2** | Python MCP client integration. |
| **Evaluation** | `evaluation.mdx` | js (likely) | **P2** | Python has evaluator support (`evaluator-demo` sample exists). Needs Python examples. |

### ➖ Language-Agnostic / Meta Pages

| Page | File | Notes |
|------|------|-------|
| **Overview** | `overview.mdx` | Conceptual overview, no code tabs needed |
| **API References** | `api-references.mdx` | Links to API docs |
| **API Stability** | `api-stability.mdx` | Policy document |
| **Error Types** | `error-types.mdx` | Reference |
| **Feedback** | `feedback.mdx` | Meta |
| **Develop with AI** | `develop-with-ai.mdx` | Meta/guide |
| **DevTools** | `devtools.mdx` | Dev UI documentation |
| **Local Observability** | `local-observability.mdx` | Observability setup |

---

## 2. Subdirectory Pages (need individual audit)

| Directory | Path | Known Pages | Python Status |
|-----------|------|-------------|---------------|
| **Deployment** | `deployment/` | Cloud Run, Firebase, etc. | 🔶 Python Flask deployment exists; verify others |
| **Frameworks** | `frameworks/` | Express, etc. | ❌ Need Flask/FastAPI/Starlette Python examples |
| **Integrations** | `integrations/` | Various provider plugins | 🔶 Some Python plugins documented; need audit |
| **Observability** | `observability/` | GCP, custom, etc. | 🔶 Python GCP telemetry plugin exists |
| **Plugin Authoring** | `plugin-authoring/` | Writing plugins | ❌ Need Python plugin authoring guide |
| **Resources** | `resources/` | Additional resources | ➖ Likely language-agnostic |
| **Tutorials** | `tutorials/` | Step-by-step guides | ❌ Need Python tutorials |

---

## 3. Feature Parity: JS Examples → Python Samples

This section maps JS features documented on genkit.dev to their Python sample
coverage status.

### `/docs/models` Features

| Feature | JS Example | Python Sample Status | Python SDK Support |
|---------|-----------|---------------------|-------------------|
| Simple generation | `ai.generate('prompt')` | ✅ All hello samples have `say_hi` | ✅ |
| System prompts | `system: "..."` | ✅ Being added to all samples | ✅ |
| Multi-turn (messages) | `messages: [...]` | ✅ Being added to all samples | ✅ |
| Model parameters | `config: {...}` | ✅ `say_hi_with_config` in most samples | ✅ |
| Structured output | `output: { schema: ... }` | ✅ `generate_character` in most samples | ✅ |
| Streaming | `ai.generateStream()` | ✅ `streaming_flow` / `say_hi_stream` | ✅ |
| Streaming + structured | `generateStream() + output schema` | ❌ No dedicated sample | ✅ (need sample) |
| Multimodal input | `prompt: [{media: ...}, {text: ...}]` | ✅ `describe_image` in google-genai, anthropic, xai, msf | ✅ |
| Generating media (images) | `output: { format: 'media' }` | ❌ No dedicated sample | ✅ (google-genai supports it) |
| Generating media (TTS) | Text-to-speech | ❌ No dedicated sample | ✅ (google-genai supports it) |
| Middleware (retry) | `use: [retry({...})]` | ❌ No sample | 🔶 Python has `use=` plumbing, but no retry/fallback middleware defined |
| Middleware (fallback) | `use: [fallback({...})]` | ❌ No sample | 🔶 Same as above |

### `/docs/tool-calling` Features

| Feature | JS Example | Python Sample Status | Python SDK Support |
|---------|-----------|---------------------|-------------------|
| Define tools | `ai.defineTool()` | ✅ All samples with tools | ✅ |
| Use tools | `tools: [getWeather]` | ✅ `weather_flow` in most samples | ✅ |
| `maxTurns` | `maxTurns: 8` | ✅ 3 samples use `max_turns=2` | ✅ |
| Dynamic tools at runtime | `tool({...})` | ❌ No sample | ❓ Need investigation |
| Interrupts | `ctx.interrupt()` | ✅ `tool-interrupts`, `google-genai-hello`, `short-n-long` | ✅ |
| `returnToolRequests` | `returnToolRequests: true` | ✅ 1 sample (`google-genai-context-caching`) | ✅ |
| Streaming + tool calling | Stream with tools | ❌ No dedicated sample | ✅ (need sample) |

### `/docs/interrupts` Features

| Feature | JS Example | Python Sample Status | Python SDK Support |
|---------|-----------|---------------------|-------------------|
| `defineInterrupt()` | Dedicated interrupt definition | ❌ No equivalent sample | ❓ `define_interrupt` not found in Python SDK |
| `ctx.interrupt()` in tool | Tool-based interrupts | ✅ `tool-interrupts`, `google-genai-hello` | ✅ |
| Restartable interrupts | `restart` option | ❌ No sample | ❓ Need investigation |
| `response.interrupts` check | Interrupt loop | ✅ Demonstrated in `tool-interrupts` | ✅ |
| `resume: { respond: [...] }` | Resume generation | ❌ No sample using `resume` | ❓ Need investigation |

### `/docs/context` Features

| Feature | JS Example | Python Sample Status | Python SDK Support |
|---------|-----------|---------------------|-------------------|
| Context in flow | `{context}` destructured | ❌ No sample | ✅ (context available via ActionRunContext) |
| Context in tool | `{context}` in tool handler | ❌ No sample | ✅ (ToolContext has context) |
| Context in prompt file | `{{@auth.name}}` | ❌ No sample | ✅ (dotprompt supports @) |
| Provide context at runtime | `context: { auth: ... }` | ❌ No sample | ✅ (`context=` supported on `generate()`) |
| Context propagation | Auto-propagation to tools | ❌ No sample | ✅ |

### `/docs/chat` Features

| Feature | JS Example | Python Sample Status | Python SDK Support |
|---------|-----------|---------------------|-------------------|
| `ai.chat()` basic | Create chat, send messages | ❌ No sample | ✅ `ai.chat()` exists |
| Chat with system prompt | `ai.chat({ system: '...' })` | ❌ No sample | ✅ |
| Stateful sessions | Session with state management | ❌ No sample | ✅ `Session` class exists |
| Multi-thread sessions | Named chat threads | ❌ No sample | ✅ `session.chat('thread')` |
| Session persistence | Custom store implementation | ❌ No sample | 🔶 Experimental |

### `/docs/flows` Features

| Feature | JS Example | Python Sample Status | Python SDK Support |
|---------|-----------|---------------------|-------------------|
| Define flows | `@ai.flow()` decorator | ✅ All samples | ✅ |
| Input/output schemas | Pydantic models | ✅ All samples | ✅ |
| Streaming flows | `ctx.send_chunk()` | ✅ Several samples | ✅ |
| Flow steps (`ai.run()`) | Named trace spans | ❌ No sample | ❓ Need investigation |
| Deploy with Flask | Flask integration | ✅ Documented on genkit.dev | ✅ |

### `/docs/dotprompt` Features

| Feature | JS Example | Python Sample Status | Python SDK Support |
|---------|-----------|---------------------|-------------------|
| Prompt files (.prompt) | `.prompt` file format | ❌ No sample | ✅ Dotprompt supported |
| Running prompts from code | `ai.prompt('name')` | ❌ No sample | ✅ |
| Input/output schemas | Picoschema/JSON Schema | ❌ No sample | ✅ |
| Tool calling in prompts | `tools: [...]` in frontmatter | ❌ No sample | ✅ |
| Multi-message prompts | `{{role "system"}}` | ❌ No sample | ✅ |
| Partials | `{{> partialName}}` | ❌ No sample | ✅ |
| Prompt variants | `.variant.prompt` files | ❌ No sample | ✅ |
| Defining prompts in code | `ai.define_prompt()` | ❌ No sample | ✅ |

### `/docs/agentic-patterns` Features

| Feature | JS Example | Python SDK Support | Sample Needed |
|---------|-----------|-------------|---------------|
| Sequential workflow | Chain of flows | ✅ | ❌ No sample |
| Conditional routing | If/else in flow | ✅ | ❌ No sample |
| Parallel execution | Multiple concurrent calls | ✅ (`asyncio.gather`) | ❌ No sample |
| Tool calling | Tools in generate | ✅ | ✅ Exists |
| Iterative refinement | Loop with evaluation | ✅ | ❌ No sample |
| Autonomous agent | Agent with tools loop | ✅ | ❌ No sample |
| Stateful interactions | Session-based | ✅ | ❌ No sample |

---

## 4. Priority Action Items

### P0: Critical (blocking feature parity)
1. **`models.mdx`** — Add Generating Media section for Python (images, TTS)
2. **`chat.mdx`** — Add Python tab with `ai.chat()` and `Session` examples
3. **`context.mdx`** — Add Python tab with context in flows, tools, and generate
4. **`dotprompt.mdx`** — Add Python tab with full dotprompt examples

### P1: High Priority (important for developer experience)
5. **`agentic-patterns.mdx`** — Add Python tab for all agentic patterns
6. **`tool-calling.mdx`** — Add `max_turns` docs, streaming + tools
7. **`models.mdx`** — Add Middleware section for Python (investigate retry/fallback)
8. **`evaluation.mdx`** — Add Python tab for evaluation
9. **`mcp-server.mdx`** / **`model-context-protocol.mdx`** — Add Python MCP examples
10. **Python samples** — Add `streaming_structured_output` flow to hello samples

### P2: Medium Priority (polish)
11. **`flows.mdx`** — Add flow steps docs for Python
12. **`multi-agent.mdx`** — Add Python tab if SDK supports agent delegation
13. **`frameworks/`** — Add Flask/FastAPI/Starlette deployment guides
14. **`plugin-authoring/`** — Add Python plugin authoring guide
15. **`interrupts.mdx`** — Verify Python section covers `defineInterrupt` equivalent and restartable interrupts

### P3: Low Priority (nice to have)
16. **`durable-streaming.mdx`** — Investigate Python support
17. **`client.mdx`** — Determine if applicable to Python
18. **`tutorials/`** — Create Python-specific tutorials
19. **`deployment/`** — Add Python Cloud Run, etc. deployment guides

---

## 5. Python Samples Gap Analysis

### Samples needing `system_prompt` flow (in progress)
- [x] `google-genai-hello`
- [x] `compat-oai-hello`
- [x] `anthropic-hello`
- [x] `ollama-hello`
- [x] `amazon-bedrock-hello`
- [x] `deepseek-hello`
- [x] `xai-hello`
- [x] `cloudflare-workers-ai-hello`
- [ ] `microsoft-foundry-hello`
- [ ] `mistral-hello`
- [ ] `huggingface-hello`
- [ ] `google-genai-vertexai-hello`
- [ ] `short-n-long`
- [ ] `model-garden`

### Samples needing `multi_turn_chat` flow (in progress)
- [x] `google-genai-hello`
- [x] `compat-oai-hello`
- [x] `anthropic-hello`
- [x] `ollama-hello`
- [x] `amazon-bedrock-hello`
- [x] `xai-hello`
- [x] `cloudflare-workers-ai-hello`
- [ ] `microsoft-foundry-hello`
- [ ] `google-genai-vertexai-hello`
- [ ] `short-n-long`
- [ ] `model-garden`

### New standalone samples needed
- [x] ~~`dotprompt-hello`~~ — Covered by `prompt-demo` sample ⚠️ (P1 bug: recursion depth exceeded)
- [ ] ~~`chat-hello`~~ — Chat/Session API deprecated, skip
- [ ] ~~`agentic-patterns`~~ — Agents not yet in Python SDK, skip
- [ ] `context-demo` — Need dedicated context flows (context in generate, flows, tools, propagation, `ai.current_context()`)
- [x] ~~`streaming-structured-output`~~ — Covered by `google-genai-hello` / hello samples
- [x] ~~`media-generation`~~ — Covered by `media-models-demo` sample
- [ ] `middleware-demo` — Custom retry/fallback middleware using `use=` parameter
- [ ] `streaming-tools` — Streaming + tool calling flow
- [ ] `eval-pipeline` — End-to-end eval: dataset → inference → metrics → results

---

### Dotprompt sample gaps (in `prompt-demo`)
- [ ] Tool calling in prompts (`tools: [...]` in frontmatter)
- [ ] Multimodal prompts (`{{media url=photoUrl}}`)
- [ ] Defining prompts in code (`ai.define_prompt()`)
- [ ] Default input values (`default:` in frontmatter)

---

## 6. Known Bugs

| Sample | Bug | Severity |
|--------|-----|----------|
| `prompt-demo` | `Failed to load lazy action recipe.robot: maximum recursion depth exceeded` / same for `story` | **P0** — Blocks all prompt feature demos |
