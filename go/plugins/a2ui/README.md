# A2UI plugin for Genkit Go

Adds [A2UI](https://a2ui.org/) ("Agent to UI") support to Genkit Go agents. A2UI
is a transport-agnostic, JSON-based streaming UI protocol. An A2UI-enabled agent
can stream not just prose, but rich, interactive UI **surfaces** that a client
renders incrementally.

> Status: experimental.

## Design principle: one representation

A2UI rides on its own part channel: a Genkit **data part** carrying the mime type
`application/a2ui+json` whose `data` is an object `{ "envelopes": [...] }`
wrapping an array of A2UI envelope messages. This maps 1:1 onto the A2A binding
of the A2UI spec, so the same envelopes are byte-compatible with the JS plugin
and the `@a2ui/*` renderers.

## Usage

The whole server-side integration is the A2UI middleware. Add it to a
`Generate` call via `ai.WithUse`:

```go
package main

import (
	"context"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/a2ui"
	"github.com/firebase/genkit/go/plugins/googlegenai"
)

func main() {
	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))

	resp, _ := genkit.Generate(ctx, g,
		ai.WithModel(googlegenai.GoogleAIModel(g, "gemini-flash-latest")),
		ai.WithSystem("You help users. Render UI when it is clearer than prose."),
		ai.WithPrompt("show me the weather in Tokyo"),
		ai.WithUse(&a2ui.Config{}), // defaults to the bundled 'basic' catalog
	)

	// A2UI envelopes ride as data parts on the response message.
	envelopes := a2ui.EnvelopesFromParts(resp.Message.Content)
	_ = envelopes
}
```

The middleware injects the catalog's capabilities into the system prompt, then
intercepts model output (streamed chunks **and** the final message), extracts
`a2ui` fenced blocks, validates them against the catalog, and rewrites them into
a2ui data parts.

### Options

`a2ui.Config` fields:

| Field          | Default            | Description                                                                                       |
| -------------- | ------------------ | ------------------------------------------------------------------------------------------------- |
| `Catalog`      | nil                | An inline catalog (code-defined use only; not serialized). Overrides `CatalogID` when set.        |
| `CatalogID`    | `"basic"`          | Id of a catalog registered with `LoadCatalog`. Resolved from the registry at call time.           |
| `Instructions` | `"system"`         | Where to inject catalog capabilities. `"none"` injects nothing.                                    |
| `Validate`     | `"warn"`           | Validate emitted envelopes. `"warn"` logs and drops bad blocks; `"strict"` returns an error; `"off"` skips checking. |
| `SurfaceID`    | fresh UUID         | Surface id policy. Provide a fixed string to reuse one id for every surface.                       |
| `Version`      | `"v0.9"`           | Protocol version stamped on envelopes.                                                            |

### Custom catalogs

The bundled `BasicCatalog()` mirrors `@a2ui/web_core`'s basic catalog. To use
your own components, register a `Catalog` with `LoadCatalog` (or `LoadCatalogFile`)
and reference it by id. Registered catalogs live in the Genkit registry under
the value type `a2ui-catalog`, so they are discoverable by tooling (the Dev UI's
`GET /api/values?type=a2ui-catalog`) and shared identically across the JS, Go,
and Dart plugins.

```go
catalog, err := a2ui.LoadCatalogFile(g, "./my-catalog.json")
if err != nil { /* ... */ }

resp, _ := genkit.Generate(ctx, g,
	ai.WithModel(m),
	ai.WithPrompt("..."),
	ai.WithUse(&a2ui.Config{CatalogID: catalog.ID}),
)
```

Or construct and register one in memory:

```go
myCatalog := &a2ui.Catalog{
	ID: "https://my-app.org/catalogs/custom.json",
	Components: []a2ui.CatalogComponent{
		{Name: "Banner", Description: "A prominent alert banner.", Props: "title: string (required)."},
	},
}
a2ui.LoadCatalog(g, myCatalog)
// ... ai.WithUse(&a2ui.Config{CatalogID: myCatalog.ID})
```

Register the bundled basic catalog too (so it shows up in the Dev UI) with
`a2ui.RegisterBasicCatalog(g)`.

The middleware resolves the catalog for each turn in this order: an inline
`Config.Catalog` (code-defined only), then `Config.CatalogID` looked up from the
registry, then the bundled basic catalog. Prefer `CatalogID`: unlike an inline
`Catalog`, it survives JSON/Dev-UI dispatch and appears in tooling.

The catalog JSON file follows the `Catalog` shape:

```json
{
  "id": "https://my-app.org/catalogs/custom.json",
  "components": [
    { "name": "Banner", "description": "A prominent alert banner.", "props": "title: string (required); severity?: info|warning|error." },
    { "name": "Text", "description": "A plain or inline-markdown text run.", "props": "text: string (required); variant?: body|caption." }
  ]
}
```

### Sending user actions back to the agent

When the user interacts with a surface (e.g. presses a `Button`), the client
renderer emits an action. Send it back as the next turn: put the action's name
as the user message text, and attach the full action as an a2ui data part. The
middleware sanitizes inbound a2ui parts into a short text summary so the model
can reason about them, and `a2ui.EnvelopesFromParts` reads envelopes back off
any message/chunk content.

## Registering as a plugin (optional)

Passing `&a2ui.Config{}` to `ai.WithUse` is all you need. If you also want the
middleware to appear in the Dev UI and be referenceable by name, register the
plugin during init:

```go
g := genkit.Init(ctx, genkit.WithPlugins(&a2ui.Plugin{}, &googlegenai.GoogleAI{}))
```

## Try it with the web UI

`go/samples/basic-middleware/a2ui` serves an A2UI agent at `POST /api/uiAgent`,
the exact endpoint the browser frontend in `js/testapps/a2ui/web` talks to via
`remoteAgent`. Because the Go agent speaks the same wire protocol as the JS
backend, the existing web UI works unchanged against it:

```sh
# 1. Start the Go backend (needs a Gemini API key in the environment):
cd go/samples/basic-middleware/a2ui
go run .

# 2. In another terminal, build and preview the web UI:
cd js/testapps/a2ui/web
pnpm install && pnpm build && pnpm preview
```

Open the printed preview URL; Vite proxies `/api` to the Go backend on `:8080`.
Ask for "the weather in Tokyo" to see a streamed, interactive surface rendered
by `@a2ui/lit`, including a Refresh button whose action round-trips back to the
Go agent.

## Note on the upstream A2UI SDKs

The A2UI team is standardizing prompt formatting, catalog management, and
inference inside official SDKs (a2ui-core / a2ui-agent). Those are the eventual
home for the prompt-rendering, parsing, and validation this plugin does today,
so treat those internals as thin and replaceable. The stable surface is the
`a2ui.Config` entrypoint and the spec-defined wire part
(`application/a2ui+json`), both of which are unaffected by an SDK swap.

## License

Apache-2.0
