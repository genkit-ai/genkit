# A2UI plugin for Genkit Go

Adds [A2UI](https://a2ui.org/) ("Agent to UI") support to Genkit Go agents. A2UI
is a transport-agnostic, JSON-based streaming UI protocol. An A2UI-enabled agent
can stream not just prose, but rich, interactive UI **surfaces** that a client
renders incrementally.

> Status: in preview. The package lives under `go/plugins/a2ui/exp` and its
> APIs may change in any minor version release. Samples import it as `a2uix`.

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
	a2uix "github.com/firebase/genkit/go/plugins/a2ui/exp"
	"github.com/firebase/genkit/go/plugins/googlegenai"
)

func main() {
	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))

	resp, _ := genkit.Generate(ctx, g,
		ai.WithModel(googlegenai.GoogleAIModel(g, "gemini-flash-latest")),
		ai.WithSystem("You help users. Render UI when it is clearer than prose."),
		ai.WithPrompt("show me the weather in Tokyo"),
		ai.WithUse(&a2uix.Surfaces{}), // defaults to the bundled 'basic' catalog
	)

	// A2UI envelopes ride as data parts on the response message.
	envelopes := a2uix.EnvelopesFromParts(resp.Message.Content)
	_ = envelopes
}
```

The middleware injects the catalog's capabilities into the system prompt, then
intercepts model output (streamed chunks **and** the final message), extracts
`a2ui` fenced blocks, validates them against the catalog, and rewrites them into
a2ui data parts.

### Options

`a2uix.Surfaces` fields:

| Field          | Default            | Description                                                                                       |
| -------------- | ------------------ | ------------------------------------------------------------------------------------------------- |
| `Catalog`      | nil                | An inline catalog (code-defined use only; not serialized). Overrides `CatalogID` when set.        |
| `CatalogID`    | `"basic"`          | Id of a catalog registered with `LoadCatalog`. Resolved from the registry at call time.           |
| `Instructions` | `"system"`         | Where to inject catalog capabilities. `"none"` injects nothing.                                    |
| `Validate`     | `"warn"`           | Validate emitted envelopes. `"warn"` logs and drops bad blocks; `"strict"` returns an error; `"off"` skips checking. This is a well-formedness check, not sanitization (see [Security and the trust boundary](#security-and-the-trust-boundary)). Invalid values are rejected by `New`. |
| `SurfaceID`    | fresh UUID         | Surface id policy. Provide a fixed string to reuse one id for every surface.                       |
| `Version`      | `"v0.9"`           | Protocol version stamped on envelopes. Must be one of the `SupportedVersions`; a typo is rejected by `New`. |


### Custom catalogs

The bundled `BasicCatalog()` mirrors `@a2ui/web_core`'s basic catalog. To use
your own components, register a `Catalog` with `LoadCatalog` (or `LoadCatalogFile`)
and reference it by id. Registered catalogs live in the Genkit registry under
the value type `a2ui-catalog`, so they are discoverable by tooling (the Dev UI's
`GET /api/values?type=a2ui-catalog`). The wire representation of a catalog and
its envelopes is identical across the JS, Go, and Dart plugins, so a surface
rendered by one is byte-compatible with the renderers of another. The registry
*key* differs by runtime, though: Go keys strictly by the catalog's own `ID` and
its config field is `CatalogID`, whereas JS's `loadCatalog` keys by a
caller-chosen lookup id and its middleware config field is `catalog`. Tooling or
shared config that matches catalogs by registry key across runtimes will not
line up; match on the catalog's `id` value instead.


```go
catalog, err := a2uix.LoadCatalogFile(g, "./my-catalog.json")
if err != nil { /* ... */ }

resp, _ := genkit.Generate(ctx, g,
	ai.WithModel(m),
	ai.WithPrompt("..."),
	ai.WithUse(&a2uix.Surfaces{CatalogID: catalog.ID}),
)
```

Or construct and register one in memory:

```go
myCatalog := &a2uix.Catalog{
	ID: "https://my-app.org/catalogs/custom.json",
	Components: []a2uix.CatalogComponent{
		{Name: "Banner", Description: "A prominent alert banner.", Props: "title: string (required)."},
	},
}
a2uix.LoadCatalog(g, myCatalog)
// ... ai.WithUse(&a2uix.Surfaces{CatalogID: myCatalog.ID})
```

Register the bundled basic catalog too (so it shows up in the Dev UI) with
`a2uix.RegisterBasicCatalog(g)`.

The middleware resolves the catalog for each turn in this order: an inline
`Surfaces.Catalog` (code-defined only), then `Surfaces.CatalogID` looked up from the
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
can reason about them, and `a2uix.EnvelopesFromParts` reads envelopes back off
any message/chunk content.

## Registering as a plugin (optional)

Passing `&a2uix.Surfaces{}` to `ai.WithUse` is all you need. If you also want the
middleware to appear in the Dev UI and be referenceable by name, register the
plugin during init:

```go
g := genkit.Init(ctx, genkit.WithPlugins(&a2uix.A2UI{}, &googlegenai.GoogleAI{}))
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

## Security and the trust boundary

`Validate` checks structure and component *type names* against the catalog. It
is a well-formedness check, **not** a sanitizer, even in `"strict"` mode:

- Model-controlled values pass through untouched. An `Image`'s `url`, a `Text`'s
  content (which "may use inline Markdown", so a renderer may turn it into
  HTML), and any other prop value are never inspected or escaped.
- Validation confirms an envelope is well-formed and its components exist in the
  catalog. It does not confirm the values are safe to render.

Treat rendered surfaces as untrusted output driven by the model:

- Prop sanitization is the **renderer/catalog's** responsibility. A catalog
  component should escape or constrain the props it accepts.
- Hosts should CSP-restrict image and other remote sources so a
  model-controlled `Image.url` cannot exfiltrate or load hostile content.

This matches the JS plugin's trust boundary exactly.

## Note on the upstream A2UI SDKs

The A2UI team is standardizing prompt formatting, catalog management, and
inference inside official SDKs (a2ui-core / a2ui-agent). Those are the eventual
home for the prompt-rendering, parsing, and validation this plugin does today,
so treat those internals as thin and replaceable. The stable surface is the
`a2uix.Surfaces` entrypoint and the spec-defined wire part
(`application/a2ui+json`), both of which are unaffected by an SDK swap.


## License

Apache-2.0
