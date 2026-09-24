# genkit-a2ui

Experimental [A2UI](https://a2ui.org/) middleware for Genkit Python.

Add `Surfaces()` to `use=[...]` on `ai.generate` or `define_agent`. The model may
emit ` ```a2ui ` fences; the middleware rewrites them into
`application/a2ui+json` data parts. On the next turn, those parts become text
again so the model can see prior surfaces and button clicks.

```python
from genkit import Genkit
from genkit_a2ui import Surfaces, envelopes_from_parts
from genkit_google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI()])

response = await ai.generate(
    model='googleai/gemini-2.5-flash',
    prompt='Show me the weather in Tokyo',
    use=[Surfaces()],
)
envelopes = envelopes_from_parts(response.message.content)
```

`Surfaces()` uses the bundled basic catalog. To use your own components, register a
catalog and pass its id:

```python
from genkit_a2ui import Surfaces, A2uiCatalog, A2uiCatalogComponent, load_catalog

catalog = A2uiCatalog(
    id='https://my-app.org/catalogs/custom.json',
    components=(
        A2uiCatalogComponent(name='Banner', description='A prominent alert.', props='title: string.'),
    ),
)
load_catalog(ai, catalog)

response = await ai.generate(
    model='googleai/gemini-2.5-flash',
    prompt='Show a warning banner',
    use=[Surfaces(catalog=catalog.id)],
)
```

`load_catalog_file(ai, './my-catalog.json')` does the same from disk.
`register_basic_catalog(ai)` puts the bundled catalog on the registry so the
Developer UI can list it next to custom ones (`GET /api/values?type=a2ui-catalog`).

> Status: experimental.
