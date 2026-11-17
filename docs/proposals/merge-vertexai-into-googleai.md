# Proposal: Merge `VertexAI` into `GoogleAI`

**Author:** Jeff Huang · **Status:** Proposed · **Applies to:** JS, Go, Python

## Summary

Genkit has two plugins for Gemini: `GoogleAI` (an AI Studio API key) and `VertexAI` (a Google Cloud project). They call the same models. The only difference is which Google API serves them.

This proposal folds them into one plugin with a switch:

```python
GoogleAI()                                                         # AI Studio key, as today
GoogleAI(enterprise=True, project='bistro-prod', location='us-central1')  # Google Cloud
```

Examples use Python, and file references point to the Python plugin. JS (`googleAI()`) and Go (`GoogleAI`) get the same option, spelled the way each language spells it.

It ships in two steps:

1. **In any 1.x release, nothing breaks:** add `enterprise=`, `project` and `location` to `GoogleAI`. `VertexAI` keeps working and logs a deprecation warning. Shipping this in Python 1.0 is nice but not required.
2. **At 2.0:** remove `VertexAI`. Old `vertexai/...` model names keep working through an alias.

Neither step blocks Python 1.0.

## Why

**Going to production means rewriting model names.** A team prototypes with `googleai/gemini-2.5-flash` on an AI Studio key. Moving to their Cloud project means changing every ref to `vertexai/gemini-2.5-flash`: in code, in `.prompt` files and in eval datasets. With one plugin, they change one line of setup and the refs stay the same (except for a few model IDs that differ by backend; see below).

**Google's own SDKs already work this way.** They use one client with a backend switch, not two clients:

- **Python `google-genai`:** `Client(enterprise=True)`. The old `vertexai=` flag is now documented as legacy.
- **Go `genai`:** a `Backend` setting, with a new `BackendEnterprise` value.
- Both SDKs read the `GOOGLE_GENAI_USE_ENTERPRISE` env var.

**The name keeps changing.** Vertex AI became Gemini Enterprise Agent Platform in April, then "Gemini platform" in October. A setting on `GoogleAI` holds up better than a plugin class named after the product.

## Design

```python
# 1. Prototype on an AI Studio key
ai = Genkit(plugins=[GoogleAI()])

# 2. Production on a Cloud project: same refs, same .prompt files
ai = Genkit(plugins=[GoogleAI(enterprise=True, project='bistro-prod', location='us-central1')])

response = await ai.generate(model='googleai/gemini-2.5-flash', prompt='Suggest a dish.')
# => runs on whichever backend the plugin was configured for

# 3. Existing VertexAI code keeps working until 2.0
ai = Genkit(plugins=[VertexAI(project='bistro-prod')])
# => DeprecationWarning: VertexAI is deprecated; use GoogleAI(enterprise=True, ...)
```

- **`GoogleAI`, not a new class.** Most users already use `GoogleAI()` and `googleai/`, and none of that changes. Only `VertexAI` users migrate, and they have until 2.0.
- **The ref names the model, and the plugin picks the backend.** The same string works in code, `.prompt` frontmatter, the Dev UI and eval datasets.
- **Traces record the backend.** The ref no longer shows which account paid, so the backend goes in span metadata.
- **Features tied to one backend fail loudly on the other.** The plugin already does this today (`_check_backend_fields` and per-backend routing). `GoogleAI` keys those checks on `enterprise`, and the messages say how to fix it.
  - **Cloud only:** Virtual Try-On, tuned Gemini endpoints, multimodal embedders, image `output_mime_type`/`output_compression_quality`, `person_generation`, `prominent_people`, `google_search_exclude_domains`/`blocking_confidence`, `safety_settings[].method`, per-request `location`, embedder `mime_type`/`auto_truncate`.
  - **Developer API only:** Lyria, Deep Research, Antigravity, `file_search_*`.
- **Internals:** we pass `enterprise=` to `genai.Client`, and internal names move from `vertex` to `enterprise`. The SDK still reports the backend as `client.vertexai`, so we read it in one helper.

## Constructor settings

`enterprise=True` uses exactly what `VertexAI` does today, and `enterprise=False` uses exactly what `GoogleAI` does today. The merged signature is `GoogleAI`'s current parameters plus `enterprise`, `project` and `location`.

```python
# before
VertexAI(project='bistro-prod', location='us-central1')
VertexAI(api_key=key)                          # Cloud express mode
GoogleAI(api_key=key)

# after
GoogleAI(enterprise=True, project='bistro-prod', location='us-central1')
GoogleAI(enterprise=True, api_key=key)         # Cloud express mode
GoogleAI(api_key=key)                          # unchanged
```

- **`enterprise=True`:**
  - `project`: the argument, then `GCLOUD_PROJECT` / `GOOGLE_CLOUD_PROJECT`, then `credentials.project_id`, then the ADC project.
  - `location`: the argument, then `GOOGLE_CLOUD_LOCATION` / `GCLOUD_LOCATION`, then `us-central1`. Multi-regions keep their dedicated endpoints.
  - Express mode only with an explicit `api_key=`. Environment keys stay ignored, as they are today, so an AI Studio `GEMINI_API_KEY` never quietly becomes a Cloud key.
- **`enterprise=False`:** the key comes from `api_key`, then `GEMINI_API_KEY`, then `GOOGLE_API_KEY`. Passing `project` or `location` raises a `ValueError` at setup.

## What still differs between backends

The namespace and refs stay the same, but some model IDs differ by backend:

- **Veo:** the IDs don't overlap. The Developer API uses `veo-3.1-generate-preview`, and Cloud uses `veo-3.1-generate-001`.
- **Gemini:** most names are shared. The `-latest` aliases (`gemini-flash-latest`, …) exist only on the Developer API.
- **Known-name lists** on `GoogleAI.gemini()` / `.veo()` become the union of both backends. A name the configured backend doesn't serve fails at lookup, and the error names the backend.

## Interactions API

The same switch covers Interactions. The request body is the same on both backends. What changes is the transport, plus a few field rules (for example, Cloud rejects `top_k`, and Cloud Interactions doesn't serve image models).

- **Developer API:** `v1beta/interactions` with an API key.
- **Cloud:** `v1beta1/projects/{p}/locations/{l}/interactions` with OAuth, or express mode at `locations/global/interactions`.

The Python Interactions work ([genkit-python#78](https://github.com/genkit-ai/genkit-python/pull/78), then [#105](https://github.com/genkit-ai/genkit-python/pull/105), then a Gemini-on-Interactions follow-up) uses the plugin's own HTTP client and wire types rather than the SDK's, matching JS and Dart. So two separate choices are in play:

- **The model picks the API.** Gemini, Antigravity, Lyria and Deep Research use Interactions through our own client. Imagen, Veo, embedders, image models and tuned endpoints use `generateContent` through `genai.Client`.
- **`enterprise=` picks the backend for both.** Either client sends to the Developer API or to Cloud based on the same setting. Gemini on Interactions works with either value.

**The ask for the Interactions work:** the Cloud client should choose its URL and auth from a backend value passed in, not from the plugin class, and the request function should take credentials instead of assuming an API key. That's the same amount of work now. `GoogleAI(enterprise=...)` then just connects the setting; nothing has to be moved out of `VertexAI`.

## At 2.0: the `vertexai/` alias

When `VertexAI` is removed, no plugin is named `vertexai`. The registry picks a plugin from the text before the first `/`, so `vertexai/gemini-2.5-flash` would never reach `GoogleAI` and would come back as a generic not-found. The core change is to let a plugin claim older namespace names:

```python
class GoogleAI(Plugin):
    name = 'googleai'
    legacy_namespaces = ('vertexai',)   # new, in genkit.plugin_api

await ai.generate(model='vertexai/gemini-2.5-flash', prompt='Suggest a dish.')
# => with enterprise=True: same action as googleai/gemini-2.5-flash
# => otherwise: GenkitError: NOT_FOUND, 'vertexai/gemini-2.5-flash' needs GoogleAI(enterprise=True)
```

- **Routing:** `vertexai/...` goes to `GoogleAI.resolve`, and the plugin is told which name was used. The backend check lives in the plugin, not in core.
- **Naming:** the resolved action is stored and traced under `googleai/...`. Today the registry names an action after whatever string was asked for.
- **Dev UI:** the model list doesn't show the alias, so models don't appear twice.
- **Cost:** new API for plugin authors, and JS and Go need the same thing at their own 2.0.

## What we give up at 2.0

- **Both backends in one app.** Until 2.0, `GoogleAI()` plus `VertexAI()` keeps working. After 2.0, two `GoogleAI` plugins collide on `googleai/`, and the registry raises at startup. We'd make that message name the fix. I expect this is rare, but we have no usage data.
- **A possible follow-up that doesn't break anyone:** a per-request backend override, if those apps show up. It's not part of this proposal. Config is merged from the ref, the `.prompt` file and the call, and middleware can rewrite it, so putting billing routing in config needs its own design. Background `Operation`s would also have to record their backend.

## Cross-SDK risk

Until JS and Go ship `enterprise=`, `googleai/gemini-2.5-flash` in a `.prompt` file means the Developer API in JS and Go but can mean Cloud in Python. The same file would bill a different account depending on which SDK loads it. In the additive phase, that only affects users who opt in with `enterprise=True`, but JS and Go should follow soon.

## Plan

1. **Now:**
   - Agree the Cloud Interactions transport shape with the Interactions migration owner.
   - Get sign-off on this proposal.
   - Merge [genkit-python#137](https://github.com/genkit-ai/genkit-python/pull/137), the Model Garden rename. It doesn't depend on this proposal, but it is breaking, so it needs to land before Python 1.0.
2. **The Python Interactions stack lands:** [#78](https://github.com/genkit-ai/genkit-python/pull/78), then [#105](https://github.com/genkit-ai/genkit-python/pull/105), then Gemini on Interactions with the Cloud transport.
3. **`GoogleAI(enterprise=...)`**, any 1.x release (ideally 1.0), stacked on the Interactions work:
   - the `enterprise`, `project` and `location` parameters;
   - `enterprise=` passed to the SDK client;
   - backend-specific checks keyed on `enterprise`;
   - the backend in traces;
   - internal renames;
   - the `VertexAI` deprecation warning.

   Additive.
4. **JS and Go:** the same additive change whenever they're ready.
5. **At 2.0, in each SDK:** core namespace aliases, then remove `VertexAI` and register the `vertexai/` alias.

## Open questions

1. **Is `enterprise` the long-term name?** The flag follows "Gemini Enterprise Agent Platform", a brand developer marketing has already moved away from. Can we confirm with the `google-genai` team, and is JS `@google/genai` moving too?
2. **Flag name in Genkit:** mirror each SDK (`enterprise=` in Python, a backend enum in Go), or use one brand-neutral `backend=` Literal everywhere?
3. **The environment variable:** should `GoogleAI()` honor `GOOGLE_GENAI_USE_ENTERPRISE` when `enterprise=` isn't passed? Honoring it lets deployment config switch the backend without code changes. Ignoring it avoids a switch caused by a variable set for some other tool. An explicit `enterprise=` wins either way. I lean toward honoring it and logging the chosen backend once.
4. **How long the alias lives after 2.0:** one major version, or for good?
