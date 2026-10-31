# PR descriptions

Reviewers mostly care about one thing: what this does to the public API. Write the description so they can judge that without reading the diff.

The title says what the user sees, in present tense, subject first: `fix(py)!: a failed agent turn returns res.error`. Add `!` when it breaks someone. Leave out the implementation; "refactor X internals" says nothing to a reviewer.

Open with one or two sentences: what a caller can do now, or what no longer breaks. Then a short snippet showing the call a user would write and what comes back, with `# before` / `# after` if the behavior changed. Include the exact error text if that's what changed.

## Changes

One bullet per change a user could notice, biggest first: a new or removed name, a new argument, a different return value, a new error, a different default. Say what it is now and why it belongs on the surface, in terms of what the user gets. Leave out internal moves and renames nobody imports. If a bullet needs more than two or three sentences, it's probably two changes.

## Breaking changes

List every change to the API surface that could break existing code: removed or renamed names, moved imports, arguments that became required or keyword-only, different return types, new errors where there used to be none. For each one, show what the old code does now and what to write instead:

```python
# before
from genkit_vertexai.model_garden import ModelGarden
# after
from genkit_vertexai import ModelGarden
```

If nothing breaks, say "None." so reviewers know you checked.

## Cross-SDK

Some of what Python writes is read by the other SDKs and by the tooling, so a change there isn't Python-only. That includes:

- the shared schema (`genkit-tools/genkit-schema.json`) and anything generated from it
- trace and span attributes the Dev UI reads
- the reflection API
- the HTTP flow format (`{"data": ...}`, `{"result": ...}`, SSE frames)
- `.prompt` frontmatter
- config keys as they show up in traces and the Dev UI form
- status names, finish reasons, and action kinds and metadata

If your PR changes any of these, add this section. Say what changed on the wire, show one example of the JSON before and after, and say what the JS, Go and Dart owners need to do: nothing, update a schema, handle a new field, or stop sending an old one. If Python now writes something the others don't yet read, or reads something they don't write, say so plainly so it doesn't get found in a broken trace.

If nothing on the wire changed, leave the section out. A Python-only rename doesn't belong here, as long as the JSON stays the same.

## Open question

Put anything you haven't decided under `## Open question` with the options, so nobody approves a choice nobody made.

Leave out the story of how you got here, review-thread history, and "matches JS" as a reason. Describe the end state and why it's right for the user.
