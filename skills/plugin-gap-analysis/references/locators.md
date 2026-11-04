# Where plugins live

Layouts differ per language and per plugin, and these tables go stale. Confirm everything against
the tracked tree at your pinned commit, not against `ls` - see below.

## Scope: JS and Go only

**Python is out of scope.** It is mid-migration to a separate `genkit-python` repo, so anything
Python-shaped in this repo is in flux and a parity claim against it would be stale on arrival.
Do not audit it, do not cite it, and do not file a row for a plugin "missing" in Python.

If a caller asks for a Python comparison, say it is out of scope and why, rather than producing
a list against a moving target.

## Confirm paths against the tracked tree, not the filesystem

Verify every path you cite is tracked at your pinned commit. An on-disk listing is not evidence:
directories exist in working copies that are not in the repo at all.

```
git cat-file -e <sha>:<path>     # exists?
git cat-file -t <sha>:<path>     # blob or tree?
```

A path that fails is one of three things, each meaning something different: an untracked local
directory (drop the citation), a file created by an unmerged PR (a signal the row is already
being worked on - see `in-flight.md`), or a typo.

One tracked path that is easy to get wrong: the CLI and tooling live at **`genkit-tools/`** in
the repo root, not under `js/`.

**Several roles a plugin appears to own actually live in the framework**, and a row is invisible
until you follow the call out of the plugin tree. Known hops: the HTTP-status-to-Genkit-status code
table (`go/core/status`), untyped-config deserialization (`go/ai/config.go` into
`go/internal/base/json.go`), media-part URI handling (`go/plugins/internal/uri`), and the response
and finish-reason types (`go/ai/gen.go`). A defect in any of these surfaces as a plugin bug and
fixes in the shared ones land across every plugin that calls them.

## Per plugin

| plugin | JS | Go |
|--------|----|----|
| anthropic | `js/plugins/anthropic` | `go/plugins/anthropic` + shared `go/plugins/internal/anthropic` |
| google-genai | `js/plugins/google-genai` (sibling `src/googleai/` and `src/vertexai/` trees over a shared `src/common/`) | `go/plugins/googlegenai` (one package, `GoogleAI` and `VertexAI` structs) |
| openai / compat-oai | `js/plugins/compat-oai` | `go/plugins/compat_oai` (+ `openai`, `xai`, `zai`, `kimi`, `deepseek`, `dashscope`, `anthropic` sub-plugins) |

Legacy plugins overlap the modern ones and are separate packages: `js/plugins/vertexai` (which
is where JS serves Vertex-hosted Claude, via `src/modelgarden/`) and `go/plugins/vertexai`
(`modelgarden`, `vectorsearch`). A row about a "missing" backend often lives in one of these.

Note the naming: JS uses kebab-case directories, Go uses no separator or an underscore. Names do
not line up across the two either - the OpenAI-compatible family is `compat-oai` in JS and
`compat_oai` in Go, and the two are not the same shape, since Go factors sub-providers into
separate sub-packages. Establish what each language actually models before diffing
feature-by-feature.

A plugin present in only one language is an inventory fact worth stating, not a gap to file
against the other.

## Additional entry points

A provider is often served from more than one place. Find them before auditing:

```
find . -maxdepth 5 -type d -iname "*<provider>*" \
  -not -path "*/node_modules/*" -not -path "./.claude/*"
```

Known cases for Claude in Go: `go/plugins/anthropic`,
`go/plugins/compat_oai/anthropic`, `go/plugins/vertexai/modelgarden/anthropic.go`.
The first and third share `go/plugins/internal/anthropic`, so a fix there lands in both.

## Do not assume a file layout

There was a file-role map here - which file holds the config schema, the catalog, the conversion
layer - and it was wrong for the second plugin anyone audited, which keeps its whole implementation
in the plugin package with no shared internal directory. A table that is wrong for the plugin in
front of you, and carries a disclaimer saying it might be, is worse than no table: it invites a
citation to a path that does not exist.

Two `git ls-tree` calls replace it. Locate the roles in the plugin you are auditing.

## Not in this repo

Doc-site pages. `docs/` here holds only internal specs (`model-spec.md`,
`reflection-v2-protocol.md`, `agents-conformance-testing.md`). Doc-site parity is tracked in
the docs repo.
