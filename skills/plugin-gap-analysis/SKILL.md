---
name: plugin-gap-analysis
description: Audit a Genkit provider plugin in one language against its reference implementation in another, adversarially verify every finding, and produce a flat, reviewable Markdown gap list. Use when asked to gap-analyse a plugin, compare Go and JS plugin parity, find what a plugin is missing, or prepare child issues under a plugin parity epic.
---

# Plugin gap analysis

Produce a **flat, severity-tagged Markdown list of gaps** between a provider plugin in a
target language and the same plugin in a reference language, then **verify every row before
handing it over**. Output is Markdown; a human reviews it before anything is filed.

**Roughly half of an unverified first pass needs rework** - some rows factually wrong, more of
them right about the fact and wrong about the detail or the consequence. Phase 6 is therefore not
optional, and a list that has not been through it must be labelled as unverified.

## Arguments

- `plugin` (required): `anthropic`, `google-genai`, `compat-oai`, ...
- `target` (default `go`), `reference` (default `js`)

## Scope

**JS and Go only. Python is out of scope** - it is mid-migration to a separate `genkit-python`
repo, so a parity claim against what is in this repo would be stale on arrival. If asked for a
Python comparison, say it is out of scope and why rather than auditing a moving target.

## Direction

Default: **JS is the reference, Go is the target** - JS is the reference implementation and Go
plugins have had less focused attention.

**Verify direction per plugin before scoping.** The anthropic audit came out one-way, Go behind
on nearly everything. The google-genai audit came out two-way: Go led on config surface,
README, samples, transport options and error metadata, while JS led on model-family coverage.
Do dimension group 0 first, tally which side leads per group, and state the verdict at the top
of the report. Where the target leads half the groups, this is a convergence, not a catch-up -
and a workplan that says "bring the target up to the reference" will *remove* capability.

Record reverse-direction gaps in the same flat list either way. The goal is experience parity,
not a one-way feature checklist.

**Do not read LOC as depth.** google-genai was 13137 JS to 3916 Go, but most of the difference
was three model families and an agents layer Go does not serve, not depth on the shared path.

## Non-goals

Do not propose designs, do not fix anything, do not file issues without being asked. Do not
compare against the provider's upstream API - the reference implementation is the baseline. If
a gap looks like it needs a redesign rather than a fix, tag it and say so; that is a signal to
stop, not to keep going.

## Workflow

**1. Pin the commit.** Record the SHA you are auditing and use it for every citation. Also
record the **pinned SDK version on each side** (`go/go.mod`, the installed
`node_modules/.../package.json`). Version skew masquerades as a plugin bug - see the SDK-skew
trap in `references/verification.md`.

**2. Locate both plugins.** See `references/locators.md`. Layouts differ per language and per
plugin, and a plugin may have more than one entry point in a language (Go serves Claude from
`go/plugins/anthropic`, `go/plugins/compat_oai/anthropic` and `go/plugins/vertexai/modelgarden`).
Name every entry point found and say which you audited.

**Never conclude "the other language has no plugin for this" from one directory listing**, and
never from an on-disk listing alone - confirm the path is tracked at your pinned commit. The same
family is named differently per language (`compat-oai` in JS, `compat_oai` in Go) and may be
modelled differently on each side.

**3. Walk `references/dimensions.md` in order.** It is the frozen checklist. **Group 0
(coverage) comes first and is not optional** - it decides both scope and direction. For each
dimension, read the reference side, read the target side, record a row or record nothing. Do
not improvise the dimension list; if the plugin forces a question the checklist does not cover,
answer it *and* append the dimension to the checklist file.

**4. Cite everything, and make citations resolvable.** Every row needs at least one `path:line`
on the side that has the feature, preferably both. Paths are repo-relative from the root.
Confirm each cited path exists at the pinned commit - one that does not is a signal, usually an
untracked directory or a file introduced by an unmerged PR.

**5. Tag severity and confidence.** Number rows with a per-section prefix (`ANT-1`, `GGA-1`,
`DEC-1`) so a row survives being pasted into an issue on its own.

| Tag | Meaning |
|-----|---------|
| **X** | Correctness bug - a valid provider response or a documented config fails |
| **T** | Table stakes - a feature or config surface users reasonably expect |
| **C** | Consistency / experience - naming, docs, samples, ergonomics, thin-but-working |

Confidence is `confirmed` (read both sides), `needs-repro` (plausible from the code, wants an
E2E reproduction first) or `not-audited` (deferred, say why). Never report `needs-repro` as
fact.

**6. Adversarially verify every row. Mandatory.** Follow `references/verification.md`. Split
the rows across several independent checkers with **fresh context**, each given the claims
alone and told to *refute* them. Reconcile the verdicts: rewrite wrong rows, tighten imprecise
ones, and keep a `corrected` marker plus a sentence saying what the original got wrong, so the
trail is auditable rather than quietly rewritten. Expect severity to move in both directions.

**7. Sweep for work already in flight.** Before anything is filed, check open PRs - see
`references/in-flight.md`. Expect a meaningful share of rows to be covered already; one trial run
found a PR containing the exact fix for a row the audit had recorded as an open bug. Also flag PRs
that are a *sequencing hazard* for a row rather than a fix for it.

**8. Emit Markdown: flat list, then buckets, then in-flight.** Format in
`references/output-template.md`. The flat list is what the reviewer validates; buckets are for
scoping afterwards. Do not curate, merge or drop rows to make the list shorter - a reviewer can
strike a row, but cannot recover one you never wrote.

**9. Stop.** Hand the list over. Filing child issues is a separate, explicitly requested step,
and rows covered by an open PR should not be filed at all.

## Rules that keep runs comparable

- **Read the code, not the README.** A README claim is a docs dimension, not evidence of a
  feature. Several rows came from a config description advertising a capability the response
  path could not handle.
- **`grep` for absence, cite for presence** - and treat every absence claim as suspect until
  you have ruled out the escape hatches in `references/verification.md`. Absence claims were
  the single largest source of wrong rows. Say which command you ran.
- **Diff catalogs both ways.** Curated model lists drift independently; neither side is
  reliably a superset.
- **Exhaustive switches are gap sites.** For every `switch` over provider block types, stop
  reasons or part types, list the cases each side handles and diff them. A missing case in a
  `default: return error` branch is an **X**, not a **C**.
- **Verify the consequence, not just the fact.** "So the request fails" must be traced to the
  point of failure. One row correctly identified a hardcoded flag but claimed a failure that
  could not occur, because no failed tool could reach that code path.
- **A missing family is one row.** Do not expand it into a row per config field it would have
  had.
- **Folding a restricted family into the generic one is an X - but check both sides do not do
  it.** A restricted family inheriting generic capabilities advertises what the model lacks;
  that is only a *parity* gap if the reference handles it correctly.
- **Shared code changes the blast radius.** In Go, `go/plugins/internal/<provider>` is shared by
  several plugins; note when a fix lands in more than one place. In JS, a plugin serving two
  backends from sibling trees can drift from *itself* - measure the overlap before calling it
  duplication, and defer it to its own same-language run.
- **Watch for rows that recur across plugins.** Recurring rows are cross-plugin decisions; say
  so, so they are not filed as per-plugin issues.
- **Keep distinct decisions distinct.** Do not fold a locally-fixable defect into a broad
  design decision because they touch the same code. One run wrongly folded an SDK-specific
  type-mapping bug into the config-provenance decision; the other plugin did not share the
  problem at all.
- **Out-of-repo dimensions get flagged, not skipped.** Doc-site pages are not in this repo -
  emit a row saying so rather than silently dropping the dimension.

## Cross-plugin decisions found so far

These recur. File them once, at the API-design level, not per plugin.

1. **Config schema provenance** - curated schema with Genkit-flavoured names (JS) vs the
   provider SDK's own struct, reflected and annotated (Go). Decides the entire user-facing
   config vocabulary. Note the evidence cuts both ways: the reference's hand-curated schemas
   were themselves found to reject values the provider accepts, which is the drift cost
   curation incurs.
2. **Per-request auth** - the reference supports a request-level API key and deferred plugin
   auth; Go resolves the key once at init. Verification found these are *two* asks - multi-tenant
   key routing and startup ordering - sharing one blocker, plus a third context-carried-secret
   mechanism for background-model check and cancel.
3. **Per-model capability override** - Go's `Models`/`Embedders` maps let a caller describe a
   model newer than the plugin; JS has no equivalent. Do not claim this causes catalog
   staleness: verification showed the reference already resolves unknown IDs and lists from the
   live models API, so the override is a user escape hatch, not a freshness mechanism.

## Prior runs

- `anthropic` (JS -> Go), 2026-08-18 at `ccfe1093d`. Direction one-way as expected. Corrected one
  seed item that had already landed. Several correctness rows were missing cases in
  response-conversion switches.
- `google-genai` (JS <-> Go), same commit. **Direction came out two-way**, which is what forced
  dimension group 0 and the direction-check step.
- **Verification pass over both.** Wrong rows pointed at the *reference* far more often than the
  target, so reverse-direction claims are the least reliable output of a first pass. Severity moved
  substantially in both directions, and the verifiers surfaced better bugs than the audit had -
  including a target-side `raw` response field that serialised to an empty object, breaking
  response inspection entirely.
