# Output template

Markdown. Row IDs carry a per-section prefix so a single row survives being pasted into an
issue on its own: `ANT-`, `GGA-`, `DEC-` for decisions, `IF-` for in-flight items.

```markdown
# Plugin gap audit

Audited <date> at `<commit>`. SDK versions pinned: <target> `<ver>`, <reference> `<ver>`.

Then **one of the two headers below**, never neither.

*If phase 6 ran:*

> Every row was re-checked against the code by <N> independent reviewers working from the claims
> alone. Rows whose original claim was wrong or materially understated carry `corrected` in their
> tag and say what the original got wrong. <Which direction the wrong rows pointed, so the reader
> knows which half of the list is least settled.>

*If phase 6 did not run* - required wording, at the top, before any row:

> **THIS LIST IS UNVERIFIED.** Phase 6 (adversarial verification) <and phase 7 (the open-PR
> sweep)> did not run. Roughly half of an unverified first pass needs rework, and wrong rows skew
> toward claims about the reference implementation - of which this list contains <n>. Treat every
> row as a hypothesis with evidence attached, not a finding. Nothing here should be filed as an
> issue until the missing phases run.

Do **not** respond to being unable to verify by shortening the list to only the rows you are
confident about. Keep everything and label harder - a reviewer can strike a row, but cannot
recover one that was never written.

Severity: **X** = correctness bug (a valid provider response or documented config fails);
**T** = table stakes; **C** = consistency/experience.
Confidence: **confirmed** = both sides read; **needs-repro** = plausible from code, wants an E2E
reproduction before filing; **not-audited** = deferred.

## <plugin>: <reference> to <target>

Reference: `<ref path>` (<LOC>). Target: `<target paths>` (<LOC>). <Direction verdict — and if
two-way, say so before the rows, because it changes scoping.>

### Corrections to the seed list

Only when the caller supplied known gaps to check. A "seed list" is whatever the requester already
believes is broken - from a meeting, an epic, or an earlier audit. **Omit this section entirely if
there was none**; do not emit the heading with a disclaimer.

- ~~"<seed claim>"~~ — **not a gap.** <evidence with `path:line`>. What is actually missing is
  <ROW-ID>.

### Gaps

**ANT-1. <One-line claim.>**  `[T, confirmed]`

<Reference behaviour with `path:line`. Target behaviour with `path:line`, or the exact grep that
shows absence. Consequence for the user.>

**ANT-2. <One-line claim.>**  `[X, confirmed, corrected]`

<Originally stated as "<the wrong claim>". <What is actually true, with evidence.>>

### Buckets

- **Missing features (T):** row IDs — one-line theme.
- **Correctness bugs (X):** row IDs — one-line theme.
- **API shape decisions (T/C):** row IDs — design decisions, not independent tickets.
- **<reference>-side gaps (reverse direction):** row IDs — the reference is not a superset.
- **Docs and samples (C):** row IDs.

### Not audited

<Entry points found but not audited; languages out of scope.>

## Cross-plugin decisions

**DEC-1. <Decision.>**  `[decision]`

<The choice, the row IDs it came from, and which rows stop being gaps under each option. Keep
decisions distinct — do not fold a locally-fixable defect in because it touches the same code.>

## Work already in flight

<Coverage table: Row | PR | Author | State and scope — including what each PR does *not* cover.>

**IF-1. <Sequencing hazard.>**  `[risk]`

<The PR, the row it triggers, and the required ordering.>

**IF-2. <Provenance to confirm.>**  `[open question]`

<What is unclear, and who to ask.>

### Rows with nothing in flight

- <Group>: row IDs. <The grep evidence that establishes it.>

### Adjacent PRs worth watching

- #NNNN (author) title — adjacent to <row IDs>.
- Discount #NNNN: <why it is a stale branch rather than a real overlap>.
```

## Row rules

- One gap per row. Do not merge related gaps to shorten the list.
- Bold the claim, then evidence, then consequence. A reviewer reads the bold text first and
  only drops into the evidence for rows they doubt.
- Absence is evidenced by a named grep, not by assertion.
- A row whose fix is a design decision goes in the flat list *and* the design bucket, so it is
  not filed as an ordinary child issue by mistake.
- Reverse-direction gaps stay in the same flat list, tagged in the text as reverse-direction.
  They are real parity gaps; they just point the other way.
- A row that verification dissolved stays in the list, restated as "no gap here, and why".
  Deleting it invites the next run to re-file it.
- `corrected` rows open with what the original claim got wrong. The audit trail is the point;
  a reviewer who spot-checks one corrected row should be able to see the correction was
  deliberate.
- Paths are repo-relative from the root, so a citation resolves without guessing. Where the
  output will be read alongside the repo, link them at the pinned commit rather than the
  default branch — line anchors drift.
- Recount every counting claim (lines, sections, samples, catalog members) rather than reasoning
  from an earlier count. They are the fastest way for a reviewer to lose trust. Check which
  spelling a file uses before counting it - counting `test(` in a suite written with `it(` undercounts
  by a factor.
- Where a defect is severe on one code path and harmless on another, tag it by its worst reachable
  path and state the condition in the evidence. A config bug that only bites through the Dev UI is
  still a correctness bug.
