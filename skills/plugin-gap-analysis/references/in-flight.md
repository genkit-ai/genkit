# Sweeping for work already in flight

Run after verification, before anything is filed. **Expect a meaningful share of rows to be
covered already** - in one trial run, a PR in review contained the exact fix for a row the audit
had recorded as an open bug. Filing those as child issues duplicates work that needs a rebase and
a review, not a ticket.

## How to sweep

PRs go to the public upstream (`genkit-ai/genkit`), not the private mirror - check `git remote -v`
and target upstream explicitly.

```
gh pr list --repo <upstream> --state open --limit 250 \
  --json number,title,author,isDraft,updatedAt,files
```

Filter by **files touched**, not title. Titles miss things and mislead: one PR titled as a
Python change carried diffs to the Go plugin, and one titled "polish X support" created X from
nothing. Match paths against the plugin directories from `locators.md`.

Then, for each row with no title-level match, check whether *any* open PR touches the specific
file the row cites. That is what establishes "nothing in flight" credibly:

> Only four open PRs touch `<the cited file>` and none address these.

Keyword-search bodies for the row's subject too, but expect mostly noise - a handful of terms
produced a couple of real hits and a lot of unrelated matches.

## What to record per covered row

Row ID, PR number, author, state, and **what the PR does not cover**. Partial coverage is the
common case:

- a PR adding one model of a family, where the row is about the family
- a PR fixing one direction of a two-way row
- a PR solving the row with a *different API shape* than the reference uses, which converts the
  remainder from missing work into a design decision
- a draft covering half of a two-part row

Also record merge state. Every feature PR in the first sweep was conflicted against main and
months stale, which turned the workplan's "coordinate with in-flight infra PRs" from a
forward-looking risk into a present blocker - and made an already-approved, merely-conflicted PR
the cheapest available win.

## Two categories beyond simple coverage

**Sequencing hazards.** A PR that is not a fix for a row but will *trigger* it. A pending
dependency bump jumped an SDK 39 minor versions, which is exactly the silent-breakage scenario
one row described, with no test guarding it - and would also invalidate two other rows'
conclusions. Emit as its own row with a `risk` tag and state the ordering.

**Provenance to confirm.** Where a PR's description implies a baseline the audit could not find,
say so and name who to ask rather than assuming either way. Do not treat the row as covered
until it is resolved.

## Noise to discount explicitly

A stale branch shows diffs for work already merged at your commit. Check whether the diff's
"new" lines are already present at the audited SHA before treating it as an overlap, and say in
the report that you discounted it - otherwise the next reader re-investigates it.
