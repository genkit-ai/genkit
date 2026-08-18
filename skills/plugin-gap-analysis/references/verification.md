# Adversarial verification

Mandatory phase. **Roughly half of an unverified first pass needs rework** - some rows factually
wrong, more of them right about the fact and wrong about the detail or the consequence. Skipping
this ships a list a reviewer will lose confidence in on the first row they check.

**Wrong rows skew heavily toward the reference implementation.** Claims that the *reference* is
missing something failed far more often than claims about the target, because the reference tends
to have more escape hatches and more places to look. Weight the checking accordingly.

## How to run it

Split the rows across **several checkers with fresh context**, each covering a bounded range
(15-25 rows). Fresh context is the point - a checker that helped write the rows will confirm its
own reasoning. Give each checker the claims *as text*, including the cited `path:line`, and tell
it to **refute**.

Bias the split so reverse-direction rows are concentrated and explicitly flagged as the least
reliable, and warn the checker that the audit's author may have been over-eager to find them.
Name the specific escape hatches to check (below) - a checker told only "be skeptical" re-runs
the same greps and reaches the same wrong answer.

## Verdict taxonomy

| Verdict | Meaning |
|---|---|
| `CONFIRMED` | Accurate as stated, both sides read |
| `WRONG` | Factually false. Must say what is actually true |
| `IMPRECISE` | Directionally right, but detail, line number or consequence is off. Must say precisely how |
| `UNVERIFIABLE` | Cannot be determined from the repo. Must say what is missing |

Require: open every cited line rather than trusting it; re-run every claimed grep; cite own
`file:line` evidence. A cited line that is off but where the claim holds elsewhere in the file
is `IMPRECISE` with the correct location, not `WRONG`.

Ask each checker for two extra sections: **where the audit understated a problem**, and **any
additional bug noticed while checking**. Both were high-yield - the verifiers surfaced better
bugs than the audit itself.

## Traps that produced the wrong rows

Every one of these caused at least one wrong or imprecise row. Check them by name.

**Passthrough and untyped escape hatches.** A schema ending `.passthrough()` that spreads
unknown keys into the wire request means a field absent from the schema may still be reachable.
Two "unreachable in the reference" rows were wrong for this reason. Check *where* the provider
places the field: one sibling field was genuinely unreachable because it sits at the request
root while passthrough only reaches the nested config object. The distinction is the row.

**Sibling backend directories.** A plugin serving two backends from sibling trees needs both
searched. One row asserted a transport option was missing when it existed on one backend and
not the other - and the row had the direction backwards.

**Second package location, and on-disk vs tracked.** Searching one plugin directory produced a
"no plugin in this language" claim that was false. Worse, a directory present on disk was
entirely untracked at the pinned commit, so a path cited from a local listing did not exist in
the repo at all. Always confirm with the version-control tree, not `ls`.

**SDK version skew.** The two languages pin different SDK versions. One row blamed the target
for a stale enum description when the real defect was the *reference* rejecting a value its own
newer SDK accepts. Record both pinned versions and attribute the gap to the right side. Check
whether a pending dependency bump would change the row's conclusion.

**Files that do not exist yet.** A path cited from an open PR's branch will not resolve at the
audited commit. That is a signal the row is already being worked on, not a broken citation.

**Deprecated fields.** One row reported a field as unpopulated on the target; the field was
deprecated on both sides, and the real defect was in its replacement, which serialised to an
empty object. Check whether the field you are comparing is the one that still matters.

**Consequences that do not follow.** "So the request fails" needs tracing. One row correctly
identified a hardcoded error flag but its stated consequence was impossible, because the
framework aborts before that code path can see a failed tool. Downgrade to a latent wart.

**Both sides doing the wrong thing.** A capability mismatch is only a *parity* gap if the
reference gets it right. One correctness row dissolved because both implementations advertised
the same wrong capability set - and the genuine divergence was elsewhere in the same file.

**Counting claims.** Line counts, section counts, sample counts and catalog membership were all
wrong somewhere. They are trivially checkable and therefore the fastest way for a reviewer to
lose trust. Recount rather than reasoning from an earlier count.

## Reconciling

- Rewrite `WRONG` rows around whatever the verifier found instead; if there is no gap, say so
  plainly and keep the row so the question is not re-asked.
- Fold `IMPRECISE` corrections into the evidence text.
- Mark every changed row `corrected` in its tag and open its evidence with one sentence on what
  the original got wrong. Silent rewriting destroys the audit trail.
- Re-derive severity afterwards; it moves in both directions. In the last run, four rows were
  promoted to correctness bugs and one demoted to a latent wart.
- Promote verifier-found bugs to full rows with their own IDs.
- Re-check the buckets and any cross-references - they go stale when severity moves.
- State the verification tally at the top of the report, including which direction the wrong
  rows pointed.
