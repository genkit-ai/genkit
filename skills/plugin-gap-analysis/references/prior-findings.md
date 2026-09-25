# Prior findings

**Spoilers. Do not open this file while auditing.** It exists so that findings
from earlier runs are not lost, and so they are not sitting in the checklist priming the next
auditor. Read it after your list is written, as a completeness check.

An independent re-run reported that findings embedded in the checklist stopped some rows from being
real derivations - it confirmed conclusions it had been handed. Two separate runs also flagged that
per-plugin verdicts elsewhere in the skill primed them before they read any code. Everything of that
kind belongs here.

Treat every entry as dated and possibly closed. Verify against the current commit before reusing.

- **C7** - a Go plugin keyed its SDK wrapper-type remap on `reflect.Type.Name()` string literals,
  which degrades silently on an SDK rename rather than failing at build time. A sibling plugin did
  not share the problem, because its SDK uses plain pointer scalars with nothing to remap.
- **C8** - a reference-side schema ending `.passthrough()` spread unknown keys into the nested
  config object, so a field the provider places at the *request root* stayed unreachable while two
  sibling fields did not. The same passthrough let config overwrite messages, model, system and
  tools after the framework had built them.
- **C10** - an inline union discriminator (`{"type":"adaptive"}`) failed to round-trip through
  `map[string]any` into the SDK params struct, so the field vanished with no error on the untyped
  path while the typed path worked. Reachable only through the Dev UI and invisible to source
  reading.
- **E1** - a Go response switch handled three of the SDK's twelve content-block variants and
  returned an error for the rest, so a safety-redacted thinking block failed the whole request.
- **E4** - a Go plugin assigned the SDK's field-presence bookkeeping struct to the raw response
  field, which serialises to empty stubs; the intended value was the SDK's raw-JSON accessor.
- **F1** - a provider's overload code (529) fell through a generic `>= 500 -> Internal` branch in
  the framework's code table, landing the one retryable condition in the wrong retry class.
- **G1/G2** - a target README asserted a capability that two correctness rows showed did not work
  end-to-end. README claims are a docs dimension, never evidence of a feature.
- **H1** - the language with no unit test for its response-conversion path was the language whose
  response path carried five separate defects. Test-coverage gaps predict where the bugs are.

## Per-plugin run history

- `anthropic` (JS -> Go), 2026-08-18. Recorded at the time as one-way with the target behind.
  **Superseded - see the re-run below.**
- `google-genai` (JS <-> Go), same date. Direction came out two-way, which is what forced dimension
  group 0 and the direction-check step.
- **Verification pass over both.** Wrong rows pointed at the *reference* far more often than the
  target, so reverse-direction claims are the least reliable output of a first pass. Severity moved
  substantially in both directions, and the verifiers surfaced better bugs than the audit had -
  including a target-side raw-response field that serialised to empty stubs, breaking response
  inspection entirely.
- **Independent re-run of `anthropic`**, 2026-08-19, by an auditor given only this skill and no
  prior findings. Reached **two-way, net reference ahead** - overturning the earlier verdict - and
  found several defects both earlier passes missed: an untyped-config union that silently drops a
  field, server-side tools unreachable from JSON config, a reference that curates three retired
  model IDs, and a reference that advertises constrained generation then hard-fails it on its own
  default surface. Two of the three errors this skill's traps were written to prevent did not recur;
  the third did, and produced the SDK-environment-defaults trap in `verification.md`. Most of its
  strongest rows came from compiling probes against the pinned SDK rather than reading it, which is
  why that is now a rule.
