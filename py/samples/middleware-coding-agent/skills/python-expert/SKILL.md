---
name: python-expert
description: Conventions for clean, idiomatic Python. Load whenever you read, edit, or write Python source files.
---

# Python expert

When working with Python in this workspace, follow these conventions:

- **Type-hint everything.** Parameters, returns, attributes, locals where the type isn't obvious.
- **Prefer dataclasses** for simple data containers over hand-written `__init__`s.
- **Raise specific exceptions** (`ValueError`, `KeyError`, `LookupError`) with informative messages. Avoid bare `Exception`.
- **Don't swallow errors.** Don't `except Exception: pass`. Let unexpected errors propagate.
- **Match the surrounding style.** If the file uses single quotes and 4-space indent, match it. Don't reformat unrelated lines.
- **Comments explain why, not what.** Skip narration like `# loop over items`; only comment non-obvious intent.
- **Small, focused edits.** When fixing a bug, change only what's necessary. Leave the rest of the file untouched so the diff stays readable.
