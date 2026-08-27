# genkit Python-floor shim

A one-time PyPI release (`genkit 0.3.0`, sdist only) that turns the
old-Python install experience from a silent wrong install into a clear
upgrade message.

## Why it exists

Genkit requires Python >= 3.10, but macOS ships Python 3.9 at
`/usr/bin/python3`. On that interpreter, `pip install genkit` used to
resolve `genkit 0.2.0` — an unrelated package from before the PyPI name
was transferred to this project ("random data like phone numbers") —
because those pre-transfer releases (0.1.0–0.2.0) declare
`Requires-Python: >=3.6` and pip silently falls back to them.

## How it works

- The shim declares `Requires-Python: <3.10`, so Python >= 3.10 can
  never select it. Fresh installs on current Pythons keep getting the
  real latest genkit.
- On Python <= 3.9, version 0.3.0 outranks the pre-transfer 0.2.0, pip
  tries to build the sdist, and `setup.py` stops with a boxed message:
  the 3.10 requirement plus Homebrew / uv / python.org instructions.
- It is published as an **sdist only**. A wheel would install cleanly as
  an empty package on old Pythons — exactly the silent failure this
  prevents. Keep the `--sdist` flag in `publish_python.yml`.

The version never changes: 0.3.0 was never published (the history goes
0.3.0.dev2 -> 0.3.1), and it sits above 0.2.0 (the last pre-transfer
release) and below 0.3.1 (the first real genkit release).
`skip-existing: true` in the publish workflow makes re-publishing it on
every release tag a no-op after the first upload.

This directory is intentionally not a uv workspace member: it shares the
`genkit` project name with `packages/genkit`, and workspace project
names must be unique.

## Verifying

```bash
# Build (any Python >= 3.10):
uv build --sdist py/floor-shim --out-dir /tmp/shim-dist

# Old Python shows the boxed message (expected: install fails with instructions):
/usr/bin/python3 -m venv /tmp/v39 && /tmp/v39/bin/pip install --no-index \
  --find-links /tmp/shim-dist genkit

# Current Python ignores the shim even when visible (expected: installs real genkit):
python3.13 -m venv /tmp/v313 && /tmp/v313/bin/pip install \
  --find-links /tmp/shim-dist genkit
```

Related cleanup: the pre-transfer releases 0.1.0, 0.1.3, 0.1.4, and
0.2.0 should be yanked on pypi.org so pinned installs are the only way
to reach them.
