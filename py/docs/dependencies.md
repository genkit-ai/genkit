# Python Dependency Policy

The Python workspace resolves every package, sample, and tool into one
`uv.lock`. CI installs with `uv sync --frozen`, so nothing changes version
until someone edits the lockfile. Users don't get the lockfile, though. They
resolve the specifiers in `packages/*/pyproject.toml`. This page covers both.

## Caps (upper bounds) in published packages

A dependency gets a cap at the next major (`<N+1`, or `<1` while it's 0.x) if
it passes either test:

1. **Integration target:** a genkit package exists to wrap it. Examples:
   `openai` (genkit-openai), `flask` (genkit-flask), `boto3`
   (genkit-amazon-bedrock).
2. **Contract:** users write code or files against it and hand them to genkit.
   Examples: `pydantic` (schemas are `BaseModel` subclasses), `dotpromptz`
   (`.prompt` files).

Everything else stays floor-only: starlette, httpx, anyio, structlog, uvicorn,
and so on. These are internal implementation details. A cap on them would stop
users from taking upstream fixes, including security fixes that only ship in a
new major, until we cut a release. It would also make their installs fail when
another library needs the newer major.

`REQUIRED_CAPS` in `scripts/check_consistency.py` lists every capped dep and
the test it passes. CI fails if a listed dep has no cap, or if an unlisted dep
has one. To add a cap, add the dep there with its reason.

If a dep is a contract in only one package, add it to `SCOPED_CAPS` instead.
That caps it in that package and leaves it floor-only everywhere else. For
example, `genkit-google-genai` caps `httpx<1` because `http_client_factory`
returns an `httpx.AsyncClient`, while core `genkit` uses httpx internally.

To exclude one known-bad release, use `!=x.y.z` instead of a cap.

## Floors (lower bounds) in published packages

A floor promises that genkit works with any version at or above it. Our code
is the only thing that breaks that promise: a PR starts using an API that the
floor version doesn't have.

1. **Detect:** the `lowest-deps` job in `python.yml` resolves every direct dep
   at its floor (`uv lock --resolution lowest-direct`) and runs the suite on
   Python 3.10. It's part of the required `python-checks` gate.
2. **Fix:** if it fails, find the real floor:

   ```bash
   py/bin/find_floor genkit-openai openai
   # => openai==1.64.0: FAIL (3 failed, 486 passed)
   #    openai==1.68.0: PASS (489 passed)
   #    genkit-openai: openai lowest passing 1.68.0 (declared floor 1.60.0)
   ```

   It binary-searches PyPI between the declared floor and the locked version.
   If it's too slow, use the locked version: a floor that's too high fails
   loudly at install time, while a floor that's too low fails at runtime.
3. **Bump in the same PR** that started using the newer API.

Don't raise a floor just because a newer version exists. Raise it on purpose
only when:

- **Dropping support:** do it at a genkit minor release and put it in the
  release notes.
- **A security fix we expose:** only when our code path reaches the
  vulnerable code. Leave a comment next to the floor saying why.

## Dev and lint groups, samples

- **Dev and lint groups (root `pyproject.toml`):** cap at the next major. These
  never ship, so a cap only limits what `uv lock --upgrade` can move.
- **Samples:** declare what the sample imports and nothing else. Samples are
  workspace members, so their specifiers join the shared resolution.
- **Imports in tests and samples:** if a test or sample imports a package,
  declare it. Don't rely on another tool happening to pull it in.

## Lockfile updates

1. **Minor and patch (weekly):** Dependabot opens one grouped PR after a 7-day
   cooldown. Merge it if CI is green. A red one means a minor release broke
   us, and that needs a fix, not a cap.
2. **Majors (weekly):** Dependabot opens a separate `py-major` PR. For capped
   deps it widens the cap (`<3` to `<4`) in every member and updates the lock.
   - Green: merge it, then release so users get the new range.
   - Red: that's the migration ticket. The cap keeps users safe until it lands.
3. **Security alerts:** Dependabot security PRs skip the cooldown and the
   major/minor grouping. To clear a batch by hand:

   ```bash
   # 1. Find vulnerable locked packages
   uv export --frozen --all-packages --all-groups --no-hashes --no-emit-workspace -o /tmp/req.txt
   uvx pip-audit -r /tmp/req.txt --no-deps --disable-pip

   # 2. Upgrade only those packages, within existing bounds
   uv lock --upgrade-package urllib3 --upgrade-package cryptography
   ```

   If a package won't move, a parent is holding it back. Run
   `uv tree --invert --package <name>` and upgrade the parent too.
4. **Newest-version canary (daily):** `python-deps-canary.yml` resolves every
   dep to the newest version our bounds allow and runs the suite. That's what
   a fresh `pip install` gets today. It isn't a PR gate, because its failures
   come from upstream releases, not from the PR.
   - On failure it opens one `py-deps-canary` issue, comments on it while the
     canary stays red, and closes it on the next green run.
   - Fix the code or exclude the bad release with `!=`. Cut a patch release if
     users are affected.

Avoid a blanket `uv lock --upgrade` in feature PRs. It moves every uncapped
dependency at once, and reviewers can't tell which bump caused a failure.
