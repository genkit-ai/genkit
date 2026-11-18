# Python Dependency Policy

The Python workspace resolves every package, sample, and tool into one
`uv.lock`. CI installs with `uv sync --frozen`, so nothing changes version
until someone edits the lockfile. This page covers who edits it and how.

## Version bounds

- **Published packages (`packages/*`)**: set a floor at the oldest version we
  support. Add an upper cap only for a dependency whose next major is known to
  break us (provider SDKs, `mcp`, `dotpromptz`). Don't cap infrastructure
  libraries like `starlette`, `httpx`, or `anyio`. A cap in a published
  package stops users from taking a fix that only ships in the next major, and
  it can make their resolver fail outright. `starlette` is an example: its
  2026 CVEs are fixed only in 1.x.
- **Dev and lint groups (root `pyproject.toml`)**: cap at the next major
  (`<N+1`, or `<1` for 0.x tools). These never ship, so a cap only limits what
  `uv lock --upgrade` can move.
- **Samples**: declare what the sample imports and nothing else. Samples are
  workspace members, so their specifiers join the shared resolution.
- **Dev-only imports**: if a test or sample imports a package, declare it. A
  package that only arrives through some other tool disappears when that
  tool is removed.

## Lockfile updates

1. **Minor and patch (weekly)**: Dependabot opens one grouped PR after a 7-day
   cooldown. Merge it if CI is green.
2. **Security alerts**: Dependabot security PRs skip the major-version ignore
   and the cooldown, so they still arrive for major-only fixes. You can also
   clear a batch by hand:

   ```bash
   # 1. Find vulnerable locked packages
   uv export --frozen --all-packages --all-groups --no-hashes --no-emit-workspace -o /tmp/req.txt
   uvx pip-audit -r /tmp/req.txt --no-deps --disable-pip

   # 2. Upgrade only those packages, within existing bounds
   uv lock --upgrade-package urllib3 --upgrade-package cryptography
   ```

   If a package won't move, a parent is holding it back. Run
   `uv tree --invert --package <name>` and upgrade the parent too.
3. **Majors (by hand)**: one PR per package. Raise the bound and the lock
   together, read the changelog, and run the full CI matrix. Dependabot
   version updates ignore majors on purpose.

Avoid a blanket `uv lock --upgrade` in feature PRs. It moves runtime majors
for every uncapped dependency at once, and reviewers can't tell which bump
caused a failure.
