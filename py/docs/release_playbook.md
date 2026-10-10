# Python Release Playbook

## Release candidates (RC)

Use the **Release Python RC** workflow in GitHub Actions:

1. Go to Actions → Release Python RC
2. Click "Run workflow"
3. Enter target version (e.g., `X.Y.Z`)
4. The workflow infers the next RC (`X.Y.Z-rc.1`, `X.Y.Z-rc.2`, …) and runs the full flow

**What the workflow does:**

1. Creates branch `release/py/X.Y.Z-rc.1` (one branch per RC)
2. Bumps all `pyproject.toml` versions from current to the RC version
3. Commits and pushes to that branch
4. Creates tag `py/vX.Y.Z-rc.1` pointing at the release branch
5. Tag push triggers **Publish Python**, which builds and publishes to PyPI

No PR required.

**If you need to re-run publish** (e.g., it failed): Go to Actions → Publish Python → Run workflow → select the release branch (e.g., `release/py/X.Y.Z-rc.1`) from the branch dropdown → Run.

## Stable release steps

There is one path. `create_release` refuses to run unless every step below was followed.

1. `py/bin/bump_version X.Y.Z` on a branch from main. It bumps every `pyproject.toml` and `uv.lock`.
2. Open a PR to main titled exactly `chore(py): release Python SDK vX.Y.Z`. Its description is the release notes, published word for word.
3. Get it approved and merged.
4. `py/bin/create_release X.Y.Z <PR_NUMBER> --dry-run`, then without `--dry-run`.
5. Approve the publish at <https://github.com/genkit-ai/genkit/actions/workflows/publish_python.yml>. The tag push already started it; don't run it by hand.
6. `pip index versions genkit` shows X.Y.Z.

`create_release` runs every check before it changes anything and prints a fix for each failure:

- `pr-merged`, `pr-base`: the PR is merged into main
- `pr-title`: the title is exactly `chore(py): release Python SDK vX.Y.Z`
- `pr-notes`: the description isn't empty
- `versions`: every `py/packages/*/pyproject.toml` is at X.Y.Z in the merge commit
- `tag-free`: tag `py/vX.Y.Z` doesn't exist yet

It tags the PR's merge commit, not the tip of main, so anything merged later stays out of the release. Exit codes: `0` ok, `1` usage or environment, `2` a check failed and nothing changed.

To fix notes after release, edit the GitHub release: `gh release edit py/vX.Y.Z --notes-file notes.md`. The PR description isn't read again.

## Workflow: `publish_python.yml`

**Triggers:** Tag push (`py/v*`) or manual `workflow_dispatch`. When triggered by tag, it checks out the tag (which points at the release branch). When run manually, **select the release branch** (e.g., `release/py/X.Y.Z-rc.1`) from the "Use workflow from" dropdown — otherwise it will build from main.

**Two jobs:** publish → verify

- **publish**: Builds all packages with `uv build`, then uploads to PyPI via `pypa/gh-action-pypi-publish`. All-or-nothing — if any build fails, the job fails.
- **verify**: `pip install genkit==<version>` + import test.

## Auth

OIDC trusted publishing. No API tokens. PyPI is configured to trust: Owner `firebase`, Repository `genkit`, Workflow `publish_python.yml`, Environment `pypi_github_publishing`. Already set up. Don't rename the workflow file or this breaks.
