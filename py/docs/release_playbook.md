# Python Release Playbook

## Release candidates (RC)

Use the **Release Python RC** workflow in GitHub Actions:

1. Go to Actions → Release Python RC
2. Click "Run workflow"
3. Enter target version (e.g., `0.5.2`)
4. The workflow infers the next RC (`0.5.2-rc.1`, `0.5.2-rc.2`, …) and runs the full flow

**What the workflow does:**

1. Creates branch `release/py/0.5.2-rc.1` (one branch per RC)
2. Bumps all `pyproject.toml` versions from current (e.g. 0.5.1) to the RC version
3. Commits and pushes to that branch
4. Creates tag `py/v0.5.2-rc.1` pointing at the release branch
5. Tag push triggers **Publish Python**, which builds and publishes to PyPI

No PR required.

**If you need to re-run publish** (e.g. it failed): Go to Actions → Publish Python → Run workflow → select the release branch (e.g. `release/py/0.5.2-rc.1`) from the branch dropdown → Run.

## Stable release steps

1. `./bin/bump_version 0.7.1` — bump all `pyproject.toml` files
2. `./bin/release_check` — preflight checks
3. Open a PR to main with release notes in the description
4. Merge the PR
5. `./bin/create_release 0.7.1` — tags `py/v0.7.1`, pushes tag, creates GitHub release
6. Approve the publish at <https://github.com/genkit-ai/genkit/actions>

## Workflow: `publish_python.yml`

**Triggers:** Tag push (`py/v*`) or manual `workflow_dispatch`. When triggered by tag, it checks out the tag (which points at the release branch). When run manually, **select the release branch** (e.g. `release/py/0.5.2-rc.1`) from the "Use workflow from" dropdown — otherwise it will build from main.

**Two jobs:** publish → verify

- **publish**: Builds all packages with `uv build`, then uploads to PyPI via `pypa/gh-action-pypi-publish`. All-or-nothing — if any build fails, the job fails.
- **verify**: `pip install genkit==<version>` + import test.

## Auth

OIDC trusted publishing. No API tokens. PyPI is configured to trust: Owner `firebase`, Repository `genkit`, Workflow `publish_python.yml`, Environment `pypi_github_publishing`. Already set up from v0.5.0. Don't rename the workflow file or this breaks.
