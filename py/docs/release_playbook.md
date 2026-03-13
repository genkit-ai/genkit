# Python Release Playbook

## Release steps

1. `./bin/bump_version 0.7.1` — bump all `pyproject.toml` files
2. `./bin/release_check` — preflight checks
3. Open a PR to main with release notes in the description
4. Merge the PR
5. `./bin/create_release 0.7.1` — tags `py/v0.7.1`, pushes tag, creates GitHub release
6. Approve the publish at <https://github.com/firebase/genkit/actions>

## Workflow: `publish_python.yml`

Three jobs: **build** → **publish** → **verify**.

- **build**: `uv build` all packages in order (starting with core genkit, then plugins, then more plugins that depend on previous plugins), upload as artifact. All-or-nothing — if one fails to build, nothing gets published.
- **publish**: `pypa/gh-action-pypi-publish@release/v1` with OIDC trusted publishing. `skip-existing: true` so re-runs are safe.
- **verify**: `pip install genkit==<version>` + import test.

## Auth

OIDC trusted publishing. No API tokens. PyPI is configured to trust:

| Field | Value |
|---|---|
| Owner | `firebase` |
| Repository | `genkit` |
| Workflow | `publish_python.yml` |
| Environment | `pypi_github_publishing` |

Already set up from v0.5.0. Don't rename the workflow file or this breaks.

## Failures

| What | Recovery |
|---|---|
| Build fails | Fix, re-run |
| Publish fails midway | Re-run publish job (`skip-existing` skips uploaded packages) |
| Bad package on PyPI | Can't overwrite. Yank it, do a patch release |


## If something goes wrong

```bash
# Delete a bad tag/release and redo
git tag -d py/v0.7.1 && git push origin :refs/tags/py/v0.7.1
```

PyPI doesn't allow re-uploading the same version. Bump to a patch release instead.
