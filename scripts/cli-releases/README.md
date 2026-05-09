# CLI Release Scripts

This directory contains scripts for managing Genkit CLI releases, including promotion from GitHub artifacts to Google Cloud Storage (GCS).

## Overview

The release management process consists of two scripts:
1. `promote_cli_gcs.sh` - Downloads binaries from GitHub and uploads to GCS
2. `update_cli_metadata.sh` - Updates metadata JSON files with version information

## Prerequisites

- Google Cloud SDK (`gcloud` and `gsutil`) installed and authenticated
- GitHub CLI (`gh`) installed and authenticated (for artifact downloads)
- Appropriate permissions on the GCS bucket

## Usage

### Promoting from a GitHub Actions Run

```bash
# Promote binaries from a specific GitHub Actions run
./scripts/cli-releases/promote_cli_gcs.sh \
  --github-run-id=123456789 \
  --channel=next \
  --version=1.15.5

# Update metadata
./scripts/cli-releases/update_cli_metadata.sh \
  --channel=next \
  --version=1.15.5
```

### Promoting from a GitHub Release

```bash
# Promote binaries from a GitHub release tag
./scripts/cli-releases/promote_cli_gcs.sh \
  --github-tag=v1.15.5 \
  --channel=prod \
  --version=1.15.5

# Update metadata
./scripts/cli-releases/update_cli_metadata.sh \
  --channel=prod \
  --version=1.15.5
```

### Dry Run Mode

Both scripts support a `--dry-run` flag to see what would be done without actually doing it:

```bash
./scripts/cli-releases/promote_cli_gcs.sh \
  --github-run-id=123456789 \
  --channel=next \
  --version=1.15.5 \
  --dry-run
```

## Script Options

### promote_cli_gcs.sh

| Option | Description | Required |
|--------|-------------|----------|
| `--github-run-id=ID` | GitHub Actions run ID to download artifacts from | One of github-run-id or github-tag |
| `--github-tag=TAG` | GitHub release tag to download artifacts from | One of github-run-id or github-tag |
| `--channel=CHANNEL` | Target channel (prod/next) | No (default: next) |
| `--version=VERSION` | Version string for GCS paths | Yes |
| `--bucket=BUCKET` | GCS bucket name | No (default: genkit-cli-binaries) |
| `--dry-run` | Show what would be done without doing it | No |

### update_cli_metadata.sh

| Option | Description | Required |
|--------|-------------|----------|
| `--channel=CHANNEL` | Target channel (prod/next) | No (default: next) |
| `--version=VERSION` | Version to mark as latest | Yes |
| `--bucket=BUCKET` | GCS bucket name | No (default: genkit-cli-binaries) |
| `--dry-run` | Show what would be done without doing it | No |

## GCS Bucket Structure

The scripts create the following structure in GCS:

```
gs://genkit-cli-binaries/
├── prod/
│   └── bin/
│       ├── linux-x64/
│       │   ├── latest
│       │   └── v1.15.5/
│       │       └── genkit
│       ├── linux-arm64/
│       │   ├── latest
│       │   └── v1.15.5/
│       │       └── genkit
│       ├── darwin-x64/
│       │   ├── latest
│       │   └── v1.15.5/
│       │       └── genkit
│       ├── darwin-arm64/
│       │   ├── latest
│       │   └── v1.15.5/
│       │       └── genkit
│       └── win32-x64/
│           ├── latest.exe
│           └── v1.15.5/
│               └── genkit.exe
├── next/
│   └── bin/
│       └── [same structure as prod]
├── metadata-prod.json
└── metadata-next.json
```

## Metadata Format

The metadata JSON files contain information about the latest version:

```json
{
  "channel": "prod",
  "latestVersion": "1.15.5",
  "lastUpdated": "2025-01-15T10:00:00Z",
  "platforms": {
    "linux-x64": {
      "url": "https://cli.genkit.dev/bin/linux-x64/latest",
      "version": "1.15.5",
      "versionedUrl": "https://cli.genkit.dev/bin/linux-x64/v1.15.5/genkit"
    },
    ...
  }
}
```

## Integration with Cloud Build

These scripts are designed to be called from Cloud Build or other CI/CD systems. Example Cloud Build step:

```yaml
steps:
  - name: 'gcr.io/cloud-builders/gcloud'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        ./scripts/cli-releases/promote_cli_gcs.sh \
          --github-run-id=${_GITHUB_RUN_ID} \
          --channel=${_CHANNEL} \
          --version=${_VERSION}
        
        ./scripts/cli-releases/update_cli_metadata.sh \
          --channel=${_CHANNEL} \
          --version=${_VERSION}
```

## Notes

- Binary naming conventions match those from `build-cli-binaries.yml`
- The domain `cli.genkit.dev` should be configured to serve from the GCS bucket
- Cache headers are set appropriately (1 hour for versioned files, 5 minutes for latest)