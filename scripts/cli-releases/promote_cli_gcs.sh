#!/bin/bash
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

set -e
set -x

# Script to promote CLI binaries from GitHub artifacts to GCS
# Usage: promote_cli_gcs.sh [options]
#
# Options:
#   --github-run-id=ID    GitHub Actions run ID to download artifacts from
#   --github-tag=TAG      GitHub release tag to download artifacts from  
#   --channel=CHANNEL     Target channel (prod/next)
#   --version=VERSION     Version string for GCS paths
#   --bucket=BUCKET       GCS bucket name (default: genkit-cli-binaries)
#   --dry-run            Show what would be done without doing it

# Default values
GITHUB_RUN_ID=""
GITHUB_TAG=""
CHANNEL="next"
VERSION=""
BUCKET="genkit-assets-cli"
DRY_RUN=false

# Parse command line arguments
for arg in "$@"; do
  case $arg in
    --github-run-id=*)
      GITHUB_RUN_ID="${arg#*=}"
      shift
      ;;
    --github-tag=*)
      GITHUB_TAG="${arg#*=}"
      shift
      ;;
    --channel=*)
      CHANNEL="${arg#*=}"
      shift
      ;;
    --version=*)
      VERSION="${arg#*=}"
      shift
      ;;
    --bucket=*)
      BUCKET="${arg#*=}"
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    *)
      echo "Unknown option: $arg"
      exit 1
      ;;
  esac
done

# Validate inputs
if [[ -z "$GITHUB_RUN_ID" && -z "$GITHUB_TAG" ]]; then
  echo "Error: Either --github-run-id or --github-tag must be specified"
  exit 1
fi

if [[ -z "$VERSION" ]]; then
  echo "Error: --version must be specified"
  exit 1
fi

if [[ "$CHANNEL" != "prod" && "$CHANNEL" != "next" ]]; then
  echo "Error: --channel must be either 'prod' or 'next'"
  exit 1
fi

# Platform list matching build-cli-binaries.yml
PLATFORMS=(
  "linux-x64"
  "linux-arm64"
  "darwin-x64"
  "darwin-arm64"
  "win32-x64"
)

echo "=== CLI Binary Promotion to GCS ==="
echo "Channel: $CHANNEL"
echo "Version: $VERSION"
echo "Bucket: $BUCKET"
echo "Dry run: $DRY_RUN"
echo ""

# Create temporary directory for downloads
TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT

# Validate that TMP_DIR exists and is writable
if [[ -z "$TMP_DIR" || ! -d "$TMP_DIR" || ! -w "$TMP_DIR" ]]; then
  echo "Error: Failed to create a writable temporary directory."
  exit 1
fi

cd "$TMP_DIR"

# Download artifacts from GitHub
if [[ -n "$GITHUB_RUN_ID" ]]; then
  echo "Downloading artifacts from GitHub run ID: $GITHUB_RUN_ID"
  
  # Check if gh CLI is available
  if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI (gh) is not installed"
    echo "Please install it from: https://cli.github.com/"
    exit 1
  fi
  
  # Download all artifacts for the run
  for platform in "${PLATFORMS[@]}"; do
    echo "Downloading artifact: genkit-$platform"
    if [[ "$DRY_RUN" == "false" ]]; then
      # Use gh CLI to download the artifact
      if gh run download "$GITHUB_RUN_ID" -n "genkit-$platform" -R firebase/genkit; then
        continue
        # The artifact is downloaded as a directory, move the binary to the current directory
        # if [[ -f "genkit-$platform" ]]; then
        #   mv "genkit-$platform" .
        # elif [[ -f "genkit-$platform.exe" ]]; then
        #   mv "genkit-$platform.exe" .
        # fi
      else
        echo "Error: Failed to download genkit-$platform"
        exit 1
      fi
    else
      echo "[DRY RUN] Would download: genkit-$platform"
    fi
  done
elif [[ -n "$GITHUB_TAG" ]]; then
  echo "Downloading artifacts from GitHub release tag: $GITHUB_TAG"
  
  # Download release assets
  for platform in "${PLATFORMS[@]}"; do
    # Determine file extension
    if [[ "$platform" == "win32-x64" ]]; then
      ext=".exe"
    else
      ext=""
    fi
    
    echo "Downloading release asset: genkit-$platform$ext"
    if [[ "$DRY_RUN" == "false" ]]; then
      gh release download "$GITHUB_TAG" -p "genkit-$platform$ext" -R firebase/genkit || {
        echo "Error: Failed to download genkit-$platform$ext"
        exit 1
      }
    else
      echo "[DRY RUN] Would download: genkit-$platform$ext"
    fi
  done
fi

# Upload to GCS
echo ""
echo "Uploading binaries to GCS..."

for platform in "${PLATFORMS[@]}"; do
  # Determine file extension and names
  if [[ "$platform" == "win32-x64" ]]; then
    ext=".exe"
    binary_name="genkit.exe"
    latest_name="latest.exe"
  else
    ext=""
    binary_name="genkit"
    latest_name="latest"
  fi
  
  source_file="genkit-$platform$ext"
  
  # Check if file exists
  if [[ ! -f "$source_file" ]]; then
    echo "Error: $source_file not found, skipping..."
    exit 1
  fi
  
  versioned_path="gs://$BUCKET/$CHANNEL/$platform/v$VERSION/$binary_name"

  # Check if versioned file already exists.
  if [[ "$DRY_RUN" == "false" ]]; then
    if gcloud storage objects list --stat --fetch-encrypted-object-hashes "$versioned_path" >/dev/null 2>&1; then
      echo "Version $VERSION for $platform already exists in GCS, skipping upload."
      continue
    fi
  fi

  # Upload versioned binary
  echo "Uploading $source_file to $versioned_path"
  if [[ "$DRY_RUN" == "false" ]]; then
    gcloud storage cp --cache-control="public, max-age=3600" "$source_file" "$versioned_path"
  else
    echo "[DRY RUN] Would upload to: $versioned_path"
  fi
  
  # Upload/copy as latest
  latest_path="gs://$BUCKET/$CHANNEL/$platform/$latest_name"
  echo "Copying to $latest_path"
  if [[ "$DRY_RUN" == "false" ]]; then
    gcloud storage cp --cache-control="public, max-age=300" "$source_file" "$latest_path"
  else
    echo "[DRY RUN] Would copy to: $latest_path (overwriting)"
  fi
  
  echo ""
done

echo "=== Promotion complete ==="
echo "Run update_cli_metadata.sh to update the metadata files."
