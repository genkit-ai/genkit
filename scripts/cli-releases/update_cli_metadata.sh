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

# Script to update CLI metadata in GCS
# Usage: update_cli_metadata.sh [options]
#
# Options:
#   --channel=CHANNEL     Target channel (prod/next)
#   --version=VERSION     Version to mark as latest
#   --bucket=BUCKET       GCS bucket name (default: genkit-cli-binaries)
#   --dry-run            Show what would be done without doing it

# Default values
CHANNEL="next"
VERSION=""
BUCKET="genkit-assets-cli"
DRY_RUN=false

# Parse command line arguments
for arg in "$@"; do
  case $arg in
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
if [[ -z "$VERSION" ]]; then
  echo "Error: --version must be specified"
  exit 1
fi

if [[ "$CHANNEL" != "prod" && "$CHANNEL" != "next" ]]; then
  echo "Error: --channel must be either 'prod' or 'next'"
  exit 1
fi

# Platform list
PLATFORMS=(
  "linux-x64"
  "linux-arm64"
  "darwin-x64"
  "darwin-arm64"
  "win32-x64"
)

echo "=== CLI Metadata Update ==="
echo "Channel: $CHANNEL"
echo "Version: $VERSION"
echo "Bucket: $BUCKET"
echo "Dry run: $DRY_RUN"
echo ""

# Create temporary directory
TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT

# Validate that TMP_DIR exists and is writable
if [[ -z "$TMP_DIR" || ! -d "$TMP_DIR" || ! -w "$TMP_DIR" ]]; then
  echo "Error: Failed to create a writable temporary directory."
  exit 1
fi

cd "$TMP_DIR"

# Get current timestamp
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Create metadata JSON
if [[ "$CHANNEL" == "prod" ]]; then
  METADATA_FILE="latest.json"
else
  METADATA_FILE="$CHANNEL.json"
fi
echo "Generating metadata file: $METADATA_FILE"

# Try to download existing metadata to preserve release history
EXISTING_METADATA=""
if [[ "$DRY_RUN" == "false" ]]; then
  echo "Checking for existing metadata..."
  if gsutil cp "gs://$BUCKET/$METADATA_FILE" "$METADATA_FILE.existing" 2>/dev/null; then
    echo "Found existing metadata, will preserve release history"
    EXISTING_METADATA=$(cat "$METADATA_FILE.existing")
  else
    echo "No existing metadata found, creating new file"
  fi
fi

# Start building the JSON
cat > "$METADATA_FILE" << EOF
{
  "channel": "$CHANNEL",
  "latestVersion": "$VERSION",
  "lastUpdated": "$TIMESTAMP",
  "platforms": {
EOF

# Add platform entries
first=true
for platform in "${PLATFORMS[@]}"; do
  if [[ "$first" == "true" ]]; then
    first=false
  else
    echo "," >> "$METADATA_FILE"
  fi
  
  # Determine binary names
  if [[ "$platform" == "win32-x64" ]]; then
    binary_name="latest.exe"
    versioned_binary_name="genkit.exe"
  else
    binary_name="latest"
    versioned_binary_name="genkit"
  fi
  
  cat >> "$METADATA_FILE" << EOF
    "$platform": {
      "url": "https://storage.googleapis.com/genkit-assets-cli/$CHANNEL/$platform/$binary_name",
      "version": "$VERSION",
      "versionedUrl": "https://storage.googleapis.com/genkit-assets-cli/$CHANNEL/$platform/v$VERSION/$versioned_binary_name"
    }
EOF
done

# Close the JSON
cat >> "$METADATA_FILE" << EOF

  }
}
EOF

# Pretty print the JSON
if command -v jq &> /dev/null; then
  jq . "$METADATA_FILE" > "$METADATA_FILE.tmp" && mv "$METADATA_FILE.tmp" "$METADATA_FILE"
fi

# Show the metadata
echo ""
echo "Generated metadata:"
cat "$METADATA_FILE"
echo ""

# Upload to GCS
METADATA_PATH="gs://$BUCKET/$METADATA_FILE"
echo "Uploading metadata to: $METADATA_PATH"

if [[ "$DRY_RUN" == "false" ]]; then
  # Upload with appropriate cache headers
  gsutil -h "Cache-Control:public, max-age=60" \
         -h "Content-Type:application/json" \
         cp "$METADATA_FILE" "$METADATA_PATH"
  
 
  echo ""
  echo "Metadata uploaded successfully!"
else
  echo "[DRY RUN] Would upload to: $METADATA_PATH"
fi

echo ""
echo "=== Metadata update complete ==="
echo ""
echo "Metadata is now available at:"
echo "  https://storage.googleapis.com/genkit-assets-cli/$METADATA_FILE"