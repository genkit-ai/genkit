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


# git clone git@github.com:firebase/genkit.git
# cd genkit
# pnpm i
# pnpm build
# pnpm test:all
# Run from root: scripts/release_main.sh
#
# Set RELEASE_SCOPE to control what gets published:
#   all (default) - publish both the JS SDK/plugins and the CLI tooling
#   js            - publish only the JS SDK/plugins
#   cli           - publish only the CLI tooling

# pnpm login --registry https://wombat-dressing-room.appspot.com

set -euo pipefail

CURRENT=`pwd`
RELEASE_BRANCH="${RELEASE_BRANCH:-main}"
RELEASE_TAG="${RELEASE_TAG:-next}"
RELEASE_SCOPE="${RELEASE_SCOPE:-all}"
REGISTRY="https://wombat-dressing-room.appspot.com"

# JS SDK and plugin packages to publish, relative to the repo root.
# Add new packages here.
JS_PACKAGES=(
  js/core
  js/ai
  js/genkit
  js/plugins/chroma
  js/plugins/dev-local-vectorstore
  js/plugins/firebase
  js/plugins/google-cloud
  js/plugins/ollama
  js/plugins/pinecone
  js/plugins/vertexai
  js/plugins/evaluators
  js/plugins/langchain
  js/plugins/checks
  js/plugins/mcp
  js/plugins/express
  js/plugins/next
  js/plugins/cloud-sql-pg
  js/plugins/compat-oai
  js/plugins/google-genai
  js/plugins/anthropic
  js/plugins/fastify
  js/plugins/fetch
  js/plugins/middleware
  js/plugins/vercel-ai
  js/plugins/a2ui
)

# CLI tooling packages to publish, relative to the repo root.
# Add new packages here.
CLI_PACKAGES=(
  genkit-tools/common
  genkit-tools/telemetry-server
  genkit-tools/cli
)

publish_packages() {
  local packages=("$@")
  for package in "${packages[@]}"; do
    echo "Publishing $package..."
    cd "$CURRENT/$package"
    pnpm publish \
      --provenance=false \
      --tag "$RELEASE_TAG" \
      --publish-branch "$RELEASE_BRANCH" \
      --registry "$REGISTRY"
    cd "$CURRENT"
  done
}

case "$RELEASE_SCOPE" in
  all)
    publish_packages "${JS_PACKAGES[@]}"
    publish_packages "${CLI_PACKAGES[@]}"
    ;;
  js)
    publish_packages "${JS_PACKAGES[@]}"
    ;;
  cli)
    publish_packages "${CLI_PACKAGES[@]}"
    ;;
  *)
    echo "Unknown RELEASE_SCOPE: '$RELEASE_SCOPE' (expected 'all', 'js', or 'cli')" >&2
    exit 1
    ;;
esac
