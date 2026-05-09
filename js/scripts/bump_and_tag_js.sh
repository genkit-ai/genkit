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

# this script lets you bump and tag all JS packages (including plugins)
# js/scripts/bump_and_tag_cli.sh prerelease rc

set -e
set -x

RELEASE_TYPE=$1
PREID="${2:-rc}"

if [[ -z "$RELEASE_TYPE" ]]
then
  echo "release type (first arg) not set"
  exit 1
fi

# import bump_version script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
. $SCRIPT_DIR/bump_version.sh

COMMIT_MSG="chore: JS version bump$NEWLINE$NEWLINE"

# core packages
bump_version js/core @genkit-ai/core core-v
bump_version js/ai @genkit-ai/ai ai-v
bump_version js/genkit genkit

# plugins
bump_version js/plugins/chroma genkitx-chromadb
bump_version js/plugins/dev-local-vectorstore @genkit-ai/dev-local-vectorstore dev-local-vectorstore-v
bump_version js/plugins/evaluators @genkit-ai/evaluator evaluator-v
bump_version js/plugins/firebase @genkit-ai/firebase firebase-v
bump_version js/plugins/google-cloud @genkit-ai/google-cloud google-cloud-v
bump_version js/plugins/langchain genkitx-langchain
bump_version js/plugins/next @genkit-ai/next next-v
bump_version js/plugins/ollama genkitx-ollama
bump_version js/plugins/pinecone genkitx-pinecone
bump_version js/plugins/vertexai @genkit-ai/vertexai vertexai-v
bump_version js/plugins/checks @genkit-ai/checks checks-v
bump_version js/plugins/mcp @genkit-ai/mcp mcp-v
bump_version js/plugins/express @genkit-ai/express express-v
bump_version js/plugins/cloud-sql-pg genkitx-cloud-sql-pg
bump_version js/plugins/compat-oai @genkit-ai/compat-oai compat-oai-v
bump_version js/plugins/google-genai @genkit-ai/google-genai google-genai-v

echo TAGS "${TAGS[*]}"

commit_and_tag
