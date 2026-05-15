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

# pnpm login --registry https://wombat-dressing-room.appspot.com

CURRENT=`pwd`
RELEASE_BRANCH="${RELEASE_BRANCH:-main}"
RELEASE_TAG="${RELEASE_TAG:-next}"

cd genkit-tools/common
pnpm publish --provenance=false --tag $RELEASE_TAG --publish-branch $RELEASE_BRANCH --registry https://wombat-dressing-room.appspot.com
cd $CURRENT

cd genkit-tools/telemetry-server
pnpm publish --provenance=false --tag $RELEASE_TAG --publish-branch $RELEASE_BRANCH --registry https://wombat-dressing-room.appspot.com
cd $CURRENT

cd genkit-tools/cli
pnpm publish --provenance=false --tag $RELEASE_TAG --publish-branch $RELEASE_BRANCH --registry https://wombat-dressing-room.appspot.com
cd $CURRENT

cd js/core
pnpm publish --provenance=false --tag $RELEASE_TAG --publish-branch $RELEASE_BRANCH --registry https://wombat-dressing-room.appspot.com
cd $CURRENT

cd js/ai
pnpm publish --provenance=false --tag $RELEASE_TAG --publish-branch $RELEASE_BRANCH --registry https://wombat-dressing-room.appspot.com
cd $CURRENT

cd js/genkit
pnpm publish --provenance=false --tag $RELEASE_TAG --publish-branch $RELEASE_BRANCH --registry https://wombat-dressing-room.appspot.com
cd $CURRENT

cd js/plugins/chroma
pnpm publish --provenance=false --tag $RELEASE_TAG --publish-branch $RELEASE_BRANCH --registry https://wombat-dressing-room.appspot.com
cd $CURRENT

cd js/plugins/dev-local-vectorstore
pnpm publish --provenance=false --tag $RELEASE_TAG --publish-branch $RELEASE_BRANCH --registry https://wombat-dressing-room.appspot.com
cd $CURRENT

cd js/plugins/firebase
pnpm publish --provenance=false --tag $RELEASE_TAG --publish-branch $RELEASE_BRANCH --registry https://wombat-dressing-room.appspot.com
cd $CURRENT

cd js/plugins/google-cloud
pnpm publish --provenance=false --tag $RELEASE_TAG --publish-branch $RELEASE_BRANCH --registry https://wombat-dressing-room.appspot.com
cd $CURRENT

cd js/plugins/ollama
pnpm publish --provenance=false --tag $RELEASE_TAG --publish-branch $RELEASE_BRANCH --registry https://wombat-dressing-room.appspot.com
cd $CURRENT

cd js/plugins/pinecone
pnpm publish --provenance=false --tag $RELEASE_TAG --publish-branch $RELEASE_BRANCH --registry https://wombat-dressing-room.appspot.com
cd $CURRENT

cd js/plugins/vertexai
pnpm publish --provenance=false --tag $RELEASE_TAG --publish-branch $RELEASE_BRANCH --registry https://wombat-dressing-room.appspot.com
cd $CURRENT

cd  js/plugins/evaluators
pnpm publish --provenance=false --tag $RELEASE_TAG --publish-branch $RELEASE_BRANCH --registry https://wombat-dressing-room.appspot.com
cd $CURRENT

cd  js/plugins/langchain
pnpm publish --provenance=false --tag $RELEASE_TAG --publish-branch $RELEASE_BRANCH --registry https://wombat-dressing-room.appspot.com
cd $CURRENT

cd  js/plugins/checks
pnpm publish --provenance=false --tag $RELEASE_TAG --publish-branch $RELEASE_BRANCH --registry https://wombat-dressing-room.appspot.com
cd $CURRENT

cd  js/plugins/mcp
pnpm publish --provenance=false --tag $RELEASE_TAG --publish-branch $RELEASE_BRANCH --registry https://wombat-dressing-room.appspot.com
cd $CURRENT

cd  js/plugins/express
pnpm publish --provenance=false --tag $RELEASE_TAG --publish-branch $RELEASE_BRANCH --registry https://wombat-dressing-room.appspot.com
cd $CURRENT

cd  js/plugins/next
pnpm publish --provenance=false --tag $RELEASE_TAG --publish-branch $RELEASE_BRANCH --registry https://wombat-dressing-room.appspot.com
cd $CURRENT

cd  js/plugins/cloud-sql-pg
pnpm publish --provenance=false --tag $RELEASE_TAG --publish-branch $RELEASE_BRANCH --registry https://wombat-dressing-room.appspot.com
cd $CURRENT

cd  js/plugins/compat-oai
pnpm publish --provenance=false --tag $RELEASE_TAG --publish-branch $RELEASE_BRANCH --registry https://wombat-dressing-room.appspot.com
cd $CURRENT

cd  js/plugins/google-genai
pnpm publish --provenance=false --tag $RELEASE_TAG --publish-branch $RELEASE_BRANCH --registry https://wombat-dressing-room.appspot.com
cd $CURRENT
