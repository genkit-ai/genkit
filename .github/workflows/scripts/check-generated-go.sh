#!/bin/bash
# Copyright 2024 Google LLC
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

# Check that the committed JSON schema and the generated Go code are up to
# date. Must run from the repo root, after the workflow's
# `npm --prefix genkit-tools run export:schemas` step.

set -euo pipefail

# export:schemas rewrites genkit-schema.json from the Zod schemas in
# genkit-tools. If it left the file dirty, the committed schema does not match
# its Zod sources and everything generated from it would silently drift, so
# fail with that remedy before checking the Go code.
if ! git diff --quiet -- genkit-tools/genkit-schema.json; then
    echo "::error::genkit-tools/genkit-schema.json does not match the Zod schemas it is exported from."
    echo "::error::Update the schemas under genkit-tools, then run 'npm --prefix genkit-tools run export:schemas' and commit the result."
    git --no-pager diff -- genkit-tools/genkit-schema.json
    exit 1
fi

cd go/core
if ! gen_output=$(go run ../internal/cmd/jsonschemagen -outdir .. -config schemas.config ../../genkit-tools/genkit-schema.json ai 2>&1); then
    echo "$gen_output"
    echo "::error::jsonschemagen failed."
    exit 1
fi
echo "$gen_output"

# Check every file the generator reported writing, not a hardcoded subset.
gen_files=$(sed -n 's/^jsonschemagen: wrote //p' <<<"$gen_output")
if [ -z "$gen_files" ]; then
    echo "::error::Could not determine which files jsonschemagen wrote; has its output format changed?"
    exit 1
fi

out_of_date=0
for f in $gen_files; do
    if ! git diff --quiet -- "$f"; then
        echo "::error::Generated $f is out of date."
        out_of_date=1
    fi
done

if [ "$out_of_date" -ne 0 ]; then
    echo "::error::Please run the following and commit the result:"
    echo "::error::cd go/core && go run ../internal/cmd/jsonschemagen -outdir .. -config schemas.config ../../genkit-tools/genkit-schema.json ai"
    git --no-pager diff -- $gen_files
    exit 1
fi
