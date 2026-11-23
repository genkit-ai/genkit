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

# Check that the generated Go code is up to date with the committed
# genkit-tools/genkit-schema.json (which builder.yml verifies against the Zod
# schemas in genkit-tools). Must run from the repo root.

set -euo pipefail

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
