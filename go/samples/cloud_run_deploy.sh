#!/bin/sh
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


# This script deploys a sample to Cloud Run.
# Run it from this directory (go/samples).
# For example
#
#    cloud_run_deploy.sh basic
#
# will deploy the basic sample as a Cloud Run service
# named genkit-basic.

sample="$1"

if [ -z "$sample" ]; then
  echo >&2 "usage: $0 SAMPLE"
  exit 1
fi

if [ ! -d "$sample" ]; then
  echo >&2 "$sample is not a subdirectory of the samples directory."
  exit 1
fi 

if [ -z "$GCLOUD_PROJECT" ]; then
  echo >&2 "Set GCLOUD_PROJECT to your project name."
  exit 1
fi

repo_root=$(git rev-parse --show-toplevel)
if [ -z "repo_root" ]; then
  echo >&2 "Could not determine git repo root directory; are you in a git repo?"
  exit 1
fi

location=${GCLOUD_LOCATION:-us-central1}

image=${location}-docker.pkg.dev/$GCLOUD_PROJECT/cloud-run-source-deploy/genkit-${sample}

gcloud --project "$GCLOUD_PROJECT" builds submit \
  --pack image=${image},env=GOOGLE_BUILDABLE=./samples/${sample} \
  "${repo_root}/go"
