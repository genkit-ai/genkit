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


# This script makes a request to a sample application deployed
# to cloud run.
# Run it from this directory.
#
# For example, after deploying the basic example (see
# cloud_run_deploy.sh):
#
#    cloud_run_request.sh basic jokesFlow '{"topic": "airplane food"}'
#
# will request the genkit-basic Cloud Run service to invoke the jokesFlow flow
# with the third argument as input.

if [ $# -ne 3 ]; then 
  echo >&2 "usage: $0 SAMPLE FLOW INPUT"
  exit 1
fi

sample=$1
flow=$2
input=$3

if [ ! -d "$sample" ]; then
  echo >&2 "$sample is not a subdirectory of the samples directory."
  exit 1
fi 

if [ -z "$GCLOUD_PROJECT" ]; then
  echo >&2 "Set GCLOUD_PROJECT to your project name."
  exit 1
fi

token=$(gcloud auth print-identity-token)
if [ -z "$token" ]; then
  echo >&2 "could not obtain identity token; have you logged in with 'gcloud auth'?"
  exit 1
fi

url=$(gcloud run services describe genkit-${sample} --format 'value(status.url)')
if [ -z "$url}" ]; then
  echo >&2 "could not get URL of Cloud Run service genkit-${sample}; are you sure it is deployed?"
  exit 1
fi

curl -H "Content-Type: application/json" \
  -H "Authorization: Bearer $token" \
  -d  "$input" \
  "$url/$flow"
