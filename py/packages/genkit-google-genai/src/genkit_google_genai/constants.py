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

"""Constants used by the Vertex AI plugin.

This module defines constants used throughout the Vertex AI plugin,
including environment variable names and configuration values.
"""

from typing import TypeGuard

GCLOUD_PROJECT = 'GCLOUD_PROJECT'
GOOGLE_CLOUD_PROJECT = 'GOOGLE_CLOUD_PROJECT'
GOOGLE_CLOUD_LOCATION = 'GOOGLE_CLOUD_LOCATION'
GCLOUD_LOCATION = 'GCLOUD_LOCATION'
DEFAULT_REGION = 'us-central1'

MULTI_REGIONAL_LOCATIONS = ('us', 'eu')

GLOBAL_LOCATION = 'global'


def is_multi_regional_location(location: str | None) -> TypeGuard[str]:
    """Whether the location is a Vertex AI multi-region ('us' or 'eu')."""
    return location in MULTI_REGIONAL_LOCATIONS


def vertex_api_host(location: str) -> str:
    """Vertex AI API host for a location.

    Multi-regions are served from dedicated hosts
    (``aiplatform.{location}.rep.googleapis.com``), 'global' from the bare
    host, and regions from the ``{location}-aiplatform.googleapis.com``
    pattern.
    """
    if location == GLOBAL_LOCATION:
        return 'aiplatform.googleapis.com'
    if is_multi_regional_location(location):
        return f'aiplatform.{location}.rep.googleapis.com'
    return f'{location}-aiplatform.googleapis.com'


def multi_regional_base_url(location: str) -> str:
    """Base URL for a Vertex AI multi-region endpoint.

    No trailing slash, so the google-genai SDK's '.googleapis.com' suffix
    checks recognize this as a Google host.
    """
    return f'https://{vertex_api_host(location)}'
