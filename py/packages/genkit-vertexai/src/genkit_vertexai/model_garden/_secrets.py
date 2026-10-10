# Copyright 2026 Google LLC
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

"""Model Garden has no per-request API key."""

from typing import Any

from genkit import GenkitError


def reject_secrets_api_key(context: dict[str, Any]) -> None:
    """Raise when ``context.secrets`` carries an API key.

    A caller passing a tenant's key expects the tenant to pay. Model Garden
    bills the app's Google Cloud project no matter what, so running the call
    anyway would charge the wrong account without anyone noticing.
    """
    secrets = context.get('secrets')
    if isinstance(secrets, dict) and (secrets.get('api_key') is not None or secrets.get('apiKey') is not None):
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=(
                'modelgarden authenticates with Google Cloud credentials; a per-request api_key in '
                "context.secrets isn't supported"
            ),
        )
