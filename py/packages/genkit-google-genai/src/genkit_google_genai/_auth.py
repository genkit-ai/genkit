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

"""Classify google.auth credential failures.

ADC lookup and token refresh can fail inside any SDK call, not only at
client construction. Every call site maps them through here so the status
stays the same across generate, embed, cache, and Veo paths.
"""

from typing import NoReturn

from google.auth.exceptions import DefaultCredentialsError, GoogleAuthError, RefreshError, TransportError

from genkit.plugin_api import provider_error

GOOGLE_AUTH_ERRORS: tuple[type[GoogleAuthError], ...] = (DefaultCredentialsError, RefreshError, TransportError)


def raise_auth_error(error: GoogleAuthError) -> NoReturn:
    """Raise UNAUTHENTICATED for a missing or rejected credential, UNAVAILABLE for a blip.

    A refresh that could not reach the token endpoint is UNAVAILABLE, so
    Retry tries again and Fallback can switch, the same as the Vertex AI
    Model Garden client. google.auth marks a token-endpoint blip
    ``retryable``; a metadata-server blip on Cloud Run or GKE is a
    ``RefreshError`` raised from a ``TransportError`` with
    ``retryable=False``, so the cause is checked too.

    Args:
        error: The DefaultCredentialsError, RefreshError, or TransportError that was caught.

    Raises:
        GenkitError: UNAVAILABLE for a transient failure, else UNAUTHENTICATED,
            with ``error`` as the cause.
    """
    if isinstance(error, TransportError) or error.retryable or isinstance(error.__cause__, TransportError):
        raise provider_error(error, status='UNAVAILABLE') from error
    raise provider_error(
        error,
        status='UNAUTHENTICATED',
        message='Google Cloud credentials are missing or were rejected',
    ) from error
