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

"""Which model failures Retry and Fallback act on."""

from collections.abc import Collection

from genkit import GenkitError

# Failures where sending the same request again can succeed.
TRANSIENT_STATUSES: tuple[str, ...] = (
    'UNAVAILABLE',
    'DEADLINE_EXCEEDED',
    'RESOURCE_EXHAUSTED',
    'ABORTED',
    'INTERNAL',
)


def status_matches(error: Exception, statuses: Collection[str], *, unclassified: bool) -> bool:
    """Whether ``error`` is one of ``statuses``.

    A ``GenkitError`` matches when its status is in ``statuses``. Any other
    exception carries no status, so ``unclassified`` is the answer for it.
    """
    if isinstance(error, GenkitError):
        return error.status in statuses
    return unclassified
