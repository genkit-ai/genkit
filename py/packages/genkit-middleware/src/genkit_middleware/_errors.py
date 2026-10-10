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

"""Error checks shared by retry and fallback."""

from genkit.model import StreamingCallbackError


def caused_by_caller_callback(exc: BaseException) -> bool:
    """True when ``exc`` is, or was raised from, the caller's ``on_chunk`` failing.

    Walks ``__cause__`` because a model plugin may re-raise the failure as
    ``GenkitError(status='UNAVAILABLE') from e`` around its stream loop.
    """
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, StreamingCallbackError):
            return True
        current = current.__cause__
    return False
