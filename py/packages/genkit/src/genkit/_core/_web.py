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

"""Reading a served flow's HTTP request."""

from typing import cast

from genkit._core._action import input_from_json
from genkit._core._error import PublicError


def read_body(body: object) -> object:
    """Return the flow input from a parsed JSON request body.

    ``{"data": x}`` gives ``x``. ``{}`` and ``{"data": null}`` mean the caller
    sent no input, so a flow with a default uses it. Any other body is a 400
    so the caller learns to wrap their input instead of the flow seeing a
    half-understood value.

    Raises:
        PublicError: INVALID_ARGUMENT when the body isn't ``{"data": ...}`` or ``{}``.
    """
    if isinstance(body, dict):
        fields = cast(dict[str, object], body)
        if 'data' in fields:
            return input_from_json(fields['data'])
        if not fields:
            return input_from_json(None)
    raise PublicError('INVALID_ARGUMENT', 'Flow request must be wrapped in {"data": ...}')


def wants_stream(*, accept: str | None, stream: str | None) -> bool:
    """Whether the caller asked for server-sent events.

    Pass the ``Accept`` header and the ``?stream=`` query value. Substring
    match so ``text/event-stream, */*`` still streams.
    """
    return 'text/event-stream' in (accept or '') or stream == 'true'
