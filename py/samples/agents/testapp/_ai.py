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

"""The one Genkit instance every agent in this folder shares.

Each agent file registers itself on this ``ai`` the moment it's imported, so the
Dev UI (running a single file) and the FastAPI server (importing all of them)
both see the same registry. Same idea as the JS testapp's ``genkit.ts``.

Requires GEMINI_API_KEY.
"""

from __future__ import annotations

from typing import Any, cast

from genkit_google_genai import GoogleAI
from genkit_middleware import Middleware

from genkit import GenkitError
from genkit.exp import Genkit
from genkit.exp.agent import AgentResponse

# The capable default; a couple of agents also reach for the lite model below
# for cheap sub-steps (decomposition, safety checks) so the main model isn't
# paying for busywork.
DEFAULT_MODEL = GoogleAI.gemini('gemini-flash-latest')
LITE_MODEL = GoogleAI.gemini('gemini-flash-lite-latest')

# The Middleware plugin lists `ToolApproval`, `Filesystem` and the rest in the
# Dev UI; the agents pass their middleware instances directly in `use=[...]`.
ai = Genkit(plugins=[GoogleAI(), Middleware()], model=DEFAULT_MODEL)


def reply_text(res: AgentResponse[Any]) -> str:
    """The turn's reply text, raising if the turn failed.

    A failed turn returns with ``res.error`` set and ``res.text == ''``. A flow
    or delegate tool that returned that ``''`` would report success, so raise
    instead and let the caller (the Dev UI, or the orchestrator's model) see it.
    """
    if res.error:
        # The wire status is a plain str; GenkitError takes the status literal.
        status = cast(Any, res.error.status)
        raise GenkitError(status=status, message=res.error.message, details=res.error.details)
    return res.text
