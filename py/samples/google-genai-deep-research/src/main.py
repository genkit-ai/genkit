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

"""Google GenAI Deep Research - start a background research job and poll to completion."""

import asyncio
import time
from typing import Any, Literal

from genkit_google_genai import GoogleAI
from pydantic import BaseModel, Field

from genkit import Genkit
from genkit.model import Operation

ai = Genkit(plugins=[GoogleAI()])


class ResearchInput(BaseModel):
    """Input for a Deep Research background model."""

    model: Literal[
        'googleai/deep-research-preview-04-2026',
        'googleai/deep-research-max-preview-04-2026',
        'googleai/deep-research-pro-preview-12-2025',
    ] = Field(
        default='googleai/deep-research-preview-04-2026',
        description='Deep Research model for the background job',
    )
    prompt: str = Field(
        default='Summarize recent advances in quantum error correction for a technical audience.',
        description='Research question or topic',
    )


def _response_text(response: Any) -> str | None:
    """Return text from a completed model response."""

    text = getattr(response, 'text', None)
    if text:
        return text
    return None


async def _poll_operation(operation: Operation, *, timeout_seconds: float = 600) -> Operation:
    """Poll a background operation until it completes or times out."""

    started_at = time.monotonic()
    while not operation.done:
        if time.monotonic() - started_at > timeout_seconds:
            raise TimeoutError('Timed out waiting for Deep Research output')
        await asyncio.sleep(5)
        operation = await ai.check_operation(operation)
    return operation


@ai.flow(name='deep_research')
async def deep_research_flow(input: ResearchInput) -> dict[str, str | None]:
    """Run Deep Research with generate_operation() and poll until the report is ready."""

    operation = await ai.generate_operation(
        model=input.model,
        prompt=input.prompt,
    )
    operation = await _poll_operation(operation)

    report = _response_text(operation.output) if operation.output else None

    return {
        'model': input.model,
        'operation_id': operation.id,
        'report': report,
    }


async def main() -> None:
    """Run one Deep Research job."""
    try:
        print(await deep_research_flow(ResearchInput()))  # noqa: T201
    except Exception as error:
        print(f'Set GEMINI_API_KEY to a valid value before running this sample directly.\n{error}')  # noqa: T201


if __name__ == '__main__':
    ai.run_main(main())
