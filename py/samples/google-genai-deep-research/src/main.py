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

"""Google GenAI Deep Research — start a background research job.

`uv run src/main.py` only starts the job so it returns immediately. Poll
from Dev UI (or call `deep_research` there) — a finished report takes minutes.
"""

from genkit_google_genai import GoogleAI
from pydantic import BaseModel, Field

from genkit import Genkit

ai = Genkit(plugins=[GoogleAI()])


class ResearchInput(BaseModel):
    """Input for a Deep Research background model."""

    model: str = Field(
        default='googleai/deep-research-preview-04-2026',
        description='Deep Research model for the background job',
    )
    prompt: str = Field(
        default='Summarize recent advances in quantum error correction for a technical audience.',
        description='Research question or topic',
    )


@ai.flow(name='deep_research')
async def deep_research_flow(input: ResearchInput) -> dict[str, str | bool | None]:
    """Start Deep Research with generate_operation(). Poll from Dev UI."""
    operation = await ai.generate_operation(
        model=input.model,
        prompt=input.prompt,
    )
    return {
        'model': input.model,
        'operation_id': operation.id,
        'done': operation.done,
    }


async def main() -> None:
    """Start one Deep Research job — does not wait for the report."""
    try:
        print(await deep_research_flow(ResearchInput()))  # noqa: T201
    except Exception as error:
        print(f'Set GEMINI_API_KEY to a valid value before running this sample directly.\n{error}')  # noqa: T201


if __name__ == '__main__':
    ai.run_main(main())
