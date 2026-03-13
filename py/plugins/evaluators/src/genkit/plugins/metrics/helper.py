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


"""Helper functions for metrics processing."""

# Run tests for all supported Python versions using tox

from typing import Any

import aiofiles
from dotpromptz import Dotprompt
from dotpromptz.typing import DataArgument, PromptFunction

dp = Dotprompt()


async def load_prompt_file(path: str) -> PromptFunction:
    """Load a prompt file from the given path."""
    # Use aiofiles for async file I/O to avoid blocking the event loop
    async with aiofiles.open(path, encoding='utf-8') as f:
        content = await f.read()
        result = await dp.compile(content)

    return result


async def render_text(prompt: PromptFunction, input_: dict[str, Any]) -> str:
    """Render text from a prompt function using provided input."""
    rendered = await prompt(
        data=DataArgument[dict[str, Any]](
            input=input_,
        )
    )
    result = []
    for message in rendered.messages:
        result.append(''.join(e.text for e in message.content if hasattr(e, 'text') and e.text))  # type: ignore[arg-type]

    return ''.join(result)
