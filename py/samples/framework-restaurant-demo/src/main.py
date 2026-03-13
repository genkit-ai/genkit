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

"""Entry point for the menu AI sample - Restaurant menu analysis with AI.

This sample demonstrates a multi-file Genkit application with prompts, flows,
and tools organized into separate modules, simulating a restaurant menu
analysis system.

See README.md for testing instructions.

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Multi-file App      │ Code split across multiple files. Each file        │
    │                     │ handles one part (prompts, flows, tools).          │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Module Organization │ Separate files for different concerns.             │
    │                     │ case_01/prompts.py, case_02/flows.py, etc.         │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Flow Registration   │ Importing a module registers its flows.            │
    │                     │ Just import it and Genkit knows about it.          │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Menu Analysis       │ AI reads menus and answers questions.              │
    │                     │ "What vegetarian options are there?"               │
    └─────────────────────┴────────────────────────────────────────────────────┘

Key Features
============
| Feature Description                     | Example Function / Code Snippet     |
|-----------------------------------------|-------------------------------------|
| Multi-file Flow Organization            | `case_01`, `case_02`, etc. imports  |
| Prompt Management                       | `prompts` module imports            |
| Tool Integration                        | `tools` module imports              |
"""

# Import all of the example prompts and flows to ensure they are registered
import asyncio

from samples.shared.logging import setup_sample

setup_sample()

# Import case modules to register flows and prompts with the ai instance
from case_01 import prompts as case_01_prompts  # noqa: F401
from case_02 import (
    flows as case_02_flows,  # noqa: F401
    prompts as case_02_prompts,  # noqa: F401
    tools as case_02_tools,  # noqa: F401
)
from case_03 import (
    flows as case_03_flows,  # noqa: F401
    prompts as case_03_prompts,  # noqa: F401
)
from case_04 import (
    flows as case_04_flows,  # noqa: F401
    prompts as case_04_prompts,  # noqa: F401
)
from case_05 import (
    flows as case_05_flows,  # noqa: F401
    prompts as case_05_prompts,  # noqa: F401
)
from menu_ai import ai


async def main() -> None:
    """Keep alive for Dev UI."""
    # Keep the process alive for Dev UI
    await asyncio.Event().wait()


if __name__ == '__main__':
    ai.run_main(main())
