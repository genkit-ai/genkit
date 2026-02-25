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

"""Dynamic tools demo - Runtime tool creation and sub-span tracing in Genkit.

This sample demonstrates two powerful Genkit features:

1. ``ai.dynamic_tool()`` -- Create tools at runtime that are NOT globally
   registered. Useful for one-off tools, user-generated tools, or tools
   whose behavior depends on runtime data.

2. ``ai.run()`` -- Wrap any function call as a named step (sub-span)
   in the trace. The step and its input/output appear in the Dev UI
   trace viewer.

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ ai.dynamic_tool()   │ Makes a tool on the spot, like writing a recipe   │
    │                     │ card right when you need it instead of using a     │
    │                     │ cookbook.                                          │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ ai.run()            │ Wraps a regular function so it shows up in the    │
    │                     │ trace, like adding a bookmark to mark your place. │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ @ai.tool()          │ The standard way to register a tool (global).     │
    │                     │ Dynamic tools skip this registration.             │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Trace / Span        │ A tree of operations recorded during a flow run.  │
    │                     │ Each ai.run() call creates a child span.          │
    └─────────────────────┴────────────────────────────────────────────────────┘

Data Flow::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    COMBINED DEMO FLOW                                    │
    │                                                                         │
    │   combined_demo(input)                                                  │
    │        │                                                                │
    │        ├── ai.run("preprocess_step", input, preprocess)                 │
    │        │       └── Returns preprocessed string                          │
    │        │                                                                │
    │        ├── ai.dynamic_tool("scaler", scale_fn)                          │
    │        │       └── Creates tool (not globally registered)               │
    │        │                                                                │
    │        ├── scaler.arun(7)                                               │
    │        │       └── Returns 7 * 10 = 70                                  │
    │        │                                                                │
    │        └── Returns {step_result, tool_result, tool_metadata}            │
    └─────────────────────────────────────────────────────────────────────────┘

Testing Instructions
====================
1. Set ``GEMINI_API_KEY`` environment variable.
2. Run ``./run.sh`` from this sample directory.
3. Open the DevUI at http://localhost:4000.
4. Run each flow:
   - ``dynamic_tool_demo``: Creates and runs a dynamic multiplier tool.
   - ``run_step_demo``: Wraps a function in a trace span.
   - ``combined_demo``: Uses both features together.
5. Click "View trace" to see the sub-spans and tool execution.

See README.md for more details.
"""

import asyncio
import os

from pydantic import BaseModel, Field

from genkit.ai import Genkit
from genkit.core.logging import get_logger
from genkit.plugins.google_genai import GoogleAI
from samples.shared.logging import setup_sample

setup_sample()

if 'GEMINI_API_KEY' not in os.environ:
    os.environ['GEMINI_API_KEY'] = input('Please enter your GEMINI_API_KEY: ')

logger = get_logger(__name__)

ai = Genkit(
    plugins=[GoogleAI()],
    model='googleai/gemini-2.5-flash',
)


class DynamicToolInput(BaseModel):
    """Input for dynamic tool demo."""

    value: int = Field(default=5, description='Value to pass to the dynamic tool')


class RunStepInput(BaseModel):
    """Input for run step demo."""

    data: str = Field(default='hello world', description='Data to process in the traced step')


class CombinedInput(BaseModel):
    """Input for combined demo."""

    input_val: str = Field(default='Dynamic tools demo', description='Input value for demo')


@ai.flow()
async def dynamic_tool_demo(input: DynamicToolInput) -> dict[str, object]:
    """Create and invoke a tool at runtime using ai.dynamic_tool().

    Unlike ``@ai.tool()`` which globally registers a tool, dynamic tools
    are created on-the-fly and exist only for the current scope. They
    can be called directly or passed to ``ai.generate(tools=[...])``.

    Args:
        input: Input with a value to pass to the dynamic tool.

    Returns:
        A dict containing the tool result and metadata.
    """

    def multiplier_fn(x: int) -> int:
        return x * 10

    dynamic_multiplier = ai.dynamic_tool(
        'dynamic_multiplier',
        multiplier_fn,
        description='Multiplies input by 10',
    )
    result = await dynamic_multiplier.arun(input.value)

    return {
        'input_value': input.value,
        'tool_result': result.response,
        'tool_name': dynamic_multiplier.metadata.get('name', 'unknown'),
        'tool_metadata': dynamic_multiplier.metadata,
    }


@ai.flow()
async def run_step_demo(input: RunStepInput) -> dict[str, str]:
    """Wrap a plain function as a traceable step using ai.run().

    ``ai.run(name, input, fn)`` creates a named sub-span in the trace.
    The step's input and output are recorded and visible in the Dev UI
    trace viewer.

    Args:
        input: Input with data to process.

    Returns:
        A dict containing the original and processed data.
    """

    def uppercase(data: str) -> str:
        return data.upper()

    def reverse(data: str) -> str:
        return data[::-1]

    step1 = await ai.run('uppercase_step', input.data, uppercase)
    step2 = await ai.run('reverse_step', step1, reverse)

    return {
        'original': input.data,
        'after_uppercase': step1,
        'after_reverse': step2,
    }


@ai.flow()
async def combined_demo(input: CombinedInput) -> dict[str, object]:
    """Combine ai.run() sub-spans with ai.dynamic_tool() in one flow.

    This flow demonstrates using both features together:
    1. ``ai.run()`` wraps a preprocessing function as a trace span.
    2. ``ai.dynamic_tool()`` creates a scaler tool at runtime.
    3. Both appear in the trace as inspectable steps.

    Args:
        input: Input with a value string.

    Returns:
        A dict with results from both the step and the dynamic tool.
    """

    def preprocess(data: str) -> str:
        return f'processed: {data}'

    step_result = await ai.run('preprocess_step', input.input_val, preprocess)

    def scale_fn(x: int) -> int:
        return x * 10

    scaler = ai.dynamic_tool('scaler', scale_fn, description='Scales input by 10')
    tool_result = await scaler.arun(7)

    return {
        'step_result': step_result,
        'tool_result': tool_result.response,
        'tool_metadata': scaler.metadata,
    }


async def main() -> None:
    """Main function -- keep alive for Dev UI."""
    await logger.ainfo('Dynamic tools demo started. Open http://localhost:4000 to test flows.')
    while True:
        await asyncio.sleep(3600)


if __name__ == '__main__':
    ai.run_main(main())
