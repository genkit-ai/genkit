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

"""Custom evaluators sample.

This sample demonstrates how to write custom evaluators using both LLM-based
and non-LLM approaches. It provides five evaluators:

1. **Regex Matchers** (non-LLM):
   - `byo/regex_match_url` - Detects URLs in output
   - `byo/regex_match_us_phone` - Detects US phone numbers

2. **PII Detection** (LLM-based):
   - `byo/pii_detection` - Detects personally identifiable information

3. **Funniness** (LLM-based):
   - `byo/funniness` - Judges if output is a joke and if it's funny

4. **Deliciousness** (LLM-based):
   - `byo/deliciousness` - Judges if output is delicious (literally or metaphorically)

Testing Instructions
====================
1. Set ``GEMINI_API_KEY`` environment variable.
2. Run ``./run.sh`` from this sample directory.
3. In a separate terminal, run evaluations:

   Regex evaluators:
   ```bash
   genkit eval:run datasets/regex_dataset.json --evaluators=byo/regex_match_url,byo/regex_match_us_phone
   ```

   PII detection:
   ```bash
   genkit eval:run datasets/pii_detection_dataset.json --evaluators=byo/pii_detection
   ```

   Funniness:
   ```bash
   genkit eval:run datasets/funniness_dataset.json --evaluators=byo/funniness
   ```

   Deliciousness:
   ```bash
   genkit eval:run datasets/deliciousness_dataset.json --evaluators=byo/deliciousness
   ```

4. View results in the Dev UI at http://localhost:4000 (Evaluations section).
"""

import asyncio
import os
from pathlib import Path

from genkit.ai import Genkit
from genkit.core.logging import get_logger
from genkit.plugins.google_genai import GoogleAI
from src.constants import PERMISSIVE_SAFETY_SETTINGS, URL_REGEX, US_PHONE_REGEX
from src.deliciousness_evaluator import register_deliciousness_evaluator
from src.funniness_evaluator import register_funniness_evaluator
from src.pii_evaluator import register_pii_evaluator
from src.regex_evaluator import regex_matcher, register_regex_evaluators

logger = get_logger(__name__)

# Get prompts directory path
current_dir = Path(__file__).resolve().parent
prompts_path = current_dir.parent / 'prompts'

# Register all evaluators
JUDGE_MODEL = os.getenv('JUDGE_MODEL', 'googleai/gemini-3-pro-preview')

# Initialize Genkit with Google AI plugin, default model, and load prompts
ai = Genkit(plugins=[GoogleAI()], model=JUDGE_MODEL, prompt_dir=prompts_path)

# Regex evaluators (non-LLM)
register_regex_evaluators(
    ai,
    [
        regex_matcher('url', URL_REGEX),
        regex_matcher('us_phone', US_PHONE_REGEX),
    ],
)

# LLM-based evaluators
register_pii_evaluator(ai, JUDGE_MODEL, PERMISSIVE_SAFETY_SETTINGS)
register_funniness_evaluator(ai, JUDGE_MODEL, PERMISSIVE_SAFETY_SETTINGS)
register_deliciousness_evaluator(ai, JUDGE_MODEL, PERMISSIVE_SAFETY_SETTINGS)


async def main() -> None:
    """Main entry point for the custom evaluators sample."""
    await logger.ainfo('Custom evaluators sample initialized')
    await logger.ainfo('Registered evaluators:')
    await logger.ainfo('  - byo/regex_match_url (non-LLM)')
    await logger.ainfo('  - byo/regex_match_us_phone (non-LLM)')
    await logger.ainfo('  - byo/pii_detection (LLM-based)')
    await logger.ainfo('  - byo/funniness (LLM-based)')
    await logger.ainfo('  - byo/deliciousness (LLM-based)')
    await logger.ainfo('Use genkit eval:run to test evaluators with datasets')

    # Keep the app running in development mode
    await asyncio.Event().wait()


if __name__ == '__main__':
    ai.run_main(main())
