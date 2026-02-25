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

"""Evaluation in code sample."""

import json
import pathlib

from evaluator_demo.genkit_demo import ai
from genkit.core.typing import EvalResponse

# Load dataset
DATA_PATH = pathlib.Path(__file__).parent.parent / 'data' / 'dogfacts.json'
if DATA_PATH.exists():
    with DATA_PATH.open() as f:
        DOG_DATASET = json.load(f)
else:
    DOG_DATASET = []


# Run this flow to programmatically execute the evaluator on the dog dataset.
@ai.flow(name='dog_facts_eval')
async def dog_facts_eval_flow() -> EvalResponse:
    """Run dog facts evaluation.

    Returns:
        The evaluation response.
    """
    # Ensure dataset is loaded as list of BaseDataPoint (or dicts which evaluate() accepts)
    # The dataset in dogfacts.json usually matches the structure needed.

    return await ai.evaluate(
        evaluator='genkitEval/faithfulness',
        dataset=DOG_DATASET,
        eval_run_id='my-dog-eval',
    )
