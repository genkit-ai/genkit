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

"""Checks for generated schema typing extensions."""

import runpy
from pathlib import Path


def test_runtime_error_reason_accessor_is_generated() -> None:
    repo = Path(__file__).resolve().parents[6]
    generator = runpy.run_path(str(repo / 'py/scripts/schema_to_typing.py'))
    generate = generator['generate']
    generated = generate(repo / 'genkit-tools/genkit-schema.json', Path('_typing.py'))
    checked_in = repo / 'py/packages/genkit/src/genkit/_core/_typing.py'
    accessor = 'def reason(self) -> RuntimeErrorReason | None:'
    assert accessor in generated
    assert accessor in checked_in.read_text()
