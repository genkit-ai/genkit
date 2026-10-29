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

"""Operation: the poll handle for a background job.

Hand-written instead of generated from the wire schema because it is generic
over its output. Depends only on ``_base`` and ``_typing`` so any module can
import it without a cycle.
"""

from __future__ import annotations

from typing import Any, ClassVar, Generic

from pydantic import ConfigDict, Field
from typing_extensions import TypeVar

from genkit._core._base import GenkitModel
from genkit._core._typing import Metadata, OperationError

# Defaults to Any, same as JS Operation<O = any>.
OutputT = TypeVar('OutputT', default=Any)


class Operation(GenkitModel, Generic[OutputT]):
    """Poll handle for a background job.

    A background action returns an ``Operation`` right away. Poll it until
    ``done`` is true, then read ``output`` or ``error``. Generic over the
    output type, like JS ``Operation<O>`` and Go ``Operation[Out]``. Background
    models return ``Operation[ModelResponse]``. The wire shape does not depend
    on ``OutputT``, so a stored handle goes back to ``ai.check_operation`` as
    the dict you saved, or reloads with
    ``Operation[ModelResponse].model_validate_json(...)``.

    Example:
        ```python
        # 1. Start a background video generation
        op = await ai.generate_operation(model='googleai/veo-3.1-generate-preview', prompt='A chef plating risotto.')
        print(op.done)
        # => False

        # 2. Store the handle; later, check it with the dict you saved
        saved = op.model_dump_json()
        op = await ai.check_operation(json.loads(saved))

        # 3. Poll until done; output is a ModelResponse
        while not op.done:
            op = await ai.check_operation(op)
        print(op.output.media[0].url)
        # => https://storage.googleapis.com/.../risotto.mp4
        ```
    """

    # Pin the schema description to the generated one. Without it Pydantic copies
    # this docstring into every background model's outputSchema. Pydantic
    # merges this with GenkitModel's config.
    model_config: ClassVar[ConfigDict] = ConfigDict(
        json_schema_extra={'description': 'Model for operation data.'},
    )
    action: str | None = None
    id: str = Field(...)
    done: bool | None = None
    output: OutputT | None = None
    error: OperationError | None = None
    metadata: Metadata | None = None
