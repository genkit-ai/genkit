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

"""Setup for evaluator demo."""

import os
import pathlib

from pydantic import BaseModel

from evaluator_demo.genkit_demo import ai
from evaluator_demo.pdf_rag import index_pdf

# Default document to index
CAT_FACTS = [str(pathlib.Path(os.path.join(pathlib.Path(__file__).parent, '..', 'docs', 'cat-handbook.pdf')).resolve())]


class SetupInput(BaseModel):
    """Input for setup flow."""

    documents: list[str] | None = None


@ai.flow(name='setup')
async def setup(options: SetupInput | None = None) -> None:
    """Run initial setup (indexing).

    Args:
        options: Setup options.

    Example:
        >>> await setup(SetupInput(documents=['doc.pdf']))
    """
    if not options or not options.documents:
        docs_to_index = CAT_FACTS
    else:
        # input overrides defaults
        docs_to_index = options.documents

    for doc in docs_to_index:
        await index_pdf(doc)
