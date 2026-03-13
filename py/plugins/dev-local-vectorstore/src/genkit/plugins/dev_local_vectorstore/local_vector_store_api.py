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


"""Base API for dev-local-vectorstore.

Provides async file-backed storage for document embeddings using
``aiofiles`` to avoid blocking the event loop during reads and writes.
"""

import json
from functools import cached_property

import aiofiles
import aiofiles.os

from genkit.codec import dump_json

from .constant import DbValue


class LocalVectorStoreAPI:
    """Base class for development local vector store operations."""

    _LOCAL_FILESTORE_TEMPLATE = '__db_{index_name}.json'

    def __init__(
        self,
        index_name: str,
    ) -> None:
        """Initialize the LocalVectorStoreAPI."""
        self.index_name = index_name

    @cached_property
    def index_file_name(self) -> str:
        """Get the filename of the index file."""
        return self._LOCAL_FILESTORE_TEMPLATE.format(index_name=self.index_name)

    async def _load_filestore(self) -> dict[str, DbValue]:
        """Load the filestore asynchronously to avoid blocking the event loop."""
        data: dict[str, object] = {}
        if await aiofiles.os.path.exists(self.index_file_name):
            async with aiofiles.open(self.index_file_name, encoding='utf-8') as f:
                contents = await f.read()
            data = json.loads(contents)
        return self._deserialize_data(data)

    async def _dump_filestore(self, data: dict[str, DbValue]) -> None:
        """Dump the filestore asynchronously to avoid blocking the event loop."""
        serialized_data = self._serialize_data(data)
        async with aiofiles.open(self.index_file_name, 'w', encoding='utf-8') as f:
            await f.write(dump_json(serialized_data, indent=2))

    @staticmethod
    def _serialize_data(data: dict[str, DbValue]) -> dict[str, object]:
        result: dict[str, object] = {}
        for k, v in data.items():
            result[k] = DbValue.model_dump(v, exclude_none=True)
        return result

    @staticmethod
    def _deserialize_data(data: dict[str, object]) -> dict[str, DbValue]:
        result: dict[str, DbValue] = {}
        for k, v in data.items():
            result[k] = DbValue.model_validate(v)
        return result
