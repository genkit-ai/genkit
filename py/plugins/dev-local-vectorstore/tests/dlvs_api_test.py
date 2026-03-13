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

"""Tests for LocalVectorStoreAPI base class."""

import json
import pathlib
import tempfile

import aiofiles.os
import pytest

from genkit.plugins.dev_local_vectorstore.constant import DbValue
from genkit.plugins.dev_local_vectorstore.local_vector_store_api import LocalVectorStoreAPI
from genkit.types import DocumentData, Embedding, TextPart


def _make_db_value(text: str = 'hello', embedding: list[float] | None = None) -> DbValue:
    """Create a DbValue for testing."""
    return DbValue(
        # Pydantic's discriminated union accepts TextPart directly at runtime,
        # but the static type is list[Part]. Wrapping in Part(root=...) causes
        # a ValidationError, so the type: ignore is intentional.
        doc=DocumentData(content=[TextPart(text=text)]),  # type: ignore[list-item]
        embedding=Embedding(embedding=embedding or [0.1, 0.2, 0.3]),
    )


class TestIndexFileName:
    """Tests for index file naming."""

    def test_index_file_name_format(self) -> None:
        """Index file name follows the expected template."""
        api = LocalVectorStoreAPI(index_name='test-index')
        assert api.index_file_name == '__db_test-index.json'

    def test_index_file_name_with_special_chars(self) -> None:
        """Index file name handles special characters in index name."""
        api = LocalVectorStoreAPI(index_name='my_index_123')
        assert api.index_file_name == '__db_my_index_123.json'

    def test_index_file_name_cached(self) -> None:
        """Index file name is cached (cached_property)."""
        api = LocalVectorStoreAPI(index_name='cached')
        name1 = api.index_file_name
        name2 = api.index_file_name
        assert name1 is name2


class TestSerializeData:
    """Tests for data serialization."""

    def test_serialize_empty_data(self) -> None:
        """Serialize empty dict returns empty dict."""
        result = LocalVectorStoreAPI._serialize_data({})
        assert result == {}

    def test_serialize_single_entry(self) -> None:
        """Serialize single DbValue entry."""
        data = {'key1': _make_db_value('test doc')}
        result = LocalVectorStoreAPI._serialize_data(data)
        assert 'key1' in result
        serialized = result['key1']
        assert isinstance(serialized, dict)

    def test_serialize_multiple_entries(self) -> None:
        """Serialize multiple DbValue entries preserves all keys."""
        data = {
            'a': _make_db_value('first'),
            'b': _make_db_value('second'),
            'c': _make_db_value('third'),
        }
        result = LocalVectorStoreAPI._serialize_data(data)
        assert set(result.keys()) == {'a', 'b', 'c'}

    def test_serialize_excludes_none(self) -> None:
        """Serialization excludes None values."""
        data = {'key1': _make_db_value('test')}
        result = LocalVectorStoreAPI._serialize_data(data)
        serialized = result['key1']
        assert isinstance(serialized, dict)
        for _key, value in _flatten_dict(serialized):
            assert value is not None


class TestDeserializeData:
    """Tests for data deserialization."""

    def test_deserialize_empty_data(self) -> None:
        """Deserialize empty dict returns empty dict."""
        result = LocalVectorStoreAPI._deserialize_data({})
        assert result == {}

    def test_roundtrip_serialize_deserialize(self) -> None:
        """Serialize then deserialize produces equivalent data."""
        original = {
            'key1': _make_db_value('hello world', [0.5, -0.3, 0.8]),
            'key2': _make_db_value('goodbye', [0.1, 0.9, -0.2]),
        }
        serialized = LocalVectorStoreAPI._serialize_data(original)
        deserialized = LocalVectorStoreAPI._deserialize_data(serialized)

        assert set(deserialized.keys()) == set(original.keys())
        for key in original:
            assert deserialized[key].embedding.embedding == original[key].embedding.embedding
            orig_text = original[key].doc.content[0].root.text
            deser_text = deserialized[key].doc.content[0].root.text
            assert deser_text == orig_text


class TestFileStoreOperations:
    """Tests for file store load/dump operations."""

    @pytest.mark.asyncio
    async def test_load_nonexistent_file_returns_empty(self) -> None:
        """Loading from nonexistent file returns empty dict."""
        api = LocalVectorStoreAPI(index_name='nonexistent_xyz_test')
        result = await api._load_filestore()
        assert result == {}

    @pytest.mark.asyncio
    async def test_dump_and_load_roundtrip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dump then load produces equivalent data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_name = 'roundtrip_test'

            new_template = str(pathlib.Path(tmpdir) / '__db_{index_name}.json')
            monkeypatch.setattr(LocalVectorStoreAPI, '_LOCAL_FILESTORE_TEMPLATE', new_template)
            api = LocalVectorStoreAPI(index_name=index_name)

            data = {
                'doc1': _make_db_value('stored document', [0.1, 0.2]),
            }
            await api._dump_filestore(data)

            loaded = await api._load_filestore()
            assert 'doc1' in loaded
            assert loaded['doc1'].embedding.embedding == [0.1, 0.2]

    @pytest.mark.asyncio
    async def test_dump_creates_valid_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dump creates a valid JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_name = 'json_test'
            new_template = str(pathlib.Path(tmpdir) / '__db_{index_name}.json')
            monkeypatch.setattr(LocalVectorStoreAPI, '_LOCAL_FILESTORE_TEMPLATE', new_template)
            api = LocalVectorStoreAPI(index_name=index_name)

            data = {'key': _make_db_value('test')}
            await api._dump_filestore(data)

            assert await aiofiles.os.path.exists(api.index_file_name)
            async with aiofiles.open(api.index_file_name, encoding='utf-8') as f:
                content = json.loads(await f.read())
            assert isinstance(content, dict)
            assert 'key' in content


def _flatten_dict(d: dict, prefix: str = '') -> list[tuple[str, object]]:
    """Flatten a nested dict for inspection."""
    items: list[tuple[str, object]] = []
    for k, v in d.items():
        new_key = f'{prefix}.{k}' if prefix else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key))
        else:
            items.append((new_key, v))
    return items
