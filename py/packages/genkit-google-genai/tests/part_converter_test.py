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

"""Tests for PartConverter utility functions.

These tests verify the edge cases documented in the utils.py module docstring,
particularly around URL classification and media part conversion.
"""

from unittest.mock import AsyncMock, patch

import pytest
from genkit_google_genai.models.utils import PartConverter
from google import genai

from genkit import Media, MediaPart, Part, ToolRequest, ToolRequestPart, ToolResponse, ToolResponsePart


class TestIsGeminiNativeUrl:
    """Tests for _is_gemini_native_url hostname classification."""

    def test_youtube_www(self) -> None:
        """YouTube www subdomain is natively resolved."""
        got = PartConverter._is_gemini_native_url('https://www.youtube.com/watch?v=abc123')
        if not got:
            pytest.fail(f'_is_gemini_native_url(www.youtube.com) = {got}, want True')

    def test_youtube_bare(self) -> None:
        """YouTube bare domain is natively resolved."""
        got = PartConverter._is_gemini_native_url('https://youtube.com/watch?v=abc123')
        if not got:
            pytest.fail(f'_is_gemini_native_url(youtube.com) = {got}, want True')

    def test_youtu_be_short(self) -> None:
        """YouTube short URL is natively resolved."""
        got = PartConverter._is_gemini_native_url('https://youtu.be/abc123')
        if not got:
            pytest.fail(f'_is_gemini_native_url(youtu.be) = {got}, want True')

    def test_files_api(self) -> None:
        """Gemini Files API URLs are natively resolved."""
        got = PartConverter._is_gemini_native_url('https://generativelanguage.googleapis.com/v1beta/files/abc123')
        if not got:
            pytest.fail(f'_is_gemini_native_url(generativelanguage.googleapis.com) = {got}, want True')

    def test_arbitrary_http_not_native(self) -> None:
        """Arbitrary HTTP URLs are NOT natively resolved."""
        got = PartConverter._is_gemini_native_url('https://example.com/image.jpg')
        if got:
            pytest.fail(f'_is_gemini_native_url(example.com) = {got}, want False')

    def test_wikipedia_not_native(self) -> None:
        """Wikipedia URLs are NOT natively resolved — they require download."""
        got = PartConverter._is_gemini_native_url('https://upload.wikimedia.org/image.jpg')
        if got:
            pytest.fail(f'_is_gemini_native_url(wikimedia.org) = {got}, want False')

    def test_invalid_url_returns_false(self) -> None:
        """Malformed URLs return False instead of raising."""
        got = PartConverter._is_gemini_native_url('not-a-url')
        if got:
            pytest.fail(f'_is_gemini_native_url(not-a-url) = {got}, want False')

    def test_empty_string_returns_false(self) -> None:
        """Empty string returns False."""
        got = PartConverter._is_gemini_native_url('')
        if got:
            pytest.fail(f'_is_gemini_native_url("") = {got}, want False')


class TestToGeminiMediaPart:
    """Tests for to_gemini media part conversion with native URL handling."""

    @pytest.mark.asyncio
    async def test_youtube_url_uses_file_data(self) -> None:
        """YouTube URLs are passed as file_data, NOT downloaded."""
        part = Part(root=MediaPart(media=Media(url='https://www.youtube.com/watch?v=abc', content_type='video/mp4')))

        result = await PartConverter.to_gemini(part)

        # Narrow to a single Part for attribute access.
        assert isinstance(result, genai.types.Part)
        # Must use file_data, not inline_data
        assert result.file_data is not None, 'YouTube URL should produce file_data, not inline_data'
        if result.inline_data is not None:
            pytest.fail('YouTube URL should NOT produce inline_data')
        if result.file_data.file_uri != 'https://www.youtube.com/watch?v=abc':
            pytest.fail(f'file_uri = {result.file_data.file_uri}, want original URL')
        if result.file_data.mime_type != 'video/mp4':
            pytest.fail(f'mime_type = {result.file_data.mime_type}, want video/mp4')

    @pytest.mark.asyncio
    async def test_youtu_be_short_url_uses_file_data(self) -> None:
        """Short youtu.be URLs are passed as file_data."""
        part = Part(root=MediaPart(media=Media(url='https://youtu.be/abc', content_type='video/mp4')))

        result = await PartConverter.to_gemini(part)

        assert isinstance(result, genai.types.Part)
        assert result.file_data is not None, 'youtu.be URL should produce file_data'
        if result.file_data.file_uri != 'https://youtu.be/abc':
            pytest.fail(f'file_uri = {result.file_data.file_uri}, want original URL')

    @pytest.mark.asyncio
    async def test_files_api_url_uses_file_data(self) -> None:
        """Gemini Files API URLs are passed as file_data."""
        url = 'https://generativelanguage.googleapis.com/v1beta/files/abc123'
        part = Part(root=MediaPart(media=Media(url=url, content_type='video/mp4')))

        result = await PartConverter.to_gemini(part)

        assert isinstance(result, genai.types.Part)
        assert result.file_data is not None, 'Files API URL should produce file_data'
        if result.file_data.file_uri != url:
            pytest.fail(f'file_uri = {result.file_data.file_uri}, want original URL')

    @pytest.mark.asyncio
    async def test_regular_http_url_downloads_inline(self) -> None:
        """Regular HTTP URLs are downloaded and sent as inline_data."""
        part = Part(root=MediaPart(media=Media(url='https://example.com/photo.jpg', content_type='image/jpeg')))

        mock_data = b'\x89PNG\r\n'
        with patch.object(
            PartConverter,
            '_download_image',
            new_callable=AsyncMock,
            return_value=(mock_data, 'image/jpeg'),
        ) as mock_download:
            result = await PartConverter.to_gemini(part)

            mock_download.assert_called_once_with('https://example.com/photo.jpg')

        assert isinstance(result, genai.types.Part)
        assert result.inline_data is not None, 'Regular HTTP URL should produce inline_data'
        if result.inline_data.data != mock_data:
            pytest.fail('inline_data.data should contain downloaded bytes')

    @pytest.mark.asyncio
    async def test_gs_uri_uses_file_data(self) -> None:
        """gs:// URIs are passed through as file_data (not downloaded)."""
        part = Part(root=MediaPart(media=Media(url='gs://bucket/video.mp4', content_type='video/mp4')))

        result = await PartConverter.to_gemini(part)

        assert isinstance(result, genai.types.Part)
        assert result.file_data is not None, 'gs:// URI should produce file_data'
        if result.file_data.file_uri != 'gs://bucket/video.mp4':
            pytest.fail(f'file_uri = {result.file_data.file_uri}, want original URI')

    @pytest.mark.asyncio
    async def test_data_uri_uses_inline_data(self) -> None:
        """data: URIs are decoded and sent as inline_data."""
        import base64

        raw = b'hello'
        b64 = base64.b64encode(raw).decode('utf-8')
        url = f'data:text/plain;base64,{b64}'
        part = Part(root=MediaPart(media=Media(url=url, content_type='text/plain')))

        result = await PartConverter.to_gemini(part)

        assert isinstance(result, genai.types.Part)
        assert result.inline_data is not None, 'data: URI should produce inline_data'
        if result.inline_data.data != raw:
            pytest.fail(f'inline_data.data = {result.inline_data.data!r}, want {raw!r}')
        if result.inline_data.mime_type != 'text/plain':
            pytest.fail(f'mime_type = {result.inline_data.mime_type}, want text/plain')


class TestFunctionCallRef:
    """Tool-request refs come from the model's call id, not a part index."""

    def test_from_gemini_uses_function_call_id(self) -> None:
        part = genai.types.Part(
            function_call=genai.types.FunctionCall(
                id='call-abc',
                name='write_file',
                args={'file_path': 'a.py', 'content': 'hi'},
            )
        )
        got = PartConverter.from_gemini(part)
        assert isinstance(got.root, ToolRequestPart)
        if got.root.tool_request.ref != 'call-abc':
            pytest.fail(f'ref = {got.root.tool_request.ref!r}, want call-abc')
        if got.root.tool_request.name != 'write_file':
            pytest.fail(f'name = {got.root.tool_request.name!r}, want write_file')

    def test_from_gemini_leaves_ref_unset_when_model_omits_id(self) -> None:
        part = genai.types.Part(
            function_call=genai.types.FunctionCall(
                name='write_file',
                args={'file_path': 'a.py', 'content': 'hi'},
            )
        )
        got = PartConverter.from_gemini(part)
        assert isinstance(got.root, ToolRequestPart)
        if got.root.tool_request.ref is not None:
            pytest.fail(f'ref = {got.root.tool_request.ref!r}, want None')

    @pytest.mark.asyncio
    async def test_to_gemini_round_trips_ref_as_function_call_id(self) -> None:
        part = Part(
            root=ToolRequestPart(
                tool_request=ToolRequest(
                    name='write_file',
                    ref='call-abc',
                    input={'file_path': 'a.py', 'content': 'hi'},
                )
            )
        )
        got = await PartConverter.to_gemini(part)
        assert isinstance(got, genai.types.Part)
        assert got.function_call is not None
        if got.function_call.id != 'call-abc':
            pytest.fail(f'function_call.id = {got.function_call.id!r}, want call-abc')

    @pytest.mark.asyncio
    async def test_to_gemini_omits_id_when_ref_unset(self) -> None:
        part = Part(
            root=ToolRequestPart(
                tool_request=ToolRequest(
                    name='write_file',
                    input={'file_path': 'a.py', 'content': 'hi'},
                )
            )
        )
        got = await PartConverter.to_gemini(part)
        assert isinstance(got, genai.types.Part)
        assert got.function_call is not None
        if got.function_call.id is not None:
            pytest.fail(f'function_call.id = {got.function_call.id!r}, want None')


class TestToolResponseToGemini:
    """Structured output is FunctionResponse.response; media is FunctionResponse.parts."""

    @pytest.mark.asyncio
    async def test_bare_output_is_one_function_response(self) -> None:
        part = Part(
            root=ToolResponsePart(
                tool_response=ToolResponse(
                    name='shot',
                    ref='s1',
                    output={'ok': True, 'label': 'lab'},
                )
            )
        )
        got = await PartConverter.to_gemini(part)
        assert isinstance(got, genai.types.Part)
        assert got.function_response is not None
        if got.function_response.name != 'shot':
            pytest.fail(f'name = {got.function_response.name!r}, want shot')
        if got.function_response.id != 's1':
            pytest.fail(f'id = {got.function_response.id!r}, want s1')
        if got.function_response.response != {'name': 'shot', 'content': {'ok': True, 'label': 'lab'}}:
            pytest.fail(f'response = {got.function_response.response!r}')
        if got.function_response.parts:
            pytest.fail(f'parts = {got.function_response.parts!r}, want empty')
        if got.inline_data is not None:
            pytest.fail('bare output must not add inline_data')

    @pytest.mark.asyncio
    async def test_string_output_is_wrapped_as_dict(self) -> None:
        part = Part(
            root=ToolResponsePart(
                tool_response=ToolResponse(name='weather', output='Sunny'),
            )
        )
        got = await PartConverter.to_gemini(part)
        assert isinstance(got, genai.types.Part)
        assert got.function_response is not None
        if got.function_response.response != {'name': 'weather', 'content': 'Sunny'}:
            pytest.fail(f'response = {got.function_response.response!r}')
        if got.function_response.parts:
            pytest.fail(f'parts = {got.function_response.parts!r}, want empty')

    @pytest.mark.asyncio
    async def test_content_media_is_function_response_parts(self) -> None:
        part = Part(
            root=ToolResponsePart(
                tool_response=ToolResponse(
                    name='shot',
                    output={'ok': True, 'label': 'lab'},
                    content=[{'media': {'contentType': 'image/png', 'url': 'data:image/png;base64,YWJj'}}],
                )
            )
        )
        got = await PartConverter.to_gemini(part)
        assert isinstance(got, genai.types.Part)
        assert got.function_response is not None
        if got.function_response.response != {'name': 'shot', 'content': {'ok': True, 'label': 'lab'}}:
            pytest.fail(f'response = {got.function_response.response!r}')
        if got.inline_data is not None:
            pytest.fail('media must live on function_response.parts, not a sibling Part')
        fr_parts = got.function_response.parts or []
        if len(fr_parts) != 1:
            pytest.fail(f'parts = {len(fr_parts)}, want 1')
        blob = fr_parts[0].inline_data
        if blob is None:
            pytest.fail('function_response.parts[0] missing inline_data')
        if blob.mime_type != 'image/png':
            pytest.fail(f'mime_type = {blob.mime_type!r}, want image/png')
        if blob.data != b'abc':
            pytest.fail(f"data = {blob.data!r}, want b'abc'")


class TestToolResponseFromGemini:
    """Inbound peels the dict wrap and keeps FunctionResponse.parts on the same tool."""

    def test_unwraps_name_content_envelope(self) -> None:
        part = genai.types.Part(
            function_response=genai.types.FunctionResponse(
                name='shot',
                id='s1',
                response={'name': 'shot', 'content': {'ok': True, 'label': 'lab'}},
            )
        )
        got = PartConverter.from_gemini(part)
        assert isinstance(got.root, ToolResponsePart)
        tr = got.root.tool_response
        if tr.name != 'shot':
            pytest.fail(f'name = {tr.name!r}, want shot')
        if tr.ref != 's1':
            pytest.fail(f'ref = {tr.ref!r}, want s1')
        if tr.output != {'ok': True, 'label': 'lab'}:
            pytest.fail(f'output = {tr.output!r}')
        if tr.content is not None:
            pytest.fail(f'content = {tr.content!r}, want None')

    def test_unwraps_string_content(self) -> None:
        part = genai.types.Part(
            function_response=genai.types.FunctionResponse(
                name='weather',
                response={'name': 'weather', 'content': 'Sunny'},
            )
        )
        got = PartConverter.from_gemini(part)
        assert isinstance(got.root, ToolResponsePart)
        if got.root.tool_response.output != 'Sunny':
            pytest.fail(f'output = {got.root.tool_response.output!r}, want Sunny')

    @pytest.mark.asyncio
    async def test_round_trip_keeps_output_and_media_together(self) -> None:
        part = Part(
            root=ToolResponsePart(
                tool_response=ToolResponse(
                    name='shot',
                    ref='s1',
                    output={'ok': True, 'label': 'lab'},
                    content=[{'media': {'contentType': 'image/png', 'url': 'data:image/png;base64,YWJj'}}],
                )
            )
        )
        outbound = await PartConverter.to_gemini(part)
        assert isinstance(outbound, genai.types.Part)
        got = PartConverter.from_gemini(outbound)
        assert isinstance(got.root, ToolResponsePart)
        tr = got.root.tool_response
        if tr.output != {'ok': True, 'label': 'lab'}:
            pytest.fail(f'output = {tr.output!r}')
        if tr.content != [{'media': {'url': 'data:image/png;base64,YWJj', 'contentType': 'image/png'}}]:
            pytest.fail(f'content = {tr.content!r}')
