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

"""Utility functions and converters for Google GenAI plugin.

Edge Cases
----------
The following edge cases have been discovered through testing and should be
kept in mind when modifying media handling or tool conversion logic:

1. **YouTube URLs must not be downloaded** (``_GEMINI_NATIVE_HOSTS``):
   YouTube watch pages (``https://www.youtube.com/watch?v=...``) serve HTML
   content, not raw video. Downloading them produces ``text/html; charset=utf-8``
   inline data, which the Gemini API rejects with ``400 INVALID_ARGUMENT:
   Unsupported MIME type``. The Gemini API natively resolves YouTube URLs when
   passed as ``file_data``, so they must bypass the download path. This matches
   the JS plugin's ``downloadRequestMedia`` middleware filter.

2. **Gemini Files API URLs must not be downloaded**:
   URLs from ``generativelanguage.googleapis.com`` (the Files API) are
   server-side references. Downloading them is unnecessary and would require
   authentication. They are passed through as ``file_data``.

3. **Tool input schemas must use object types, not bare primitives**:
   LLMs always send tool arguments as JSON objects with named keys (e.g.
   ``{'celsius': 21.5}``). A tool with a bare ``float`` input generates
   a ``{'type': 'number'}`` schema, which causes a validation mismatch when
   the model sends ``{'celsius': 21.5}``. Use Pydantic models for tool inputs.

4. **GoogleSearch vs GoogleSearchRetrieval type mismatch**:
   The ``google.genai`` SDK's ``Tool.google_search`` field expects a
   ``GoogleSearch`` object, not the legacy ``GoogleSearchRetrieval``. Using
   the wrong type produces a silent type mismatch warning.
"""

import base64
from typing import cast
from urllib.parse import urlparse

from google import genai

from genkit.core.http_client import get_cached_client
from genkit.core.typing import DocumentPart, Metadata
from genkit.types import (
    CustomPart,
    Media,
    MediaPart,
    Part,
    ReasoningPart,
    TextPart,
    ToolRequest,
    ToolRequestPart,
    ToolResponse,
    ToolResponsePart,
)


class PartConverter:
    """Converts content parts between Genkit's internal representation and Gemini's API format.

    This class provides static methods to facilitate the translation of various
    content types (text, tool requests/responses, media, custom data) into the
    `genai.types.Part` format required by the Gemini API, and vice-versa.

    Attributes:
        EXECUTABLE_CODE (str): Key for executable code within custom parts.
        CODE_EXECUTION_RESULT (str): Key for code execution results within custom parts.
        OUTCOME (str): Key for execution outcome within code execution results.
        OUTPUT (str): Key for output within code execution results.
        LANGUAGE (str): Key for programming language within executable code.
        CODE (str): Key for code string within executable code.
        DATA (str): Prefix used for inline data URLs.
    """

    EXECUTABLE_CODE = 'executableCode'
    CODE_EXECUTION_RESULT = 'codeExecutionResult'
    OUTCOME = 'outcome'
    OUTPUT = 'output'
    LANGUAGE = 'language'
    CODE = 'code'
    DATA = 'data:'

    # Hostnames that the Gemini API can natively resolve via file_data.
    # These must NOT be downloaded and inlined — the API handles them directly.
    # Matches the JS plugin's downloadRequestMedia filter (gemini.ts).
    _GEMINI_NATIVE_HOSTS: frozenset[str] = frozenset({
        'generativelanguage.googleapis.com',
        'www.youtube.com',
        'youtube.com',
        'youtu.be',
    })

    @classmethod
    async def to_gemini(cls, part: Part | DocumentPart) -> genai.types.Part | list[genai.types.Part]:
        """Maps a Genkit Part to a Gemini Part.

        This method inspects the root type of the Genkit Part and converts it
        into the corresponding `genai.types.Part` structure, which includes
        text, function calls, function responses, inline media data, or custom
        parts.

        Args:
            part: The Genkit Part object to convert.

        Returns:
            A `genai.types.Part` object representing the converted content.
        """
        if isinstance(part.root, TextPart):
            return genai.types.Part(text=part.root.text or ' ')
        if isinstance(part.root, ToolRequestPart):
            return genai.types.Part(
                function_call=genai.types.FunctionCall(
                    # Gemini throws on '/' in tool name
                    name=part.root.tool_request.name.replace('/', '__'),
                    args=part.root.tool_request.input,
                ),
                thought_signature=cls._extract_thought_signature(part.root.metadata),
            )
        if isinstance(part.root, ReasoningPart):
            return genai.types.Part(
                thought=True,
                text=part.root.reasoning,
                thought_signature=cls._extract_thought_signature(part.root.metadata),
            )
        if isinstance(part.root, ToolResponsePart):
            tool_output = part.root.tool_response.output
            parts_to_return = []

            # Check for multimodal content structure {content: [{media: ...}]}
            if isinstance(tool_output, dict) and 'content' in tool_output:
                content_list = tool_output['content']
                if isinstance(content_list, list):
                    # Create a copy to avoid mutating original if that matters,
                    # but here we just want to separate content from other fields.
                    clean_output = tool_output.copy()
                    clean_output.pop('content')

                    # Heuristic: if media found, extract it to separate parts.
                    has_media = False
                    for item in content_list:
                        if isinstance(item, dict) and 'media' in item:
                            has_media = True
                            media_info = item['media']
                            url = media_info.get('url')
                            content_type = media_info.get('contentType') or media_info.get('content_type')

                            if url and url.startswith(cls.DATA):
                                _, data_str = url.split(',', 1)
                                data = base64.b64decode(data_str)
                                parts_to_return.append(
                                    genai.types.Part(inline_data=genai.types.Blob(mime_type=content_type, data=data))
                                )

                    if has_media:
                        # Append the function response part FIRST (contextually correct)
                        parts_to_return.insert(
                            0,
                            genai.types.Part(
                                function_response=genai.types.FunctionResponse(
                                    id=part.root.tool_response.ref,
                                    name=part.root.tool_response.name.replace('/', '__'),
                                    response=clean_output,
                                )
                            ),
                        )
                        return parts_to_return

            # Default behavior for standard tool responses
            # FunctionResponse.response must be a dict, not a raw value
            output = tool_output
            if not isinstance(output, dict):
                output = {'result': output}

            return genai.types.Part(
                function_response=genai.types.FunctionResponse(
                    id=part.root.tool_response.ref,
                    name=part.root.tool_response.name.replace('/', '__'),
                    response=output,
                )
            )
        if isinstance(part.root, MediaPart):
            url = part.root.media.url
            if url.startswith(cls.DATA):
                # Extract mime type and data from data:mime_type;base64,data
                metadata, data_str = url.split(',', 1)
                mime_type = part.root.media.content_type or metadata.split(':', 1)[1].split(';', 1)[0]
                data = base64.b64decode(data_str)

                return genai.types.Part(
                    inline_data=genai.types.Blob(
                        mime_type=mime_type,
                        data=data,
                    )
                )

            if url.startswith('http'):
                # URLs from hosts the Gemini API can natively resolve (YouTube,
                # Files API) are passed as file_data — downloading them would
                # fetch HTML pages instead of actual media content.
                if cls._is_gemini_native_url(url):
                    return genai.types.Part(
                        file_data=genai.types.FileData(
                            mime_type=part.root.media.content_type,
                            file_uri=url,
                        )
                    )

                # TODO(#4360): Replace inline download with downloadRequestMedia
                # middleware (JS parity) once model middleware is implemented.
                # The Gemini API cannot fetch arbitrary HTTP URLs via file_uri,
                # so we must download the content and send it as inline_data.
                data, mime_type = await cls._download_image(url)
                mime_type = mime_type or part.root.media.content_type or 'image/jpeg'
                return genai.types.Part(
                    inline_data=genai.types.Blob(
                        mime_type=mime_type,
                        data=data,
                    )
                )

            # Non-HTTP, non-data URIs (e.g. gs://, Files API URIs) are
            # passed through as file_data — the Gemini API can resolve these.
            return genai.types.Part(
                file_data=genai.types.FileData(
                    mime_type=part.root.media.content_type,
                    file_uri=url,
                )
            )
        if isinstance(part.root, CustomPart):
            return cls._to_gemini_custom(part)
        # Default fallback for unknown part types
        return genai.types.Part()

    @classmethod
    def _to_gemini_custom(cls, part: Part | DocumentPart) -> genai.types.Part:
        """Converts a Genkit CustomPart into a Gemini Part.

        This internal helper method handles the conversion of custom part types,
        specifically `executableCode` and `codeExecutionResult`, into their
        corresponding Gemini Part representations.

        Args:
            part: The Genkit Part object with a CustomPart root to convert.

        Returns:
            A `genai.types.Part` object representing the converted custom content.
        """
        # pyrefly: ignore[unsupported-operation] - Custom is RootModel[dict] which supports 'in'
        if part.root.custom and cls.EXECUTABLE_CODE in part.root.custom:
            custom_data = cast(dict, part.root.custom)
            return genai.types.Part(
                executable_code=genai.types.ExecutableCode(
                    code=custom_data[cls.EXECUTABLE_CODE][cls.CODE],
                    language=custom_data[cls.EXECUTABLE_CODE][cls.LANGUAGE],
                )
            )
        # pyrefly: ignore[unsupported-operation] - Custom is RootModel[dict] which supports 'in'
        if part.root.custom and cls.CODE_EXECUTION_RESULT in part.root.custom:
            custom_data = cast(dict, part.root.custom)
            return genai.types.Part(
                code_execution_result=genai.types.CodeExecutionResult(
                    outcome=custom_data[cls.CODE_EXECUTION_RESULT][cls.OUTCOME],
                    output=custom_data[cls.CODE_EXECUTION_RESULT][cls.OUTPUT],
                )
            )
        return genai.types.Part()

    @classmethod
    def from_gemini(cls, part: genai.types.Part, ref: str | None = None) -> Part:
        """Maps a Gemini Part back to a Genkit Part.

        This method inspects the type of the Gemini Part and converts it into
        the corresponding Genkit Part structure, handling text, function calls,
        function responses, inline media data, executable code, and code execution results.

        Args:
            part: The `genai.types.Part` object to convert.
            ref: The tool call reference ID.

        Returns:
            A Genkit `Part` object representing the converted content.
        """
        if part.thought:
            return Part(
                root=ReasoningPart(
                    reasoning=part.text or '',
                    metadata=cls._encode_thought_signature(part.thought_signature),
                )
            )
        if part.text is not None:
            return Part(root=TextPart(text=part.text))
        if part.function_call:
            return Part(
                root=ToolRequestPart(
                    tool_request=ToolRequest(
                        ref=ref or getattr(part.function_call, 'id', None),
                        # restore slashes
                        name=(part.function_call.name or '').replace('__', '/'),
                        input=part.function_call.args if part.function_call.args is not None else {},
                    ),
                    metadata=cls._encode_thought_signature(part.thought_signature),
                )
            )
        if part.function_response:
            return Part(
                root=ToolResponsePart(
                    tool_response=ToolResponse(
                        ref=getattr(part.function_response, 'id', None),
                        # restore slashes
                        name=(part.function_response.name or '').replace('__', '/'),
                        output=part.function_response.response,
                    )
                )
            )
        if part.inline_data and part.inline_data.data:
            b64_data = base64.b64encode(part.inline_data.data).decode('utf-8')
            return Part(
                root=MediaPart(
                    media=Media(
                        url=f'data:{part.inline_data.mime_type};base64,{b64_data}',
                        content_type=part.inline_data.mime_type,
                    )
                )
            )
        if part.executable_code:
            return Part(
                root=CustomPart(
                    custom={
                        cls.EXECUTABLE_CODE: {
                            cls.LANGUAGE: part.executable_code.language,
                            cls.CODE: part.executable_code.code,
                        }
                    }
                )
            )
        if part.code_execution_result:
            return Part(
                root=CustomPart(
                    custom={
                        cls.CODE_EXECUTION_RESULT: {
                            cls.OUTCOME: part.code_execution_result.outcome,
                            cls.OUTPUT: part.code_execution_result.output,
                        }
                    }
                )
            )

        return Part(root=TextPart(text=''))

    @classmethod
    def _extract_thought_signature(cls, metadata: Metadata | None) -> bytes | None:
        """Extracts and decodes the thought signature from metadata."""
        thought_sig = metadata.root.get('thoughtSignature') if metadata else None
        if isinstance(thought_sig, str):
            return base64.b64decode(thought_sig)
        return None

    @classmethod
    def _encode_thought_signature(cls, thought_signature: bytes | None) -> Metadata | None:
        """Encodes the thought signature into metadata format."""
        if thought_signature:
            return Metadata(root={'thoughtSignature': base64.b64encode(thought_signature).decode('utf-8')})
        return None

    # TODO(#4360): Replace with downloadRequestMedia middleware (JS parity).
    # User-Agent is required because many servers (e.g. Wikipedia) return
    # 403 Forbidden for the default httpx user-agent string.
    _DOWNLOAD_HEADERS: dict[str, str] = {
        'User-Agent': 'Genkit/1.0 (https://github.com/firebase/genkit; genkit@google.com)',
    }

    @classmethod
    def _is_gemini_native_url(cls, url: str) -> bool:
        """Returns True if the Gemini API can natively resolve this URL.

        YouTube and Gemini Files API URLs are handled server-side by the
        Gemini API via ``file_data``.  Downloading them would fetch HTML
        pages (YouTube) or require authentication (Files API) instead of
        the actual media content.

        Args:
            url: An HTTP/HTTPS URL to check.

        Returns:
            True if the URL's hostname is in ``_GEMINI_NATIVE_HOSTS``.
        """
        try:
            hostname = urlparse(url).hostname or ''
            return hostname in cls._GEMINI_NATIVE_HOSTS
        except ValueError:
            return False

    @classmethod
    async def _download_image(cls, url: str) -> tuple[bytes, str | None]:
        """Downloads media content from a URL and returns raw bytes with MIME type.

        Args:
            url: The URL to download.

        Returns:
            A tuple containing the content (bytes) and its MIME type (str or None).

        Raises:
            httpx.HTTPStatusError: If the server returns an error status code.
        """
        client = get_cached_client(
            cache_key='google_genai_media',
            headers=cls._DOWNLOAD_HEADERS,
            follow_redirects=True,
        )
        response = await client.get(url, timeout=60.0)
        response.raise_for_status()
        return response.content, response.headers.get('content-type')
