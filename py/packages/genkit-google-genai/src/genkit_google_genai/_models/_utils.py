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

3. **Tool inputs that aren't objects ride under an ``input`` field**:
   Gemini reads a tool's parameters as named fields and always sends call
   arguments as an object, so a bare ``{'type': 'number'}`` looks like a tool
   that takes nothing. A non-object input is declared as
   ``{'input': <schema>}``, unwrapped from ``args['input']`` when the model
   calls the tool, and re-wrapped when an earlier call is sent back in history.

4. **GoogleSearch vs GoogleSearchRetrieval type mismatch**:
   The ``google.genai`` SDK's ``Tool.google_search`` field expects a
   ``GoogleSearch`` object, not the legacy ``GoogleSearchRetrieval``. Using
   the wrong type produces a silent type mismatch warning.
"""

import base64
import binascii
import re
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import unquote_to_bytes, urlparse, urlsplit, urlunsplit

import httpx
from google import genai

from genkit import GenkitError, Part, get_logger
from genkit.model import ToolRequest, ToolResponse
from genkit.plugin_api import loop_local_client

logger = get_logger(__name__)


# TODO(#4360): Replace with downloadRequestMedia middleware.
# Some media hosts (Wikipedia, for one) answer 403 to requests without a real
# User-Agent.
@loop_local_client
def _media_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={
            'User-Agent': 'Genkit/1.0 (https://github.com/genkit-ai/genkit; genkit@google.com)',
            'Accept-Encoding': 'identity',
        },
        follow_redirects=True,
    )


# The field a non-object tool input rides under on the wire.
TOOL_INPUT_FIELD = 'input'


@dataclass(frozen=True)
class ToolWire:
    """One tool as declared to Gemini this turn."""

    original_name: str
    wire_name: str
    wrapped: bool


class ToolTable:
    """Per-request map from Gemini wire names back to the tools we declared."""

    def __init__(self, entries: Sequence[ToolWire] = ()) -> None:
        self._by_wire = {e.wire_name: e for e in entries}
        self._by_original = {e.original_name: e for e in entries}

    def for_wire(self, name: str) -> ToolWire | None:
        return self._by_wire.get(name)

    def for_original(self, name: str) -> ToolWire | None:
        return self._by_original.get(name)


def _function_response_part(part: genai.types.Part) -> genai.types.FunctionResponsePart | None:
    """Media from a converted Genkit part, in the FunctionResponse.parts shape.

    Gemini's FunctionResponse.parts wire format only accepts media attachments
    (inline bytes or file URI). Non-media parts (text, data, reasoning) are
    omitted here.
    """
    if part.inline_data and part.inline_data.data is not None:
        return genai.types.FunctionResponsePart.from_bytes(
            data=part.inline_data.data,
            mime_type=part.inline_data.mime_type or 'application/octet-stream',
        )
    if part.file_data and part.file_data.file_uri:
        return genai.types.FunctionResponsePart.from_uri(
            file_uri=part.file_data.file_uri,
            mime_type=part.file_data.mime_type,
        )
    logger.debug('Skipping non-media part for FunctionResponse.parts: %s', part)
    return None


def _media_from_function_response_part(part: genai.types.FunctionResponsePart) -> dict[str, object] | None:
    """Wire media dict from a FunctionResponse.parts item."""
    if part.inline_data and part.inline_data.data is not None:
        b64_data = base64.b64encode(part.inline_data.data).decode('utf-8')
        mime = part.inline_data.mime_type or 'application/octet-stream'
        return {'media': {'url': f'data:{mime};base64,{b64_data}', 'contentType': mime}}
    if part.file_data and part.file_data.file_uri:
        media: dict[str, object] = {'url': part.file_data.file_uri}
        if part.file_data.mime_type:
            media['contentType'] = part.file_data.mime_type
        return {'media': media}
    return None


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
    # Server-side (built-in) tool invocations, as in the JS plugin.
    GEMINI_TOOL_CALL = 'geminiToolCall'
    GEMINI_TOOL_RESPONSE = 'geminiToolResponse'
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
    async def to_gemini(
        cls, part: Part, *, tools: ToolTable | None = None
    ) -> genai.types.Part | list[genai.types.Part]:
        """Maps a Genkit Part to a Gemini Part.

        This method inspects the root type of the Genkit Part and converts it
        into the corresponding `genai.types.Part` structure, which includes
        text, function calls, function responses, inline media data, or custom
        parts.

        Args:
            part: The Genkit Part object to convert.
            tools: Tools declared this turn; drives wrap and wire names.

        Returns:
            A `genai.types.Part` object representing the converted content.
        """
        if part.text is not None:
            return genai.types.Part(text=part.text or ' ')
        if part.tool_request is not None:
            args = part.tool_request.input
            entry = tools.for_original(part.tool_request.name) if tools else None
            if entry and entry.wrapped:
                # declared wrapped: always send {input: value}, None included,
                # so a second turn doesn't send args=None to a required field
                args = {TOOL_INPUT_FIELD: args}
            elif args is not None and not isinstance(args, dict):
                # Gemini rejects call args that aren't an object, so a string
                # from history (Any tool, or a tool dropped from this turn)
                # still goes under input
                args = {TOOL_INPUT_FIELD: args}
            wire_name = entry.wire_name if entry else part.tool_request.name.replace('/', '__')
            # Round-trip the call id when we have one so the model can correlate
            # tool responses to the original request.
            return genai.types.Part(
                function_call=genai.types.FunctionCall(
                    name=wire_name,
                    args=args,
                    id=part.tool_request.ref,
                ),
                thought_signature=cls._extract_thought_signature(part.metadata),
            )
        if part.reasoning is not None:
            return genai.types.Part(
                thought=True,
                text=part.reasoning,
                thought_signature=cls._extract_thought_signature(part.metadata),
            )
        if part.tool_response is not None:
            tool_response = part.tool_response
            tool_output = tool_response.output

            # Media next to structured output lives on FunctionResponse.parts
            # so the model sees one function result, not a loose follow-on part.
            fn_media: list[genai.types.FunctionResponsePart] = []
            if tool_response.content:
                for item in tool_response.content:
                    try:
                        genkit_part = Part.model_validate(item)
                        converted = await cls.to_gemini(genkit_part)
                        items = converted if isinstance(converted, list) else [converted]
                        for gemini_part in items:
                            media = _function_response_part(gemini_part)
                            if media is not None:
                                fn_media.append(media)
                    except Exception as exc:
                        logger.debug('Skipping unrecognised tool-response content part: %s', exc)

            # Older tools that don't fill in tool_response.content stash media
            # as data URLs inside output['content'] instead. Only runs when
            # the primary path came up empty: lift the data URLs onto
            # FunctionResponse.parts and strip 'content' from the dict so the
            # model doesn't see the same media twice.
            if not fn_media and isinstance(tool_output, dict) and 'content' in tool_output:
                content_list = tool_output['content']
                if isinstance(content_list, list):
                    clean_output = {k: v for k, v in tool_output.items() if k != 'content'}
                    for item in content_list:
                        if isinstance(item, dict) and 'media' in item:
                            media_info = item['media']
                            url = media_info.get('url') or ''
                            content_type = media_info.get('contentType') or media_info.get('content_type')
                            if url.startswith(cls.DATA):
                                _, data_str = url.split(',', 1)
                                data = base64.b64decode(data_str)
                                fn_media.append(
                                    genai.types.FunctionResponsePart.from_bytes(
                                        data=data,
                                        mime_type=content_type or 'application/octet-stream',
                                    )
                                )
                    if fn_media:
                        tool_output = clean_output

            # Gemini's FunctionResponse requires a dict-shaped ``response``,
            # but a tool can legitimately hand back any JSON value (string,
            # list, int, None, ...). Envelope it as ``{name, content}`` so
            # the wire payload is always a dict; the inbound converter
            # unwraps the same envelope so callers see the original value.
            response_entry = tools.for_original(tool_response.name) if tools else None
            gemini_tool_name = response_entry.wire_name if response_entry else tool_response.name.replace('/', '__')
            return genai.types.Part(
                function_response=genai.types.FunctionResponse(
                    id=tool_response.ref,
                    name=gemini_tool_name,
                    response={'name': gemini_tool_name, 'content': tool_output},
                    parts=fn_media or None,
                )
            )
        if part.media is not None:
            url = part.media.url
            if url.startswith(cls.DATA):
                uri_mime, data = _decode_data_uri(url)
                return genai.types.Part(
                    inline_data=genai.types.Blob(
                        mime_type=_media_type(part.media.content_type) or uri_mime or 'application/octet-stream',
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
                            mime_type=part.media.content_type,
                            file_uri=url,
                        )
                    )

                # TODO(#4360): Replace inline download with downloadRequestMedia
                # middleware (JS parity) once model middleware is implemented.
                # The Gemini API cannot fetch arbitrary HTTP URLs via file_uri,
                # so we must download the content and send it as inline_data.
                data, served_type = await cls._download_image(url)
                # The caller's content_type wins: object stores (S3, R2, GCS)
                # often serve application/octet-stream, which Gemini rejects.
                mime_type = (
                    _media_type(part.media.content_type)
                    or _media_type(served_type, skip_generic=True)
                    or 'image/jpeg'
                )
                return genai.types.Part(
                    inline_data=genai.types.Blob(
                        mime_type=mime_type,
                        data=data,
                    )
                )

            # Non-HTTP, non-data URIs (e.g. gs://, Files API URIs) are
            # passed through as file_data — the Gemini API can resolve these.
            # Local paths and raw base64 are not URIs; refuse them here rather
            # than send them to Gemini and get them echoed back in a 400.
            _reject_non_uri_media(url)
            return genai.types.Part(
                file_data=genai.types.FileData(
                    mime_type=part.media.content_type,
                    file_uri=url,
                )
            )
        if part.custom is not None:
            return cls._to_gemini_custom(part)
        # Default fallback for unknown part types
        return genai.types.Part()

    @classmethod
    def _to_gemini_custom(cls, part: Part) -> genai.types.Part:
        """Converts a Genkit CustomPart into a Gemini Part.

        This internal helper method handles the conversion of custom part types,
        specifically `executableCode` and `codeExecutionResult`, into their
        corresponding Gemini Part representations.

        Args:
            part: The Genkit Part with a custom payload to convert.

        Returns:
            A `genai.types.Part` object representing the converted custom content.
        """
        if part.custom and cls.EXECUTABLE_CODE in part.custom:
            custom_data = cast(dict, part.custom)
            return genai.types.Part(
                executable_code=genai.types.ExecutableCode(
                    code=custom_data[cls.EXECUTABLE_CODE][cls.CODE],
                    language=custom_data[cls.EXECUTABLE_CODE][cls.LANGUAGE],
                )
            )
        if part.custom and cls.CODE_EXECUTION_RESULT in part.custom:
            custom_data = cast(dict, part.custom)
            return genai.types.Part(
                code_execution_result=genai.types.CodeExecutionResult(
                    outcome=custom_data[cls.CODE_EXECUTION_RESULT][cls.OUTCOME],
                    output=custom_data[cls.CODE_EXECUTION_RESULT][cls.OUTPUT],
                )
            )
        if part.custom and cls.GEMINI_TOOL_CALL in part.custom:
            return genai.types.Part(
                tool_call=genai.types.ToolCall.model_validate(part.custom[cls.GEMINI_TOOL_CALL]),
                thought_signature=cls._extract_thought_signature(part.metadata),
            )
        if part.custom and cls.GEMINI_TOOL_RESPONSE in part.custom:
            return genai.types.Part(
                tool_response=genai.types.ToolResponse.model_validate(part.custom[cls.GEMINI_TOOL_RESPONSE]),
                thought_signature=cls._extract_thought_signature(part.metadata),
            )
        return genai.types.Part()

    @classmethod
    def from_gemini(cls, part: genai.types.Part, *, tools: ToolTable | None = None) -> Part:
        """Maps a Gemini Part back to a Genkit Part.

        This method inspects the type of the Gemini Part and converts it into
        the corresponding Genkit Part structure, handling text, function calls,
        function responses, inline media data, executable code, and code execution results.

        Args:
            part: The `genai.types.Part` object to convert.
            tools: Tools declared this turn; drives unwrap and original names.

        Returns:
            A Genkit `Part` object representing the converted content.
        """
        if part.thought:
            return Part.from_reasoning(part.text or '', metadata=cls._encode_thought_signature(part.thought_signature))
        if part.text is not None:
            return Part.from_text(part.text)
        if part.function_call:
            # Tool refs come only from the model's call id. A synthetic part
            # index isn't unique across turns, so resume can't tell repeated
            # calls to the same tool apart.
            wire_name = part.function_call.name or ''
            entry = tools.for_wire(wire_name) if tools else None
            name = entry.original_name if entry else wire_name.replace('__', '/')
            args: object = part.function_call.args if part.function_call.args is not None else {}
            # only tools we declared wrapped are unwrapped, so an object tool
            # with its own `input` field still gets all of its args
            if entry and entry.wrapped and isinstance(args, dict) and TOOL_INPUT_FIELD in args:
                # a stray key next to input is ignored, same as an object tool
                # ignoring a field the model added that the tool never declared
                args = args[TOOL_INPUT_FIELD]
            return Part(
                tool_request=ToolRequest(
                    ref=getattr(part.function_call, 'id', None),
                    name=name,
                    input=args,
                ),
                metadata=cls._encode_thought_signature(part.thought_signature),
            )
        if part.function_response:
            # If the model echoes back the ``{name, content}`` envelope we
            # used on the outbound side, peel it off so the caller sees the
            # original tool output.
            output = part.function_response.response
            if isinstance(output, dict) and output.get('name') == part.function_response.name and 'content' in output:
                output = output['content']
            # FunctionResponse.parts stay on the same tool result as content.
            content = []
            for fr_part in part.function_response.parts or []:
                media = _media_from_function_response_part(fr_part)
                if media is not None:
                    content.append(media)
            response_wire = part.function_response.name or ''
            response_entry = tools.for_wire(response_wire) if tools else None
            return Part(
                tool_response=ToolResponse(
                    ref=getattr(part.function_response, 'id', None),
                    name=response_entry.original_name if response_entry else response_wire.replace('__', '/'),
                    output=output,
                    content=content or None,
                )
            )
        if part.inline_data and part.inline_data.data:
            b64_data = base64.b64encode(part.inline_data.data).decode('utf-8')
            return Part.from_media(
                f'data:{part.inline_data.mime_type};base64,{b64_data}', content_type=part.inline_data.mime_type
            )
        if part.executable_code:
            return Part.from_custom({
                cls.EXECUTABLE_CODE: {
                    cls.LANGUAGE: part.executable_code.language,
                    cls.CODE: part.executable_code.code,
                }
            })
        if part.code_execution_result:
            return Part.from_custom({
                cls.CODE_EXECUTION_RESULT: {
                    cls.OUTCOME: part.code_execution_result.outcome,
                    cls.OUTPUT: part.code_execution_result.output,
                }
            })
        if part.tool_call:
            # A built-in tool the server ran (include_server_side_tool_invocations).
            # Kept wire-shaped so the next turn can send it back unchanged.
            return Part(
                custom={cls.GEMINI_TOOL_CALL: part.tool_call.model_dump(mode='json', by_alias=True, exclude_none=True)},
                metadata=cls._encode_thought_signature(part.thought_signature),
            )
        if part.tool_response:
            return Part(
                custom={
                    cls.GEMINI_TOOL_RESPONSE: part.tool_response.model_dump(mode='json', by_alias=True, exclude_none=True)
                },
                metadata=cls._encode_thought_signature(part.thought_signature),
            )

        return Part.from_text('')

    @classmethod
    def _extract_thought_signature(cls, metadata: dict[str, Any] | None) -> bytes | None:
        """Extracts and decodes the thought signature from metadata."""
        thought_sig = metadata.get('thoughtSignature') if metadata else None
        if isinstance(thought_sig, str):
            return base64.b64decode(thought_sig)
        return None

    @classmethod
    def _encode_thought_signature(cls, thought_signature: bytes | None) -> dict[str, Any] | None:
        """Encodes the thought signature into metadata format."""
        if thought_signature:
            return {'thoughtSignature': base64.b64encode(thought_signature).decode('utf-8')}
        return None

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

        Redirects are followed and private or loopback addresses are not
        blocked, so apps that pass end-user URLs here should vet them first;
        upfront vetting of the initial URL does not cover 302 redirects to
        loopback or link-local addresses (such as ``169.254.169.254``).

        Args:
            url: The URL to download.

        Returns:
            A tuple containing the content (bytes) and its MIME type (str or None).

        Raises:
            GenkitError: INVALID_ARGUMENT if the body is larger than 20MB,
                if the host sends a compressed ``Content-Encoding``, or when
                the media host answers with a 4xx other than 408/429.
            httpx.HTTPStatusError: A 5xx, 408, or 429 from the media host, left
                raw. It is not Gemini's failure, so Fallback must not switch
                models over it (the next model needs the same URL); Retry
                still tries again. The Ollama plugin does the same.
            httpx.TransportError: A timeout or transport failure, left
                unclassified because it may pass on retry.
        """
        client = _media_client()
        async with client.stream('GET', url, timeout=60.0) as response:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                code = e.response.status_code
                where = _redact_url(str(e.request.url))
                sanitized = httpx.HTTPStatusError(
                    f'HTTP {code} for {where}',
                    request=e.request,
                    response=e.response,
                )
                if 400 <= code < 500 and code not in (408, 429):
                    raise GenkitError(
                        status='INVALID_ARGUMENT',
                        message=f'Could not download request media (HTTP {code} for {where})',
                        cause=sanitized,
                    ) from sanitized
                raise sanitized from None
            encoding = response.headers.get('content-encoding', '').strip().lower()
            if encoding not in ('', 'identity', 'gzip', 'x-gzip', 'deflate'):
                raise GenkitError(
                    status='INVALID_ARGUMENT',
                    message=(
                        f'media at {_redact_url(url)} sent unsupported '
                        f'Content-Encoding {encoding!r}; expected identity, gzip or deflate'
                    ),
                )
            declared = response.headers.get('content-length', '')
            if declared.isdigit() and int(declared) > _MAX_MEDIA_DOWNLOAD_BYTES:
                raise _media_too_large(url)
            # Some CDNs gzip despite Accept-Encoding: identity. Decompress here,
            # bounded, so the cap applies to the decompressed size and a small
            # compressed body can't expand past it in one step.
            decoder = _bounded_decoder(encoding)
            chunks = response.aiter_raw()
            if response.is_stream_consumed:
                # Already read (mock transports): httpx has decoded it.
                decoder, chunks = None, response.aiter_bytes()
            body = bytearray()
            async for chunk in chunks:
                if decoder is None:
                    body.extend(chunk)
                else:
                    data = chunk
                    while data:
                        body.extend(decoder.decompress(data, _MAX_MEDIA_DOWNLOAD_BYTES + 1 - len(body)))
                        if len(body) > _MAX_MEDIA_DOWNLOAD_BYTES:
                            raise _media_too_large(url)
                        data = decoder.unconsumed_tail
                if len(body) > _MAX_MEDIA_DOWNLOAD_BYTES:
                    raise _media_too_large(url)
            if decoder is not None:
                body.extend(decoder.flush())
                if len(body) > _MAX_MEDIA_DOWNLOAD_BYTES:
                    raise _media_too_large(url)
            return bytes(body), response.headers.get('content-type')


# A truncated image or video is a corrupt file, so a URL over the cap fails
# instead of sending the first 20MB.
_MAX_MEDIA_DOWNLOAD_BYTES = 20 * 1024 * 1024


def _redact_url(url: str) -> str:
    # Signed URLs carry credentials in the query string, so error messages only name host and path.
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc.rpartition('@')[2], parts.path, '', ''))


def _media_too_large(url: str) -> GenkitError:
    return GenkitError(status='INVALID_ARGUMENT', message=f'media at {_redact_url(url)} is larger than 20MB')


def _bounded_decoder(encoding: str) -> Any:  # noqa: ANN401
    """A zlib decompressobj for a gzip/deflate Content-Encoding, or None for identity."""
    if encoding in ('gzip', 'x-gzip'):
        return zlib.decompressobj(16 + zlib.MAX_WBITS)
    if encoding == 'deflate':
        # RFC 9110 deflate is zlib-wrapped; 32 + MAX_WBITS also accepts gzip headers.
        return zlib.decompressobj(32 + zlib.MAX_WBITS)
    return None


# Media types that say nothing about the content; a caller's or the URI's type is better.
_GENERIC_MEDIA_TYPES = frozenset({'application/octet-stream', 'binary/octet-stream', 'application/x-binary'})


def _media_type(value: str | None, *, skip_generic: bool = False) -> str | None:
    """*value* without parameters (``; charset=...``), lowercased; None if empty or generic when *skip_generic*."""
    if not value:
        return None
    media_type = value.split(';', 1)[0].strip().lower()
    if not media_type or (skip_generic and media_type in _GENERIC_MEDIA_TYPES):
        return None
    return media_type


def _invalid_media(message: str) -> GenkitError:
    return GenkitError(status='INVALID_ARGUMENT', message=message)


def _decode_data_uri(url: str) -> tuple[str | None, bytes]:
    """The media type and bytes of an RFC 2397 ``data:`` URI.

    Raises:
        GenkitError: INVALID_ARGUMENT for a URI with no ``,`` or a payload
            that doesn't decode. The payload is not echoed back.
    """
    header, sep, payload = url.partition(',')
    if not sep:
        raise _invalid_media("media data: URI has no ',' before its payload; expected data:<type>;base64,<data>")
    params = header[len('data:') :].split(';')
    is_base64 = any(p.strip().lower() == 'base64' for p in params[1:])
    try:
        data = base64.b64decode(payload) if is_base64 else unquote_to_bytes(payload)
    except (binascii.Error, ValueError) as e:
        raise _invalid_media('media data: URI payload is not valid base64') from e
    return _media_type(params[0]), data


# A Windows drive path (C:\x or C:/x) parses as a one-letter URI scheme.
_WINDOWS_PATH = re.compile(r'^[A-Za-z]:[\\/]')
# Longer than any real URI without a scheme; most likely raw base64.
_MAX_SCHEMELESS_MEDIA_URL = 2048


def _reject_non_uri_media(url: str) -> None:
    """Raise for media URLs Gemini can't resolve: ``file://``, local paths, raw base64.

    The message names the problem, not the value, so a path or payload
    isn't echoed into logs and traces.
    """
    scheme = urlsplit(url).scheme.lower() if not _WINDOWS_PATH.match(url) else ''
    if scheme == 'file' or (not scheme and (url.startswith(('/', './', '../', '~')) or _WINDOWS_PATH.match(url))):
        raise _invalid_media(
            'media url is a local file path; Gemini cannot read local files. '
            'Read the bytes and pass a data: URI, or upload the file and pass its URI.'
        )
    if not scheme and len(url) > _MAX_SCHEMELESS_MEDIA_URL:
        raise _invalid_media(
            f'media url has no scheme and is {len(url)} characters long; '
            'pass inline bytes as a data:<type>;base64,<data> URI.'
        )
