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

"""Implementation of Enum output format."""

from typing import Any

from genkit._ai._formats._types import FormatDef, Formatter, FormatterConfig
from genkit._ai._model import (
    Message,
    ModelResponseChunk,
)
from genkit._core._compat import override
from genkit._core._error import GenkitError, RuntimeErrorReason


class EnumFormat(FormatDef):
    """Defines an Enum format for use with AI models.

    This format instructs the model to output a single value from a specified list of enum options.
    It is useful for classification tasks or choosing from a fixed set of actions.

    The formatter handles:
    1.  Validating that the schema contains an `enum` property.
    2.  Injecting instructions to output ONLY one of the enum values.
    3.  Cleaning the response (removing quotes) to return the raw enum string.

    Usage:
        ai.generate(
            output_format='enum',
            output_schema={
                'type': 'string',
                'enum': ['positive', 'negative', 'neutral']
            }
        )
    """

    def __init__(self) -> None:
        """Initializes the EnumFormat.

        Configures the format with:
        - name: 'enum'
        - content_type: 'text/enum'
        - constrained: True
        """
        super().__init__(
            'enum',
            FormatterConfig(
                content_type='text/enum',
                constrained=True,
            ),
        )

    @override
    def handle(self, schema: dict[str, object] | None) -> Formatter[Any, Any]:
        """Creates a Formatter for handling Enum values.

        Args:
            schema: The JSON schema. Must be type 'string' with an 'enum' property listing allowed values.

        Returns:
            A Formatter configured to parse enum values.

        Raises:
            GenkitError: If the schema type is not 'string' or 'enum'.
        """
        if schema and schema.get('type') not in ('string', 'enum'):
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message="Must supply a schema of type 'string' with an 'enum' property when using the enum format.",
                reason=RuntimeErrorReason.INVALID_SCHEMA,
            )

        def parse_value(text: str, *, partial: bool = False) -> str:
            value = text.strip()
            enum_values = schema.get('enum') if schema else None
            if isinstance(enum_values, list | tuple) and value in enum_values:
                return value
            if value and value[0] in ("'", '"'):
                quote = value[0]
                value = value[1:]
                if value.endswith(quote) and isinstance(enum_values, list | tuple):
                    unquoted = value[:-1].strip()
                    if unquoted in enum_values:
                        return unquoted
                if partial and isinstance(enum_values, list | tuple):
                    if any(isinstance(option, str) and option.startswith(value) for option in enum_values):
                        return value.strip()
                if value.endswith(quote):
                    value = value[:-1]
            return value.strip()

        def message_parser(msg: Message) -> str:
            """Parses a complete message, removing quotes."""
            return parse_value(msg.text)

        def chunk_parser(chunk: ModelResponseChunk) -> str:
            """Parses a chunk, removing quotes from accumulated text."""
            return parse_value(chunk.accumulated_text, partial=True)

        instructions = None
        if schema:
            enum_values = schema.get('enum')
            if isinstance(enum_values, list | tuple) and enum_values:
                enum_text = '\n'.join(str(v) for v in enum_values)
                instructions = (
                    'Output should be ONLY one of the following enum values. '
                    'Do not output any additional information or add quotes.\n\n'
                    f'{enum_text}'
                )

        return Formatter(
            chunk_parser=chunk_parser,
            message_parser=message_parser,
            instructions=instructions,
        )
