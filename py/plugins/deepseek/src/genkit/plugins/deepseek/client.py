# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the Apache License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek API client."""

from typing import Any, cast

from openai import AsyncOpenAI as _AsyncOpenAI

# Official DeepSeek API endpoint
# This is the standard endpoint and doesn't vary by region
DEFAULT_DEEPSEEK_API_URL = 'https://api.deepseek.com'


class DeepSeekClient:
    """DeepSeek API client initialization."""

    def __new__(cls, **deepseek_params: object) -> _AsyncOpenAI:
        """Initialize the DeepSeek client.

        Args:
            **deepseek_params: Client configuration parameters including:
                - api_key: DeepSeek API key.
                - base_url: API base URL (defaults to DEFAULT_DEEPSEEK_API_URL).
                - Additional OpenAI client parameters.

        Returns:
            Configured async OpenAI client instance.
        """
        api_key = cast(str | None, deepseek_params.pop('api_key', None))
        base_url = cast(str, deepseek_params.pop('base_url', DEFAULT_DEEPSEEK_API_URL))

        return _AsyncOpenAI(api_key=api_key, base_url=base_url, **cast(dict[str, Any], deepseek_params))
