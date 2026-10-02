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

"""Tests for action context definitions."""

from genkit import RequestData


def test_request_data_subclass_with_super_init_still_builds() -> None:
    """A RequestData subclass that calls super().__init__(request=...) still builds."""

    class LegacyRequestData(RequestData):
        def __init__(self, request: object) -> None:
            super().__init__(request=request)
            self.method = 'POST'
            self.headers = {'authorization': 'token'}
            self.input = {'key': 'val'}

    data = LegacyRequestData('req')
    assert data.request == 'req'
    assert data.method == 'POST'
    assert data.headers == {'authorization': 'token'}
    assert data.input == {'key': 'val'}


def test_request_data_defaults() -> None:
    """RequestData has default empty values for method, headers, and input."""
    data = RequestData(request='req')
    assert data.request == 'req'
    assert data.method == ''
    assert data.headers == {}
    assert data.input is None
