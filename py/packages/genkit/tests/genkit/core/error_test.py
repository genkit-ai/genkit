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

"""Unit tests for the error module."""

from genkit._core._error import (
    GenkitError,
    PublicError,
    ReflectionError,
    get_callable_json,
    get_error_stack,
    get_http_status,
)


def test_genkit_error() -> None:
    error = GenkitError(
        status='INVALID_ARGUMENT',
        message='Test message',
        details={'extra_msg': 'Test detail'},
        source='test_source',
    )
    assert error.original_message == 'Test message'
    assert error.http_code == 400
    assert error.status == 'INVALID_ARGUMENT'
    assert error.details['extra_msg'] == 'Test detail'
    assert error.source == 'test_source'
    assert str(error) == 'test_source: INVALID_ARGUMENT: Test message'

    error_no_source = GenkitError(status='INTERNAL', message='Test message 2')
    assert str(error_no_source) == 'INTERNAL: Test message 2'


def test_genkit_error_to_json() -> None:
    # NOT_FOUND is a valid gRPC-style status (maps to HTTP 404).
    error = GenkitError(status='NOT_FOUND', message='Resource not found', details={'id': 123})
    serializable = error.to_serializable()
    assert isinstance(serializable, ReflectionError)
    assert serializable.code == 5
    assert serializable.message == 'Resource not found'
    assert serializable.details is not None
    assert serializable.details.model_dump()['id'] == 123


def test_public_error() -> None:
    error = PublicError(
        status='UNAUTHENTICATED',
        message='Please log in',
        details={'extra_msg': 'Session expired'},
    )
    assert error.status == 'UNAUTHENTICATED'
    assert error.original_message == 'Please log in'
    assert error.details['extra_msg'] == 'Session expired'


def test_get_http_status() -> None:
    genkit_error = GenkitError(status='PERMISSION_DENIED', message='No access')
    assert get_http_status(genkit_error) == 403

    non_genkit_error = ValueError('Some other error')
    assert get_http_status(non_genkit_error) == 500


def test_get_callable_json() -> None:
    genkit_error = GenkitError(status='INVALID_ARGUMENT', message='Oops')
    json_data = get_callable_json(genkit_error)
    assert isinstance(json_data, dict)
    assert json_data['status'] == 'INVALID_ARGUMENT'
    assert json_data['message'] == 'Oops'
    assert 'details' in json_data

    non_genkit_error = TypeError('Type error')
    json_data = get_callable_json(non_genkit_error)
    assert isinstance(json_data, dict)
    assert json_data['status'] == 'INTERNAL'
    assert json_data['message'] == 'Type error'
    assert 'details' in json_data


def test_get_error_stack() -> None:
    try:
        raise ValueError('Example Error')
    except ValueError as e:
        tb = get_error_stack(e)
        assert tb == ''
