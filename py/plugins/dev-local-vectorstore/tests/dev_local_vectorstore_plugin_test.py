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

"""Tests for Dev Local Vector Store plugin."""

from genkit.plugins.dev_local_vectorstore import define_dev_local_vector_store


def test_define_dev_local_vector_store_callable() -> None:
    """Test define_dev_local_vector_store is callable."""
    assert callable(define_dev_local_vector_store)


def test_define_dev_local_vector_store_exported() -> None:
    """Test define_dev_local_vector_store is exported from package."""
    from genkit.plugins.dev_local_vectorstore import define_dev_local_vector_store as func

    assert func is not None
    assert callable(func)
