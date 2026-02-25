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

"""Tests for Cloudflare Workers AI typing and config schemas."""

import pytest
from pydantic import ValidationError

from genkit.plugins.cloudflare_workers_ai.typing import (
    CloudflareConfig,
    CloudflareEmbedConfig,
)


class TestCloudflareConfig:
    """Tests for CloudflareConfig validation."""

    def test_defaults(self) -> None:
        """Test Defaults."""
        cfg = CloudflareConfig()
        assert cfg.top_k is None
        assert cfg.seed is None
        assert cfg.repetition_penalty is None
        assert cfg.frequency_penalty is None
        assert cfg.presence_penalty is None
        assert cfg.lora is None
        assert cfg.raw is None

    def test_valid_top_k(self) -> None:
        """Test Valid top k."""
        cfg = CloudflareConfig(top_k=10)
        assert cfg.top_k == 10

    def test_top_k_lower_bound(self) -> None:
        """Test Top k lower bound."""
        with pytest.raises(ValidationError):
            CloudflareConfig(top_k=0)

    def test_top_k_upper_bound(self) -> None:
        """Test Top k upper bound."""
        with pytest.raises(ValidationError):
            CloudflareConfig(top_k=51)

    def test_valid_seed(self) -> None:
        """Test Valid seed."""
        cfg = CloudflareConfig(seed=42)
        assert cfg.seed == 42

    def test_seed_lower_bound(self) -> None:
        """Test Seed lower bound."""
        with pytest.raises(ValidationError):
            CloudflareConfig(seed=0)

    def test_seed_upper_bound(self) -> None:
        """Test Seed upper bound."""
        with pytest.raises(ValidationError):
            CloudflareConfig(seed=10000000000)

    def test_repetition_penalty_bounds(self) -> None:
        """Test Repetition penalty bounds."""
        cfg = CloudflareConfig(repetition_penalty=1.5)
        assert cfg.repetition_penalty == 1.5
        with pytest.raises(ValidationError):
            CloudflareConfig(repetition_penalty=3.0)

    def test_frequency_penalty_bounds(self) -> None:
        """Test Frequency penalty bounds."""
        cfg = CloudflareConfig(frequency_penalty=0.5)
        assert cfg.frequency_penalty == 0.5
        with pytest.raises(ValidationError):
            CloudflareConfig(frequency_penalty=-1.0)

    def test_presence_penalty_bounds(self) -> None:
        """Test Presence penalty bounds."""
        cfg = CloudflareConfig(presence_penalty=1.0)
        assert cfg.presence_penalty == 1.0
        with pytest.raises(ValidationError):
            CloudflareConfig(presence_penalty=2.5)

    def test_lora_string(self) -> None:
        """Test Lora string."""
        cfg = CloudflareConfig(lora='my-lora-adapter')
        assert cfg.lora == 'my-lora-adapter'

    def test_raw_flag(self) -> None:
        """Test Raw flag."""
        cfg = CloudflareConfig(raw=True)
        assert cfg.raw is True

    def test_inherits_common_config(self) -> None:
        """Test Inherits common config."""
        cfg = CloudflareConfig(temperature=0.7, max_output_tokens=512)
        assert cfg.temperature == 0.7
        assert cfg.max_output_tokens == 512


class TestCloudflareEmbedConfig:
    """Tests for CloudflareEmbedConfig validation."""

    def test_defaults(self) -> None:
        """Test Defaults."""
        cfg = CloudflareEmbedConfig()
        assert cfg.pooling is None

    def test_valid_mean_pooling(self) -> None:
        """Test Valid mean pooling."""
        cfg = CloudflareEmbedConfig(pooling='mean')
        assert cfg.pooling == 'mean'

    def test_valid_cls_pooling(self) -> None:
        """Test Valid cls pooling."""
        cfg = CloudflareEmbedConfig(pooling='cls')
        assert cfg.pooling == 'cls'

    def test_invalid_pooling_rejected(self) -> None:
        """Test Invalid pooling rejected."""
        with pytest.raises(ValidationError):
            CloudflareEmbedConfig(pooling='max')
