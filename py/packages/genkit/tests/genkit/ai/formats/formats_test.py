#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the built-in formats."""

import genkit
import genkit.formats
from genkit._ai._formats._array import ArrayFormat
from genkit._ai._formats._builtin import built_in_formats
from genkit._ai._formats._enum import EnumFormat
from genkit._ai._formats._json import JsonFormat
from genkit._ai._formats._jsonl import JsonlFormat
from genkit._ai._formats._text import TextFormat
from genkit.formats import Formatter


class TestBuiltInFormats:
    """Test built-in format registration."""

    def test_built_in_formats_contains_all_expected_formats(self) -> None:
        """Test that built_in_formats list contains all expected formats."""
        format_names = [f.name for f in built_in_formats]

        assert 'array' in format_names
        assert 'enum' in format_names
        assert 'json' in format_names
        assert 'jsonl' in format_names
        assert 'text' in format_names

    def test_built_in_formats_count(self) -> None:
        """Test that there are exactly 5 built-in formats."""
        assert len(built_in_formats) == 5

    def test_built_in_formats_are_formatter_instances(self) -> None:
        """Test that all built-in formats are Formatter instances."""
        for fmt in built_in_formats:
            assert isinstance(fmt, Formatter)


class TestJsonFormatConfig:
    """Test JSON format default configuration."""

    def test_json_format_config(self) -> None:
        """Test that JSON format has correct default config."""
        json_format = JsonFormat()

        assert json_format.name == 'json'
        assert json_format.config.content_type == 'application/json'
        assert json_format.config.constrained is True
        assert json_format.config.format == 'json'
        assert json_format.config.default_instructions is False


class TestArrayFormatConfig:
    """Test Array format default configuration."""

    def test_array_format_config(self) -> None:
        """Test that Array format has correct default config."""
        array_format = ArrayFormat()

        assert array_format.name == 'array'
        assert array_format.config.content_type == 'application/json'
        assert array_format.config.constrained is True


class TestEnumFormatConfig:
    """Test Enum format default configuration."""

    def test_enum_format_config(self) -> None:
        """Test that Enum format has correct default config."""
        enum_format = EnumFormat()

        assert enum_format.name == 'enum'
        assert enum_format.config.content_type == 'text/enum'
        assert enum_format.config.constrained is True


class TestJsonlFormatConfig:
    """Test JSONL format default configuration."""

    def test_jsonl_format_config(self) -> None:
        """Test that JSONL format has correct default config."""
        jsonl_format = JsonlFormat()

        assert jsonl_format.name == 'jsonl'
        assert jsonl_format.config.content_type == 'application/jsonl'


class TestTextFormatConfig:
    """Test Text format default configuration."""

    def test_text_format_config(self) -> None:
        """Test that Text format has correct default config."""
        text_format = TextFormat()

        assert text_format.name == 'text'
        assert text_format.config.content_type == 'text/plain'
        assert text_format.config.constrained is None


class TestModuleExports:
    """Test that all required types are exported from the module."""

    def test_genkit_formats_exports(self) -> None:
        """genkit.formats exports exactly the three custom-format types."""
        assert sorted(genkit.formats.__all__) == ['FormatHandler', 'Formatter', 'FormatterConfig']

    def test_root_does_not_export_format_types(self) -> None:
        """Format types live in genkit.formats, not the root package."""
        for name in ('Formatter', 'FormatterConfig', 'FormatHandler', 'FormatDef'):
            assert name not in genkit.__all__

    def test_all_format_classes_exported(self) -> None:
        """Test that all format classes are exported."""
        assert ArrayFormat is not None
        assert EnumFormat is not None
        assert JsonFormat is not None
        assert JsonlFormat is not None
        assert TextFormat is not None
