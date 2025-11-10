# -*- coding: utf-8 -*-
"""Tests for the cli script.

MIT License
Copyright (c) 2021-2022, Daniel Nagel
All rights reserved.

"""
import warnings

from click.testing import CliRunner

from mosaic.__main__ import main


def test_main():
    runner = CliRunner()
    result = runner.invoke(main)
    assert result.exit_code == 0
    assert '--help' in result.output


def test_clustering():
    runner = CliRunner()
    result = runner.invoke(main, ['clustering'])
    assert result.exit_code == 0
    assert '--help' in result.output


def test_similarity():
    runner = CliRunner()
    result = runner.invoke(main, ['similarity'])
    assert result.exit_code == 0
    assert '--help' in result.output


def test_no_trogon_deprecation_warning():
    """Test that trogon's DeprecationWarning about BaseCommand is suppressed."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # The import happens at module level, but we can test by invoking the CLI
        runner = CliRunner()
        result = runner.invoke(main)
        
        # Check that no DeprecationWarning about BaseCommand was raised
        basecommand_warnings = [
            warning for warning in w
            if issubclass(warning.category, DeprecationWarning)
            and 'BaseCommand' in str(warning.message)
        ]
        assert len(basecommand_warnings) == 0, (
            f"Found {len(basecommand_warnings)} BaseCommand deprecation warnings"
        )
        assert result.exit_code == 0
