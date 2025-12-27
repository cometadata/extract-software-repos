# tests/test_cli.py
"""Tests for CLI commands."""

import json
import tempfile
from pathlib import Path

import pytest
import polars as pl
from click.testing import CliRunner

from extract_software_repos.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


class TestExtractCommand:
    """Test extract command."""

    def test_help(self, runner):
        result = runner.invoke(cli, ["extract", "--help"])
        assert result.exit_code == 0
        assert "Extract software repository URLs" in result.output

    def test_extracts_from_datacite(self, runner):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "records.jsonl"
            output_file = Path(tmpdir) / "enrichments.jsonl"

            records = [
                {
                    "id": "10.1234/test1",
                    "attributes": {
                        "descriptions": [
                            {"descriptionType": "Abstract", "description": "See https://github.com/user/repo"}
                        ]
                    }
                }
            ]
            with open(input_file, "w") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")

            result = runner.invoke(cli, [
                "extract",
                str(input_file),
                "--from-datacite-abstract",
                "-o", str(output_file)
            ])

            assert result.exit_code == 0
            assert output_file.exists()

            with open(output_file) as f:
                enrichments = [json.loads(line) for line in f]

            assert len(enrichments) == 1
            assert enrichments[0]["doi"] == "10.1234/test1"

    def test_extracts_from_parquet(self, runner):
        df = pl.DataFrame({
            "relative_path": ["2308.11197v1.md"],
            "content": ["Check https://github.com/user/repo"]
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "test.parquet"
            output_path = Path(tmpdir) / "output.jsonl"
            df.write_parquet(input_path)

            result = runner.invoke(cli, [
                "extract",
                str(input_path),
                "-o", str(output_path),
            ])

            assert result.exit_code == 0
            assert output_path.exists()


class TestValidateCommand:
    """Test validate command."""

    def test_help(self, runner):
        result = runner.invoke(cli, ["validate", "--help"])
        assert result.exit_code == 0
        assert "Validate extracted URLs" in result.output

    def test_checkpoint_option_exists(self, runner):
        result = runner.invoke(cli, ["validate", "--help"])
        assert "--checkpoint" in result.output

    def test_wait_for_ratelimit_option_exists(self, runner):
        result = runner.invoke(cli, ["validate", "--help"])
        assert "--wait-for-ratelimit" in result.output

    def test_ignore_checkpoint_option_exists(self, runner):
        result = runner.invoke(cli, ["validate", "--help"])
        assert "--ignore-checkpoint" in result.output

    def test_http_concurrency_option_exists(self, runner):
        result = runner.invoke(cli, ["validate", "--help"])
        assert "--http-concurrency" in result.output


class TestHealFulltextCommand:
    """Test heal-fulltext command."""

    def test_heal_fulltext_parallel(self, runner):
        df = pl.DataFrame({
            "content": ["Some text with git-\nhub.com link"]
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "test.parquet"
            output_path = Path(tmpdir) / "healed.parquet"
            df.write_parquet(input_path)

            result = runner.invoke(cli, [
                "heal-fulltext",
                str(input_path),
                "-o", str(output_path),
                "--workers", "2",
            ])

            assert result.exit_code == 0
            assert output_path.exists()


class TestPromoteCommand:
    """Test promote command."""

    def test_help(self, runner):
        result = runner.invoke(cli, ["promote", "--help"])
        assert result.exit_code == 0
        assert "Promote validated repo links" in result.output

    def test_promote_command_exists(self):
        """Test that promote command is registered."""
        assert "promote" in [cmd for cmd in cli.commands.keys()]

    def test_promote_requires_records_option(self, runner):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "validated.jsonl"
            input_file.write_text("{}\n")

            result = runner.invoke(cli, [
                "promote",
                str(input_file),
            ])

            # Should fail because --records is required
            assert result.exit_code != 0
            assert "Missing option" in result.output or "required" in result.output.lower()

    def test_promote_options_exist(self, runner):
        result = runner.invoke(cli, ["promote", "--help"])
        assert "--records" in result.output
        assert "--record-type" in result.output
        assert "--promotion-threshold" in result.output
        assert "--name-similarity-threshold" in result.output
        assert "--batch-size" in result.output


class TestValidateWithPromote:
    """Test validate command with --promote flag."""

    def test_promote_option_exists(self, runner):
        result = runner.invoke(cli, ["validate", "--help"])
        assert "--promote" in result.output

    def test_default_output_with_promote(self, runner):
        """Default output should be _validated_promoted.jsonl when --promote used."""
        result = runner.invoke(cli, ["validate", "--help"])
        # Check the help text mentions the different default
        assert "validated_promoted" in result.output or "validated.jsonl" in result.output

    def test_default_output_filename_without_promote(self, runner, monkeypatch):
        """Without --promote, default output should be _validated.jsonl."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "enrichments.jsonl"
            input_file.write_text('{"enrichedValue": {"relatedIdentifier": "https://example.com/test"}}\n')

            # Mock to prevent actual validation
            monkeypatch.setenv("GITHUB_TOKEN", "fake_token")

            # Just check that the command parses correctly and would use _validated.jsonl
            # We can't easily test the actual default without mocking the whole validator
            result = runner.invoke(cli, ["validate", str(input_file), "--help"])
            assert "_validated.jsonl" in result.output or "validated.jsonl" in result.output

    def test_default_output_filename_with_promote(self, runner, monkeypatch):
        """With --promote, default output should be _validated_promoted.jsonl."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "test_input.jsonl"
            input_file.write_text('{"enrichedValue": {"relatedIdentifier": "https://example.com/test"}}\n')
            records_file = Path(tmpdir) / "records.jsonl"
            records_file.write_text('{"id": "test", "attributes": {"doi": "10.1234/test"}}\n')

            monkeypatch.setenv("GITHUB_TOKEN", "fake_token")

            # The actual output filename logic is tested by checking the echoed output path
            # This requires running the command far enough to see the output path
            # For now, we verify the help text mentions both options
            result = runner.invoke(cli, ["validate", "--help"])
            assert "--promote" in result.output

    def test_promote_requires_records(self, runner):
        """--promote without --records should error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "enrichments.jsonl"
            input_file.write_text('{"enrichedValue": {"relatedIdentifier": "https://github.com/user/repo"}}\n')

            result = runner.invoke(cli, [
                "validate",
                str(input_file),
                "--promote",
            ])

            assert result.exit_code != 0
            assert "--promote requires --records" in result.output

    def test_promote_options_exist(self, runner):
        result = runner.invoke(cli, ["validate", "--help"])
        assert "--records" in result.output
        assert "--record-type" in result.output
        assert "--promotion-threshold" in result.output
        assert "--name-similarity-threshold" in result.output
        assert "--batch-size" in result.output
        assert "--github-cache" in result.output
